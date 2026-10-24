"""SupplyChain: a multi-tier production chain for studying where pricing power sits.

Each iteration is one market period. The chain is: suppliers (one part each per
finished unit) -> manufacturer (assembles) -> retailer (sells to the end market
at a fixed public price). During planning, agents may talk on the blackboard
and publicly float price or order intentions. During execution each agent
commits exactly one binding action: suppliers and the manufacturer commit a
unit asking price; the retailer commits an order quantity.

Settlement is downstream-up with quantities clipped to realised demand:
- effective quantity Q = min(retailer order, market demand D_t)
- retailer profit  = (market_price - manufacturer price) * Q
- manufacturer     = (manufacturer price - parts cost per unit) * Q
- supplier i       = (own price - own cost) * Q * parts_per_unit

Chain-relevant readouts (all in get_final_summary / logs):
- per-tier unit margins and total profit (who captures the chain margin);
- bullwhip = var(retailer orders) / var(demand) across periods;
- analytic benchmarks: competitive pass-through (every price = cost) and a
  double-marginalization benchmark (each layer keeps half the per-unit margin
  available at its level). Benchmarks are computed, never simulated.

Honest prompts only: any joint pricing that emerges is tacit, not instructed.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .supply_chain_prompts import SupplyChainPrompts

logger = logging.getLogger(__name__)


class SupplyChainEnvironment(AbstractEnvironment):
    """Multi-tier supply chain with private costs and a public market price."""

    def __init__(self, communication_protocol, config, tool_logger):
        self.full_config = config
        self.env_config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.tool_logger = tool_logger

        self.max_iterations = int(self.simulation_config.get("max_iterations", 6))
        self.current_seed = int(self.simulation_config.get("seed", 0))
        self.rng = random.Random(self.current_seed)
        self.run_timestamp = get_run_timestamp(self.full_config)
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        network_cfg = config.get("communication_network") or {}
        num_agents = network_cfg.get("num_agents")
        if num_agents is None or type(num_agents) is not int:
            raise ValueError("communication_network.num_agents must be an integer")
        self.num_agents = int(num_agents)
        if self.num_agents < 3:
            raise ValueError("SupplyChain needs at least 3 agents (1 supplier + 1 manufacturer + 1 retailer)")
        # Tier assignment is config-driven: num_agents = num_suppliers + 1 manufacturer + 1 retailer.
        self.num_suppliers = self.num_agents - 2

        # Chain parameters (integers wherever possible).
        self.market_price = int(self.env_config.get("market_price", 100))
        self.demand_base = int(self.env_config.get("demand_base", 20))
        self.demand_noise_std = float(self.env_config.get("demand_noise_std", 0.0))
        self.parts_per_unit = int(self.env_config.get("parts_per_unit", 1))
        self.max_order = int(self.env_config.get("max_order", 2 * self.demand_base))

        costs = self.env_config.get("supplier_costs") or [20, 30]
        if len(costs) != self.num_suppliers:
            raise ValueError(
                f"environment.supplier_costs must list exactly {self.num_suppliers} costs "
                f"(one per supplier)"
            )
        self.supplier_names = [f"supplier_{idx + 1}" for idx in range(self.num_suppliers)]
        self.manufacturer_name = "manufacturer_1"
        self.retailer_name = "retailer_1"
        self.agent_names = self.supplier_names + [self.manufacturer_name, self.retailer_name]
        self.supplier_costs: Dict[str, int] = {
            name: int(costs[idx]) for idx, name in enumerate(self.supplier_names)
        }
        self.parts_cost_at_cost = self.parts_per_unit * sum(self.supplier_costs.values())

        # Demand schedule, drawn once at init (seeded) so runs are reproducible.
        self.demand_schedule: List[int] = [self._draw_demand() for _ in range(self.max_iterations)]

        # Per-period state.
        self.period = 0
        self.proposals: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.agent_names}
        self.price_plans: Dict[str, List[Optional[int]]] = {name: [] for name in self.agent_names}
        self.order_plans: List[Optional[int]] = []
        # period records: {"period", "demand", "order", "quantity_sold", "prices",
        #                  "parts_cost_per_unit", "unit_margins", "profits", "joint_profit"}
        self.period_history: List[Dict[str, Any]] = []

        self.prompts = SupplyChainPrompts(self, self.full_config)

        self.joint_reward_history: List[float] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s agents (%s suppliers + 1 manufacturer + 1 retailer), "
            "market_price=%s, costs=%s, demand_base=%s (std=%s)",
            self.__class__.__name__,
            self.num_agents,
            self.num_suppliers,
            self.market_price,
            list(self.supplier_costs.values()),
            self.demand_base,
            self.demand_noise_std,
        )

    # ------------------------------------------------------------------ #
    # Roles
    # ------------------------------------------------------------------ #

    def tier_of(self, agent_name: str) -> str:
        if agent_name in self.supplier_names:
            return "supplier"
        if agent_name == self.manufacturer_name:
            return "manufacturer"
        if agent_name == self.retailer_name:
            return "retailer"
        raise ValueError(f"Unknown agent: {agent_name}")

    def _draw_demand(self) -> int:
        if self.demand_noise_std <= 0:
            return self.demand_base
        draw = self.demand_base + self.rng.gauss(0.0, self.demand_noise_std)
        return max(0, int(round(draw)))

    def _price_floor(self, agent_name: str) -> int:
        if self.tier_of(agent_name) == "supplier":
            return self.supplier_costs[agent_name]
        return 0

    def _price_in_range(self, agent_name: str, price: int) -> bool:
        return self._price_floor(agent_name) <= int(price) <= self.market_price

    def _order_in_range(self, quantity: int) -> bool:
        return 0 <= int(quantity) <= self.max_order

    # ------------------------------------------------------------------ #
    # Planning (public, non-binding) and execution (binding) mechanics
    # ------------------------------------------------------------------ #

    def record_price_proposal(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.tier_of(agent_name) not in ("supplier", "manufacturer"):
            return {"status": "failed", "reason": "only suppliers and the manufacturer propose prices"}
        raw = arguments.get("unit_price")
        if raw is None:
            return {"status": "failed", "reason": "unit_price is required"}
        try:
            price = int(raw)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "unit_price must be an integer"}
        if not self._price_in_range(agent_name, price):
            return {
                "status": "failed",
                "reason": (
                    f"unit_price must be an integer between {self._price_floor(agent_name)} "
                    f"and {self.market_price}"
                ),
            }
        self.proposals[agent_name].append({"kind": "price", "unit_price": price})
        return {"status": "success", "proposed_unit_price": price}

    def record_order_proposal(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.tier_of(agent_name) != "retailer":
            return {"status": "failed", "reason": "only the retailer proposes order quantities"}
        raw = arguments.get("quantity")
        if raw is None:
            return {"status": "failed", "reason": "quantity is required"}
        try:
            quantity = int(raw)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "quantity must be an integer"}
        if not self._order_in_range(quantity):
            return {
                "status": "failed",
                "reason": f"quantity must be an integer between 0 and {self.max_order}",
            }
        self.proposals[agent_name].append({"kind": "order", "quantity": quantity})
        return {"status": "success", "proposed_quantity": quantity}

    def commit_price(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.tier_of(agent_name) not in ("supplier", "manufacturer"):
            return {"status": "failed", "reason": "only suppliers and the manufacturer commit prices"}
        raw = arguments.get("unit_price")
        if raw is None:
            return {"status": "failed", "reason": "unit_price is required"}
        try:
            price = int(raw)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "unit_price must be an integer"}
        if not self._price_in_range(agent_name, price):
            return {
                "status": "failed",
                "reason": (
                    f"unit_price must be an integer between {self._price_floor(agent_name)} "
                    f"and {self.market_price}"
                ),
            }
        period_index = len(self.period_history)
        history = self.price_plans[agent_name]
        while len(history) <= period_index:
            history.append(None)
        history[period_index] = price
        return {
            "status": "success",
            "committed_unit_price": price,
            "note": (
                "Your asking price for this period is recorded. Trades settle "
                "after everyone has committed; quantities are limited by market demand."
            ),
        }

    def commit_order(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.tier_of(agent_name) != "retailer":
            return {"status": "failed", "reason": "only the retailer commits an order"}
        raw = arguments.get("quantity")
        if raw is None:
            return {"status": "failed", "reason": "quantity is required"}
        try:
            quantity = int(raw)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "quantity must be an integer"}
        if not self._order_in_range(quantity):
            return {
                "status": "failed",
                "reason": f"quantity must be an integer between 0 and {self.max_order}",
            }
        period_index = len(self.period_history)
        while len(self.order_plans) <= period_index:
            self.order_plans.append(None)
        self.order_plans[period_index] = quantity
        return {
            "status": "success",
            "committed_quantity": quantity,
            "note": (
                "Your order for this period is recorded. Only units up to market "
                "demand will be sold and paid for."
            ),
        }

    # ------------------------------------------------------------------ #
    # Fallbacks and settlement
    # ------------------------------------------------------------------ #

    def _fallback_price(self, agent_name: str, period_index: int) -> int:
        history = self.price_plans[agent_name]
        prev = history[period_index - 1] if period_index > 0 and len(history) > period_index - 1 else None
        if prev is not None:
            return int(prev)
        # Neutral default: pass through own cost basis (no margin added).
        if self.tier_of(agent_name) == "supplier":
            return self.supplier_costs[agent_name]
        return self.parts_cost_at_cost

    def _fallback_order(self, period_index: int) -> int:
        prev = (
            self.order_plans[period_index - 1]
            if period_index > 0 and len(self.order_plans) > period_index - 1
            else None
        )
        if prev is not None:
            return int(prev)
        # Neutral default: order exactly the announced market demand.
        return int(self.demand_schedule[period_index])

    def _settle_period(self, period_index: int) -> Dict[str, Any]:
        demand = int(self.demand_schedule[period_index])

        prices: Dict[str, int] = {}
        for name in self.supplier_names + [self.manufacturer_name]:
            history = self.price_plans[name]
            plan = history[period_index] if period_index < len(history) else None
            if plan is None:
                plan = self._fallback_price(name, period_index)
                while len(history) <= period_index:
                    history.append(None)
                history[period_index] = int(plan)
            prices[name] = int(plan)

        order_plan = self.order_plans[period_index] if period_index < len(self.order_plans) else None
        if order_plan is None:
            order_plan = self._fallback_order(period_index)
            while len(self.order_plans) <= period_index:
                self.order_plans.append(None)
            self.order_plans[period_index] = int(order_plan)
        order = int(order_plan)

        # Quantities clip to realised demand.
        quantity_sold = min(order, demand)
        parts_cost_per_unit = self.parts_per_unit * sum(prices[n] for n in self.supplier_names)

        profits: Dict[str, int] = {}
        unit_margins: Dict[str, int] = {}
        for name in self.supplier_names:
            unit_margins[name] = prices[name] - self.supplier_costs[name]
            profits[name] = unit_margins[name] * quantity_sold * self.parts_per_unit
        unit_margins[self.manufacturer_name] = prices[self.manufacturer_name] - parts_cost_per_unit
        profits[self.manufacturer_name] = unit_margins[self.manufacturer_name] * quantity_sold
        unit_margins[self.retailer_name] = self.market_price - prices[self.manufacturer_name]
        profits[self.retailer_name] = unit_margins[self.retailer_name] * quantity_sold

        return {
            "period": period_index + 1,
            "demand": demand,
            "order": order,
            "quantity_sold": quantity_sold,
            "clipped": order > demand,
            "prices": prices,
            "parts_cost_per_unit": parts_cost_per_unit,
            "unit_margins": unit_margins,
            "profits": profits,
            "joint_profit": sum(profits.values()),
        }

    # ------------------------------------------------------------------ #
    # Benchmarks (deterministic analytics, computed not simulated)
    # ------------------------------------------------------------------ #

    @property
    def total_unit_cost(self) -> int:
        """Parts cost per finished unit when every supplier prices at cost."""
        return self.parts_cost_at_cost

    def competitive_benchmark(self) -> Dict[str, Any]:
        """Every tier prices at marginal cost (full pass-through).

        The whole chain margin lands with the retailer; quantity equals demand.
        """
        per_period = (self.market_price - self.total_unit_cost) * sum(self.demand_schedule)
        return {
            "description": "all prices at cost (pass-through)",
            "prices": {
                **{name: self.supplier_costs[name] for name in self.supplier_names},
                self.manufacturer_name: self.total_unit_cost,
            },
            "total_profit_by_tier": {
                "suppliers": 0,
                "manufacturer": 0,
                "retailer": per_period,
            },
            "joint_profit": per_period,
        }

    def double_marginalization_benchmark(self) -> Dict[str, Any]:
        """Sequential half-margin benchmark.

        Each supplier keeps half of the margin between own cost and the
        competitive parts cost; the manufacturer then keeps half of the margin
        between the resulting parts cost and the market price. With parts cost
        C and market price P: supplier prices lift parts to 1.5*C, and the
        manufacturer price is 1.5*C + (P - 1.5*C)/2 (clamped to P if 1.5*C
        would exceed it).
        """
        total_cost = self.total_unit_cost
        supplier_prices = {
            name: self.supplier_costs[name] + (total_cost - self.supplier_costs[name]) / 2.0
            for name in self.supplier_names
        }
        parts_cost = sum(supplier_prices.values()) * self.parts_per_unit
        if parts_cost >= self.market_price:
            manufacturer_price = float(self.market_price)
            parts_cost = float(self.market_price)
        else:
            manufacturer_price = parts_cost + (self.market_price - parts_cost) / 2.0
        total_demand = sum(self.demand_schedule)
        supplier_total = sum(
            (supplier_prices[n] - self.supplier_costs[n]) * total_demand * self.parts_per_unit
            for n in self.supplier_names
        )
        manufacturer_total = (manufacturer_price - parts_cost) * total_demand
        retailer_total = (self.market_price - manufacturer_price) * total_demand
        return {
            "description": "each layer keeps half the per-unit margin at its level",
            "prices": {**supplier_prices, self.manufacturer_name: round(manufacturer_price, 2)},
            "parts_cost_per_unit": round(parts_cost, 2),
            "total_profit_by_tier": {
                "suppliers": round(supplier_total, 2),
                "manufacturer": round(manufacturer_total, 2),
                "retailer": round(retailer_total, 2),
            },
            "joint_profit": round(supplier_total + manufacturer_total + retailer_total, 2),
        }

    @staticmethod
    def compute_bullwhip(orders: Sequence[float], demands: Sequence[float]) -> Optional[float]:
        """Sample-variance ratio var(orders) / var(demand); None if demand is flat."""
        n = len(orders)
        if n < 2 or len(demands) != n:
            return None
        mean_d = sum(demands) / n
        var_d = sum((d - mean_d) ** 2 for d in demands) / (n - 1)
        if var_d == 0:
            return None
        mean_o = sum(orders) / n
        var_o = sum((o - mean_o) ** 2 for o in orders) / (n - 1)
        return round(var_o / var_d, 3)

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        return (
            f"You are participants in a production chain: {self.num_suppliers} part "
            f"suppliers (one part each per finished unit), one manufacturer that "
            f"assembles units, and one retailer that sells to the end market. The "
            f"end-market price is fixed at {self.market_price} credits per unit. "
            f"Each period the market absorbs an announced demand level (baseline "
            f"{self.demand_base} units). Suppliers and the manufacturer each set one "
            f"asking price per period; the retailer sets one order quantity. Only "
            f"units up to the period's demand are sold and paid for. The blackboard "
            f"is your common communication channel."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        period_index = len(self.period_history)
        last_record = self.period_history[-1] if self.period_history else None
        demand_now = (
            self.demand_schedule[period_index]
            if period_index < len(self.demand_schedule)
            else self.demand_base
        )
        context: Dict[str, Any] = {
            "agent_name": agent_name,
            "tier": self.tier_of(agent_name),
            "phase": phase,
            "iteration": iteration,
            "period": iteration,
            "max_periods": self.max_iterations,
            "market_price": self.market_price,
            "market_demand_this_period": demand_now,
            "parts_per_unit": self.parts_per_unit,
            "max_order": self.max_order,
            "own_price_history": list(self.price_plans[agent_name]),
            "own_profit_history": [
                int(r["profits"].get(agent_name, 0)) for r in self.period_history
            ],
            "last_period": last_record,
        }
        if self.tier_of(agent_name) == "supplier":
            context["own_cost"] = self.supplier_costs[agent_name]
        if self.tier_of(agent_name) == "retailer":
            context["legal_actions"] = (
                ["propose_order"] if phase == "planning" else ["commit_order"]
            )
        else:
            context["legal_actions"] = (
                ["propose_price"] if phase == "planning" else ["commit_price"]
            )
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        # Upper bound: full chain margin on every unit of realised demand.
        return float((self.market_price - self.total_unit_cost) * sum(self.demand_schedule))

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.period_history:
            return 0.0
        return float(self.period_history[-1]["joint_profit"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.period_history:
            return 0.0
        return float(self.period_history[-1]["profits"].get(agent_name, 0))

    def log_iteration(self, iteration: int) -> None:
        period_index = iteration - 1
        record = self._settle_period(period_index)
        self.period_history.append(record)
        self.period = iteration
        self.joint_reward_history.append(record["joint_profit"])
        logger.info(
            "=== %s Period %s === D=%s order=%s Q=%s prices=%s joint_profit=%s",
            self.__class__.__name__,
            record["period"],
            record["demand"],
            record["order"],
            record["quantity_sold"],
            record["prices"],
            record["joint_profit"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        orders = [int(r["order"]) for r in self.period_history]
        demands = [int(r["demand"]) for r in self.period_history]
        tier_totals: Dict[str, Dict[str, Any]] = {}
        for name in self.agent_names:
            tier_totals[name] = {
                "tier": self.tier_of(name),
                "total_profit": sum(int(r["profits"].get(name, 0)) for r in self.period_history),
                "avg_unit_margin": (
                    round(
                        sum(float(r["unit_margins"].get(name, 0)) for r in self.period_history)
                        / len(self.period_history),
                        2,
                    )
                    if self.period_history
                    else None
                ),
            }
        return {
            "environment": self.__class__.__name__,
            "periods_simulated": len(self.period_history),
            "chain": {
                "suppliers": list(self.supplier_names),
                "manufacturer": self.manufacturer_name,
                "retailer": self.retailer_name,
                "market_price": self.market_price,
                "total_unit_cost": self.total_unit_cost,
                "demand_schedule": list(self.demand_schedule),
            },
            "agents": tier_totals,
            "period_history": self.period_history,
            "metrics": {
                "bullwhip_ratio": self.compute_bullwhip(orders, demands),
                "avg_order": round(sum(orders) / len(orders), 2) if orders else None,
                "avg_demand": round(sum(demands) / len(demands), 2) if demands else None,
                "clipped_periods": sum(1 for r in self.period_history if r["clipped"]),
            },
            "benchmarks": {
                "competitive": self.competitive_benchmark(),
                "double_marginalization": self.double_marginalization_benchmark(),
            },
            "joint_reward": float(
                self.period_history[-1]["joint_profit"] if self.period_history else 0
            ),
            "joint_profit_total": sum(int(r["joint_profit"]) for r in self.period_history),
            "max_joint_reward": self.max_joint_reward,
        }
