"""AdAuctionGSP: repeated generalized second-price (GSP) ad-slot auction.

Terrarium idea #9 (terrarium-ideas/environments-tasks.md): "Do agents discover
generalized second-price equilibria, or get exploited?" — ad-slot auction with
private values, sweeping agent sophistication.

Each iteration every advertiser submits one sealed per-click bid. Bids are
ranked from highest to lowest; the k-th highest bidder wins slot k (slots
ordered by click-through rate, ctr_1 >= ctr_2 >= ... >= ctr_K). Following the
standard GSP pricing rule (Edelman-Ostrovsky-Schwarz / Varian), the winner of
slot k pays per click the bid of the advertiser ranked directly below them,
floored at the reserve price::

    payment_k = max(reserve_price, bid_(k+1))

The lowest-ranked winner (when every bidder fills a slot) pays the reserve. A
per-click ``value_i`` is private to each advertiser and stays private forever;
bids and payments are public after settlement. Profit for the iteration is
``(value_i - payment_i) * clicks_i`` with ``clicks_i = clicks_per_iteration *
ctr_slot``.

Analytic benchmarks (never LLM-derived), all over single-iteration profiles:
- TRUTHFUL: every advertiser bids their own value.
- OPTIMAL_WELFARE: the injective agent->slot allocation maximising delivered
  value (sum of value * clicks), found by enumeration (tiny instances).
- NASH: pure-strategy bid profile found by deterministic best-response
  dynamics over a documented candidate bid grid, started from truthful.
- MAX_JOINT_PROFIT: bid profile over the same grid maximising summed
  advertiser profit — the coordinated ceiling bidders could reach with
  jointly chosen bids. Realized total profit is reported against it.

Honest prompts only: bidders know the rules, the slot table, and their own
history. Any equilibrium-like play that emerges is tacit, not instructed.

Assumptions (the idea text is thin on specifics):
- one auctioneer, K ranked slots, per-click bidding, static private
  valuations repeated across iterations (repeated one-shot game);
- sealed bids per iteration; all bids revealed at settlement;
- payments follow the classic "pay the bid below you" GSP rule with a
  reserve-price floor (no VCG variant);
- values are either given explicitly in the config (``values``) or drawn
  i.i.d. uniform from ``value_range`` with the simulation seed.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from typing import Any, Dict, List, Mapping, Optional

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .ad_auction_gsp_prompts import AdAuctionGSPPrompts

logger = logging.getLogger(__name__)

DEFAULT_SLOTS: List[Dict[str, Any]] = [
    {"name": "slot_1", "ctr": 0.5},
    {"name": "slot_2", "ctr": 0.3},
    {"name": "slot_3", "ctr": 0.1},
]


class AdAuctionGSPEnvironment(AbstractEnvironment):
    """Repeated GSP ad-slot auction with private per-click values."""

    def __init__(self, communication_protocol, config, tool_logger):
        self.full_config = config
        self.env_config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.tool_logger = tool_logger

        self.max_iterations = int(self.simulation_config.get("max_iterations", 5))
        self.current_seed = int(self.simulation_config.get("seed", 0))
        self.rng = random.Random(self.current_seed)
        self.run_timestamp = get_run_timestamp(self.full_config)
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        network_cfg = config.get("communication_network") or {}
        num_agents = network_cfg.get("num_agents")
        if num_agents is None or type(num_agents) is not int:
            raise ValueError("communication_network.num_agents must be an integer")
        self.num_agents = int(num_agents)
        if self.num_agents < 2:
            raise ValueError("AdAuctionGSP needs at least 2 advertisers")

        # Slots: ranked by descending click-through rate.
        raw_slots = self.env_config.get("slots") or DEFAULT_SLOTS
        self.slots: List[Dict[str, Any]] = []
        seen = set()
        for idx, entry in enumerate(raw_slots):
            name = str(entry.get("name", f"slot_{idx + 1}"))
            if name in seen:
                raise ValueError(f"duplicate slot name: {name}")
            seen.add(name)
            ctr = float(entry.get("ctr", 0.0))
            if not (0.0 < ctr <= 1.0):
                raise ValueError("slot ctr must be in (0, 1]")
            self.slots.append({"name": name, "ctr": ctr})
        if not self.slots:
            raise ValueError("AdAuctionGSP needs at least one slot")
        for prev, cur in zip(self.slots, self.slots[1:]):
            if cur["ctr"] > prev["ctr"]:
                raise ValueError("slot ctrs must be in non-increasing order")
        self.slot_names: List[str] = [slot["name"] for slot in self.slots]
        self.num_slots = len(self.slots)
        self.clicks_per_iteration = float(self.env_config.get("clicks_per_iteration", 100))
        if self.clicks_per_iteration <= 0:
            raise ValueError("clicks_per_iteration must be > 0")
        self.slot_clicks: List[float] = [
            round(self.clicks_per_iteration * slot["ctr"], 6) for slot in self.slots
        ]
        self.reserve_price = float(self.env_config.get("reserve_price", 1.0))
        if self.reserve_price < 0:
            raise ValueError("reserve_price must be >= 0")
        self.bid_epsilon = float(self.env_config.get("bid_epsilon", 0.01))
        if self.bid_epsilon <= 0:
            raise ValueError("bid_epsilon must be > 0")

        self.agent_names = [f"bidder_{idx + 1}" for idx in range(self.num_agents)]

        # Private per-click values (never shared between agents).
        value_range = self.env_config.get("value_range") or [2.0, 15.0]
        explicit_values = self.env_config.get("values")
        self.agent_values: Dict[str, float] = {}
        if explicit_values is not None:
            if len(explicit_values) != self.num_agents:
                raise ValueError("environment.values must have one entry per agent")
            for name, raw in zip(self.agent_names, explicit_values):
                value = float(raw)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("values must be finite and >= 0")
                self.agent_values[name] = round(value, 6)
        else:
            lo, hi = float(value_range[0]), float(value_range[1])
            if not (0 <= lo < hi):
                raise ValueError("value_range must satisfy 0 <= lo < hi")
            for name in self.agent_names:
                self.agent_values[name] = round(self.rng.uniform(lo, hi), 2)

        # Candidate bid grid for the analytic benchmarks (BR dynamics and
        # coordinated ceiling): reserve neighbourhood plus value neighbourhoods.
        self.bid_grid: List[float] = self._build_bid_grid()

        # Per-iteration state.
        self.iteration = 0
        self.proposals: Dict[str, List[float]] = {name: [] for name in self.agent_names}
        self.committed_bids: Dict[str, List[Optional[float]]] = {
            name: [] for name in self.agent_names
        }
        # iteration records: {"iteration", "bids", "ranking", "allocation",
        #                     "payments", "clicks", "profits", "total_profit",
        #                     "welfare", "revenue"}
        self.auction_history: List[Dict[str, Any]] = []

        self.prompts = AdAuctionGSPPrompts(self, self.full_config)

        # Analytic benchmarks (deterministic, values + slot table only).
        self.benchmarks = self.compute_benchmarks()
        self.joint_reward_history: List[float] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s bidders, slots=%s, clicks_per_iteration=%s, "
            "reserve=%s, truthful_welfare=%s, max_joint_profit=%s",
            self.__class__.__name__,
            self.num_agents,
            self.slot_names,
            self.clicks_per_iteration,
            self.reserve_price,
            self.benchmarks["truthful_welfare"],
            self.benchmarks["max_joint_profit"],
        )

    # ------------------------------------------------------------------ #
    # Auction mechanics
    # ------------------------------------------------------------------ #

    def _build_bid_grid(self) -> List[float]:
        """Deterministic candidate bid set for benchmark search."""
        eps = self.bid_epsilon
        candidates = {0.0, self.reserve_price, self.reserve_price - eps, self.reserve_price + eps}
        for value in self.agent_values.values():
            candidates.update({value - eps, value, value + eps})
        candidates = {round(c, 6) for c in candidates if c >= 0}
        grid = sorted(candidates)
        # Keep the coordinated-ceiling enumeration tractable: fall back to a
        # coarser grid for large populations.
        while len(grid) ** self.num_agents > 300_000 and len(grid) > 2:
            grid = grid[::2]
        return grid

    def _coerce_bid(self, raw: Any) -> Optional[float]:
        try:
            bid = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(bid) or bid < 0:
            return None
        return round(bid, 6)

    def record_proposal(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        bid = self._coerce_bid(arguments.get("bid"))
        if bid is None:
            return {
                "status": "failed",
                "reason": "bid is required and must be a non-negative number",
            }
        self.proposals[agent_name].append(bid)
        return {"status": "success", "proposed_bid": bid}

    def commit_bid(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        bid = self._coerce_bid(arguments.get("bid"))
        if bid is None:
            return {
                "status": "failed",
                "reason": "bid is required and must be a non-negative number",
            }
        # Keep exactly one pending commit per iteration.
        it_index = len(self.auction_history)
        while len(self.committed_bids[agent_name]) <= it_index:
            self.committed_bids[agent_name].append(None)
        self.committed_bids[agent_name][it_index] = bid
        return {"status": "success", "committed_bid": bid}

    def _profile_from_bids(self, bids: Mapping[str, float]) -> Dict[str, Any]:
        """Rank bids, allocate slots and compute GSP payments/profits."""
        index_of = {name: i for i, name in enumerate(self.agent_names)}
        ranking = sorted(self.agent_names, key=lambda n: (-bids[n], index_of[n]))
        allocation: Dict[str, Optional[int]] = {}
        payments: Dict[str, float] = {}
        clicks: Dict[str, float] = {}
        for k, name in enumerate(ranking):
            if k < self.num_slots:
                allocation[name] = k
                below = bids[ranking[k + 1]] if k + 1 < len(ranking) else self.reserve_price
                payments[name] = round(max(self.reserve_price, below), 6)
                clicks[name] = self.slot_clicks[k]
            else:
                allocation[name] = None
                payments[name] = 0.0
                clicks[name] = 0.0
        profits = {
            name: round((self.agent_values[name] - payments[name]) * clicks[name], 6)
            for name in self.agent_names
        }
        welfare = round(
            sum(self.agent_values[name] * clicks[name] for name in self.agent_names), 6
        )
        revenue = round(sum(payments[name] * clicks[name] for name in self.agent_names), 6)
        return {
            "bids": {name: float(bids[name]) for name in self.agent_names},
            "ranking": ranking,
            "allocation": allocation,
            "payments": payments,
            "clicks": clicks,
            "profits": profits,
            "total_profit": round(sum(profits.values()), 6),
            "welfare": welfare,
            "revenue": revenue,
        }

    def _resolve_round_bids(self, it_index: int) -> Dict[str, float]:
        """Committed bids with fallback: repeat previous bid, else own value."""
        bids: Dict[str, float] = {}
        for name in self.agent_names:
            history = self.committed_bids[name]
            bid = history[it_index] if it_index < len(history) else None
            if bid is None:
                prev = history[it_index - 1] if it_index > 0 else None
                bid = prev if prev is not None else self.agent_values[name]
                while len(history) <= it_index:
                    history.append(None)
                history[it_index] = bid
            bids[name] = float(bid)
        return bids

    def _settle_round(self, it_index: int) -> Dict[str, Any]:
        record = self._profile_from_bids(self._resolve_round_bids(it_index))
        record = {"iteration": it_index + 1, **record}
        return record

    # ------------------------------------------------------------------ #
    # Analytic benchmarks
    # ------------------------------------------------------------------ #

    def _truthful_profile(self) -> Dict[str, Any]:
        return self._profile_from_bids(dict(self.agent_values))

    def _optimal_welfare(self) -> float:
        """Max delivered value over injective agent->slot assignments."""
        best = 0.0
        fill = min(self.num_slots, self.num_agents)
        for perm in itertools.permutations(self.agent_names, fill):
            welfare = sum(
                self.agent_values[name] * self.slot_clicks[k] for k, name in enumerate(perm)
            )
            best = max(best, round(welfare, 6))
        return best

    def _best_response_bid(self, name: str, others: Dict[str, float], current: float) -> float:
        current_profit = self._profile_from_bids({**others, name: current})["profits"][name]
        best_bid, best_profit = current, current_profit
        for candidate in self.bid_grid:
            profit = self._profile_from_bids({**others, name: candidate})["profits"][name]
            if profit > best_profit + 1e-9:
                best_profit, best_bid = profit, candidate
        return best_bid

    def _nash_bids(self) -> Dict[str, float]:
        """Pure Nash by deterministic best-response dynamics from truthful."""
        bids = dict(self.agent_values)
        for _ in range(100 * self.num_agents):
            changed = False
            for name in self.agent_names:
                others = {k: v for k, v in bids.items() if k != name}
                best_bid = self._best_response_bid(name, others, bids[name])
                if best_bid != bids[name]:
                    bids[name] = best_bid
                    changed = True
            if not changed:
                break
        return bids

    def _max_joint_profit(self) -> float:
        """Coordinated ceiling: best summed advertiser profit on the grid."""
        names = self.agent_names
        best = 0.0
        for combo in itertools.product(self.bid_grid, repeat=self.num_agents):
            bids = dict(zip(names, combo))
            total = self._profile_from_bids(bids)["total_profit"]
            best = max(best, total)
        return best

    def compute_benchmarks(self) -> Dict[str, Any]:
        truthful = self._truthful_profile()
        nash = self._profile_from_bids(self._nash_bids())
        optimal_welfare = self._optimal_welfare()
        return {
            "truthful_bids": truthful["bids"],
            "truthful_allocation": truthful["allocation"],
            "truthful_payments": truthful["payments"],
            "truthful_profits": truthful["profits"],
            "truthful_welfare": truthful["welfare"],
            "truthful_revenue": truthful["revenue"],
            "nash_bids": nash["bids"],
            "nash_allocation": nash["allocation"],
            "nash_payments": nash["payments"],
            "nash_profits": nash["profits"],
            "nash_welfare": nash["welfare"],
            "nash_revenue": nash["revenue"],
            "optimal_welfare": optimal_welfare,
            "truthful_welfare_ratio": (
                round(truthful["welfare"] / optimal_welfare, 6) if optimal_welfare else None
            ),
            "nash_welfare_ratio": (
                round(nash["welfare"] / optimal_welfare, 6) if optimal_welfare else None
            ),
            "max_joint_profit": self._max_joint_profit(),
        }

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        return (
            f"You are advertisers bidding in a per-click ad-slot auction. There are "
            f"{self.num_agents} advertisers competing for {self.num_slots} ad slots each "
            f"iteration. Each iteration every advertiser submits one per-click bid; bids "
            f"are ranked and slots are filled from the top. The blackboard is your public "
            f"communication channel."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        last_record = self.auction_history[-1] if self.auction_history else None
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "max_iterations": self.max_iterations,
            "own_value": self.agent_values[agent_name],
            "reserve_price": self.reserve_price,
            "slots": [
                {"name": slot["name"], "ctr": slot["ctr"], "clicks": self.slot_clicks[k]}
                for k, slot in enumerate(self.slots)
            ],
            "own_bid_history": list(self.committed_bids[agent_name]),
            "own_profit_history": [
                float(r["profits"].get(agent_name, 0.0)) for r in self.auction_history
            ],
            "last_iteration": last_record,
            "legal_actions": ["propose_bid"] if phase == "planning" else ["commit_bid"],
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        # Best joint (summed) advertiser profit reachable on the bid grid.
        return float(self.benchmarks["max_joint_profit"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.auction_history:
            return 0.0
        return float(self.auction_history[-1]["total_profit"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.auction_history:
            return 0.0
        return float(self.auction_history[-1]["profits"].get(agent_name, 0.0))

    def log_iteration(self, iteration: int) -> None:
        it_index = iteration - 1
        record = self._settle_round(it_index)
        self.auction_history.append(record)
        self.iteration = iteration
        self.joint_reward_history.append(record["total_profit"])
        logger.info(
            "=== %s Iteration %s === bids=%s ranking=%s payments=%s profits=%s "
            "total_profit=%s welfare=%s",
            self.__class__.__name__,
            record["iteration"],
            record["bids"],
            record["ranking"],
            record["payments"],
            record["profits"],
            record["total_profit"],
            record["welfare"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        total_profits = [r["total_profit"] for r in self.auction_history]
        welfares = [r["welfare"] for r in self.auction_history]
        avg_total_profit = (
            round(sum(total_profits) / len(total_profits), 6) if total_profits else None
        )
        avg_welfare = round(sum(welfares) / len(welfares), 6) if welfares else None
        # Wins per slot across iterations.
        slot_wins = {slot["name"]: 0 for slot in self.slots}
        for r in self.auction_history:
            for k in r["allocation"].values():
                if k is not None:
                    slot_wins[self.slot_names[k]] += 1
        max_joint = self.benchmarks["max_joint_profit"]
        return {
            "environment": self.__class__.__name__,
            "iterations_simulated": len(self.auction_history),
            "bidders": {
                name: {
                    "own_value": self.agent_values[name],
                    "bid_history": list(self.committed_bids[name]),
                    "profit_history": [
                        float(r["profits"].get(name, 0.0)) for r in self.auction_history
                    ],
                    "total_profit": round(
                        sum(float(r["profits"].get(name, 0.0)) for r in self.auction_history), 6
                    ),
                }
                for name in self.agent_names
            },
            "auction_history": self.auction_history,
            "slot_usage": slot_wins,
            "auction_stats": {
                "average_total_profit_per_iteration": avg_total_profit,
                "average_welfare_per_iteration": avg_welfare,
                "truthful_welfare": self.benchmarks["truthful_welfare"],
                "truthful_total_profit": self.benchmarks["truthful_profits"] and round(
                    sum(self.benchmarks["truthful_profits"].values()), 6
                ),
                "nash_welfare": self.benchmarks["nash_welfare"],
                "optimal_welfare": self.benchmarks["optimal_welfare"],
                "max_joint_profit": max_joint,
                "realized_profit_vs_max": (
                    round(avg_total_profit / max_joint, 6)
                    if avg_total_profit is not None and max_joint
                    else None
                ),
            },
            "joint_reward": (
                self.auction_history[-1]["total_profit"] if self.auction_history else 0.0
            ),
            "benchmarks": self.benchmarks,
        }
