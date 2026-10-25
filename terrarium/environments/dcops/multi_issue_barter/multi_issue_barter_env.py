"""MultiIssueBarterEnvironment: multi-issue bargaining with asymmetric private valuations.

Agents negotiate over ``num_issues`` (default 5) divisible resource issues
(``issue_1..issue_5``), each with ``total_units`` (default 10) integer units to
split among ``num_agents`` (default 2: ``trader_1``, ``trader_2``).

Traders hold asymmetric private per-unit valuations ``v_{i, k}`` and start from
an initial ``endowment`` where each trader starts with issues they value less
than the other trader does. Consequently, multi-issue package trades that swap
low-value issues for high-value issues yield higher joint and individual
payoffs than both the status-quo endowment and issue-by-issue 50/50 splits.

Supports two agenda modes via ``agenda_mode``:
- ``"package"`` (default): each iteration is a full 5-issue bargaining episode
  where traders can propose and commit full allocation vectors across all issues.
- ``"sequential"``: iteration ``t`` settles one issue at a time in
  ``agenda_order`` (`issue_1..issue_5` or configured order).

Settlement rule per issue ``k``:
Each trader ``i`` commits a demanded unit count ``d_{i, k}`` in ``0..total_units``.
- If ``sum_i d_{i, k} <= total_units`` (compatible claims), the deal succeeds on
  issue ``k``: each trader receives ``d_{i, k}`` plus an integer-split share of
  any unclaimed remainder ``total_units - sum_i d_{i, k}``.
- If ``sum_i d_{i, k} > total_units`` (incompatible claims / disagreement),
  issue ``k`` reverts to the initial ``endowment`` for that issue.

Analytic benchmarks (computed from valuations and endowments, never from the LLM):
- ``endowment_benchmark()``: status-quo no-trade payoff.
- ``equal_split_benchmark()``: 50/50 integer split on every issue.
- ``pareto_package_optimum_benchmark()``: maximum joint welfare allocation
  subject to individual rationality (every trader >= endowment payoff), breaking
  ties by maximum Nash bargaining product over gains from trade.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .multi_issue_barter_prompts import MultiIssueBarterPrompts

logger = logging.getLogger(__name__)


class MultiIssueBarterEnvironment(AbstractEnvironment):
    """Multi-issue barter environment comparing package vs. issue-by-issue agendas."""

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
            raise ValueError("MultiIssueBarterEnvironment needs at least 2 traders")

        self.num_issues = int(self.env_config.get("num_issues", 5))
        if self.num_issues < 1:
            raise ValueError("num_issues must be at least 1")
        self.total_units = int(self.env_config.get("total_units", 10))
        if self.total_units < 1:
            raise ValueError("total_units must be at least 1")

        self.agenda_mode = str(self.env_config.get("agenda_mode", "package")).strip().lower()
        if self.agenda_mode not in ("package", "sequential"):
            raise ValueError("agenda_mode must be 'package' or 'sequential'")

        raw_order = self.env_config.get("agenda_order")
        if raw_order is not None:
            order = [int(idx) for idx in raw_order]
            if sorted(order) != list(range(self.num_issues)):
                raise ValueError(
                    f"agenda_order must be a permutation of 0..{self.num_issues - 1}"
                )
            self.agenda_order: List[int] = order
        else:
            self.agenda_order = list(range(self.num_issues))

        self.issue_names: List[str] = [f"issue_{k + 1}" for k in range(self.num_issues)]
        self.agent_names: List[str] = [f"trader_{idx + 1}" for idx in range(self.num_agents)]

        self.valuations: Dict[str, List[int]] = self._init_valuations()
        self.endowments: Dict[str, List[int]] = self._init_endowments()

        # Per-round state.
        self.round_num = 0
        self.proposals: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.agent_names}
        self.pending_actions: Dict[str, Optional[Dict[str, Any]]] = {
            name: None for name in self.agent_names
        }
        self.agenda_setter: Optional[str] = None
        self.round_agenda_setter: Optional[str] = None
        self.round_history: List[Dict[str, Any]] = []

        self.prompts = MultiIssueBarterPrompts(self, self.full_config)

        self.joint_reward_history: List[int] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s traders, issues=%s, units/issue=%s, mode=%s, seed=%s",
            self.__class__.__name__,
            self.num_agents,
            self.num_issues,
            self.total_units,
            self.agenda_mode,
            self.current_seed,
        )

    # ------------------------------------------------------------------ #
    # Valuations and endowments initialization
    # ------------------------------------------------------------------ #

    def _init_valuations(self) -> Dict[str, List[int]]:
        """Initialize private per-unit valuations across issues."""
        cfg_vals = self.env_config.get("valuations")
        if isinstance(cfg_vals, dict):
            out: Dict[str, List[int]] = {}
            for name in self.agent_names:
                if name not in cfg_vals or len(cfg_vals[name]) != self.num_issues:
                    raise ValueError(
                        f"valuations[{name!r}] must be a list of length {self.num_issues}"
                    )
                out[name] = [int(v) for v in cfg_vals[name]]
            return out

        # Default opposed valuations for 2 traders and 5 issues:
        # trader_1 values issues 1..2 low [2, 3], issues 3..4 high [9, 8], issue 5 medium [5].
        # trader_2 values issues 1..2 high [8, 9], issues 3..4 low [3, 2], issue 5 medium [5].
        if self.num_agents == 2 and self.num_issues == 5:
            return {
                "trader_1": [2, 3, 9, 8, 5],
                "trader_2": [8, 9, 3, 2, 5],
            }

        # General N >= 2 or custom num_issues: cyclically opposed valuations.
        base_profile = [2, 3, 9, 8, 5]
        out = {}
        for idx, name in enumerate(self.agent_names):
            vals = []
            for k in range(self.num_issues):
                # Shift preferences by agent index so traders value different issues highest
                shifted = base_profile[(k + 2 * idx) % len(base_profile)]
                vals.append(int(shifted))
            out[name] = vals
        return out

    def _init_endowments(self) -> Dict[str, List[int]]:
        """Initialize status-quo unit endowments per issue (summing to total_units)."""
        cfg_endow = self.env_config.get("endowments")
        if isinstance(cfg_endow, dict):
            out: Dict[str, List[int]] = {}
            for name in self.agent_names:
                if name not in cfg_endow or len(cfg_endow[name]) != self.num_issues:
                    raise ValueError(
                        f"endowments[{name!r}] must be a list of length {self.num_issues}"
                    )
                out[name] = [int(u) for u in cfg_endow[name]]
            for k in range(self.num_issues):
                col_sum = sum(out[name][k] for name in self.agent_names)
                if col_sum != self.total_units:
                    raise ValueError(
                        f"endowments for issue_{k + 1} must sum to {self.total_units}, got {col_sum}"
                    )
            return out

        # Default for 2 traders, 5 issues:
        # trader_1 starts with all units of issues 1 & 2 (which it values low) and 50% of issue 5.
        # trader_2 starts with all units of issues 3 & 4 (which it values low) and 50% of issue 5.
        if self.num_agents == 2 and self.num_issues == 5:
            half_a = self.total_units // 2
            half_b = self.total_units - half_a
            return {
                "trader_1": [self.total_units, self.total_units, 0, 0, half_a],
                "trader_2": [0, 0, self.total_units, self.total_units, half_b],
            }

        # General N >= 2 or custom num_issues: assign each issue k to the trader with the
        # lowest valuation on issue k (creating strong gains from multi-issue exchange).
        out = {name: [0] * self.num_issues for name in self.agent_names}
        for k in range(self.num_issues):
            if k == self.num_issues - 1:
                base, rem = divmod(self.total_units, self.num_agents)
                for idx, name in enumerate(self.agent_names):
                    out[name][k] = base + (1 if idx < rem else 0)
            else:
                holder = min(
                    self.agent_names,
                    key=lambda n: (self.valuations[n][k], self.agent_names.index(n)),
                )
                out[holder][k] = self.total_units
        return out

    # ------------------------------------------------------------------ #
    # Agenda helpers
    # ------------------------------------------------------------------ #

    def active_issue_indices(self, iteration: int) -> List[int]:
        """Return the 0-based issue indices negotiated in a given 1-based iteration."""
        if self.agenda_mode == "package":
            return list(self.agenda_order)
        # Sequential mode: iteration t settles the (t-1)-th issue in agenda_order (wrapping if > num_issues)
        pos = (max(1, int(iteration)) - 1) % self.num_issues
        return [self.agenda_order[pos]]

    def _parse_allocations_arg(
        self, arguments: Dict[str, Any], active_indices: List[int]
    ) -> Tuple[Optional[List[int]], Optional[str]]:
        """Validate and normalize an allocations list from tool arguments."""
        raw = arguments.get("allocations")
        if raw is None:
            raw = arguments.get("my_units")
        if not isinstance(raw, (list, tuple)):
            return None, "allocations must be a list of integers"

        try:
            vals = [int(x) for x in raw]
        except (TypeError, ValueError):
            return None, "allocations must contain only integers"

        if any(isinstance(x, bool) for x in raw):
            return None, "allocations must contain integers, not booleans"

        if len(vals) not in (len(active_indices), self.num_issues):
            if len(active_indices) == 1 and self.num_issues != 1:
                return (
                    None,
                    f"allocations must have length 1 (for active issue) or {self.num_issues}",
                )
            return None, f"allocations must have length {self.num_issues}"

        for v in vals:
            if v < 0 or v > self.total_units:
                return None, f"each allocation must be between 0 and {self.total_units}"

        return vals, None

    def _expand_to_full_vector(
        self, agent_name: str, vals: List[int], active_indices: List[int]
    ) -> List[int]:
        """Expand a 1-issue sequential vector or full vector into length num_issues."""
        if len(vals) == self.num_issues:
            return list(vals)
        full = list(self.endowments[agent_name])
        for pos, issue_idx in enumerate(active_indices):
            full[issue_idx] = vals[pos]
        return full

    # ------------------------------------------------------------------ #
    # Action recording (called by the tools)
    # ------------------------------------------------------------------ #

    def record_proposal(
        self, agent_name: str, arguments: Dict[str, Any], iteration: Optional[int] = None
    ) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown trader {agent_name}"}

        iter_num = int(iteration) if iteration is not None else (self.round_num + 1)
        active_indices = self.active_issue_indices(iter_num)
        vals, err = self._parse_allocations_arg(arguments, active_indices)
        if err is not None or vals is None:
            return {"status": "failed", "reason": err or "invalid allocations"}

        full_vals = self._expand_to_full_vector(agent_name, vals, active_indices)
        active_demands = {f"issue_{k + 1}": full_vals[k] for k in active_indices}

        # Track who proposes a package deal first
        is_multi_issue_proposal = len(active_indices) > 1 or len(vals) > 1
        if is_multi_issue_proposal:
            if self.agenda_setter is None:
                self.agenda_setter = agent_name
            if self.round_agenda_setter is None:
                self.round_agenda_setter = agent_name

        entry: Dict[str, Any] = {
            "iteration": iter_num,
            "allocations": full_vals,
            "active_demands": active_demands,
        }
        message = str(arguments.get("message") or "").strip()
        if message:
            entry["message"] = message

        self.proposals[agent_name].append(entry)
        return {
            "status": "success",
            "proposal": entry,
            "agenda_setter": self.agenda_setter,
        }

    def commit_deal(
        self, agent_name: str, arguments: Dict[str, Any], iteration: Optional[int] = None
    ) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown trader {agent_name}"}

        iter_num = int(iteration) if iteration is not None else (self.round_num + 1)
        active_indices = self.active_issue_indices(iter_num)
        vals, err = self._parse_allocations_arg(arguments, active_indices)
        if err is not None or vals is None:
            return {"status": "failed", "reason": err or "invalid allocations"}

        full_vals = self._expand_to_full_vector(agent_name, vals, active_indices)
        self.pending_actions[agent_name] = {
            "action": "commit_deal",
            "allocations": full_vals,
        }
        return {
            "status": "success",
            "action": "commit_deal",
            "allocations": full_vals,
            "active_issues": [f"issue_{k + 1}" for k in active_indices],
        }

    # ------------------------------------------------------------------ #
    # Settlement mechanics (pure integer arithmetic)
    # ------------------------------------------------------------------ #

    def _settle_single_issue(
        self, issue_idx: int, demands: Dict[str, int]
    ) -> Dict[str, Any]:
        """Settle one issue k given each trader's demanded units."""
        total_demand = sum(int(demands[name]) for name in self.agent_names)
        agreed = total_demand <= self.total_units

        final_units: Dict[str, int] = {}
        if agreed:
            unclaimed = self.total_units - total_demand
            base_extra, rem = divmod(unclaimed, self.num_agents)
            for idx, name in enumerate(self.agent_names):
                extra = base_extra + (1 if idx < rem else 0)
                final_units[name] = int(demands[name]) + extra
        else:
            # Disagreement -> revert to status-quo endowment on this issue
            for name in self.agent_names:
                final_units[name] = int(self.endowments[name][issue_idx])

        issue_payoffs: Dict[str, int] = {
            name: final_units[name] * self.valuations[name][issue_idx]
            for name in self.agent_names
        }
        endowment_payoffs: Dict[str, int] = {
            name: self.endowments[name][issue_idx] * self.valuations[name][issue_idx]
            for name in self.agent_names
        }

        return {
            "issue_index": issue_idx,
            "issue": f"issue_{issue_idx + 1}",
            "demands": {name: int(demands[name]) for name in self.agent_names},
            "total_demand": total_demand,
            "agreed": agreed,
            "final_units": final_units,
            "payoffs": issue_payoffs,
            "endowment_payoffs": endowment_payoffs,
            "joint_payoff": sum(issue_payoffs.values()),
            "endowment_joint_payoff": sum(endowment_payoffs.values()),
        }

    def _settle_round(self, iteration: int) -> Dict[str, Any]:
        """Settle active issues for the current iteration."""
        active_indices = self.active_issue_indices(iteration)
        actions: Dict[str, Dict[str, Any]] = {}
        for name in self.agent_names:
            act = self.pending_actions.get(name)
            if act is None:
                # Neutral fallback: claim status-quo endowment
                act = {
                    "action": "commit_deal",
                    "allocations": list(self.endowments[name]),
                    "fallback": True,
                }
            actions[name] = act

        issue_records: Dict[str, Dict[str, Any]] = {}
        for issue_idx in active_indices:
            demands = {
                name: int(actions[name]["allocations"][issue_idx])
                for name in self.agent_names
            }
            rec = self._settle_single_issue(issue_idx, demands)
            issue_records[rec["issue"]] = rec

        payoffs: Dict[str, int] = {name: 0 for name in self.agent_names}
        endowment_payoffs: Dict[str, int] = {name: 0 for name in self.agent_names}
        final_allocations: Dict[str, Dict[str, int]] = {name: {} for name in self.agent_names}

        for issue_name, rec in issue_records.items():
            for name in self.agent_names:
                payoffs[name] += rec["payoffs"][name]
                endowment_payoffs[name] += rec["endowment_payoffs"][name]
                final_allocations[name][issue_name] = rec["final_units"][name]

        agreed_issues = [
            issue_name for issue_name, rec in issue_records.items() if rec["agreed"]
        ]
        disagreed_issues = [
            issue_name for issue_name, rec in issue_records.items() if not rec["agreed"]
        ]

        joint_payoff = sum(payoffs.values())
        endowment_joint = sum(endowment_payoffs.values())
        gains_from_trade = {
            name: payoffs[name] - endowment_payoffs[name] for name in self.agent_names
        }
        pareto_improving = (
            all(gains_from_trade[name] >= 0 for name in self.agent_names)
            and any(gains_from_trade[name] > 0 for name in self.agent_names)
        )

        # Detect whether a logrolling / package swap occurred across active issues:
        # at least one trader gave up units relative to endowment on one issue and gained on another.
        package_swap_occurred = False
        if len(active_indices) > 1:
            for name in self.agent_names:
                gave_up = any(
                    issue_records[f"issue_{k + 1}"]["final_units"][name]
                    < self.endowments[name][k]
                    for k in active_indices
                )
                gained = any(
                    issue_records[f"issue_{k + 1}"]["final_units"][name]
                    > self.endowments[name][k]
                    for k in active_indices
                )
                if gave_up and gained:
                    package_swap_occurred = True
                    break

        record = {
            "round": iteration,
            "agenda_mode": self.agenda_mode,
            "active_issues": [f"issue_{k + 1}" for k in active_indices],
            "round_agenda_setter": self.round_agenda_setter,
            "actions": {n: dict(a) for n, a in actions.items()},
            "issues": issue_records,
            "agreed_issues": agreed_issues,
            "disagreed_issues": disagreed_issues,
            "agreement_fraction": (
                round(len(agreed_issues) / len(active_indices), 3)
                if active_indices
                else None
            ),
            "final_allocations": final_allocations,
            "payoffs": payoffs,
            "endowment_payoffs": endowment_payoffs,
            "gains_from_trade": gains_from_trade,
            "joint_payoff": joint_payoff,
            "endowment_joint_payoff": endowment_joint,
            "joint_surplus_gained": joint_payoff - endowment_joint,
            "pareto_improving": pareto_improving,
            "package_swap_occurred": package_swap_occurred,
        }
        return record

    # ------------------------------------------------------------------ #
    # Analytic benchmarks (never from the LLM)
    # ------------------------------------------------------------------ #

    def _single_episode_endowment(self) -> Dict[str, Any]:
        """Status-quo endowment outcome for one full 5-issue set."""
        payoffs = {
            name: sum(
                self.endowments[name][k] * self.valuations[name][k]
                for k in range(self.num_issues)
            )
            for name in self.agent_names
        }
        return {
            "allocations": {name: list(self.endowments[name]) for name in self.agent_names},
            "payoffs": payoffs,
            "joint_payoff": sum(payoffs.values()),
        }

    def _single_episode_equal_split(self) -> Dict[str, Any]:
        """50/50 integer split on every issue for one full 5-issue set."""
        allocs: Dict[str, List[int]] = {name: [] for name in self.agent_names}
        for _k in range(self.num_issues):
            base, rem = divmod(self.total_units, self.num_agents)
            for idx, name in enumerate(self.agent_names):
                allocs[name].append(base + (1 if idx < rem else 0))
        payoffs = {
            name: sum(
                allocs[name][k] * self.valuations[name][k]
                for k in range(self.num_issues)
            )
            for name in self.agent_names
        }
        return {
            "allocations": allocs,
            "payoffs": payoffs,
            "joint_payoff": sum(payoffs.values()),
        }

    def _single_episode_pareto_optimum(self) -> Dict[str, Any]:
        """Maximum joint welfare package allocation (individually rational, Nash-tiebroken)."""
        endow_episode = self._single_episode_endowment()
        endow_payoffs = endow_episode["payoffs"]

        # Fast exact enumeration for 2 traders and moderate num_issues, or greedy/exact per-issue
        # Note: unconstrained welfare max assigns all units of issue k to argmax_i v_{i, k}.
        # Check if assigning each issue k to the highest-valuation trader is already IR!
        unconstrained_allocs: Dict[str, List[int]] = {
            name: [0] * self.num_issues for name in self.agent_names
        }
        for k in range(self.num_issues):
            max_v = max(self.valuations[name][k] for name in self.agent_names)
            winners = [
                name for name in self.agent_names if self.valuations[name][k] == max_v
            ]
            if len(winners) == 1:
                unconstrained_allocs[winners[0]][k] = self.total_units
            else:
                # Tie in highest valuation: split evenly among tied winners (matching endowment if all tied)
                base, rem = divmod(self.total_units, len(winners))
                for idx, w in enumerate(winners):
                    unconstrained_allocs[w][k] = base + (1 if idx < rem else 0)

        unconstrained_payoffs = {
            name: sum(
                unconstrained_allocs[name][k] * self.valuations[name][k]
                for k in range(self.num_issues)
            )
            for name in self.agent_names
        }
        if all(
            unconstrained_payoffs[name] >= endow_payoffs[name]
            for name in self.agent_names
        ):
            nash_prod = 1
            for name in self.agent_names:
                nash_prod *= max(1, unconstrained_payoffs[name] - endow_payoffs[name])
            return {
                "allocations": unconstrained_allocs,
                "payoffs": unconstrained_payoffs,
                "joint_payoff": sum(unconstrained_payoffs.values()),
                "nash_product": nash_prod,
            }

        # If unconstrained optimum violates IR for some trader (e.g. heavily skewed valuations),
        # search over integer allocations for 2 traders to maximize (joint_payoff, nash_product).
        if self.num_agents == 2 and (self.total_units + 1) ** self.num_issues <= 200000:
            t1, t2 = self.agent_names[0], self.agent_names[1]
            best_tuple: Optional[Tuple[int, int]] = None
            best_alloc_t1: List[int] = list(self.endowments[t1])
            for vec in itertools.product(range(self.total_units + 1), repeat=self.num_issues):
                p1 = sum(vec[k] * self.valuations[t1][k] for k in range(self.num_issues))
                if p1 < endow_payoffs[t1]:
                    continue
                p2 = sum(
                    (self.total_units - vec[k]) * self.valuations[t2][k]
                    for k in range(self.num_issues)
                )
                if p2 < endow_payoffs[t2]:
                    continue
                joint = p1 + p2
                nash = (p1 - endow_payoffs[t1]) * (p2 - endow_payoffs[t2])
                cand = (joint, nash)
                if best_tuple is None or cand > best_tuple:
                    best_tuple = cand
                    best_alloc_t1 = list(vec)

            allocs = {
                t1: best_alloc_t1,
                t2: [self.total_units - x for x in best_alloc_t1],
            }
            payoffs = {
                name: sum(
                    allocs[name][k] * self.valuations[name][k]
                    for k in range(self.num_issues)
                )
                for name in self.agent_names
            }
            return {
                "allocations": allocs,
                "payoffs": payoffs,
                "joint_payoff": sum(payoffs.values()),
                "nash_product": best_tuple[1] if best_tuple else 0,
            }

        return {
            "allocations": unconstrained_allocs,
            "payoffs": unconstrained_payoffs,
            "joint_payoff": sum(unconstrained_payoffs.values()),
            "nash_product": 0,
        }

    def _scale_benchmark_to_horizon(self, single_episode: Dict[str, Any]) -> Dict[str, Any]:
        """Scale single-episode benchmark across max_iterations according to agenda_mode."""
        allocs = single_episode["allocations"]
        total_payoffs = {name: 0 for name in self.agent_names}
        for iteration in range(1, self.max_iterations + 1):
            for k in self.active_issue_indices(iteration):
                for name in self.agent_names:
                    total_payoffs[name] += allocs[name][k] * self.valuations[name][k]
        return {
            "per_episode_allocations": allocs,
            "per_episode_payoffs": dict(single_episode["payoffs"]),
            "per_episode_joint_payoff": int(single_episode["joint_payoff"]),
            "payoffs": total_payoffs,
            "joint_payoff": sum(total_payoffs.values()),
        }

    def endowment_benchmark(self) -> Dict[str, Any]:
        """Status-quo no-trade payoff benchmark across the simulation horizon."""
        return self._scale_benchmark_to_horizon(self._single_episode_endowment())

    def equal_split_benchmark(self) -> Dict[str, Any]:
        """Issue-by-issue 50/50 split benchmark across the simulation horizon."""
        return self._scale_benchmark_to_horizon(self._single_episode_equal_split())

    def pareto_package_optimum_benchmark(self) -> Dict[str, Any]:
        """Maximum joint welfare package deal benchmark across the simulation horizon."""
        return self._scale_benchmark_to_horizon(self._single_episode_pareto_optimum())

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        mode_desc = (
            "all issues simultaneously as a package deal each round"
            if self.agenda_mode == "package"
            else "one issue per round in sequential agenda order"
        )
        return (
            f"You are {self.num_agents} traders negotiating over {self.num_issues} resource "
            f"issues (issue_1..issue_{self.num_issues}), each with {self.total_units} integer "
            f"units available. Negotiation proceeds over {self.max_iterations} rounds ({mode_desc}). "
            f"Each trader has private per-unit valuations and a status-quo endowment. On each "
            f"active issue, if the sum of units claimed by all traders is at most {self.total_units}, "
            f"every trader receives their claimed units (plus an equal share of any unclaimed units). "
            f"If total claims exceed {self.total_units}, that issue reverts to the status-quo endowment. "
            f"The blackboard is your shared communication channel."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        active_indices = self.active_issue_indices(iteration)
        active_issues = [f"issue_{k + 1}" for k in active_indices]
        last_record = self.round_history[-1] if self.round_history else None
        endow_episode = self._single_episode_endowment()
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round": iteration,
            "max_rounds": self.max_iterations,
            "agenda_mode": self.agenda_mode,
            "agenda_order": [f"issue_{k + 1}" for k in self.agenda_order],
            "active_issues": active_issues,
            "active_issue_indices": active_indices,
            "num_issues": self.num_issues,
            "total_units": self.total_units,
            "own_valuations": list(self.valuations[agent_name]),
            "own_endowment": list(self.endowments[agent_name]),
            "all_endowments": {n: list(self.endowments[n]) for n in self.agent_names},
            "own_endowment_payoff_per_episode": endow_episode["payoffs"][agent_name],
            "own_payoff_history": [
                int(r["payoffs"].get(agent_name, 0)) for r in self.round_history
            ],
            "last_round": last_record,
            "agenda_setter": self.agenda_setter,
            "legal_actions": ["propose_package"] if phase == "planning" else ["commit_deal"],
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        return float(self.pareto_package_optimum_benchmark()["joint_payoff"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["joint_payoff"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["payoffs"].get(agent_name, 0))

    def log_iteration(self, iteration: int) -> None:
        record = self._settle_round(iteration)
        self.round_history.append(record)
        self.round_num = iteration
        self.pending_actions = {name: None for name in self.agent_names}
        self.round_agenda_setter = None
        self.joint_reward_history.append(record["joint_payoff"])
        logger.info(
            "=== %s Round %s (%s) === agreed=%s payoffs=%s joint=%s swap=%s",
            self.__class__.__name__,
            record["round"],
            record["agenda_mode"],
            record["agreed_issues"],
            record["payoffs"],
            record["joint_payoff"],
            record["package_swap_occurred"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        joint_total = sum(r["joint_payoff"] for r in self.round_history)
        endow_bm = self.endowment_benchmark()
        equal_bm = self.equal_split_benchmark()
        pareto_bm = self.pareto_package_optimum_benchmark()

        endow_joint = endow_bm["joint_payoff"]
        pareto_joint = pareto_bm["joint_payoff"]
        max_surplus = pareto_joint - endow_joint
        realized_surplus = joint_total - endow_joint
        trade_efficiency = (
            round(realized_surplus / max_surplus, 3) if max_surplus > 0 else 0.0
        )

        total_active = sum(len(r["active_issues"]) for r in self.round_history)
        total_agreed = sum(len(r["agreed_issues"]) for r in self.round_history)

        return {
            "environment": self.__class__.__name__,
            "rounds_simulated": len(self.round_history),
            "agenda_mode": self.agenda_mode,
            "agenda_order": [f"issue_{k + 1}" for k in self.agenda_order],
            "agenda_setter": self.agenda_setter,
            "traders": {
                name: {
                    "valuations": list(self.valuations[name]),
                    "endowment": list(self.endowments[name]),
                    "total_payoff": sum(
                        int(r["payoffs"].get(name, 0)) for r in self.round_history
                    ),
                    "endowment_benchmark_payoff": endow_bm["payoffs"][name],
                    "gain_over_endowment": sum(
                        int(r["payoffs"].get(name, 0)) for r in self.round_history
                    )
                    - endow_bm["payoffs"][name],
                    "proposals_made": len(self.proposals[name]),
                }
                for name in self.agent_names
            },
            "round_history": self.round_history,
            "trade_stats": {
                "issues_negotiated": total_active,
                "issues_agreed": total_agreed,
                "agreement_rate": (
                    round(total_agreed / total_active, 3) if total_active else None
                ),
                "pareto_improving_rounds": sum(
                    1 for r in self.round_history if r["pareto_improving"]
                ),
                "package_swap_rounds": sum(
                    1 for r in self.round_history if r["package_swap_occurred"]
                ),
                "realized_surplus": realized_surplus,
                "max_possible_surplus": max_surplus,
                "trade_efficiency": trade_efficiency,
            },
            "benchmarks": {
                "endowment": endow_bm,
                "equal_split": equal_bm,
                "pareto_package_optimum": pareto_bm,
            },
            "joint_reward": joint_total,
            "max_joint_reward": self.max_joint_reward,
            "joint_reward_vs_optimum": (
                round(joint_total / self.max_joint_reward, 3)
                if self.max_joint_reward
                else None
            ),
        }
