"""NetworkRouting: repeated selfish routing over parallel congestible links.

Each iteration, every router must send its ``units_per_agent`` traffic units
from source to sink over exactly one of E parallel links. A link's per-unit
latency is affine in the total load placed on it::

    latency(link, load) = base_latency + congestion_rate * load

where ``load`` counts the units of ALL routers on that link this iteration
(congestion externality). During planning, routers may talk on the blackboard
and publicly float route proposals (non-binding). During execution each router
commits one link with ``commit_route``.

Analytic benchmarks (never LLM-derived), all over single-iteration profiles:
- OPTIMAL: the assignment of routers to links minimising total latency
  (sum over links of load * latency(load)); computed by full enumeration over
  E^N profiles (N and E are tiny).
- GREEDY (all-shortest-path): every router picks the free-flow (uncongested)
  shortest link ignoring congestion.
- NASH: pure-strategy equilibrium found by deterministic best-response
  dynamics started from the greedy profile.
- Price of anarchy = NASH total cost / OPTIMAL total cost (greedy ratio also
  reported).

Honest prompts only: any load-spreading that emerges is tacit, not instructed.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any, Dict, List, Mapping, Optional

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .network_routing_prompts import NetworkRoutingPrompts

logger = logging.getLogger(__name__)

DEFAULT_LINKS: List[Dict[str, Any]] = [
    {"name": "link_alpha", "base_latency": 10.0, "congestion_rate": 1.2},
    {"name": "link_beta", "base_latency": 28.0, "congestion_rate": 0.6},
    {"name": "link_gamma", "base_latency": 55.0, "congestion_rate": 0.3},
    {"name": "link_delta", "base_latency": 75.0, "congestion_rate": 0.0},
]


class NetworkRoutingEnvironment(AbstractEnvironment):
    """Repeated selfish routing over parallel links with congestion."""

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
            raise ValueError("NetworkRouting needs at least 2 routers")

        # Topology: ordered list of parallel source->sink links.
        raw_links = self.env_config.get("links") or DEFAULT_LINKS
        self.links: List[Dict[str, Any]] = []
        seen = set()
        for entry in raw_links:
            name = str(entry["name"])
            if name in seen:
                raise ValueError(f"duplicate link name: {name}")
            seen.add(name)
            base = float(entry.get("base_latency", 0.0))
            rate = float(entry.get("congestion_rate", 0.0))
            if base < 0 or rate < 0:
                raise ValueError("base_latency and congestion_rate must be >= 0")
            self.links.append({"name": name, "base_latency": base, "congestion_rate": rate})
        if not self.links:
            raise ValueError("NetworkRouting needs at least one link")
        self.link_names: List[str] = [link["name"] for link in self.links]
        self.link_index: Dict[str, int] = {name: i for i, name in enumerate(self.link_names)}
        self.units_per_agent = int(self.env_config.get("units_per_agent", 6))

        self.agent_names = [f"router_{idx + 1}" for idx in range(self.num_agents)]

        # Per-iteration state.
        self.iteration = 0
        self.proposals: Dict[str, List[Optional[str]]] = {name: [] for name in self.agent_names}
        self.committed_links: Dict[str, List[Optional[str]]] = {
            name: [] for name in self.agent_names
        }
        # iteration records: {"iteration", "choices", "loads", "latencies",
        #                     "agent_costs", "total_cost"}
        self.routing_history: List[Dict[str, Any]] = []

        self.prompts = NetworkRoutingPrompts(self, self.full_config)

        # Analytic benchmarks (deterministic, topology-only).
        self.benchmarks = self.compute_benchmarks()
        self.joint_reward_history: List[float] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s routers, units_per_agent=%s, links=%s, "
            "optimal_cost=%s nash_cost=%s greedy_cost=%s",
            self.__class__.__name__,
            self.num_agents,
            self.units_per_agent,
            self.link_names,
            self.benchmarks["optimal_total_cost"],
            self.benchmarks["nash_total_cost"],
            self.benchmarks["greedy_total_cost"],
        )

    # ------------------------------------------------------------------ #
    # Routing mechanics
    # ------------------------------------------------------------------ #

    def latency(self, link_name: str, load: float) -> float:
        """Affine congestion: base + rate * load (load = ALL units on link)."""
        link = self.links[self.link_index[link_name]]
        return round(link["base_latency"] + link["congestion_rate"] * float(load), 6)

    def free_flow_shortest_link(self) -> str:
        """Link with minimal base latency (ties broken by link order)."""
        return min(self.link_names, key=lambda name: (self.links[self.link_index[name]]["base_latency"], self.link_index[name]))

    def record_proposal(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        link = arguments.get("link")
        if link is None:
            return {"status": "failed", "reason": "link is required"}
        link = str(link)
        if link not in self.link_index:
            return {
                "status": "failed",
                "reason": f"unknown link '{link}'; valid links: {self.link_names}",
            }
        self.proposals[agent_name].append(link)
        return {"status": "success", "proposed_link": link}

    def commit_route(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        link = arguments.get("link")
        if link is None:
            return {"status": "failed", "reason": "link is required"}
        link = str(link)
        if link not in self.link_index:
            return {
                "status": "failed",
                "reason": f"unknown link '{link}'; valid links: {self.link_names}",
            }
        # Keep exactly one pending commit per iteration.
        it_index = len(self.routing_history)
        while len(self.committed_links[agent_name]) <= it_index:
            self.committed_links[agent_name].append(None)
        self.committed_links[agent_name][it_index] = link
        return {"status": "success", "committed_link": link}

    def _settle_round(self, it_index: int) -> Dict[str, Any]:
        """Resolve link choices into loads, latencies and costs."""
        choices: Dict[str, str] = {}
        for name in self.agent_names:
            history = self.committed_links[name]
            link = history[it_index] if it_index < len(history) else None
            if link is None:
                # Fallback: repeat previous iteration's link, else the
                # free-flow shortest link.
                prev = history[it_index - 1] if it_index > 0 else None
                link = prev if prev is not None else self.free_flow_shortest_link()
                while len(history) <= it_index:
                    history.append(None)
                history[it_index] = link
            choices[name] = link

        counts: Dict[str, int] = {name: 0 for name in self.link_names}
        for link in choices.values():
            counts[link] += 1
        loads = {name: counts[name] * self.units_per_agent for name in self.link_names}
        latencies = {name: self.latency(name, loads[name]) for name in self.link_names}
        agent_costs = {name: latencies[choices[name]] for name in self.agent_names}
        total_cost = round(
            sum(loads[link] * latencies[link] for link in self.link_names),
            6,
        )
        record = {
            "iteration": it_index + 1,
            "choices": choices,
            "counts": counts,
            "loads": loads,
            "latencies": latencies,
            "agent_costs": agent_costs,
            "total_cost": total_cost,
        }
        return record

    # ------------------------------------------------------------------ #
    # Analytic benchmarks
    # ------------------------------------------------------------------ #

    def _profile_costs(self, assignment: List[str]) -> Dict[str, Any]:
        """Loads, latencies and total cost for a full agent->link assignment."""
        counts = {name: 0 for name in self.link_names}
        for link in assignment:
            counts[link] += 1
        loads = {name: counts[name] * self.units_per_agent for name in self.link_names}
        latencies = {name: self.latency(name, loads[name]) for name in self.link_names}
        total_cost = round(
            sum(loads[name] * latencies[name] for name in self.link_names), 6
        )
        return {"counts": counts, "loads": loads, "latencies": latencies, "total_cost": total_cost}

    def _optimal_assignment(self) -> List[str]:
        """Global minimum total latency; exhaustive over E^N profiles (tiny)."""
        best: Optional[List[str]] = None
        best_cost: Optional[float] = None
        for assignment in itertools.product(self.link_names, repeat=self.num_agents):
            cost = self._profile_costs(list(assignment))["total_cost"]
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = list(assignment)
        assert best is not None
        return best

    def _greedy_assignment(self) -> List[str]:
        """All-shortest-path: everyone on the free-flow shortest link."""
        return [self.free_flow_shortest_link() for _ in self.agent_names]

    def _nash_assignment(self) -> List[str]:
        """Pure Nash by deterministic best-response dynamics from greedy."""
        assignment = self._greedy_assignment()
        for _ in range(100 * self.num_agents):
            changed = False
            for i, name in enumerate(self.agent_names):
                current = assignment[i]
                counts = {link: 0 for link in self.link_names}
                for j, other in enumerate(assignment):
                    if j != i:
                        counts[other] += 1
                # Latency the agent would experience on each link after moving.
                best_link = min(
                    self.link_names,
                    key=lambda link: (
                        self.latency(link, (counts[link] + 1) * self.units_per_agent),
                        self.link_index[link],
                    ),
                )
                if best_link != current:
                    assignment[i] = best_link
                    changed = True
            if not changed:
                break
        return assignment

    def compute_benchmarks(self) -> Dict[str, Any]:
        optimal = self._profile_costs(self._optimal_assignment())
        greedy = self._profile_costs(self._greedy_assignment())
        nash = self._profile_costs(self._nash_assignment())
        optimal_total = optimal["total_cost"]
        return {
            "optimal_counts": optimal["counts"],
            "optimal_total_cost": optimal_total,
            "greedy_counts": greedy["counts"],
            "greedy_total_cost": greedy["total_cost"],
            "nash_counts": nash["counts"],
            "nash_total_cost": nash["total_cost"],
            "price_of_anarchy": (
                round(nash["total_cost"] / optimal_total, 6) if optimal_total else None
            ),
            "greedy_over_optimal": (
                round(greedy["total_cost"] / optimal_total, 6) if optimal_total else None
            ),
        }

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        return (
            f"You are traffic routers in a common network. There are {self.num_agents} routers. "
            f"Each iteration every router sends {self.units_per_agent} units of traffic from "
            f"source to sink over exactly one of {len(self.links)} parallel links. A link's "
            f"per-unit latency depends on the TOTAL units placed on it by all routers that "
            f"iteration. The blackboard is your public communication channel."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        last_record = self.routing_history[-1] if self.routing_history else None
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "max_iterations": self.max_iterations,
            "units_per_agent": self.units_per_agent,
            "links": [dict(link) for link in self.links],
            "own_route_history": list(self.committed_links[agent_name]),
            "own_latency_history": [
                float(r["agent_costs"].get(agent_name, 0.0)) for r in self.routing_history
            ],
            "last_iteration": last_record,
            "legal_actions": ["propose_route"] if phase == "planning" else ["commit_route"],
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        # Rewards are negative costs: best joint reward = -optimal total cost.
        return -float(self.benchmarks["optimal_total_cost"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.routing_history:
            return 0.0
        return -float(self.routing_history[-1]["total_cost"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.routing_history:
            return 0.0
        return -float(self.routing_history[-1]["agent_costs"].get(agent_name, 0.0))

    def log_iteration(self, iteration: int) -> None:
        it_index = iteration - 1
        record = self._settle_round(it_index)
        self.routing_history.append(record)
        self.iteration = iteration
        self.joint_reward_history.append(-record["total_cost"])
        logger.info(
            "=== %s Iteration %s === choices=%s loads=%s latencies=%s total_cost=%s",
            self.__class__.__name__,
            record["iteration"],
            record["choices"],
            record["loads"],
            record["latencies"],
            record["total_cost"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        avg_total_cost = (
            round(sum(r["total_cost"] for r in self.routing_history) / len(self.routing_history), 6)
            if self.routing_history
            else None
        )
        per_agent_latency = {
            name: [
                float(r["agent_costs"].get(name, 0.0)) for r in self.routing_history
            ]
            for name in self.agent_names
        }
        all_latencies = [lat for hist in per_agent_latency.values() for lat in hist]
        avg_agent_latency = (
            round(sum(all_latencies) / len(all_latencies), 6) if all_latencies else None
        )
        link_usage = {
            name: sum(r["counts"].get(name, 0) for r in self.routing_history)
            for name in self.link_names
        }
        optimal_total = self.benchmarks["optimal_total_cost"]
        return {
            "environment": self.__class__.__name__,
            "iterations_simulated": len(self.routing_history),
            "routers": {
                name: {
                    "route_history": list(self.committed_links[name]),
                    "mean_latency": (
                        round(sum(per_agent_latency[name]) / len(per_agent_latency[name]), 6)
                        if per_agent_latency[name]
                        else None
                    ),
                }
                for name in self.agent_names
            },
            "routing_history": self.routing_history,
            "link_usage": link_usage,
            "routing_stats": {
                "average_total_cost_per_iteration": avg_total_cost,
                "average_agent_latency": avg_agent_latency,
                "optimal_total_cost": optimal_total,
                "greedy_total_cost": self.benchmarks["greedy_total_cost"],
                "nash_total_cost": self.benchmarks["nash_total_cost"],
                "price_of_anarchy_benchmark": self.benchmarks["price_of_anarchy"],
                "realized_over_optimal": (
                    round(avg_total_cost / optimal_total, 6)
                    if avg_total_cost is not None and optimal_total
                    else None
                ),
            },
            "joint_reward": (
                -self.routing_history[-1]["total_cost"] if self.routing_history else 0.0
            ),
            "benchmarks": self.benchmarks,
        }
