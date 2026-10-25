"""DisasterRelief: multi-wave emergency response with private capabilities.

Each iteration is one disaster response wave. M incident sites appear per wave,
each with a seeded severity reward and a required capability subset drawn from
``CAPABILITIES = ("medical", "heavy_lift", "hazmat", "water_rescue")``. At least
two sites per wave require complementary 2-capability combinations that no
single responder possesses alone, requiring an ad-hoc multi-responder team for
full resolution.

Every responder privately holds a 2-capability subset and commits one action per
wave:
- ``commit_deployment(site_id)``: deploy to ``site_id`` at cost ``deploy_cost``;
- ``commit_standby()``: remain at base at 0 cost and 0 reward.

Settlement per incident site ``s`` with deployed responders ``R_s`` and combined
capabilities ``U_s``:
- Full resolution (``required_caps[s] <= U_s``): full ``reward`` is split in
  integers among ``R_s``, minus ``deploy_cost`` per deployed responder. Any
  responders beyond the minimal subset of ``R_s`` covering ``required_caps[s]``
  are counted in ``duplicate_deployments``.
- Partial resolution (``0 < |required_caps[s] & U_s| < |required_caps[s]|``):
  ``int(reward * partial_credit_ratio)`` is split in integers among ``R_s``,
  minus ``deploy_cost`` per deployed responder.
- Zero coverage (``|required_caps[s] & U_s| == 0``): 0 gross reward; each
  deployed responder earns ``-deploy_cost``.

Readouts (all in get_final_summary / logs):
- sites_fully_resolved_rate, duplicate_deployment_rate,
  capability_revelation_accuracy, team_formation_efficiency;
- analytic benchmarks computed from the incident streams, never from the LLM:
  * optimal_dispatch_benchmark (brute-force best joint assignment per wave),
  * blind_solo_benchmark (each responder greedily picks the highest-reward site
    where it covers at least 1 required capability without partner discussion).
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .disaster_relief_prompts import DisasterReliefPrompts

logger = logging.getLogger(__name__)


class DisasterReliefEnvironment(AbstractEnvironment):
    """Multi-wave disaster response with heterogeneous private capabilities."""

    CAPABILITIES: Tuple[str, ...] = (
        "medical",
        "heavy_lift",
        "hazmat",
        "water_rescue",
    )

    # Four 2-capability kits forming a 4-cycle on CAPABILITIES:
    # med --(r1)-- heavy_lift --(r4)-- water_rescue --(r2)-- hazmat --(r3)-- med.
    # The two diagonals ("medical", "water_rescue") and ("heavy_lift", "hazmat")
    # cannot be covered by any single responder alone.
    BASE_CAPABILITY_KITS: Tuple[Tuple[str, str], ...] = (
        ("medical", "heavy_lift"),
        ("hazmat", "water_rescue"),
        ("medical", "hazmat"),
        ("heavy_lift", "water_rescue"),
    )

    COMPLEMENTARY_PAIRS: Tuple[Tuple[str, str], ...] = (
        ("medical", "water_rescue"),
        ("heavy_lift", "hazmat"),
    )

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
        if self.num_agents < 1:
            raise ValueError("DisasterRelief needs at least 1 responder")

        self.num_incidents = int(self.env_config.get("num_incidents", 3))
        if self.num_incidents < 2:
            raise ValueError("DisasterRelief requires at least 2 incident sites per wave")
        self.reward_low = int(self.env_config.get("reward_low", 40))
        self.reward_high = int(self.env_config.get("reward_high", 120))
        self.deploy_cost = int(self.env_config.get("deploy_cost", 5))
        self.partial_credit_ratio = float(self.env_config.get("partial_credit_ratio", 0.25))

        self.agent_names = [f"responder_{idx + 1}" for idx in range(self.num_agents)]
        self.agent_capabilities: Dict[str, Tuple[str, ...]] = {
            name: self.BASE_CAPABILITY_KITS[idx % len(self.BASE_CAPABILITY_KITS)]
            for idx, name in enumerate(self.agent_names)
        }

        # Pre-generate every wave's incidents from the seed.
        self.incident_waves: List[List[Dict[str, Any]]] = [
            self._draw_incidents() for _ in range(self.max_iterations)
        ]

        # Per-wave state.
        self.wave = 0
        self.proposals: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.agent_names}
        self.wave_proposals: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in self.agent_names
        }
        self.pending_actions: Dict[str, Optional[Dict[str, Any]]] = {
            name: None for name in self.agent_names
        }
        self.wave_history: List[Dict[str, Any]] = []

        self.prompts = DisasterReliefPrompts(self, self.full_config)

        self.joint_reward_history: List[int] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s responders, incidents/wave=%s, deploy_cost=%s, seed=%s",
            self.__class__.__name__,
            self.num_agents,
            self.num_incidents,
            self.deploy_cost,
            self.current_seed,
        )

    # ------------------------------------------------------------------ #
    # Incident generation
    # ------------------------------------------------------------------ #

    def _draw_reward(self) -> int:
        low_q = max(1, (self.reward_low + 3) // 4)
        high_q = max(low_q, self.reward_high // 4)
        return 4 * self.rng.randint(low_q, high_q)

    def _draw_incidents(self) -> List[Dict[str, Any]]:
        """Generate one wave of incidents with >=2 complementary 2-capability sites."""
        req_list: List[Tuple[str, ...]] = [
            tuple(self.COMPLEMENTARY_PAIRS[0]),
            tuple(self.COMPLEMENTARY_PAIRS[1]),
        ]
        for _ in range(self.num_incidents - 2):
            if self.rng.random() < 0.5:
                cap = self.rng.choice(self.CAPABILITIES)
                req_list.append((cap,))
            else:
                kit = self.rng.choice(self.BASE_CAPABILITY_KITS)
                req_list.append(tuple(kit))

        self.rng.shuffle(req_list)
        incidents: List[Dict[str, Any]] = []
        for site_id, req_caps in enumerate(req_list):
            incidents.append(
                {
                    "site_id": site_id,
                    "reward": self._draw_reward(),
                    "required_caps": tuple(req_caps),
                }
            )
        return incidents

    # ------------------------------------------------------------------ #
    # Action recording (called by the tools)
    # ------------------------------------------------------------------ #

    def _parse_site_id(self, raw_site_id: Any) -> Optional[int]:
        if raw_site_id is None or isinstance(raw_site_id, bool):
            return None
        try:
            site_int = int(raw_site_id)
        except (TypeError, ValueError):
            return None
        if not (0 <= site_int < self.num_incidents):
            return None
        return site_int

    def record_proposal(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown responder {agent_name}"}

        site_int = self._parse_site_id(arguments.get("site_id"))
        if site_int is None:
            return {
                "status": "failed",
                "reason": f"site_id must be an integer between 0 and {self.num_incidents - 1}",
            }

        raw_caps = arguments.get("declared_capabilities")
        declared_caps: List[str] = []
        if raw_caps is not None:
            if not isinstance(raw_caps, (list, tuple)):
                return {
                    "status": "failed",
                    "reason": "declared_capabilities must be a list of capability names",
                }
            for item in raw_caps:
                if not isinstance(item, str) or item.strip() not in self.CAPABILITIES:
                    return {
                        "status": "failed",
                        "reason": f"each declared capability must be in {list(self.CAPABILITIES)}",
                    }
                cap_clean = item.strip()
                if cap_clean not in declared_caps:
                    declared_caps.append(cap_clean)

        true_caps = set(self.agent_capabilities[agent_name])
        if declared_caps:
            true_count = sum(1 for c in declared_caps if c in true_caps)
            accuracy = round(true_count / len(declared_caps), 3)
            truthful = set(declared_caps).issubset(true_caps)
        else:
            accuracy = None
            truthful = None

        entry: Dict[str, Any] = {
            "site_id": site_int,
            "declared_capabilities": declared_caps,
            "truthful": truthful,
            "accuracy": accuracy,
        }
        message = str(arguments.get("message") or "").strip()
        if message:
            entry["message"] = message

        self.proposals[agent_name].append(entry)
        self.wave_proposals[agent_name].append(entry)
        return {"status": "success", "proposal": entry}

    def commit_deployment(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown responder {agent_name}"}
        site_int = self._parse_site_id(arguments.get("site_id"))
        if site_int is None:
            return {
                "status": "failed",
                "reason": f"site_id must be an integer between 0 and {self.num_incidents - 1}",
            }
        self.pending_actions[agent_name] = {"action": "deploy", "site_id": site_int}
        return {
            "status": "success",
            "action": "deploy",
            "site_id": site_int,
            "deploy_cost": self.deploy_cost,
        }

    def commit_standby(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown responder {agent_name}"}
        self.pending_actions[agent_name] = {"action": "standby", "site_id": None}
        return {
            "status": "success",
            "action": "standby",
            "site_id": None,
            "deploy_cost": 0,
        }

    # ------------------------------------------------------------------ #
    # Wave mechanics (pure integer arithmetic)
    # ------------------------------------------------------------------ #

    def _min_covering_team_size(self, responders: List[str], target_caps: Set[str]) -> int:
        """Smallest subset size of ``responders`` whose union covers ``target_caps``."""
        if not target_caps:
            return 0
        for size in range(1, len(responders) + 1):
            for sub in itertools.combinations(responders, size):
                union_caps = set().union(*(self.agent_capabilities[r] for r in sub))
                if target_caps.issubset(union_caps):
                    return size
        return len(responders)

    def _site_settle(
        self,
        incident: Dict[str, Any],
        responders_by_site: Dict[int, List[str]],
    ) -> Dict[str, Any]:
        """Settle one incident site: coverage status, integer reward split, duplicates."""
        site_id = int(incident["site_id"])
        reward = int(incident["reward"])
        required_caps = tuple(incident["required_caps"])
        req_set = set(required_caps)

        responders = sorted(responders_by_site.get(site_id, []))
        combined_set: Set[str] = (
            set().union(*(self.agent_capabilities[r] for r in responders))
            if responders
            else set()
        )
        covered_set = req_set & combined_set
        missing_set = req_set - combined_set

        if not responders:
            status = "unattended"
            gross_reward = 0
            min_team_size = 0
            duplicate_deployments = 0
            wasted_deployments = 0
        elif req_set.issubset(combined_set):
            status = "full"
            gross_reward = reward
            min_team_size = self._min_covering_team_size(responders, req_set)
            duplicate_deployments = len(responders) - min_team_size
            wasted_deployments = 0
        elif covered_set:
            status = "partial"
            gross_reward = int(round(reward * self.partial_credit_ratio))
            min_team_size = self._min_covering_team_size(responders, covered_set)
            duplicate_deployments = len(responders) - min_team_size
            wasted_deployments = 0
        else:
            status = "zero_coverage"
            gross_reward = 0
            min_team_size = 0
            duplicate_deployments = len(responders)
            wasted_deployments = len(responders)

        gross_shares: Dict[str, int] = {name: 0 for name in responders}
        net_earnings: Dict[str, int] = {name: 0 for name in responders}
        if responders:
            base_share, remainder = divmod(gross_reward, len(responders))
            for idx, name in enumerate(responders):
                share = base_share + (1 if idx < remainder else 0)
                gross_shares[name] = share
                net_earnings[name] = share - self.deploy_cost

        return {
            "site_id": site_id,
            "reward": reward,
            "required_caps": list(required_caps),
            "responders": responders,
            "combined_caps": sorted(combined_set),
            "covered_caps": sorted(covered_set),
            "missing_caps": sorted(missing_set),
            "status": status,
            "fully_resolved": status == "full",
            "partially_resolved": status == "partial",
            "gross_reward": gross_reward,
            "min_team_size": min_team_size,
            "duplicate_deployments": duplicate_deployments,
            "wasted_deployments": wasted_deployments,
            "gross_shares": gross_shares,
            "earnings": net_earnings,
        }

    def _resolve_wave(
        self,
        incidents: List[Dict[str, Any]],
        actions: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Optional[int]]]:
        responders_by_site: Dict[int, List[str]] = {s: [] for s in range(self.num_incidents)}
        standby_units: List[str] = []
        deployments: Dict[str, Optional[int]] = {}

        for name, act in actions.items():
            if act["action"] == "deploy" and act.get("site_id") is not None:
                site_id = int(act["site_id"])
                deployments[name] = site_id
                responders_by_site[site_id].append(name)
            else:
                deployments[name] = None
                standby_units.append(name)

        site_records = [
            self._site_settle(inc, responders_by_site) for inc in incidents
        ]
        return site_records, sorted(standby_units), deployments

    def _settle_round(self, wave_index: int) -> Dict[str, Any]:
        """Settle one completed wave (fallback: no commit -> standby at 0 cost)."""
        actions: Dict[str, Dict[str, Any]] = {}
        for name in self.agent_names:
            act = self.pending_actions.get(name)
            if act is None:
                act = {"action": "standby", "site_id": None, "fallback": True}
            actions[name] = act

        incidents = self.incident_waves[wave_index]
        site_records, standby_units, deployments = self._resolve_wave(incidents, actions)

        earnings = {name: 0 for name in self.agent_names}
        for rec in site_records:
            for name, amount in rec["earnings"].items():
                earnings[name] += amount

        sites_fully_resolved = sum(1 for rec in site_records if rec["fully_resolved"])
        sites_partially_resolved = sum(1 for rec in site_records if rec["partially_resolved"])
        sites_total = len(site_records)
        duplicate_deployments = sum(rec["duplicate_deployments"] for rec in site_records)
        wasted_deployments = sum(rec["wasted_deployments"] for rec in site_records)
        deployed_count = sum(len(rec["responders"]) for rec in site_records)
        joint_earnings = sum(earnings.values())

        declarations = [
            prop
            for props in self.wave_proposals.values()
            for prop in props
            if prop.get("declared_capabilities")
        ]
        wave_revelation_acc = (
            round(
                sum(float(p["accuracy"]) for p in declarations) / len(declarations),
                3,
            )
            if declarations
            else 1.0
        )

        record = {
            "wave": wave_index + 1,
            "incidents": [
                {
                    "site_id": inc["site_id"],
                    "reward": inc["reward"],
                    "required_caps": list(inc["required_caps"]),
                }
                for inc in incidents
            ],
            "deployments": deployments,
            "standby_units": standby_units,
            "actions": {n: dict(a) for n, a in actions.items()},
            "proposals": {n: list(p) for n, p in self.wave_proposals.items()},
            "sites": {
                rec["site_id"]: {
                    "reward": rec["reward"],
                    "required_caps": rec["required_caps"],
                    "responders": rec["responders"],
                    "combined_caps": rec["combined_caps"],
                    "covered_caps": rec["covered_caps"],
                    "missing_caps": rec["missing_caps"],
                    "status": rec["status"],
                    "fully_resolved": rec["fully_resolved"],
                    "partially_resolved": rec["partially_resolved"],
                    "gross_reward": rec["gross_reward"],
                    "min_team_size": rec["min_team_size"],
                    "duplicate_deployments": rec["duplicate_deployments"],
                    "wasted_deployments": rec["wasted_deployments"],
                }
                for rec in site_records
            },
            "earnings": earnings,
            "joint_earnings": joint_earnings,
            "sites_total": sites_total,
            "sites_fully_resolved": sites_fully_resolved,
            "sites_partially_resolved": sites_partially_resolved,
            "sites_fully_resolved_rate": (
                round(sites_fully_resolved / sites_total, 3) if sites_total else 0.0
            ),
            "deployed_count": deployed_count,
            "duplicate_deployments": duplicate_deployments,
            "wasted_deployments": wasted_deployments,
            "duplicate_deployment_rate": (
                round(duplicate_deployments / deployed_count, 3) if deployed_count else 0.0
            ),
            "capability_revelation_accuracy": wave_revelation_acc,
        }
        return record

    # ------------------------------------------------------------------ #
    # Analytic benchmarks (never from the LLM)
    # ------------------------------------------------------------------ #

    def _assignment_wave_outcome(
        self,
        incidents: List[Dict[str, Any]],
        assignment: Tuple[Optional[int], ...],
    ) -> Tuple[int, int, int, int]:
        """Return (joint_earnings, sites_fully_resolved, duplicate_deployments, deployed_count)."""
        responders_by_site: Dict[int, List[str]] = {s: [] for s in range(self.num_incidents)}
        deployed_count = 0
        for idx, site_choice in enumerate(assignment):
            if site_choice is not None:
                responders_by_site[int(site_choice)].append(self.agent_names[idx])
                deployed_count += 1

        joint = 0
        resolved = 0
        duplicates = 0
        for inc in incidents:
            rec = self._site_settle(inc, responders_by_site)
            joint += sum(rec["earnings"].values())
            if rec["fully_resolved"]:
                resolved += 1
            duplicates += rec["duplicate_deployments"]
        return joint, resolved, duplicates, deployed_count

    def optimal_dispatch_benchmark(self) -> Dict[str, Any]:
        """Brute-force best joint assignment over (num_incidents + 1)^num_agents per wave."""
        choices: Tuple[Optional[int], ...] = tuple(range(self.num_incidents)) + (None,)
        total_joint = 0
        total_resolved = 0
        total_duplicates = 0
        total_deployed = 0
        total_sites = 0

        for incidents in self.incident_waves:
            best_key: Optional[Tuple[int, int, int]] = None
            best_outcome: Tuple[int, int, int, int] = (0, 0, 0, 0)
            for assignment in itertools.product(choices, repeat=self.num_agents):
                outcome = self._assignment_wave_outcome(incidents, assignment)
                key = (outcome[0], outcome[1], -outcome[2])
                if best_key is None or key > best_key:
                    best_key = key
                    best_outcome = outcome
            total_joint += best_outcome[0]
            total_resolved += best_outcome[1]
            total_duplicates += best_outcome[2]
            total_deployed += best_outcome[3]
            total_sites += len(incidents)

        return {
            "joint_reward": total_joint,
            "joint_earnings": total_joint,
            "sites_fully_resolved": total_resolved,
            "sites_total": total_sites,
            "sites_fully_resolved_rate": (
                round(total_resolved / total_sites, 3) if total_sites else 0.0
            ),
            "duplicate_deployments": total_duplicates,
            "deployments_total": total_deployed,
            "duplicate_deployment_rate": (
                round(total_duplicates / total_deployed, 3) if total_deployed else 0.0
            ),
        }

    def blind_solo_benchmark(self) -> Dict[str, Any]:
        """Each responder greedily picks the highest-reward site where it covers >=1 capability."""
        total_joint = 0
        total_resolved = 0
        total_duplicates = 0
        total_deployed = 0
        total_sites = 0

        for incidents in self.incident_waves:
            assignment_list: List[Optional[int]] = []
            for name in self.agent_names:
                own_caps = set(self.agent_capabilities[name])
                eligible = [
                    inc
                    for inc in incidents
                    if own_caps & set(inc["required_caps"])
                ]
                if not eligible:
                    assignment_list.append(None)
                else:
                    best_inc = max(
                        eligible,
                        key=lambda inc: (int(inc["reward"]), -int(inc["site_id"])),
                    )
                    assignment_list.append(int(best_inc["site_id"]))

            outcome = self._assignment_wave_outcome(incidents, tuple(assignment_list))
            total_joint += outcome[0]
            total_resolved += outcome[1]
            total_duplicates += outcome[2]
            total_deployed += outcome[3]
            total_sites += len(incidents)

        return {
            "joint_reward": total_joint,
            "joint_earnings": total_joint,
            "sites_fully_resolved": total_resolved,
            "sites_total": total_sites,
            "sites_fully_resolved_rate": (
                round(total_resolved / total_sites, 3) if total_sites else 0.0
            ),
            "duplicate_deployments": total_duplicates,
            "deployments_total": total_deployed,
            "duplicate_deployment_rate": (
                round(total_duplicates / total_deployed, 3) if total_deployed else 0.0
            ),
        }

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        partial_pct = int(round(self.partial_credit_ratio * 100))
        return (
            f"You are {self.num_agents} emergency response units operating across "
            f"{self.num_incidents} disaster incident sites (0..{self.num_incidents - 1}). "
            f"Each responder privately holds a subset of capabilities drawn from "
            f"{list(self.CAPABILITIES)}. Each wave, {self.num_incidents} incidents appear "
            f"with severity rewards and required capability sets. Deploying to a site "
            f"costs {self.deploy_cost} credits; standby costs 0. If all required "
            f"capabilities at a site are covered by the responders deployed there, the "
            f"full site reward is split in whole credits among them; partial coverage "
            f"yields {partial_pct}% of the site reward; zero coverage yields 0. "
            f"The blackboard is your shared communication channel."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        wave_index = min(max(iteration - 1, 0), self.max_iterations - 1)
        incidents = self.incident_waves[wave_index]
        last_record = self.wave_history[-1] if self.wave_history else None
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "wave": wave_index + 1,
            "max_waves": self.max_iterations,
            "num_incidents": self.num_incidents,
            "own_capabilities": list(self.agent_capabilities.get(agent_name, ())),
            "deploy_cost": self.deploy_cost,
            "partial_credit_ratio": self.partial_credit_ratio,
            "incidents": [
                {
                    "site_id": inc["site_id"],
                    "reward": inc["reward"],
                    "required_caps": list(inc["required_caps"]),
                }
                for inc in incidents
            ],
            "own_earnings_history": [
                int(r["earnings"].get(agent_name, 0)) for r in self.wave_history
            ],
            "last_wave": last_record,
            "legal_actions": (
                ["propose_deployment"]
                if phase == "planning"
                else ["commit_deployment", "commit_standby"]
            ),
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        return float(self.optimal_dispatch_benchmark()["joint_reward"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.wave_history:
            return 0.0
        return float(self.wave_history[-1]["joint_earnings"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.wave_history:
            return 0.0
        return float(self.wave_history[-1]["earnings"].get(agent_name, 0))

    def log_iteration(self, iteration: int) -> None:
        wave_index = iteration - 1
        record = self._settle_round(wave_index)
        self.wave_history.append(record)
        self.wave = iteration
        self.pending_actions = {name: None for name in self.agent_names}
        self.wave_proposals = {name: [] for name in self.agent_names}
        self.joint_reward_history.append(record["joint_earnings"])
        logger.info(
            "=== %s Wave %s === resolved=%s/%s duplicates=%s joint_earnings=%s",
            self.__class__.__name__,
            record["wave"],
            record["sites_fully_resolved"],
            record["sites_total"],
            record["duplicate_deployments"],
            record["joint_earnings"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        total_resolved = sum(r["sites_fully_resolved"] for r in self.wave_history)
        total_partial = sum(r["sites_partially_resolved"] for r in self.wave_history)
        total_sites = sum(r["sites_total"] for r in self.wave_history)
        total_duplicates = sum(r["duplicate_deployments"] for r in self.wave_history)
        total_wasted = sum(r["wasted_deployments"] for r in self.wave_history)
        total_deployed = sum(r["deployed_count"] for r in self.wave_history)
        joint_total = sum(r["joint_earnings"] for r in self.wave_history)

        all_declarations = [
            prop
            for props in self.proposals.values()
            for prop in props
            if prop.get("declared_capabilities")
        ]
        revelation_accuracy = (
            round(
                sum(float(p["accuracy"]) for p in all_declarations) / len(all_declarations),
                3,
            )
            if all_declarations
            else 1.0
        )

        optimal = self.optimal_dispatch_benchmark()
        blind_solo = self.blind_solo_benchmark()

        sites_fully_resolved_rate = (
            round(total_resolved / total_sites, 3) if total_sites else 0.0
        )
        duplicate_deployment_rate = (
            round(total_duplicates / total_deployed, 3) if total_deployed else 0.0
        )
        team_formation_efficiency = (
            round(joint_total / self.max_joint_reward, 3) if self.max_joint_reward else 0.0
        )

        return {
            "environment": self.__class__.__name__,
            "waves_simulated": len(self.wave_history),
            "responders": {
                name: {
                    "capabilities": list(self.agent_capabilities[name]),
                    "total_earnings": sum(
                        int(r["earnings"].get(name, 0)) for r in self.wave_history
                    ),
                    "waves_deployed": sum(
                        1 for r in self.wave_history if r["deployments"].get(name) is not None
                    ),
                    "waves_standby": sum(
                        1 for r in self.wave_history if r["deployments"].get(name) is None
                    ),
                    "proposals_made": len(self.proposals[name]),
                }
                for name in self.agent_names
            },
            "wave_history": self.wave_history,
            "sites_fully_resolved_rate": sites_fully_resolved_rate,
            "duplicate_deployment_rate": duplicate_deployment_rate,
            "capability_revelation_accuracy": revelation_accuracy,
            "team_formation_efficiency": team_formation_efficiency,
            "relief_stats": {
                "sites_total": total_sites,
                "sites_fully_resolved": total_resolved,
                "sites_partially_resolved": total_partial,
                "sites_fully_resolved_rate": sites_fully_resolved_rate,
                "deployments_total": total_deployed,
                "duplicate_deployments": total_duplicates,
                "wasted_deployments": total_wasted,
                "duplicate_deployment_rate": duplicate_deployment_rate,
                "declarations_total": len(all_declarations),
                "declarations_truthful": sum(
                    1 for p in all_declarations if p.get("truthful") is True
                ),
                "capability_revelation_accuracy": revelation_accuracy,
                "team_formation_efficiency": team_formation_efficiency,
            },
            "benchmarks": {
                "optimal_dispatch": optimal,
                "blind_solo": blind_solo,
            },
            "joint_reward": joint_total,
            "max_joint_reward": self.max_joint_reward,
        }
