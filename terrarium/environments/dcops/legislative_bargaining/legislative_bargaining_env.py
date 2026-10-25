"""LegislativeBargaining: multi-issue roll-call voting and vote-trading game.

Each iteration is one legislative session with `num_bills` (default 3:
`bill_A`, `bill_B`, `bill_C`) on the docket before a committee of `num_agents`
(default 5: `legislator_1..legislator_5`) operating under strict majority rule
(`majority_threshold = num_agents // 2 + 1`, i.e. 3 of 5).

Classic Buchanan-Tullock / Riker setup:
- A special-interest bill confers a concentrated gross benefit (default +35
  credits) on its sponsor(s) while imposing a uniform tax cost (default 10
  credits) on EVERY legislator in the committee (including the sponsor).
- Thus the sponsor's standalone net payoff if their bill passes is +25 credits
  (+35 - 10), while every non-sponsor's standalone net payoff is -10 credits,
  and the bill's net social surplus across all 5 legislators is
  +25 + 4*(-10) = -15 < 0.
- Under issue-by-issue sincere voting (vote 'yes' iff own standalone net payoff
  > 0), each special-interest bill receives only 1 'yes' vote against 4 'no'
  votes and fails: 0 bills pass and joint social welfare is 0.
- If a minimal winning coalition of 3 sponsors trades votes and all 3 vote
  'yes' on all 3 of their bills (`bill_A`, `bill_B`, `bill_C`), all 3 bills
  pass (3-2 each): each coalition member nets +35 - 3*10 = +5 credits (+15
  coalition total), while the 2 non-coalition legislators each suffer -30
  credits (-60 minority externality burden), driving joint social welfare down
  to -45 credits per session.

The environment also supports seeded rotation of bill sponsors across sessions
and optional socially efficient bills (`social_surplus > 0`) so benchmarks can
separate welfare-reducing vote trading from socially beneficial bill passage.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .legislative_bargaining_prompts import LegislativeBargainingPrompts

logger = logging.getLogger(__name__)


class LegislativeBargainingEnvironment(AbstractEnvironment):
    """Multi-session legislative roll-call voting environment."""

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
        if self.num_agents < 3:
            raise ValueError("LegislativeBargaining needs at least 3 legislators")

        self.num_bills = int(self.env_config.get("num_bills", 3))
        if self.num_bills < 1 or self.num_bills > 26:
            raise ValueError("num_bills must be between 1 and 26")

        default_majority = self.num_agents // 2 + 1
        self.majority_threshold = int(
            self.env_config.get("majority_threshold", default_majority)
        )
        if not (1 <= self.majority_threshold <= self.num_agents):
            raise ValueError("majority_threshold must be between 1 and num_agents")

        self.sponsor_benefit = int(
            self.env_config.get("sponsor_benefit", self.env_config.get("benefit", 35))
        )
        self.tax_cost = int(self.env_config.get("tax_cost", 10))
        self.public_good_benefit = int(self.env_config.get("public_good_benefit", 16))
        raw_pg_sessions = self.env_config.get("public_good_sessions") or []
        self.public_good_sessions = {int(s) for s in raw_pg_sessions}
        # Board-visibility arm: when False, legislators cannot read the shared
        # blackboard feed (other legislators' messages / proposed slates).
        self.board_visibility = bool(self.env_config.get("board_visibility", True))

        self.agent_names = [f"legislator_{idx + 1}" for idx in range(self.num_agents)]
        self.bill_ids = [f"bill_{chr(ord('A') + k)}" for k in range(self.num_bills)]

        # Pre-generate all sessions' dockets from the seeded RNG.
        self.session_dockets: List[Dict[str, Dict[str, Any]]] = [
            self._generate_session_docket(session_idx)
            for session_idx in range(self.max_iterations)
        ]

        self.session = 0
        self.proposals: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in self.agent_names
        }
        self.pending_actions: Dict[str, Optional[Dict[str, str]]] = {
            name: None for name in self.agent_names
        }
        self.session_history: List[Dict[str, Any]] = []

        self.prompts = LegislativeBargainingPrompts(self, self.full_config)

        self.joint_reward_history: List[int] = []
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s legislators (majority=%s), %s bills/session, "
            "sponsor_benefit=%s, tax_cost=%s, seed=%s",
            self.__class__.__name__,
            self.num_agents,
            self.majority_threshold,
            self.num_bills,
            self.sponsor_benefit,
            self.tax_cost,
            self.current_seed,
        )

    # ------------------------------------------------------------------ #
    # Bill and docket construction
    # ------------------------------------------------------------------ #

    def make_special_interest_bill(
        self,
        bill_id: str,
        sponsors: List[str],
        benefit: Optional[int] = None,
        tax_cost: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Construct a bill conferring `benefit` on `sponsors` and `tax_cost` on all."""
        b_val = self.sponsor_benefit if benefit is None else int(benefit)
        t_val = self.tax_cost if tax_cost is None else int(tax_cost)
        sponsor_set = set(sponsors)
        gross_benefits = {
            name: (b_val if name in sponsor_set else 0) for name in self.agent_names
        }
        net_payoffs = {name: gross_benefits[name] - t_val for name in self.agent_names}
        social_surplus = sum(net_payoffs.values())
        return {
            "bill_id": bill_id,
            "bill_type": "special_interest" if social_surplus < 0 else "efficient_targeted",
            "sponsors": list(sponsors),
            "gross_benefits": gross_benefits,
            "tax_cost": t_val,
            "net_payoffs": net_payoffs,
            "social_surplus": social_surplus,
        }

    def make_public_good_bill(
        self,
        bill_id: str,
        benefit_per_agent: Optional[int] = None,
        tax_cost: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Construct a broad public-good bill benefiting every legislator."""
        b_val = (
            self.public_good_benefit
            if benefit_per_agent is None
            else int(benefit_per_agent)
        )
        t_val = self.tax_cost if tax_cost is None else int(tax_cost)
        gross_benefits = {name: b_val for name in self.agent_names}
        net_payoffs = {name: b_val - t_val for name in self.agent_names}
        social_surplus = sum(net_payoffs.values())
        return {
            "bill_id": bill_id,
            "bill_type": "public_good",
            "sponsors": list(self.agent_names),
            "gross_benefits": gross_benefits,
            "tax_cost": t_val,
            "net_payoffs": net_payoffs,
            "social_surplus": social_surplus,
        }

    def _generate_session_docket(self, session_idx: int) -> Dict[str, Dict[str, Any]]:
        """Generate the bill docket for `session_idx` (0-indexed)."""
        if session_idx == 0:
            # Session 1 is always the canonical assignment: legislator_1..legislator_K
            sponsors_order = [
                self.agent_names[k % self.num_agents] for k in range(self.num_bills)
            ]
        else:
            if self.num_bills <= self.num_agents:
                sponsors_order = self.rng.sample(self.agent_names, k=self.num_bills)
            else:
                sponsors_order = [
                    self.rng.choice(self.agent_names) for _ in range(self.num_bills)
                ]

        docket: Dict[str, Dict[str, Any]] = {}
        session_num = session_idx + 1
        for k, bill_id in enumerate(self.bill_ids):
            if session_num in self.public_good_sessions and k == self.num_bills - 1:
                docket[bill_id] = self.make_public_good_bill(bill_id)
            else:
                docket[bill_id] = self.make_special_interest_bill(
                    bill_id, sponsors=[sponsors_order[k]]
                )
        return docket

    def _current_docket(self, iteration: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        if iteration is not None:
            idx = min(max(int(iteration) - 1, 0), self.max_iterations - 1)
        else:
            idx = min(max(self.session, 0), self.max_iterations - 1)
        return self.session_dockets[idx]

    # ------------------------------------------------------------------ #
    # Tool callbacks (planning proposal & execution vote commit)
    # ------------------------------------------------------------------ #

    def _validate_votes_dict(
        self,
        raw_votes: Any,
        valid_bill_ids: List[str],
        require_all_bills: bool,
    ) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        if not isinstance(raw_votes, Mapping) or not raw_votes:
            return None, "votes must be a non-empty dictionary mapping bill_id to 'yes' or 'no'"

        valid_set = set(valid_bill_ids)
        unknown = [str(k) for k in raw_votes.keys() if str(k) not in valid_set]
        if unknown:
            return (
                None,
                f"unknown bill_id(s): {unknown}; valid bills are {valid_bill_ids}",
            )

        if require_all_bills:
            missing = [b for b in valid_bill_ids if b not in raw_votes]
            if missing:
                return (
                    None,
                    f"cast_votes must include every bill on the docket; missing: {missing}",
                )

        normalized: Dict[str, str] = {}
        for bill_id in valid_bill_ids:
            if bill_id not in raw_votes:
                continue
            val = raw_votes[bill_id]
            vote_str = str(val).strip().lower() if val is not None else ""
            if vote_str not in ("yes", "no"):
                return (
                    None,
                    f"vote for {bill_id} must be 'yes' or 'no' (got {val!r})",
                )
            normalized[bill_id] = vote_str

        return normalized, None

    def record_proposal(
        self,
        agent_name: str,
        arguments: Dict[str, Any],
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown legislator {agent_name}"}

        docket = self._current_docket(iteration)
        normalized, err = self._validate_votes_dict(
            arguments.get("votes"),
            list(docket.keys()),
            require_all_bills=False,
        )
        if err is not None:
            return {"status": "failed", "reason": err}

        entry: Dict[str, Any] = {"votes": normalized}
        message = str(arguments.get("message") or "").strip()
        if message:
            entry["message"] = message
        self.proposals[agent_name].append(entry)
        return {"status": "success", "proposal": entry}

    def commit_votes(
        self,
        agent_name: str,
        arguments: Dict[str, Any],
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown legislator {agent_name}"}

        docket = self._current_docket(iteration)
        normalized, err = self._validate_votes_dict(
            arguments.get("votes"),
            list(docket.keys()),
            require_all_bills=True,
        )
        if err is not None:
            return {"status": "failed", "reason": err}

        self.pending_actions[agent_name] = normalized
        return {
            "status": "success",
            "agent": agent_name,
            "votes": dict(normalized),
        }

    # ------------------------------------------------------------------ #
    # Roll-call settlement mechanics
    # ------------------------------------------------------------------ #

    def _sincere_votes_for_agent(
        self, docket: Dict[str, Dict[str, Any]], agent_name: str
    ) -> Dict[str, str]:
        """Vote 'yes' on bill k iff standalone net payoff for `agent_name` > 0."""
        return {
            bill_id: ("yes" if bill["net_payoffs"][agent_name] > 0 else "no")
            for bill_id, bill in docket.items()
        }

    def _resolve_session(
        self,
        session_index: int,
        docket: Dict[str, Dict[str, Any]],
        votes_by_agent: Dict[str, Dict[str, str]],
        fallbacks: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Tally votes on every bill, determine passage, and compute payoffs."""
        fallbacks = fallbacks or {name: False for name in self.agent_names}
        bill_outcomes: Dict[str, Dict[str, Any]] = {}
        passed_bills: List[str] = []
        pork_bills_passed = 0
        efficient_bills_passed = 0
        insincere_yes_votes = 0
        insincere_no_votes = 0
        insincere_yes_by_agent: Dict[str, int] = {name: 0 for name in self.agent_names}

        for bill_id, bill in docket.items():
            yes_voters = [
                name for name in self.agent_names if votes_by_agent[name].get(bill_id) == "yes"
            ]
            no_voters = [
                name for name in self.agent_names if votes_by_agent[name].get(bill_id) != "yes"
            ]
            passed = len(yes_voters) >= self.majority_threshold

            bill_insincere_yes = [
                name for name in yes_voters if bill["net_payoffs"][name] < 0
            ]
            bill_insincere_no = [
                name for name in no_voters if bill["net_payoffs"][name] > 0
            ]
            insincere_yes_votes += len(bill_insincere_yes)
            insincere_no_votes += len(bill_insincere_no)
            for name in bill_insincere_yes:
                insincere_yes_by_agent[name] += 1

            if passed:
                passed_bills.append(bill_id)
                if bill["social_surplus"] < 0:
                    pork_bills_passed += 1
                elif bill["social_surplus"] > 0:
                    efficient_bills_passed += 1

            bill_outcomes[bill_id] = {
                "bill_id": bill_id,
                "bill_type": bill["bill_type"],
                "sponsors": list(bill["sponsors"]),
                "tax_cost": bill["tax_cost"],
                "gross_benefits": dict(bill["gross_benefits"]),
                "net_payoffs": dict(bill["net_payoffs"]),
                "social_surplus": bill["social_surplus"],
                "yes_votes": len(yes_voters),
                "no_votes": len(no_voters),
                "yes_voters": yes_voters,
                "no_voters": no_voters,
                "insincere_yes_voters": bill_insincere_yes,
                "insincere_no_voters": bill_insincere_no,
                "passed": passed,
            }

        payoffs: Dict[str, int] = {name: 0 for name in self.agent_names}
        for bill_id in passed_bills:
            bill = docket[bill_id]
            for name in self.agent_names:
                payoffs[name] += int(bill["net_payoffs"][name])

        joint_welfare = sum(payoffs.values())
        minority_externality_burden = sum(max(0, -p) for p in payoffs.values())
        # Logrolling occurs when at least one socially inefficient (social_surplus < 0)
        # bill passes with the support of insincere 'yes' votes from non-beneficiaries.
        logrolling_occurred = bool(
            any(
                out["passed"]
                and out["social_surplus"] < 0
                and len(out["insincere_yes_voters"]) > 0
                for out in bill_outcomes.values()
            )
        )

        return {
            "session": session_index + 1,
            "votes": {name: dict(votes_by_agent[name]) for name in self.agent_names},
            "fallbacks": dict(fallbacks),
            "bills": bill_outcomes,
            "passed_bills": passed_bills,
            "pork_bills_passed": pork_bills_passed,
            "efficient_bills_passed": efficient_bills_passed,
            "insincere_yes_votes": insincere_yes_votes,
            "insincere_no_votes": insincere_no_votes,
            "insincere_yes_by_agent": insincere_yes_by_agent,
            "logrolling_occurred": logrolling_occurred,
            "payoffs": payoffs,
            "joint_welfare": joint_welfare,
            "minority_externality_burden": minority_externality_burden,
        }

    def _settle_round(self, session_index: int) -> Dict[str, Any]:
        """Settle one completed legislative session (fallback: sincere voting)."""
        docket = self.session_dockets[session_index]
        votes_by_agent: Dict[str, Dict[str, str]] = {}
        fallbacks: Dict[str, bool] = {}
        for name in self.agent_names:
            committed = self.pending_actions.get(name)
            if committed is None:
                votes_by_agent[name] = self._sincere_votes_for_agent(docket, name)
                fallbacks[name] = True
            else:
                votes_by_agent[name] = dict(committed)
                fallbacks[name] = False
        return self._resolve_session(session_index, docket, votes_by_agent, fallbacks)

    # ------------------------------------------------------------------ #
    # Analytic benchmarks (computed from session dockets, never from LLM)
    # ------------------------------------------------------------------ #

    def sincere_voting_benchmark(self) -> Dict[str, Any]:
        """Issue-by-issue sincere voting: every legislator votes 'yes' iff own net > 0."""
        total_joint = 0
        total_passed = 0
        total_pork = 0
        total_efficient = 0
        per_agent = {name: 0 for name in self.agent_names}

        for idx, docket in enumerate(self.session_dockets):
            votes = {
                name: self._sincere_votes_for_agent(docket, name)
                for name in self.agent_names
            }
            rec = self._resolve_session(idx, docket, votes)
            total_joint += rec["joint_welfare"]
            total_passed += len(rec["passed_bills"])
            total_pork += rec["pork_bills_passed"]
            total_efficient += rec["efficient_bills_passed"]
            for name, p in rec["payoffs"].items():
                per_agent[name] += p

        return {
            "joint_welfare": total_joint,
            "bills_passed": total_passed,
            "pork_bills_passed": total_pork,
            "efficient_bills_passed": total_efficient,
            "per_agent_payoffs": per_agent,
        }

    def social_optimum_benchmark(self) -> Dict[str, Any]:
        """Utilitarian optimum: pass a bill iff its net social surplus across all legislators > 0."""
        total_joint = 0
        total_passed = 0
        total_efficient = 0
        per_agent = {name: 0 for name in self.agent_names}

        for docket in self.session_dockets:
            for bill in docket.values():
                if bill["social_surplus"] > 0:
                    total_passed += 1
                    total_efficient += 1
                    total_joint += bill["social_surplus"]
                    for name in self.agent_names:
                        per_agent[name] += int(bill["net_payoffs"][name])

        return {
            "joint_welfare": total_joint,
            "bills_passed": total_passed,
            "pork_bills_passed": 0,
            "efficient_bills_passed": total_efficient,
            "per_agent_payoffs": per_agent,
        }

    def logrolling_cartel_benchmark(self) -> Dict[str, Any]:
        """Minimal winning coalition of sponsors trades 'yes' votes on their pork bills."""
        total_joint = 0
        total_coalition_welfare = 0
        total_minority_burden = 0
        total_pork = 0
        total_insincere_yes = 0
        per_agent = {name: 0 for name in self.agent_names}

        for idx, docket in enumerate(self.session_dockets):
            # Identify special-interest bills and their sponsors to form a
            # minimal winning coalition of size `majority_threshold`.
            pork_bills = [
                bill_id
                for bill_id, bill in docket.items()
                if bill["social_surplus"] < 0
            ]
            coalition: List[str] = []
            for bill_id in pork_bills:
                for sp in docket[bill_id]["sponsors"]:
                    if sp not in coalition and len(coalition) < self.majority_threshold:
                        coalition.append(sp)
            for name in self.agent_names:
                if len(coalition) >= self.majority_threshold:
                    break
                if name not in coalition:
                    coalition.append(name)
            coalition_set = set(coalition)

            # Coalition bills: any special-interest bill sponsored by a coalition member.
            coalition_bills = {
                bill_id
                for bill_id in pork_bills
                if any(sp in coalition_set for sp in docket[bill_id]["sponsors"])
            }

            votes: Dict[str, Dict[str, str]] = {}
            for name in self.agent_names:
                sincere = self._sincere_votes_for_agent(docket, name)
                if name in coalition_set:
                    for b_id in coalition_bills:
                        sincere[b_id] = "yes"
                votes[name] = sincere

            rec = self._resolve_session(idx, docket, votes)
            total_joint += rec["joint_welfare"]
            total_coalition_welfare += sum(rec["payoffs"][n] for n in coalition)
            total_minority_burden += rec["minority_externality_burden"]
            total_pork += rec["pork_bills_passed"]
            total_insincere_yes += rec["insincere_yes_votes"]
            for name, p in rec["payoffs"].items():
                per_agent[name] += p

        return {
            "joint_welfare": total_joint,
            "coalition_welfare": total_coalition_welfare,
            "minority_externality_burden": total_minority_burden,
            "pork_bills_passed": total_pork,
            "insincere_yes_votes": total_insincere_yes,
            "per_agent_payoffs": per_agent,
        }

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        return (
            f"You are a legislative committee of {self.num_agents} legislators "
            f"({', '.join(self.agent_names)}) voting across {self.max_iterations} sessions. "
            f"In each session, {self.num_bills} bills ({', '.join(self.bill_ids)}) are on "
            f"the docket. A bill passes if and only if at least {self.majority_threshold} "
            f"legislators vote 'yes' on it. Each passed bill grants its gross credit benefit "
            f"to its designated beneficiary legislator(s) and deducts its uniform tax cost "
            f"from every legislator in the committee. The shared blackboard is your "
            f"communication channel."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        session_index = min(max(iteration - 1, 0), self.max_iterations - 1)
        docket = self.session_dockets[session_index]
        last_record = self.session_history[-1] if self.session_history else None
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "session": session_index + 1,
            "max_sessions": self.max_iterations,
            "num_agents": self.num_agents,
            "majority_threshold": self.majority_threshold,
            "num_bills": self.num_bills,
            "docket": docket,
            "sincere_votes": self._sincere_votes_for_agent(docket, agent_name),
            "own_payoff_history": [
                int(r["payoffs"].get(agent_name, 0)) for r in self.session_history
            ],
            "last_session": last_record,
            "legal_actions": (
                ["propose_vote_slate"] if phase == "planning" else ["cast_votes"]
            ),
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        return float(self.social_optimum_benchmark()["joint_welfare"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.session_history:
            return 0.0
        return float(self.session_history[-1]["joint_welfare"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.session_history:
            return 0.0
        return float(self.session_history[-1]["payoffs"].get(agent_name, 0))

    def log_iteration(self, iteration: int) -> None:
        session_index = iteration - 1
        record = self._settle_round(session_index)
        self.session_history.append(record)
        self.session = iteration
        self.pending_actions = {name: None for name in self.agent_names}
        self.joint_reward_history.append(record["joint_welfare"])
        logger.info(
            "=== %s Session %s === passed=%s pork=%s insincere_yes=%s joint_welfare=%s",
            self.__class__.__name__,
            record["session"],
            record["passed_bills"],
            record["pork_bills_passed"],
            record["insincere_yes_votes"],
            record["joint_welfare"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        sessions_run = len(self.session_history)
        joint_total = sum(r["joint_welfare"] for r in self.session_history)
        pork_passed = sum(r["pork_bills_passed"] for r in self.session_history)
        efficient_passed = sum(r["efficient_bills_passed"] for r in self.session_history)
        total_passed = sum(len(r["passed_bills"]) for r in self.session_history)
        total_bills = sessions_run * self.num_bills
        insincere_yes_total = sum(r["insincere_yes_votes"] for r in self.session_history)
        logrolling_sessions = sum(
            1 for r in self.session_history if r["logrolling_occurred"]
        )
        minority_burden_total = sum(
            r["minority_externality_burden"] for r in self.session_history
        )

        # Count total negative-standalone-payoff voting opportunities across simulated sessions
        negative_payoff_opportunities = 0
        for idx in range(sessions_run):
            for bill in self.session_dockets[idx].values():
                negative_payoff_opportunities += sum(
                    1 for net in bill["net_payoffs"].values() if net < 0
                )

        sincere_bm = self.sincere_voting_benchmark()
        social_bm = self.social_optimum_benchmark()
        cartel_bm = self.logrolling_cartel_benchmark()

        legislators_summary: Dict[str, Dict[str, Any]] = {}
        for name in self.agent_names:
            sponsored_count = 0
            sponsored_passed = 0
            yes_cast = 0
            insincere_yes = 0
            for idx, r in enumerate(self.session_history):
                docket = self.session_dockets[idx]
                for b_id, bill in docket.items():
                    if name in bill["sponsors"] and bill["bill_type"] != "public_good":
                        sponsored_count += 1
                        if r["bills"][b_id]["passed"]:
                            sponsored_passed += 1
                    if r["votes"][name].get(b_id) == "yes":
                        yes_cast += 1
                insincere_yes += r["insincere_yes_by_agent"].get(name, 0)

            legislators_summary[name] = {
                "total_payoff": sum(
                    int(r["payoffs"].get(name, 0)) for r in self.session_history
                ),
                "bills_sponsored": sponsored_count,
                "sponsored_bills_passed": sponsored_passed,
                "yes_votes_cast": yes_cast,
                "insincere_yes_votes": insincere_yes,
            }

        return {
            "environment": self.__class__.__name__,
            "sessions_simulated": sessions_run,
            "majority_threshold": self.majority_threshold,
            "legislators": legislators_summary,
            "session_history": self.session_history,
            "legislative_stats": {
                "total_bills_considered": total_bills,
                "total_bills_passed": total_passed,
                "pork_bills_passed": pork_passed,
                "efficient_bills_passed": efficient_passed,
                "logrolling_sessions": logrolling_sessions,
                "logrolling_incidence_rate": (
                    round(logrolling_sessions / sessions_run, 3)
                    if sessions_run
                    else None
                ),
                "insincere_yes_votes": insincere_yes_total,
                "insincere_yes_vote_rate": (
                    round(insincere_yes_total / negative_payoff_opportunities, 3)
                    if negative_payoff_opportunities
                    else None
                ),
                "minority_externality_burden": minority_burden_total,
                "welfare_loss_vs_social_optimum": (
                    social_bm["joint_welfare"] - joint_total
                ),
            },
            "benchmarks": {
                "sincere_voting": sincere_bm,
                "social_optimum": social_bm,
                "logrolling_cartel": cartel_bm,
            },
            "joint_reward": joint_total,
            "max_joint_reward": self.max_joint_reward,
        }
