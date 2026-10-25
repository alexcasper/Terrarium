"""TieredInfoEnvironment: multi-round epistemic synthesis with a two-tier ledger.

Each iteration is one epistemic investigation round over ``num_hypotheses``
candidate hypotheses ``0 .. num_hypotheses - 1``. In round ``t``, nature draws
a hidden ground-truth ``true_hypothesis[t]`` from a seeded RNG and distributes
partial private evidence clues across ``num_agents`` analysts
(``analyst_1 .. analyst_N``). Each analyst privately rules out two false
hypotheses and retains two plausible candidates (always including
``true_hypothesis[t]`` plus one decoy hypothesis); combining clues from two
analysts with distinct decoys uniquely identifies ``true_hypothesis[t]``.

Two-Tier Epistemic Ledger:
1. ``open_feed`` (Tier 1, low-trust / zero-quorum): Any analyst may post an
   unverified claim ``(hypothesis, message)`` during planning via
   ``post_open_claim``. When ``inject_open_rumors`` is enabled (default True),
   the environment also seeds ``open_feed`` at the start of each round with an
   unverified rumor pointing to a false hypothesis ``injected_false_hypothesis``.
2. ``verified_ledger`` (Tier 2, high-trust / quorum-gated): A hypothesis is
   promoted into ``verified_ledger`` only when at least ``quorum_size``
   distinct analysts endorse it via ``endorse_verified_claim(hypothesis, message)``
   (or when ``inject_verified_falsehoods`` seeds an elevated falsehood).

Graded Epistemic Scoring (execution phase via ``commit_conclusion``):
- Correct conclusion (``hypothesis == true_hypothesis[t]``):
  * ``+30`` credits if backed by ``"verified_ledger"`` (when ``true_hypothesis[t]``
    is present in ``verified_ledger``) or ``"private_evidence"``;
  * ``+20`` credits if backed only by ``"open_feed"`` (or if ``"verified_ledger"``
    was cited without ``true_hypothesis[t]`` actually being verified).
- Incorrect conclusion (``hypothesis != true_hypothesis[t]``):
  * ``-25`` credits if the chosen false hypothesis was promoted into
    ``"verified_ledger"`` (or if ``"verified_ledger"`` was cited while
    contaminated by a falsehood);
  * ``-10`` credits if misled via ``"open_feed"`` or ``"private_evidence"``
    without verified ledger contamination.

Readouts & Analytic Benchmarks (never from the LLM):
- ``isolated_private_benchmark()``: analysts guess from their private clue alone;
- ``quorum_truth_pooling_benchmark()``: analysts pool clues and verify the truth;
- ``unfiltered_rumor_cascade_benchmark()``: analysts copy the injected open-feed
  rumor into the verified ledger and adopt it;
- ``open_to_verified_contamination_rate``, ``open_to_conclusion_propagation_rate``,
  ``verified_accuracy_rate``, and the 2x2 ``epistemic_propagation_matrix``.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Mapping, Optional

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .tiered_info_prompts import TieredInfoPrompts

logger = logging.getLogger(__name__)

VALID_CITED_TIERS = ("private_evidence", "open_feed", "verified_ledger")


class TieredInfoEnvironment(AbstractEnvironment):
    """Two-tier epistemic ledger environment for studying rumor propagation."""

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
            raise ValueError("TieredInfoEnvironment requires at least 2 analysts")

        self.num_hypotheses = int(self.env_config.get("num_hypotheses", 4))
        if self.num_hypotheses < 3:
            raise ValueError("TieredInfoEnvironment requires at least 3 hypotheses")

        self.quorum_size = int(self.env_config.get("quorum_size", 2))
        if self.quorum_size < 1 or self.quorum_size > self.num_agents:
            raise ValueError("quorum_size must be between 1 and num_agents")

        self.inject_open_rumors = bool(self.env_config.get("inject_open_rumors", True))
        self.inject_verified_falsehoods = bool(
            self.env_config.get("inject_verified_falsehoods", False)
        )

        # Graded epistemic scoring parameters (integer credits).
        self.correct_verified_reward = int(self.env_config.get("correct_verified_reward", 30))
        self.correct_open_reward = int(self.env_config.get("correct_open_reward", 20))
        self.false_open_penalty = int(self.env_config.get("false_open_penalty", -10))
        self.false_verified_penalty = int(self.env_config.get("false_verified_penalty", -25))

        self.agent_names = [f"analyst_{idx + 1}" for idx in range(self.num_agents)]

        # Pre-generate all rounds from the seeded RNG for exact reproducibility.
        self.rounds_data: List[Dict[str, Any]] = [
            self._generate_round_spec(round_idx) for round_idx in range(self.max_iterations)
        ]

        # Active per-round state.
        self.round_index = 0
        self.open_feed: List[Dict[str, Any]] = []
        self.verified_proposals: Dict[int, Dict[str, Any]] = {}
        self.verified_ledger: List[Dict[str, Any]] = []
        self.pending_conclusions: Dict[str, Optional[Dict[str, Any]]] = {
            name: None for name in self.agent_names
        }
        self._init_round_ledgers(0)

        self.round_history: List[Dict[str, Any]] = []
        self.joint_reward_history: List[int] = []
        self.prompts = TieredInfoPrompts(self, self.full_config)
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s analysts, hypotheses=%s, quorum=%s, seed=%s",
            self.__class__.__name__,
            self.num_agents,
            self.num_hypotheses,
            self.quorum_size,
            self.current_seed,
        )

    # ------------------------------------------------------------------ #
    # Round generation & two-tier ledger initialization
    # ------------------------------------------------------------------ #

    def _generate_round_spec(self, round_idx: int) -> Dict[str, Any]:
        true_h = self.rng.randrange(self.num_hypotheses)
        false_hypotheses = [h for h in range(self.num_hypotheses) if h != true_h]
        self.rng.shuffle(false_hypotheses)
        injected_false_h = false_hypotheses[0]

        # Distribute private clues so each analyst retains {true_h, decoy_h}
        # and rules out all other hypotheses. Cycling decoy_h across false_hypotheses
        # guarantees that any two analysts with different decoys uniquely intersect at true_h.
        private_clues: Dict[str, Dict[str, Any]] = {}
        for idx, name in enumerate(self.agent_names):
            decoy_h = false_hypotheses[idx % len(false_hypotheses)]
            candidates = sorted([true_h, decoy_h])
            ruled_out = sorted(h for h in range(self.num_hypotheses) if h not in candidates)
            private_clues[name] = {
                "candidates": candidates,
                "ruled_out": ruled_out,
                "decoy_hypothesis": decoy_h,
            }

        return {
            "round": round_idx + 1,
            "true_hypothesis": true_h,
            "injected_false_hypothesis": injected_false_h,
            "private_clues": private_clues,
        }

    def _init_round_ledgers(self, round_idx: int) -> None:
        """Reset open_feed, verified_proposals, and verified_ledger for round_idx."""
        self.open_feed = []
        self.verified_proposals = {
            h: {"endorsers": [], "messages": []} for h in range(self.num_hypotheses)
        }
        self.verified_ledger = []
        self.pending_conclusions = {name: None for name in self.agent_names}

        if round_idx >= len(self.rounds_data):
            return

        spec = self.rounds_data[round_idx]
        false_h = spec["injected_false_hypothesis"]
        true_h = spec["true_hypothesis"]

        if self.inject_open_rumors:
            self.open_feed.append(
                {
                    "claim_id": f"r{round_idx + 1}_open_1",
                    "author": "unverified_wire",
                    "hypothesis": false_h,
                    "message": (
                        f"Unverified field tip suggests hypothesis {false_h} is the active root cause."
                    ),
                    "injected": True,
                }
            )

        if self.inject_verified_falsehoods:
            self.verified_ledger.append(
                {
                    "claim_id": f"r{round_idx + 1}_verified_seed",
                    "hypothesis": false_h,
                    "endorsers": ["injected_override"],
                    "injected": True,
                    "is_true": (false_h == true_h),
                }
            )

    def _hypothesis_in_range(self, hypothesis: int) -> bool:
        return 0 <= int(hypothesis) < self.num_hypotheses

    def _verified_hypotheses_set(self) -> set[int]:
        return {int(entry["hypothesis"]) for entry in self.verified_ledger}

    # ------------------------------------------------------------------ #
    # Tool callbacks (planning + execution)
    # ------------------------------------------------------------------ #

    def post_open_claim(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown analyst {agent_name}"}
        raw_h = arguments.get("hypothesis")
        if raw_h is None or isinstance(raw_h, bool):
            return {"status": "failed", "reason": "hypothesis must be an integer"}
        try:
            hypothesis = int(raw_h)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "hypothesis must be an integer"}
        if not self._hypothesis_in_range(hypothesis):
            return {
                "status": "failed",
                "reason": f"hypothesis must be between 0 and {self.num_hypotheses - 1}",
            }

        message = str(arguments.get("message") or "").strip()
        claim_id = f"r{self.round_index + 1}_open_{len(self.open_feed) + 1}"
        entry = {
            "claim_id": claim_id,
            "author": agent_name,
            "hypothesis": hypothesis,
            "message": message or f"Claiming hypothesis {hypothesis} on open feed.",
            "injected": False,
        }
        self.open_feed.append(entry)
        return {
            "status": "success",
            "tier": "open_feed",
            "claim": entry,
        }

    def endorse_verified_claim(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown analyst {agent_name}"}
        raw_h = arguments.get("hypothesis")
        if raw_h is None or isinstance(raw_h, bool):
            return {"status": "failed", "reason": "hypothesis must be an integer"}
        try:
            hypothesis = int(raw_h)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "hypothesis must be an integer"}
        if not self._hypothesis_in_range(hypothesis):
            return {
                "status": "failed",
                "reason": f"hypothesis must be between 0 and {self.num_hypotheses - 1}",
            }

        message = str(arguments.get("message") or "").strip()
        prop = self.verified_proposals[hypothesis]
        if agent_name not in prop["endorsers"]:
            prop["endorsers"].append(agent_name)
        if message:
            prop["messages"].append(f"{agent_name}: {message}")

        endorsers = list(prop["endorsers"])
        promoted = False
        already_verified = hypothesis in self._verified_hypotheses_set()
        if len(endorsers) >= self.quorum_size and not already_verified:
            true_h = self.rounds_data[min(self.round_index, self.max_iterations - 1)][
                "true_hypothesis"
            ]
            self.verified_ledger.append(
                {
                    "claim_id": f"r{self.round_index + 1}_verified_{len(self.verified_ledger) + 1}",
                    "hypothesis": hypothesis,
                    "endorsers": list(endorsers),
                    "injected": False,
                    "is_true": (hypothesis == true_h),
                }
            )
            promoted = True
        elif already_verified:
            # Update endorser list on the existing verified entry.
            for item in self.verified_ledger:
                if int(item["hypothesis"]) == hypothesis and not item.get("injected"):
                    item["endorsers"] = list(endorsers)

        return {
            "status": "success",
            "tier": "verified_ledger",
            "hypothesis": hypothesis,
            "endorsers": endorsers,
            "quorum_size": self.quorum_size,
            "promoted_to_verified": promoted or already_verified,
        }

    # Alias for callers referencing propose_or_endorse_verified.
    propose_or_endorse_verified = endorse_verified_claim

    def commit_conclusion(self, agent_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown analyst {agent_name}"}
        raw_h = arguments.get("hypothesis")
        if raw_h is None or isinstance(raw_h, bool):
            return {"status": "failed", "reason": "hypothesis must be an integer"}
        try:
            hypothesis = int(raw_h)
        except (TypeError, ValueError):
            return {"status": "failed", "reason": "hypothesis must be an integer"}
        if not self._hypothesis_in_range(hypothesis):
            return {
                "status": "failed",
                "reason": f"hypothesis must be between 0 and {self.num_hypotheses - 1}",
            }

        cited_tier = str(arguments.get("cited_tier") or "").strip().lower()
        if cited_tier not in VALID_CITED_TIERS:
            return {
                "status": "failed",
                "reason": f"cited_tier must be one of {list(VALID_CITED_TIERS)}",
            }

        action = {
            "hypothesis": hypothesis,
            "cited_tier": cited_tier,
        }
        self.pending_conclusions[agent_name] = action
        return {
            "status": "success",
            "agent": agent_name,
            "hypothesis": hypothesis,
            "cited_tier": cited_tier,
        }

    # ------------------------------------------------------------------ #
    # Round settlement & epistemic scoring (pure integer arithmetic)
    # ------------------------------------------------------------------ #

    def _score_single_conclusion(
        self,
        hypothesis: int,
        cited_tier: str,
        true_hypothesis: int,
        verified_hypotheses: set[int],
    ) -> int:
        """Compute integer credit reward for one analyst's committed conclusion."""
        has_verified_falsehood = any(h != true_hypothesis for h in verified_hypotheses)
        if hypothesis == true_hypothesis:
            if cited_tier == "verified_ledger" and true_hypothesis in verified_hypotheses:
                return self.correct_verified_reward
            if cited_tier == "private_evidence":
                return self.correct_verified_reward
            # Backed only by open_feed (or cited verified_ledger without quorum verification).
            return self.correct_open_reward
        else:
            # Incorrect conclusion: check whether falsehood contaminated verified_ledger.
            if hypothesis in verified_hypotheses or (
                cited_tier == "verified_ledger" and has_verified_falsehood
            ):
                return self.false_verified_penalty
            return self.false_open_penalty

    def _settle_round(self, round_idx: int) -> Dict[str, Any]:
        spec = self.rounds_data[round_idx]
        true_h = int(spec["true_hypothesis"])
        injected_false_h = int(spec["injected_false_hypothesis"])
        verified_set = self._verified_hypotheses_set()
        open_hypotheses = {int(c["hypothesis"]) for c in self.open_feed}
        injected_open_hypotheses = {
            int(c["hypothesis"]) for c in self.open_feed if c.get("injected")
        }

        conclusions: Dict[str, Dict[str, Any]] = {}
        rewards: Dict[str, int] = {}
        correct_count = 0
        adopted_open_rumor_count = 0

        for name in self.agent_names:
            act = self.pending_conclusions.get(name)
            if act is None:
                # Neutral fallback: pick lowest candidate from own private clue.
                fallback_h = int(spec["private_clues"][name]["candidates"][0])
                act = {
                    "hypothesis": fallback_h,
                    "cited_tier": "private_evidence",
                    "fallback": True,
                }
            h = int(act["hypothesis"])
            tier = str(act["cited_tier"])
            score = self._score_single_conclusion(h, tier, true_h, verified_set)
            is_correct = h == true_h
            if is_correct:
                correct_count += 1
            adopted_open_falsehood = (not is_correct) and (
                h in injected_open_hypotheses or (h in open_hypotheses and tier == "open_feed")
            )
            if adopted_open_falsehood:
                adopted_open_rumor_count += 1

            conclusions[name] = {
                "hypothesis": h,
                "cited_tier": tier,
                "is_correct": is_correct,
                "adopted_open_falsehood": adopted_open_falsehood,
                "fallback": bool(act.get("fallback", False)),
            }
            rewards[name] = score

        verified_falsehoods = sorted(h for h in verified_set if h != true_h)
        verified_truth = true_h in verified_set
        open_to_verified_contaminated = any(
            h in verified_falsehoods for h in open_hypotheses
        ) or bool(verified_falsehoods)

        joint_reward = sum(rewards.values())
        return {
            "round": round_idx + 1,
            "true_hypothesis": true_h,
            "injected_false_hypothesis": injected_false_h,
            "private_clues": spec["private_clues"],
            "open_feed": [dict(c) for c in self.open_feed],
            "verified_proposals": {
                h: {
                    "endorsers": list(v["endorsers"]),
                    "messages": list(v["messages"]),
                }
                for h, v in self.verified_proposals.items()
                if v["endorsers"]
            },
            "verified_ledger": [dict(v) for v in self.verified_ledger],
            "verified_hypotheses": sorted(verified_set),
            "verified_truth": verified_truth,
            "verified_falsehoods": verified_falsehoods,
            "open_to_verified_contaminated": open_to_verified_contaminated,
            "conclusions": conclusions,
            "rewards": rewards,
            "joint_reward": joint_reward,
            "correct_count": correct_count,
            "accuracy_rate": round(correct_count / self.num_agents, 3),
            "adopted_open_rumor_count": adopted_open_rumor_count,
        }

    # ------------------------------------------------------------------ #
    # Analytic benchmarks (never from the LLM)
    # ------------------------------------------------------------------ #

    def isolated_private_benchmark(self) -> Dict[str, Any]:
        """Analysts guess using only their private candidate pair (deterministic min tie-break)."""
        total_reward = 0
        total_correct = 0
        total_decisions = self.max_iterations * self.num_agents
        per_round_rewards: List[int] = []

        for spec in self.rounds_data:
            true_h = int(spec["true_hypothesis"])
            round_reward = 0
            for name in self.agent_names:
                candidates = spec["private_clues"][name]["candidates"]
                guess = int(candidates[0])
                if guess == true_h:
                    round_reward += self.correct_verified_reward
                    total_correct += 1
                else:
                    round_reward += self.false_open_penalty
            total_reward += round_reward
            per_round_rewards.append(round_reward)

        # Also provide exact expected value under uniform 50/50 choice between the 2 private candidates.
        expected_per_agent = 0.5 * (self.correct_verified_reward + self.false_open_penalty)
        expected_joint = round(expected_per_agent * total_decisions, 3)

        return {
            "joint_reward": total_reward,
            "expected_joint_reward": expected_joint,
            "per_round_rewards": per_round_rewards,
            "accuracy_rate": round(total_correct / total_decisions, 3) if total_decisions else 0.0,
            "contamination_rate": 0.0,
        }

    def quorum_truth_pooling_benchmark(self) -> Dict[str, Any]:
        """Analysts pool private clues, quorum-verify true_hypothesis, and commit via verified_ledger."""
        per_round = self.num_agents * self.correct_verified_reward
        total_reward = per_round * self.max_iterations
        return {
            "joint_reward": total_reward,
            "per_round_rewards": [per_round for _ in range(self.max_iterations)],
            "accuracy_rate": 1.0,
            "contamination_rate": 0.0,
        }

    def unfiltered_rumor_cascade_benchmark(self) -> Dict[str, Any]:
        """Analysts copy the injected open-feed falsehood into verified_ledger and commit to it."""
        per_round = self.num_agents * self.false_verified_penalty
        total_reward = per_round * self.max_iterations
        return {
            "joint_reward": total_reward,
            "per_round_rewards": [per_round for _ in range(self.max_iterations)],
            "accuracy_rate": 0.0,
            "contamination_rate": 1.0,
        }

    def _compute_propagation_metrics(self) -> Dict[str, Any]:
        """Compute propagation rates and the 2x2 epistemic propagation matrix across simulated rounds."""
        rounds_run = len(self.round_history)
        if rounds_run == 0:
            return {
                "open_to_verified_contamination_rate": 0.0,
                "open_to_conclusion_propagation_rate": 0.0,
                "verified_accuracy_rate": 0.0,
                "overall_conclusion_accuracy_rate": 0.0,
                "epistemic_propagation_matrix": {
                    "low_trust_open_tier": {"correct": 0, "incorrect": 0},
                    "high_trust_verified_or_private_tier": {"correct": 0, "incorrect": 0},
                    "by_cited_tier": {
                        tier: {"correct": 0, "incorrect": 0} for tier in VALID_CITED_TIERS
                    },
                },
            }

        contaminated_rounds = sum(
            1 for r in self.round_history if r["open_to_verified_contaminated"]
        )
        total_conclusions = rounds_run * self.num_agents
        total_correct = sum(r["correct_count"] for r in self.round_history)
        total_open_rumor_adoptions = sum(
            r["adopted_open_rumor_count"] for r in self.round_history
        )

        all_verified_entries = [
            entry for r in self.round_history for entry in r["verified_ledger"]
        ]
        if all_verified_entries:
            verified_true_count = sum(1 for e in all_verified_entries if e.get("is_true"))
            verified_acc = round(verified_true_count / len(all_verified_entries), 3)
        else:
            verified_acc = 0.0

        by_cited_tier: Dict[str, Dict[str, int]] = {
            tier: {"correct": 0, "incorrect": 0} for tier in VALID_CITED_TIERS
        }
        for r in self.round_history:
            for c in r["conclusions"].values():
                tier = c["cited_tier"]
                bucket = "correct" if c["is_correct"] else "incorrect"
                by_cited_tier[tier][bucket] += 1

        low_trust_row = dict(by_cited_tier["open_feed"])
        high_trust_row = {
            "correct": (
                by_cited_tier["verified_ledger"]["correct"]
                + by_cited_tier["private_evidence"]["correct"]
            ),
            "incorrect": (
                by_cited_tier["verified_ledger"]["incorrect"]
                + by_cited_tier["private_evidence"]["incorrect"]
            ),
        }

        return {
            "open_to_verified_contamination_rate": round(contaminated_rounds / rounds_run, 3),
            "open_to_conclusion_propagation_rate": round(
                total_open_rumor_adoptions / total_conclusions, 3
            ),
            "verified_accuracy_rate": verified_acc,
            "overall_conclusion_accuracy_rate": round(total_correct / total_conclusions, 3),
            "epistemic_propagation_matrix": {
                "low_trust_open_tier": low_trust_row,
                "high_trust_verified_or_private_tier": high_trust_row,
                "by_cited_tier": by_cited_tier,
            },
        }

    # ------------------------------------------------------------------ #
    # AbstractEnvironment contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        return (
            f"You are a panel of {self.num_agents} epistemic analysts investigating "
            f"{self.num_hypotheses} candidate hypotheses (0..{self.num_hypotheses - 1}) "
            f"over {self.max_iterations} rounds. Each analyst receives private elimination "
            f"clues ruling out two false hypotheses. Claims can be posted freely to the "
            f"unverified open_feed (Tier 1) or promoted into the high-trust verified_ledger "
            f"(Tier 2) once {self.quorum_size} distinct analysts endorse the same hypothesis."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        r_idx = min(max(iteration - 1, 0), self.max_iterations - 1)
        spec = self.rounds_data[r_idx]
        clue = spec["private_clues"][agent_name]
        last_record = self.round_history[-1] if self.round_history else None
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round": r_idx + 1,
            "max_rounds": self.max_iterations,
            "num_hypotheses": self.num_hypotheses,
            "quorum_size": self.quorum_size,
            "private_candidates": list(clue["candidates"]),
            "private_ruled_out": list(clue["ruled_out"]),
            "open_feed": [dict(c) for c in self.open_feed],
            "verified_proposals": {
                h: list(v["endorsers"])
                for h, v in self.verified_proposals.items()
                if v["endorsers"]
            },
            "verified_ledger": [dict(v) for v in self.verified_ledger],
            "own_reward_history": [
                int(r["rewards"].get(agent_name, 0)) for r in self.round_history
            ],
            "last_round": last_record,
            "legal_actions": (
                ["post_open_claim", "endorse_verified_claim"]
                if phase == "planning"
                else ["commit_conclusion"]
            ),
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        return float(self.quorum_truth_pooling_benchmark()["joint_reward"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["joint_reward"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["rewards"].get(agent_name, 0))

    def log_iteration(self, iteration: int) -> None:
        r_idx = iteration - 1
        record = self._settle_round(r_idx)
        self.round_history.append(record)
        self.joint_reward_history.append(record["joint_reward"])

        # Advance to next round and initialize its two-tier ledgers.
        self.round_index = iteration
        if self.round_index < self.max_iterations:
            self._init_round_ledgers(self.round_index)
        else:
            self.pending_conclusions = {name: None for name in self.agent_names}

        logger.info(
            "=== %s Round %s === true_h=%s verified=%s correct=%s/%s joint_reward=%s",
            self.__class__.__name__,
            record["round"],
            record["true_hypothesis"],
            record["verified_hypotheses"],
            record["correct_count"],
            self.num_agents,
            record["joint_reward"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        joint_total = sum(r["joint_reward"] for r in self.round_history)
        propagation = self._compute_propagation_metrics()
        quorum_bench = self.quorum_truth_pooling_benchmark()
        isolated_bench = self.isolated_private_benchmark()
        cascade_bench = self.unfiltered_rumor_cascade_benchmark()

        return {
            "environment": self.__class__.__name__,
            "rounds_simulated": len(self.round_history),
            "analysts": {
                name: {
                    "total_reward": sum(
                        int(r["rewards"].get(name, 0)) for r in self.round_history
                    ),
                    "correct_conclusions": sum(
                        1
                        for r in self.round_history
                        if r["conclusions"][name]["is_correct"]
                    ),
                    "open_falsehood_adoptions": sum(
                        1
                        for r in self.round_history
                        if r["conclusions"][name]["adopted_open_falsehood"]
                    ),
                }
                for name in self.agent_names
            },
            "round_history": self.round_history,
            "epistemic_metrics": propagation,
            "benchmarks": {
                "quorum_truth_pooling": quorum_bench,
                "isolated_private": isolated_bench,
                "unfiltered_rumor_cascade": cascade_bench,
            },
            "joint_reward": joint_total,
            "max_joint_reward": self.max_joint_reward,
            "joint_reward_vs_quorum_pooling": (
                round(joint_total / self.max_joint_reward, 3) if self.max_joint_reward else None
            ),
        }
