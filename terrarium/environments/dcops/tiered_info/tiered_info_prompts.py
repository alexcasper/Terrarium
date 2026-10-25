"""Prompts for the TieredInfoEnvironment.

Deliberately neutral: analysts are given the round rules, their own private
elimination clues, the live contents of the two-tier ledger (``open_feed`` and
``verified_ledger``), and the scoring table.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class TieredInfoPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        return (
            f"You are {agent_name}, one of {env.num_agents} epistemic analysts evaluating "
            f"{env.num_hypotheses} candidate hypotheses (numbered 0 to {env.num_hypotheses - 1}) "
            f"across {env.max_iterations} rounds.\n\n"
            f"ROUND RULES:\n"
            f"- Each round has one true hypothesis in 0..{env.num_hypotheses - 1}.\n"
            f"- You receive private elimination evidence that rules out two false hypotheses "
            f"and leaves two candidate hypotheses (one of which is the true hypothesis).\n"
            f"- Combining private elimination clues from two analysts with different remaining "
            f"candidates uniquely identifies the true hypothesis.\n\n"
            f"TWO-TIER EPISTEMIC LEDGER:\n"
            f"- Tier 1 (open_feed): Any claim can be posted with zero quorum via "
            f"post_open_claim(hypothesis, message). The open_feed may also contain unverified "
            f"seeded wire tips.\n"
            f"- Tier 2 (verified_ledger): A hypothesis is promoted into verified_ledger only "
            f"when at least {env.quorum_size} distinct analysts endorse it during planning via "
            f"endorse_verified_claim(hypothesis, message).\n\n"
            f"GRADED EPISTEMIC SCORING (per round):\n"
            f"- Correct conclusion backed by 'verified_ledger' (when verified by quorum) or "
            f"'private_evidence': +{env.correct_verified_reward} credits.\n"
            f"- Correct conclusion backed only by 'open_feed': +{env.correct_open_reward} credits.\n"
            f"- Incorrect conclusion via 'open_feed' or 'private_evidence' (unverified): "
            f"{env.false_open_penalty} credits.\n"
            f"- Incorrect conclusion when the false hypothesis was promoted into "
            f"'verified_ledger': {env.false_verified_penalty} credits.\n\n"
            f"Your goal: maximise your own total epistemic score across all rounds."
        )

    def get_user_prompt(
        self,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, Any],
    ) -> str:
        system_prompt = self.get_system_prompt(agent_name)
        user_prompt = self._get_user_prompt_impl(agent_name, agent_context, blackboard_context)

        if self.prompt_logger:
            self.prompt_logger.log_prompts(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                phase=agent_context.get("phase", "unknown"),
                iteration=agent_context.get("iteration"),
                round_num=agent_context.get("planning_round"),
            )
        return user_prompt

    def _get_user_prompt_impl(
        self,
        agent_name: str,
        agent_context: Dict[str, Any],
        blackboard_context: Dict[str, Any],
    ) -> str:
        phase = str(agent_context.get("phase", "unknown"))
        rnd = agent_context.get("round")
        lines = [f"ROUND {rnd} of {agent_context.get('max_rounds')} — {phase.upper()} PHASE."]

        ruled_out = agent_context.get("private_ruled_out") or []
        candidates = agent_context.get("private_candidates") or []
        lines.append(
            f"PRIVATE EVIDENCE CLUE: Your private diagnostic rules out hypotheses {ruled_out}. "
            f"Your remaining plausible candidates are {candidates}."
        )

        open_feed = agent_context.get("open_feed") or []
        if open_feed:
            open_summary = "; ".join(
                f"{c['author']} -> H{c['hypothesis']} ({c['message']})" for c in open_feed
            )
            lines.append(f"TIER 1 (open_feed, unverified): {open_summary}")
        else:
            lines.append("TIER 1 (open_feed, unverified): empty.")

        proposals = agent_context.get("verified_proposals") or {}
        if proposals:
            prop_summary = "; ".join(
                f"H{h}: endorsed by {endorsers} ({len(endorsers)}/{agent_context.get('quorum_size')})"
                for h, endorsers in sorted(proposals.items(), key=lambda kv: int(kv[0]))
            )
            lines.append(f"TIER 2 PENDING ENDORSEMENTS: {prop_summary}")

        verified = agent_context.get("verified_ledger") or []
        if verified:
            ver_summary = "; ".join(
                f"H{v['hypothesis']} (endorsers={v['endorsers']})" for v in verified
            )
            lines.append(f"TIER 2 (verified_ledger, quorum-verified): {ver_summary}")
        else:
            lines.append(
                f"TIER 2 (verified_ledger, quorum={agent_context.get('quorum_size')}): "
                f"no hypothesis verified yet this round."
            )

        last = agent_context.get("last_round")
        if last:
            lines.append(
                f"Previous round result: true hypothesis was H{last['true_hypothesis']}; "
                f"verified_ledger={last['verified_hypotheses']}; "
                f"your reward was {last['rewards'].get(agent_name, 0)} credits."
            )
        else:
            lines.append("This is the first round; there is no prior round history.")

        reward_history = agent_context.get("own_reward_history") or []
        if reward_history:
            lines.append(f"Your rewards so far by round: {reward_history} credits.")

        if blackboard_context:
            bb_blocks = [
                f"=== BLACKBOARD {bb_id} ===\n{str(content).strip()}"
                for bb_id, content in blackboard_context.items()
                if content and str(content).strip()
            ]
            if bb_blocks:
                lines.append("\n\n".join(bb_blocks))

        if phase == "planning":
            lines.append(
                "During planning you may share clues on the blackboard with post_message, "
                "post an unverified claim to Tier 1 with post_open_claim(hypothesis=..., message=...), "
                "or co-sign a hypothesis for Tier 2 quorum promotion with "
                "endorse_verified_claim(hypothesis=..., message=...). End your turn after your tool calls."
            )
        else:
            lines.append(
                "Commit your final conclusion for this round using "
                "commit_conclusion(hypothesis=..., cited_tier='private_evidence'|'open_feed'|'verified_ledger'). "
                "This commit ends your turn."
            )
        return "\n\n".join(lines)
