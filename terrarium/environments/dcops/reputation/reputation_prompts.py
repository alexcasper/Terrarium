"""Prompts for the Reputation environment.

Deliberately neutral: participants are told the rules, the payoffs, and what
the public record contains — never to build reputation, coordinate, or
cooperate. Any reputation behaviour in the transcripts is tacit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ReputationPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        return (
            f"You are {agent_name}, a participant in a repeated favour-exchange setting. "
            f"Each round you are matched with one other participant and each of you "
            f"commits one action toward the other.\n\n"
            f"ACTIONS AND PAYOFFS (per round, to you):\n"
            f"- honor: you pay {env.cost:g}; the other member of your pair gains "
            f"{env.benefit:g}. You receive the same from them if they honor you.\n"
            f"- betray: you pay nothing and give nothing; you still receive whatever "
            f"the other member of your pair does toward you.\n\n"
            f"THE PUBLIC RECORD:\n"
            f"- Every settled action is written to a public, append-only record that "
            f"all participants can read.\n"
            f"- The record carries over between episodes; it is not erased when a new "
            f"episode starts.\n"
            + (
                f"- You only see the most recent {env.memory_length} entries of the record.\n"
                if env.memory_length
                else "- You see the full record.\n"
            )
            + f"\nYou and the other participants may talk on the public blackboard before "
            f"each round. You are free to discuss whatever you like.\n\n"
            f"Your goal: maximise your own total payoff across all rounds."
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
        iteration = agent_context.get("iteration")
        max_iterations = agent_context.get("max_iterations")
        lines = [
            f"ROUND {iteration} of {max_iterations} — {phase.upper()} PHASE.",
            f"Your partner this round is {agent_context.get('partner')}.",
        ]

        own_history = agent_context.get("own_action_history") or []
        if own_history:
            lines.append(
                f"Your own past actions (oldest first): {list(own_history)}. "
                f"Your payoffs so far: "
                f"{[float(v) for v in agent_context.get('own_payoff_history') or []]}."
            )
        else:
            lines.append("This is your first recorded round; you have no history yet.")

        board = agent_context.get("reputation_board") or []
        if board:
            rendered = [
                f"ep{e['episode_index']} r{e['round']} {e['agent']}->{e['partner']}: {e['action']}"
                for e in board
            ]
            lines.append("Public record (most recent last): " + " | ".join(rendered))
        else:
            lines.append("The public record is empty.")

        last = agent_context.get("last_round")
        if last:
            lines.append(
                f"Previous round result: actions {last['actions']}; payoffs {last['payoffs']}."
            )

        if phase == "planning":
            lines.append(
                "The blackboard above shows the conversation so far. You may send messages "
                "with post_message and/or announce an intended action with "
                "pledge_action(action=...). An announcement is not binding; your actual "
                "action is set with commit_action. Settled actions are written to the "
                "public record that all participants read. End your turn after your tool calls."
            )
        else:
            lines.append(
                "Commit your action toward your partner with "
                "commit_action(action=honor|betray). Your action is written to the public "
                "record once the round settles. This ends your turn."
            )
        return "\n\n".join(lines)
