"""Prompts for the MultiIssueBarterEnvironment.

Deliberately neutral: traders are given the bargaining rules, their private
per-unit valuations, the status-quo endowments, and their own round history —
never instructions on how to package issues or split surplus.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class MultiIssueBarterPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        vals = env.valuations.get(agent_name) or []
        endow = env.endowments.get(agent_name) or []
        val_str = ", ".join(
            f"issue_{k + 1}: {v} credits/unit" for k, v in enumerate(vals)
        )
        endow_str = ", ".join(
            f"issue_{k + 1}: {u}/{env.total_units} units" for k, u in enumerate(endow)
        )
        endow_payoff = sum(u * v for u, v in zip(endow, vals))

        if env.agenda_mode == "package":
            agenda_text = (
                f"- Agenda mode is 'package': in each of the {env.max_iterations} rounds, all "
                f"{env.num_issues} issues (issue_1..issue_{env.num_issues}) are open for "
                f"simultaneous negotiation and settlement."
            )
        else:
            order_names = [f"issue_{k + 1}" for k in env.agenda_order]
            agenda_text = (
                f"- Agenda mode is 'sequential': each round settles one issue at a time "
                f"following the agenda order {order_names}."
            )

        return (
            f"You are {agent_name}, one of {env.num_agents} traders negotiating over "
            f"{env.num_issues} divisible resource issues (issue_1 to issue_{env.num_issues}). "
            f"Each issue has {env.total_units} integer units available.\n\n"
            f"PRIVATE INFORMATION (known only to you):\n"
            f"- Your per-unit valuations: {val_str}.\n"
            f"- Your status-quo endowment: {endow_str} "
            f"(status-quo payoff = {endow_payoff} credits per full 5-issue set).\n\n"
            f"BARGAINING RULES:\n"
            f"{agenda_text}\n"
            f"- During planning, you may communicate on the shared blackboard using "
            f"post_message and float non-binding proposals using "
            f"propose_package(allocations=[...], message=...).\n"
            f"- During execution, you submit your binding unit claims using "
            f"commit_deal(allocations=[...]), where each integer (0 to {env.total_units}) is "
            f"the number of units YOU claim for yourself on that issue.\n"
            f"- Settlement per issue: if the sum of units claimed by all traders on an issue "
            f"is at most {env.total_units}, the deal succeeds on that issue and every trader "
            f"receives their claimed units (plus an equal integer split of any unclaimed "
            f"units). If total claims exceed {env.total_units}, there is no agreement on that "
            f"issue and it reverts to its status-quo endowment.\n\n"
            f"Your goal: maximise your own total credits earned across all rounds."
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
        max_rounds = agent_context.get("max_rounds")
        mode = agent_context.get("agenda_mode", "package")
        active_issues = agent_context.get("active_issues") or []
        total_units = agent_context.get("total_units", 10)
        num_issues = agent_context.get("num_issues", 5)

        lines = [
            f"ROUND {rnd} of {max_rounds} ({mode.upper()} AGENDA) — {phase.upper()} PHASE."
        ]

        lines.append(
            f"Active issue(s) this round: {', '.join(active_issues)} "
            f"({total_units} units per issue)."
        )

        vals = agent_context.get("own_valuations") or []
        endow = agent_context.get("own_endowment") or []
        lines.append(
            "Your private valuations (credits/unit) and status-quo endowment (units): "
            + ", ".join(
                f"issue_{k + 1}: val={vals[k]}, endow={endow[k]}/{total_units}"
                for k in range(len(vals))
            )
            + "."
        )

        last = agent_context.get("last_round")
        if last:
            lines.append(
                f"Previous round result: agreed issues={last['agreed_issues']}; "
                f"disagreed issues={last['disagreed_issues']}; "
                f"your payoff={last['payoffs'].get(agent_name, 0)} credits "
                f"(status-quo baseline={last['endowment_payoffs'].get(agent_name, 0)} credits)."
            )
        else:
            lines.append("This is the first round; there is no round history yet.")

        payoff_history = agent_context.get("own_payoff_history") or []
        if payoff_history:
            lines.append(f"Your payoffs by round so far: {payoff_history} credits.")

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
                f"Review the blackboard messages above. You may discuss terms with "
                f"post_message and/or post a non-binding proposal with "
                f"propose_package(allocations=[u_1, ..., u_{num_issues}], message=...) "
                f"specifying the units (0..{total_units}) you would receive on each issue. "
                f"End your turn after your tool calls."
            )
        else:
            lines.append(
                f"Commit your binding unit claims for this round by calling "
                f"commit_deal(allocations=[u_1, ..., u_{num_issues}]) where each u_k "
                f"(0..{total_units}) is the number of units you claim for yourself on "
                f"issue_k. Calling commit_deal ends your turn."
            )
        return "\n\n".join(lines)
