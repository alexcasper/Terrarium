"""Prompts for the LegislativeBargaining environment.

Deliberately neutral: legislators are told the session rules, the bill docket
(gross benefits, per-legislator tax cost, and net payoffs), the majority voting
threshold, and their own payoff history — never how to form coalitions or trade
votes across bills.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class LegislativeBargainingPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        return (
            f"You are {agent_name}, one of {env.num_agents} legislators in a committee "
            f"({', '.join(env.agent_names)}).\n\n"
            f"SESSION RULES:\n"
            f"- The committee meets for {env.max_iterations} legislative sessions.\n"
            f"- In each session, {env.num_bills} bills ({', '.join(env.bill_ids)}) are on the "
            f"docket for simultaneous roll-call votes.\n"
            f"- A bill passes if and only if at least {env.majority_threshold} of the "
            f"{env.num_agents} legislators vote 'yes' on that bill.\n"
            f"- If a bill passes, its designated beneficiary legislator(s) receive its gross "
            f"benefit in credits, and EVERY legislator in the committee pays the bill's "
            f"uniform tax cost. Thus your standalone net payoff from a passed bill is "
            f"(your gross benefit from that bill) minus (the bill's tax cost).\n"
            f"- If a bill fails to reach {env.majority_threshold} 'yes' votes, it does not "
            f"pass and produces 0 benefit and 0 tax cost for everyone.\n"
            f"- Your session payoff is the sum of your net payoffs across all bills that pass "
            f"in that session.\n\n"
            f"Before casting binding roll-call votes in the execution phase, you may discuss "
            f"the docket with other legislators on the shared blackboard during the planning "
            f"phase and optionally publish a non-binding intended vote slate.\n\n"
            f"Your goal: maximise your own total credits across all {env.max_iterations} sessions."
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
        session = agent_context.get("session")
        max_sessions = agent_context.get("max_sessions")
        majority_threshold = agent_context.get("majority_threshold")
        lines = [f"SESSION {session} of {max_sessions} — {phase.upper()} PHASE."]
        lines.append(
            f"Majority threshold to pass a bill: {majority_threshold} 'yes' votes "
            f"out of {agent_context.get('num_agents')} legislators."
        )

        docket = agent_context.get("docket") or {}
        bill_lines = ["Current session bill docket:"]
        for bill_id, bill in docket.items():
            sponsors = ", ".join(bill.get("sponsors") or []) or "all legislators"
            own_net = bill.get("net_payoffs", {}).get(agent_name, 0)
            own_gross = bill.get("gross_benefits", {}).get(agent_name, 0)
            tax_cost = bill.get("tax_cost", 0)
            net_summary = ", ".join(
                f"{leg}: {net:+d}" for leg, net in bill.get("net_payoffs", {}).items()
            )
            bill_lines.append(
                f"- {bill_id} (primary beneficiary: {sponsors}): tax cost = {tax_cost} per "
                f"legislator; your gross benefit = {own_gross}, your standalone net payoff if "
                f"passed = {own_net:+d} credits | all legislators' net payoffs: [{net_summary}]"
            )
        lines.append("\n".join(bill_lines))

        last = agent_context.get("last_session")
        if last:
            passed_str = ", ".join(last.get("passed_bills") or []) or "none"
            lines.append(
                f"Previous session result: passed bills = [{passed_str}]; "
                f"your session payoff = {last['payoffs'].get(agent_name, 0):+d} credits."
            )
        else:
            lines.append("This is the first session; there is no prior session history.")

        payoff_history = agent_context.get("own_payoff_history") or []
        if payoff_history:
            lines.append(
                f"Your session payoffs so far: {payoff_history} "
                f"(cumulative: {sum(payoff_history):+d} credits)."
            )

        bill_ids = list(docket.keys())
        example_slate = ", ".join(f"'{b}': 'yes'|'no'" for b in bill_ids)
        board_visible = bool(getattr(self.environment, "board_visibility", True))
        if board_visible:
            if blackboard_context:
                bb_blocks = [
                    f"=== BLACKBOARD {bb_id} ===\n{str(content).strip()}"
                    for bb_id, content in blackboard_context.items()
                    if content and str(content).strip()
                ]
                if bb_blocks:
                    lines.append("\n\n".join(bb_blocks))
        else:
            lines.append(
                "Note: the shared blackboard feed is disabled for this session; "
                "you cannot read other legislators' messages or proposed slates."
            )

        if phase == "planning":
            if board_visible:
                lines.append(
                    "The blackboard above shows the committee conversation so far. You may post "
                    "messages with post_message and/or announce a non-binding intended vote slate "
                    f"with propose_vote_slate(votes={{{example_slate}}}, message=...). "
                    "End your turn after your tool calls."
                )
            else:
                lines.append(
                    "You may announce a non-binding intended vote slate with "
                    f"propose_vote_slate(votes={{{example_slate}}}, message=...). "
                    "End your turn after your tool calls."
                )
        else:
            lines.append(
                "Commit your binding roll-call votes for all bills in this session using "
                f"cast_votes(votes={{{example_slate}}}). This ends your turn."
            )
        return "\n\n".join(lines)
