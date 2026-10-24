"""Prompts for the NetworkRouting environment.

Deliberately neutral: routers are told the rules, the topology, and their own
history — never to spread load, coordinate, or cooperate. Any load-balancing
in the transcripts is tacit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class NetworkRoutingPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def _link_table(self) -> str:
        env = self.environment
        rows = [
            f"- {link['name']}: base latency {link['base_latency']:g}, "
            f"congestion rate {link['congestion_rate']:g}"
            for link in env.links
        ]
        return "\n".join(rows)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        return (
            f"You are {agent_name}, a traffic operator routing {env.units_per_agent} units of "
            f"traffic per iteration from a source node to a sink node. "
            f"{env.num_agents - 1} other operators route the same amount through the same "
            f"network.\n\n"
            f"NETWORK TOPOLOGY — {len(env.links)} parallel links from source to sink:\n"
            f"{self._link_table()}\n\n"
            f"RULES:\n"
            f"- Each iteration you must place ALL your units on exactly one link.\n"
            f"- A link's per-unit latency is: base latency + congestion rate x (total units "
            f"placed on that link by ALL operators this iteration).\n"
            f"- Your latency for the iteration is the per-unit latency of the link you chose.\n"
            f"- The network repeats for {env.max_iterations} iterations.\n\n"
            f"Before each iteration you may talk with the other operators on the public "
            f"blackboard. You are free to discuss whatever you like.\n\n"
            f"Your goal: minimise your own total latency across all iterations."
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
            f"ITERATION {iteration} of {max_iterations} — {phase.upper()} PHASE.",
            f"You route {agent_context.get('units_per_agent')} units this iteration.",
        ]

        own_history = agent_context.get("own_latency_history") or []
        if own_history:
            lines.append(
                f"Your latency history so far (per iteration): "
                f"{[float(v) for v in own_history]}."
            )
        else:
            lines.append("This is the first iteration; you have no history yet.")

        last = agent_context.get("last_iteration")
        if last:
            lines.append(
                f"Previous iteration result: link loads {last['loads']}; "
                f"per-unit latencies {last['latencies']}."
            )

        if phase == "planning":
            lines.append(
                "The blackboard above shows the conversation so far. You may send messages "
                "with post_message and/or announce a link you are considering with "
                "propose_route(link=...). Discussion is allowed but never required. "
                "End your turn after your tool calls."
            )
        else:
            lines.append(
                "Commit your link for this iteration with commit_route(link=...). Remember: "
                "the per-unit latency of a link depends on the total units all operators "
                "place on it. This ends your turn."
            )
        return "\n\n".join(lines)
