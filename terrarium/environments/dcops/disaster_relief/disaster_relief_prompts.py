"""Prompts for the DisasterRelief environment.

Deliberately neutral: responders are told the rules, their private capability
kit, the current wave's incident sites and required capabilities, and their own
earnings history. Any ad-hoc team formation or capability disclosure that
emerges on the blackboard is endogenous.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class DisasterReliefPrompts:
    def __init__(self, environment, config):
        self.environment = environment
        self.full_config = config
        self.prompt_logger = getattr(environment, "prompt_logger", None)

    def get_system_prompt(self, agent_name: Optional[str] = None) -> str:
        env = self.environment
        caps = sorted(env.agent_capabilities.get(agent_name, ()))
        caps_str = ", ".join(caps) if caps else "none"
        partial_pct = int(round(env.partial_credit_ratio * 100))
        system_lines = [
            f"You are {agent_name}, one of {env.num_agents} emergency response units "
            f"operating across {env.num_incidents} incident sites (numbered 0 to "
            f"{env.num_incidents - 1}).\n\n"
            f"PRIVATE INFORMATION: your unit possesses capabilities [{caps_str}] and "
            f"your deployment cost is {env.deploy_cost} credits per wave when deployed "
            f"to a site (0 credits on standby). Other responders do not see your "
            f"capabilities unless you share them.\n\n"
            f"WAVE RULES:\n"
            f"- Each wave, {env.num_incidents} disaster incidents appear at sites "
            f"0..{env.num_incidents - 1}. Each site has a severity reward between "
            f"{env.reward_low} and {env.reward_high} credits and a set of required "
            f"capabilities drawn from {list(env.CAPABILITIES)}.\n"
            f"- Each wave you take exactly one action: deploy to a single site "
            f"(0..{env.num_incidents - 1}) at cost {env.deploy_cost} credits, or stay "
            f"on standby at 0 cost.\n"
            f"- For each site, the combined capabilities of all units deployed there "
            f"are compared against the site's required capabilities:\n"
            f"  * Full resolution: if all required capabilities of the site are covered "
            f"by the deployed units, the full site reward is split in whole credits "
            f"among the deployed units (minus each unit's {env.deploy_cost}-credit "
            f"deployment cost).\n"
            f"  * Partial resolution: if at least one required capability is covered "
            f"but not all, {partial_pct}% of the site reward is split in whole credits "
            f"among the deployed units (minus each unit's {env.deploy_cost}-credit "
            f"deployment cost).\n"
            f"  * Zero coverage: if none of the site's required capabilities are "
            f"covered by the deployed units, deployed units earn 0 reward and still "
            f"pay the {env.deploy_cost}-credit deployment cost.\n"
            f"- The response operation runs for {env.max_iterations} waves.\n\n",
        ]
        planning_rounds = int(
            (self.full_config.get("simulation") or {}).get("max_planning_rounds", 1) or 0
        )
        if planning_rounds > 0:
            system_lines.append(
                "Before each wave's action you may talk with the other responders on the "
                "shared blackboard. You are free to discuss whatever you like."
            )
        else:
            system_lines.append(
                "This operation has no communication phase; each unit decides on its "
                "own private information."
            )
        system_lines.append(
            "Your goal: maximise your own total net credits across all waves."
        )
        return "\n".join(system_lines)

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
        wave = agent_context.get("wave")
        lines = [f"WAVE {wave} of {agent_context.get('max_waves')} — {phase.upper()} PHASE."]

        own_caps = agent_context.get("own_capabilities") or []
        lines.append(
            f"Your private capabilities: {list(own_caps)} "
            f"(deployment cost: {agent_context.get('deploy_cost')} credits; standby cost: 0)."
        )

        incidents = agent_context.get("incidents") or []
        incident_strs = [
            f"site {inc['site_id']}: reward={inc['reward']} credits, "
            f"required_caps={list(inc['required_caps'])}"
            for inc in incidents
        ]
        lines.append("Active incident sites this wave: " + "; ".join(incident_strs) + ".")

        last = agent_context.get("last_wave")
        if last:
            site_statuses = (
                {s: rec["status"] for s, rec in last["sites"].items()} if last.get("sites") else {}
            )
            lines.append(
                f"Previous wave result: site statuses {site_statuses}; "
                f"sites fully resolved {last['sites_fully_resolved']}/{last['sites_total']}; "
                f"your net earnings {last['earnings'].get(agent_name, 0)} credits."
            )
        else:
            lines.append("This is the first wave; there is no history yet.")

        earnings_history = agent_context.get("own_earnings_history") or []
        if earnings_history:
            lines.append(f"Your net earnings so far by wave: {earnings_history} credits.")

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
                "The blackboard above shows the conversation so far. You may send messages "
                "with post_message and/or publicly announce an intended deployment with "
                "propose_deployment(site_id=..., declared_capabilities=[...], message=...). "
                "Talk is allowed but never required. End your turn after your tool calls."
            )
        else:
            lines.append(
                "Commit exactly one action for this wave: commit_deployment(site_id=...) to "
                "deploy to an incident site, or commit_standby() to stay at base at 0 cost. "
                "Either commit ends your turn."
            )
        return "\n\n".join(lines)
