"""Tools for the PublicGoods environment.

Planning phase: `propose_contribution` lets an agent float a public
contribution level on the blackboard (standard blackboard tools are also
available). Execution phase: `contribute` commits the agent's contribution
(and optional punishment allocations when the punishment arm is enabled).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_contribution"})
EXECUTION_TOOLS = frozenset({"contribute"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class PublicGoodsTools:
    def __init__(self, blackboard_manager, environment=None):
        self.blackboard_manager = blackboard_manager
        self.environment = environment

    def get_tool_names(self) -> Set[str]:
        return set(PLANNING_TOOLS) | set(EXECUTION_TOOLS)

    def get_tools(self, phase: str) -> List[Dict[str, Any]]:
        if phase == "planning":
            return self._planning_tools()
        if phase == "execution":
            return self._execution_tools()
        return []

    def _planning_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "propose_contribution",
                    "description": (
                        "Publicly announce a suggested contribution level for "
                        "this round on the shared blackboard. This is a "
                        "non-binding signal; your committed contribution is set "
                        "separately with contribute during execution. Use it to "
                        "coordinate (or not) with the other participants."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "integer",
                                "description": (
                                    "Proposed contribution in tokens "
                                    "(integer, 0-endowment)."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short rationale posted alongside the proposal.",
                            },
                        },
                        "required": ["amount"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        env = self.environment
        props: Dict[str, Any] = {
            "amount": {
                "type": "integer",
                "description": "Tokens to contribute to the group account this round (integer, 0-endowment).",
            },
        }
        required = ["amount"]
        desc = (
            "Commit your contribution to the group account for this round. The "
            "group total is multiplied and split equally among all "
            "participants. This ends your turn."
        )
        if env is not None and getattr(env, "punishment_enabled", False):
            props["punishments"] = {
                "type": "array",
                "description": (
                    f"Optional: up to {env.punish_points} punishment points "
                    f"spread across other participants. Each point costs you "
                    f"{int(env.punish_cost)} credit and deducts "
                    f"{int(env.punish_damage)} credits from the target."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "peer": {"type": "string", "description": "Participant to punish."},
                        "points": {"type": "integer", "description": "Points to assign (>= 0)."},
                    },
                    "required": ["peer", "points"],
                },
            }
            desc = (
                "Commit your contribution to the group account for this round, "
                "optionally assigning punishment points to other participants. "
                "This ends your turn."
            )
        return [
            {
                "type": "function",
                "function": {
                    "name": "contribute",
                    "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            },
        ]

    def _post_public_message(
        self,
        agent_name: str,
        message: str,
        phase: Optional[str],
        iteration: Optional[int],
    ) -> None:
        if self.blackboard_manager is None:
            return
        blackboard_ids = self.blackboard_manager.get_agent_blackboards(agent_name)
        for bb_id in blackboard_ids:
            if self.blackboard_manager.is_private_channel(bb_id):
                continue
            try:
                self.blackboard_manager.post(
                    int(bb_id),
                    agent_name,
                    "communication",
                    {"content": message},
                    phase=phase,
                    iteration=iteration,
                )
            except Exception:
                continue

    def handle_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        arguments: Dict[str, Any],
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.environment is None:
            return {"status": "failed", "reason": "Environment not initialized for tools"}

        if tool_name not in self.get_tool_names():
            return {"error": f"PublicGoods environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_contribution":
            if phase != "planning":
                return {"status": "failed", "reason": "propose_contribution is only available during planning"}
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                amount = int(args["amount"])
                message = str(args.get("message") or "").strip()
                text = f"{agent_name} proposes a contribution of {amount} tokens."
                if message:
                    text = f"{text} Rationale: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "contribute":
            if phase != "execution":
                return {"status": "failed", "reason": "contribute is only available during execution"}
            result = self.environment.commit_contribution(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                amount = int(args["amount"])
                n_pun = result.get("punishments_registered", 0)
                text = f"{agent_name} committed a contribution of {amount} tokens for round {iteration}."
                if n_pun:
                    text += f" ({n_pun} punishment allocation(s) registered)"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
