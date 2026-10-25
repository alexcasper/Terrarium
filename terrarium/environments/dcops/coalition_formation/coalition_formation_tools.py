"""Tools for the CoalitionFormation environment.

Planning phase: ``propose_charter`` lets an agent announce a non-binding team
charter proposal (member roster, credit shares or split mode, and optional
message). Because ``CoalitionFormationTools.supports_private_channels = True``,
``ToolsetDiscovery`` also exposes ``create_channel`` and ``post_message`` during
planning so agents can open private side channels.

Execution phase: ``commit_charter`` records the agent's binding team roster and
split mode (``"equal"`` or ``"proportional"``) for the round. A multi-agent
team forms if and only if all listed members commit to the exact same roster and
split mode; otherwise non-matching members fall back to solo operation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_charter"})
EXECUTION_TOOLS = frozenset({"commit_charter"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class CoalitionFormationTools:
    """Toolset for CoalitionFormationEnvironment."""

    supports_private_channels: bool = True

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
                    "name": "propose_charter",
                    "description": (
                        "Announce a non-binding charter proposal for a team of specialists "
                        "(must include yourself). Specify the member list, optional split_mode "
                        "('equal' or 'proportional'), optional share breakdown summing to 100 "
                        "(percentages) or to the team value v(C), and an optional message."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "members": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "List of agent names in the proposed team (must include yourself)."
                                ),
                            },
                            "split_mode": {
                                "type": "string",
                                "enum": ["equal", "proportional"],
                                "description": (
                                    "Proposed rule for dividing team value: 'equal' or 'proportional' "
                                    "to standalone weights."
                                ),
                            },
                            "shares": {
                                "type": "object",
                                "additionalProperties": {"type": "integer"},
                                "description": (
                                    "Optional mapping from each member name in members to an integer "
                                    "share summing to 100 (percentage points) or to v(members)."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional note posted alongside the charter proposal.",
                            },
                        },
                        "required": ["members"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_charter",
                    "description": (
                        "Commit to a final team roster (must include yourself) and split_mode "
                        "('equal' or 'proportional') for this round. A multi-agent team forms "
                        "only if every member listed in members commits to the exact same "
                        "members set and split_mode; otherwise you work solo for your standalone "
                        "weight. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "members": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Exact roster of agent names for your team this round "
                                    "(must include yourself; pass [your_name] for solo)."
                                ),
                            },
                            "split_mode": {
                                "type": "string",
                                "enum": ["equal", "proportional"],
                                "description": (
                                    "Division rule for team output: 'equal' (default) or "
                                    "'proportional' to standalone weights."
                                ),
                            },
                        },
                        "required": ["members"],
                    },
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
            return {"error": f"CoalitionFormation environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_charter":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_charter is only available during planning",
                }
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                prop = result["proposal"]
                members_str = ", ".join(prop["members"])
                text = (
                    f"{agent_name} proposed charter [{members_str}] "
                    f"(v={prop['coalition_value']}, split_mode={prop['split_mode']}, "
                    f"shares={prop['shares']})."
                )
                note = prop.get("message")
                if note:
                    text = f"{text} Note: {note}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_charter":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "commit_charter is only available during execution",
                }
            result = self.environment.commit_charter(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                members_str = ", ".join(result["members"])
                self._post_public_message(
                    agent_name,
                    f"{agent_name} submitted charter commitment [{members_str}] "
                    f"(split_mode={result['split_mode']}).",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
