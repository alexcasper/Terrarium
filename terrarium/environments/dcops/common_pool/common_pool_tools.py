"""Tools for the CommonPool environment.

Planning phase: `propose_quota` lets an agent float a public extraction level
on the blackboard (standard get_messages/post_message blackboard tools are also
available). Execution phase: `extract` commits the agent's extraction for the
current season.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_quota"})
EXECUTION_TOOLS = frozenset({"extract"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class CommonPoolTools:
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
                    "name": "propose_quota",
                    "description": (
                        "Publicly announce a suggested per-harvester extraction "
                        "level for this season on the shared blackboard. This is a "
                        "non-binding signal; your committed extraction is set "
                        "separately with extract during execution. Use it to "
                        "coordinate (or not) with the other harvesters."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "integer",
                                "description": (
                                    "Proposed extraction per harvester in units "
                                    "(integer, 0-max_extract)."
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
        return [
            {
                "type": "function",
                "function": {
                    "name": "extract",
                    "description": (
                        "Commit your extraction from the shared pool for this "
                        "season. If total requests exceed the pool, units are "
                        "rationed proportionally. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "integer",
                                "description": (
                                    "Units to extract this season "
                                    "(integer, 0-max_extract)."
                                ),
                            },
                        },
                        "required": ["amount"],
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
            return {"error": f"CommonPool environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_quota":
            if phase != "planning":
                return {"status": "failed", "reason": "propose_quota is only available during planning"}
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                amount = int(args["amount"])
                message = str(args.get("message") or "").strip()
                text = f"{agent_name} proposes an extraction level of {amount} units per harvester."
                if message:
                    text = f"{text} Rationale: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "extract":
            if phase != "execution":
                return {"status": "failed", "reason": "extract is only available during execution"}
            result = self.environment.commit_extraction(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                amount = int(args["amount"])
                self._post_public_message(
                    agent_name,
                    f"{agent_name} requested {amount} units from the pool for season {iteration}.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
