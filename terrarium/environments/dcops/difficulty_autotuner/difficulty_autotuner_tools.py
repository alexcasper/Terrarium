"""Tools for the InstanceDifficultyAutotuner environment.

Planning phase: `propose_assignment` lets a solver announce a tentative color
choice on the shared blackboard (non-binding). Execution phase: `commit_assignment`
locks in the solver's color index for the current round.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_assignment"})
EXECUTION_TOOLS = frozenset({"commit_assignment"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class InstanceDifficultyAutotunerTools:
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
                    "name": "propose_assignment",
                    "description": (
                        "Publicly announce a tentative color index on the shared blackboard "
                        "for the current round. This is a non-binding signal; your binding "
                        "choice is submitted during execution with commit_assignment."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "color": {
                                "type": "integer",
                                "description": "Tentative integer color index in 0..(num_colors - 1).",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short note posted alongside the proposal.",
                            },
                        },
                        "required": ["color"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_assignment",
                    "description": (
                        "Commit your binding integer color index in 0..(num_colors - 1) for "
                        "this round. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "color": {
                                "type": "integer",
                                "description": "Binding integer color index in 0..(num_colors - 1).",
                            },
                        },
                        "required": ["color"],
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
            return {
                "error": f"InstanceDifficultyAutotuner environment does not support tool: {tool_name}"
            }

        args = dict(arguments or {})

        if tool_name == "propose_assignment":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_assignment is only available during planning",
                }
            result = self.environment.record_proposal(
                agent_name=agent_name, arguments=args, iteration=iteration
            )
            if result.get("status") == "success":
                proposal = result["proposal"]
                text = f"{agent_name} proposes selecting color {proposal['color']} this round."
                if proposal.get("message"):
                    text = f"{text} Note: {proposal['message']}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_assignment":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "commit_assignment is only available during execution",
                }
            result = self.environment.commit_assignment(
                agent_name=agent_name, arguments=args, iteration=iteration
            )
            if result.get("status") == "success":
                self._post_public_message(
                    agent_name,
                    f"{agent_name} committed color {result['color']} for this round.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
