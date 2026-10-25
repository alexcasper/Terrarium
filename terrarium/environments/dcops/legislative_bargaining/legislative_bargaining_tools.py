"""Tools for the LegislativeBargaining environment.

Planning phase: `propose_vote_slate` lets a legislator publish a non-binding
intended vote slate across the bills on the current session docket.
Execution phase: `cast_votes` commits the legislator's binding 'yes'/'no'
roll-call votes on all bills for the session.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_vote_slate"})
EXECUTION_TOOLS = frozenset({"cast_votes"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class LegislativeBargainingTools:
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
                    "name": "propose_vote_slate",
                    "description": (
                        "Publicly announce a non-binding intended vote slate ('yes' or 'no' "
                        "per bill) on the shared blackboard for the current session. Binding "
                        "votes are submitted separately with cast_votes during execution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "votes": {
                                "type": "object",
                                "description": (
                                    "Mapping from bill ID (e.g. 'bill_A', 'bill_B', 'bill_C') "
                                    "to intended vote ('yes' or 'no')."
                                ),
                                "additionalProperties": {
                                    "type": "string",
                                    "enum": ["yes", "no"],
                                },
                            },
                            "message": {
                                "type": "string",
                                "description": (
                                    "Optional short message or rationale posted alongside the "
                                    "proposed vote slate."
                                ),
                            },
                        },
                        "required": ["votes"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "cast_votes",
                    "description": (
                        "Commit your binding roll-call votes ('yes' or 'no') for every bill "
                        "on the docket in the current legislative session. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "votes": {
                                "type": "object",
                                "description": (
                                    "Mapping from every bill ID on the current session docket "
                                    "(e.g. 'bill_A', 'bill_B', 'bill_C') to 'yes' or 'no'."
                                ),
                                "additionalProperties": {
                                    "type": "string",
                                    "enum": ["yes", "no"],
                                },
                            },
                        },
                        "required": ["votes"],
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
                "error": f"LegislativeBargaining environment does not support tool: {tool_name}"
            }

        args = dict(arguments or {})

        if tool_name == "propose_vote_slate":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_vote_slate is only available during planning",
                }
            result = self.environment.record_proposal(
                agent_name=agent_name, arguments=args, iteration=iteration
            )
            if result.get("status") == "success":
                proposal = result["proposal"]
                slate_str = ", ".join(
                    f"{bill_id}={vote}" for bill_id, vote in proposal["votes"].items()
                )
                text = f"{agent_name} proposes vote slate: {slate_str}."
                if proposal.get("message"):
                    text = f"{text} Note: {proposal['message']}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "cast_votes":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "cast_votes is only available during execution",
                }
            result = self.environment.commit_votes(
                agent_name=agent_name, arguments=args, iteration=iteration
            )
            if result.get("status") == "success":
                slate_str = ", ".join(
                    f"{bill_id}={vote}" for bill_id, vote in result["votes"].items()
                )
                self._post_public_message(
                    agent_name,
                    f"{agent_name} casts roll-call votes: {slate_str}.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
