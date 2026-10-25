"""Tools for the FederatedUpdate environment.

Planning phase: `propose_update_summary` lets a node post a structured summary
of its intended update (L2 norm, dominant coordinate index, optional anomalous
peer suspect, and short note) on the shared blackboard.
Execution phase: `submit_update` commits the node's numeric update vector in
R^D and an optional peer anomaly flag for the current aggregation round.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_update_summary"})
EXECUTION_TOOLS = frozenset({"submit_update"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class FederatedUpdateTools:
    supports_private_channels = True

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
                    "name": "propose_update_summary",
                    "description": (
                        "Share a compact summary of your local gradient update on the public "
                        "blackboard (estimated L2 norm, dominant coordinate index, optional "
                        "anomalous peer node ID, and optional note). Non-binding signal."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "norm": {
                                "type": "number",
                                "description": "Non-negative L2 norm of the update vector you plan to submit.",
                            },
                            "dominant_coord": {
                                "type": "integer",
                                "description": "0-based coordinate index with the largest magnitude in your update.",
                            },
                            "suspect_node": {
                                "type": "string",
                                "description": "Optional peer node ID (e.g. 'node_4') whose updates appear anomalous.",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short note posted with your summary.",
                            },
                        },
                        "required": ["norm", "dominant_coord"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_update",
                    "description": (
                        "Commit your numeric update vector for this aggregation round, and "
                        "optionally flag at most one peer node as anomalous. Ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vector": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Numeric update vector of length dim, each entry within bounds.",
                            },
                            "flag_peer": {
                                "type": "string",
                                "description": "Optional peer node ID (other than yourself) to flag as anomalous.",
                            },
                        },
                        "required": ["vector"],
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
            return {"error": f"FederatedUpdate environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_update_summary":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_update_summary is only available during planning",
                }
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                proposal = result["proposal"]
                text = (
                    f"{agent_name} reports update summary: norm={proposal['norm']}, "
                    f"dominant_coord={proposal['dominant_coord']}"
                )
                if proposal.get("suspect_node"):
                    text += f", suspect_node={proposal['suspect_node']}"
                text += "."
                if proposal.get("message"):
                    text += f" Note: {proposal['message']}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "submit_update":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "submit_update is only available during execution",
                }
            result = self.environment.commit_update(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                flag_note = (
                    f" and flagged {result['flag_peer']}" if result.get("flag_peer") else ""
                )
                self._post_public_message(
                    agent_name,
                    f"{agent_name} submitted its {len(result['vector'])}-D update vector{flag_note}.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
