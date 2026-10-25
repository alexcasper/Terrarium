"""Tools for the DisasterRelief environment.

Planning phase: `propose_deployment` lets a responder float a public,
non-binding deployment intent and optional capability declaration on the
blackboard (standard get_messages / post_message blackboard tools are also
available). Execution phase: exactly one binding commit — `commit_deployment`
(deploy to an incident site) or `commit_standby` (remain at base at 0 cost).
The first successful commit ends the turn.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_deployment"})
EXECUTION_TOOLS = frozenset({"commit_deployment", "commit_standby"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class DisasterReliefTools:
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
                    "name": "propose_deployment",
                    "description": (
                        "Publicly announce your intended incident site deployment and "
                        "optionally declare capabilities for this wave on the shared "
                        "blackboard. This is a non-binding signal; your actual move is "
                        "set during execution with commit_deployment or commit_standby."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "site_id": {
                                "type": "integer",
                                "description": "Target incident site index (0..num_incidents-1).",
                            },
                            "declared_capabilities": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "medical",
                                        "heavy_lift",
                                        "hazmat",
                                        "water_rescue",
                                    ],
                                },
                                "description": (
                                    "Optional list of capabilities you wish to declare "
                                    "publicly alongside your proposal."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short note posted alongside the proposal.",
                            },
                        },
                        "required": ["site_id"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_deployment",
                    "description": (
                        "Deploy your response unit to the specified incident site this "
                        "wave. You pay the deployment cost and share any site reward "
                        "earned by the units deployed there. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "site_id": {
                                "type": "integer",
                                "description": "Target incident site index (0..num_incidents-1).",
                            },
                        },
                        "required": ["site_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "commit_standby",
                    "description": (
                        "Keep your response unit on standby at base this wave at 0 cost "
                        "and 0 reward. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
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
            return {"error": f"DisasterRelief environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_deployment":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_deployment is only available during planning",
                }
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                proposal = result["proposal"]
                caps = proposal.get("declared_capabilities") or []
                caps_note = f" declaring capabilities {caps}" if caps else ""
                text = (
                    f"{agent_name} proposes deploying to site {proposal['site_id']}{caps_note}."
                )
                message = str(args.get("message") or "").strip()
                if message:
                    text = f"{text} Note: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_deployment":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "commit_deployment is only available during execution",
                }
            result = self.environment.commit_deployment(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                self._post_public_message(
                    agent_name,
                    f"{agent_name} commits deployment to site {result['site_id']} this wave.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        if tool_name == "commit_standby":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "commit_standby is only available during execution",
                }
            result = self.environment.commit_standby(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                self._post_public_message(
                    agent_name,
                    f"{agent_name} remains on standby this wave.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
