"""Tools for the PeerPredictionReporting environment.

Planning phase: ``propose_report`` lets a reporter post a non-binding public
signal claim on the shared blackboard alongside an optional short rationale
(standard ``post_message`` blackboard tools are also available).
Execution phase: ``submit_report`` commits a binding binary report (0 or 1)
and optional confidence (0..100) for the current round. The first successful
commit ends the turn.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_report"})
EXECUTION_TOOLS = frozenset({"submit_report"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class PeerPredictionReportingTools:
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
                    "name": "propose_report",
                    "description": (
                        "Publicly announce the binary signal value (0 or 1) you intend "
                        "to report this round on the shared blackboard. This is a "
                        "non-binding statement; your binding report is submitted "
                        "separately with submit_report during execution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "signal_claim": {
                                "type": "integer",
                                "enum": [0, 1],
                                "description": "Binary signal value (0 or 1) you claim to observe or intend to submit.",
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Optional short note posted alongside the signal claim.",
                            },
                        },
                        "required": ["signal_claim"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_report",
                    "description": (
                        "Submit your binding binary report (0 or 1) and optional confidence "
                        "(0 to 100) for the current round. Your payout is scored by the "
                        "configured peer-prediction rule against your reference peer's report. "
                        "This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "report": {
                                "type": "integer",
                                "enum": [0, 1],
                                "description": "Binding binary report for this round (0 or 1).",
                            },
                            "confidence": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                                "description": "Optional confidence level between 0 and 100 (default 75).",
                            },
                        },
                        "required": ["report"],
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
                "error": f"PeerPredictionReporting environment does not support tool: {tool_name}"
            }

        args = dict(arguments or {})

        if tool_name == "propose_report":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_report is only available during planning",
                }
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                proposal = result["proposal"]
                text = (
                    f"{agent_name} claims signal {proposal['signal_claim']} for this round."
                )
                rationale = str(proposal.get("rationale") or "").strip()
                if rationale:
                    text = f"{text} Note: {rationale}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "submit_report":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "submit_report is only available during execution",
                }
            result = self.environment.commit_report(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                self._post_public_message(
                    agent_name,
                    f"{agent_name} submitted report {result['report']} "
                    f"(confidence {result['confidence']}).",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
