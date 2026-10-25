"""Tools for the MultiIssueBarterEnvironment.

Planning phase: ``propose_package`` lets a trader float a non-binding allocation
proposal (own requested units across the issues) on the shared blackboard
(standard ``post_message`` is also available). Execution phase: ``commit_deal``
commits the trader's binding unit claim vector for the active issue(s).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_package"})
EXECUTION_TOOLS = frozenset({"commit_deal"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class MultiIssueBarterTools:
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
                    "name": "propose_package",
                    "description": (
                        "Publicly propose a unit allocation on the shared blackboard. "
                        "Pass 'allocations' as a list of integer units (each 0 to total_units) "
                        "that you would receive on each issue (with the remaining units "
                        "available to the other trader(s)). Non-binding during planning; "
                        "final settlement uses commit_deal during execution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "allocations": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": (
                                    "List of integer unit counts (0..total_units) you propose "
                                    "to receive for each issue (length num_issues in package "
                                    "mode, or length 1 or num_issues in sequential mode)."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional rationale or counter-offer note posted with the proposal.",
                            },
                        },
                        "required": ["allocations"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_deal",
                    "description": (
                        "Commit your binding unit claim for each active issue. Pass "
                        "'allocations' as a list of integer units (0..total_units) that "
                        "you claim for yourself on each issue. On any issue where the sum "
                        "of claims across all traders is at most total_units, each trader "
                        "receives their claimed units (plus an even split of any unclaimed "
                        "units). On any issue where total claims exceed total_units, that "
                        "issue reverts to its status-quo endowment. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "allocations": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": (
                                    "List of integer unit counts (0..total_units) you claim "
                                    "for yourself on each issue (length num_issues, or length 1 "
                                    "for the active issue in sequential mode)."
                                ),
                            },
                        },
                        "required": ["allocations"],
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
            return {"error": f"MultiIssueBarter environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_package":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_package is only available during planning",
                }
            result = self.environment.record_proposal(
                agent_name=agent_name, arguments=args, iteration=iteration
            )
            if result.get("status") == "success":
                proposal = result["proposal"]
                demands_str = ", ".join(
                    f"{k}: {v}/{self.environment.total_units}"
                    for k, v in proposal["active_demands"].items()
                )
                text = f"{agent_name} proposes claiming [{demands_str}] (own units)."
                message = str(args.get("message") or "").strip()
                if message:
                    text = f"{text} Note: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_deal":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "commit_deal is only available during execution",
                }
            result = self.environment.commit_deal(
                agent_name=agent_name, arguments=args, iteration=iteration
            )
            if result.get("status") == "success":
                self._post_public_message(
                    agent_name,
                    f"{agent_name} has submitted a binding deal commitment for "
                    f"{result['active_issues']}.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
