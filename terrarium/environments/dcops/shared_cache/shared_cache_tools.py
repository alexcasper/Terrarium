"""Tools for the SharedCacheEnvironment.

Planning phase: `propose_cache_plan` lets an agent announce which keys they
intend to write to the shared cache this round on the blackboard (non-binding).
Execution phase: `commit_cache_action` commits any cache writes (`write_entries`)
and selects how to resolve the assigned query key (`use_cache` or `compute_locally`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_cache_plan"})
EXECUTION_TOOLS = frozenset({"commit_cache_action"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class SharedCacheTools:
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
                    "name": "propose_cache_plan",
                    "description": (
                        "Publicly announce on the blackboard which keys (if any) you plan "
                        "to write to the shared cache this round. This is a non-binding "
                        "announcement; actual writes and query resolution are committed "
                        "during execution with commit_cache_action."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keys_to_write": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "List of key identifiers (e.g. ['key_0']) you plan to "
                                    "write to the shared cache this round, or [] if none."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional note or query request posted alongside the plan.",
                            },
                        },
                        "required": ["keys_to_write"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_cache_action",
                    "description": (
                        "Commit your cache writes and query resolution mode for this round. "
                        "All committed write_entries are applied to the shared cache before "
                        "queries are resolved. Each entry written incurs write_cost credits. "
                        "Then your target query key is resolved via resolution_mode "
                        "('use_cache' or 'compute_locally'). This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "write_entries": {
                                "type": "object",
                                "additionalProperties": {"type": "integer"},
                                "description": (
                                    "Optional dictionary mapping key names (e.g. 'key_2') to "
                                    "integer values to write into the shared cache this round. "
                                    "Omit or pass {} to write nothing."
                                ),
                            },
                            "resolution_mode": {
                                "type": "string",
                                "enum": ["use_cache", "compute_locally"],
                                "description": (
                                    "How to resolve your assigned target query key: 'use_cache' "
                                    "reads the shared cache (falling back to local compute on a "
                                    "cache miss), or 'compute_locally' bypasses the cache and "
                                    "computes locally."
                                ),
                            },
                        },
                        "required": ["resolution_mode"],
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
            return {"error": f"SharedCache environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_cache_plan":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_cache_plan is only available during planning",
                }
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                proposal = result["proposal"]
                keys = proposal["keys_to_write"]
                if keys:
                    text = f"{agent_name} plans to write keys {keys} to the shared cache."
                else:
                    text = f"{agent_name} plans to write no keys to the shared cache this round."
                if proposal.get("message"):
                    text = f"{text} Note: {proposal['message']}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_cache_action":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "commit_cache_action is only available during execution",
                }
            result = self.environment.commit_cache_action(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                written_keys = sorted(result["write_entries"].keys())
                self._post_public_message(
                    agent_name,
                    f"{agent_name} committed cache action (wrote keys={written_keys}, "
                    f"resolution_mode={result['resolution_mode']}).",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
