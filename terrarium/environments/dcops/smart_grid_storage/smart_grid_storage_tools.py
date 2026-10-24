"""Tools for the SmartGridStorageArbitrage environment.

Planning phase: `propose_action` floats a public, non-binding charge/discharge/
hold signal on the blackboard (standard get_messages/post_message blackboard
tools are also available). Execution phase: exactly one of `commit_charge`,
`commit_discharge` or `hold` binds the owner's action for the current block.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_action"})
EXECUTION_TOOLS = frozenset({"commit_charge", "commit_discharge", "hold"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class SmartGridStorageTools:
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
                    "name": "propose_action",
                    "description": (
                        "Publicly announce a suggested battery action for the upcoming block on "
                        "the shared blackboard. This is a non-binding signal; your actual move is "
                        "made separately during execution with commit_charge / commit_discharge / "
                        "hold. Use it to share your thinking with the other owners."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["charge", "discharge", "hold"],
                                "description": "The suggested action for the upcoming block.",
                            },
                            "amount": {
                                "type": "integer",
                                "description": (
                                    "Suggested MWh for charge/discharge (1 to your battery "
                                    "capacity); omit for hold."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short rationale posted alongside the proposal.",
                            },
                        },
                        "required": ["action"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_charge",
                    "description": (
                        "Bind your action for this block: buy and store up to your remaining "
                        "room. You pay the block's realized price per MWh. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "integer",
                                "description": "MWh to charge (integer, 1 to remaining room).",
                            },
                        },
                        "required": ["amount"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "commit_discharge",
                    "description": (
                        "Bind your action for this block: sell stored energy. You are paid the "
                        "block's realized price per MWh. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "integer",
                                "description": "MWh to discharge (integer, 1 to your stored level).",
                            },
                        },
                        "required": ["amount"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "hold",
                    "description": (
                        "Bind your action for this block: do nothing, keep your battery as it "
                        "is. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
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
            return {"error": f"SmartGridStorage environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_action":
            if phase != "planning":
                return {"status": "failed", "reason": "propose_action is only available during planning"}
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                action = result["proposed_action"]
                amount = result["amount"]
                text = f"{agent_name} suggests {action}"
                text += f" {amount} MWh" if amount else ""
                text += " for the upcoming block."
                message = str(args.get("message") or "").strip()
                if message:
                    text = f"{text} Rationale: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if phase != "execution":
            return {
                "status": "failed",
                "reason": f"{tool_name} is only available during execution",
            }

        if tool_name == "commit_charge":
            result = self.environment.commit_charge(agent_name=agent_name, arguments=args)
        elif tool_name == "commit_discharge":
            result = self.environment.commit_discharge(agent_name=agent_name, arguments=args)
        else:  # hold
            result = self.environment.commit_hold(agent_name)

        if result.get("status") == "success":
            action = result["action"]
            amount = result["amount"]
            text = f"{agent_name} commits to {action}"
            text += f" {amount} MWh" if amount else ""
            text += f" for block {iteration}."
            self._post_public_message(agent_name, text, phase, iteration)
        return _mark_committed(result)
