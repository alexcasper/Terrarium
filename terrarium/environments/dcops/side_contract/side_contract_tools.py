"""Tools for the SideContract environment.

Planning phase:
- ``propose_allocation`` announces a non-binding intended slot on the shared
  blackboard;
- ``propose_transfer`` (when ``side_contracts_enabled=True``) offers a binding
  conditional credit transfer to another participant if they commit to a
  specified slot in execution;
- ``accept_transfer`` (when ``side_contracts_enabled=True``) accepts a pending
  transfer contract addressed to the caller.

Execution phase:
- ``commit_slot`` commits the participant's binding slot choice for the round.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_allocation", "propose_transfer", "accept_transfer"})
EXECUTION_TOOLS = frozenset({"commit_slot"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class SideContractTools:
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
        tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "propose_allocation",
                    "description": (
                        "Publicly announce which task slot you intend to claim this "
                        "round on the shared blackboard. This is a non-binding signal; "
                        "your actual slot is set with commit_slot during execution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slot": {
                                "type": "integer",
                                "description": "Task slot index you intend to claim (0 to num_slots - 1).",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short note posted alongside the proposal.",
                            },
                        },
                        "required": ["slot"],
                    },
                },
            },
        ]
        side_enabled = (
            True
            if self.environment is None
            else bool(getattr(self.environment, "side_contracts_enabled", True))
        )
        if side_enabled:
            tools.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "propose_transfer",
                            "description": (
                                "Propose a conditional bilateral transfer to another participant "
                                "for this round. If the recipient accepts the contract during "
                                "planning and commits to condition_recipient_slot during execution, "
                                "amount credits are automatically transferred from you to the "
                                "recipient after slots settle."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "recipient": {
                                        "type": "string",
                                        "description": "Participant identifier to receive the conditional transfer offer.",
                                    },
                                    "amount": {
                                        "type": "integer",
                                        "description": "Positive integer credit amount to transfer if the condition holds.",
                                    },
                                    "condition_recipient_slot": {
                                        "type": "integer",
                                        "description": "Slot index the recipient must commit to in execution for the transfer to settle.",
                                    },
                                    "message": {
                                        "type": "string",
                                        "description": "Optional note posted with the contract offer.",
                                    },
                                },
                                "required": [
                                    "recipient",
                                    "amount",
                                    "condition_recipient_slot",
                                ],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "accept_transfer",
                            "description": (
                                "Accept a pending bilateral transfer contract addressed to you "
                                "this round. Once accepted, the transfer settles automatically "
                                "after execution if you commit to the specified slot."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "contract_id": {
                                        "type": "integer",
                                        "description": "Identifier of the proposed contract to accept.",
                                    },
                                },
                                "required": ["contract_id"],
                            },
                        },
                    },
                ]
            )
        return tools

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_slot",
                    "description": (
                        "Commit your binding task slot claim for this round. After all "
                        "participants commit, slot payoffs and any accepted conditional "
                        "transfers settle. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slot": {
                                "type": "integer",
                                "description": "Task slot index to claim this round (0 to num_slots - 1).",
                            },
                        },
                        "required": ["slot"],
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
            return {"error": f"SideContract environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_allocation":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_allocation is only available during planning",
                }
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                slot = result["proposal"]["slot"]
                text = f"{agent_name} intends to claim slot {slot} this round."
                message = str(args.get("message") or "").strip()
                if message:
                    text = f"{text} Note: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "propose_transfer":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "propose_transfer is only available during planning",
                }
            result = self.environment.propose_transfer(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                c = result["contract"]
                text = (
                    f"{agent_name} proposed contract #{c['contract_id']} to {c['recipient']}: "
                    f"transfer {c['amount']} credits if {c['recipient']} commits to slot "
                    f"{c['condition_recipient_slot']}."
                )
                if c.get("message"):
                    text = f"{text} Note: {c['message']}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "accept_transfer":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "accept_transfer is only available during planning",
                }
            result = self.environment.accept_transfer(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                c = result["contract"]
                text = (
                    f"{agent_name} accepted contract #{c['contract_id']} from {c['proposer']} "
                    f"({c['amount']} credits conditional on slot {c['condition_recipient_slot']})."
                )
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_slot":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "commit_slot is only available during execution",
                }
            result = self.environment.commit_slot(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                self._post_public_message(
                    agent_name,
                    f"{agent_name} committed to slot {result['slot']} this round.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
