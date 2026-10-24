"""Tools for the AuctionHouse environment.

Planning phase: `signal_bid` floats a non-binding bid intention on the public
blackboard (standard get_messages/post_message are also available). Execution
phase: `submit_bid` seals a bid — the amount is NOT posted to the blackboard,
only the fact of participation, so bids stay sealed from rivals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"signal_bid"})
EXECUTION_TOOLS = frozenset({"submit_bid"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class AuctionHouseTools:
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
                    "name": "signal_bid",
                    "description": (
                        "Publicly announce an intended bid for today's auction on the shared "
                        "blackboard. Signals are non-binding: your actual sealed bid is placed "
                        "separately with submit_bid during execution. Use it to probe or "
                        "coordinate (or not) with the other bidders."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "integer",
                                "description": "Intended bid amount in credits.",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short rationale posted alongside the signal.",
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
                    "name": "submit_bid",
                    "description": (
                        "Seal your bid for today's auction. The highest bid wins the item and "
                        "pays its own bid. Your bid amount is kept sealed from the other "
                        "bidders. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bid": {
                                "type": "integer",
                                "description": "Your sealed bid in credits (>= minimum bid).",
                            },
                        },
                        "required": ["bid"],
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
            return {"error": f"AuctionHouse environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "signal_bid":
            if phase != "planning":
                return {"status": "failed", "reason": "signal_bid is only available during planning"}
            result = self.environment.record_signal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                amount = int(args["amount"])
                message = str(args.get("message") or "").strip()
                text = f"{agent_name} signals an intent to bid {amount} credits."
                if message:
                    text = f"{text} Rationale: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "submit_bid":
            if phase != "execution":
                return {"status": "failed", "reason": "submit_bid is only available during execution"}
            result = self.environment.commit_bid(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                # Sealed: announce participation, never the amount.
                self._post_public_message(
                    agent_name,
                    f"{agent_name} submitted a sealed bid for auction {iteration}.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
