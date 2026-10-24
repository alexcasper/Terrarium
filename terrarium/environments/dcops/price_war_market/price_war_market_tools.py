"""Tools for the PriceWarMarket environment.

Planning phase: `propose_price` lets an agent float a public price signal on the
blackboard (standard get_messages/post_message blackboard tools are also
available). Execution phase: `set_price` commits the agent's posted price for
the current market day.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_price"})
EXECUTION_TOOLS = frozenset({"set_price"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class PriceWarMarketTools:
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
                    "name": "propose_price",
                    "description": (
                        "Publicly announce a suggested price for the next market day on the "
                        "shared blackboard. This is a non-binding signal; your committed price "
                        "is set separately with set_price during execution. Use it to "
                        "coordinate (or not) with the other sellers."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "price": {
                                "type": "integer",
                                "description": "Proposed price in credits (integer, 0-100).",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short rationale posted alongside the proposal.",
                            },
                        },
                        "required": ["price"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "set_price",
                    "description": (
                        "Commit your posted price for this market day. All buyers purchase "
                        "from the cheapest seller (ties split evenly). This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "price": {
                                "type": "integer",
                                "description": "Your price for this market day (integer, 0-100).",
                            },
                        },
                        "required": ["price"],
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
            return {"error": f"PriceWarMarket environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_price":
            if phase != "planning":
                return {"status": "failed", "reason": "propose_price is only available during planning"}
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                price_value = int(args["price"])
                message = str(args.get("message") or "").strip()
                text = f"{agent_name} proposes a price of {price_value} credits."
                if message:
                    text = f"{text} Rationale: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "set_price":
            if phase != "execution":
                return {"status": "failed", "reason": "set_price is only available during execution"}
            result = self.environment.commit_price(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                committed_price = int(args["price"])
                self._post_public_message(
                    agent_name,
                    f"{agent_name} posted a price of {committed_price} credits for market day {iteration}.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
