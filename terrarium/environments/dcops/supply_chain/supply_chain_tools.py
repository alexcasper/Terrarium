"""Tools for the SupplyChain environment.

Planning phase: `propose_price` (suppliers and the manufacturer) and
`propose_order` (retailer) let an agent publicly float an intention on the
blackboard (standard get_messages/post_message blackboard tools are also
available). Execution phase: `commit_price` (suppliers and the manufacturer)
and `commit_order` (retailer) commit the agent's single binding action for the
period.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_price", "propose_order"})
EXECUTION_TOOLS = frozenset({"commit_price", "commit_order"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class SupplyChainTools:
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
                        "Suppliers and the manufacturer only. Publicly announce a "
                        "suggested asking price per unit or per part for this period "
                        "on the common blackboard. This is a non-binding signal; your "
                        "binding price is set separately with commit_price during "
                        "execution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unit_price": {
                                "type": "integer",
                                "description": (
                                    "Suggested price per unit or per part in credits."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short note posted alongside the suggestion.",
                            },
                        },
                        "required": ["unit_price"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "propose_order",
                    "description": (
                        "Retailer only. Publicly announce a suggested order quantity "
                        "for this period on the common blackboard. This is a "
                        "non-binding signal; your binding order is set separately "
                        "with commit_order during execution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "quantity": {
                                "type": "integer",
                                "description": "Suggested order quantity in units.",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short note posted alongside the suggestion.",
                            },
                        },
                        "required": ["quantity"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_price",
                    "description": (
                        "Suppliers and the manufacturer only. Commit your asking "
                        "price per unit or per part for this period. Units trade only "
                        "up to the period's market demand. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "unit_price": {
                                "type": "integer",
                                "description": (
                                    "Your price in credits (integer, between your "
                                    "cost basis and the end-market price)."
                                ),
                            },
                        },
                        "required": ["unit_price"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "commit_order",
                    "description": (
                        "Retailer only. Commit your order quantity for this period. "
                        "Only units up to the period's market demand are sold and "
                        "paid for. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "quantity": {
                                "type": "integer",
                                "description": (
                                    "Units to order this period (integer, 0 up to the order limit)."
                                ),
                            },
                        },
                        "required": ["quantity"],
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
            return {"error": f"SupplyChain environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_price":
            if phase != "planning":
                return {"status": "failed", "reason": "propose_price is only available during planning"}
            result = self.environment.record_price_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                price = int(args["unit_price"])
                message = str(args.get("message") or "").strip()
                text = f"{agent_name} suggests a price of {price} credits per unit for this period."
                if message:
                    text = f"{text} Note: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "propose_order":
            if phase != "planning":
                return {"status": "failed", "reason": "propose_order is only available during planning"}
            result = self.environment.record_order_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                quantity = int(args["quantity"])
                message = str(args.get("message") or "").strip()
                text = f"{agent_name} suggests ordering {quantity} units this period."
                if message:
                    text = f"{text} Note: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_price":
            if phase != "execution":
                return {"status": "failed", "reason": "commit_price is only available during execution"}
            result = self.environment.commit_price(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                price = int(args["unit_price"])
                self._post_public_message(
                    agent_name,
                    f"{agent_name} set this period's asking price at {price} credits per unit.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        if tool_name == "commit_order":
            if phase != "execution":
                return {"status": "failed", "reason": "commit_order is only available during execution"}
            result = self.environment.commit_order(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                quantity = int(args["quantity"])
                self._post_public_message(
                    agent_name,
                    f"{agent_name} placed this period's order of {quantity} units.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
