"""Tools for the AdAuctionGSP environment.

Planning phase: `propose_bid` lets an advertiser float a public bid signal on
the blackboard (standard get_messages/post_message blackboard tools are also
available). Execution phase: `commit_bid` seals the advertiser's per-click bid
for the current iteration. Committed amounts are never posted publicly before
settlement — bids are sealed and revealed in the iteration result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_bid"})
EXECUTION_TOOLS = frozenset({"commit_bid"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class AdAuctionGSPTools:
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
                    "name": "propose_bid",
                    "description": (
                        "Publicly announce a per-click bid you are considering for this "
                        "iteration on the blackboard. This is a non-binding signal; your "
                        "actual bid is sealed with commit_bid during execution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bid": {
                                "type": "number",
                                "description": "Per-click bid you are considering.",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short rationale posted alongside the proposal.",
                            },
                        },
                        "required": ["bid"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_bid",
                    "description": (
                        "Seal your per-click bid for this iteration. Bids are ranked from "
                        "highest to lowest; if you win a slot, your cost per click is the "
                        "bid of the advertiser ranked directly below you (never below the "
                        "reserve price). This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bid": {
                                "type": "number",
                                "description": "Your sealed per-click bid for this iteration.",
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
            return {"error": f"AdAuctionGSP environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_bid":
            if phase != "planning":
                return {"status": "failed", "reason": "propose_bid is only available during planning"}
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                message = str(args.get("message") or "").strip()
                text = f"{agent_name} floated a bid signal for iteration {iteration}."
                if message:
                    text = f"{text} Rationale: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_bid":
            if phase != "execution":
                return {"status": "failed", "reason": "commit_bid is only available during execution"}
            result = self.environment.commit_bid(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                # Sealed-bid: never reveal the amount before settlement.
                self._post_public_message(
                    agent_name,
                    f"{agent_name} submitted a sealed bid for iteration {iteration}.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
