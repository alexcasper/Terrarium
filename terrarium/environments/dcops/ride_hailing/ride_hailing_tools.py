"""Tools for the RideHailingDispatch environment.

Planning phase: `propose_action` lets a driver float a public, non-binding
intent (serve / reposition) on the blackboard (standard get_messages /
post_message blackboard tools are also available). Execution phase: exactly one
binding commit — `commit_serve` (stay and serve) or `commit_move` (reposition
to an adjacent zone). The first successful commit ends the turn.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_action"})
EXECUTION_TOOLS = frozenset({"commit_serve", "commit_move"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class RideHailingDispatchTools:
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
                        "Publicly announce your intended move for this wave on the shared "
                        "blackboard. This is a non-binding signal; what you actually do is "
                        "set separately with commit_serve or commit_move during execution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["serve", "move"],
                                "description": "The action you intend for this wave.",
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["cw", "ccw"],
                                "description": "Required when action is 'move': cw = next zone index, ccw = previous zone index.",
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
                    "name": "commit_serve",
                    "description": (
                        "Stay in your current zone and serve riders there this wave (up to "
                        "your capacity). Earnings are the zone revenue split among the "
                        "drivers serving that zone. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "commit_move",
                    "description": (
                        "Reposition to an adjacent zone on the ring (cw = +1, ccw = -1, "
                        "wrapping around). You earn nothing this wave and arrive in the "
                        "new zone for the next wave. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["cw", "ccw"],
                                "description": "Direction around the ring: cw (+1 zone) or ccw (-1 zone).",
                            },
                        },
                        "required": ["direction"],
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
            return {"error": f"RideHailingDispatch environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_action":
            if phase != "planning":
                return {"status": "failed", "reason": "propose_action is only available during planning"}
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                proposal = result["proposal"]
                if proposal["action"] == "serve":
                    text = f"{agent_name} is thinking of serving riders this wave."
                else:
                    text = (
                        f"{agent_name} is thinking of repositioning {proposal['direction']} "
                        f"this wave."
                    )
                message = str(args.get("message") or "").strip()
                if message:
                    text = f"{text} Note: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_serve":
            if phase != "execution":
                return {"status": "failed", "reason": "commit_serve is only available during execution"}
            result = self.environment.commit_serve(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                self._post_public_message(
                    agent_name,
                    f"{agent_name} is serving riders in zone {result['zone']} this wave.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        if tool_name == "commit_move":
            if phase != "execution":
                return {"status": "failed", "reason": "commit_move is only available during execution"}
            result = self.environment.commit_move(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                self._post_public_message(
                    agent_name,
                    f"{agent_name} is repositioning {result['direction']} to zone "
                    f"{result['to_zone']} this wave (no earnings while moving).",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
