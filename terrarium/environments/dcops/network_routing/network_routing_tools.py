"""Tools for the NetworkRouting environment.

Planning phase: `propose_route` lets an agent float a public link signal on the
blackboard (standard get_messages/post_message blackboard tools are also
available). Execution phase: `commit_route` binds the agent's traffic to one
link for the current iteration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"propose_route"})
EXECUTION_TOOLS = frozenset({"commit_route"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class NetworkRoutingTools:
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
                    "name": "propose_route",
                    "description": (
                        "Publicly announce a link you are considering for this iteration on "
                        "the blackboard. This is a non-binding signal; your actual route is "
                        "fixed with commit_route during execution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "link": {
                                "type": "string",
                                "description": "Name of the link you are considering.",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional short rationale posted alongside the proposal.",
                            },
                        },
                        "required": ["link"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_route",
                    "description": (
                        "Commit the link that carries your traffic units for this iteration. "
                        "A link's per-unit latency depends on the total units all routers "
                        "place on it. This ends your turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "link": {
                                "type": "string",
                                "description": "Name of the link your traffic takes this iteration.",
                            },
                        },
                        "required": ["link"],
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
            return {"error": f"NetworkRouting environment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "propose_route":
            if phase != "planning":
                return {"status": "failed", "reason": "propose_route is only available during planning"}
            result = self.environment.record_proposal(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                link = str(args["link"])
                message = str(args.get("message") or "").strip()
                text = f"{agent_name} is considering link {link}."
                if message:
                    text = f"{text} Rationale: {message}"
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_route":
            if phase != "execution":
                return {"status": "failed", "reason": "commit_route is only available during execution"}
            result = self.environment.commit_route(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                committed_link = str(args["link"])
                self._post_public_message(
                    agent_name,
                    f"{agent_name} routed its traffic over {committed_link} for iteration {iteration}.",
                    phase,
                    iteration,
                )
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
