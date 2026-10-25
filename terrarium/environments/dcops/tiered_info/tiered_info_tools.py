"""Tools for the TieredInfoEnvironment.

Planning phase:
- ``post_open_claim``: post a zero-quorum claim to ``open_feed`` (Tier 1).
- ``endorse_verified_claim``: endorse a hypothesis for promotion into
  ``verified_ledger`` (Tier 2); once ``quorum_size`` distinct analysts endorse
  the same hypothesis, it enters ``verified_ledger``.

Execution phase:
- ``commit_conclusion``: submit the analyst's final hypothesis conclusion and
  cite the backing epistemic tier (``private_evidence``, ``open_feed``, or
  ``verified_ledger``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

PLANNING_TOOLS = frozenset({"post_open_claim", "endorse_verified_claim"})
EXECUTION_TOOLS = frozenset({"commit_conclusion"})


def _mark_committed(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a successful action so BaseAgent ends the turn."""
    if isinstance(result, dict) and result.get("status") == "success":
        return {**result, "action_committed": True}
    return result


class TieredInfoTools:
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
                    "name": "post_open_claim",
                    "description": (
                        "Post an unverified hypothesis claim directly to the Tier-1 "
                        "open_feed with zero quorum requirement."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hypothesis": {
                                "type": "integer",
                                "description": "Candidate hypothesis index (e.g. 0..3).",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional rationale or clue summary for the claim.",
                            },
                        },
                        "required": ["hypothesis"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "endorse_verified_claim",
                    "description": (
                        "Add your endorsement signature to a candidate hypothesis for "
                        "the Tier-2 verified_ledger. When enough distinct analysts "
                        "endorse the same hypothesis to meet the quorum threshold, it "
                        "is promoted to verified_ledger."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hypothesis": {
                                "type": "integer",
                                "description": "Candidate hypothesis index to endorse for verification.",
                            },
                            "message": {
                                "type": "string",
                                "description": "Optional evidence note supporting the endorsement.",
                            },
                        },
                        "required": ["hypothesis"],
                    },
                },
            },
        ]

    def _execution_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "commit_conclusion",
                    "description": (
                        "Commit your final hypothesis conclusion for this round and "
                        "record which epistemic tier you are citing as support "
                        "('private_evidence', 'open_feed', or 'verified_ledger'). "
                        "This ends your execution turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hypothesis": {
                                "type": "integer",
                                "description": "Chosen hypothesis index for this round.",
                            },
                            "cited_tier": {
                                "type": "string",
                                "enum": ["private_evidence", "open_feed", "verified_ledger"],
                                "description": (
                                    "Epistemic tier backing your conclusion: "
                                    "'private_evidence', 'open_feed', or 'verified_ledger'."
                                ),
                            },
                        },
                        "required": ["hypothesis", "cited_tier"],
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
            return {"error": f"TieredInfoEnvironment does not support tool: {tool_name}"}

        args = dict(arguments or {})

        if tool_name == "post_open_claim":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "post_open_claim is only available during planning",
                }
            result = self.environment.post_open_claim(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                claim = result["claim"]
                text = (
                    f"[OPEN_FEED] {agent_name} posted unverified claim for hypothesis "
                    f"{claim['hypothesis']}: {claim['message']}"
                )
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "endorse_verified_claim":
            if phase != "planning":
                return {
                    "status": "failed",
                    "reason": "endorse_verified_claim is only available during planning",
                }
            result = self.environment.endorse_verified_claim(
                agent_name=agent_name, arguments=args
            )
            if result.get("status") == "success":
                status_tag = (
                    "PROMOTED TO VERIFIED_LEDGER"
                    if result["promoted_to_verified"]
                    else f"endorsers={len(result['endorsers'])}/{result['quorum_size']}"
                )
                note = str(args.get("message") or "").strip()
                suffix = f" ({note})" if note else ""
                text = (
                    f"[VERIFIED_LEDGER] {agent_name} endorsed hypothesis "
                    f"{result['hypothesis']} [{status_tag}]{suffix}"
                )
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        if tool_name == "commit_conclusion":
            if phase != "execution":
                return {
                    "status": "failed",
                    "reason": "commit_conclusion is only available during execution",
                }
            result = self.environment.commit_conclusion(agent_name=agent_name, arguments=args)
            if result.get("status") == "success":
                text = (
                    f"[CONCLUSION] {agent_name} committed hypothesis "
                    f"{result['hypothesis']} citing {result['cited_tier']}."
                )
                self._post_public_message(agent_name, text, phase, iteration)
            return _mark_committed(result)

        return {"error": f"Unknown action type: {tool_name}"}
