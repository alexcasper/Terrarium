"""DCOP environments module - MeetingScheduling, PersonalAssistant, SmartGrid, JiraTicket, and Hospital.

This package depends on an external CoLLAB checkout for instance generation
and scoring. CoLLAB is not a Python package at its root, so we add CoLLAB's
repo root to `sys.path` to make `problem_layer.*` imports available.

When running from a source checkout, the default location is `external/CoLLAB`.
When installed from PyPI, set `TERRARIUM_COLLAB_PATH=/path/to/CoLLAB`.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import os
import sys
from typing import Any

_COLLAB_ENV_VARS = ("TERRARIUM_COLLAB_PATH", "TERRARIUM_COLLAB_ROOT")


def _candidate_collab_roots() -> list[Path]:
    for var in _COLLAB_ENV_VARS:
        raw = os.getenv(var)
        if raw:
            return [Path(raw).expanduser()]

    return [
        # Source checkout default: repo-root/external/CoLLAB
        Path(__file__).resolve().parents[3] / "external" / "CoLLAB",
        # Common invocation pattern: run from a repo root with submodule
        Path.cwd() / "external" / "CoLLAB",
    ]


def _maybe_add_collab_to_syspath() -> None:
    for root in _candidate_collab_roots():
        try:
            root = root.resolve()
        except OSError:
            continue
        if not (root / "problem_layer").exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return


_maybe_add_collab_to_syspath()

__all__ = [
    "MeetingSchedulingEnvironment",
    "PersonalAssistantEnvironment",
    "SmartGridEnvironment",
    "JiraTicketEnvironment",
    "HospitalEnvironment",
    "IsThisSeatTakenEnvironment",
    "PriceWarMarketEnvironment",
    "AuctionHouseEnvironment",
    "CommonPoolEnvironment",
    "PublicGoodsEnvironment",
    "NetworkRoutingEnvironment",
    "SupplyChainEnvironment",
    "SmartGridStorageEnvironment",
    "RideHailingDispatchEnvironment",
    "ReputationEnvironment",
    "SideContractEnvironment",
    "PeerPredictionReportingEnvironment",
    "MultiIssueBarterEnvironment",
    "CoalitionFormationEnvironment",
    "LegislativeBargainingEnvironment",
    "SharedCacheEnvironment",
    "FederatedUpdateEnvironment",
    "DisasterReliefEnvironment",
    "TieredInfoEnvironment",
    "InstanceDifficultyAutotunerEnvironment",
    "AdAuctionGSPEnvironment",
]

_LAZY_ATTRS = {
    "MeetingSchedulingEnvironment": ".meeting_scheduling.meeting_scheduling_env:MeetingSchedulingEnvironment",
    "PersonalAssistantEnvironment": ".personal_assistant.personal_assistant_env:PersonalAssistantEnvironment",
    "SmartGridEnvironment": ".smart_grid.smart_grid_env:SmartGridEnvironment",
    "JiraTicketEnvironment": ".jira_ticket.jira_ticket_env:JiraTicketEnvironment",
    "HospitalEnvironment": ".hospital.hospital_env:HospitalEnvironment",
    "IsThisSeatTakenEnvironment": ".is_this_seat_taken.is_this_seat_taken_env:IsThisSeatTakenEnvironment",
    "PriceWarMarketEnvironment": ".price_war_market.price_war_market_env:PriceWarMarketEnvironment",
    "AuctionHouseEnvironment": ".auction_house.auction_house_env:AuctionHouseEnvironment",
    "CommonPoolEnvironment": ".common_pool.common_pool_env:CommonPoolEnvironment",
    "PublicGoodsEnvironment": ".public_goods.public_goods_env:PublicGoodsEnvironment",
    "NetworkRoutingEnvironment": ".network_routing.network_routing_env:NetworkRoutingEnvironment",
    "SupplyChainEnvironment": ".supply_chain.supply_chain_env:SupplyChainEnvironment",
    "SmartGridStorageEnvironment": ".smart_grid_storage.smart_grid_storage_env:SmartGridStorageEnvironment",
    "RideHailingDispatchEnvironment": ".ride_hailing.ride_hailing_env:RideHailingDispatchEnvironment",
    "ReputationEnvironment": ".reputation.reputation_env:ReputationEnvironment",
    "SideContractEnvironment": ".side_contract.side_contract_env:SideContractEnvironment",
    "PeerPredictionReportingEnvironment": ".peer_prediction.peer_prediction_env:PeerPredictionReportingEnvironment",
    "MultiIssueBarterEnvironment": ".multi_issue_barter.multi_issue_barter_env:MultiIssueBarterEnvironment",
    "CoalitionFormationEnvironment": ".coalition_formation.coalition_formation_env:CoalitionFormationEnvironment",
    "LegislativeBargainingEnvironment": ".legislative_bargaining.legislative_bargaining_env:LegislativeBargainingEnvironment",
    "SharedCacheEnvironment": ".shared_cache.shared_cache_env:SharedCacheEnvironment",
    "FederatedUpdateEnvironment": ".federated_update.federated_update_env:FederatedUpdateEnvironment",
    "DisasterReliefEnvironment": ".disaster_relief.disaster_relief_env:DisasterReliefEnvironment",
    "TieredInfoEnvironment": ".tiered_info.tiered_info_env:TieredInfoEnvironment",
    "InstanceDifficultyAutotunerEnvironment": ".difficulty_autotuner.difficulty_autotuner_env:InstanceDifficultyAutotunerEnvironment",
    "AdAuctionGSPEnvironment": ".ad_auction_gsp.ad_auction_gsp_env:AdAuctionGSPEnvironment",
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, _, attr = target.partition(":")
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
