from .battle_manager import (
    BattleManager,
    BattleManagerConfig,
    BattleResult,
    ThreatTrack,
    Battery,
    greedy_allocate,
    hungarian_allocate,
)
from .campaign import run_campaign, CampaignThreat, CampaignResult
from .discrete_event import run_discrete_event, C2Scenario
from .layers import (
    Tier,
    Layer,
    DefenseArchitecture,
    SpaceSensor,
    DistributedC2Config,
    build_architecture_from_locations,
    distributed_handoff,
    run_layered_campaign,
    architecture_summary,
    national_metrics,
)

from .persistence import save_campaign_hdf5, load_campaign_hdf5
from .visualization import build_national_scene, coverage_summary_table
from .dashboard import NationalDashboard, make_dashboard

__all__ = [
    "BattleManager",
    "BattleManagerConfig",
    "BattleResult",
    "ThreatTrack",
    "Battery",
    "greedy_allocate",
    "hungarian_allocate",
    "run_campaign",
    "CampaignThreat",
    "CampaignResult",
    "run_discrete_event",
    "C2Scenario",
    "Tier",
    "Layer",
    "DefenseArchitecture",
    "SpaceSensor",
    "DistributedC2Config",
    "build_architecture_from_locations",
    "distributed_handoff",
    "run_layered_campaign",
    "architecture_summary",
    "national_metrics",

    "save_campaign_hdf5",
    "load_campaign_hdf5",
    "build_national_scene",
    "coverage_summary_table",
    "NationalDashboard",
    "make_dashboard",
]
