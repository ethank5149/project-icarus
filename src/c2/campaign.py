from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np

from .battle_manager import BattleManager, BattleManagerConfig, BattleResult, ThreatTrack, Battery


@dataclass
class CampaignThreat:
    """A raid element: a target scenario launched at a defended aim point."""
    target: Any
    aim_point: np.ndarray
    launch_site: np.ndarray = field(default_factory=lambda: np.zeros(3))
    priority: float = 1.0
    label: str = ""


@dataclass
class CampaignResult:
    battle: BattleResult
    engagements: List[Any] = field(default_factory=list)
    config: Any = None

    def summary(self) -> Dict[str, Any]:
        return self.battle.summary()


def _default_assess_factory(engagements_by_threat: Dict[int, List[Any]],
                            kill_radius: float):
    """Build an ``assess(threat_track, battery_index) -> miss_m`` callback.

    Batteries in this lightweight campaign are fungible: the miss distance for a
    given threat comes from the precomputed engagement results indexed by threat
    id. ``battery_index`` selects which *shot* (engagement replicate) to read so
    shoot-look-shoot can consume additional interceptors.
    """
    def assess(threat_track: ThreatTrack, battery_index: int) -> float:
        tid = threat_track.threat_id
        reps = engagements_by_threat.get(tid, [])
        if not reps:
            return float("inf")
        idx = min(threat_track.shots_fired - 1, len(reps) - 1)
        idx = max(idx, 0)
        eng = reps[idx]
        miss = getattr(eng, "miss_distance", np.inf)
        if miss is None or not np.isfinite(miss):
            return float("inf")
        return float(miss)
    return assess


def run_campaign(
    threats: List[CampaignThreat],
    batteries: List[Battery],
    scenario_builder: Callable[[CampaignThreat, Battery], Any],
    cfg: Optional[BattleManagerConfig] = None,
    n_trials: int = 20,
    perturbations: Optional[Dict[str, float]] = None,
    parallel: bool = False,
    n_jobs: int = -1,
    location_fn: Optional[Callable[[Any], np.ndarray]] = None,
) -> CampaignResult:
    """Run a saturation-raid campaign (Phase 5C).

    Extends the single-engagement philosophy of ``run_sweep`` to a *system* of
    threats vs a battery of interceptors. For each (threat, battery) pair we build
    an ``EngagementScenario`` via ``scenario_builder`` and run ``run_engagement``
    to obtain per-pair miss-distance statistics, then hand those statistics to the
    ``BattleManager`` which applies the C2 doctrine (allocation + salvo / shoot-
    look-shoot) and reports system-level leakage / battery-utilization metrics.

    Parameters
    ----------
    threats : list[CampaignThreat]
        Inbound raid elements (target scenarios + aim points).
    batteries : list[Battery]
        Defender batteries (each a finite magazine of one interceptor type).
    scenario_builder : callable
        ``scenario_builder(threat, battery) -> EngagementScenario``.
    cfg : BattleManagerConfig, optional
        C2 doctrine. Defaults to greedy + shoot-look-shoot, salvo_size=1.
    n_trials : int
        Monte Carlo trials per pairwise engagement.
    perturbations, parallel, n_jobs
        Forwarded to the engagement runner.

    Returns
    -------
    CampaignResult
    """
    cfg = cfg or BattleManagerConfig()

    # --- 1. Precompute pairwise engagement miss distances -----------------
    engagements_by_threat: Dict[int, List[Any]] = {i: [] for i in range(len(threats))}
    all_engagements: List[Any] = []
    from ..sim.api import run_engagement  # local import keeps layer additive
    from ..guidance.law import GuidanceLaw

    for ti, th in enumerate(threats):
        for bat in batteries:
            scenario = scenario_builder(th, bat)
            guidance = bat.guidance_config
            if not isinstance(guidance, GuidanceLaw):
                guidance = GuidanceLaw(guidance)
            eng = run_engagement(
                interceptor=bat.interceptor_config,
                guidance=guidance,
                target=th.target,
                scenario=scenario,
                n_trials=n_trials,
                perturbations=perturbations,
            )
            engagements_by_threat[ti].append(eng)
            all_engagements.append(eng)

    # --- 2. Build C2 tracks + batteries ----------------------------------
    tracks: List[ThreatTrack] = []
    for ti, th in enumerate(threats):
        tracks.append(ThreatTrack(
            threat_id=ti,
            target=th.target,
            launch_site=np.asarray(th.launch_site, dtype=float),
            aim_point=np.asarray(th.aim_point, dtype=float),
            priority=float(th.priority),
        ))

    bm = BattleManager(tracks, batteries, cfg=cfg, location_fn=location_fn)
    assess = _default_assess_factory(engagements_by_threat, 0.5)
    battle = bm.run(assess)

    return CampaignResult(battle=battle, engagements=all_engagements, config=cfg)
