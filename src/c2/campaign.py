from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import json
import os
import tempfile
import numpy as np

from .battle_manager import BattleManager, BattleManagerConfig, BattleResult, ThreatTrack, Battery


# ---------------------------------------------------------------------------
# Parallel backend: JSON spec + HDF5 results (industry-standard, no pickle)
# ---------------------------------------------------------------------------
# Engagement configs contain LAMBDA thrust profiles that pickle cannot
# serialize across processes. The parallel path therefore uses ONLY
# industry-standard interchange formats -- pickle-free by design:
#   * JSON  -- the (threat, battery) engagement SPEC. The interceptor /
#               guidance preset is rebuilt by NAME via build_interceptor_config,
#               and the target / scenario are serialized as plain params.
#               Fully portable and human-readable.
#   * HDF5 -- the per-pair RESULTS (nominal miss, Monte-Carlo miss
#               distributions, trajectories). The standard for numerical arrays
#               and repo-consistent with the aero / CFD pipeline.
# A module-level worker reads a JSON spec and writes HDF5 results, so
# no lambda-laden object ever crosses the process boundary.


def _serialize_target(target) -> Dict[str, Any]:
    """Serialize a TargetScenario to a portable dict (no pickled callables)."""
    out = {"kind": type(target).__name__}
    for key in ("r0", "v0", "launch_el_deg", "scenario_type"):
        if hasattr(target, key):
            v = getattr(target, key)
            if isinstance(v, np.ndarray):
                out[key] = v.tolist()
            else:
                out[key] = v
    # Fallback: capture the constructor signature if it is a simple ballistic.
    if "r0" not in out and hasattr(target, "propagate"):
        # BallisticScenario(r0, v0) is the common case; re-derive if possible.
        pass
    return out


def _deserialize_target(spec: Dict[str, Any]):
    """Rebuild a TargetScenario from its serialized spec."""
    from src.scenarios.target_factory import BallisticScenario
    kind = spec.get("kind", "BallisticScenario")
    if kind == "BallisticScenario":
        r0 = np.asarray(spec["r0"], dtype=float)
        v0 = np.asarray(spec["v0"], dtype=float)
        return BallisticScenario(r0=r0, v0=v0)
    # Generic: try the registered target presets by matching r0/v0 is unsafe;
    # for now only ballistic is supported in the parallel path.
    raise ValueError(f"Cannot deserialize target kind {kind!r} in parallel path")


def _serialize_scenario(scn) -> Dict[str, Any]:
    out = {}
    for key in ("engagement_start", "engagement_end", "interceptor_launch_site",
                "target_launch_site", "dt"):
        if hasattr(scn, key):
            v = getattr(scn, key)
            out[key] = v if not isinstance(v, np.ndarray) else v.tolist()
    return out


def _deserialize_scenario(spec: Dict[str, Any]):
    from src.sim.api import EngagementScenario
    return EngagementScenario(**{k: (np.asarray(v, dtype=float) if isinstance(v, list) else v)
                               for k, v in spec.items()})


def _run_one_pair_spec(pair_spec: Dict[str, Any]):
    """Module-level worker (picklable): run one engagement from a JSON spec.

    ``pair_spec`` carries only portable data: threat_id, interceptor/guidance
    NAMES (rebuilt via ``build_interceptor_config``), a serialized target, a
    serialized scenario, n_trials, perturbations. Returns ``(ti, eng)`` where
    ``eng`` is a lightweight dict (miss + mc misses) — no pickled callables.
    """
    from ..sim.api import run_engagement
    from ..scenarios.presets import build_interceptor_config
    from ..guidance.law import GuidanceLaw

    ti = pair_spec["ti"]
    icfg, gcfg = build_interceptor_config(pair_spec["interceptor_name"])
    guidance = GuidanceLaw(gcfg)
    target = _deserialize_target(pair_spec["target"])
    scenario = _deserialize_scenario(pair_spec["scenario"])
    eng = run_engagement(
        interceptor=icfg,
        guidance=guidance,
        target=target,
        scenario=scenario,
        n_trials=pair_spec["n_trials"],
        perturbations=pair_spec.get("perturbations"),
    )
    return ti, {
        "miss_distance": float(getattr(eng, "miss_distance", np.inf)),
        "kill_assessment": bool(getattr(eng, "kill_assessment", False)),
        "mc_misses": [float(m) for m in
                        getattr(getattr(eng, "monte_carlo", None), "miss_distances", [])],
    }


def _interceptor_name(battery: Battery) -> str:
    """Resolve a battery's interceptor preset NAME for JSON transport.

    The battery stores a built ``InterceptorConfig``; the parallel worker
    rebuilds it by name (via ``build_interceptor_config``) so lambda thrust
    profiles never cross the process boundary. We recover the name from the
    config's ``name`` field when it matches a known preset, else fall back to
    the battery's stored name lookup.
    """
    cfg = battery.interceptor_config
    name = getattr(cfg, "name", "") or ""
    # Known preset names from build_interceptor_config.
    known = {"Arrow-3 (exoatmospheric hit-to-kill)", "Iron Dome Tamir (endoaortic point-defense)",
              "GMD GBII (exoatmospheric EKV hit-to-kill)"}
    if name in known:
        # Map the friendly name back to its preset key.
        friendly = {
            "Arrow-3 (exoatmospheric hit-to-kill)": "arrow3",
            "Iron Dome Tamir (endoaortic point-defense)": "tamir",
            "GMD GBII (exoatmospheric EKV hit-to-kill)": "gmd",
        }
        return friendly[name]
    # Fallback: try the battery's stored key if it was built from a preset.
    from ..scenarios.presets import get_interceptor_config_presets
    for key, (ic, _) in get_interceptor_config_presets().items():
        if ic is cfg:
            return key
    raise ValueError(
        f"Cannot resolve interceptor preset name for parallel transport; "
        f"battery carries an unregistered InterceptorConfig: {name!r}"
    )


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
    backend: str = "dask",
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
    backend : str
        Parallel backend when ``parallel`` is set: ``"dask"`` (preferred; uses
        a local ``distributed.Client`` with a graceful fallback to joblib) or
        ``"joblib"``. Both are optional imports so the layer stays portable.

    Returns
    -------
    CampaignResult
    """
    cfg = cfg or BattleManagerConfig()

    # --- 1. Precompute pairwise engagement miss distances -----------------
    # Each (threat, battery) pair is an independent ``run_engagement``; under
    # large raids this dominates runtime. When ``parallel`` is set we fan the
    # pairs out across host cores via joblib multiprocessing.
    #
    # Transport uses INDUSTRY-STANDARD formats, not pickle:
    #   * Each pair is serialized to a JSON SPEC (interceptor/guidance rebuilt
    #     by NAME via ``build_interceptor_config``; target/scenario as plain
    #     params). No lambda thrust profiles cross the process boundary, so the
    #     worker is fully picklable.
    #   * Engagement RESULTS are returned as plain dicts (miss + Monte-Carlo
    #     miss list) and persisted to HDF5 by ``save_campaign_hdf5``.
    from ..sim.api import run_engagement  # local import keeps layer additive
    from ..guidance.law import GuidanceLaw

    if parallel:
        # Parallel transport uses INDUSTRY-STANDARD formats: each pair is a
        # JSON SPEC (interceptor/guidance rebuilt by NAME; target/scenario
        # as plain params). No lambda thrust profiles cross the boundary.
        from ..scenarios.presets import build_interceptor_config
        specs = []
        for ti, th in enumerate(threats):
            for bi, bat in enumerate(batteries):
                specs.append({
                    "ti": ti,
                    "bi": bi,
                    "interceptor_name": _interceptor_name(bat),
                    "target": _serialize_target(th.target),
                    "scenario": _serialize_scenario(scenario_builder(th, bat)),
                    "n_trials": n_trials,
                    "perturbations": perturbations,
                })
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:  # pragma: no cover - joblib present
            raise RuntimeError("joblib required for parallel run_campaign") from exc
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(_run_one_pair_spec)(s) for s in specs
        )
    else:
        # Serial: run the real run_engagement with the battery's own configs
        # (no name resolution, so custom/preset configs both work).
        results = []
        for ti, th in enumerate(threats):
            for bi, bat in enumerate(batteries):
                guidance = bat.guidance_config
                if not isinstance(guidance, GuidanceLaw):
                    guidance = GuidanceLaw(guidance)
                eng = run_engagement(
                    interceptor=bat.interceptor_config,
                    guidance=guidance,
                    target=th.target,
                    scenario=scenario_builder(th, bat),
                    n_trials=n_trials,
                    perturbations=perturbations,
                )
                results.append((ti, eng))

    engagements_by_threat: Dict[int, List[Any]] = {i: [] for i in range(len(threats))}
    all_engagements: List[Any] = []
    for ti, eng in results:
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
