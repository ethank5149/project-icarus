"""Phase 6 — National "Golden Dome" layered defense architecture.

Wraps the heterogeneous interceptor mixes defined in ``reference/locations.yml``
into a tiered, layered architecture and adds the distributed-C2 machinery (space
sensor warning + ground-handoff with latency / bandwidth constraints) on top of
the Phase 5 ``BattleManager``.

Design notes
------------
* A :class:`Tier` is a single interceptor type deployed at one or more bases
  (e.g. GMD boost/exo, Arrow-3 upper, THAAD mid, Iron Dome lower). Each tier
  owns a magazine split across its bases.
* A :class:`Layer` groups tiers by engagement regime (boost / upper / mid /
  lower) and exposes the unified ``batteries`` list consumed by the existing
  ``BattleManager`` / ``run_campaign`` API. This keeps the layer strictly
  additive — no changes to the engagement-scale engine are required.
* Distributed C2 (:func:`distributed_handoff`) models a three-echelon chain:
  space-based early-warning (SBIRS-like) -> regional ground C2 -> battery.
  Each handoff carries a latency and a finite bandwidth (tracks/min) so that
  under saturation a fraction of tracks are dropped before they reach a battery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np

from .battle_manager import Battery, BattleManager, BattleManagerConfig, ThreatTrack


# ---------------------------------------------------------------------------
# Tiers & Layers
# ---------------------------------------------------------------------------

_TIER_KINDS = ("boost", "upper", "mid", "lower")


@dataclass
class Tier:
    """A single interceptor type deployed across one or more bases."""
    kind: str                       # "boost" | "upper" | "mid" | "lower"
    interceptor_name: str           # preset key in build_interceptor_config
    bases: List[np.ndarray] = field(default_factory=list)
    magazine_per_base: int = 10
    salvo_size: int = 1
    # C2 handoff cost to reach this tier (s); higher tiers (space) add latency.
    c2_latency_s: float = 2.0
    label: str = ""

    def __post_init__(self):
        if self.kind not in _TIER_KINDS:
            raise ValueError(f"Tier kind must be one of {_TIER_KINDS}, got {self.kind!r}")
        if not self.bases:
            raise ValueError("A Tier needs at least one base location")

    @property
    def total_magazine(self) -> int:
        return self.magazine_per_base * len(self.bases)

    def to_batteries(self) -> List[Battery]:
        """Expand this tier into one Battery per base (same interceptor type)."""
        from ..scenarios.presets import build_interceptor_config
        cfg, gcfg = build_interceptor_config(self.interceptor_name)
        bats: List[Battery] = []
        for i, loc in enumerate(self.bases):
            bats.append(Battery(
                name=f"{self.label or self.interceptor_name}-b{i}",
                interceptor_config=cfg,
                guidance_config=gcfg,
                location=np.asarray(loc, dtype=float),
                magazine=self.magazine_per_base,
                salvo_size=self.salvo_size,
            ))
        return bats


@dataclass
class Layer:
    """A regime grouping of tiers (boost / upper / mid / lower)."""
    name: str
    tiers: List[Tier] = field(default_factory=list)

    def add(self, tier: Tier) -> "Layer":
        self.tiers.append(tier)
        return self

    @property
    def batteries(self) -> List[Battery]:
        out: List[Battery] = []
        for t in self.tiers:
            out.extend(t.to_batteries())
        return out

    @property
    def total_magazine(self) -> int:
        return sum(t.total_magazine for t in self.tiers)


@dataclass
class DefenseArchitecture:
    """The full national layered architecture (one or more layers)."""
    name: str = "golden_dome"
    layers: List[Layer] = field(default_factory=list)

    def add(self, layer: Layer) -> "DefenseArchitecture":
        self.layers.append(layer)
        return self

    @property
    def batteries(self) -> List[Battery]:
        out: List[Battery] = []
        for ly in self.layers:
            out.extend(ly.batteries)
        return out

    @property
    def total_magazine(self) -> int:
        return sum(ly.total_magazine for ly in self.layers)


# ---------------------------------------------------------------------------
# Layered builders from reference/locations.yml
# ---------------------------------------------------------------------------

def build_architecture_from_locations(
    defended_names: Optional[List[str]] = None,
    interceptor_kind_map: Optional[Dict[str, str]] = None,
    magazine_per_base: int = 10,
    salvo_size: int = 1,
) -> DefenseArchitecture:
    """Construct a tiered architecture from the locations database.

    Parameters
    ----------
    defended_names
        Optional subset of ``defended-target`` site names to protect. (Currently
        used for reporting; aim points are supplied by the campaign caller.)
    interceptor_kind_map
        Mapping ``interceptor-launch-site name -> tier kind``. Sites not listed
        default to ``upper``. This lets the operator classify GBI fields as
        ``boost``/``upper`` and Aegis/THAAD sites as ``mid``/``lower``.
    magazine_per_base, salvo_size
        Per-base magazine and salvo sizing for every tier.
    """
    from reference.locations import locations_by_designation, coordinates_to_ecef

    groups = locations_by_designation()
    sites = groups.get("interceptor-launch-site", [])
    interceptor_kind_map = interceptor_kind_map or {}

    by_kind: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in _TIER_KINDS}
    for rec in sites:
        kind = interceptor_kind_map.get(rec["name"], "upper")
        by_kind[kind][rec["name"]] = coordinates_to_ecef(rec)

    arch = DefenseArchitecture(name="golden_dome")
    # Order matters: boost -> upper -> mid -> lower (outside-in engagement).
    layer_for: Dict[str, Layer] = {
        "boost": Layer("boost_layer"),
        "upper": Layer("upper_layer"),
        "mid": Layer("mid_layer"),
        "lower": Layer("mid_layer"),  # mid + lower share the regional layer
    }
    for kind in ("boost", "upper", "mid", "lower"):
        locs = by_kind[kind]
        if not locs:
            continue
        tier = Tier(
            kind=kind,
            interceptor_name=_interceptor_for_kind(kind),
            bases=list(locs.values()),
            magazine_per_base=magazine_per_base,
            salvo_size=salvo_size,
            c2_latency_s={"boost": 4.0, "upper": 3.0, "mid": 2.0, "lower": 1.0}[kind],
            label=kind,
        )
        layer_for[kind].add(tier)
    for ly in (layer_for["boost"], layer_for["upper"], layer_for["mid"]):
        if ly.tiers:
            arch.add(ly)
    return arch


def _interceptor_for_kind(kind: str) -> str:
    # OSINT-approximate default mapping of tier kind -> preset interceptor.
    return {
        "boost": "gmd",
        "upper": "arrow3",
        "mid": "arrow3",
        "lower": "tamir",
    }[kind]


# ---------------------------------------------------------------------------
# Distributed C2: space sensor warning + ground handoff
# ---------------------------------------------------------------------------

@dataclass
class SpaceSensor:
    """An SBIRS-like early-warning satellite with a global track capacity."""
    name: str = "SBIRS"
    # Mean endo-to-exo handoff latency (s): detection -> ground C2 report.
    warning_latency_s: float = 10.0
    # Track-handling bandwidth (tracks reported per second) under saturation.
    bandwidth_tracks_per_s: float = 5.0
    # Probability a given threat is detected by the space layer at all.
    p_detect: float = 0.95


@dataclass
class DistributedC2Config:
    """Latency / bandwidth model for the distributed C2 chain."""
    space: SpaceSensor = field(default_factory=SpaceSensor)
    ground_latency_s: float = 2.0     # ground C2 -> battery release
    # Fraction of tracks dropped when bandwidth is exceeded (saturation loss).
    saturation_drop_enabled: bool = True
    # Inbound raid arrival spread (s): threats appear uniformly across
    # ``[0, raid_arrival_window_s]`` so the space layer's finite bandwidth can
    # actually cue a realistic fraction (an instantaneous salvo saturates fully).
    raid_arrival_window_s: float = 0.0


def distributed_handoff(
    threats: List[ThreatTrack],
    cfg: DistributedC2Config,
    t_contact: Optional[Callable[[ThreatTrack], float]] = None,
) -> Tuple[List[ThreatTrack], Dict[str, Any]]:
    """Pass a raid through the distributed C2 chain.

    Models two handoffs:
      1. Space early-warning detects each threat (``p_detect``) and reports it
         after ``warning_latency_s``. The space layer's finite
         ``bandwidth_tracks_per_s`` means that under a saturation raid it can only
         hand off ``bandwidth * dt`` tracks per second; the rest are *dropped*
         (not yet cued to ground), realizing a real saturation effect.
      2. Surviving tracks are handed to ground C2 (``ground_latency_s``) before a
         battery can be tasked.

    ``t_contact(threat) -> t`` gives the time each threat appears to the space
    layer (default 0). The returned threat list carries ``_cue_latency_s`` so the
    BattleManager can delay engagement correspondingly.

    Returns
    -------
    (tasked_threats, diagnostics) where ``tasked_threats`` is the subset that
    reaches a battery and ``diagnostics`` records dropped-count / latency stats.
    """
    if t_contact is None:
        t_contact = lambda t: 0.0

    # Sort by contact time (FIFO through the space pipeline).
    ordered = sorted(threats, key=lambda t: t_contact(t))

    tasked: List[ThreatTrack] = []
    dropped = 0
    total_latency: List[float] = []

    last_report_t: Optional[float] = None
    # Bandwidth budget accrues from the moment the first track is detected and
    # refills at ``bandwidth`` tracks/sec. An instantaneous salvo (all tc==0)
    # therefore cues only the tracks covered by the accrued budget, realising a
    # true saturation drop; a spread raid accrues budget across the arrival
    # window and cues a realistic fraction. The pipeline is assumed to have
    # been monitoring for ~1 s before the raid, so ``budget`` is seeded with
    # one slot (bandwidth>=1) — the very first detected track is always cued.
    budget = cfg.space.bandwidth_tracks_per_s * 1.0
    for t in ordered:
        # Space detection gate.
        if np.random.random() > cfg.space.p_detect:
            dropped += 1
            continue
        tc = t_contact(t)
        if cfg.saturation_drop_enabled:
            if last_report_t is None:
                last_report_t = tc
            dt = max(tc - last_report_t, 0.0)
            budget += cfg.space.bandwidth_tracks_per_s * dt
            if budget < 1.0:
                dropped += 1
                continue
            budget -= 1.0
            last_report_t = tc
        cue = cfg.space.warning_latency_s + cfg.ground_latency_s
        t._cue_latency_s = cue
        total_latency.append(cue)
        tasked.append(t)

    diag = {
        "n_inbound": len(threats),
        "n_tasked": len(tasked),
        "n_dropped": dropped,
        "mean_cue_latency_s": float(np.mean(total_latency)) if total_latency else 0.0,
        "space_bandwidth_tracks_per_s": cfg.space.bandwidth_tracks_per_s,
    }
    return tasked, diag


# ---------------------------------------------------------------------------
# Tiered engagement: run a layered architecture against a raid.
# ---------------------------------------------------------------------------

def run_layered_campaign(
    threats: List[ThreatTrack],
    architecture: DefenseArchitecture,
    assess: Callable[[ThreatTrack, int], float],
    cfg: Optional[BattleManagerConfig] = None,
    c2: Optional[DistributedC2Config] = None,
    location_fn: Optional[Callable[[Any], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Engage a raid with the full layered architecture + distributed C2.

    This is the Phase-6 counterpart of :func:`run_campaign`: it first passes the
    inbound threats through :func:`distributed_handoff` (space warning + ground
    handoff, with saturation drop), then runs the existing ``BattleManager``
    doctrine across *all* batteries in the architecture so the allocator can
    commit boost/upper/mid/lower interceptors against the cued subset.

    Returns a dict with ``battle`` (BattleResult over the *declared* threats, so
    leakage includes dropped tracks), ``c2_diag`` (handoff diagnostics), and
    ``tasked_ids`` (which threats reached a battery).
    """
    cfg = cfg or BattleManagerConfig()
    c2 = c2 or DistributedC2Config()

    declared = list(threats)
    if c2.raid_arrival_window_s > 0:
        # Spread the inbound raid across the arrival window so the space layer's
        # finite bandwidth cues a realistic (sub-saturation) fraction.
        span = float(c2.raid_arrival_window_s)

        def _t_contact(t: ThreatTrack) -> float:
            return span * (float(t.threat_id) / max(len(declared) - 1, 1))
    else:
        _t_contact = None
    tasked, c2_diag = distributed_handoff(declared, c2, t_contact=_t_contact)

    batteries = architecture.batteries
    if not batteries:
        raise RuntimeError("Architecture has no deployed batteries")

    bm = BattleManager(tasked, batteries, cfg=cfg, location_fn=location_fn)
    battle = bm.run(assess)

    # BattleResult is built over ``tasked`` (declared == tasked here inside BM),
    # but we want leakage relative to the *original* declared raid.
    from .battle_manager import BattleResult
    final_battle = BattleResult(threats=declared, batteries=batteries, shots=battle.shots)

    tasked_ids = [t.threat_id for t in tasked]
    return {
        "battle": final_battle,
        "c2_diag": c2_diag,
        "tasked_ids": tasked_ids,
        "architecture": architecture,
    }
