from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np


@dataclass
class ThreatTrack:
    """A declared inbound threat (raid element) to be engaged by the C2 layer."""
    threat_id: int
    target: Any  # TargetScenario-like: propagate(t) -> (6,) [r, v]
    launch_site: np.ndarray = field(default_factory=lambda: np.zeros(3))
    aim_point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    priority: float = 1.0  # higher = more urgent (e.g. populated target)
    # Bookkeeping filled by the BattleManager.
    assigned_interceptors: List[int] = field(default_factory=list)
    shots_fired: int = 0
    defeated: bool = False


@dataclass
class Battery:
    """A defended-point interceptor battery with a finite magazine."""
    name: str
    interceptor_config: Any
    guidance_config: Any
    location: np.ndarray = field(default_factory=lambda: np.zeros(3))
    magazine: int = 10
    # Salvo doctrine: how many interceptors to commit per threat on a "look".
    salvo_size: int = 1

    def can_fire(self, n: int = 1) -> bool:
        return self.magazine >= n

    def fire(self, n: int = 1) -> bool:
        if not self.can_fire(n):
            return False
        self.magazine -= n
        return True


def _range_km(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))) / 1000.0


def greedy_allocate(
    threats: List[ThreatTrack],
    batteries: List[Battery],
    location_fn: Callable[[Any], np.ndarray],
    salvo_size: int = 1,
) -> List[Tuple[int, int, int]]:
    """Greedy threat->battery->count assignment (doctrine-agnostic).

    Returns a list of (threat_index, battery_index, count) commitments sorted by
    descending threat priority. Each battery contributes at most ``salvo_size``
    per pass, up to remaining magazine.
    """
    salvo_size = max(salvo_size, 1)
    commitments: List[Tuple[int, int, int]] = []
    ordered = sorted(range(len(threats)), key=lambda i: -threats[i].priority)
    for ti in ordered:
        if threats[ti].defeated:
            continue
        remaining_want = max(salvo_size - threats[ti].shots_fired, 1)
        for bi, bat in enumerate(batteries):
            if remaining_want <= 0:
                break
            if not bat.can_fire():
                continue
            give = min(bat.salvo_size, bat.magazine, remaining_want)
            if give <= 0:
                continue
            bat.fire(give)
            threats[ti].assigned_interceptors.extend([bi] * give)
            threats[ti].shots_fired += give
            commitments.append((ti, bi, give))
            remaining_want -= give
    return commitments


def hungarian_allocate(
    threats: List[ThreatTrack],
    batteries: List[Battery],
    location_fn: Callable[[Any], np.ndarray],
    max_per_battery: int = 1,
) -> List[Tuple[int, int, int]]:
    """Optimal one-shot assignment minimising total engagement range.

    Uses scipy's linear_sum_assignment over a padded cost matrix. At most
    ``max_per_battery`` interceptors are committed initially (salvo follow-up is
    handled by the shoot-look-shoot loop in ``BattleManager``).
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - scipy always present
        raise RuntimeError("scipy is required for Hungarian allocation") from exc

    n = len(threats)
    m = len(batteries) * max_per_battery
    cost = np.full((max(n, 1), max(m, 1)), np.inf)
    # Build battery slots.
    slots: List[Tuple[int, np.ndarray]] = []
    for bi, bat in enumerate(batteries):
        for _ in range(max_per_battery):
            slots.append((bi, location_fn(bat.location)))
    for ti in range(n):
        tloc = location_fn(threats[ti].aim_point)
        for sj, (bi, bloc) in enumerate(slots):
            cost[ti, sj] = _range_km(tloc, bloc)
    commitments: List[Tuple[int, int, int]] = []
    if n == 0 or m == 0:
        return commitments
    row, col = linear_sum_assignment(cost)
    for ti, sj in zip(row, col):
        if ti >= n or sj >= len(slots):
            continue
        if not np.isfinite(cost[ti, sj]):
            continue
        bi, _ = slots[sj]
        bat = batteries[bi]
        if not bat.can_fire():
            continue
        bat.fire(1)
        threats[ti].assigned_interceptors.append(bi)
        threats[ti].shots_fired += 1
        commitments.append((ti, bi, 1))
    return commitments


@dataclass
class BattleManagerConfig:
    doctrine: str = "shoot_look_shoot"  # "shoot_look_shoot" | "salvo"
    allocator: str = "greedy"  # "greedy" | "hungarian"
    salvo_size: int = 1
    max_rounds: int = 4
    kill_assessment_fn: Callable[[Any, float], bool] = field(default_factory=lambda: (lambda m, k: bool(m <= k)))


class BattleManager:
    """C2 layer: assess a raid, allocate interceptors, assess leakage.

    The manager is *orchestration only* — it does not integrate trajectories.
    It receives threat tracks and batteries, runs the allocation doctrine, and
    applies a caller-supplied ``assess`` callback that returns per-shot miss
    distance (metres) and the battery kill radius so the manager can mark
    threats defeated and count leakage.
    """

    def __init__(self, threats: List[ThreatTrack], batteries: List[Battery],
                 cfg: Optional[BattleManagerConfig] = None,
                 location_fn: Optional[Callable[[Any], np.ndarray]] = None):
        self.threats = threats
        self.batteries = batteries
        self.cfg = cfg or BattleManagerConfig()
        self.cfg.salvo_size = max(self.cfg.salvo_size, 1)
        # Default location fn: if arg is an ndarray use directly, else .location.
        self._loc = location_fn or (lambda x: np.asarray(x, dtype=float))

    # --- assessment hook --------------------------------------------------
    def _location(self, obj: Any) -> np.ndarray:
        return self._loc(obj)

    def allocate(self) -> List[Tuple[int, int, int]]:
        if self.cfg.allocator == "hungarian":
            return hungarian_allocate(self.threats, self.batteries, self._location,
                                     max_per_battery=self.cfg.salvo_size)
        return greedy_allocate(self.threats, self.batteries, self._location,
                               salvo_size=self.cfg.salvo_size)

    def run(self, assess: Callable[[ThreatTrack, int], float]) -> "BattleResult":
        """Execute the engagement doctrine.

        ``assess(threat, battery_index)`` returns the miss distance (m) for the
        next shot committed against ``threat`` by ``battery_index``. The manager
        fires according to doctrine, marks threats defeated when a shot lands
        within the applicable kill radius, and stops early once all threats are
        resolved or rounds are exhausted.
        """
        # Snapshot capacity for utilization reporting.
        for bat in self.batteries:
            bat._capacity = bat.magazine
        results: List[Dict[str, Any]] = []
        for _round in range(self.cfg.max_rounds):
            active = [t for t in self.threats if not t.defeated]
            if not active:
                break
            # Reset per-round salvo expectation.
            for t in self.threats:
                if not hasattr(t, "_round_fired"):
                    t._round_fired = 0
            commitments = self.allocate()
            if not commitments:
                break
            for ti, bi, count in commitments:
                t = self.threats[ti]
                for _ in range(count):
                    miss = assess(t, bi)
                    bat = self.batteries[bi]
                    kill_radius = getattr(bat.interceptor_config, "kill_radius", 0.5)
                    results.append({
                        "threat_id": t.threat_id,
                        "battery": bat.name,
                        "miss_distance_m": float(miss),
                        "kill_radius_m": float(kill_radius),
                        "kill": bool(miss <= kill_radius),
                    })
                    if miss <= kill_radius:
                        t.defeated = True
                        break
            # Doctrine gate.
            if self.cfg.doctrine == "shoot_look_shoot":
                # re-evaluate after this round; continue if threats remain.
                continue
            else:
                # salvo: one round only.
                break
        return BattleResult.from_shots(self.threats, self.batteries, results)


@dataclass
class BattleResult:
    threats: List[ThreatTrack] = field(default_factory=list)
    batteries: List[Battery] = field(default_factory=list)
    shots: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_shots(cls, threats, batteries, shots) -> "BattleResult":
        return cls(threats=threats, batteries=batteries, shots=list(shots))

    @property
    def n_threats(self) -> int:
        return len(self.threats)

    @property
    def n_defeated(self) -> int:
        return sum(1 for t in self.threats if t.defeated)

    @property
    def n_leakage(self) -> int:
        return sum(1 for t in self.threats if not t.defeated)

    @property
    def leakage_fraction(self) -> float:
        if not self.threats:
            return 0.0
        return self.n_leakage / len(self.threats)

    @property
    def shots_fired(self) -> int:
        return len(self.shots)

    @property
    def kill_probability(self) -> float:
        if not self.threats:
            return 0.0
        return self.n_defeated / len(self.threats)

    @property
    def battery_utilization(self) -> Dict[str, float]:
        out = {}
        for bat in self.batteries:
            cap = getattr(bat, "_capacity", None)
            if cap is None:
                out[bat.name] = float(bat.magazine)
            else:
                used = cap - bat.magazine
                out[bat.name] = used / cap if cap else 0.0
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "n_threats": self.n_threats,
            "n_defeated": self.n_defeated,
            "n_leakage": self.n_leakage,
            "leakage_fraction": self.leakage_fraction,
            "shots_fired": self.shots_fired,
            "kill_probability": self.kill_probability,
            "battery_utilization": self.battery_utilization,
        }
