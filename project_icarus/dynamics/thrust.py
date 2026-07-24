from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np
from .coordinate_systems import quat_normalize


G0 = 9.80665


class StageSeparation:
    def __init__(self, time, mass_drop, impulse=None, state_jump=None,
                 separation_delay=0.0, spin_impulse=(0.0, 0.0, 0.0)):
        self.time = time
        self.mass_drop = mass_drop
        self.impulse = impulse if impulse is not None else np.zeros(3)
        self.state_jump = state_jump if state_jump is not None else {}
        self.separation_delay = separation_delay
        self.spin_impulse = np.asarray(spin_impulse, dtype=float)
        self.applied = False

    def apply(self, state):
        new_state = dict(state)
        new_state["m"] = state["m"] - self.mass_drop
        new_state["v"] = state["v"] + self.impulse / max(state["m"], 1e-6)
        for key, jump in self.state_jump.items():
            if key in new_state:
                new_state[key] = new_state[key] + jump
        if "omega" in new_state:
            new_state["omega"] = new_state["omega"] + self.spin_impulse / max(state["m"], 1e-6)
        return new_state


class MKVSystem:
    def __init__(self, kv_mass=15.0, v_rel=1.5, divert_thrust=85.0, divert_gimbal=np.radians(15.0)):
        self.kv_mass = kv_mass
        self.v_rel = v_rel
        self.divert_thrust = divert_thrust
        self.divert_gimbal = divert_gimbal
        self.separated = False

    def separate(self, state):
        if self.separated:
            return state
        new_state = dict(state)
        new_state["m"] = state["m"] - self.kv_mass
        new_state["v"] = state["v"] + self.v_rel * np.array([1.0, 0.0, 0.0])
        new_state["q"] = quat_normalize(state["q"])
        self.separated = True
        return new_state

    def divert(self, state, direction_body, dt=1.0):
        new_state = dict(state)
        dir_b = np.asarray(direction_body, dtype=float)
        dir_b = dir_b / max(np.linalg.norm(dir_b), 1e-12)
        dir_b = np.clip(dir_b, -np.tan(self.divert_gimbal), np.tan(self.divert_gimbal))
        impulse = self.divert_thrust * dt * dir_b
        new_state["v"] = state["v"] + impulse / max(state["m"], 1e-6)
        return new_state


class ThrustModel:
    def __init__(self, thrust_profile, mass_flow=None, Isp=250.0,
                 gimbal_limits=(np.radians(15), np.radians(15)),
                 gimbal_rate=np.radians(30)):
        self.thrust_profile = thrust_profile
        self.mass_flow = mass_flow
        self.Isp = Isp
        self.gimbal_limits = np.asarray(gimbal_limits, dtype=float)
        self.gimbal_rate = gimbal_rate
        self.separations = []
        self._gimbal = np.zeros(2)

    def thrust(self, t, state):
        return self.thrust_profile(t)

    def gimbal(self, t, commanded_angles):
        cmd = np.clip(np.asarray(commanded_angles, dtype=float), -self.gimbal_limits, self.gimbal_limits)
        max_step = self.gimbal_rate
        clamped = np.clip(cmd, self._gimbal - max_step, self._gimbal + max_step)
        self._gimbal = clamped
        return clamped

    def thrust_vector(self, t, state, gimbal_angles=None):
        T = self.thrust(t, state)
        if gimbal_angles is None:
            gimbal_angles = self._gimbal
        pitch, yaw = gimbal_angles
        v = np.array([np.cos(pitch) * np.cos(yaw),
                      np.sin(yaw),
                      np.sin(pitch) * np.cos(yaw)])
        return T * v, np.array([pitch, yaw])

    def add_separation(self, sep: StageSeparation):
        self.separations.append(sep)

    def mass_rate(self, t, state):
        T = self.thrust_profile(t)
        if T <= 0.0:
            return 0.0
        if self.mass_flow is not None:
            return -self.mass_flow
        return -T / (self.Isp * G0)


@dataclass
class StageSpec:
    """Definition of a single rocket stage in a multi-stage vehicle.

    ``thrust`` is the sea-level/instantaneous thrust magnitude (N) as a function
    of time-since-stage-ignition. ``burn_time`` is the nominal burn duration; the
    stage is considered burnt out when ``t_stage >= burn_time`` or thrust hits 0.
    ``wet_mass``/``dry_mass`` bound the propellant mass available for this stage.

    ``inertia`` is the residual bus inertia (kg m^2, diagonal) AFTER this stage
    and all spent stages below it have separated — i.e. the inertia the vehicle
    flies with once ``burn_time`` elapses and the stage is jettisoned. ``cg`` is
    the corresponding centre-of-gravity offset (m, body frame) used for the
    gravity-gradient torque. Both feed the post-separation inertia update in the
    integrator loop (Phase 1B.2).

    ``thrust_table`` is an optional callable ``(alt_m, mach) -> float`` that
    returns a thrust multiplier (1.0 = nominal).  When ``None`` the stage uses
    constant thrust as specified by ``thrust``.
    """

    thrust: Callable[[float], float]
    burn_time: float
    wet_mass: float
    dry_mass: float
    Isp: float = 250.0
    gimbal_limits: tuple = (np.radians(15), np.radians(15))
    gimbal_rate: float = np.radians(30)
    name: str = "stage"
    inertia: Optional[np.ndarray] = None
    cg: Optional[np.ndarray] = None
    thrust_table: Optional[Callable[[float, float], float]] = None
    mass_flow: Optional[float] = None


class MultiStageThrustModel:
    """True multi-stage thrust model with sequential burns and separation.

    Stages burn in order. After a stage's ``burn_time`` elapses (or thrust
    returns to ~0) the next stage ignites. At each ignition boundary a
    :class:`StageSeparation` impulse is applied (mass drop + delta-v/spin) via
    the integrator event loop. Common-bus inertial properties are updated by the
    caller after each separation so mass flow is consistent.
    """

    def __init__(self, stages: List[StageSpec], sep_impulse_per_stage: Optional[List[np.ndarray]] = None):
        if not stages:
            raise ValueError("MultiStageThrustModel requires at least one stage.")
        self.stages = stages
        self.sep_impulse_per_stage = sep_impulse_per_stage or [np.zeros(3) for _ in stages]
        self._ignition_times: List[float] = []
        self._current = 0
        self._compute_ignition_times()

    def _compute_ignition_times(self):
        self._ignition_times = []
        t = 0.0
        for s in self.stages:
            self._ignition_times.append(t)
            t += s.burn_time

    @property
    def total_burn_time(self) -> float:
        return sum(s.burn_time for s in self.stages)

    def stage_index_at(self, t: float) -> int:
        idx = 0
        for i, t0 in enumerate(self._ignition_times):
            if t >= t0:
                idx = i
        return idx

    def thrust(self, t: float, state=None) -> float:
        idx = self.stage_index_at(t)
        s = self.stages[idx]
        t_stage = t - self._ignition_times[idx]
        if t_stage < 0.0:
            return 0.0
        if t_stage > s.burn_time:
            return 0.0
        T_base = float(s.thrust(t_stage))
        if s.thrust_table is None or state is None:
            return T_base
        try:
            alt = float(np.linalg.norm(state.get("r", np.zeros(3))) - 6371e3)
            v = state.get("v", np.zeros(3))
            vmag = float(np.linalg.norm(v))
            mach = vmag / 295.0 if vmag > 1e-6 else 0.0
            mult = float(s.thrust_table(alt, mach))
            return T_base * max(mult, 0.0)
        except Exception:
            return T_base

    def thrust_vector(self, t, state, gimbal_angles=None):
        T = self.thrust(t, state)
        idx = self.stage_index_at(t)
        if gimbal_angles is None:
            gimbal_angles = np.zeros(2)
        pitch, yaw = np.asarray(gimbal_angles, dtype=float)
        v = np.array([
            -np.cos(pitch) * np.cos(yaw),
            np.sin(yaw),
            np.sin(pitch) * np.cos(yaw),
        ])
        return T * v, np.array([pitch, yaw])

    def mass_rate(self, t, state) -> float:
        T = self.thrust(t, state)
        if T <= 0.0:
            return 0.0
        idx = self.stage_index_at(t)
        stage = self.stages[idx]
        if stage.mass_flow is not None:
            return stage.mass_flow
        derived = -T / (stage.Isp * G0)
        max_allowed = -(stage.wet_mass - stage.dry_mass) / max(stage.burn_time, 1e-9)
        if abs(derived) > abs(max_allowed):
            return max_allowed
        return derived

    def separation_for_stage(self, stage_idx: int) -> Optional[StageSeparation]:
        if stage_idx >= len(self.stages):
            return None
        s = self.stages[stage_idx]
        mass_drop = max(s.wet_mass - s.dry_mass, 0.0)
        t_sep = self._ignition_times[stage_idx] + s.burn_time
        return StageSeparation(
            time=t_sep,
            mass_drop=mass_drop,
            impulse=self.sep_impulse_per_stage[stage_idx],
            separation_delay=0.0,
        )

    @property
    def separations(self) -> List[StageSeparation]:
        return [self.separation_for_stage(i) for i in range(len(self.stages))]

    def inertia_after_separation(self, stage_idx: int) -> Optional[np.ndarray]:
        n = len(self.stages)
        if stage_idx < 0 or stage_idx >= n:
            return None
        for idx in (stage_idx + 1, stage_idx):
            if idx < n:
                inert = self.stages[idx].inertia
                if inert is not None:
                    return np.asarray(inert, dtype=float)
        return None

    def cg_after_separation(self, stage_idx: int) -> Optional[np.ndarray]:
        n = len(self.stages)
        if stage_idx < 0 or stage_idx >= n:
            return None
        for idx in (stage_idx + 1, stage_idx):
            if idx < n:
                 cg = self.stages[idx].cg
                 if cg is not None:
                     return np.asarray(cg, dtype=float)
        return None


def sarmat_stage_specs():
    """Authoritative RS-28 Sarmat stage specifications.

    Refs: Astronautica RD-274 (4,952 kN), Missile Defense Advocacy (PDU-99),
    Army Recognition (four RS-99, "over 100 tons thrust").
    """
    return [
        StageSpec(thrust=lambda t: 5.0e6, burn_time=90.0,
                  wet_mass=140_000.0, dry_mass=11_340.0, Isp=315.0,
                  mass_flow=1430.0),
        StageSpec(thrust=lambda t: 1.2e6, burn_time=180.0,
                  wet_mass=48_000.0, dry_mass=3_600.0, Isp=325.0,
                  mass_flow=247.0),
        StageSpec(thrust=lambda t: 2.5e5, burn_time=90.0,
                  wet_mass=10_100.0, dry_mass=2_160.0, Isp=330.0,
                  mass_flow=77.2),
    ]


def sarmat_stage_dicts():
    specs = sarmat_stage_specs()
    t_cur = 0.0
    out = []
    for s in specs:
        t_start = t_cur
        t_end = t_cur + s.burn_time
        out.append({
            "t_start": t_start,
            "t_end": t_end,
            "thrust": float(s.thrust(0.0)),
            "m_dot": abs(s.mass_flow),
            "Isp": s.Isp,
        })
        t_cur = t_end
    return out
