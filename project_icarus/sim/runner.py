from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
import numpy as np
from scipy.integrate import RK45, solve_ivp

from ..dynamics.eom_6dof import EOM6DOF
from ..guidance.boost_guidance import BoostGuidance
from ..guidance.midcourse_guidance import MidcourseGuidance
from ..guidance.terminal_guidance import TerminalGuidance
from ..interceptors.config import InterceptorConfig
from ..dynamics.thrust import MultiStageThrustModel, MKVSystem
from ..guidance.law import GuidanceLaw
from ..scenarios.target_factory import TargetScenario
from ..scenarios.scenario import EngagementScenario
from ..aero.aero_analytical import blended_aero
from .config import SimConfig, get_config, logger


# ---------------------------------------------------------------------------
# Phase-transition events (replace the old time-based phase switch)
# ---------------------------------------------------------------------------

PHASES = ("boost", "midcourse", "terminal")


class PhaseEvent:
    """Base class for an event that can trigger a phase transition.

    ``should_trigger`` inspects the current integration state and returns True
    when the transition should fire. Subclasses implement domain logic.
    """

    next_phase: str = "midcourse"

    def __init__(self, next_phase: str = "midcourse"):
        self.next_phase = next_phase

    def should_trigger(self, t: float, y: np.ndarray, ctx: Dict[str, Any]) -> bool:
        raise NotImplementedError


class ThrustCutoffEvent(PhaseEvent):
    """Boost -> midcourse when thrust has effectively ended.

    Fires when the instantaneous thrust is below ``frac`` of the peak observed
    thrust this run OR the vehicle mass has reached its dry mass.
    """

    def __init__(self, frac: float = 1e-3, next_phase: str = "midcourse"):
        super().__init__(next_phase)
        self.frac = frac

    def should_trigger(self, t, y, ctx):
        peak = ctx.get("peak_thrust", 0.0)
        if peak > 0.0 and ctx.get("thrust", 0.0) < self.frac * peak:
            return True
        dry = ctx.get("dry_mass", -1.0)
        if dry > 0.0 and y[13] <= dry:
            return True
        return False


class ReentryEvent(PhaseEvent):
    """Midcourse -> terminal when altitude drops below ``alt``."""

    def __init__(self, alt: float = 100e3, next_phase: str = "terminal"):
        super().__init__(next_phase)
        self.alt = alt

    def should_trigger(self, t, y, ctx):
        return (np.linalg.norm(y[:3]) - 6371e3) < self.alt


class RangeEvent(PhaseEvent):
    """Midcourse -> terminal when slant range-to-target drops below ``range_``."""

    def __init__(self, range_: float = 50e3, next_phase: str = "terminal"):
        super().__init__(next_phase)
        self.range_ = range_

    def should_trigger(self, t, y, ctx):
        target_state = ctx.get("target_state")
        if target_state is None:
            return False
        return np.linalg.norm(target_state[:3] - y[:3]) < self.range_


class SpeedEvent(PhaseEvent):
    """Midcourse -> terminal once homing-speed regime is reached."""

    def __init__(self, speed: float = 4000.0, next_phase: str = "terminal"):
        super().__init__(next_phase)
        self.speed = speed

    def should_trigger(self, t, y, ctx):
        return np.linalg.norm(y[3:6]) < self.speed


class SeparationEvent(PhaseEvent):
    """Fires when a stage burns out so its separation impulse can be injected.

    Unlike the time-based ``StageSeparation.time`` default, this event triggers
    on the *physics* of burnout: the active stage's thrust has dropped to
    ``frac`` of the observed peak AND ``t`` has reached (or passed) the nominal
    ignition+burn boundary for ``stage_idx``. This makes stage jettison robust to
    a coasting/low-thrust vehicle that would otherwise spuriously fire a timed
    separation (Phase 1A.3 / 1B.1). The integrator loop consumes this event to
    apply the mass drop + delta-v and recompute the bus inertia/CG.
    """

    def __init__(self, stage_idx: int, thrust_model, cfg: "SimConfig" = None,
                 frac: float = 1e-3, next_phase: str = "midcourse"):
        super().__init__(next_phase)
        self.stage_idx = stage_idx
        self.thrust_model = thrust_model
        self.frac = frac
        self._ignition = 0.0
        if thrust_model is not None and hasattr(thrust_model, "_ignition_times"):
            times = thrust_model._ignition_times
            if 0 <= stage_idx < len(times):
                self._ignition = times[stage_idx]
            burn = getattr(thrust_model, "stages", [None])[stage_idx].burn_time if (
                hasattr(thrust_model, "stages") and 0 <= stage_idx < len(thrust_model.stages)
            ) else 0.0
            self._burnout = self._ignition + burn
        else:
            self._burnout = float("inf")

    def should_trigger(self, t, y, ctx):
        if t < self._burnout:
            return False
        peak = ctx.get("peak_thrust", 0.0)
        if peak > 0.0 and ctx.get("thrust", 0.0) < self.frac * peak:
            return True
        return False


def _default_events(cfg: SimConfig):
    return [
        ThrustCutoffEvent(frac=cfg.thrust_cutoff_frac, next_phase="midcourse"),
        ReentryEvent(alt=cfg.reentry_alt, next_phase="terminal"),
        RangeEvent(range_=cfg.terminal_range, next_phase="terminal"),
    ]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    miss_distances: List[float]
    kill_assessments: List[bool]
    perturbations: List[Dict[str, Any]]
    mean_miss: float = 0.0
    std_miss: float = 0.0
    kill_probability: float = 0.0
    n_rejected: int = 0

    def __post_init__(self):
        finite = [m for m in self.miss_distances if np.isfinite(m)]
        self.n_rejected = len(self.miss_distances) - len(finite)
        if finite:
            self.mean_miss = float(np.mean(finite))
            self.std_miss = float(np.std(finite))
            kills_finite = [
                k for m, k in zip(self.miss_distances, self.kill_assessments)
                if np.isfinite(m)
            ]
            self.kill_probability = float(np.mean(kills_finite)) if kills_finite else 0.0


@dataclass
class EngagementResult:
    nominal_trajectory: Dict[str, np.ndarray]
    target_trajectory: Dict[str, np.ndarray]
    miss_distance: float
    kill_assessment: bool
    monte_carlo: Optional[MonteCarloResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def plot_3d(self, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        r_i = self.nominal_trajectory.get("r", np.zeros((1, 3)))
        r_t = self.target_trajectory.get("r", np.zeros((1, 3)))
        ax.plot(r_i[:, 0], r_i[:, 1], r_i[:, 2], label="Interceptor", color="blue")
        ax.plot(r_t[:, 0], r_t[:, 1], r_t[:, 2], label="Target", color="red", linestyle="--")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        return ax

    def plot_miss_distance_distribution(self, ax=None):
        if self.monte_carlo is None:
            print("No Monte Carlo data available")
            return None
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return None
        if ax is None:
            fig, ax = plt.subplots()
        misses = np.asarray(self.monte_carlo.miss_distances, dtype=float)
        finite_mask = np.isfinite(misses)
        if not np.all(finite_mask):
            n_bad = int(np.sum(~finite_mask))
            print(f"Warning: {n_bad} non-finite miss distance(s) excluded from histogram")
            misses = misses[finite_mask]
        if misses.size == 0:
            print("No finite miss distances to plot")
            return ax
        ax.hist(misses, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(self.miss_distance, color="red", linestyle="--", label="Nominal")
        ax.set_xlabel("Miss Distance (m)")
        ax.set_ylabel("Count")
        ax.legend()
        return ax

    def plot_kill_probability_vs_param(self, param_name: str, param_values: List[float], ax=None):
        if self.monte_carlo is None:
            print("No Monte Carlo data available")
            return None
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return None
        if ax is None:
            fig, ax = plt.subplots()
        probs = [self.monte_carlo.kill_probability for _ in param_values]
        ax.plot(param_values, probs, marker="o")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Kill Probability")
        return ax


# ---------------------------------------------------------------------------
# Closed-loop RHS: event-driven phase selection
# ---------------------------------------------------------------------------

def _compute_v0(scenario: EngagementScenario, magnitude: float = 1500.0) -> np.ndarray:
    """Initial interceptor speed from launch site toward threat axis."""
    site = np.asarray(scenario.interceptor_launch_site, dtype=float)
    axis = np.asarray(scenario.threat_axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return magnitude * axis


def _surrogate(mach, alpha, beta, alt):
    return blended_aero(mach, alpha, beta, alt)[:3]


def _phase_stateless(t, y, phase_events, ctx):
    """Stateless phase determination (Phase 3.3 / hardening).

    The active phase is derived purely from the registered events evaluated at
    ``(t, y)`` rather than from mutable module-level state. The only allowed
    transition order is boost -> midcourse -> terminal, so the phase is simply:
    boost until a boost->midcourse event fires, then midcourse until a
    midcourse->terminal event fires, then terminal. This keeps the RHS safe
    under adaptive integrators (``solve_ivp``/DOP853) that evaluate ``f`` at
    arbitrary points out of time order.
    """
    to_midcourse = any(
        ev.next_phase == "midcourse" and ev.should_trigger(t, y, ctx)
        for ev in phase_events
    )
    if not to_midcourse:
        return "boost"
    to_terminal = any(
        ev.next_phase == "terminal" and ev.should_trigger(t, y, ctx)
        for ev in phase_events
    )
    return "terminal" if to_terminal else "midcourse"


def _closed_loop_rhs(t, y, interceptor, guidance_law, target_fn, eom, thrust_fn,
                     peak_thrust, phase_events, mass_floor=1e-3, wind_model=None):
    r = y[:3]
    v = y[3:6]
    q = y[6:10]
    omega = y[10:13]
    m = max(float(y[13]), mass_floor)  # Phase 3.4 mass floor (prevents 1/m blowup)
    q = q / max(np.linalg.norm(q), 1e-12)

    target_state = target_fn(t)
    target_r = target_state[:3]
    target_v = target_state[3:6]

    alt = np.linalg.norm(r) - 6371e3

    # Stateless event-driven phase determination (no time-based switch).
    ctx = {
        "thrust": float(thrust_fn(t, {"m": m})) if thrust_fn is not None else 0.0,
        "peak_thrust": peak_thrust,
        "dry_mass": getattr(eom, "dry_mass", -1.0),
        "target_state": target_state,
    }
    phase = _phase_stateless(t, y, phase_events, ctx)

    rho = 1.225 * np.exp(-max(alt, 0.0) / 8500.0) if alt < 100e3 else 0.0

    eom_state = {"r": r, "v": v, "q": q, "omega": omega, "m": m}
    if wind_model is not None:
        eom_state["wind"] = wind_model

    f_thrust = np.zeros(3)
    autopilot = getattr(guidance_law, "autopilot", None)
    tracker = getattr(guidance_law, "tracker", None)
    if phase == "boost":
        cmd = guidance_law.boost.commanded_gimbal(t, eom_state, rho, v)
        thrust_val = thrust_fn(t, eom_state) if thrust_fn is not None else 0.0
        f_thrust = np.array([-thrust_val, 0.0, 0.0])
    else:
        # Midcourse / terminal guidance commands an acceleration; the full-fidelity
        # autopilot (if present) converts the command into a realized acceleration
        # with second-order actuator dynamics and body-rate feedback.
        avail_thrust = float(thrust_fn(t, eom_state)) if thrust_fn is not None else 0.0
        max_accel_force = avail_thrust if avail_thrust > 0.0 else 0.0
        if phase == "midcourse":
            guidance_law.midcourse.update_target(target_state, t)
            accel_cmd = guidance_law.midcourse.commanded_accel(t, eom_state)
            # Feed the data-link 3D position measurement into the target tracker.
            if tracker is not None:
                try:
                    meas = guidance_law.midcourse.measurement_noise()
                    if meas is not None:
                        tracker.update(t, meas[:3])
                except Exception:
                    pass
        else:
            seeker = getattr(guidance_law, "seeker", None)
            if seeker is not None:
                los_rate = seeker.update_tracker(eom_state, {"r": target_r, "v": target_v})
                _disc = getattr(guidance_law, "discriminate_target", None)
                _tgt_scenario = getattr(guidance_law, "_target_scenario", None)
                if _disc is not None and _tgt_scenario is not None:
                    feats = getattr(_tgt_scenario, "decoy_features", None)
                    if feats is not None:
                        for f in feats(t, seed=guidance_law.config.seeker_noise_seed):
                            guidance_law._decoy_rejects = getattr(
                                guidance_law, "_decoy_rejects", 0
                            ) + (0 if _disc(f) else 1)
                accel_cmd = guidance_law.terminal.commanded_accel_seeker(
                    eom_state, {"r": target_r, "v": target_v}, los_rate=los_rate
                )
            else:
                accel_cmd = guidance_law.terminal.commanded_accel(t, eom_state, target_state)

        # Apply full-fidelity autopilot inner-loop dynamics with body-rate feedback.
        if autopilot is not None:
            dt_auto = max(t - getattr(autopilot, "_last_t", t), 1e-6)
            autopilot._last_t = t
            accel_cmd = autopilot.update(
                accel_cmd,
                dt=dt_auto,
                quat=q,
                omega_body=omega,
            )

        # Convert the acceleration command to a force, capped by available thrust.
        desired_force = accel_cmd * m
        f_mag = np.linalg.norm(desired_force)
        if f_mag > 1e-9 and f_mag > max_accel_force:
            f_thrust = desired_force * (max_accel_force / f_mag)
        else:
            f_thrust = desired_force

    derivs = eom.compute(t, eom_state, _surrogate)
    dr_dt = derivs["r"]
    dv_dt = derivs["v"] + f_thrust / max(m, 1e-6)
    dq_dt = derivs["q"]
    domega_dt = derivs["omega"]
    dm_dt = derivs["m"]

    return np.concatenate([dr_dt, dv_dt, dq_dt, domega_dt, [dm_dt]])


def _track_phase(t_grid, y_grid, phase_events, peak_thrust, thrust_fn, eom, mass_floor):
    """Forward-pass phase tracker over a saved trajectory (monotonic).

    Used by the DOP853 path to recover the phase at each saved point so the
    separation/MKV impulses can be injected in the correct phase (Phase 1A.3).
    """
    phase = "boost"
    out = []
    for t, y in zip(t_grid, y_grid.T):
        m = max(float(y[13]), mass_floor)
        ctx = {
            "thrust": float(thrust_fn(t, {"m": m})) if thrust_fn is not None else 0.0,
            "peak_thrust": peak_thrust,
            "dry_mass": getattr(eom, "dry_mass", -1.0),
            "target_state": None,
        }
        if phase == "boost":
            if any(ev.next_phase == "midcourse" and ev.should_trigger(t, y, ctx)
                   for ev in phase_events):
                phase = "midcourse"
        elif phase == "midcourse":
            if any(ev.next_phase == "terminal" and ev.should_trigger(t, y, ctx)
                   for ev in phase_events):
                phase = "terminal"
        out.append(phase)
    return out


def _apply_separations(times, states, separations, separation_events, stage_thrust_model,
                       eom, mkv, cfg, peak_thrust, thrust_fn, mass_floor):
    """Inject stage-separation + MKV impulses into a saved (time, state) grid.

    Shared by both integrator backends. Because separations are impulsive
    (instantaneous mass drop + delta-v that persist for the rest of flight), the
    corrections are *accumulated* and applied to the entire post-separation
    trajectory, not just the single grid point where burnout is detected. This
    matches what the RK45 stepper does by re-injecting ``integrator.y``.

    Returns ``(times, states, final_phase)``.
    """
    orig = states.copy()
    phase_by_idx = _track_phase(times, states, separation_events, peak_thrust,
                                thrust_fn, eom, mass_floor) if (separations or mkv) else ["boost"] * len(times)
    applied = set()
    cum_dv = np.zeros(3)        # cumulative velocity delta-v (inertial, m/s)
    cum_domega = np.zeros(3)    # cumulative spin impulse / mass
    cum_dm = 0.0                # cumulative mass drop (kg)
    mkv_done = False
    for k in range(len(times)):
        t = times[k]
        y = orig[:, k].copy()
        thrust_now = float(thrust_fn(t, {"m": y[13]})) if thrust_fn is not None else 0.0
        sep_ctx = {"thrust": thrust_now, "peak_thrust": peak_thrust}
        for i, sep in enumerate(separations):
            if i in applied:
                continue
            if (separation_events and separation_events[i].should_trigger(t, orig[:, k], sep_ctx)):
                applied.add(i)
                m_now = max(y[13] - cum_dm, mass_floor)
                cum_dm += sep.mass_drop
                cum_dv += sep.impulse / max(m_now, 1e-6)
                if getattr(sep, "spin_impulse", None) is not None:
                    cum_domega += np.asarray(sep.spin_impulse) / max(m_now, 1e-6)
                if stage_thrust_model is not None:
                    inert = stage_thrust_model.inertia_after_separation(i)
                    if inert is not None:
                        cg = stage_thrust_model.cg_after_separation(i)
                        eom.set_inertia(inert, cg)
        # Multi-KV payload separation at terminal phase (1A.3).
        if (mkv is not None and not mkv_done and phase_by_idx[k] == "terminal"):
            mkv_done = True
            m_now = max(y[13] - cum_dm, mass_floor)
            cum_dm += mkv.kv_mass
            cum_dv += mkv.v_rel * np.array([1.0, 0.0, 0.0])
        # Overlay the accumulated impulsive corrections on the integrated state.
        y[13] = max(y[13] - cum_dm, mass_floor)
        y[3:6] = y[3:6] + cum_dv
        y[10:13] = y[10:13] + cum_domega
        states[:, k] = y
    return times, states, phase_by_idx[-1] if phase_by_idx else "boost"


def _integrate_trajectory(interceptor, guidance_law, target, scenario, perturb=None,
                          cfg: Optional[SimConfig] = None, rng=None):
    cfg = cfg or get_config()
    rng = rng or cfg.rng
    site = np.asarray(scenario.interceptor_launch_site, dtype=float)
    v0 = _compute_v0(scenario, magnitude=1500.0)

    state0 = np.concatenate([
        site,
        v0,
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [interceptor.mass],
    ])
    if perturb is not None:
        state0 = state0.copy()
        if "position_sigma" in perturb:
            state0[:3] += rng.normal(0, perturb["position_sigma"], 3)
        if "velocity_sigma" in perturb:
            state0[3:6] += rng.normal(0, perturb["velocity_sigma"], 3)
        if "mass_sigma" in perturb:
            state0[13] += rng.normal(0, perturb["mass_sigma"])

    eom = EOM6DOF(
        mass=interceptor.mass,
        inertia=interceptor.inertia,
        area=interceptor.area,
        ref_length=interceptor.ref_length,
    )
    eom.dry_mass = getattr(interceptor, "dry_mass", -1.0)

    # Wire the thrust model if the interceptor exposes one (multi-stage or scalar).
    thrust_fn = getattr(interceptor, "_thrust_callable", None)
    peak_thrust = float(getattr(interceptor, "peak_thrust", 0.0) or 0.0)

    stage_thrust_model = None
    if getattr(interceptor, "stages", None):
        stage_thrust_model = MultiStageThrustModel(interceptor.stages, interceptor.sep_impulses)

    separations = getattr(interceptor, "_separations", []) or []
    mkv = getattr(interceptor, "_mkv", None)

    separation_events = [
        SeparationEvent(i, stage_thrust_model, cfg)
        for i in range(len(separations))
    ] if stage_thrust_model is not None else []

    phase_events = _default_events(cfg)
    mass_floor = cfg.mass_floor
    wind_model = cfg.wind_model if cfg.use_wind else None

    t_span = [scenario.engagement_start, scenario.engagement_end]
    t_end = min(t_span[1], cfg.t_max)
    target_fn = lambda t: target.propagate(t)

    # Expose the target scenario so the terminal seeker can run RV/decoy
    # discrimination (2C) and the event evaluator can read target state.
    guidance_law._target_scenario = target
    guidance_law._decoy_rejects = 0

    def _make_rhs():
        return lambda t, y: _closed_loop_rhs(
            t, y, interceptor, guidance_law, target_fn, eom,
            thrust_fn, peak_thrust, phase_events, mass_floor,
            wind_model=wind_model,
        )

    if cfg.integrator == "dop853":
        # Phase 3.3: adaptive 8th-order Dormand-Prince for stiff/hypersonic endo.
        sol = solve_ivp(
            _make_rhs(), (t_span[0], t_end), state0, method="DOP853",
            rtol=cfg.rtol, atol=cfg.atol, max_step=cfg.max_step,
            dense_output=False,
            t_eval=np.linspace(t_span[0], t_end, max(2, int((t_end - t_span[0]) / 0.5) + 1)),
        )
        if not sol.success:
            logger.warning("DOP853 integration failed: %s", sol.message)
        times = sol.t
        states = sol.y
        times, states, _ = _apply_separations(
            times, states, separations, separation_events, stage_thrust_model,
            eom, mkv, cfg, peak_thrust, thrust_fn, mass_floor,
        )
    else:
        # Default RK45 (historical explicit Runge-Kutta stepper).
        integrator = RK45(_make_rhs(), t_span[0], state0, t_end,
                          max_step=cfg.max_step, rtol=cfg.rtol, atol=cfg.atol)
        times = [integrator.t]
        states = [integrator.y.copy()]
        applied = set()
        while integrator.status == "running":
            integrator.step()
            t = integrator.t
            y = integrator.y

            thrust_now = float(thrust_fn(t, {"m": y[13]})) if thrust_fn is not None else 0.0
            sep_ctx = {"thrust": thrust_now, "peak_thrust": peak_thrust}
            for i, sep in enumerate(separations):
                if i in applied:
                    continue
                if (separation_events and separation_events[i].should_trigger(t, y, sep_ctx)) or (
                    integrator.status != "running"
                ):
                    applied.add(i)
                    y = np.array(y, copy=True)
                    y[13] = max(y[13] - sep.mass_drop, mass_floor)
                    y[3:6] = y[3:6] + sep.impulse / max(y[13], 1e-6)
                    if getattr(sep, "spin_impulse", None) is not None:
                        y[10:13] = y[10:13] + np.asarray(sep.spin_impulse) / max(y[13], 1e-6)
                    if stage_thrust_model is not None:
                        inert = stage_thrust_model.inertia_after_separation(i)
                        if inert is not None:
                            cg = stage_thrust_model.cg_after_separation(i)
                            eom.set_inertia(inert, cg)
                    integrator.y = y

            if mkv is not None and not getattr(mkv, "separated", False):
                # Terminal phase for RK45: recompute statelessly at this state.
                ctx = {"thrust": thrust_now, "peak_thrust": peak_thrust,
                       "dry_mass": getattr(eom, "dry_mass", -1.0), "target_state": None}
                phase_now = _phase_stateless(t, integrator.y, phase_events, ctx)
                if phase_now == "terminal":
                    y = np.array(integrator.y, copy=True)
                    y[13] = max(y[13] - mkv.kv_mass, mass_floor)
                    y[3:6] = y[3:6] + mkv.v_rel * np.array([1.0, 0.0, 0.0])
                    integrator.y = y
                    mkv.separated = True

            times.append(integrator.t)
            states.append(integrator.y.copy())

            if integrator.t >= t_end:
                integrator.status = "finished"

        times = np.array(times)
        states = np.column_stack(states)

    sol_t = np.asarray(times)
    sol_y = np.asarray(states)
    if sol_y.ndim == 1:
        sol_y = sol_y.reshape(-1, 1)

    r = sol_y[:3, -1]
    target_final = target_fn(t_end)
    miss = float(np.linalg.norm(r - target_final[:3]))
    kill = guidance_law.terminal.kill_assessment(miss)

    traj = {
        "t": sol_t,
        "r": sol_y[:3, :].T,
        "v": sol_y[3:6, :].T,
        "q": sol_y[6:10, :].T,
        "omega": sol_y[10:13, :].T,
        "m": sol_y[13, :],
    }
    target_traj = {
        "t": sol_t,
        "r": target.propagate_batch(sol_t)[:, :3] if hasattr(target, "propagate_batch") else np.array([target_fn(ti)[:3] for ti in sol_t]),
        "v": target.propagate_batch(sol_t)[:, 3:] if hasattr(target, "propagate_batch") else np.array([target_fn(ti)[3:] for ti in sol_t]),
    }
    return traj, target_traj, miss, kill


from .parallel_mc import ParallelMonteCarloRunner


class EngagementRunner:
    """End-to-end engagement simulation runner (event-driven)."""

    def __init__(
        self,
        interceptor: InterceptorConfig,
        guidance: GuidanceLaw,
        target: TargetScenario,
        scenario: EngagementScenario,
        cfg: Optional[SimConfig] = None,
    ):
        self.interceptor = interceptor
        self.guidance = guidance
        self.target = target
        self.scenario = scenario
        self.cfg = cfg or get_config()
        self._parallel_runner = ParallelMonteCarloRunner()

    def run(self, n_trials: int = 50, perturbations: Optional[Dict[str, float]] = None,
            cfg: Optional[SimConfig] = None, parallel: bool = True) -> EngagementResult:
        cfg = cfg or self.cfg
        if perturbations is None:
            perturbations = dict(cfg.perturbations)

        # Dedicated, seedable RNG for this run (does not advance global state).
        rng = np.random.default_rng(cfg.seed)

        nominal_traj, nominal_target_traj, nominal_miss, nominal_kill = _integrate_trajectory(
            self.interceptor, self.guidance, self.target, self.scenario,
            cfg=cfg, rng=rng,
        )

        if parallel and n_trials > 1:
            mc_result = self._parallel_runner.run(
                self.interceptor, self.guidance, self.target, self.scenario,
                n_trials=n_trials, perturbations=perturbations, cfg=cfg,
                base_seed=cfg.seed + 1000,
            )
            mc_misses = mc_result["miss_distances"]
            mc_kills = mc_result["kill_assessments"]
            mc_perturbs = mc_result["perturbations"]
        else:
            mc_misses = []
            mc_kills = []
            mc_perturbs = []
            for _ in range(n_trials):
                _, _, miss, kill = _integrate_trajectory(
                    self.interceptor, self.guidance, self.target, self.scenario,
                    perturb=perturbations, cfg=cfg, rng=rng,
                )
                if not np.isfinite(miss):
                    if cfg.reject_nonfinite:
                        logger.warning("rejecting non-finite miss distance (trial); "
                                       "cf. numerical safeguard 3.4")
                        continue
                    mc_misses.append(float("nan"))
                    mc_kills.append(False)
                    mc_perturbs.append(perturbations)
                    continue
                mc_misses.append(miss)
                mc_kills.append(kill)
                mc_perturbs.append(perturbations)

        mc = MonteCarloResult(
            miss_distances=mc_misses,
            kill_assessments=mc_kills,
            perturbations=mc_perturbs,
        )

        return EngagementResult(
            nominal_trajectory=nominal_traj,
            target_trajectory=nominal_target_traj,
            miss_distance=nominal_miss,
            kill_assessment=nominal_kill,
            monte_carlo=mc,
            metadata={"n_trials": n_trials, "n_rejected": mc.n_rejected},
        )
