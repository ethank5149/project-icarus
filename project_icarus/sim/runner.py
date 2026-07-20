from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
import numpy as np
from scipy.integrate import RK45

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

    def __post_init__(self):
        if self.miss_distances:
            self.mean_miss = float(np.mean(self.miss_distances))
            self.std_miss = float(np.std(self.miss_distances))
            self.kill_probability = float(np.mean(self.kill_assessments))


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


def _current_phase(t, y, phase_events, ctx):
    """Evaluate which phase is active given the registered transition events.

    Only ONE monotonic transition boost -> midcourse -> terminal is permitted.
    Boost can leave for midcourse (thrust cutoff); midcourse can leave for
    terminal (reentry / range / speed). Terminal is terminal. This prevents a
    launch at sub-terminal speed from spuriously jumping straight to terminal
    guidance (which would command huge accelerations on a coasting body).
    """
    current = ctx.get("phase", "boost")
    if current == "terminal":
        return "terminal"
    for ev in phase_events:
        if ev.next_phase == current:
            continue  # a same-phase event is not a transition
        if current == "boost" and ev.next_phase != "midcourse":
            continue  # boost must go to midcourse first
        if ev.should_trigger(t, y, ctx):
            logger.debug("phase transition %s -> %s at t=%.2f", current, ev.next_phase, t)
            return ev.next_phase
    return current


def _closed_loop_rhs(t, y, interceptor, guidance_law, target_fn, eom, thrust_fn, peak_thrust):
    r = y[:3]
    v = y[3:6]
    q = y[6:10]
    omega = y[10:13]
    m = y[13]
    q = q / max(np.linalg.norm(q), 1e-12)

    target_state = target_fn(t)
    target_r = target_state[:3]
    target_v = target_state[3:6]

    alt = np.linalg.norm(r) - 6371e3

    # Event-driven phase determination (no time-based switch).
    ctx = {
        "phase": getattr(_closed_loop_rhs, "_phase", "boost"),
        "thrust": float(thrust_fn(t, {"m": m})) if thrust_fn is not None else 0.0,
        "peak_thrust": peak_thrust,
        "dry_mass": getattr(eom, "dry_mass", -1.0),
        "target_state": target_state,
    }
    phase = _current_phase(t, y, _closed_loop_rhs._events, ctx)
    _closed_loop_rhs._phase = phase

    rho = 1.225 * np.exp(-max(alt, 0.0) / 8500.0) if alt < 100e3 else 0.0

    eom_state = {"r": r, "v": v, "q": q, "omega": omega, "m": m}

    f_thrust = np.zeros(3)
    if phase == "boost":
        cmd = guidance_law.boost.commanded_gimbal(t, eom_state, rho, v)
        thrust_val = thrust_fn(t, eom_state) if thrust_fn is not None else 0.0
        # Base thrust along -x body (project convention).
        f_thrust = np.array([-thrust_val, 0.0, 0.0])
    else:
        # Midcourse / terminal guidance command an acceleration; realize it only
        # up to the thrust actually available. An unpowered (coasting) interceptor
        # cannot accelerate, so the command is inert (no fictitious force). This
        # also prevents the integrator from stiffening on large PN commands.
        avail_thrust = float(thrust_fn(t, eom_state)) if thrust_fn is not None else 0.0
        max_accel_force = avail_thrust if avail_thrust > 0.0 else 0.0
        if phase == "midcourse":
            guidance_law.midcourse.update_target(target_state, t)
            accel_cmd = guidance_law.midcourse.commanded_accel(t, eom_state)
        else:
            # Terminal guidance. If a seeker is attached (2B.1/2B.2) the UKF
            # tracks the target and supplies a smoothed LOS rate to the
            # selected terminal backend; otherwise use the analytic PN law.
            seeker = getattr(guidance_law, "seeker", None)
            if seeker is not None:
                los_rate = seeker.update_tracker(eom_state, {"r": target_r, "v": target_v})
                # 2C: when the target carries decoys, score them with the RV/decoy
                # discriminator; the interceptor prefers the contact classified as
                # the RV (the guidance law continues homing the true RV state).
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

    # Keep the multi-stage model object so the loop can recompute bus inertia/CG
    # after each separation (Phase 1B.2). The property rebuilds a fresh model,
    # which is fine since separation state is keyed by index, not mutated here.
    stage_thrust_model = None
    if getattr(interceptor, "stages", None):
        stage_thrust_model = MultiStageThrustModel(interceptor.stages, interceptor.sep_impulses)

    # Collect separation / MKV events to inject mid-integration.
    separations = getattr(interceptor, "_separations", []) or []

    # Optional multi-KV (MKVSystem) separation at terminal, if the vehicle
    # declares an MKV payload mass (1A.3).
    mkv = getattr(interceptor, "_mkv", None)

    # Event-driven stage separations (Phase 1A.3): one SeparationEvent per stage,
    # fired on physical burnout rather than a hard-coded time.
    separation_events = [
        SeparationEvent(i, stage_thrust_model, cfg)
        for i in range(len(separations))
    ] if stage_thrust_model is not None else []

    t_span = [scenario.engagement_start, scenario.engagement_end]
    target_fn = lambda t: target.propagate(t)

    # Expose the target scenario so the terminal seeker can run RV/decoy
    # discrimination (2C) and the event evaluator can read target state.
    guidance_law._target_scenario = target
    guidance_law._decoy_rejects = 0

    # Reset event-driven phase state on the RHS function object.
    _closed_loop_rhs._phase = "boost"
    _closed_loop_rhs._events = _default_events(cfg)

    integrator = RK45(
        lambda t, y: _closed_loop_rhs(t, y, interceptor, guidance_law, target_fn, eom,
                                      thrust_fn, peak_thrust),
        t_span[0], state0, min(t_span[1], cfg.t_max),
        max_step=cfg.max_step, rtol=cfg.rtol, atol=cfg.atol,
    )

    times = [integrator.t]
    states = [integrator.y.copy()]
    applied = set()

    while integrator.status == "running":
        integrator.step()
        t = integrator.t
        y = integrator.y

        # Inject separation / MKV impulses when their burnout event crosses.
        thrust_now = float(thrust_fn(t, {"m": y[13]})) if thrust_fn is not None else 0.0
        sep_ctx = {"thrust": thrust_now, "peak_thrust": peak_thrust}
        for i, sep in enumerate(separations):
            if i in applied:
                continue
            # Event-driven: fire only when the stage's SeparationEvent triggers
            # (physical burnout: past ignition+burn AND thrust ~0), so a low-thrust
            # or coasting vehicle cannot spuriously jettison a live stage.
            if (separation_events and separation_events[i].should_trigger(t, y, sep_ctx)) or (
                integrator.status != "running"
            ):
                applied.add(i)
                y = np.array(y, copy=True)
                y[13] = max(y[13] - sep.mass_drop, 1e-3)
                y[3:6] = y[3:6] + sep.impulse / max(y[13], 1e-6)
                if getattr(sep, "spin_impulse", None) is not None:
                    y[10:13] = y[10:13] + np.asarray(sep.spin_impulse) / max(y[13], 1e-6)
                # Phase 1B.2: recompute the flying-bus inertia + CG after the
                # spent stage is jettisoned so the angular EOM stays physical.
                if stage_thrust_model is not None:
                    inert = stage_thrust_model.inertia_after_separation(i)
                    if inert is not None:
                        cg = stage_thrust_model.cg_after_separation(i)
                        eom.set_inertia(inert, cg)
                # Re-inject as the integrator's current state for the next step.
                integrator.y = y

        # Multi-KV payload separation: once the event loop reaches terminal
        # phase (or burnouts), eject the KV bus and hand momentum to the KV
        # (1A.3). Guarded so it fires exactly once.
        if mkv is not None and not getattr(mkv, "separated", False) and _closed_loop_rhs._phase == "terminal":
            y = np.array(y, copy=True)
            y[13] = max(y[13] - mkv.kv_mass, 1e-3)
            y[3:6] = y[3:6] + mkv.v_rel * np.array([1.0, 0.0, 0.0])
            integrator.y = y
            mkv.separated = True

        times.append(integrator.t)
        states.append(integrator.y.copy())

        if integrator.t >= min(t_span[1], cfg.t_max):
            integrator.status = "finished"

    sol_t = np.array(times)
    sol_y = np.column_stack(states)

    r = sol_y[:3, -1]
    target_final = target_fn(min(t_span[1], cfg.t_max))
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
        "r": np.array([target_fn(ti)[:3] for ti in sol_t]),
        "v": np.array([target_fn(ti)[3:] for ti in sol_t]),
    }
    return traj, target_traj, miss, kill


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

    def run(self, n_trials: int = 50, perturbations: Optional[Dict[str, float]] = None,
            cfg: Optional[SimConfig] = None) -> EngagementResult:
        cfg = cfg or self.cfg
        if perturbations is None:
            perturbations = dict(cfg.perturbations)

        # Dedicated, seedable RNG for this run (does not advance global state).
        rng = np.random.default_rng(cfg.seed)

        nominal_traj, nominal_target_traj, nominal_miss, nominal_kill = _integrate_trajectory(
            self.interceptor, self.guidance, self.target, self.scenario,
            cfg=cfg, rng=rng,
        )

        mc_misses = []
        mc_kills = []
        mc_perturbs = []
        for _ in range(n_trials):
            _, _, miss, kill = _integrate_trajectory(
                self.interceptor, self.guidance, self.target, self.scenario,
                perturb=perturbations, cfg=cfg, rng=rng,
            )
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
            metadata={"n_trials": n_trials},
        )
