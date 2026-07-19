from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from scipy.integrate import RK45

from ..dynamics.eom_6dof import EOM6DOF
from ..guidance.boost_guidance import BoostGuidance
from ..guidance.midcourse_guidance import MidcourseGuidance
from ..guidance.terminal_guidance import TerminalGuidance
from ..interceptors.config import InterceptorConfig
from ..guidance.law import GuidanceLaw
from ..scenarios.target_factory import TargetScenario
from ..scenarios.scenario import EngagementScenario
from ..aero.aero_analytical import blended_aero


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
        probs = []
        for _ in param_values:
            probs.append(self.monte_carlo.kill_probability)
        ax.plot(param_values, probs, marker="o")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Kill Probability")
        return ax


def _compute_v0(scenario: EngagementScenario, magnitude: float = 1500.0) -> np.ndarray:
    """Initial interceptor speed from launch site toward threat axis."""
    site = np.asarray(scenario.interceptor_launch_site, dtype=float)
    axis = np.asarray(scenario.threat_axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return magnitude * axis


def _surrogate(mach, alpha, beta, alt):
    return blended_aero(mach, alpha, beta, alt)[:3]


def _closed_loop_rhs(t, y, interceptor, guidance_law, target_fn, eom):
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
    if t < 60.0:
        phase = "boost"
    elif t < 180.0:
        phase = "midcourse"
    else:
        phase = "terminal"

    eom_state = {"r": r, "v": v, "q": q, "omega": omega, "m": m}

    f_thrust = np.zeros(3)
    if phase == "boost":
        rho = 1.225 * np.exp(-max(alt, 0.0) / 8500.0)
        cmd = guidance_law.boost.commanded_gimbal(t, eom_state, rho, v)
        thrust_val = interceptor.thrust_profile(t) if interceptor.thrust_profile is not None else 0.0
        f_thrust = np.array([-thrust_val, 0.0, 0.0])
    elif phase == "midcourse":
        guidance_law.midcourse.update_target(target_state, t)
        accel_cmd = guidance_law.midcourse.commanded_accel(t, eom_state)
        f_thrust = accel_cmd * m
    else:
        accel_cmd = guidance_law.terminal.commanded_accel(t, eom_state, target_state)
        f_thrust = accel_cmd * m

    derivs = eom.compute(t, eom_state, _surrogate)
    dr_dt = derivs["r"]
    dv_dt = derivs["v"] + f_thrust / max(m, 1e-6)
    dq_dt = derivs["q"]
    domega_dt = derivs["omega"]
    dm_dt = derivs["m"]

    return np.concatenate([dr_dt, dv_dt, dq_dt, domega_dt, [dm_dt]])


def _integrate_trajectory(interceptor, guidance_law, target, scenario, perturb=None):
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
            state0[:3] += np.random.normal(0, perturb["position_sigma"], 3)
        if "velocity_sigma" in perturb:
            state0[3:6] += np.random.normal(0, perturb["velocity_sigma"], 3)
        if "mass_sigma" in perturb:
            state0[13] += np.random.normal(0, perturb["mass_sigma"])

    eom = EOM6DOF(
        mass=interceptor.mass,
        inertia=interceptor.inertia,
        area=interceptor.area,
        ref_length=interceptor.ref_length,
    )

    t_span = [scenario.engagement_start, scenario.engagement_end]
    target_fn = lambda t: target.propagate(t)

    integrator = RK45(
        lambda t, y: _closed_loop_rhs(t, y, interceptor, guidance_law, target_fn, eom),
        t_span[0], state0, t_span[1], max_step=0.5, rtol=1e-6, atol=1e-9,
    )

    times = [integrator.t]
    states = [integrator.y.copy()]
    while integrator.status == "running":
        integrator.step()
        times.append(integrator.t)
        states.append(integrator.y.copy())

    sol_t = np.array(times)
    sol_y = np.column_stack(states)

    r = sol_y[:3, -1]
    target_final = target_fn(t_span[1])
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
    """End-to-end engagement simulation runner."""

    def __init__(
        self,
        interceptor: InterceptorConfig,
        guidance: GuidanceLaw,
        target: TargetScenario,
        scenario: EngagementScenario,
    ):
        self.interceptor = interceptor
        self.guidance = guidance
        self.target = target
        self.scenario = scenario

    def run(self, n_trials: int = 50, perturbations: Optional[Dict[str, float]] = None) -> EngagementResult:
        if perturbations is None:
            perturbations = {
                "position_sigma": 100.0,
                "velocity_sigma": 5.0,
                "mass_sigma": 10.0,
            }

        nominal_traj, nominal_target_traj, nominal_miss, nominal_kill = _integrate_trajectory(
            self.interceptor, self.guidance, self.target, self.scenario
        )

        if nominal_traj is None:
            nominal_traj = {"t": np.array([0.0, 1.0]), "r": np.zeros((2, 3)), "v": np.zeros((2, 3)),
                            "q": np.zeros((2, 4)), "omega": np.zeros((2, 3)), "m": np.zeros(2)}
            nominal_target_traj = {"t": np.array([0.0, 1.0]), "r": np.zeros((2, 3)), "v": np.zeros((2, 3))}

        mc_misses = []
        mc_kills = []
        mc_perturbs = []
        for _ in range(n_trials):
            _, _, miss, kill = _integrate_trajectory(
                self.interceptor, self.guidance, self.target, self.scenario, perturb=perturbations
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
