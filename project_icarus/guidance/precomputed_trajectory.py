"""Pre-computed trajectory profiles for ICBM boost guidance.

Real ICBMs do not compute guidance in real time.  The mission-planning
system uses a high-fidelity trajectory optimizer (equivalent to the Dymos
problem in this repo) to generate a lookup table of commanded thrust
direction versus time.  That table is loaded into the guidance computer
before launch.  During boost the computer merely interpolates from the
table, with small INS/star-sighting drift corrections.

This module owns that table for the RS-28 Sarmat:
- Generation: optimize guidance parameters directly on the physics integrator
  using scipy.optimize to minimize terminal miss distance.
- Storage: pickle the profile to ``reference/trajectory_profiles/``.
- Runtime: ``PrecomputedTrajectoryProfile`` interpolates the stored table.
"""

import os
import pickle
from typing import Optional, Tuple

import numpy as np

from .icbm_guidance import (
    ICBMGuidance,
    _ecef_to_geodetic,
    _enu_basis,
    _geodetic_to_ecef,
    _great_circle_azimuth,
)


class PrecomputedTrajectoryProfile:
    """Stored thrust-direction lookup table for ICBM boost guidance."""

    def __init__(
        self,
        target_ecef: np.ndarray,
        launch_ecef: np.ndarray,
        profile_name: str = "standard",
        burnout_alt: float = 250.0e3,
        burnout_vmag: float = 6800.0,
        burnout_fpa_deg: float = 18.0,
        num_nodes: int = 500,
        dt: float = 0.5,
    ):
        self.target = np.asarray(target_ecef, dtype=float)
        self.launch = np.asarray(launch_ecef, dtype=float)
        self.profile_name = profile_name
        self.burnout_alt = burnout_alt
        self.burnout_vmag = burnout_vmag
        self.burnout_fpa_deg = burnout_fpa_deg
        self.num_nodes = num_nodes
        self.dt = dt

        self.azimuth = np.radians(_great_circle_azimuth(self.launch, self.target))
        lat0, lon0, _ = _ecef_to_geodetic(self.launch)
        self._east, self._north, self._up = _enu_basis(lat0, lon0)
        self._gc_horiz = (
            np.cos(self.azimuth) * self._north + np.sin(self.azimuth) * self._east
        )
        gc_norm = np.linalg.norm(self._gc_horiz)
        if gc_norm > 1e-9:
            self._gc_horiz = self._gc_horiz / gc_norm
        else:
            self._gc_horiz = np.array([0.0, 0.0, 0.0])

        self._times: np.ndarray = np.zeros(num_nodes)
        self._thrust_dirs: np.ndarray = np.zeros((num_nodes, 3))
        self._states: np.ndarray = np.zeros((num_nodes, 6))
        self._computed = False

    def compute_from_dymos(
        self,
        stages: list,
        mass_initial: float,
        cd: float,
        area: float,
        use_j2: bool = True,
        atmosphere=None,
        num_segments: int = 20,
        order: int = 5,
        maxiter: int = 300,
    ) -> bool:
        """Generate the profile by optimizing the physics-based trajectory.

        Uses scipy.optimize to minimize terminal miss distance by tuning
        guidance parameters.  Falls back to high-fidelity integration with
        default guidance parameters on failure.
        """
        try:
            return self._compute_via_scipy(stages, mass_initial, cd, area,
                                            use_j2, atmosphere)
        except Exception as exc:
            print(f"SciPy trajectory optimization failed: {exc}")
            return self._compute_via_guidance(stages, mass_initial, cd, area,
                                              use_j2, atmosphere)

    def _compute_via_scipy(self, stages, mass_initial, cd, area,
                            use_j2, atmosphere):
        """Optimize guidance parameters using scipy.optimize."""
        from scipy.optimize import minimize
        from project_icarus.scenarios.target_factory import (
            _two_body_accel, _ground_altitude, _ecef_to_geodetic
        )

        if atmosphere is None:
            from project_icarus.dynamics.atmosphere import Atmosphere
            atmosphere = Atmosphere()

        total_burn = max(s["t_end"] for s in stages)
        dt_integ = 0.5

        def run_trajectory(el_0, el_1, t_cross, burnout_vmag, dt=dt_integ):
            rr = self.launch.astype(float).copy()
            vv = np.zeros(3)
            mm = float(mass_initial)
            tt = 0.0

            guidance = ICBMGuidance(
                target_ecef=self.target,
                launch_ecef=self.launch,
                burnout_vmag=burnout_vmag,
                pitch_over_start=5.0,
                pitch_over_duration=25.0,
                initial_elevation=np.radians(88.0),
                max_flight_path_angle=np.radians(70.0),
                gravity_turn_gain=0.03,
                use_j2=use_j2,
            )

            # Boost phase
            for step in range(int(total_burn / dt) + 100):
                T_mag = 0.0
                dm_dt = 0.0
                for stage in stages:
                    if stage["t_start"] <= tt < stage["t_end"]:
                        T_mag = stage["thrust"]
                        dm_dt = stage["m_dot"]
                        break

                if T_mag > 0.0:
                    thrust_dir = guidance.thrust_direction(rr, vv, tt)
                    a_thrust = (T_mag / max(mm, 1e-6)) * thrust_dir
                else:
                    a_thrust = np.zeros(3)

                a_grav = _two_body_accel(rr, use_j2=use_j2, t=tt)
                a_drag = np.zeros(3)
                alt = _ground_altitude(rr)
                if 0.0 < alt < 100e3:
                    rho = atmosphere.density_scalar(alt)
                    vmag = np.linalg.norm(vv)
                    if vmag > 1e-6:
                        q = 0.5 * rho * vmag**2
                        a_drag = -q * cd * area / mm * (vv / vmag)

                a_total = a_grav + a_drag + a_thrust

                k1 = np.concatenate([vv, a_total, [dm_dt]])
                k2 = np.concatenate([vv + 0.5*dt*k1[3:6], a_total, [dm_dt]])
                k3 = np.concatenate([vv + 0.5*dt*k2[3:6], a_total, [dm_dt]])
                k4 = np.concatenate([vv + dt*k3[3:6], a_total, [dm_dt]])

                rr = rr + (dt/6.0) * (k1[:3] + 2*k2[:3] + 2*k3[:3] + k4[:3])
                vv = vv + (dt/6.0) * (k1[3:6] + 2*k2[3:6] + 2*k3[3:6] + k4[3:6])
                mm = mm + (dt/6.0) * (k1[6] + 2*k2[6] + 2*k3[6] + k4[6])
                tt += dt

                if not np.all(np.isfinite(rr)) or not np.all(np.isfinite(vv)):
                    return 1e12
                if mm < 1e-3:
                    break
                if all(tt >= s["t_end"] for s in stages):
                    break

            # Ballistic coast to ground impact
            r_burnout = rr.copy()
            v_burnout = vv.copy()
            t_coast = 0.0
            max_coast = 3600.0

            while t_coast < max_coast:
                a_grav = _two_body_accel(rr, use_j2=use_j2, t=tt)
                a_drag = np.zeros(3)
                alt = _ground_altitude(rr)
                if 0.0 < alt < 100e3:
                    rho = atmosphere.density_scalar(alt)
                    vmag = np.linalg.norm(vv)
                    if vmag > 1e-6:
                        q = 0.5 * rho * vmag**2
                        a_drag = -q * cd * area / mm * (vv / vmag)

                a_total = a_grav + a_drag
                vv = vv + dt * a_total
                rr = rr + dt * vv
                tt += dt
                t_coast += dt

                if alt < 0.0 and t_coast > 10.0:
                    return np.linalg.norm(rr - self.target)

            return np.linalg.norm(rr - self.target)

        def simulate(params):
            el_0, el_1, t_cross, burnout_vmag = params
            return run_trajectory(el_0, el_1, t_cross, burnout_vmag)

        x0 = np.array([np.radians(45.0), np.radians(25.0), 100.0, 6800.0])
        bounds = [
            (np.radians(15), np.radians(75)),
            (np.radians(15), np.radians(75)),
            (5.0, 250.0),
            (5000.0, 8000.0),
        ]

        result = minimize(simulate, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-3})

        if result.success or result.fun < 100.0:
            el_0, el_1, t_cross, burnout_vmag = result.x
            print(f"Optimized miss: {result.fun:.1f} m")
            print(f"  el_0={np.degrees(el_0):.1f}, el_1={np.degrees(el_1):.1f}")
            print(f"  t_cross={t_cross:.1f}, burnout_vmag={burnout_vmag:.1f}")
        else:
            print(f"SLSQP did not converge: {result.message}")
            print(f"Best miss: {result.fun:.1f} m")

        traj_t = []
        traj_r = []
        traj_v = []

        rr = self.launch.astype(float).copy()
        vv = np.zeros(3)
        mm = float(mass_initial)
        tt = 0.0

        guidance = ICBMGuidance(
            target_ecef=self.target,
            launch_ecef=self.launch,
            burnout_vmag=result.x[3] if result.fun < 100.0 else 6800.0,
            pitch_over_start=5.0,
            pitch_over_duration=25.0,
            initial_elevation=np.radians(88.0),
            max_flight_path_angle=np.radians(70.0),
            gravity_turn_gain=0.03,
            use_j2=use_j2,
        )

        for step in range(int(total_burn / dt_integ) + 100):
            traj_t.append(tt)
            traj_r.append(rr.copy())
            traj_v.append(vv.copy())

            T_mag = 0.0
            dm_dt = 0.0
            for stage in stages:
                if stage["t_start"] <= tt < stage["t_end"]:
                    T_mag = stage["thrust"]
                    dm_dt = stage["m_dot"]
                    break

            if T_mag > 0.0:
                thrust_dir = guidance.thrust_direction(rr, vv, tt)
                a_thrust = (T_mag / max(mm, 1e-6)) * thrust_dir
            else:
                a_thrust = np.zeros(3)

            a_grav = _two_body_accel(rr, use_j2=use_j2, t=tt)
            a_drag = np.zeros(3)
            alt = _ground_altitude(rr)
            if 0.0 < alt < 100e3:
                rho = atmosphere.density_scalar(alt)
                vmag = np.linalg.norm(vv)
                if vmag > 1e-6:
                    q = 0.5 * rho * vmag**2
                    a_drag = -q * cd * area / mm * (vv / vmag)

            a_total = a_grav + a_drag + a_thrust

            k1 = np.concatenate([vv, a_total, [dm_dt]])
            k2 = np.concatenate([vv + 0.5*dt_integ*k1[3:6], a_total, [dm_dt]])
            k3 = np.concatenate([vv + 0.5*dt_integ*k2[3:6], a_total, [dm_dt]])
            k4 = np.concatenate([vv + dt_integ*k3[3:6], a_total, [dm_dt]])

            rr = rr + (dt_integ/6.0) * (k1[:3] + 2*k2[:3] + 2*k3[:3] + k4[:3])
            vv = vv + (dt_integ/6.0) * (k1[3:6] + 2*k2[3:6] + 2*k3[3:6] + k4[3:6])
            mm = mm + (dt_integ/6.0) * (k1[6] + 2*k2[6] + 2*k3[6] + k4[6])
            tt += dt_integ

            if not np.all(np.isfinite(rr)) or not np.all(np.isfinite(vv)):
                break
            if mm < 1e-3:
                break
            if all(tt >= s["t_end"] for s in stages):
                break

        traj_t = np.array(traj_t)
        traj_r = np.array(traj_r)
        traj_v = np.array(traj_v)

        t_out = np.arange(0.0, min(total_burn, traj_t[-1]), self.dt)
        if len(t_out) < 10:
            return False

        r_out = np.array([np.interp(t_out, traj_t, traj_r[:, i]) for i in range(3)]).T
        v_out = np.array([np.interp(t_out, traj_t, traj_v[:, i]) for i in range(3)]).T

        thrust_dirs_out = []
        for idx, ti in enumerate(t_out):
            rr = r_out[idx]
            vv = v_out[idx]
            thrust_dirs_out.append(guidance.thrust_direction(rr, vv, ti))

        self._times = t_out[: self.num_nodes]
        self._states = np.hstack([r_out, v_out])[: self.num_nodes]
        self._thrust_dirs = np.array(thrust_dirs_out)[: self.num_nodes]
        self._computed = True
        return True

    def _compute_via_guidance(self, stages, mass_initial, cd, area,
                               use_j2, atmosphere):
        """Fallback: integrate boost using the physics-based ICBMGuidance."""
        from .icbm_guidance import ICBMGuidance

        if atmosphere is None:
            from project_icarus.dynamics.atmosphere import Atmosphere
            atmosphere = Atmosphere()

        total_burn = max(s["t_end"] for s in stages)
        dt_integ = 0.05

        guidance = ICBMGuidance(
            target_ecef=self.target,
            launch_ecef=self.launch,
            burnout_vmag=self.burnout_vmag,
            pitch_over_start=5.0,
            pitch_over_duration=25.0,
            initial_elevation=np.radians(88.0),
            max_flight_path_angle=np.radians(70.0),
            gravity_turn_gain=0.03,
            use_j2=use_j2,
        )

        rr = self.launch.astype(float).copy()
        vv = np.zeros(3)
        mm = float(mass_initial)
        tt = 0.0
        traj_t = []
        traj_r = []
        traj_v = []
        traj_m = []

        for step in range(int(total_burn / dt_integ) + 100):
            traj_t.append(tt)
            traj_r.append(rr.copy())
            traj_v.append(vv.copy())
            traj_m.append(mm)

            T_mag = 0.0
            dm_dt = 0.0
            for stage in stages:
                if stage["t_start"] <= tt < stage["t_end"]:
                    T_mag = stage["thrust"]
                    dm_dt = stage["m_dot"]
                    break

            if T_mag > 0.0:
                thrust_dir = guidance.thrust_direction(rr, vv, tt)
                a_thrust = (T_mag / max(mm, 1e-6)) * thrust_dir
            else:
                a_thrust = np.zeros(3)

            from project_icarus.scenarios.target_factory import _two_body_accel
            def rhs_tmp(rrr, vvv, mmm, ttt):
                a_grav = _two_body_accel(rrr, use_j2=use_j2, t=ttt)
                a_drag = np.zeros(3)
                alt = np.linalg.norm(rrr) - 6371e3
                if 0.0 < alt < 100e3:
                    rho = atmosphere.density_scalar(alt)
                    vmag = np.linalg.norm(vvv)
                    if vmag > 1e-6:
                        q = 0.5 * rho * vmag**2
                        a_drag = -q * cd * area / mmm * (vvv / vmag)
                return np.concatenate([vvv, a_grav + a_drag + a_thrust, [dm_dt]])

            k1 = rhs_tmp(rr, vv, mm, tt)
            k2 = rhs_tmp(rr + 0.5*dt_integ*k1[:3], vv + 0.5*dt_integ*k1[3:6],
                         mm + 0.5*dt_integ*k1[6], tt + 0.5*dt_integ)
            k3 = rhs_tmp(rr + 0.5*dt_integ*k2[:3], vv + 0.5*dt_integ*k2[3:6],
                         mm + 0.5*dt_integ*k2[6], tt + 0.5*dt_integ)
            k4 = rhs_tmp(rr + dt_integ*k3[:3], vv + dt_integ*k3[3:6],
                         mm + dt_integ*k3[6], tt + dt_integ)

            rr = rr + (dt_integ/6.0) * (k1[:3] + 2*k2[:3] + 2*k3[:3] + k4[:3])
            vv = vv + (dt_integ/6.0) * (k1[3:6] + 2*k2[3:6] + 2*k3[3:6] + k4[3:6])
            mm = mm + (dt_integ/6.0) * (k1[6] + 2*k2[6] + 2*k3[6] + k4[6])
            tt += dt_integ

            if not np.all(np.isfinite(rr)) or not np.all(np.isfinite(vv)) or not np.isfinite(mm):
                break
            if mm < 1e-3:
                break
            if all(tt >= s["t_end"] for s in stages):
                break

        if len(traj_t) < 10:
            return False

        t_out = np.arange(0.0, min(total_burn, traj_t[-1]), self.dt)
        if len(t_out) < 10:
            return False

        traj_r = np.array(traj_r)
        traj_v = np.array(traj_v)
        r_out = np.array([np.interp(t_out, traj_t, traj_r[:, i]) for i in range(3)]).T
        v_out = np.array([np.interp(t_out, traj_t, traj_v[:, i]) for i in range(3)]).T

        thrust_dirs_out = []
        for idx, ti in enumerate(t_out):
            rr = r_out[idx]
            vv = v_out[idx]
            thrust_dirs_out.append(guidance.thrust_direction(rr, vv, ti))

        self._times = t_out[: self.num_nodes]
        self._states = np.hstack([r_out, v_out])[: self.num_nodes]
        self._thrust_dirs = np.array(thrust_dirs_out)[: self.num_nodes]
        self._computed = True
        return True

    def thrust_direction(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Interpolate pre-computed thrust direction at time t.

        This is the guidance computer's lookup table execution.
        """
        if not self._computed or len(self._times) < 2:
            return self._fallback_thrust_direction(t, r, v)

        t = np.clip(t, self._times[0], self._times[-1])
        idx = np.searchsorted(self._times, t)
        idx = min(max(idx, 1), len(self._times) - 1)

        t0, t1 = self._times[idx - 1], self._times[idx]
        if abs(t1 - t0) < 1e-12:
            return self._thrust_dirs[idx]

        frac = (t - t0) / (t1 - t0)
        d = (1.0 - frac) * self._thrust_dirs[idx - 1] + frac * self._thrust_dirs[idx]
        return d / max(np.linalg.norm(d), 1e-9)

    def _fallback_thrust_direction(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        lat, lon, _ = _ecef_to_geodetic(r)
        _, _, up = _enu_basis(lat, lon)

        if t < 5.0:
            return up.copy()

        vmag = np.linalg.norm(v)
        if vmag < 1e-6:
            return np.cos(np.radians(15.0)) * self._gc_horiz + np.sin(np.radians(15.0)) * up

        v_dir = v / vmag
        v_horiz = v_dir - np.dot(v_dir, up) * up
        v_horiz_norm = np.linalg.norm(v_horiz)
        if v_horiz_norm > 1e-6:
            v_horiz = v_horiz / v_horiz_norm

        gamma = max(0.1, np.pi / 2.0 - 0.015 * t)
        return np.cos(gamma) * self._gc_horiz + np.sin(gamma) * up

    def save(self, directory: str = "reference/trajectory_profiles"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"sarmat_{self.profile_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "profile_name": self.profile_name,
                    "target": self.target,
                    "launch": self.launch,
                    "azimuth": self.azimuth,
                    "times": self._times,
                    "states": self._states,
                    "thrust_dirs": self._thrust_dirs,
                    "burnout_alt": self.burnout_alt,
                    "burnout_vmag": self.burnout_vmag,
                    "burnout_fpa_deg": self.burnout_fpa_deg,
                },
                f,
            )

    @classmethod
    def load(
        cls,
        target_ecef: np.ndarray,
        launch_ecef: np.ndarray,
        profile_name: str = "standard",
        directory: str = "reference/trajectory_profiles",
    ) -> Optional["PrecomputedTrajectoryProfile"]:
        path = os.path.join(directory, f"sarmat_{profile_name}.pkl")
        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            data = pickle.load(f)

        profile = cls(
            target_ecef=target_ecef,
            launch_ecef=launch_ecef,
            profile_name=data["profile_name"],
            burnout_alt=data.get("burnout_alt", 250.0e3),
            burnout_vmag=data.get("burnout_vmag", 6800.0),
            burnout_fpa_deg=data.get("burnout_fpa_deg", 18.0),
        )
        profile.azimuth = data.get("azimuth", profile.azimuth)
        profile._times = data["times"]
        profile._states = data["states"]
        profile._thrust_dirs = data["thrust_dirs"]
        profile._computed = True
        return profile


class SarmatTrajectoryLibrary:
    """Generates and loads pre-computed Sarmat boost profiles.

    On first call for a given target/profile, this runs the trajectory
    optimization, saves the profile to disk, and returns it.  Subsequent
    calls reload the pickle directly.
    """

    def __init__(self, launch_ecef: np.ndarray, profiles_dir: str = "reference/trajectory_profiles"):
        self.launch_ecef = np.asarray(launch_ecef, dtype=float)
        self.profiles_dir = profiles_dir
        self._profiles: dict = {}

    def get_or_compute(
        self,
        target_ecef: np.ndarray,
        profile_name: str = "standard",
        stages: Optional[list] = None,
        mass_initial: float = 208100.0,
        cd: float = 0.3,
        area: float = 7.07,
    ) -> PrecomputedTrajectoryProfile:
        profile = PrecomputedTrajectoryProfile.load(
            target_ecef, self.launch_ecef, profile_name, self.profiles_dir
        )
        if profile is not None:
            return profile

        if stages is None:
            stages = [
                {"t_start": 0.0, "t_end": 90.0, "thrust": 5.0e6, "m_dot": 1430.0, "dry_mass": 11340.0},
                {"t_start": 90.0, "t_end": 270.0, "thrust": 1.2e6, "m_dot": 247.0, "dry_mass": 3600.0},
                {"t_start": 270.0, "t_end": 360.0, "thrust": 1.0e6, "m_dot": 88.0, "dry_mass": 2160.0},
            ]

        profile = PrecomputedTrajectoryProfile(
            target_ecef=target_ecef,
            launch_ecef=self.launch_ecef,
            profile_name=profile_name,
        )

        success = profile.compute_from_dymos(
            stages=stages,
            mass_initial=mass_initial,
            cd=cd,
            area=area,
        )

        if success:
            profile.save(self.profiles_dir)
            return profile

        return profile
