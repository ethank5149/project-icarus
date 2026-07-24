"""Legacy archive-only module for pre-computed Sarmat boost profiles.

Production code must use ``project_icarus.guidance.icbm_guidance.ICBMGuidance``
directly.  This module is retained only as a pickle archive; generation and
fallback paths have been removed.
"""

import os
import pickle
from typing import Optional

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

    def thrust_direction(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Interpolate pre-computed thrust direction at time t.

        Raises if the profile has not been computed.
        """
        if not self._computed or len(self._times) < 2:
            raise RuntimeError(
                "PrecomputedTrajectoryProfile not computed. "
                "Production code must use ICBMGuidance.thrust_direction(r, v, t) directly."
            )

        t = np.clip(t, self._times[0], self._times[-1])
        idx = np.searchsorted(self._times, t)
        idx = min(max(idx, 1), len(self._times) - 1)

        t0, t1 = self._times[idx - 1], self._times[idx]
        if abs(t1 - t0) < 1e-12:
            return self._thrust_dirs[idx]

        frac = (t - t0) / (t1 - t0)
        d = (1.0 - frac) * self._thrust_dirs[idx - 1] + frac * self._thrust_dirs[idx]
        return d / max(np.linalg.norm(d), 1e-9)

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

        from project_icarus.optimization.direct_trajectory_optimizer import compute_miss_distance
        from project_icarus.guidance.icbm_guidance import ICBMGuidance

        guidance = ICBMGuidance(
            target_ecef=target_ecef,
            launch_ecef=self.launch_ecef,
            use_j2=True,
        )

        traj_t = []
        traj_r = []
        traj_v = []

        rr = self.launch_ecef.astype(float).copy()
        vv = np.zeros(3)
        mm = float(mass_initial)
        tt = 0.0
        dt_integ = 0.05
        total_burn = max(s["t_end"] for s in stages)

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

            from project_icarus.scenarios.target_factory import _two_body_accel, _ground_altitude
            a_grav = _two_body_accel(rr, use_j2=True, t=tt)
            a_drag = np.zeros(3)
            alt = _ground_altitude(rr)
            if 0.0 < alt < 100e3:
                from project_icarus.dynamics.atmosphere import Atmosphere
                rho = Atmosphere().density_scalar(alt)
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

            if not np.all(np.isfinite(rr)) or not np.all(np.isfinite(vv)) or not np.isfinite(mm):
                break
            if mm < 1e-3:
                break
            if all(tt >= s["t_end"] for s in stages):
                break

        if len(traj_t) < 10:
            return profile

        t_out = np.arange(0.0, min(total_burn, traj_t[-1]), profile.dt)
        if len(t_out) < 10:
            return profile

        traj_r = np.array(traj_r)
        traj_v = np.array(traj_v)
        r_out = np.array([np.interp(t_out, traj_t, traj_r[:, i]) for i in range(3)]).T
        v_out = np.array([np.interp(t_out, traj_t, traj_v[:, i]) for i in range(3)]).T

        thrust_dirs_out = []
        for idx, ti in enumerate(t_out):
            rr = r_out[idx]
            vv = v_out[idx]
            thrust_dirs_out.append(guidance.thrust_direction(rr, vv, ti))

        profile._times = t_out[: profile.num_nodes]
        profile._states = np.hstack([r_out, v_out])[: profile.num_nodes]
        profile._thrust_dirs = np.array(thrust_dirs_out)[: profile.num_nodes]
        profile._computed = True
        profile.save(self.profiles_dir)
        return profile
