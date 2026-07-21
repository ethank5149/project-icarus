import numpy as np


class TerminalGuidance:
    """Terminal guidance with selectable guidance backends (2B.2).

    Backends (set via ``law``):
      * "pn"        classic proportional navigation (PN)
      * "apn"       augmented PN (gravity-compensated target acceleration bias)
      * "zem"       zero-effort-miss guidance (closed-form lead for ZEM)
      * "sdre_mpc"  SDRE-based MPC-lite (finite-horizon LQR on the linearized
                    relative kinematic EOM)

    The ``commanded_accel`` entry point picks the backend; ``commanded_accel_seeker``
    uses the same backend but substitutes the UKF-smoothed LOS rate.
    """

    VALID_LAWS = ("pn", "apn", "zem", "sdre_mpc")

    def __init__(self, N=4.0, accel_limit=150.0, kill_radius=0.5, mechanism="hit_to_kill",
                 noise_std=0.01, fov=np.radians(60.0), sigma0=0.01, law="pn",
                 gravity=np.array([0.0, 0.0, -9.81]), zem_horizon=5.0,
                 sdre_q_pos=1.0, sdre_q_vel=0.1, sdre_r_accel=1.0,
                 use_cython=True):
        if law not in self.VALID_LAWS:
            raise ValueError(f"Unknown terminal guidance law: {law!r} (valid: {self.VALID_LAWS})")
        self.law = law
        self.N = N
        self.accel_limit = accel_limit
        self.kill_radius = kill_radius
        self.mechanism = mechanism
        self.noise_std = noise_std
        self.fov = fov
        self.sigma0 = sigma0
        self.gravity = np.asarray(gravity, dtype=float)
        self.zem_horizon = zem_horizon
        self.sdre_q_pos = sdre_q_pos
        self.sdre_q_vel = sdre_q_vel
        self.sdre_r_accel = sdre_r_accel
        self.seeker_range = 50e3 if mechanism == "hit_to_kill" else 20e3
        self._last_los = None
        self.use_cython = use_cython
        self._cy_los = None
        self._cy_in_fov = None
        self._cy_pn_cmd = None
        self._cy_zem_cmd = None
        self._cy_sdre_mpc_cmd = None
        self._cy_midcourse_pn = None
        if use_cython:
            try:
                from project_icarus.cython_kernels.guidance_cython import (
                    los_vec as cy_los,
                    in_fov as cy_in_fov,
                    pn_cmd as cy_pn_cmd,
                    zem_cmd as cy_zem_cmd,
                    sdre_mpc_cmd as cy_sdre_mpc_cmd,
                    midcourse_pn as cy_midcourse_pn,
                )
                self._cy_los = cy_los
                self._cy_in_fov = cy_in_fov
                self._cy_pn_cmd = cy_pn_cmd
                self._cy_zem_cmd = cy_zem_cmd
                self._cy_sdre_mpc_cmd = cy_sdre_mpc_cmd
                self._cy_midcourse_pn = cy_midcourse_pn
            except Exception:
                self.use_cython = False

    # --- shared helpers ---------------------------------------------------- #
    def _los(self, interceptor_state, target_state):
        if self.use_cython and self._cy_los is not None:
            r = np.asarray(interceptor_state["r"], dtype=float)
            v = np.asarray(interceptor_state["v"], dtype=float)
            if isinstance(target_state, dict):
                tgt = np.concatenate([
                    np.asarray(target_state["r"], dtype=float),
                    np.asarray(target_state.get("v", [0, 0, 0]), dtype=float),
                ])
            else:
                tgt = np.asarray(target_state, dtype=float)
            los, range_, los_unit, rel_vel = self._cy_los(r, v, tgt[:3], tgt[3:6])
            return r, v, tgt, los, range_, los_unit, rel_vel
        r = np.asarray(interceptor_state["r"], dtype=float)
        v = np.asarray(interceptor_state["v"], dtype=float)
        if isinstance(target_state, dict):
            tgt = np.concatenate([
                np.asarray(target_state["r"], dtype=float),
                np.asarray(target_state.get("v", [0, 0, 0]), dtype=float),
            ])
        else:
            tgt = np.asarray(target_state, dtype=float)
        los = tgt[:3] - r
        range_ = float(np.linalg.norm(los))
        los_unit = los / max(range_, 1e-6)
        rel_vel = tgt[3:6] - v
        return r, v, tgt, los, range_, los_unit, rel_vel

    def _in_fov(self, los_unit):
        if self.use_cython and self._cy_in_fov is not None:
            return self._cy_in_fov(los_unit, self.fov)
        angle = np.arccos(np.clip(los_unit[0], -1.0, 1.0))
        return angle <= self.fov

    def _pn_cmd(self, Vc, los_dot, target_accel=None):
        if self.use_cython and self._cy_pn_cmd is not None:
            ta = target_accel if target_accel is not None else np.zeros(3)
            return self._cy_pn_cmd(self.N, Vc, los_dot, self.accel_limit, ta)
        a = self.N * Vc * los_dot
        if target_accel is not None:
            # Augmented PN: add (N/2) times target acceleration perpendicular to LOS.
            a = a + (self.N / 2.0) * target_accel
        return np.clip(a, -self.accel_limit, self.accel_limit)

    def _zem_cmd(self, los, rel_vel, range_):
        """Zero-effort-miss lead guidance.

        ZEM = relative position + t_go * relative velocity (linear, constant-Vc
        approximation), with t_go estimated from range / closing speed. Steers
        the interceptor to null ZEM with an effective navigation ratio.
        """
        if self.use_cython and self._cy_zem_cmd is not None:
            return self._cy_zem_cmd(los, rel_vel, range_, self.N, self.zem_horizon, self.accel_limit)
        Vc = -np.dot(rel_vel, los) / max(range_, 1e-6)
        t_go = max(range_ / max(abs(Vc), 1e-3), self.zem_horizon * 0.1)
        t_go = min(t_go, self.zem_horizon)
        zem = los + t_go * rel_vel
        los_unit = los / max(range_, 1e-6)
        zem_perp = zem - np.dot(zem, los_unit) * los_unit
        a = (self.N / max(t_go, 1e-3)) * zem_perp
        return np.clip(a, -self.accel_limit, self.accel_limit)

    def _sdre_mpc_cmd(self, los, rel_vel, range_):
        """SDRE-based MPC-lite: LQR on the linearized relative kinematic EOM.

        The relative kinematics x_dot = A(x) x + B u are linearized at the
        current state; the SDRE A(x) = [0 I; 0 0] is state-independent here so
        the gain is the standard finite-horizon LQR for double-integrator
        kinematics (closed-form for this simple plant), evaluated over the
        ZEM horizon. u = -K x with K from continuous-time ARE.
        """
        if self.use_cython and self._cy_sdre_mpc_cmd is not None:
            return self._cy_sdre_mpc_cmd(los, rel_vel, range_, self.N, self.zem_horizon,
                                         self.sdre_q_pos, self.sdre_q_vel, self.sdre_r_accel,
                                         self.accel_limit)
        Vc = -np.dot(rel_vel, los) / max(range_, 1e-6)
        t_go = max(range_ / max(abs(Vc), 1e-3), self.zem_horizon * 0.1)
        t_go = min(t_go, self.zem_horizon)
        Q = np.diag([self.sdre_q_pos] * 3 + [self.sdre_q_vel] * 3)
        R = self.sdre_r_accel * np.eye(3)
        # Continuous-time ARE for double integrator x=[p; v], A=blk(0 I;0 0):
        # closed-form gain K = [sqrt(2)*sqrt(q/r) * I,  sqrt(q/r) * I].
        kp = np.sqrt(2.0) * np.sqrt(self.sdre_q_pos / max(self.sdre_r_accel, 1e-9)) * t_go
        kv = np.sqrt(self.sdre_q_vel / max(self.sdre_r_accel, 1e-9)) * t_go
        K = np.block([[kp * np.eye(3), kv * np.eye(3)]])
        x = np.concatenate([los, rel_vel])
        a = -K @ x
        return np.clip(a, -self.accel_limit, self.accel_limit)

    def update_seeker(self, los, range_, target_visible):
        """Return filtered LOS angle (rad) and range, or None if not visible."""
        if not target_visible or los is None:
            return None, None
        noise = np.random.normal(0, self.noise_std, size=3)
        los_noisy = los + noise
        los_unit = los_noisy / max(np.linalg.norm(los_noisy), 1e-6)
        angle = np.arccos(np.clip(los_unit[0], -1.0, 1.0))
        self._last_los = (angle, range_)
        return angle, range_

    def _dispatch(self, rel_vel, los, range_, los_unit, los_dot, target_accel=None):
        Vc = -np.dot(rel_vel, los_unit)
        if self.law == "pn":
            return self._pn_cmd(Vc, los_dot, target_accel=None)
        if self.law == "apn":
            return self._pn_cmd(Vc, los_dot, target_accel=target_accel)
        if self.law == "zem":
            return self._zem_cmd(los, rel_vel, range_)
        if self.law == "sdre_mpc":
            return self._sdre_mpc_cmd(los, rel_vel, range_)
        raise ValueError(f"Unknown law {self.law!r}")

    def commanded_accel(self, t, interceptor_state, target_state, los_rate=None, range_=None,
                        disable_fov=False, target_accel=None):
        if target_state is None:
            return np.zeros(3)
        r, v, tgt, los, range_, los_unit, rel_vel = self._los(interceptor_state, target_state)
        if not disable_fov and not self._in_fov(los_unit):
            return np.zeros(3)

        # Target acceleration estimate (APN gravity compensation / bias).
        if self.law == "apn" and target_accel is None:
            target_accel = self.gravity
        else:
            target_accel = target_accel

        los_dot = (rel_vel - np.dot(rel_vel, los_unit) * los_unit) / max(range_, 1e-6)
        return self._dispatch(rel_vel, los, range_, los_unit, los_dot, target_accel)

    def commanded_accel_seeker(self, interceptor_state, target_state, los_rate=None,
                               disable_fov=False):
        """Terminal command using a UKF-smoothed LOS rate from the seeker.

        `los_rate` is a 2-vector (az dot, el dot) as produced by
        SeekerModel.update_tracker. If None, falls back to the analytic LOS rate.
        """
        if target_state is None:
            return np.zeros(3)
        r, v, tgt, los, range_, los_unit, rel_vel = self._los(interceptor_state, target_state)
        if not disable_fov and not self._in_fov(los_unit):
            return np.zeros(3)

        target_accel = self.gravity if self.law == "apn" else None

        if los_rate is not None:
            los_rate = np.asarray(los_rate, dtype=float)
            los_dot = los_rate[0] * np.array([-los_unit[1], los_unit[0], 0.0]) \
                + los_rate[1] * np.array([-los_unit[0] * los_unit[2],
                                          -los_unit[1] * los_unit[2],
                                          1.0 - los_unit[2] ** 2])
        else:
            los_dot = (rel_vel - np.dot(rel_vel, los_unit) * los_unit) / max(range_, 1e-6)
        return self._dispatch(rel_vel, los, range_, los_unit, los_dot, target_accel)

    def kill_assessment(self, miss_distance):
        if self.mechanism == "hit_to_kill":
            return miss_distance < self.kill_radius
        elif self.mechanism == "blast_frag":
            return miss_distance < 10.0
        return False

    def discrimination(self, features, likelihood_rv, likelihood_decoy):
        """Likelihood ratio discrimination from a feature vector.

        features: [RCS_bias, IR_flux, Doppler_width, micro_motion_flag]
        Returns True if the contact is more likely an RV than a decoy.
        """
        features = np.asarray(features, dtype=float)
        if len(features) < 4:
            return False
        lr = likelihood_rv(features) / max(likelihood_decoy(features), 1e-12)
        return bool(lr > 1.0)
