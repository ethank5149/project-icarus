import numpy as np


class TerminalGuidance:
    def __init__(self, N=4.0, accel_limit=150.0, kill_radius=0.5, mechanism="hit_to_kill",
                 noise_std=0.01, fov=np.radians(60.0), sigma0=0.01):
        self.N = N
        self.accel_limit = accel_limit
        self.kill_radius = kill_radius
        self.mechanism = mechanism
        self.noise_std = noise_std
        self.fov = fov
        self.sigma0 = sigma0
        self.seeker_range = 50e3 if mechanism == "hit_to_kill" else 20e3
        self._last_los = None

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

    def commanded_accel(self, t, interceptor_state, target_state, los_rate=None, range_=None,
                        disable_fov=False):
        if target_state is None:
            return np.zeros(3)
        r = np.asarray(interceptor_state["r"], dtype=float)
        v = np.asarray(interceptor_state["v"], dtype=float)
        tgt = np.asarray(target_state, dtype=float)
        los = tgt[:3] - r
        range_ = np.linalg.norm(los)
        los_unit = los / max(range_, 1e-6)
        if not disable_fov:
            angle = np.arccos(np.clip(los_unit[0], -1.0, 1.0))
            if angle > self.fov:
                # Out of field of view: hold last command (zero as safe default).
                return np.zeros(3)
        rel_vel = tgt[3:6] - v
        los_dot = (rel_vel - np.dot(rel_vel, los_unit) * los_unit) / max(range_, 1e-6)
        Vc = -np.dot(rel_vel, los_unit)
        a_c = self.N * Vc * los_dot
        return np.clip(a_c, -self.accel_limit, self.accel_limit)

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
