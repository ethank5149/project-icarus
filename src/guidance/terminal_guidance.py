import numpy as np


class BoostGuidance:
    def __init__(self, pitch_over_q=15000.0, pitch_over_angle=np.radians(5.0), gimbal_limits=np.radians(15)):
        self.pitch_over_q = pitch_over_q
        self.pitch_over_angle = pitch_over_angle
        self.gimbal_limits = gimbal_limits
        self.pitched = False

    def commanded_gimbal(self, t, state, rho, v_inertial):
        q_dyn = 0.5 * rho * np.dot(v_inertial, v_inertial)
        if not self.pitched and q_dyn > self.pitch_over_q:
            self.pitched = True
        if self.pitched:
            angle = self.pitch_over_angle
        else:
            angle = 0.0
        return np.clip([angle, 0.0], -self.gimbal_limits, self.gimbal_limits)


class MidcourseGuidance:
    def __init__(self, N=5.0, accel_limit=50.0, update_interval=2.0):
        self.N = N
        self.accel_limit = accel_limit
        self.update_interval = update_interval
        self.target_state = None

    def update_target(self, target_state):
        self.target_state = np.asarray(target_state, dtype=float)

    def commanded_accel(self, t, interceptor_state, los_rate, range_):
        if self.target_state is None:
            return np.zeros(3)
        r = interceptor_state["r"]
        v = interceptor_state["v"]
        los = self.target_state[:3] - r
        los_unit = los / max(np.linalg.norm(los), 1e-6)
        rel_vel = self.target_state[3:6] - v
        closing = -np.dot(rel_vel, los_unit)
        accel = self.N * closing * np.cross(rel_vel, los_unit)
        accel = np.clip(accel, -self.accel_limit, self.accel_limit)
        return accel


class TerminalGuidance:
    def __init__(self, N=4.0, accel_limit=150.0, kill_radius=0.5, mechanism="hit_to_kill", noise_std=0.01):
        self.N = N
        self.accel_limit = accel_limit
        self.kill_radius = kill_radius
        self.mechanism = mechanism
        self.noise_std = noise_std
        self.seeker_range = 50e3 if mechanism == "hit_to_kill" else 20e3

    def update_seeker(self, los, range_, target_visible):
        noise = np.random.normal(0, self.noise_std, size=3)
        return los + noise if target_visible else None

    def commanded_accel(self, t, interceptor_state, target_state, los_rate, range_):
        if target_state is None:
            return np.zeros(3)
        r = interceptor_state["r"]
        v = interceptor_state["v"]
        los = target_state[:3] - r
        range_ = np.linalg.norm(los)
        los_unit = los / max(range_, 1e-6)
        rel_vel = target_state[3:6] - v
        closing = -np.dot(rel_vel, los_unit)
        if self.mechanism == "blast_frag":
            accel = 3.0 * self.N * closing * np.cross(rel_vel, los_unit)
        else:
            accel = self.N * closing * np.cross(rel_vel, los_unit)
        accel = np.clip(accel, -self.accel_limit, self.accel_limit)
        return accel

    def kill_assessment(self, miss_distance):
        if self.mechanism == "hit_to_kill":
            return miss_distance < self.kill_radius
        elif self.mechanism == "blast_frag":
            return miss_distance < 10.0
        return False

    def discrimination(self, target_state, decoy_states, seeker_snr):
        scores = []
        for i, tgt in enumerate([target_state] + decoy_states):
            range_ = np.linalg.norm(tgt[:3])
            rr = np.dot(tgt[3:6], tgt[:3] / max(range_, 1e-6))
            scores.append(seeker_snr * rr / max(range_, 1e-3))
        best = int(np.argmax(scores))
        return best == 0
