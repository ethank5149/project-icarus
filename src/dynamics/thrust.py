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
        # Spring ejection: relative velocity along bus +x (body frame).
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
        # First-order gimbal rate limit (per-step approximation).
        max_step = self.gimbal_rate
        clamped = np.clip(cmd, self._gimbal - max_step, self._gimbal + max_step)
        self._gimbal = clamped
        return clamped

    def thrust_vector(self, t, state, gimbal_angles=None):
        T = self.thrust(t, state)
        if gimbal_angles is None:
            gimbal_angles = self._gimbal
        pitch, yaw = gimbal_angles
        # Rotate base [-x body] thrust by gimbal pitch/yaw.
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
