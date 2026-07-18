import numpy as np
from .coordinate_systems import quat_normalize


class StageSeparation:
    def __init__(self, time, mass_drop, impulse=None, state_jump=None):
        self.time = time
        self.mass_drop = mass_drop
        self.impulse = impulse if impulse is not None else np.zeros(3)
        self.state_jump = state_jump if state_jump is not None else {}

    def apply(self, state):
        new_state = dict(state)
        new_state["m"] = state["m"] - self.mass_drop
        for key, jump in self.state_jump.items():
            if key in new_state:
                new_state[key] = new_state[key] + jump
        return new_state


class MKVSystem:
    def __init__(self, kv_mass=15.0, divert_impulse=85.0):
        self.kv_mass = kv_mass
        self.divert_impulse = divert_impulse
        self.separated = False

    def separate(self, state):
        if self.separated:
            return state
        new_state = dict(state)
        new_state["m"] = state["m"] - self.kv_mass
        new_state["r"] = state["r"] + np.array([0.1, 0.0, 0.0])
        new_state["q"] = quat_normalize(state["q"])
        self.separated = True
        return new_state

    def divert(self, state, direction_body):
        new_state = dict(state)
        impulse = self.divert_impulse * np.asarray(direction_body, dtype=float)
        new_state["v"] = state["v"] + impulse / max(state["m"], 1e-6)
        return new_state


class ThrustModel:
    def __init__(self, thrust_profile, mass_flow, gimbal_limits=(np.radians(15), np.radians(15))):
        self.thrust_profile = thrust_profile
        self.mass_flow = mass_flow
        self.gimbal_limits = gimbal_limits
        self.separations = []

    def thrust(self, t, state):
        T = self.thrust_profile(t)
        return T

    def gimbal(self, t, commanded_angles):
        return np.clip(commanded_angles, -self.gimbal_limits, self.gimbal_limits)

    def add_separation(self, sep: StageSeparation):
        self.separations.append(sep)

    def mass_rate(self, t, state):
        return -self.mass_flow if self.thrust_profile(t) > 0 else 0.0
