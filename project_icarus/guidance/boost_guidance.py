import numpy as np


class BoostGuidance:
    def __init__(self, pitch_over_q=15000.0, pitch_over_angle=np.radians(5.0),
                 gimbal_limits=(np.radians(15), np.radians(15)), mu=3.986004418e14,
                 r0=6371e3, g0=9.80665):
        self.pitch_over_q = pitch_over_q
        self.pitch_over_angle = pitch_over_angle
        self.gimbal_limits = np.asarray(gimbal_limits, dtype=float)
        self.mu = mu
        self.r0 = r0
        self.g0 = g0
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

    def gravity_turn_angle(self, v0_mag, r0=None):
        """Pitch-over (flight-path) angle for a gravity-turn to desired apogee.

        theta0 = arcsin[1 / (1 + v0^2 / (g0 * r0))] (flat-Earth energy estimate).
        """
        r0 = r0 if r0 is not None else self.r0
        denom = 1.0 + v0_mag**2 / max(self.g0 * r0, 1e-6)
        return np.arcsin(np.clip(1.0 / denom, -1.0, 1.0))

    def pitch_rate(self, lift_accel, mass, gamma, speed):
        """Gravity-turn flight-path rate: gamma_dot = (L/m) - g*cos(gamma)/V."""
        g_local = self.mu / max(self.r0**2, 1e-6)
        return (lift_accel / max(mass, 1e-6)) - g_local * np.cos(gamma) / max(speed, 1e-6)
