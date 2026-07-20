import numpy as np
from openmdao.api import ExplicitComponent


MU_EARTH = 3.986004418e14


class InterceptConstraint(ExplicitComponent):
    def initialize(self):
        self.options.declare("boundary_alt", default=100e3)
        self.options.declare("max_q", default=50000.0)
        self.options.declare("max_alpha", default=np.radians(15.0))
        self.options.declare("kill_radius", default=0.5)

    def setup(self):
        self.add_input("terminal_r", val=np.zeros(3))
        self.add_input("terminal_v", val=np.zeros(3))
        self.add_input("target_r", val=np.zeros(3))
        self.add_input("target_v", val=np.zeros(3))
        self.add_input("max_dyn_pressure_in", val=0.0)
        self.add_input("max_alpha_in", val=0.0)
        self.add_output("miss_distance", val=100.0)
        self.add_output("intercept_altitude", val=0.0)
        self.add_output("max_dyn_pressure", val=0.0)
        self.add_output("max_alpha", val=0.0)
        self.add_output("kill_confirmed", val=0.0)

    def compute(self, inputs, outputs):
        miss = np.linalg.norm(inputs["terminal_r"] - inputs["target_r"])
        alt = np.linalg.norm(inputs["terminal_r"]) - 6371e3
        outputs["miss_distance"] = miss
        outputs["intercept_altitude"] = alt
        outputs["max_dyn_pressure"] = inputs["max_dyn_pressure_in"]
        outputs["max_alpha"] = inputs["max_alpha_in"]
        outputs["kill_confirmed"] = 1.0 if miss < self.options["kill_radius"] else 0.0


class QuatNormConstraint(ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input("q", val=np.array([1.0, 0.0, 0.0, 0.0]))
        self.add_output("q_norm", val=1.0)

    def compute(self, inputs, outputs):
        outputs["q_norm"] = np.linalg.norm(inputs["q"]) - 1.0


class ControlSaturation(ExplicitComponent):
    def initialize(self):
        self.options.declare("accel_limit", default=150.0)
        self.options.declare("gimbal_limit", default=np.radians(15))

    def setup(self):
        self.add_input("accel_cmd", val=np.zeros(3))
        self.add_input("gimbal_cmd", val=np.zeros(2))
        self.add_output("accel_sat", val=0.0)
        self.add_output("gimbal_sat", val=0.0)

    def compute(self, inputs, outputs):
        outputs["accel_sat"] = np.max(np.abs(inputs["accel_cmd"])) / max(self.options["accel_limit"], 1e-6)
        outputs["gimbal_sat"] = np.max(np.abs(inputs["gimbal_cmd"])) / max(self.options["gimbal_limit"], 1e-6)


def add_mission_constraints(prob, kill_radius=0.5, max_q=50000.0, max_alpha=np.radians(15.0),
                            alt_min=100e3, alt_max=200e3, homing_time=10.0):
    """Add intercept feasibility constraints to an OpenMDAO problem."""
    prob.model.add_constraint("intercept_altitude", lower=alt_min, upper=alt_max)
    prob.model.add_constraint("max_dyn_pressure", upper=max_q)
    prob.model.add_constraint("max_alpha", upper=max_alpha)
    prob.model.add_constraint("homing_time", lower=homing_time)
    return prob
