import numpy as np
import openmdao.api as om
from ..dynamics.eom_6dof import EOM6DOF
from ..guidance.midcourse_guidance import MidcourseGuidance


class MidcourseODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("boundary_alt", default=100e3)

    def setup(self):
        self.add_input("r", val=np.zeros(3))
        self.add_input("v", val=np.zeros(3))
        self.add_input("q", val=np.array([1.0, 0.0, 0.0, 0.0]))
        self.add_input("omega", val=np.zeros(3))
        self.add_input("m", val=100.0)
        self.add_input("accel_x", val=0.0)
        self.add_input("accel_y", val=0.0)
        self.add_input("accel_z", val=0.0)
        self.add_input("time", val=0.0)

        self.add_output("dr_dt", val=np.zeros(3))
        self.add_output("dv_dt", val=np.zeros(3))
        self.add_output("dq_dt", val=np.array([0.0, 0.0, 0.0, 0.0]))
        self.add_output("domega_dt", val=np.zeros(3))
        self.add_output("dm_dt", val=0.0)

        self.eom = EOM6DOF(boundary_alt=self.options["boundary_alt"])
        self.guidance = MidcourseGuidance()

    def compute(self, inputs, outputs):
        r = inputs["r"]
        v = inputs["v"]
        q = inputs["q"]
        omega = inputs["omega"]
        m = inputs["m"]
        t = inputs["time"]
        accel = np.array([inputs["accel_x"], inputs["accel_y"], inputs["accel_z"]])

        state = {"r": r, "v": v, "q": q, "omega": omega, "m": m}

        def surrogate(mach, alpha, beta, alt):
            return 0.05 + 0.1 * mach**2, 0.5 * alpha, 0.0

        derivs = self.eom.compute(t, state, surrogate)
        outputs["dr_dt"] = derivs["r"]
        outputs["dv_dt"] = derivs["v"] + accel / max(m, 1e-6)
        outputs["dq_dt"] = derivs["q"]
        outputs["domega_dt"] = derivs["omega"]
        outputs["dm_dt"] = derivs["m"]
