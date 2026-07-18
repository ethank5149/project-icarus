import numpy as np
import openmdao.api as om
from ..dynamics.eom_6dof import EOM6DOF
from ..guidance.boost_guidance import BoostGuidance
from ..surrogates.aero_surrogate import AeroSurrogateComponent


class BoostODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("surrogate_path", default="aero_surrogate.pkl")
        self.options.declare("boundary_alt", default=100e3)

    def setup(self):
        self.add_input("r", val=np.zeros(3))
        self.add_input("v", val=np.zeros(3))
        self.add_input("q", val=np.array([1.0, 0.0, 0.0, 0.0]))
        self.add_input("omega", val=np.zeros(3))
        self.add_input("m", val=1000.0)
        self.add_input("thrust", val=0.0)
        self.add_input("gimbal_beta", val=0.0)
        self.add_input("gimbal_delta", val=0.0)
        self.add_input("time", val=0.0)

        self.add_output("dr_dt", val=np.zeros(3))
        self.add_output("dv_dt", val=np.zeros(3))
        self.add_output("dq_dt", val=np.array([0.0, 0.0, 0.0, 0.0]))
        self.add_output("domega_dt", val=np.zeros(3))
        self.add_output("dm_dt", val=0.0)

        self.add_output("Cd", val=0.0)
        self.add_output("Cl", val=0.0)
        self.add_output("Cm", val=0.0)

        self.eom = EOM6DOF(boundary_alt=self.options["boundary_alt"])
        self.guidance = BoostGuidance()

    def compute(self, inputs, outputs):
        r = inputs["r"]
        v = inputs["v"]
        q = inputs["q"]
        omega = inputs["omega"]
        m = inputs["m"]
        t = inputs["time"]
        thrust = inputs["thrust"]
        gimbal = np.array([inputs["gimbal_beta"], inputs["gimbal_delta"]])

        state = {"r": r, "v": v, "q": q, "omega": omega, "m": m}
        v_inertial = np.linalg.norm(v)
        rho = 1.225 * np.exp(-np.linalg.norm(r) / 8500.0)

        def surrogate(mach, alpha, beta, alt):
            return 0.05 + 0.1 * mach**2, 0.5 * alpha, 0.0

        derivs = self.eom.compute(t, state, surrogate)
        outputs["dr_dt"] = derivs["r"]
        outputs["dv_dt"] = derivs["v"]
        outputs["dq_dt"] = derivs["q"]
        outputs["domega_dt"] = derivs["omega"]
        outputs["dm_dt"] = derivs["m"]

        alt = np.linalg.norm(r) - 6371e3
        mach = v_inertial / 300.0
        alpha = np.degrees(np.arctan2(v[2], v[0]))
        beta = np.degrees(np.arcsin(np.clip(v[1] / max(v_inertial, 1e-6), -1.0, 1.0)))
        cd, cl, cm = surrogate(mach, alpha, beta, alt)
        outputs["Cd"] = cd
        outputs["Cl"] = cl
        outputs["Cm"] = cm
