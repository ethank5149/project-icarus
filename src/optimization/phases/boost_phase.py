import numpy as np
import openmdao.api as om
from ...dynamics.eom_6dof import EOM6DOF
from ...guidance.boost_guidance import BoostGuidance
from ...aero.aero_analytical import blended_aero


R_EARTH = 6371e3


class BoostODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare("surrogate_path", default="aero_surrogate.pkl")
        self.options.declare("boundary_alt", default=100e3)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("r", val=np.zeros((nn, 3)))
        self.add_input("v", val=np.zeros((nn, 3)))
        self.add_input("q", val=np.zeros((nn, 4)))
        self.add_input("omega", val=np.zeros((nn, 3)))
        self.add_input("m", val=np.full(nn, 1000.0))
        self.add_input("thrust", val=np.zeros(nn))
        self.add_input("gimbal_beta", val=np.zeros(nn))
        self.add_input("gimbal_delta", val=np.zeros(nn))
        self.add_input("time", val=np.zeros(nn))

        self.add_output("dr_dt", val=np.zeros((nn, 3)))
        self.add_output("dv_dt", val=np.zeros((nn, 3)))
        self.add_output("dq_dt", val=np.zeros((nn, 4)))
        self.add_output("domega_dt", val=np.zeros((nn, 3)))
        self.add_output("dm_dt", val=np.zeros(nn))

        self.add_output("Cd", val=np.zeros(nn))
        self.add_output("Cy", val=np.zeros(nn))
        self.add_output("Cm", val=np.zeros(nn))

        self.eom = EOM6DOF(boundary_alt=self.options["boundary_alt"])
        self.guidance = BoostGuidance()

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        r = inputs["r"]
        v = inputs["v"]
        q = inputs["q"]
        omega = inputs["omega"]
        m = inputs["m"]
        thrust = inputs["thrust"]
        gimbal_beta = inputs["gimbal_beta"]
        gimbal_delta = inputs["gimbal_delta"]
        t = inputs["time"]

        boundary_alt = self.options["boundary_alt"]

        def surrogate(mach, alpha, beta, alt):
            cd, cy, cm, _, _ = blended_aero(
                mach, alpha, beta, alt, boundary_alt=boundary_alt
            )
            return cd, cy, cm

        for i in range(nn):
            state = {"r": r[i], "v": v[i], "q": q[i], "omega": omega[i], "m": float(m[i])}
            derivs = self.eom.compute(t[i], state, surrogate)
            outputs["dr_dt"][i] = derivs["r"]
            outputs["dv_dt"][i] = derivs["v"]
            outputs["dq_dt"][i] = derivs["q"]
            outputs["domega_dt"][i] = derivs["omega"]
            outputs["dm_dt"][i] = derivs["m"]

            v_inertial = np.linalg.norm(v[i])
            alt = np.linalg.norm(r[i]) - R_EARTH
            mach = v_inertial / 300.0
            alpha = np.degrees(np.arctan2(v[i, 2], v[i, 0]))
            beta = np.degrees(np.arcsin(np.clip(v[i, 1] / max(v_inertial, 1e-6), -1.0, 1.0)))
            cd, cy, cm = surrogate(mach, alpha, beta, alt)
            outputs["Cd"][i] = cd
            outputs["Cy"][i] = cy
            outputs["Cm"][i] = cm
