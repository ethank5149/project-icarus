import numpy as np
import openmdao.api as om
from ...dynamics.eom_6dof import EOM6DOF
from ...guidance.midcourse_guidance import MidcourseGuidance
from ...aero.aero_analytical import blended_aero
from ...surrogates.train_gpr import load_vehicle_gpr


R_EARTH = 6371e3


class MidcourseODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare("boundary_alt", default=100e3)
        self.options.declare("surrogate_path", default="aero_surrogate.pkl")
        self.options.declare("geometry_key", default="generic")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("r", val=np.zeros((nn, 3)))
        self.add_input("v", val=np.zeros((nn, 3)))
        self.add_input("q", val=np.zeros((nn, 4)))
        self.add_input("omega", val=np.zeros((nn, 3)))
        self.add_input("m", val=np.full(nn, 100.0))
        self.add_input("accel_x", val=np.zeros(nn))
        self.add_input("accel_y", val=np.zeros(nn))
        self.add_input("accel_z", val=np.zeros(nn))
        self.add_input("time", val=np.zeros(nn))

        self.add_output("dr_dt", val=np.zeros((nn, 3)))
        self.add_output("dv_dt", val=np.zeros((nn, 3)))
        self.add_output("dq_dt", val=np.zeros((nn, 4)))
        self.add_output("domega_dt", val=np.zeros((nn, 3)))
        self.add_output("dm_dt", val=np.zeros(nn))

        self.eom = EOM6DOF(boundary_alt=self.options["boundary_alt"])
        self.guidance = MidcourseGuidance()
        self._surrogate_model = None
        self._geometry_key = self.options["geometry_key"]

    def _get_surrogate(self):
        if self._surrogate_model is None:
            try:
                self._surrogate_model = load_vehicle_gpr(self._geometry_key)
            except Exception:
                pass
        return self._surrogate_model

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        r = inputs["r"]
        v = inputs["v"]
        q = inputs["q"]
        omega = inputs["omega"]
        m = inputs["m"]
        accel = np.stack([inputs["accel_x"], inputs["accel_y"], inputs["accel_z"]], axis=1)
        t = inputs["time"]

        boundary_alt = self.options["boundary_alt"]
        gpr = self._get_surrogate()

        def surrogate(mach, alpha, beta, alt):
            if gpr is not None:
                X = np.array([[mach, alpha, beta, alt, 0.0]])
                try:
                    pred = gpr.predict(X, return_std=False)
                    return float(pred[0, 0]), float(pred[0, 1]), float(pred[0, 2])
                except Exception:
                    pass
            cd, cy, cm, _, _ = blended_aero(
                mach, alpha, beta, alt, boundary_alt=boundary_alt
            )
            return cd, cy, cm

        for i in range(nn):
            state = {"r": r[i], "v": v[i], "q": q[i], "omega": omega[i], "m": float(m[i])}

            derivs = self.eom.compute(t[i], state, surrogate)
            outputs["dr_dt"][i] = derivs["r"]
            outputs["dv_dt"][i] = derivs["v"] + accel[i] / max(float(m[i]), 1e-6)
            outputs["dq_dt"][i] = derivs["q"]
            outputs["domega_dt"][i] = derivs["omega"]
            outputs["dm_dt"][i] = derivs["m"]
