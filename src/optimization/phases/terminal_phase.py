import numpy as np
import openmdao.api as om
from ...dynamics.eom_6dof import EOM6DOF
from ...guidance.terminal_guidance import TerminalGuidance
from ...aero.aero_analytical import blended_aero


R_EARTH = 6371e3


class TerminalODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare("boundary_alt", default=100e3)
        self.options.declare("surrogate_path", default="aero_surrogate.pkl")
        self.options.declare("kill_mechanism", default="hit_to_kill")
        self.options.declare("kill_radius", default=0.5)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("r", val=np.zeros((nn, 3)))
        self.add_input("v", val=np.zeros((nn, 3)))
        self.add_input("q", val=np.zeros((nn, 4)))
        self.add_input("omega", val=np.zeros((nn, 3)))
        self.add_input("m", val=np.full(nn, 15.0))
        self.add_input("accel_x", val=np.zeros(nn))
        self.add_input("accel_y", val=np.zeros(nn))
        self.add_input("accel_z", val=np.zeros(nn))
        # Aim point (defended location), held fixed across nodes.
        self.add_input("target_r", val=np.zeros((nn, 3)))
        self.add_input("target_v", val=np.zeros((nn, 3)))
        self.add_input("time", val=np.zeros(nn))

        self.add_output("dr_dt", val=np.zeros((nn, 3)))
        self.add_output("dv_dt", val=np.zeros((nn, 3)))
        self.add_output("dq_dt", val=np.zeros((nn, 4)))
        self.add_output("domega_dt", val=np.zeros((nn, 3)))
        self.add_output("dm_dt", val=np.zeros(nn))
        self.add_output("miss_distance", val=np.full(nn, 100.0))

        self.eom = EOM6DOF(boundary_alt=self.options["boundary_alt"])
        self.guidance = TerminalGuidance(
            mechanism=self.options["kill_mechanism"],
            kill_radius=self.options["kill_radius"],
        )

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        r = inputs["r"]
        v = inputs["v"]
        q = inputs["q"]
        omega = inputs["omega"]
        m = inputs["m"]
        accel = np.stack([inputs["accel_x"], inputs["accel_y"], inputs["accel_z"]], axis=1)
        target_r = inputs["target_r"]
        t = inputs["time"]

        for i in range(nn):
            state = {"r": r[i], "v": v[i], "q": q[i], "omega": omega[i], "m": float(m[i])}

            def surrogate(mach, alpha, beta, alt):
                cd, cy, cm, _, _ = blended_aero(
                    mach, alpha, beta, alt, boundary_alt=self.options["boundary_alt"]
                )
                return cd, cy, cm

            derivs = self.eom.compute(t[i], state, surrogate)
            outputs["dr_dt"][i] = derivs["r"]
            outputs["dv_dt"][i] = derivs["v"] + accel[i] / max(float(m[i]), 1e-6)
            outputs["dq_dt"][i] = derivs["q"]
            outputs["domega_dt"][i] = derivs["omega"]
            outputs["dm_dt"][i] = derivs["m"]
            outputs["miss_distance"][i] = np.linalg.norm(r[i] - target_r[i])
