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



class ClosedLoopTerminalODE(om.ExplicitComponent):
    """Terminal phase ODE driven by the *closed-loop* seeker guidance.

    This is the Dymos counterpart of ``EngagementRunner``'s terminal branch:
    instead of an open-loop ``accel_*`` control, the interceptor acceleration is
    computed each node by :class:`TerminalGuidance` from the relative
    kinematics (LOS range / rate) to the target. The optimizer therefore tunes
    the *guidance* (navigation ratio ``N`` and ``accel_limit``) to drive the
    final ``miss_distance`` to zero, exactly as the README's Phase-7 next step
    requires — "wire ``EngagementRunner`` into the Dymos terminal phase as the
    closed-loop ODE so the optimizer directly minimizes miss distance of a
    seeker-guided solution".

    The open-loop ``accel_*`` inputs are retained (optional additive bias) so
    the phase stays differentiable and can be compared against the open-loop
    :class:`TerminalODE`, but by default the commanded acceleration is fully
    guidance-generated.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare("boundary_alt", default=100e3)
        self.options.declare("surrogate_path", default="aero_surrogate.pkl")
        self.options.declare("kill_mechanism", default="hit_to_kill")
        self.options.declare("kill_radius", default=0.5)
        # Closed-loop guidance law (pn | apn | zem | sdre_mpc).
        self.options.declare("law", default="pn")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("r", val=np.zeros((nn, 3)))
        self.add_input("v", val=np.zeros((nn, 3)))
        self.add_input("q", val=np.zeros((nn, 4)))
        self.add_input("omega", val=np.zeros((nn, 3)))
        self.add_input("m", val=np.full(nn, 15.0))
        # Optional open-loop bias (kept for differentiability/comparison).
        self.add_input("accel_x", val=np.zeros(nn))
        self.add_input("accel_y", val=np.zeros(nn))
        self.add_input("accel_z", val=np.zeros(nn))
        # Aim point (defended location), held fixed across nodes.
        self.add_input("target_r", val=np.zeros((nn, 3)))
        self.add_input("target_v", val=np.zeros((nn, 3)))
        self.add_input("time", val=np.zeros(nn))
        # Guidance gains exposed as optimizer parameters.
        self.add_input("N", val=4.0 * np.ones(nn))
        self.add_input("accel_limit", val=150.0 * np.ones(nn))

        self.add_output("dr_dt", val=np.zeros((nn, 3)))
        self.add_output("dv_dt", val=np.zeros((nn, 3)))
        self.add_output("dq_dt", val=np.zeros((nn, 4)))
        self.add_output("domega_dt", val=np.zeros((nn, 3)))
        self.add_output("dm_dt", val=np.zeros(nn))
        self.add_output("cmd_accel", val=np.zeros((nn, 3)))
        self.add_output("miss_distance", val=np.full(nn, 100.0))

        self.eom = EOM6DOF(boundary_alt=self.options["boundary_alt"])
        self.guidance = TerminalGuidance(
            mechanism=self.options["kill_mechanism"],
            kill_radius=self.options["kill_radius"],
            law=self.options["law"],
        )

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        r = inputs["r"]
        v = inputs["v"]
        q = inputs["q"]
        omega = inputs["omega"]
        m = inputs["m"]
        bias = np.stack([inputs["accel_x"], inputs["accel_y"], inputs["accel_z"]], axis=1)
        target_r = inputs["target_r"]
        target_v = inputs["target_v"]
        t = inputs["time"]

        for i in range(nn):
            state = {"r": r[i], "v": v[i], "q": q[i], "omega": omega[i], "m": float(m[i])}

            def surrogate(mach, alpha, beta, alt):
                cd, cy, cm, _, _ = blended_aero(
                    mach, alpha, beta, alt, boundary_alt=self.options["boundary_alt"]
                )
                return cd, cy, cm

            # Closed-loop guidance: tune this node's gains from the parameters.
            self.guidance.N = float(inputs["N"][i])
            self.guidance.accel_limit = float(inputs["accel_limit"][i])
            target_state = {"r": target_r[i], "v": target_v[i]}
            cmd = self.guidance.commanded_accel(
                float(t[i]), state, target_state, disable_fov=True
            )
            accel = cmd + bias[i]

            # ``accel`` is already an acceleration command (m/s^2); this matches
            # EngagementRunner, which converts the guidance command to a force
            # (cmd * m) and then divides by m back to acceleration. No extra
            # mass division here.
            derivs = self.eom.compute(t[i], state, surrogate)
            outputs["dr_dt"][i] = derivs["r"]
            outputs["dv_dt"][i] = derivs["v"] + accel
            outputs["dq_dt"][i] = derivs["q"]
            outputs["domega_dt"][i] = derivs["omega"]
            outputs["dm_dt"][i] = derivs["m"]
            outputs["cmd_accel"][i] = accel
            outputs["miss_distance"][i] = np.linalg.norm(r[i] - target_r[i])