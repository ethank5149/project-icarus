import numpy as np
import openmdao.api as om
from ...dynamics.eom_6dof import EOM6DOF
from ...aero.aero_analytical import blended_aero
from ...dynamics.thrust import StageSpec, MultiStageThrustModel, sarmat_stage_specs
from ...guidance.icbm_guidance import ICBMGuidance
from ...scenarios.target_factory import _ecef_to_geodetic, _enu_basis, _geodetic_to_ecef_simple
from ...reference.surface_elevation import get_surface_elevation

R_EARTH = 6371e3


class SarmatBoostODE(om.ExplicitComponent):
    """Offensive boost-phase ODE for RS-28 Sarmat.

    Uses the existing ``EOM6DOF`` for aero/gravity/rotational dynamics and
    ``MultiStageThrustModel`` for staged thrust and mass tracking.  The
    guidance parameters ``pitch_over_start``, ``initial_elevation``, and
    ``burnout_vmag`` are exposed as Dymos parameters so the optimizer can
    tune the real ``ICBMGuidance`` pitch schedule to minimize terminal miss
    distance.

    States:  r (3), v (3), q (4), omega (3), m (1)
    Controls: none (thrust direction is parameterized via ICBMGuidance)
    Parameters: pitch_over_start, initial_elevation, burnout_vmag, target_r
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare("boundary_alt", default=100e3)
        self.options.declare("surrogate_path", default="aero_surrogate.pkl")
        self.options.declare("geometry_key", default="generic")
        self.options.declare("target_lat", default=38.90)
        self.options.declare("target_lon", default=-77.04)

    def setup(self):
        nn = self.options["num_nodes"]
        # States
        self.add_input("r", val=np.zeros((nn, 3)))
        self.add_input("v", val=np.zeros((nn, 3)))
        self.add_input("q", val=np.zeros((nn, 4)))
        self.add_input("omega", val=np.zeros((nn, 3)))
        self.add_input("m", val=np.full(nn, 208100.0))
        self.add_input("time", val=np.zeros(nn))

        # Guidance / vehicle parameters (optimizer-tunable).
        # Dymos passes these as static inputs: shape (nn,) with all values equal.
        self.add_input("pitch_over_start", val=np.full(nn, 5.0))
        self.add_input("initial_elevation", val=np.full(nn, np.radians(55.0)))
        self.add_input("burnout_vmag", val=np.full(nn, 6200.0))
        # Fixed target position in ECEF
        self.add_input("target_r", val=np.zeros((nn, 3)))

        # Outputs
        self.add_output("dr_dt", val=np.zeros((nn, 3)))
        self.add_output("dv_dt", val=np.zeros((nn, 3)))
        self.add_output("dq_dt", val=np.zeros((nn, 4)))
        self.add_output("domega_dt", val=np.zeros((nn, 3)))
        self.add_output("dm_dt", val=np.zeros(nn))
        self.add_output("miss_distance", val=np.zeros(nn))

        # 6-DOF EOM (no thrust model — we add thrust externally)
        # use_cython=False ensures OpenMDAO complex-step derivatives work
        self.eom = EOM6DOF(boundary_alt=self.options["boundary_alt"], use_cython=False)

        # Authoritative RS-28 Sarmat multi-stage thrust model
        self._thrust_model = MultiStageThrustModel(sarmat_stage_specs())

        self._surrogate_model = None
        self._geometry_key = self.options["geometry_key"]
        self._target_lat = self.options["target_lat"]
        self._target_lon = self.options["target_lon"]

        # Real ICBM guidance instance (single source of truth for thrust direction)
        target_lat = float(self._target_lat)
        target_lon = float(self._target_lon)
        target_elev = get_surface_elevation(target_lat, target_lon)
        launch_lat = 54.07
        launch_lon = 35.73
        launch_elev = get_surface_elevation(launch_lat, launch_lon)
        self._guidance = ICBMGuidance(
            target_ecef=_geodetic_to_ecef_simple(target_lat, target_lon, target_elev),
            launch_ecef=_geodetic_to_ecef_simple(launch_lat, launch_lon, launch_elev),
            burnout_vmag=6200.0,
            pitch_over_start=5.0,
            initial_elevation=np.radians(55.0),
            use_j2=True,
        )

        self.declare_partials('*', '*', method='fd')

    def _get_surrogate(self):
        if self._surrogate_model is None:
            try:
                from ...surrogates.train_gpr import load_vehicle_gpr
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
        t = inputs["time"]

        # Parameters are static (same across all nodes); read from index 0
        pitch_over_start = float(inputs["pitch_over_start"][0])
        initial_elevation = float(inputs["initial_elevation"][0])
        burnout_vmag = float(inputs["burnout_vmag"][0])
        target_r = inputs["target_r"][0]

        self._guidance.pitch_over_start = pitch_over_start
        self._guidance.initial_elevation = initial_elevation
        self._guidance.burnout_vmag = burnout_vmag

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
            return blended_aero(mach, alpha, beta, alt, boundary_alt=boundary_alt)[:3]

        for i in range(nn):
            state = {"r": r[i], "v": v[i], "q": q[i],
                     "omega": omega[i], "m": float(m[i])}

            # EOM6DOF returns aero + gravity + quaternion + rotational dynamics
            derivs = self.eom.compute(t[i], state, surrogate)

            # Thrust magnitude from multi-stage model (authoritative stage specs)
            T_mag = self._thrust_model.thrust(t[i], state)
            # Mass rate from the underlying stage (explicit mass_flow or Isp-based clamp)
            dm_dt = self._thrust_model.mass_rate(t[i], state)

            # Thrust direction from real ICBM guidance
            thrust_dir = self._guidance.thrust_direction(r[i], v[i], t[i])
            a_thrust = (T_mag / max(float(m[i]), 1e-6)) * thrust_dir

            outputs["dr_dt"][i] = derivs["r"]
            outputs["dv_dt"][i] = derivs["v"] + a_thrust
            outputs["dq_dt"][i] = derivs["q"]
            outputs["domega_dt"][i] = derivs["omega"]
            outputs["dm_dt"][i] = dm_dt
            outputs["miss_distance"][i] = np.linalg.norm(r[i] - target_r)
