import numpy as np
import openmdao.api as om
from ...dynamics.eom_6dof import EOM6DOF
from ...aero.aero_analytical import blended_aero
from ...dynamics.thrust import StageSpec, MultiStageThrustModel
from ...scenarios.target_factory import _ecef_to_geodetic, _enu_basis

R_EARTH = 6371e3


class SarmatBoostODE(om.ExplicitComponent):
    """Offensive boost-phase ODE for RS-28 Sarmat.

    Uses the existing ``EOM6DOF`` for aero/gravity/rotational dynamics and
    ``MultiStageThrustModel`` for staged thrust and mass tracking.  The
    guidance parameters ``el_0``, ``el_1``, ``t_cross``, and ``T3_scale``
    are exposed as Dymos parameters so the optimizer can tune the pitch
    schedule and stage-3 thrust level to minimize terminal miss distance.

    States:  r (3), v (3), q (4), omega (3), m (1)
    Controls: none (thrust direction is parameterized)
    Parameters: el_0, el_1, t_cross, T3_scale, target_r
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
        self.add_input("el_0", val=np.full(nn, np.radians(45.0)))
        self.add_input("el_1", val=np.full(nn, np.radians(25.0)))
        self.add_input("t_cross", val=np.full(nn, 100.0))
        self.add_input("T3_scale", val=np.full(nn, 1.0))
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

        # RS-28 Sarmat multi-stage thrust model (authoritative source data):
        # - Stage 1: PDU-99 derived from RD-274 (~4,952 kN, 90s burn)
        # - Stage 2: RD-250 class (~1,200 kN, 180s burn)
        # - Stage 3: Four RS-99 engines, "over 100 tons thrust" (~1,000+ kN, 90s burn)
        # Refs: Astronautica RD-274 (4,952 kN), Missile Defense Advocacy (PDU-99),
        #       Army Recognition ("four engines ... over 100 tons thrust")
        self._thrust_model = MultiStageThrustModel([
            StageSpec(thrust=lambda t: 5.0e6, burn_time=90.0,
                      wet_mass=140000.0, dry_mass=11340.0, Isp=315.0),
            StageSpec(thrust=lambda t: 1.2e6, burn_time=180.0,
                      wet_mass=48000.0, dry_mass=3600.0, Isp=325.0),
            StageSpec(thrust=lambda t: 1.0e6, burn_time=90.0,
                      wet_mass=10100.0, dry_mass=2160.0, Isp=330.0),
        ])

        self._surrogate_model = None
        self._geometry_key = self.options["geometry_key"]
        self._target_lat = self.options["target_lat"]
        self._target_lon = self.options["target_lon"]

        self.declare_partials('*', '*', method='fd')

    def _get_surrogate(self):
        if self._surrogate_model is None:
            try:
                from ...surrogates.train_gpr import load_vehicle_gpr
                self._surrogate_model = load_vehicle_gpr(self._geometry_key)
            except Exception:
                pass
        return self._surrogate_model

    @staticmethod
    def _compute_thrust_direction(t, r, v, el_0, el_1, t_cross, target_r):
        """Onboard guidance: staged elevation + great-circle azimuth.

        Mirrors the logic in ``SarmatScenario._current_thrust_dir`` but
        parameterized by Dymos design variables instead of hard-coded values.
        """
        to_target = target_r - r
        to_target_dir = to_target / max(np.linalg.norm(to_target), 1e-9)

        lat, lon, _ = _ecef_to_geodetic(r)
        east, north, up = _enu_basis(lat, lon)

        gc_horiz = to_target_dir - np.dot(to_target_dir, up) * up
        gc_horiz_norm = np.linalg.norm(gc_horiz)
        if gc_horiz_norm > 1e-6:
            gc_horiz = gc_horiz / gc_horiz_norm
        else:
            gc_horiz = np.array([0.0, 0.0, 0.0])

        if t < 5.0:
            return up

        # Linear staged elevation from el_0 to el_1 between t=5 and t_cross
        frac = np.clip((t - 5.0) / max(t_cross - 5.0, 1.0), 0.0, 1.0)
        el = el_0 + frac * (el_1 - el_0)
        el = np.clip(el, np.radians(15.0), np.radians(75.0))

        return np.cos(el) * gc_horiz + np.sin(el) * up

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        r = inputs["r"]
        v = inputs["v"]
        q = inputs["q"]
        omega = inputs["omega"]
        m = inputs["m"]
        t = inputs["time"]

        # Parameters are static (same across all nodes); read from index 0
        el_0 = float(inputs["el_0"][0])
        el_1 = float(inputs["el_1"][0])
        t_cross = float(inputs["t_cross"][0])
        T3_scale = float(inputs["T3_scale"][0])
        target_r = inputs["target_r"][0]

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

            # Thrust magnitude from multi-stage model (with T3 scaling)
            T_mag = self._thrust_model.thrust(t[i], state) * T3_scale
            # Mass rate from the underlying stage (unscaled Isp-based)
            dm_dt = self._thrust_model.mass_rate(t[i], state)

            # Thrust direction from parameterized guidance
            thrust_dir = self._compute_thrust_direction(
                t[i], r[i], v[i], el_0, el_1, t_cross, target_r
            )
            a_thrust = (T_mag / max(float(m[i]), 1e-6)) * thrust_dir

            outputs["dr_dt"][i] = derivs["r"]
            outputs["dv_dt"][i] = derivs["v"] + a_thrust
            outputs["dq_dt"][i] = derivs["q"]
            outputs["domega_dt"][i] = derivs["omega"]
            outputs["dm_dt"][i] = dm_dt
            outputs["miss_distance"][i] = np.linalg.norm(r[i] - target_r)
