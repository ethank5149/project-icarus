import os
import numpy as np
import joblib
from openmdao.api import ExplicitComponent


COEFF_NAMES = ["Cd", "Cy", "Cm", "Cn", "Cl_roll"]
SIGMA_NAMES = [f"sigma_{n}" for n in COEFF_NAMES]
FEAT = ["mach", "alpha", "beta", "altitude", "delta"]


class AeroSurrogateComponent(ExplicitComponent):
    """OpenMDAO wrapper for the multi-output GPR aero surrogate.

    Returns Cd, Cy, Cm, Cn, Cl_roll plus predictive standard deviations.
    Mean partials are supplied analytically (finite-difference Jacobian from
    the GPR); sigma partials are left to OpenMDAO's own finite differences.
    """

    def initialize(self):
        self.options.declare("model_path", default="aero_surrogate.pkl")
        self.options.declare("vehicle_key", default=None)

    def setup(self):
        if self.options["vehicle_key"] is not None:
            model_path = self.options["model_path"]
            base = os.path.dirname(model_path) if os.path.dirname(model_path) else "."
            vkey = self.options["vehicle_key"]
            candidate = os.path.join(base, f"aero_surrogate_{vkey}.pkl")
            if os.path.exists(candidate):
                model_path = candidate
        else:
            model_path = self.options["model_path"]
        self.gpr = joblib.load(model_path)

        self.add_input("mach", val=2.0)
        self.add_input("alpha", val=0.0)
        self.add_input("beta", val=0.0)
        self.add_input("altitude", val=0.0)
        self.add_input("delta", val=0.0)

        for name in COEFF_NAMES:
            self.add_output(name, val=0.0)
            self.add_output(f"sigma_{name}", val=0.0)

        # Mean partials are exact (from the GPR Jacobian).
        self.declare_partials(COEFF_NAMES, FEAT, method="exact")
        # Sigma partials are left to OpenMDAO finite differences.
        self.declare_partials(SIGMA_NAMES, FEAT, method="fd")

    def compute(self, inputs, outputs):
        X = np.array([[
            inputs["mach"][0],
            inputs["alpha"][0],
            inputs["beta"][0],
            inputs["altitude"][0],
            inputs["delta"][0],
        ]])
        means, stds = self.gpr.predict(X, return_std=True)
        for j, name in enumerate(COEFF_NAMES):
            outputs[name] = means[0, j]
            outputs[f"sigma_{name}"] = stds[0, j]

    def compute_partials(self, inputs, partials):
        """Finite-difference Jacobian of the GPR mean from the trained model."""
        X = np.array([[
            inputs["mach"][0],
            inputs["alpha"][0],
            inputs["beta"][0],
            inputs["altitude"][0],
            inputs["delta"][0],
        ]])
        J = self.gpr.analytical_jacobian(X)  # (1, n_coeff, n_features)
        for j, cname in enumerate(COEFF_NAMES):
            for i, fname in enumerate(FEAT):
                partials[cname, fname] = J[0, j, i]
