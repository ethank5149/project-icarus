import numpy as np
import joblib
from openmdao.api import ExplicitComponent


class AeroSurrogateComponent(ExplicitComponent):
    """
    OpenMDAO wrapper for multi-output GPR aero surrogate.
    Returns Cd, Cy, Cm + predictive standard deviations.
    """

    def initialize(self):
        self.options.declare("model_path", default="aero_surrogate.pkl")

    def setup(self):
        model_path = self.options["model_path"]
        self.gpr = joblib.load(model_path)

        self.add_input("mach", val=2.0)
        self.add_input("alpha", val=0.0)
        self.add_input("beta", val=0.0)
        self.add_input("altitude", val=0.0)

        self.add_output("Cd", val=0.0)
        self.add_output("Cy", val=0.0)
        self.add_output("Cm", val=0.0)
        self.add_output("sigma_Cd", val=0.0)
        self.add_output("sigma_Cy", val=0.0)
        self.add_output("sigma_Cm", val=0.0)

        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        X = np.array(
            [
                [
                    inputs["mach"][0],
                    inputs["alpha"][0],
                    inputs["beta"][0],
                    inputs["altitude"][0],
                ]
            ]
        )
        means, stds = self.gpr.predict(X, return_std=True)
        outputs["Cd"] = means[0, 0]
        outputs["Cy"] = means[0, 1]
        outputs["Cm"] = means[0, 2]
        outputs["sigma_Cd"] = stds[0, 0]
        outputs["sigma_Cy"] = stds[0, 1]
        outputs["sigma_Cm"] = stds[0, 2]
