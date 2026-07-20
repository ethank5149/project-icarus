import numpy as np
import pytest
from project_icarus.constraints import InterceptConstraint, QuatNormConstraint, ControlSaturation


class TestConstraints:
    def test_intercept_constraint(self):
        comp = InterceptConstraint(kill_radius=0.5)
        comp.setup()
        inputs = {
            "terminal_r": np.array([1.0, 0.0, 0.0]),
            "terminal_v": np.zeros(3),
            "target_r": np.array([1.01, 0.0, 0.0]),
            "target_v": np.zeros(3),
            "max_dyn_pressure_in": 0.0,
            "max_alpha_in": 0.0,
        }
        outputs = {}
        comp.compute(inputs, outputs)
        assert outputs["miss_distance"] < 0.5
        assert outputs["kill_confirmed"] == 1.0

    def test_quat_norm(self):
        comp = QuatNormConstraint()
        comp.setup()
        inputs = {"q": np.array([1.0, 0.0, 0.0, 0.0])}
        outputs = {}
        comp.compute(inputs, outputs)
        assert np.isclose(outputs["q_norm"], 0.0)

    def test_control_saturation(self):
        comp = ControlSaturation(accel_limit=10.0, gimbal_limit=1.0)
        comp.setup()
        inputs = {
            "accel_cmd": np.array([5.0, 0.0, 0.0]),
            "gimbal_cmd": np.array([0.5, 0.0]),
        }
        outputs = {}
        comp.compute(inputs, outputs)
        assert outputs["accel_sat"] < 1.0
        assert outputs["gimbal_sat"] < 1.0
