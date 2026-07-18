import numpy as np
import pytest

dymos = pytest.importorskip("dymos")
from src.optimization.trajectory_optimization import build_trajectory_problem


class TestDymosTrajectory:
    def test_problem_setup(self):
        p = build_trajectory_problem()
        assert p is not None
