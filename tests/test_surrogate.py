import numpy as np
import pytest
from src.aero.aero_analytical import newtonian_cd_exo, newtonian_cl_exo, linear_viscous_endo, blended_aero
from src.surrogates.uncertainty import UncertaintyPropagator


class TestAeroAnalytical:
    def test_newtonian_cd_exo(self):
        cd = newtonian_cd_exo(5.0, 0.0, 0.0)
        assert np.isclose(cd, 0.0)

    def test_newtonian_cl_exo(self):
        cl, cm = newtonian_cl_exo(5.0, 15.0, 0.0)
        assert np.isclose(cl, 2.0 * np.sin(np.radians(15.0)) * np.cos(np.radians(15.0)))

    def test_linear_viscous_endo(self):
        cd, cl, cm = linear_viscous_endo(2.0, 5.0, 0.0)
        assert cd > 0
        assert cl != 0

    def test_blended_aero(self):
        cd, cl, cm = blended_aero(2.0, 5.0, 0.0, 50e3)
        assert np.isfinite(cd)
        assert np.isfinite(cl)
        assert np.isfinite(cm)


class TestUncertaintyPropagator:
    def test_propagate_shape(self):
        up = UncertaintyPropagator(n_samples=10)
        def eom(cd, cl, cm):
            return 1.0
        mean, std = up.propagate(np.zeros(3), 0.1, 0.1, 0.1, eom)
        assert np.isfinite(mean)
        assert np.isfinite(std)

    def test_sobol_indices_sum(self):
        up = UncertaintyPropagator(n_samples=10)
        def eom(cd, cl, cm):
            return float(cd + cl + cm)
        indices = up.sobol_indices(np.zeros(3), 0.1, 0.1, 0.1, eom)
        assert np.isclose(sum(indices), 1.0)
