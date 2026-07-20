import numpy as np
import pytest
from project_icarus.aero.aero_analytical import (
    newtonian_cd_exo,
    newtonian_sideforce_moments_exo,
    linear_viscous_endo,
    blended_aero,
)
from project_icarus.surrogates.uncertainty import UncertaintyPropagator


class TestAeroAnalytical:
    def test_newtonian_cd_exo(self):
        cd = newtonian_cd_exo(5.0, 0.0, 0.0)
        assert np.isclose(cd, 0.0)

    def test_newtonian_sideforce_moments_exo(self):
        cy, cn, cl_roll = newtonian_sideforce_moments_exo(5.0, 15.0, 0.0)
        assert np.isclose(cy, 2.0 * np.sin(np.radians(15.0)) * np.cos(np.radians(15.0)))
        assert np.isclose(cn, 0.0)
        assert np.isclose(cl_roll, 0.0)

    def test_linear_viscous_endo(self):
        cd, cy, cm, cn, cl_roll = linear_viscous_endo(2.0, 5.0, 0.0)
        assert cd > 0
        assert np.isfinite(cm)

    def test_blended_aero(self):
        cd, cy, cm, cn, cl_roll = blended_aero(2.0, 5.0, 0.0, 50e3)
        assert np.isfinite(cd)
        assert np.isfinite(cy)
        assert np.isfinite(cm)
        assert np.isfinite(cn)
        assert np.isfinite(cl_roll)

    def test_compressibility_correction(self):
        cd_sub, _, _, _, _ = linear_viscous_endo(0.5, 5.0, 0.0)
        cd_super, _, _, _, _ = linear_viscous_endo(2.0, 5.0, 0.0)
        assert cd_sub != cd_super


class TestUncertaintyPropagator:
    def test_propagate_shape(self):
        up = UncertaintyPropagator(n_samples=10)
        def eom(cd, cy, cm):
            return 1.0
        mean, std = up.propagate(np.zeros(3), 0.1, 0.1, 0.1, eom)
        assert np.isfinite(mean)
        assert np.isfinite(std)

    def test_elementary_effects_sum(self):
        up = UncertaintyPropagator(n_samples=10)
        def eom(cd, cy, cm):
            return float(cd + cy + cm)
        indices = up.elementary_effects(np.zeros(3), 0.1, 0.1, 0.1, eom)
        assert np.isclose(sum(indices), 1.0)
