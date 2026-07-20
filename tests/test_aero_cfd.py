import os
import tempfile

import numpy as np
import pytest

from project_icarus.aero.geometry import (
    VEHICLE_PRESETS,
    get_vehicle,
    build_surface_mesh,
    surface_area,
    body_profile,
    VehicleGeometry,
)
from project_icarus.aero.cfd_generators import (
    SweepSpec,
    run_sweep,
    save_sweep_hdf5,
    sweep_to_hdf5,
    cached_sweep,
    default_spec,
    COEFF_NAMES,
)


class TestGeometry:
    def test_presets_exist(self):
        for k in ("arrow3", "tamir", "gmd_gbi", "sm3", "threat_rv"):
            assert k in VEHICLE_PRESETS
            g = get_vehicle(k)
            assert g.body_length > 0 and g.body_diameter > 0
            assert g.reference_area > 0

    def test_unknown_vehicle_raises(self):
        with pytest.raises(KeyError):
            get_vehicle("does_not_exist")

    def test_body_profile_monotone_and_bounded(self):
        g = get_vehicle("arrow3")
        prof = body_profile(g, n_axial=50)
        assert prof.shape == (50, 2)
        # Max radius never exceeds body radius; nose starts at 0.
        assert prof[0, 1] <= 1e-9
        assert np.all(prof[:, 1] <= g.body_diameter / 2.0 + 1e-9)

    def test_numpy_mesh_has_faces(self):
        g = get_vehicle("tamir")
        v, f = build_surface_mesh(g, backend="numpy")
        assert v.ndim == 2 and v.shape[1] == 3
        assert f.ndim == 2 and f.shape[1] == 3
        assert f.max() < v.shape[0]
        assert surface_area(v, f) > 0.0

    def test_finned_mesh_larger_than_body(self):
        g = get_vehicle("arrow3")  # 4 fins
        v, f = build_surface_mesh(g, backend="numpy")
        # Body alone (no fins) has fewer faces than finned vehicle.
        g_nofin = VehicleGeometry(
            name="nofin", body_length=g.body_length,
            body_diameter=g.body_diameter, nose_length=g.nose_length,
            nose_type=g.nose_type,
        )
        v2, f2 = build_surface_mesh(g_nofin, backend="numpy")
        assert f.shape[0] > f2.shape[0]

    def test_blunt_nose_profile(self):
        g = get_vehicle("threat_rv")
        prof = body_profile(g, n_axial=40)
        assert np.all(prof[:, 1] >= 0.0)


class TestCFDSweep:
    def test_analytic_sweep_shape(self):
        spec = SweepSpec(
            vehicle="tamir", backend="analytic",
            mach_range=(0.5, 2.0, 3), alpha_range=(-5.0, 5.0, 3),
            beta_range=(-2.0, 2.0, 3), altitude_range=(0.0, 100e3, 2),
            delta_range=(0.0, 0.0, 1),
        )
        res = run_sweep(spec)
        n = 3 * 3 * 3 * 2 * 1
        assert res["coeffs"].shape == (n, 5)
        assert list(res["coeff_names"]) == COEFF_NAMES
        assert np.all(res["coeffs"][:, 0] >= 0.0)  # Cd non-negative
        assert np.all(np.isfinite(res["coeffs"]))

    def test_control_deflection_shifts_cy(self):
        base = SweepSpec(
            vehicle="tamir", backend="analytic",
            mach_range=(2.0, 2.0, 1), alpha_range=(0.0, 0.0, 1),
            beta_range=(0.0, 0.0, 1), altitude_range=(0.0, 0.0, 1),
            delta_range=(0.0, 0.0, 1),
        )
        defl = SweepSpec(
            vehicle="tamir", backend="analytic",
            mach_range=(2.0, 2.0, 1), alpha_range=(0.0, 0.0, 1),
            beta_range=(0.0, 0.0, 1), altitude_range=(0.0, 0.0, 1),
            delta_range=(2.0, 2.0, 1),
        )
        rb = run_sweep(base)
        rd = run_sweep(defl)
        # Positive deflection raises Cy (and lowers Cm via restoring moment).
        assert rd["coeffs"][0, 1] > rb["coeffs"][0, 1]
        assert rd["coeffs"][0, 2] < rb["coeffs"][0, 2]

    def test_hdf5_roundtrip(self):
        spec = default_spec("threat_rv", backend="analytic",
                            mach_range=(1.0, 3.0, 2),
                            alpha_range=(0.0, 5.0, 2))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "aero.h5")
            sweep_to_hdf5(spec, path)
            assert os.path.exists(path)
            import h5py
            with h5py.File(path, "r") as hf:
                coeffs = hf["coeffs"][()]
                assert coeffs.shape[1] == 5
                assert hf.attrs["backend_kind"] == "analytic"

    def test_cached_sweep_generates_once(self):
        spec = default_spec("tamir", backend="analytic",
                            mach_range=(1.0, 2.0, 2),
                            alpha_range=(0.0, 2.0, 2))
        with tempfile.TemporaryDirectory() as td:
            p1 = cached_sweep(spec, cache_dir=td)
            mtime1 = os.path.getmtime(p1)
            p2 = cached_sweep(spec, cache_dir=td)
            assert p1 == p2
            assert os.path.getmtime(p2) == mtime1  # not regenerated

    def test_unknown_backend_raises(self):
        spec = SweepSpec(vehicle="tamir", backend="bogus")
        with pytest.raises(ValueError):
            run_sweep(spec)


class TestScipyInterpBackend:
    def test_scipy_interp_shape(self):
        spec = SweepSpec(
            vehicle="tamir", backend="scipy_interp",
            mach_range=(0.5, 3.0, 3), alpha_range=(-5.0, 5.0, 3),
            beta_range=(-2.0, 2.0, 2), altitude_range=(0.0, 100e3, 2),
            delta_range=(0.0, 0.0, 1),
        )
        res = run_sweep(spec)
        n = 3 * 3 * 2 * 2 * 1
        assert res["coeffs"].shape == (n, 5)
        assert list(res["coeff_names"]) == COEFF_NAMES
        assert np.all(res["coeffs"][:, 0] >= 0.0)
        assert np.all(np.isfinite(res["coeffs"]))

    def test_scipy_interp_differs_from_analytic(self):
        base = SweepSpec(
            vehicle="tamir", backend="analytic",
            mach_range=(2.0, 2.0, 1), alpha_range=(3.0, 3.0, 1),
            beta_range=(1.0, 1.0, 1), altitude_range=(50e3, 50e3, 1),
            delta_range=(1.0, 1.0, 1),
        )
        interp = SweepSpec(
            vehicle="tamir", backend="scipy_interp",
            mach_range=(2.0, 2.0, 1), alpha_range=(3.0, 3.0, 1),
            beta_range=(1.0, 1.0, 1), altitude_range=(50e3, 50e3, 1),
            delta_range=(1.0, 1.0, 1),
        )
        rb = run_sweep(base)
        ri = run_sweep(interp)
        assert not np.allclose(rb["coeffs"], ri["coeffs"])

    def test_scipy_interp_control_deflection(self):
        base = SweepSpec(
            vehicle="tamir", backend="scipy_interp",
            mach_range=(2.0, 2.0, 1), alpha_range=(0.0, 0.0, 1),
            beta_range=(0.0, 0.0, 1), altitude_range=(0.0, 0.0, 1),
            delta_range=(0.0, 0.0, 1),
        )
        defl = SweepSpec(
            vehicle="tamir", backend="scipy_interp",
            mach_range=(2.0, 2.0, 1), alpha_range=(0.0, 0.0, 1),
            beta_range=(0.0, 0.0, 1), altitude_range=(0.0, 0.0, 1),
            delta_range=(3.0, 3.0, 1),
        )
        rb = run_sweep(base)
        rd = run_sweep(defl)
        assert rd["coeffs"][0, 1] > rb["coeffs"][0, 1]
        assert rd["coeffs"][0, 2] < rb["coeffs"][0, 2]
