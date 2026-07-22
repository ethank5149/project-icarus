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
    nose_profile,
    composite_profile,
    VehicleGeometry,
)
from project_icarus.aero.cfd_generators import (
    SweepSpec,
    run_sweep,
    run_vehicle_sweep,
    vehicle_sweep_to_hdf5,
    default_spec,
    COEFF_NAMES,
)
from project_icarus.surrogates.train_gpr import (
    MultiOutputGPR,
    load_data,
    train_vehicle_gpr,
    load_vehicle_gpr,
    default_model_path,
    default_h5_path,
    COEFF_NAMES as GPR_COEFF_NAMES,
)


class TestAllVehiclePresets:
    """Every preset must have positive dimensions, a non-negative reference area,
    and build a valid numpy mesh with at least one face."""

    @pytest.mark.parametrize("key", sorted(VEHICLE_PRESETS.keys()))
    def test_preset_valid(self, key):
        g = get_vehicle(key)
        assert g.body_length > 0, f"{key}: body_length <= 0"
        assert g.body_diameter > 0, f"{key}: body_diameter <= 0"
        assert g.reference_area > 0, f"{key}: reference_area <= 0"

        v, f = build_surface_mesh(g, backend="numpy")
        assert v.ndim == 2 and v.shape[1] == 3, f"{key}: bad vertex shape"
        assert f.ndim == 2 and f.shape[1] == 3, f"{key}: bad face shape"
        assert f.max() < v.shape[0], f"{key}: face index out of range"
        assert f.shape[0] > 0, f"{key}: zero faces"

    def test_interceptor_keys_exist(self):
        for k in ("arrow2", "arrow3", "gmd_gbi", "gmd_ce2", "patriot",
                  "tamir", "thaad", "sm3"):
            assert k in VEHICLE_PRESETS

    def test_threat_keys_exist(self):
        for k in ("rs28_sarmat", "avangard", "kalibr_3m14", "kh101", "kh102",
                  "zircon_3m22", "burevestnik", "cj10", "cj1000", "yj12", "yj18"):
            assert k in VEHICLE_PRESETS

    def test_decoy_keys_exist(self):
        for k in ("signature_balloon_decoy", "foam_decoy", "swarm_bus", "generic_rv"):
            assert k in VEHICLE_PRESETS


class TestBodyProfile:
    def test_profile_shape(self):
        g = get_vehicle("arrow3")
        prof = body_profile(g, n_axial=50)
        assert prof.shape == (50, 2)

    def test_profile_monotone_z(self):
        g = get_vehicle("arrow3")
        prof = body_profile(g, n_axial=50)
        assert np.all(np.diff(prof[:, 0]) > 0)

    def test_profile_radius_bounded(self):
        g = get_vehicle("arrow3")
        prof = body_profile(g, n_axial=50)
        assert np.all(prof[:, 1] <= g.body_diameter / 2.0 + 1e-9)

    def test_boattail_reduces_radius(self):
        g = get_vehicle("arrow3")
        prof = body_profile(g, n_axial=50)
        tail_z = g.body_length - g.tail_length
        tail_start = prof[prof[:, 0] >= tail_z - 1e-9]
        if tail_start.shape[0] > 0:
            assert tail_start[-1, 1] < tail_start[0, 1] - 1e-9

    def test_blunt_nose_starts_at_zero(self):
        g = get_vehicle("generic_rv")
        prof = body_profile(g, n_axial=40)
        assert prof[0, 1] <= 1e-9

    def test_nose_profile_blunt_cap(self):
        g = get_vehicle("generic_rv")
        prof = nose_profile(g, n_axial=40)
        assert np.all(prof[:, 1] >= 0.0)

    def test_composite_profile_returns_none_when_no_bus(self):
        g = get_vehicle("arrow3")
        assert composite_profile(g) is None

    def test_composite_profile_returns_array_when_bus(self):
        g = get_vehicle("sarmat_bus")
        prof = composite_profile(g)
        assert prof is not None
        assert prof.ndim == 2 and prof.shape[1] == 2


class TestMeshQuality:
    def test_sarmat_panel_count(self):
        g = get_vehicle("rs28_sarmat")
        v, f = build_surface_mesh(g, n_axial=40, n_circ=32, backend="numpy")
        n_panels = int(f.shape[0]) if f.size > 0 else 0
        assert n_panels > 0
        assert n_panels <= 500_000  # cap check

    @pytest.mark.parametrize("key", ["arrow3", "avangard", "yj12"])
    def test_mesh_positive_area(self, key):
        g = get_vehicle(key)
        v, f = build_surface_mesh(g, backend="numpy")
        assert surface_area(v, f) > 0.0

    def test_mesh_no_degenerate_panels(self):
        g = get_vehicle("zircon_3m22")
        v, f = build_surface_mesh(g, backend="numpy")
        tri = v[f]
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        areas = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
        frac_degen = np.sum(areas <= 1e-10) / areas.shape[0]
        assert frac_degen < 0.2
        assert np.median(areas) > 1e-4


class TestRunVehicleSweep:
    def test_analytic_vehicle_sweep_shape(self):
        res = run_vehicle_sweep(
            "tamir", backend="analytic",
            mach_range=(0.5, 2.0, 3), alpha_range=(-5.0, 5.0, 3),
            beta_range=(-2.0, 2.0, 2), altitude_range=(0.0, 50e3, 2),
            delta_range=(0.0, 0.0, 1),
        )
        n = 3 * 3 * 2 * 2 * 1
        assert res["coeffs"].shape == (n, 5)
        assert "mesh_stats" in res
        assert "sweep_metadata" in res
        assert res["sweep_metadata"]["vehicle_key"] == "tamir"

    def test_vehicle_sweep_hdf5_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "aero_avangard.h5")
            vehicle_sweep_to_hdf5("avangard", path, backend="analytic",
                                  mach_range=(1.0, 3.0, 2),
                                  alpha_range=(0.0, 5.0, 2))
            assert os.path.exists(path)
            import h5py
            with h5py.File(path, "r") as hf:
                assert hf["coeffs"].shape[1] == 5
                assert hf.attrs["backend_kind"] == "analytic"


class TestPerVehicleGPR:
    def test_default_paths(self):
        assert default_model_path("arrow3") == "reference/surrogates/aero_surrogate_arrow3.pkl"
        assert default_h5_path("arrow3") == "reference/surrogates/aero_arrow3.h5"

    def test_train_and_load_vehicle_model(self):
        with tempfile.TemporaryDirectory() as td:
            h5_path = os.path.join(td, "aero_tamir.h5")
            spec = default_spec(
                "tamir", backend="analytic",
                mach_range=(0.5, 3.0, 3),
                alpha_range=(-5.0, 5.0, 3),
                beta_range=(-3.0, 3.0, 3),
                altitude_range=(0.0, 50e3, 3),
                delta_range=(0.0, 0.0, 1),
            )
            from project_icarus.aero.cfd_generators import sweep_to_hdf5
            sweep_to_hdf5(spec, h5_path)

            model_path = os.path.join(td, "aero_surrogate_tamir.pkl")
            model = train_vehicle_gpr("tamir", h5_path=h5_path, model_dir=td)
            assert os.path.exists(model_path)

            loaded = load_vehicle_gpr("tamir", model_dir=td)
            assert loaded is not None

            X = np.array([[1.0, 0.0, 0.0, 10e3, 0.0]])
            pred1 = model.predict(X)
            pred2 = loaded.predict(X)
            np.testing.assert_allclose(pred1, pred2, atol=1e-6)

    def test_analytical_jacobian_shape(self):
        with tempfile.TemporaryDirectory() as td:
            h5_path = os.path.join(td, "aero_tamir.h5")
            spec = default_spec(
                "tamir", backend="analytic",
                mach_range=(0.5, 3.0, 3),
                alpha_range=(-5.0, 5.0, 3),
                beta_range=(-3.0, 3.0, 3),
                altitude_range=(0.0, 50e3, 3),
                delta_range=(0.0, 0.0, 1),
            )
            from project_icarus.aero.cfd_generators import sweep_to_hdf5
            sweep_to_hdf5(spec, h5_path)
            model_path = os.path.join(td, "aero_surrogate_tamir.pkl")
            model = train_vehicle_gpr("tamir", h5_path=h5_path, model_dir=td)
            X = np.array([[1.0, 0.0, 0.0, 10e3, 0.0],
                          [2.0, 3.0, 1.0, 30e3, 2.0]])
            J = model.analytical_jacobian(X)
            assert J.shape == (2, 5, 5)
            assert np.all(np.isfinite(J))

    def test_k_fold_reports_per_vehicle(self):
        with tempfile.TemporaryDirectory() as td:
            h5_path = os.path.join(td, "aero_tamir.h5")
            spec = default_spec(
                "tamir", backend="analytic",
                mach_range=(0.5, 3.0, 3),
                alpha_range=(-5.0, 5.0, 3),
                beta_range=(-3.0, 3.0, 3),
                altitude_range=(0.0, 50e3, 3),
                delta_range=(0.0, 0.0, 1),
            )
            from project_icarus.aero.cfd_generators import sweep_to_hdf5
            sweep_to_hdf5(spec, h5_path)
            model = train_vehicle_gpr("tamir", h5_path=h5_path, model_dir=td)
            assert model is not None
