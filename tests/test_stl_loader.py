import math

import numpy as np
import pytest

from project_icarus.aero.stl_loader import (
    CAD_MANIFEST,
    VehicleSTLInfo,
    load_cad_vehicle,
    normalize_stl,
    read_stl,
    surface_area,
)


def _primary_z_aligned(vertices: np.ndarray) -> bool:
    """Return True if Z is the longest axis after normalization."""
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    if extents[2] <= 0.0:
        return False
    return bool(np.argmax(extents) == 2)


class TestReadStl:
    def test_kh101_binary_loads(self):
        vertices, faces = read_stl("reference/models/kh-101-cruise-missile.stl")
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert vertices.shape[0] > 0
        assert faces.shape[0] > 0
        assert not np.any(np.isnan(vertices))
        assert faces.max() < vertices.shape[0]

    def test_returns_numpy_arrays(self):
        vertices, faces = read_stl("reference/models/kh-101-cruise-missile.stl")
        assert isinstance(vertices, np.ndarray)
        assert isinstance(faces, np.ndarray)

    def test_kh101_has_15690_faces(self):
        vertices, faces = read_stl("reference/models/kh-101-cruise-missile.stl")
        assert faces.shape[0] == 15690


class TestNormalizeStl:
    def test_centroid_near_origin(self):
        vertices, _ = read_stl("reference/models/kh-101-cruise-missile.stl")
        vertices, _ = normalize_stl(vertices, np.zeros((0, 3), dtype=int), scale=0.001)
        centroid = vertices.mean(axis=0)
        assert np.linalg.norm(centroid) < 1e-3

    def test_z_is_longest_axis_kh101(self):
        vertices, _ = read_stl("reference/models/kh-101-cruise-missile.stl")
        vertices, _ = normalize_stl(vertices, np.zeros((0, 3), dtype=int), scale=0.001)
        assert _primary_z_aligned(vertices)

    def test_no_nans_after_normalize(self):
        vertices, faces = read_stl("reference/models/kh-101-cruise-missile.stl")
        vertices, faces = normalize_stl(vertices, faces, scale=0.001)
        assert not np.any(np.isnan(vertices))

    def test_length_sanity_kh101(self):
        vertices, _ = read_stl("reference/models/kh-101-cruise-missile.stl")
        vertices, _ = normalize_stl(vertices, np.zeros((0, 3), dtype=int), scale=0.001)
        z_extent = float(vertices[:, 2].max() - vertices[:, 2].min())
        assert 5.0 < z_extent < 10.0

    def test_force_axis_overrides(self):
        vertices, _ = read_stl("reference/models/kh-101-cruise-missile.stl")
        v1, _ = normalize_stl(vertices, np.zeros((0, 3), dtype=int), scale=0.001, force_axis=0)
        assert _primary_z_aligned(v1)

    def test_scale_changes_extent(self):
        vertices, _ = read_stl("reference/models/kh-101-cruise-missile.stl")
        v1, _ = normalize_stl(vertices, np.zeros((0, 3), dtype=int), scale=1.0)
        v2, _ = normalize_stl(vertices, np.zeros((0, 3), dtype=int), scale=0.001)
        z1 = float(v1[:, 2].max() - v1[:, 2].min())
        z2 = float(v2[:, 2].max() - v2[:, 2].min())
        assert z1 > 0.0
        assert abs(z2 - z1 * 0.001) / max(z2, 1e-9) < 1e-6


class TestLoadCadVehicle:
    def test_kh101_loads(self):
        vertices, faces = load_cad_vehicle("kh101")
        assert vertices.shape[0] > 0
        assert faces.shape[0] > 0
        assert _primary_z_aligned(vertices)
        assert surface_area(vertices, faces) > 0.0

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            load_cad_vehicle("nonexistent_vehicle")

    def test_z_extent_matches_preset_order(self):
        vertices, _ = load_cad_vehicle("kh101")
        z_extent = float(vertices[:, 2].max() - vertices[:, 2].min())
        assert 5.0 < z_extent < 10.0


class TestManifest:
    def test_kh101_entry_exists(self):
        assert "kh101" in CAD_MANIFEST
        info = CAD_MANIFEST["kh101"]
        assert isinstance(info, VehicleSTLInfo)
        assert info.scale == 1.0
        assert info.proxy_for == "kh101"
        assert info.filename == "model.stl"

    def test_manifest_values_are_sane(self):
        for key, info in CAD_MANIFEST.items():
            assert key == info.key
            assert info.scale > 0.0

    def test_surface_area_positive(self):
        for key in CAD_MANIFEST:
            vertices, faces = load_cad_vehicle(key)
            assert surface_area(vertices, faces) > 0.0


class TestCfdIntegration:
    def test_build_stl_mesh_uses_stl_alias(self):
        from project_icarus.aero.geometry import VehicleGeometry, build_surface_mesh

        g = VehicleGeometry(
            name="kh101 STL",
            body_length=7.45,
            body_diameter=0.74,
            nose_length=1.5,
            nose_type="ogive",
            stl_alias="kh101",
        )
        v, f = build_surface_mesh(g, backend="stl")
        assert v.shape[0] > 0
        assert f.shape[0] > 0
        assert _primary_z_aligned(v)

    def test_dolfinx_solve_kh101_produces_finite_coeffs(self):
        from project_icarus.aero.cfd_generators import (
            _dolfinx_solve,
            SweepSpec,
        )
        from project_icarus.aero.geometry import VehicleGeometry

        g = VehicleGeometry(
            name="kh101 STL",
            body_length=7.45,
            body_diameter=0.74,
            nose_length=1.5,
            nose_type="ogive",
            stl_alias="kh101",
        )
        spec = SweepSpec(vehicle="kh101", backend="dolfinx")
        coeffs = _dolfinx_solve(g, 1.5, 2.5, 0.0, 10e3, 0.0, spec)
        assert np.all(np.isfinite(coeffs))
        assert coeffs.shape == (5,)

    def test_dolfinx_solve_ss18_satan_produces_finite_coeffs(self):
        from project_icarus.aero.cfd_generators import (
            _dolfinx_solve,
            SweepSpec,
        )
        from project_icarus.aero.geometry import VehicleGeometry

        g = VehicleGeometry(
            name="SS-18 STL",
            body_length=16.85,
            body_diameter=3.0,
            nose_length=4.0,
            nose_type="blunt",
            stl_alias="ss18_satan",
        )
        spec = SweepSpec(vehicle="ss18_satan", backend="dolfinx")
        coeffs = _dolfinx_solve(g, 20.0, 0.0, 0.0, 10e3, 0.0, spec)
        assert np.all(np.isfinite(coeffs))
        assert coeffs.shape == (5,)

    def test_stl_sweep_and_gpr_training(self, tmp_path):
        import os
        from project_icarus.aero.cfd_generators import (
            SweepSpec,
            run_sweep,
            save_sweep_hdf5,
        )
        from project_icarus.aero.geometry import VehicleGeometry, get_vehicle
        from project_icarus.surrogates.train_gpr import train_vehicle_gpr

        orig = get_vehicle("kh101")
        patched = VehicleGeometry(
            name=orig.name + " STL",
            body_length=orig.body_length,
            body_diameter=orig.body_diameter,
            nose_length=orig.nose_length,
            nose_type=orig.nose_type,
            stl_alias="kh101",
        )

        import project_icarus.aero.geometry as geom_mod
        orig_get = geom_mod.get_vehicle

        def _patched(name):
            g = orig_get(name)
            if name == "kh101":
                return patched
            return g

        geom_mod.get_vehicle = _patched
        try:
            spec = SweepSpec(
                vehicle="kh101",
                backend="dolfinx",
                mach_range=(1.0, 2.0, 3),
                alpha_range=(0.0, 5.0, 2),
                beta_range=(0.0, 0.0, 1),
                altitude_range=(10e3, 10e3, 1),
                delta_range=(0.0, 0.0, 1),
            )
            result = run_sweep(spec)
            assert result["coeffs"].shape == (6, 5)
            assert np.all(np.isfinite(result["coeffs"]))

            h5_path = str(tmp_path / "stl_kh101.h5")
            save_sweep_hdf5(result, h5_path)
            assert os.path.exists(h5_path)

            model = train_vehicle_gpr("kh101_stl", h5_path=h5_path, base_dir=str(tmp_path))
            pred = model.predict(np.array([[1.5, 2.5, 0.0, 10e3, 0.0]]))
            assert pred.shape == (1, 5)
            assert np.all(np.isfinite(pred))
        finally:
            geom_mod.get_vehicle = orig_get
