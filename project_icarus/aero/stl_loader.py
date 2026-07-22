"""Lightweight STL loader and normalizer for CAD meshes.

All operations use only numpy; no external mesh libraries are required.
Output is in meters.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VehicleSTLInfo:
    key: str
    filename: str
    scale: float
    force_axis: Optional[int] = None  # 0=X, 1=Y, 2=Z
    proxy_for: str = ""
    notes: str = ""
    original_filename: str = ""
    raw_scale: float = 1.0


def read_stl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read binary or ASCII STL and return (vertices, faces) as float64.

    For binary STL, triangle vertex ordering is preserved so face normals
    remain consistent with the file.
    """
    path = str(path)
    with open(path, "rb") as f:
        header = f.read(80)

    if header[:5].lower() == b"solid":
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            text = f.read()
        return _parse_ascii_stl(text)

    return _parse_binary_stl(path)


def _parse_binary_stl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "rb") as f:
        f.read(80)
        data = f.read(4)
        if len(data) < 4:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=int)
        num_triangles = int.from_bytes(data, "little")
        for _ in range(num_triangles):
            chunk = f.read(50)
            if len(chunk) < 50:
                break
            nx, ny, nz, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z = (
                np.frombuffer(chunk[:48], dtype=np.float32)
            )
            base = len(verts)
            verts.append([v1x, v1y, v1z])
            verts.append([v2x, v2y, v2z])
            verts.append([v3x, v3y, v3z])
            faces.append([base, base + 1, base + 2])
    vertices = np.array(verts, dtype=np.float64).reshape(-1, 3)
    faces_arr = np.array(faces, dtype=int)
    return vertices, faces_arr


def _parse_ascii_stl(text: str) -> Tuple[np.ndarray, np.ndarray]:
    import re

    pattern = re.compile(
        r"vertex\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
    )
    matches = pattern.findall(text)
    verts: list[list[float]] = []
    for m in matches:
        verts.append([float(m[0]), float(m[1]), float(m[2])])
    vertices = np.array(verts, dtype=np.float64)
    faces = np.arange(vertices.shape[0], dtype=int).reshape(-1, 3)
    return vertices, faces


def normalize_stl(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    scale: float = 1.0,
    force_axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize STL mesh: center at origin, align longest axis to +Z, scale.

    After axis alignment the function automatically detects the nose direction
    by comparing average cross-section radius at each end of the longitudinal
    axis. The end with smaller radius is assumed to be the nose and is placed
    at +Z so forward always points upward.
    """
    if vertices.size == 0:
        return vertices.copy(), faces.copy()

    vertices = vertices.astype(np.float64, copy=True)
    faces = faces.astype(int, copy=True)

    vertices *= float(scale)

    centroid = vertices.mean(axis=0)
    vertices -= centroid

    extents = vertices.max(axis=0) - vertices.min(axis=0)
    if extents.max() <= 0.0:
        return vertices, faces

    primary_axis = int(force_axis) if force_axis is not None else int(np.argmax(extents))
    target_axis = 2  # +Z

    if primary_axis != target_axis:
        perm = [0, 1, 2]
        src = primary_axis
        dst = target_axis
        # rotate axes so src -> dst
        for _ in range(3):
            if perm[src] == dst:
                break
            perm[dst], perm[src] = perm[src], perm[dst]
        vertices = vertices[:, perm]
        extents = vertices.max(axis=0) - vertices.min(axis=0)

    if extents[target_axis] < 0.0:
        vertices[:, target_axis] *= -1.0

    # Orient so the nose (smaller cross-section radius) points to +Z
    z_min = float(vertices[:, target_axis].min())
    z_max = float(vertices[:, target_axis].max())
    slab = 0.05 * (z_max - z_min)
    if slab > 1e-9:
        nose_slice = vertices[:, target_axis] < z_min + slab
        tail_slice = vertices[:, target_axis] > z_max - slab
        nose_r = float(np.linalg.norm(vertices[nose_slice][:, :2], axis=1).mean()) if nose_slice.any() else 0.0
        tail_r = float(np.linalg.norm(vertices[tail_slice][:, :2], axis=1).mean()) if tail_slice.any() else 0.0
        if nose_r < tail_r and z_min < z_max:
            vertices[:, target_axis] *= -1.0

    return vertices, faces


def load_cad_vehicle(
    key: str,
    manifest: Optional[Dict[str, VehicleSTLInfo]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and normalize a CAD vehicle by manifest key."""
    if manifest is None:
        manifest = CAD_MANIFEST
    info = manifest.get(key)
    if info is None:
        raise KeyError(f"No CAD manifest entry for '{key}'. Known: {sorted(manifest)}")

    base = Path(__file__).resolve().parent.parent.parent / "reference" / "models"
    path = base / info.filename
    if not path.exists():
        raise FileNotFoundError(f"STL not found: {path}")

    vertices, faces = read_stl(str(path))
    vertices, faces = normalize_stl(
        vertices,
        faces,
        scale=float(info.scale),
        force_axis=info.force_axis,
    )
    return vertices.astype(np.float64), faces.astype(int)


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Total surface area of a triangle mesh."""
    if faces.size == 0:
        return 0.0
    tri = vertices[faces]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def write_stl(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary STL file from vertices and faces."""
    path = str(path)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(np.uint32(faces.shape[0]).tobytes())
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            a = v1 - v0
            b = v2 - v0
            normal = np.cross(a, b)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                normal = normal / norm_len
            else:
                normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            chunk = struct.pack(
                "<12fH",
                float(normal[0]), float(normal[1]), float(normal[2]),
                float(v0[0]), float(v0[1]), float(v0[2]),
                float(v1[0]), float(v1[1]), float(v1[2]),
                float(v2[0]), float(v2[1]), float(v2[2]),
                0,
            )
            f.write(chunk)


def normalized_path_for(key: str, base_dir: Optional[Path] = None) -> Path:
    """Return path to canonical normalized STL for a manifest key."""
    if base_dir is None:
        base_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "reference" / "vehicles" / key
        )
    info = CAD_MANIFEST.get(key)
    if info is None:
        raise KeyError(f"No CAD manifest entry for '{key}'")
    return base_dir / "model.stl"


def ensure_normalized(key: str, force: bool = False) -> Path:
    """Ensure a normalized STL exists on disk for the given manifest key.

    If the normalized file already exists and ``force`` is False, returns
    the existing path. Otherwise reads the raw STL, normalizes it, writes
    the canonical file, and returns its path.
    """
    out_path = normalized_path_for(key)
    if out_path.exists() and not force:
        return out_path

    info = CAD_MANIFEST[key]
    vertices, faces = read_stl_for_key(key)
    vertices, faces = normalize_stl(
        vertices,
        faces,
        scale=float(info.raw_scale if info.raw_scale else info.scale),
        force_axis=info.force_axis,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_stl(str(out_path), vertices, faces)
    return out_path


def read_stl_for_key(key: str, manifest: Optional[Dict[str, VehicleSTLInfo]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Read raw STL for a manifest key without normalizing."""
    if manifest is None:
        manifest = CAD_MANIFEST
    info = manifest.get(key)
    if info is None:
        raise KeyError(f"No CAD manifest entry for '{key}'. Known: {sorted(manifest)}")

    base = Path(__file__).resolve().parent.parent.parent / "reference" / "models"
    filename = info.original_filename if info.original_filename else info.filename
    # If manifest points to normalized, still fall back to original raw source
    if filename.startswith("normalized/") and not info.original_filename:
        filename = Path(info.filename).name
    path = base / filename
    if not path.exists():
        raise FileNotFoundError(f"STL not found: {path}")
    return read_stl(str(path))


def load_cad_vehicle(
    key: str,
    manifest: Optional[Dict[str, VehicleSTLInfo]] = None,
    prefer_normalized: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CAD vehicle by manifest key, preferring normalized on-disk assets."""
    if manifest is None:
        manifest = CAD_MANIFEST
    info = manifest.get(key)
    if info is None:
        raise KeyError(f"No CAD manifest entry for '{key}'. Known: {sorted(manifest)}")

    if prefer_normalized:
        norm_path = normalized_path_for(key)
        if norm_path.exists():
            return read_stl(str(norm_path))

    base = Path(__file__).resolve().parent.parent.parent / "reference" / "models"
    path = base / info.original_filename
    if not path.exists():
        raise FileNotFoundError(f"STL not found: {path}")

    vertices, faces = read_stl(str(path))
    vertices, faces = normalize_stl(
        vertices,
        faces,
        scale=float(info.scale),
        force_axis=info.force_axis,
    )
    return vertices.astype(np.float64), faces.astype(int)


_MANIFEST: Dict[str, VehicleSTLInfo] = {
    "kh101": VehicleSTLInfo(
        key="kh101",
        filename="model.stl",
        scale=1.0,
        proxy_for="kh101",
        notes="OSINT Kh-101 cruise missile. Normalized: centered, +Z axis, 7.446 m.",
        original_filename="kh101.stl",
        raw_scale=0.001,
    ),
    "ss18_satan": VehicleSTLInfo(
        key="ss18_satan",
        filename="model.stl",
        scale=1.0,
        proxy_for="ss18_satan",
        notes="OSINT RS-36 SS-18 Satan. Normalized: centered, +Z axis, 34.004 m.",
        original_filename="ss18-satan.stl",
        raw_scale=2.0181,
    ),
    "topol_m": VehicleSTLInfo(
        key="topol_m",
        filename="model.stl",
        scale=1.0,
        proxy_for="topol_m",
        notes="OSINT RT-2PM2 Topol-M. Normalized: centered, +Z axis, 22.708 m.",
        original_filename="topolm.stl",
        raw_scale=2.4705,
    ),
    "df41": VehicleSTLInfo(
        key="df41",
        filename="model.stl",
        scale=1.0,
        proxy_for="df41",
        notes="OSINT DF-41. Normalized: centered, +Z axis, 16.503 m.",
        original_filename="df41.stl",
        raw_scale=2.2694,
    ),
    "df5": VehicleSTLInfo(
        key="df5",
        filename="model.stl",
        scale=1.0,
        proxy_for="df5",
        notes="OSINT DF-5A. Normalized: centered, +Z axis, 32.004 m.",
        original_filename="df5a.stl",
        raw_scale=1.1461,
    ),
    "df31": VehicleSTLInfo(
        key="df31",
        filename="model.stl",
        scale=1.0,
        proxy_for="df31",
        notes="OSINT DF-31AG. Normalized: centered, +Z axis, 12.999 m.",
        original_filename="df31ag.stl",
        raw_scale=1.5758,
    ),
    "kinzhal": VehicleSTLInfo(
        key="kinzhal",
        filename="model.stl",
        scale=1.0,
        proxy_for="kinzhal",
        notes="OSINT Kh-47M2 Kinzhal. Normalized: centered, +Z axis, 7.798 m.",
        original_filename="kh47m2.stl",
        raw_scale=0.4830,
    ),
    "aim120": VehicleSTLInfo(
        key="aim120",
        filename="model.stl",
        scale=1.0,
        proxy_for="aim120",
        notes="OSINT AIM-120 (AMRAAM). Normalized: centered, +Z axis, 3.650 m.",
        original_filename="aim120-amraam.stl",
        raw_scale=0.0106,
    ),
    "atacms": VehicleSTLInfo(
        key="atacms",
        filename="model.stl",
        scale=1.0,
        proxy_for="atacms",
        notes="OSINT MGM-140 (ATACMS). Normalized: centered, +Z axis, 4.000 m.",
        original_filename="mgm140-atacms.stl",
        raw_scale=0.0028,
    ),
    "tomahawk": VehicleSTLInfo(
        key="tomahawk",
        filename="model.stl",
        scale=1.0,
        proxy_for="tomahawk",
        notes="OSINT BGM-109 (Tomahawk). Normalized: centered, +Z axis, 5.560 m.",
        original_filename="bgm109-tomahawk.stl",
        raw_scale=0.0429,
    ),
    "harpoon": VehicleSTLInfo(
        key="harpoon",
        filename="model.stl",
        scale=1.0,
        proxy_for="harpoon",
        notes="OSINT RGM-84 (Harpoon). Normalized: centered, +Z axis, 4.600 m.",
        original_filename="rgm84-harpoon.stl",
        raw_scale=0.3803,
    ),
    "hwasong12": VehicleSTLInfo(
        key="hwasong12",
        filename="model.stl",
        scale=1.0,
        proxy_for="hwasong12",
        notes="OSINT Hwasong-12 (KN-17). Normalized: centered, +Z axis, 17.400 m.",
        original_filename="hwasong12.stl",
        raw_scale=1.0886,
    ),
    "hwasong14": VehicleSTLInfo(
        key="hwasong14",
        filename="model.stl",
        scale=1.0,
        proxy_for="hwasong14",
        notes="OSINT Hwasong-14 (KN-20). Normalized: centered, +Z axis, 19.800 m.",
        original_filename="hwasong14.stl",
        raw_scale=1.1161,
    ),
    "hwasong15": VehicleSTLInfo(
        key="hwasong15",
        filename="model.stl",
        scale=1.0,
        proxy_for="hwasong15",
        notes="OSINT Hwasong-15 (KN-22). Normalized: centered, +Z axis, 22.000 m.",
        original_filename="hwasong15.stl",
        raw_scale=1.1180,
    ),
    "hwasong17": VehicleSTLInfo(
        key="hwasong17",
        filename="model.stl",
        scale=1.0,
        proxy_for="hwasong17",
        notes="OSINT Hwasong-17. Normalized: centered, +Z axis, 25.000 m.",
        original_filename="hwasong17.stl",
        raw_scale=2.1315,
    ),
    "kh47m2": VehicleSTLInfo(
        key="kh47m2",
        filename="model.stl",
        scale=1.0,
        proxy_for="kh47m2",
        notes="OSINT Kh-47M2 (Kinzhal). Normalized: centered, +Z axis, 8.000 m.",
        original_filename="kh47m2.stl",
        raw_scale=0.4955,
    ),
    "slam": VehicleSTLInfo(
        key="slam",
        filename="model.stl",
        scale=1.0,
        proxy_for="slam",
        notes="OSINT SLAM (Project Pluto). Normalized: centered, +Z axis.",
        original_filename="project-pluto-slam.stl",
        raw_scale=1.0000,
    ),
    "skybolt": VehicleSTLInfo(
        key="skybolt",
        filename="model.stl",
        scale=1.0,
        proxy_for="skybolt",
        notes="OSINT GAM-87 (AGM-48) Skybolt. Normalized: centered, +Z axis, 11.660 m.",
        original_filename="skybolt.stl",
        raw_scale=0.9777,
    ),
    "titan2": VehicleSTLInfo(
        key="titan2",
        filename="model.stl",
        scale=1.0,
        proxy_for="titan2",
        notes="OSINT LGM-25C Titan II. Normalized: centered, +Z axis, 32.920 m.",
        original_filename="titan-ii.stl",
        raw_scale=1.2953,
    ),
    "lance": VehicleSTLInfo(
        key="lance",
        filename="model.stl",
        scale=1.0,
        proxy_for="lance",
        notes="OSINT MGM-52 (Lance). Normalized: centered, +Z axis, 6.410 m.",
        original_filename="mgm52.stl",
        raw_scale=0.5572,
    ),
    "corporal": VehicleSTLInfo(
        key="corporal",
        filename="model.stl",
        scale=1.0,
        proxy_for="corporal",
        notes="OSINT MGM-5 Corporal. Normalized: centered, +Z axis, 13.700 m.",
        original_filename="mgm5-corporal.stl",
        raw_scale=0.0010,
    ),
    "midgetman": VehicleSTLInfo(
        key="midgetman",
        filename="model.stl",
        scale=1.0,
        proxy_for="midgetman",
        notes="OSINT MGM-134A (Midgetman). Normalized: centered, +Z axis, 14.000 m.",
        original_filename="mgm134a-midgetman.stl",
        raw_scale=1.2058,
    ),
    "pac2": VehicleSTLInfo(
        key="pac2",
        filename="model.stl",
        scale=1.0,
        proxy_for="pac2",
        notes="OSINT MIM-104 PAC-2 (Patriot). Normalized: centered, +Z axis, 5.790 m.",
        original_filename="pac2.stl",
        raw_scale=0.3944,
    ),
    "pac3_cri": VehicleSTLInfo(
        key="pac3_cri",
        filename="model.stl",
        scale=1.0,
        proxy_for="pac3_cri",
        notes="OSINT MIM-104 PAC-3 CRI. Normalized: centered, +Z axis, 5.200 m.",
        original_filename="pac3cri.stl",
        raw_scale=0.3505,
    ),
    "pac3_mse": VehicleSTLInfo(
        key="pac3_mse",
        filename="model.stl",
        scale=1.0,
        proxy_for="pac3_mse",
        notes="OSINT MIM-104 PAC-3 MSE. Normalized: centered, +Z axis, 5.200 m.",
        original_filename="pac3mse.stl",
        raw_scale=0.3494,
    ),
    "redstone": VehicleSTLInfo(
        key="redstone",
        filename="model.stl",
        scale=1.0,
        proxy_for="redstone",
        notes="OSINT PGM-11 Redstone. Normalized: centered, +Z axis, 21.100 m.",
        original_filename="pgm11-redstone.stl",
        raw_scale=1.7684,
    ),
    "v2": VehicleSTLInfo(
        key="v2",
        filename="model.stl",
        scale=1.0,
        proxy_for="v2",
        notes="OSINT V-2 (A-4 rocket). Normalized: centered, +Z axis, 14.000 m.",
        original_filename="v2.stl",
        raw_scale=0.8800,
    ),
    "ababeel": VehicleSTLInfo(
        key="ababeel",
        filename="model.stl",
        scale=1.0,
        proxy_for="ababeel",
        notes="OSINT Ababeel. Normalized: centered, +Z axis.",
        original_filename="ababeel.stl",
        raw_scale=1.0000,
    ),
    "jassm": VehicleSTLInfo(
        key="jassm",
        filename="model.stl",
        scale=1.0,
        proxy_for="jassm",
        notes="OSINT AGM-158 (JASSM). Normalized: centered, +Z axis, 4.270 m.",
        original_filename="agm158.stl",
        raw_scale=0.0247,
    ),
    "sram": VehicleSTLInfo(
        key="sram",
        filename="model.stl",
        scale=1.0,
        proxy_for="sram",
        notes="OSINT AGM-69A (SRAM). Normalized: centered, +Z axis, 4.830 m.",
        original_filename="agm69a-sram.stl",
        raw_scale=0.0288,
    ),
    "thor": VehicleSTLInfo(
        key="thor",
        filename="model.stl",
        scale=1.0,
        proxy_for="thor",
        notes="OSINT PGM-17 Thor. Normalized: centered, +Z axis, 19.810 m.",
        original_filename="thor.stl",
        raw_scale=1.9973,
    ),
    "jupiter": VehicleSTLInfo(
        key="jupiter",
        filename="model.stl",
        scale=1.0,
        proxy_for="jupiter",
        notes="OSINT PGM-19 Jupiter. Normalized: centered, +Z axis, 18.390 m.",
        original_filename="jupiter.stl",
        raw_scale=1.8035,
    ),
    "titan1": VehicleSTLInfo(
        key="titan1",
        filename="model.stl",
        scale=1.0,
        proxy_for="titan1",
        notes="OSINT HGM-25A Titan I. Normalized: centered, +Z axis, 29.870 m.",
        original_filename="titan-i.stl",
        raw_scale=1.2311,
    ),
    "pershing2": VehicleSTLInfo(
        key="pershing2",
        filename="model.stl",
        scale=1.0,
        proxy_for="pershing2",
        notes="OSINT Pershing II. Normalized: centered, +Z axis, 10.610 m.",
        original_filename="pershing-ii.stl",
        raw_scale=0.1262,
    ),
    "qiam1": VehicleSTLInfo(
        key="qiam1",
        filename="model.stl",
        scale=1.0,
        proxy_for="qiam1",
        notes="OSINT Qiam-1. Normalized: centered, +Z axis, 11.850 m.",
        original_filename="qiam-1.stl",
        raw_scale=1.1810,
    ),
    "r14": VehicleSTLInfo(
        key="r14",
        filename="model.stl",
        scale=1.0,
        proxy_for="r14",
        notes="OSINT R-14 (SS-5 Skean). Normalized: centered, +Z axis, 24.400 m.",
        original_filename="r14.stl",
        raw_scale=1.5788,
    ),
    "scudb": VehicleSTLInfo(
        key="scudb",
        filename="model.stl",
        scale=1.0,
        proxy_for="scudb",
        notes="OSINT Scud-B (R-17E). Normalized: centered, +Z axis, 11.250 m.",
        original_filename="scudb.stl",
        raw_scale=0.9742,
    ),
    "shaheen3": VehicleSTLInfo(
        key="shaheen3",
        filename="model.stl",
        scale=1.0,
        proxy_for="shaheen3",
        notes="OSINT Shaheen-3. Normalized: centered, +Z axis, 19.300 m.",
        original_filename="shaheen3.stl",
        raw_scale=2.4610,
    ),

    "df3a": VehicleSTLInfo(
        key="df3a",
        filename="model.stl",
        scale=1.0,
        proxy_for="df3a",
        notes="OSINT DF-3A (CSS-2). Normalized: centered, +Z axis, 24.000 m.",
        original_filename="df3a.stl",
        raw_scale=1.1093,
    ),
    "df3": VehicleSTLInfo(
        key="df3",
        filename="model.stl",
        scale=1.0,
        proxy_for="df3",
        notes="OSINT DF-3 (CSS-2). Normalized: centered, +Z axis, 24.000 m.",
        original_filename="df3.stl",
        raw_scale=1.1853,
    ),
    "df5a": VehicleSTLInfo(
        key="df5a",
        filename="model.stl",
        scale=1.0,
        proxy_for="df5a",
        notes="OSINT DF-5A (CSS-4). Normalized: centered, +Z axis, 32.600 m.",
        original_filename="df5a.stl",
        raw_scale=1.1674,
    ),
    "df5b": VehicleSTLInfo(
        key="df5b",
        filename="model.stl",
        scale=1.0,
        proxy_for="df5b",
        notes="OSINT DF-5B (CSS-4). Normalized: centered, +Z axis, 32.600 m.",
        original_filename="df5b.stl",
        raw_scale=1.1577,
    ),
    "df31ag": VehicleSTLInfo(
        key="df31ag",
        filename="model.stl",
        scale=1.0,
        proxy_for="df31ag",
        notes="OSINT DF-31AG. Normalized: centered, +Z axis, 14.500 m.",
        original_filename="df31ag.stl",
        raw_scale=1.7578,
    ),
    "mx": VehicleSTLInfo(
        key="mx",
        filename="model.stl",
        scale=1.0,
        proxy_for="mx",
        notes="OSINT LGM-118 (MX). Normalized: centered, +Z axis, 21.100 m.",
        original_filename="lgm118.stl",
        raw_scale=2.8620,
    ),
    "minuteman2": VehicleSTLInfo(
        key="minuteman2",
        filename="model.stl",
        scale=1.0,
        proxy_for="minuteman2",
        notes="OSINT LGM-30F (Minuteman II). Normalized: centered, +Z axis, 18.200 m.",
        original_filename="lgm30f.stl",
        raw_scale=2.1037,
    ),
    "minuteman3": VehicleSTLInfo(
        key="minuteman3",
        filename="model.stl",
        scale=1.0,
        proxy_for="minuteman3",
        notes="OSINT LGM-30G (Minuteman III). Normalized: centered, +Z axis, 18.200 m.",
        original_filename="lgm30g.stl",
        raw_scale=1.9320,
    ),
    "minuteman1a": VehicleSTLInfo(
        key="minuteman1a",
        filename="model.stl",
        scale=1.0,
        proxy_for="minuteman1a",
        notes="OSINT LGM-30A (Minuteman I). Normalized: centered, +Z axis, 16.450 m.",
        original_filename="lgm30a.stl",
        raw_scale=2.0555,
    ),
    "minuteman1b": VehicleSTLInfo(
        key="minuteman1b",
        filename="model.stl",
        scale=1.0,
        proxy_for="minuteman1b",
        notes="OSINT LGM-30B (Minuteman I). Normalized: centered, +Z axis, 17.000 m.",
        original_filename="lgm30b.stl",
        raw_scale=2.0625,
    ),
    "rs28_sarmat": VehicleSTLInfo(
        key="rs28_sarmat",
        filename="model.stl",
        scale=1.0,
        proxy_for="rs28_sarmat",
        notes="OSINT RS-28 Sarmat. Normalized: centered, +Z axis, 35.000 m.",
        original_filename="rs28-sarmat.stl",
        raw_scale=0.025392,
    ),
}

CAD_MANIFEST: Dict[str, VehicleSTLInfo] = dict(_MANIFEST)
