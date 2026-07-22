"""Lightweight STL loader and normalizer for CAD meshes.

All operations use only numpy; no external mesh libraries are required.
Output is in meters.
"""

from __future__ import annotations

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
    """Normalize STL mesh: center at origin, align longest axis to +Z, scale."""
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


_MANIFEST: Dict[str, VehicleSTLInfo] = {
    "kh101": VehicleSTLInfo(
        key="kh101",
        filename="kh-101-cruise-missile.stl",
        scale=0.001,
        proxy_for="kh101",
        notes="OSINT Kh-101 cruise missile. Centered, mm units.",
    ),
    "ss18_satan": VehicleSTLInfo(
        key="ss18_satan",
        filename="ss-18-satan-missile.stl",
        scale=2.0181,
        proxy_for="ss18_satan",
        notes="OSINT RS-36 SS-18 Satan. Scaled to 34.0 m preset length.",
    ),
    "topol_m": VehicleSTLInfo(
        key="topol_m",
        filename="topol-m-missile.stl",
        scale=2.3936,
        proxy_for="topol_m",
        notes="OSINT RT-2PM2 Topol-M. Z-oriented, scaled to 22.0 m preset length.",
    ),
    "df41": VehicleSTLInfo(
        key="df41",
        filename="df-41-missile.stl",
        scale=2.2694,
        proxy_for="df41",
        notes="OSINT DF-41. Y-centroid offset present; normalizer recenters. Scaled to 16.5 m preset length.",
    ),
}

CAD_MANIFEST: Dict[str, VehicleSTLInfo] = dict(_MANIFEST)
