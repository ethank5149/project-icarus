"""Parametric CAD geometry for interceptor / threat vehicles.

Generates OSINT-approximate surface meshes for the vehicles used across
Project Icarus (Arrow 3, Tamir, GMD-GBI, SM-3, generic threat RV) from
publicly available dimensions. Meshes are produced with Gmsh (lazy import)
and exposed as numpy vertex/face arrays so downstream CFD or ray-tracing
consumers do not depend on a single meshing library.

All dimensions are approximations drawn from open sources and are intended
for research / fidelity demonstration only. No controlled or ITAR data is
used; every vertex is generated from public length/diameter/fin inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class VehicleGeometry:
    """Parametric cone-cylinder-fin vehicle description (all SI / metres)."""

    name: str
    body_length: float
    body_diameter: float
    nose_length: float
    nose_type: str = "cone"  # "cone" | "ogive" | "blunt"
    nose_radius: float = 0.0  # blunt-nose spherical radius
    tail_length: float = 0.0  # boat-tail / flare length at rear
    fin_count: int = 0
    fin_span: float = 0.0  # fin half-span (root-to-tip)
    fin_root_chord: float = 0.0
    fin_tip_chord: float = 0.0
    fin_leading_edge: float = 0.0  # axial offset of fin root LE from tail
    fin_thickness: float = 0.0
    control_surface: bool = False  # movable rear control flaps
    control_span: float = 0.0
    control_chord: float = 0.0
    reference_area: float = 0.0  # cross-section ref area (computed if 0)
    source: str = "OSINT-approximate"
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_area <= 0.0:
            self.reference_area = 0.25 * np.pi * self.body_diameter**2
        self.metadata.setdefault("source", self.source)

    @property
    def length(self) -> float:
        return self.body_length

    @property
    def diameter(self) -> float:
        return self.body_diameter


# --- Public OSINT-approximate presets ------------------------------------- #
# Figures are representative public dimensions (diameter / length order of
# magnitude) and are NOT engineering drawings.
VEHICLE_PRESETS: Dict[str, VehicleGeometry] = {
    "arrow3": VehicleGeometry(
        name="Arrow 3",
        body_length=7.0,
        body_diameter=1.0,
        nose_length=1.4,
        nose_type="ogive",
        tail_length=0.6,
        fin_count=4,
        fin_span=0.55,
        fin_root_chord=1.2,
        fin_tip_chord=0.5,
        fin_leading_edge=0.4,
        fin_thickness=0.05,
        source="public estimates",
    ),
    "tamir": VehicleGeometry(
        name="Iron Dome Tamir",
        body_length=3.0,
        body_diameter=0.16,
        nose_length=0.4,
        nose_type="cone",
        tail_length=0.2,
        fin_count=4,
        fin_span=0.18,
        fin_root_chord=0.45,
        fin_tip_chord=0.18,
        fin_leading_edge=0.05,
        fin_thickness=0.012,
    ),
    "gmd_gbi": VehicleGeometry(
        name="GMD Ground-Based Interceptor",
        body_length=16.0,
        body_diameter=1.3,
        nose_length=3.0,
        nose_type="cone",
        tail_length=1.0,
        fin_count=0,
        source="public estimates",
    ),
    "sm3": VehicleGeometry(
        name="SM-3",
        body_length=6.5,
        body_diameter=0.34,
        nose_length=1.0,
        nose_type="cone",
        tail_length=0.4,
        fin_count=4,
        fin_span=0.16,
        fin_root_chord=0.5,
        fin_tip_chord=0.2,
        fin_leading_edge=0.1,
        fin_thickness=0.02,
    ),
    "threat_rv": VehicleGeometry(
        name="Generic Threat RV",
        body_length=1.8,
        body_diameter=0.5,
        nose_length=1.0,
        nose_type="blunt",
        nose_radius=0.25,
        tail_length=0.0,
        fin_count=0,
        source="OSINT representative",
    ),
}


def get_vehicle(name: str) -> VehicleGeometry:
    """Return a known vehicle preset by key (case-insensitive)."""
    key = name.lower().strip()
    if key not in VEHICLE_PRESETS:
        raise KeyError(
            f"Unknown vehicle '{name}'. Known: {sorted(VEHICLE_PRESETS)}"
        )
    return VEHICLE_PRESETS[key]


# --- Mesh generation ------------------------------------------------------ #
def _ogive_radius(x_rel: np.ndarray, nose_len: float, radius: float) -> np.ndarray:
    """Tangent ogive radius profile (0 at tip -> radius at base)."""
    rho = (radius**2 + nose_len**2) / (2.0 * radius)
    x = np.clip(x_rel, 0.0, nose_len)
    arg = np.clip(rho**2 - (nose_len - x) ** 2, 0.0, None)
    return np.sqrt(arg) + radius - rho


def body_profile(g: VehicleGeometry, n_axial: int = 80) -> np.ndarray:
    """Return (n_axial, 2) [z, r] axial profile of the body (nose->tail)."""
    z = np.linspace(0.0, g.body_length, n_axial)
    r = np.zeros(n_axial)
    nl = max(g.nose_length, 1e-6)
    in_nose = z < nl
    if g.nose_type == "cone":
        r[in_nose] = (z[in_nose] / nl) * (g.body_diameter / 2.0)
    elif g.nose_type == "ogive":
        r[in_nose] = _ogive_radius(z[in_nose], nl, g.body_diameter / 2.0)
    elif g.nose_type == "blunt":
        # Spherical-cap nose blended into cylinder.
        R = max(g.nose_radius, 1e-6)
        zn = z[in_nose]
        r[in_nose] = np.sqrt(np.clip(R**2 - (R - zn) ** 2, 0.0, None))
        # Allow partial spherical cap (tip height < R).
        cap_h = R - np.sqrt(max(R**2 - (g.body_diameter / 2.0) ** 2, 0.0))
        if cap_h > 0:
            r[in_nose] = np.sqrt(np.clip(R**2 - (R - (zn + (nl - cap_h))) ** 2, 0.0, None))
    else:
        raise ValueError(f"Unsupported nose_type {g.nose_type!r}")
    # Cylindrical mid-body.
    mid = (z >= nl) & (z <= g.body_length - g.tail_length)
    r[mid] = g.body_diameter / 2.0
    # Optional boat-tail / flare.
    if g.tail_length > 0:
        tail = z > g.body_length - g.tail_length
        zt = (z[tail] - (g.body_length - g.tail_length)) / g.tail_length
        r[tail] = (g.body_diameter / 2.0) * (1.0 - 0.3 * zt)
    return np.column_stack([z, r])


def build_surface_mesh(
    g: VehicleGeometry,
    n_axial: int = 80,
    n_circ: int = 48,
    backend: str = "numpy",
) -> "tuple[np.ndarray, np.ndarray]":
    """Build a surface mesh (vertices, faces) for the body + fins.

    Returns (vertices, faces):
      vertices : (V, 3) float array (x=circumferential, y=vertical, z=axial)
      faces    : (F, 3) int array of vertex indices

    `backend="gmsh"` builds a watertight CAD mesh via Gmsh (lazy import);
    `"numpy"` builds a lightweight parametric surface mesh that is sufficient
    for area / reference calculation and downstream ray or panel methods.
    """
    if backend == "gmsh":
        return _build_gmsh_mesh(g, n_axial, n_circ)
    return _build_numpy_mesh(g, n_axial, n_circ)


def _build_numpy_mesh(g: VehicleGeometry, n_axial: int, n_circ: int):
    prof = body_profile(g, n_axial)
    z = prof[:, 0]
    r = prof[:, 1]
    theta = np.linspace(0.0, 2.0 * np.pi, n_circ, endpoint=False)
    # Body vertices (n_axial x n_circ) with z along body axis.
    zz, tt = np.meshgrid(z, theta, indexing="ij")
    rr = np.broadcast_to(r[:, None], zz.shape)
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    body_v = np.stack([x, y, zz], axis=-1).reshape(-1, 3)
    n_body = body_v.shape[0]

    faces: List[List[int]] = []
    for i in range(n_axial - 1):
        for j in range(n_circ):
            j2 = (j + 1) % n_circ
            a = i * n_circ + j
            b = i * n_circ + j2
            c = (i + 1) * n_circ + j
            d = (i + 1) * n_circ + j2
            faces.append([a, b, c])
            faces.append([b, d, c])
    vertices = [body_v]

    # Simple flat fins as triangular prisms (root chord along body).
    if g.fin_count > 0 and g.fin_span > 0:
        le = g.body_length - g.tail_length - g.fin_leading_edge - g.fin_root_chord
        for k in range(g.fin_count):
            phi = 2.0 * np.pi * k / g.fin_count
            cphi, sphi = np.cos(phi), np.sin(phi)
            for sgn in (1.0, -1.0):
                # Parenthesized so sign multiplies only the radial offset term.
                fin = _fin_panel(g, le, cphi, sphi, sgn)
                base = n_body + sum(len(v) for v in vertices[1:])
                faces.append([base, base + 1, base + 2])
                faces.append([base + 1, base + 3, base + 2])
                faces.append([base + 4, base + 5, base + 6])
                faces.append([base + 5, base + 7, base + 6])
                vertices.append(fin)

    all_v = np.vstack(vertices) if len(vertices) > 1 else vertices[0]
    return all_v, np.asarray(faces, dtype=int)


def _fin_panel(g: VehicleGeometry, le_z: float, cphi: float, sphi: float, sgn: float):
    R = g.body_diameter / 2.0
    span = g.fin_span * sgn
    rc = g.fin_root_chord
    tc = g.fin_tip_chord
    th = max(g.fin_thickness, 1e-3)
    z0 = le_z
    # 4 corners: rootLE, rootTE, tipLE, tipTE (then mirrored for thickness).
    corners = np.array(
        [
            [R * cphi, R * sphi, z0],
            [R * cphi, R * sphi, z0 + rc],
            [(R + span) * cphi, (R + span) * sphi, z0 + (rc - tc) / 2.0],
            [(R + span) * cphi, (R + span) * sphi, z0 + (rc + tc) / 2.0],
        ]
    )
    # Thickness offset along body normal (radial).
    nrm = np.array([cphi, sphi, 0.0])
    thick = np.outer(np.array([0.0, 0.0, th, th]), nrm)
    return np.vstack([corners, corners + thick])


def _build_gmsh_mesh(g: VehicleGeometry, n_axial: int, n_circ: int):
    """Watertight CAD mesh via Gmsh (lazy import; needs `gmsh` installed)."""
    try:
        import gmsh  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Gmsh backend requires the 'gmsh' package") from exc
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    prof = body_profile(g, n_axial)
    pts = []
    for z, r in prof:
        pts.append(gmsh.model.geo.addPoint(r, 0.0, z, 1.0))
    spline = gmsh.model.geo.addSpline(pts)
    loop = gmsh.model.geo.addCurveLoop([spline])
    surf = gmsh.model.geo.addSurfaceFilling([loop])
    gmsh.model.geo.extrude(
        [(2, surf)], dx=0.0, dy=0.0, dz=0.0,
        axis=[0.0, 0.0, 1.0], angle=2.0 * np.pi,
        numElements=[n_circ], recombine=False,
    )
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    nodes = gmsh.model.mesh.getNodes()
    coords = nodes[1].reshape(-1, 3)
    elem_types, _, elem_tags = gmsh.model.mesh.getElements(2)
    faces = np.concatenate([t.reshape(-1, 3) - 1 for t in elem_tags]) if elem_tags else np.zeros((0, 3), dtype=int)
    gmsh.finalize()
    return coords, faces


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Total surface area of a triangle mesh (mean edge cross product)."""
    v = vertices[faces]
    a = v[:, 1] - v[:, 0]
    b = v[:, 2] - v[:, 0]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def wetted_area(g: VehicleGeometry, **mesh_kw) -> float:
    """Approximate wetted area for reference-length scaling."""
    v, f = build_surface_mesh(g, **mesh_kw)
    return surface_area(v, f)


def reference_dimensions(g: VehicleGeometry) -> "tuple[float, float, float]":
    """(reference_area, reference_length, reference_span)."""
    return g.reference_area, g.body_diameter, g.body_length
