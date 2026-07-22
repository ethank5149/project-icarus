"""Parametric CAD geometry for interceptor / threat vehicles.

Generates OSINT-approximate surface meshes for the vehicles used across
Project Icarus (Arrow 2, Arrow 3, GMD GBI, GMD CE-2, Patriot PAC-3 MSE, Tamir,
THAAD, SM-3, RS-28 Sarmat, Avangard, Kalibr, Kh-101/102, Zircon, Burevestnik,
CJ-10, CJ-1000, YJ-12, YJ-18, plus decoy/swarm buses) from publicly available
dimensions. Meshes are produced with Gmsh (lazy import) and exposed as numpy
vertex/face arrays so downstream CFD or ray-tracing consumers do not depend on
a single meshing library.

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
    control_deflection_deg: float = 0.0  # reversible control deflection
    reference_area: float = 0.0  # cross-section ref area (computed if 0)
    reference_area_override: float = 0.0  # manual ref area override
    boattail_fraction: float = 0.3  # diameter reduction over tail_length
    body_flair_deg: float = 0.0  # body-side fairing half-angle (deg)
    bus_length: float = 0.0  # post-boost bus length for composite shapes
    bus_diameter: float = 0.0  # bus diameter for composite shapes
    bus_nose: str = "ogive"  # bus nose type
    bus_tail: str = "cone"  # bus tail type
    source: str = "OSINT-approximate"
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_area <= 0.0 and self.reference_area_override > 0.0:
            self.reference_area = self.reference_area_override
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
    # --- Interceptors ---
    "arrow2": VehicleGeometry(
        name="Arrow 2",
        body_length=5.5,
        body_diameter=0.8,
        nose_length=1.0,
        nose_type="ogive",
        tail_length=0.5,
        fin_count=4,
        fin_span=0.45,
        fin_root_chord=1.0,
        fin_tip_chord=0.4,
        fin_leading_edge=0.3,
        fin_thickness=0.04,
        source="public estimates",
    ),
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
    "gmd_ce2": VehicleGeometry(
        name="GMD CE-2",
        body_length=16.5,
        body_diameter=1.3,
        nose_length=3.2,
        nose_type="cone",
        tail_length=1.1,
        fin_count=0,
        source="public estimates",
    ),
    "patriot": VehicleGeometry(
        name="PAC-3 MSE",
        body_length=5.2,
        body_diameter=0.26,
        nose_length=0.7,
        nose_type="cone",
        tail_length=0.3,
        fin_count=8,
        fin_span=0.18,
        fin_root_chord=0.5,
        fin_tip_chord=0.2,
        fin_leading_edge=0.1,
        fin_thickness=0.015,
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
    "thaad": VehicleGeometry(
        name="THAAD",
        body_length=6.3,
        body_diameter=0.34,
        nose_length=0.9,
        nose_type="ogive",
        tail_length=0.4,
        fin_count=4,
        fin_span=0.22,
        fin_root_chord=0.55,
        fin_tip_chord=0.22,
        fin_leading_edge=0.12,
        fin_thickness=0.02,
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
    # --- Strategic / hypersonic threats ---
    "rs28_sarmat": VehicleGeometry(
        name="RS-28 Sarmat",
        body_length=35.0,
        body_diameter=3.0,
        nose_length=8.0,
        nose_type="blunt",
        nose_radius=1.5,
        tail_length=3.0,
        fin_count=0,
        source="OSINT representative",
    ),
    "sarmat_bus": VehicleGeometry(
        name="Sarmat post-boost bus",
        body_length=4.0,
        body_diameter=3.0,
        nose_length=1.2,
        nose_type="ogive",
        bus_length=3.5,
        bus_diameter=2.4,
        bus_nose="ogive",
        bus_tail="cone",
        source="OSINT representative",
    ),
    "avangard": VehicleGeometry(
        name="Avangard HGV",
        body_length=5.4,
        body_diameter=0.75,
        nose_length=1.8,
        nose_type="blunt",
        nose_radius=0.4,
        tail_length=0.8,
        fin_count=0,
        source="OSINT representative",
    ),
    # --- Cruise / ASCM ---
    "kalibr_3m14": VehicleGeometry(
        name="Kalibr 3M-14",
        body_length=6.2,
        body_diameter=0.51,
        nose_length=1.2,
        nose_type="ogive",
        tail_length=0.6,
        fin_count=4,
        fin_span=0.25,
        fin_root_chord=0.5,
        fin_tip_chord=0.2,
        fin_leading_edge=0.15,
        fin_thickness=0.02,
        source="public estimates",
    ),
    "kh101": VehicleGeometry(
        name="Kh-101",
        body_length=7.45,
        body_diameter=0.74,
        nose_length=1.5,
        nose_type="ogive",
        tail_length=0.8,
        fin_count=4,
        fin_span=0.3,
        fin_root_chord=0.6,
        fin_tip_chord=0.25,
        fin_leading_edge=0.2,
        fin_thickness=0.025,
        source="public estimates",
    ),
    "kh102": VehicleGeometry(
        name="Kh-102",
        body_length=7.45,
        body_diameter=0.74,
        nose_length=1.5,
        nose_type="ogive",
        tail_length=0.8,
        fin_count=4,
        fin_span=0.3,
        fin_root_chord=0.6,
        fin_tip_chord=0.25,
        fin_leading_edge=0.2,
        fin_thickness=0.025,
        source="OSINT representative",
    ),
    "zircon_3m22": VehicleGeometry(
        name="3M22 Zircon",
        body_length=8.0,
        body_diameter=0.77,
        nose_length=1.6,
        nose_type="blunt",
        nose_radius=0.3,
        tail_length=0.7,
        fin_count=4,
        fin_span=0.3,
        fin_root_chord=0.7,
        fin_tip_chord=0.3,
        fin_leading_edge=0.25,
        fin_thickness=0.025,
        source="public estimates",
    ),
    "burevestnik": VehicleGeometry(
        name="Burevestnik",
        body_length=10.0,
        body_diameter=0.95,
        nose_length=1.8,
        nose_type="ogive",
        tail_length=1.0,
        fin_count=4,
        fin_span=0.35,
        fin_root_chord=0.8,
        fin_tip_chord=0.35,
        fin_leading_edge=0.3,
        fin_thickness=0.03,
        source="OSINT representative",
    ),
    "cj10": VehicleGeometry(
        name="CJ-10",
        body_length=6.5,
        body_diameter=0.75,
        nose_length=1.3,
        nose_type="ogive",
        tail_length=0.7,
        fin_count=4,
        fin_span=0.28,
        fin_root_chord=0.55,
        fin_tip_chord=0.22,
        fin_leading_edge=0.2,
        fin_thickness=0.022,
        source="public estimates",
    ),
    "cj1000": VehicleGeometry(
        name="CJ-1000",
        body_length=7.5,
        body_diameter=0.82,
        nose_length=1.5,
        nose_type="ogive",
        tail_length=0.9,
        fin_count=4,
        fin_span=0.32,
        fin_root_chord=0.65,
        fin_tip_chord=0.28,
        fin_leading_edge=0.25,
        fin_thickness=0.025,
        source="OSINT representative",
    ),
    "yj12": VehicleGeometry(
        name="YJ-12",
        body_length=5.5,
        body_diameter=0.55,
        nose_length=1.1,
        nose_type="ogive",
        tail_length=0.5,
        fin_count=4,
        fin_span=0.22,
        fin_root_chord=0.45,
        fin_tip_chord=0.18,
        fin_leading_edge=0.15,
        fin_thickness=0.018,
        source="public estimates",
    ),
    "yj18": VehicleGeometry(
        name="YJ-18",
        body_length=6.0,
        body_diameter=0.54,
        nose_length=1.2,
        nose_type="ogive",
        tail_length=0.6,
        fin_count=4,
        fin_span=0.24,
        fin_root_chord=0.5,
        fin_tip_chord=0.2,
        fin_leading_edge=0.18,
        fin_thickness=0.02,
        source="public estimates",
    ),
    # --- Decoys / buses ---
    "signature_balloon_decoy": VehicleGeometry(
        name="Signature balloon decoy",
        body_length=1.5,
        body_diameter=1.5,
        nose_length=0.5,
        nose_type="blunt",
        nose_radius=0.75,
        tail_length=0.5,
        fin_count=0,
        source="OSINT representative",
    ),
    "foam_decoy": VehicleGeometry(
        name="Foam reentry decoy",
        body_length=0.8,
        body_diameter=0.5,
        nose_length=0.6,
        nose_type="blunt",
        nose_radius=0.3,
        tail_length=0.0,
        fin_count=0,
        source="OSINT representative",
    ),
    "swarm_bus": VehicleGeometry(
        name="Swarm post-boost bus",
        body_length=3.5,
        body_diameter=1.8,
        nose_length=1.0,
        nose_type="ogive",
        bus_length=2.5,
        bus_diameter=1.4,
        bus_nose="cone",
        bus_tail="cone",
        tail_length=0.5,
        fin_count=0,
        source="OSINT representative",
    ),
    "generic_rv": VehicleGeometry(
        name="Generic RV",
        body_length=1.8,
        body_diameter=0.5,
        nose_length=1.0,
        nose_type="blunt",
        nose_radius=0.25,
        tail_length=0.0,
        fin_count=0,
        source="OSINT representative",
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


# --- Profile functions ------------------------------------------------------ #
def _ogive_radius(x_rel: np.ndarray, nose_len: float, radius: float) -> np.ndarray:
    """Tangent ogive radius profile (0 at tip -> radius at base)."""
    rho = (radius**2 + nose_len**2) / (2.0 * radius)
    x = np.clip(x_rel, 0.0, nose_len)
    arg = np.clip(rho**2 - (nose_len - x) ** 2, 0.0, None)
    return np.sqrt(arg) + radius - rho


def nose_profile(g: VehicleGeometry, n_axial: int = 80) -> np.ndarray:
    """Return (n_nose, 2) [z, r] axial profile of the nose section."""
    z = np.linspace(0.0, g.nose_length, n_axial)
    r = np.zeros(n_axial)
    nl = max(g.nose_length, 1e-6)
    if g.nose_type == "cone":
        r = (z / nl) * (g.body_diameter / 2.0)
    elif g.nose_type == "ogive":
        r = _ogive_radius(z, nl, g.body_diameter / 2.0)
    elif g.nose_type == "blunt":
        R = max(g.nose_radius, 1e-6)
        r = np.sqrt(np.clip(R**2 - (R - np.clip(z, 0.0, R)) ** 2, 0.0, None))
        cap_h = R - np.sqrt(max(R**2 - (g.body_diameter / 2.0) ** 2, 0.0))
        if cap_h > 0 and g.nose_length > cap_h:
            z2 = z - (g.nose_length - cap_h)
            z2 = np.clip(z2, 0.0, R)
            r = np.sqrt(np.clip(R**2 - (R - z2) ** 2, 0.0, None))
    else:
        raise ValueError(f"Unsupported nose_type {g.nose_type!r}")
    return np.column_stack([z, r])


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
        R = max(g.nose_radius, 1e-6)
        zn = z[in_nose]
        r[in_nose] = np.sqrt(np.clip(R**2 - (R - zn) ** 2, 0.0, None))
        cap_h = R - np.sqrt(max(R**2 - (g.body_diameter / 2.0) ** 2, 0.0))
        if cap_h > 0 and g.nose_length > cap_h:
            z2 = zn - (g.nose_length - cap_h)
            z2 = np.clip(z2, 0.0, R)
            r[in_nose] = np.sqrt(np.clip(R**2 - (R - z2) ** 2, 0.0, None))
    else:
        raise ValueError(f"Unsupported nose_type {g.nose_type!r}")

    r[0] = max(r[0], 1e-9)

    mid = (z >= nl) & (z <= g.body_length - g.tail_length)
    r[mid] = g.body_diameter / 2.0

    if g.body_flair_deg > 0.0:
        flair_dist = 0.05 * g.body_length
        flair_mask = (z >= nl) & (z <= nl + flair_dist)
        z_rel = (z[flair_mask] - nl) / max(flair_dist, 1e-6)
        r[flair_mask] += (g.body_diameter / 2.0) * np.tan(np.radians(g.body_flair_deg)) * z_rel

    if g.tail_length > 0:
        tail = z > g.body_length - g.tail_length
        zt = (z[tail] - (g.body_length - g.tail_length)) / g.tail_length
        r[tail] = (g.body_diameter / 2.0) * (1.0 - g.boattail_fraction * zt)

    return np.column_stack([z, r])


def composite_profile(g: VehicleGeometry, n_axial: int = 80) -> Optional[np.ndarray]:
    """Return stacked [bus + payloads] axial profile for swarm/FOBS shapes."""
    if g.bus_length <= 0.0 or g.bus_diameter <= 0.0:
        return None
    z_body, r_body = body_profile(g, n_axial).T
    bus_start = z_body[-1]
    z_bus = np.linspace(bus_start, bus_start + g.bus_length, n_axial)
    r_bus = np.zeros(n_axial)
    bl = max(g.bus_length, 1e-6)
    bnl = min(g.nose_length * 0.5, bl * 0.3)
    bus_nose_idx = z_bus < bus_start + bnl
    if g.bus_nose == "cone":
        z_n = z_bus[bus_nose_idx] - bus_start
        r_bus[bus_nose_idx] = (z_n / max(bnl, 1e-6)) * (g.bus_diameter / 2.0)
    else:
        z_n = z_bus[bus_nose_idx] - bus_start
        r_bus[bus_nose_idx] = _ogive_radius(z_n, bnl, g.bus_diameter / 2.0)
    bus_mid = (z_bus >= bus_start + bnl) & (z_bus <= bus_start + g.bus_length - g.tail_length * 0.5)
    r_bus[bus_mid] = g.bus_diameter / 2.0

    tail_len = max(g.tail_length * 0.5, 1e-6)
    tail_mask = z_bus >= bus_start + g.bus_length - tail_len
    if np.any(tail_mask):
        zt = (z_bus[tail_mask] - (bus_start + g.bus_length - tail_len)) / tail_len
        r_bus[tail_mask] = (g.bus_diameter / 2.0) * (1.0 - 0.3 * zt)

    return np.column_stack([z_bus, r_bus])


# --- Mesh generation ------------------------------------------------------ #
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

    if g.control_surface and g.control_span > 0.0 and g.control_chord > 0.0:
        z_hinge = g.body_length - g.tail_length - g.fin_leading_edge - g.control_chord
        if 0.0 < z_hinge < g.body_length:
            r_hinge = float(np.interp(z_hinge, z, r))
            idx = np.searchsorted(z, z_hinge)
            if idx >= len(z) or abs(z[idx] - z_hinge) > 1e-6:
                z = np.insert(z, idx, z_hinge)
                r = np.insert(r, idx, r_hinge)

    theta = np.linspace(0.0, 2.0 * np.pi, n_circ, endpoint=False)
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

    comp = composite_profile(g, n_axial)
    if comp is not None:
        zc = comp[:, 0]
        rc = comp[:, 1]
        zzc, ttc = np.meshgrid(zc, theta, indexing="ij")
        rrc = np.broadcast_to(rc[:, None], zzc.shape)
        xc = rrc * np.cos(ttc)
        yc = rrc * np.sin(ttc)
        bus_v = np.stack([xc, yc, zzc], axis=-1).reshape(-1, 3)
        n_bus = bus_v.shape[0]
        for i in range(n_axial - 1):
            for j in range(n_circ):
                j2 = (j + 1) % n_circ
                a = n_body + i * n_circ + j
                b = n_body + i * n_circ + j2
                c = n_body + (i + 1) * n_circ + j
                d = n_body + (i + 1) * n_circ + j2
                faces.append([a, b, c])
                faces.append([b, d, c])
        vertices.append(bus_v)
        n_body += bus_v.shape[0]

    if g.fin_count > 0 and g.fin_span > 0:
        le = g.body_length - g.tail_length - g.fin_leading_edge - g.fin_root_chord
        for k in range(g.fin_count):
            phi = 2.0 * np.pi * k / g.fin_count
            cphi, sphi = np.cos(phi), np.sin(phi)
            for sgn in (1.0, -1.0):
                fin = _fin_panel(g, le, cphi, sphi, sgn)
                base = sum(len(v) for v in vertices)
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
    corners = np.array(
        [
            [R * cphi, R * sphi, z0],
            [R * cphi, R * sphi, z0 + rc],
            [(R + span) * cphi, (R + span) * sphi, z0 + (rc - tc) / 2.0],
            [(R + span) * cphi, (R + span) * sphi, z0 + (rc + tc) / 2.0],
        ]
    )
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
