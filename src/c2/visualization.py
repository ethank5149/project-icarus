"""Phase 6.4 — National "Golden Dome" visualization (PyVista + Panel).

Builds a 3-D ECEF coverage map of the defended architecture: interceptor
bases (colored by tier kind), defended targets, and an optional raid of threat
aim points. :func:`build_national_scene` produces a PyVista
``MultiBlock`` ready for ``plot()`` or offscreen export; ``dashboard.py`` wraps
it in a live Panel app (``panel serve src/c2/dashboard.py``). The scene reuses
``reference/locations.yml`` as the single source of truth for site coordinates.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import numpy as np


# Tier-kind colors (RGB 0..1) for consistent map rendering.
_TIER_COLORS: Dict[str, tuple] = {
    "boost": (0.85, 0.15, 0.15),   # red
    "upper": (0.95, 0.6, 0.1),     # orange
    "mid": (0.2, 0.6, 0.9),       # blue
    "lower": (0.2, 0.8, 0.4),      # green
}


def _earth_sphere(radius: float = 6.371e6, n_theta: int = 60, n_phi: int = 60):
    """A coarse ECEF unit-ish sphere mesh for geographic context."""
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - pyvista present in env
        raise RuntimeError("pyvista is required for the national map") from exc
    sphere = pv.Sphere(radius=radius, theta_resolution=n_theta,
                       phi_resolution=n_phi)
    return sphere


def build_national_scene(
    architecture: Any,
    defended_points: Optional[List[np.ndarray]] = None,
    threat_points: Optional[List[np.ndarray]] = None,
    earth_radius: float = 6.371e6,
) -> Any:
    """Assemble a PyVista scene of the layered architecture.

    Parameters
    ----------
    architecture : DefenseArchitecture
        The deployed layered defense (tiers -> batteries with ECEF locations).
    defended_points : list[ndarray], optional
        ECEF coordinates of defended targets (rendered as cyan stars).
    threat_points : list[ndarray], optional
        ECEF coordinates of inbound threat aim points (rendered as red markers).
    earth_radius : float
        Sphere radius (m); sites are plotted at their true ECEF magnitude so
        they sit on the globe surface.

    Returns
    -------
    A ``pyvista.MultiBlock`` scene ready for ``plot()`` / offscreen export.
    """
    import pyvista as pv

    blocks = pv.MultiBlock()
    blocks["earth"] = _earth_sphere(earth_radius)

    # --- interceptor bases, colored by tier kind -------------------------
    spokes = pv.MultiBlock()
    for ly in architecture.layers:
        for tier in ly.tiers:
            color = _TIER_COLORS.get(tier.kind, (0.9, 0.9, 0.9))
            pts = np.asarray(tier.bases, dtype=float)
            cloud = pv.PolyData(pts)
            cloud["tier"] = [tier.kind] * len(pts)
            cloud["magazine"] = [tier.magazine_per_base] * len(pts)
            blocks[f"{tier.kind}_bases"] = cloud
            # thin radial spokes from the globe surface to the site for visibility
            for p in pts:
                tip = p * (earth_radius + 3.0e5) / np.linalg.norm(p)
                spokes.append(pv.Line(p, tip))
    blocks["spokes"] = spokes

    # --- defended targets (cyan) --------------------------------------
    if defended_points:
        dp = np.asarray(defended_points, dtype=float)
        dblk = pv.PolyData(dp)
        blocks["defended"] = dblk

    # --- inbound threats (red) ----------------------------------------
    if threat_points:
        tp = np.asarray(threat_points, dtype=float)
        tblk = pv.PolyData(tp)
        blocks["threats"] = tblk

    return blocks


def coverage_summary_table(architecture: Any) -> List[Dict[str, Any]]:
    """Plain-data per-tier coverage summary (no rendering needed)."""
    rows: List[Dict[str, Any]] = []
    for ly in architecture.layers:
        for tier in ly.tiers:
            locs = np.asarray(tier.bases, dtype=float)
            # Mean great-circle reach proxy: spread of base locations (km).
            centroid = locs.mean(axis=0)
            spread_km = float(np.mean(
                [np.linalg.norm(p - centroid) for p in locs]
            )) / 1000.0
            rows.append({
                "kind": tier.kind,
                "interceptor": tier.interceptor_name,
                "n_bases": len(tier.bases),
                "magazine_per_base": tier.magazine_per_base,
                "total_magazine": tier.total_magazine,
                "mean_base_spread_km": spread_km,
                "c2_latency_s": tier.c2_latency_s,
            })
    return rows
