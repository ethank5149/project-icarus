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
    stl_path: str = ""  # optional CAD mesh path for visualization / ray-tracing
    stl_alias: str = ""  # manifest key in stl_loader.CAD_MANIFEST

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
        metadata={"range_km": ">18000", "warheads": "10-15x750 kt MIRVs", "mass_t": ">200", "cep_m": 400, "stage": "3-stage liquid", "status": "testing 2018-"},
        source="reference/threat-vehicle-deep-research-report.md",
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
        metadata={"speed_mach": "20+", "warhead_yield_kt": 750, "cep_m": "tens", "status": "active 2019", "booster": "UR-100N/Sarmat"},
        source="reference/threat-vehicle-deep-research-report.md",
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
        metadata={"range_km": "1500-2500", "speed_mach": 0.8, "propulsion": "solid boost + turbofan", "warhead_kg": 450, "warhead_yield_kt": "500 possible", "guidance": "INS+GLONASS/TERCOM", "cep_m": "2-3", "status": "active 2015-"},
        source="reference/threat-vehicle-deep-research-report.md",
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
        stl_alias="kh101",
        metadata={"range_km": "2500-3500", "speed_mach": 0.8, "mass_kg": 2300, "warhead_kg": 450, "guidance": "inertial+GLONASS/TERCOM", "cep_m": "6-10", "launch_platform": "Tu-95MS, Tu-160"},
        source="reference/threat-vehicle-deep-research-report.md",
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
        metadata={"range_km": "2500-3500", "speed_mach": 0.8, "mass_kg": 2300, "warhead_yield_kt": "~250", "guidance": "inertial+GLONASS/TERCOM", "cep_m": "6-10", "launch_platform": "Tu-95MS, Tu-160"},
        source="reference/threat-vehicle-deep-research-report.md",
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
        metadata={"range_km": 1000, "speed_mach": "8-9", "propulsion": "solid booster + scramjet", "warhead_kg": 300, "warhead_yield_kt": "1-3 possible", "status": "deployed 2022", "guidance": "INS+active radar"},
        source="reference/threat-vehicle-deep-research-report.md",
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
        metadata={"speed_mach": "<0.5", "propulsion": "nuclear reactor + turbofan", "warhead_yield_kt": "<5", "range_km": "unlimited (14000 test)", "status": "development suspended after 2019"},
        source="reference/threat-vehicle-deep-research-report.md",
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
        metadata={"range_km": "1700-2000", "speed_mach": 0.8, "warhead_kg": 500, "guidance": "inertial+BeiDou/GPS+TERCOM", "cep_m": "5-10", "status": "active mid-2000s"},
        source="reference/threat-vehicle-deep-research-report.md",
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
        metadata={"range_km": "400-460", "speed_mach": 3, "warhead_kg": 500, "guidance": "INS/GPS", "cep_m": "5-7", "status": "active 2015-", "launch_platform": "H-6K"},
        source="reference/threat-vehicle-deep-research-report.md",
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
        metadata={"range_km": 540, "speed_mach": "2.5-3 terminal", "warhead_kg": 300, "guidance": "INS + radar/IR seeker", "cep_m": 30, "status": "active 2015-"},
        source="reference/threat-vehicle-deep-research-report.md",
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
    # --- Russian ICBMs ---
    "ss18_satan": VehicleGeometry(
        name="R-36M2 SS-18 Satan",
        body_length=34.0,
        body_diameter=3.0,
        nose_length=7.5,
        nose_type="blunt",
        nose_radius=1.4,
        tail_length=2.5,
        fin_count=0,
        stl_alias="ss18_satan",
        metadata={"range_km": "11000-16000", "speed_mach": "~20 reentry", "warheads": "10x500-750 kt MIRVs", "cep_m": 0.5, "stage": "2-stage liquid", "status": "active (~46 mod6)", "service_since": 1988},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "topol_m": VehicleGeometry(
        name="RT-2PM2 Topol-M",
        body_length=22.0,
        body_diameter=1.8,
        nose_length=4.5,
        nose_type="cone",
        tail_length=1.5,
        fin_count=0,
        stl_alias="topol_m",
        metadata={"range_km": 11000, "speed_mach": "~20 reentry", "warhead_yield_kt": 550, "cep_m": 0.2, "stage": "3-stage solid", "launcher": "road-mobile or silo", "status": "active 1997-", "deployment": "~18 mobile, 60 silo (2016)"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "yars": VehicleGeometry(
        name="RS-24 Yars",
        body_length=21.0,
        body_diameter=1.8,
        nose_length=4.2,
        nose_type="cone",
        tail_length=1.4,
        fin_count=0,
        metadata={"range_km": 10500, "speed_mach": "~20 reentry", "warheads": "3x150-200 kt MIRVs", "payload_kg": 1200, "cep_m": "0.2-0.5", "stage": "3-stage solid", "launcher": "road-mobile and silo", "status": "active since 2010", "deployment": "~80 operational by 2013"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "rs26_rubezh": VehicleGeometry(
        name="RS-26 Rubezh",
        body_length=12.0,
        body_diameter=1.8,
        nose_length=2.8,
        nose_type="cone",
        tail_length=0.8,
        fin_count=4,
        fin_span=0.6,
        fin_root_chord=1.2,
        fin_tip_chord=0.5,
        fin_leading_edge=0.4,
        fin_thickness=0.04,
        metadata={"range_km": "2000-5800", "speed_mach": "~20 reentry", "warhead_kg": 800, "warheads": "single or MIRV (3-6)", "stage": "3-stage solid (shortened)", "launcher": "road-mobile", "status": "development inactive since mid-2010s", "tests": "first test 2011 (failed), successful 5800 km test May 2012"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    # --- Chinese ICBMs / MRBMs / SLBMs ---
    "df5": VehicleGeometry(
        name="DF-5 series",
        body_length=32.0,
        body_diameter=3.0,
        nose_length=7.0,
        nose_type="blunt",
        nose_radius=1.3,
        tail_length=2.2,
        fin_count=0,
        metadata={"range_km": "12000-13000", "speed_mach": "~20 reentry", "stage": "2-stage liquid", "warheads": "DF-5A single 3-5 Mt; DF-5B 3x150-200 kt MIRVs", "cep_m": 0.8, "launcher": "silo", "status": "active 1981-", "deployment": "~20 launchers (10 DF-5A, 10 DF-5B) as of 2016"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "df31": VehicleGeometry(
        name="DF-31 series",
        body_length=13.0,
        body_diameter=2.0,
        nose_length=2.5,
        nose_type="cone",
        tail_length=0.8,
        fin_count=4,
        fin_span=0.5,
        fin_root_chord=1.0,
        fin_tip_chord=0.4,
        fin_leading_edge=0.3,
        fin_thickness=0.03,
        metadata={"range_km": "7200-8000 (DF-31A ~13000)", "speed_mach": "~20 reentry", "stage": "3-stage solid", "warhead_yield_kt": 425, "guidance": "inertial", "cep_m": 100, "launcher": "road-mobile 8x8 TEL", "status": "active 2006-", "deployment": "~36 launchers (3 brigades) by 2019"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "df41": VehicleGeometry(
        name="DF-41",
        body_length=16.5,
        body_diameter=2.0,
        nose_length=3.2,
        nose_type="cone",
        tail_length=1.0,
        fin_count=4,
        fin_span=0.55,
        fin_root_chord=1.1,
        fin_tip_chord=0.45,
        fin_leading_edge=0.35,
        fin_thickness=0.035,
        stl_alias="df41",
        metadata={"range_km": "12000-15000", "speed_mach": "~20 reentry", "propulsion": "3-stage solid", "payload_kg": 2500, "warheads": "up to 10 MIRVs (20-150 kt)", "guidance": "inertial/GPS", "cep_m": "100-500", "launcher": "road/rail/silo", "status": "testing/limited, possibly ~20 launchers as of 2022"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "df17_dfzf": VehicleGeometry(
        name="DF-17 w/ DF-ZF HGV",
        body_length=11.0,
        body_diameter=1.4,
        nose_length=2.8,
        nose_type="blunt",
        nose_radius=0.6,
        tail_length=0.8,
        fin_count=4,
        fin_span=0.4,
        fin_root_chord=0.9,
        fin_tip_chord=0.35,
        fin_leading_edge=0.25,
        fin_thickness=0.025,
        metadata={"range_km": "1800-2500", "hgw_speed_mach": "5-10", "propulsion": "2-stage solid + HGV", "warhead": "single conventional or nuclear", "guidance": "INS+BeiDou", "status": "active 2019-"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    # --- SLBMs ---
    "jl2": VehicleGeometry(
        name="JL-2 SLBM",
        body_length=13.0,
        body_diameter=1.4,
        nose_length=2.6,
        nose_type="cone",
        tail_length=0.9,
        fin_count=0,
        metadata={"range_km": "8000-9000", "speed_mach": "~20 reentry", "warheads": "1x1 Mt or 3-8 MIRVs (20-150 kt)", "guidance": "astro-inertial + BeiDou", "cep_m": "150-300", "stage": "3-stage solid", "status": "active 2015-", "platform": "Type 094 Jin SSBN (~16 missiles)"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "jl3": VehicleGeometry(
        name="JL-3 SLBM",
        body_length=14.5,
        body_diameter=1.8,
        nose_length=3.0,
        nose_type="cone",
        tail_length=1.0,
        fin_count=0,
        metadata={"range_km": ">10000", "speed_mach": "~20 reentry", "warheads": "likely 3 MIRVs", "guidance": "astro-inertial + BeiDou", "stage": "3-stage solid", "status": "testing 2020s, may be operational ~2025", "platform": "Type 096 SSBN (in development)"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    # --- Hypersonics / special ---
    "df_zf_hgv": VehicleGeometry(
        name="DF-ZF Hypersonic Glide Vehicle",
        body_length=5.0,
        body_diameter=0.6,
        nose_length=1.5,
        nose_type="blunt",
        nose_radius=0.3,
        tail_length=0.6,
        fin_count=0,
        metadata={"speed_mach": "5-10", "range_km": "~2000 (DF-17 boost)", "warhead": "conventional or nuclear", "booster": "DF-17", "status": "active 2019-", "first_test": "2014-2018"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "kinzhal": VehicleGeometry(
        name="Kinzhal Kh-47M2",
        body_length=7.8,
        body_diameter=1.2,
        nose_length=1.8,
        nose_type="cone",
        tail_length=0.6,
        fin_count=4,
        fin_span=0.5,
        fin_root_chord=1.0,
        fin_tip_chord=0.4,
        fin_leading_edge=0.3,
        fin_thickness=0.04,
        metadata={"range_km": "1500-2000", "speed_mach": "4 climb, ~10 descent", "warhead_kg": 480, "warhead_yield_kt": "conventional or nuclear", "guidance": "inertial + possible IR terminal", "status": "active 2017-", "launch_platform": "MiG-31K, Tu-22M3", "launchers": "~100 MiG-31K believed"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "cj1000": VehicleGeometry(
        name="CJ-1000 scramjet cruise",
        body_length=8.5,
        body_diameter=0.8,
        nose_length=1.6,
        nose_type="ogive",
        tail_length=0.8,
        fin_count=4,
        fin_span=0.28,
        fin_root_chord=0.6,
        fin_tip_chord=0.24,
        fin_leading_edge=0.22,
        fin_thickness=0.022,
        metadata={"range_km": 6000, "speed_mach": 6, "propulsion": "scramjet", "warhead": "unknown (few hundred kg)", "guidance": "unspecified", "status": "unveiled 2025", "launcher": "road-mobile TEL"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "df100": VehicleGeometry(
        name="DF-100 CJ-100 supersonic cruise",
        body_length=7.5,
        body_diameter=0.7,
        nose_length=1.4,
        nose_type="ogive",
        tail_length=0.7,
        fin_count=4,
        fin_span=0.26,
        fin_root_chord=0.55,
        fin_tip_chord=0.22,
        fin_leading_edge=0.2,
        fin_thickness=0.02,
        metadata={"range_km": "3000-4000", "speed_mach": 5, "warhead_kg": 1000, "guidance": "inertial/BeiDou", "status": "active 2019-"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    # --- Additional Russian systems from threat report ---
    "ur100n_ss19": VehicleGeometry(
        name="UR-100N SS-19 Stiletto",
        body_length=20.0,
        body_diameter=2.0,
        nose_length=4.0,
        nose_type="cone",
        tail_length=1.2,
        fin_count=0,
        metadata={"range_km": 10000, "speed_mach": "~18 reentry", "warheads": "up to 6 MIRVs (500-750 kt)", "stage": "2-stage liquid", "launcher": "silo", "status": "active Mod3 since 1980", "note": "Avangard HGV deployed on UR-100N boosters"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "topol_ss25": VehicleGeometry(
        name="RT-2PM Topol SS-25",
        body_length=19.0,
        body_diameter=1.8,
        nose_length=3.8,
        nose_type="cone",
        tail_length=1.2,
        fin_count=0,
        metadata={"range_km": ">=11000", "speed_mach": "~20 reentry", "stage": "3-stage solid", "warhead_yield_kt": "550-800", "cep_m": 0.9, "launcher": "road-mobile TEL", "status": "active 1988-, now being retired", "deployment": "~90 launchers as of 2016"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "kh32": VehicleGeometry(
        name="Kh-32",
        body_length=5.8,
        body_diameter=0.9,
        nose_length=1.2,
        nose_type="cone",
        tail_length=0.5,
        fin_count=4,
        fin_span=0.4,
        fin_root_chord=0.8,
        fin_tip_chord=0.35,
        fin_leading_edge=0.25,
        fin_thickness=0.03,
        metadata={"range_km": 1000, "speed_mach": 4, "warhead": "possibly nuclear-capable", "status": "limited info", "launch_platform": "Tu-22M3M"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "oniks": VehicleGeometry(
        name="P-800 Oniks (3M-54E/3M55)",
        body_length=8.0,
        body_diameter=0.67,
        nose_length=1.4,
        nose_type="ogive",
        tail_length=0.6,
        fin_count=4,
        fin_span=0.3,
        fin_root_chord=0.7,
        fin_tip_chord=0.3,
        fin_leading_edge=0.25,
        fin_thickness=0.025,
        metadata={"range_km": 300, "speed_mach": "2-2.5", "propulsion": "solid boost + ramjet", "warhead_kg": 300, "status": "active 2000s-"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "starriy_sky2": VehicleGeometry(
        name="Starry Sky-2 (Xingkong-2)",
        body_length=4.0,
        body_diameter=0.5,
        nose_length=1.2,
        nose_type="ogive",
        tail_length=0.4,
        fin_count=0,
        metadata={"speed_mach": "~5.5-6", "altitude_km": 30, "flight_time_min": 10, "warhead": "none (research testbed)", "status": "experimental 2018-", "launch": "small rocket (likely B-611)"},
        source="reference/threat-vehicle-deep-research-report.md",
    ),
    "ch_as_x": VehicleGeometry(
        name="CH-AS-X-13 (JL-1) ASBM",
        body_length=6.0,
        body_diameter=1.4,
        nose_length=1.8,
        nose_type="blunt",
        nose_radius=0.5,
        tail_length=0.8,
        fin_count=4,
        fin_span=0.35,
        fin_root_chord=0.8,
        fin_tip_chord=0.35,
        fin_leading_edge=0.25,
        fin_thickness=0.025,
        metadata={"range_km": 1500, "warhead": "maneuvering reentry, nuclear-capable", "booster": "derivative of DF-21D", "status": "developmental, flight tests undisclosed", "launch_platform": "H-6N bomber"},
        source="reference/threat-vehicle-deep-research-report.md",
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
    if backend == "stl":
        from .stl_loader import load_cad_vehicle, read_stl, normalize_stl
        if g.stl_alias:
            return load_cad_vehicle(g.stl_alias)
        if g.stl_path:
            vertices, faces = read_stl(g.stl_path)
            return normalize_stl(vertices, faces, scale=1.0)
        raise ValueError(
            "VehicleGeometry has no stl_path or stl_alias for backend='stl'"
        )
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
