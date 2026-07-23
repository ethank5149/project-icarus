#!/usr/bin/env python3
import sys
sys.path.insert(0, '/mnt/user/public/project-icarus')

import numpy as np
from project_icarus.scenarios.target_factory import SarmatScenario, _geodetic_to_ecef_simple, _ecef_to_geodetic, _enu_basis, R_EARTH
from project_icarus.reference.surface_elevation import get_surface_elevation

print("=== RS-28 SARMAT UNINTERCEPTED TRAJECTORY VERIFICATION ===")

# Launch and target sites
lat0, lon0 = 54.07, 35.73  # Kozelsk
lat1, lon1 = 38.90, -77.04  # DC

# Use actual terrain elevation where available
elev0 = get_surface_elevation(lat0, lon0)
elev1 = get_surface_elevation(lat1, lon1)
print(f"Kozelsk elevation:  {elev0:.1f} m")
print(f"DC elevation:       {elev1:.1f} m")

# ECEF positions
r0 = _geodetic_to_ecef_simple(lat0, lon0, elev0)
r_target = _geodetic_to_ecef_simple(lat1, lon1, elev1)

print(f"Range: {np.linalg.norm(r0 - r_target)/1000:.0f} km")

# Get local ENU basis at launch
east, north, up = _enu_basis(lat0, lon0)
print(f"Launch up vector: {up}")

# Great-circle azimuth to target
dLon = np.radians(lon1 - lon0)
latR1 = np.radians(lat0)
latR2 = np.radians(lat1)
az = np.degrees(np.arctan2(
    np.sin(dLon) * np.cos(latR2),
    np.cos(latR1) * np.sin(latR2) - np.sin(latR1) * np.cos(latR2) * np.cos(dLon)
)) % 360.0
print(f"Great-circle azimuth: {az:.1f}°")

# Initial thrust direction = straight up (silo launch)
thrust_dir = up
print(f"Initial thrust dir: {thrust_dir}")

# Create scenario — NO estimated v0, v0 = zeros
scenario = SarmatScenario(
    r0=r0,
    v0=np.zeros(3),
    use_j2=True,
    initial_thrust_dir=thrust_dir,
)
print(f"\nScenario created")
print(f"  Boost stages: {len(scenario._SARMAT_STAGES)}")
print(f"  Initial mass: {scenario._initial_mass():.0f} kg")
print(f"  Drag coeff: {scenario.cd}, Area: {scenario.area} m^2")

# Propagate full trajectory
wall_start = np.datetime64('now')
t_query = 3000.0  # max TOF

import time
t0 = time.time()
state = scenario.propagate(t_query)
wall = time.time() - t0

r_impact = state[:3]
v_impact = state[3:6]
miss = np.linalg.norm(r_impact - r_target)

geodetic_impact = _ecef_to_geodetic(r_impact)
geodetic_target = _ecef_to_geodetic(r_target)

print(f"\n=== IMPACT ===")
print(f"Time of flight:  propagated to t={t_query}s, wall={wall:.2f}s")
print(f"Impact position (ECEF): {r_impact}")
print(f"Impact geodetic: lat={geodetic_impact[0]:.6f}, lon={geodetic_impact[1]:.6f}, alt={geodetic_impact[2]:.1f} m")
print(f"Target geodetic:  lat={geodetic_target[0]:.6f}, lon={geodetic_target[1]:.6f}, alt={geodetic_target[2]:.1f} m")
print(f"Impact speed:     {np.linalg.norm(v_impact):.1f} m/s")
print(f"Miss distance:    {miss:.1f} m")
print(f"Within 100m?:     {'YES' if miss < 100.0 else 'NO — REQUIRES GUIDANCE OR v0 OPTIMIZATION'}")

if miss >= 100.0:
    print(f"\n=== DIAGNOSTICS ===")
    # Check trajectory at burnout and apogee
    s_temp = SarmatScenario(r0=r0, v0=np.zeros(3), use_j2=True, initial_thrust_dir=thrust_dir)
    
    for check_t in [100.0, 250.0, 310.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0]:
        st = s_temp.propagate(check_t)
        pos = st[:3]
        vel = st[3:6]
        alt = _ecef_to_geodetic(pos)[2]
        spd = np.linalg.norm(vel)
        dist_to_target = np.linalg.norm(pos - r_target)
        print(f"  t={check_t:6.1f}s: alt={alt:10.1f} m, speed={spd:7.1f} m/s, dist_to_target={dist_to_target/1000:7.1f} km")
