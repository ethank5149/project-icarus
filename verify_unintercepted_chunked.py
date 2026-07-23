#!/usr/bin/env python3
import sys
sys.path.insert(0, '/mnt/user/public/project-icarus')

import numpy as np
from project_icarus.scenarios.target_factory import SarmatScenario, _geodetic_to_ecef_simple, _ecef_to_geodetic, _enu_basis, R_EARTH
from project_icarus.reference.surface_elevation import get_surface_elevation
import time

print("=== RS-28 SARMAT UNINTERCEPTED TRAJECTORY VERIFICATION ===")

# Launch and target sites
lat0, lon0 = 54.07, 35.73  # Kozelsk
lat1, lon1 = 38.90, -77.04  # DC

elev0 = get_surface_elevation(lat0, lon0)
elev1 = get_surface_elevation(lat1, lon1)
print(f"Kozelsk elevation:  {elev0:.1f} m")
print(f"DC elevation:       {elev1:.1f} m")

r0 = _geodetic_to_ecef_simple(lat0, lon0, elev0)
r_target = _geodetic_to_ecef_simple(lat1, lon1, elev1)

print(f"Range: {np.linalg.norm(r0 - r_target)/1000:.0f} km")

# Initial thrust = straight up
_, _, up = _enu_basis(lat0, lon0)
thrust_dir = up
print(f"Initial thrust dir: {thrust_dir}")

# Create scenario — NO estimated v0, v0 = zeros
scenario = SarmatScenario(
    r0=r0,
    v0=np.zeros(3),
    use_j2=True,
    initial_thrust_dir=thrust_dir,
)
print(f"Scenario created")

# Propagate in chunks to monitor progress
check_times = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 2400.0, 3000.0]
all_states = {}

print("\n=== PROPAGATION ===")
for t_query in check_times:
    print(f"  Propagating to t={t_query:.0f}s...", end=" ", flush=True)
    t0 = time.time()
    state = scenario.propagate(t_query)
    wall = time.time() - t0
    all_states[t_query] = state
    
    r = state[:3]
    v = state[3:6]
    alt = _ecef_to_geodetic(r)[2]
    spd = np.linalg.norm(v)
    dist = np.linalg.norm(r - r_target)
    print(f"wall={wall:.2f}s, alt={alt:.0f}m, speed={spd:.0f}m/s, dist_to_target={dist/1000:.0f}km")

# Final assessment
final_state = all_states[3000.0]
r_impact = final_state[:3]
miss = np.linalg.norm(r_impact - r_target)
geodetic_impact = _ecef_to_geodetic(r_impact)

print(f"\n=== IMPACT (t=3000s) ===")
print(f"Impact position: lat={geodetic_impact[0]:.6f}, lon={geodetic_impact[1]:.6f}, alt={geodetic_impact[2]:.1f}m")
print(f"Target position: lat=38.900000, lon=-77.040000, alt=31.0m")
print(f"Miss distance:   {miss:.1f} m")
print(f"Within 100m?:    {'YES' if miss < 100.0 else 'NO'}")
