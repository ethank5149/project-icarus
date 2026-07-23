import sys
sys.path.insert(0, '/mnt/user/public/project-icarus')

import numpy as np
from project_icarus.scenarios.target_factory import SarmatScenario, _geodetic_to_ecef_simple, _ecef_to_geodetic
from project_icarus.reference.surface_elevation import get_surface_elevation

lat, lon = 54.07, 35.73
elev = get_surface_elevation(lat, lon)
r0 = _geodetic_to_ecef_simple(lat, lon, elev)

scenario = SarmatScenario(r0=r0, v0=np.zeros(3), use_j2=True)

print("Aim point:", scenario._aim_point)
print("Aim point (geodetic):", _ecef_to_geodetic(scenario._aim_point))

# Test thrust direction at t=0
d0 = scenario._current_thrust_dir(0.0, r0, np.zeros(3))
print("Thrust dir at t=0:", d0)
print("Magnitude:", np.linalg.norm(d0))

# Test at t=5
r5 = r0 + d0 * 100  # rough guess
d5 = scenario._current_thrust_dir(5.0, r5, d0 * 100)
print("Thrust dir at t=5:", d5)
