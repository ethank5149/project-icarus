import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xarray")
pytest.importorskip("cfgrib")


GRIB = None
_ERA5_DIR = Path(__file__).resolve().parent.parent / "reference" / "ERA5"
if _ERA5_DIR.exists():
    gribs = list(_ERA5_DIR.glob("*.grib"))
    if gribs:
        GRIB = str(gribs[0])


def test_era5_interpolator_basic():
    if GRIB is None:
        pytest.skip("no ERA5 GRIB in reference/ERA5/")
    from project_icarus.reference.era5 import ERA5Interpolator
    interp = ERA5Interpolator(GRIB)
    u, v, w = interp.wind(39.9, 116.4, 0.0, "2015-01-15T12:00")
    assert np.isfinite(u) and np.isfinite(v)
    assert w == 0.0
    assert -80.0 <= float(u) <= 80.0
    assert -80.0 <= float(v) <= 80.0
    interp.close()


def test_era5_time_clamp():
    if GRIB is None:
        pytest.skip("no ERA5 GRIB in reference/ERA5/")
    from project_icarus.reference.era5 import ERA5Interpolator
    interp = ERA5Interpolator(GRIB)
    u1, v1, _ = interp.wind(39.9, 116.4, 0.0, "2014-12-31T00:00")
    assert np.isfinite(u1) and np.isfinite(v1)
    u2, v2, _ = interp.wind(39.9, 116.4, 0.0, "2016-01-01T00:00")
    assert np.isfinite(u2) and np.isfinite(v2)
    interp.close()


def test_era5_temperature_and_pressure():
    if GRIB is None:
        pytest.skip("no ERA5 GRIB in reference/ERA5/")
    from project_icarus.reference.era5 import ERA5Interpolator
    interp = ERA5Interpolator(GRIB)
    T = interp.temperature(39.9, 116.4, 0.0, "2015-01-15T12:00")
    p = interp.pressure(39.9, 116.4, 0.0, "2015-01-15T12:00")
    assert np.isfinite(T) and T > 100.0
    assert np.isfinite(p) and p > 50_000.0
    interp.close()


def test_era5_density():
    if GRIB is None:
        pytest.skip("no ERA5 GRIB in reference/ERA5/")
    from project_icarus.reference.era5 import ERA5Interpolator
    interp = ERA5Interpolator(GRIB)
    rho = interp.density(39.9, 116.4, 0.0, "2015-01-15T12:00")
    assert np.isfinite(rho) and rho > 0.0
    interp.close()


def test_atmosphere_era5_wind():
    if GRIB is None:
        pytest.skip("no ERA5 GRIB in reference/ERA5/")
    from project_icarus.dynamics.atmosphere import Atmosphere
    from project_icarus.reference.era5 import ERA5Interpolator
    atm = Atmosphere()
    atm.set_era5(GRIB)
    t_str = "2015-01-15T12:00"
    u_atm, v_atm, w_atm = atm.wind(39.9, 116.4, 0.0, t_str)
    era5 = ERA5Interpolator(GRIB)
    u_era5, v_era5, w_era5 = era5.wind(39.9, 116.4, 0.0, t_str)
    assert np.isclose(u_atm, u_era5, atol=0.1)
    assert np.isclose(v_atm, v_era5, atol=0.1)
    era5.close()


def test_target_wind_drift():
    from project_icarus.scenarios.target_factory import BallisticScenario

    class _ShearWind:
        def wind(self, lat, lon, alt, time):
            return 20.0 * (1.0 + lat / 90.0), -5.0 * (lat / 90.0), 0.0

    wind = _ShearWind()
    r0 = np.array([6_371_000.0 + 100_000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7000.0, 0.0])
    tgt_no_wind = BallisticScenario(r0=r0, v0=v0)
    tgt_with_wind = BallisticScenario(r0=r0.copy(), v0=v0.copy(), wind_model=wind)
    state_no = tgt_no_wind.propagate(60.0)
    state_wind = tgt_with_wind.propagate(60.0)
    drift = np.linalg.norm(state_no[:3] - state_wind[:3])
    assert drift > 10.0


def test_cep_calculation():
    from project_icarus.c2.campaign import _compute_cep, CampaignThreat

    class _Eng:
        def __init__(self, misses):
            self.monte_carlo = _MC(misses)

    class _MC:
        def __init__(self, m):
            self.miss_distances = m

    threats = [
        CampaignThreat(target=None, aim_point=np.array([6.371e6, 0.0, 0.0])),
        CampaignThreat(target=None, aim_point=np.array([6.371e6, 0.0, 0.0])),
    ]
    engs = [
        _Eng([100.0, 200.0, 150.0]),
        _Eng([50.0, 80.0]),
    ]
    cep = _compute_cep(engs, threats)
    assert cep is not None and cep > 0.0
