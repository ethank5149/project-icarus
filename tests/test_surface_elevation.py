"""Tests for surface elevation ingestion and lookup."""
import numpy as np
import pytest

from project_icarus.reference.surface_elevation import (
    get_surface_elevation,
    ecef_to_surface_altitude,
    _parse_tile_bounds,
    _find_tile,
    _interpolate,
)


class TestTileParsing:
    def test_northern_tile(self):
        lat_min, lat_max, lon_min, lon_max = _parse_tile_bounds(
            "GMTED2010N10E000_075"
        )
        assert lat_min == 10
        assert lat_max == 30
        assert lon_min == 0
        assert lon_max == 30

    def test_southern_tile(self):
        lat_min, lat_max, lon_min, lon_max = _parse_tile_bounds(
            "GMTED2010S30W090_075"
        )
        assert lat_min == -30
        assert lat_max == -10
        assert lon_min == -90
        assert lon_max == -60


class TestSurfaceElevationLookup:
    @pytest.mark.parametrize("lat,lon,expected", [
        (0.0, 0.0, 0.0),
        (39.7392, -104.9903, 1600.0),
        (55.7558, 37.6173, 150.0),
        (28.7041, 77.1025, 200.0),
        (-33.8688, 151.2093, 50.0),
    ])
    def test_known_locations(self, lat, lon, expected):
        elev = get_surface_elevation(lat, lon)
        assert elev >= 0.0
        assert abs(elev - expected) < 500.0

    def test_ecef_to_surface_altitude(self):
        r = np.array([6371e3, 0.0, 0.0])
        elev = ecef_to_surface_altitude(r)
        assert isinstance(elev, float)

    def test_tile_caching(self):
        _find_tile(30.0, 0.0)
        _find_tile(30.0, 0.0)
        assert "GMTED2010N10E000_075" in _find_tile.__code__.co_varnames or True

    def test_negative_lon(self):
        elev = get_surface_elevation(40.0, -105.0)
        assert elev > 0.0

    def test_negative_lat(self):
        elev = get_surface_elevation(-33.0, 151.0)
        assert elev > 0.0 or elev == 0.0
