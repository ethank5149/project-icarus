"""
Cython-accelerated atmosphere model.

Provides drop-in replacements for the vectorized NumPy atmosphere functions
with C-level performance. The Python API remains identical.
"""

import numpy as np

# Pre-computed layer arrays (filled at import time from the Python module).
_USSA76_BOTTOMS = None
_USSA76_TOPS = None
_USSA76_LAPSES = None
_USSA76_T_BOTTOMS = None
_USSA76_P_BOTTOMS = None


def _init_layer_arrays():
    global _USSA76_BOTTOMS, _USSA76_TOPS, _USSA76_LAPSES, _USSA76_T_BOTTOMS, _USSA76_P_BOTTOMS
    if _USSA76_BOTTOMS is not None:
        return
    from project_icarus.dynamics.atmosphere import (
        _USSA76_BOTTOMS as b,
        _USSA76_TOPS as t,
        _USSA76_LAPSES as l,
        _USSA76_T_BOTTOMS as tb,
        _USSA76_P_BOTTOMS as pb,
    )
    _USSA76_BOTTOMS = b
    _USSA76_TOPS = t
    _USSA76_LAPSES = l
    _USSA76_T_BOTTOMS = tb
    _USSA76_P_BOTTOMS = pb


def density(h):
    """Vectorized density calculation using pre-built NumPy arrays."""
    _init_layer_arrays()
    h = np.asarray(h, dtype=np.float64)
    idx = np.searchsorted(_USSA76_BOTTOMS, h, side='right') - 1
    idx = np.clip(idx, 0, 7)
    h_bottom = _USSA76_BOTTOMS[idx]
    lapse = _USSA76_LAPSES[idx]
    T_bottom = _USSA76_T_BOTTOMS[idx]
    P_bottom = _USSA76_P_BOTTOMS[idx]
    T = T_bottom + lapse * (h - h_bottom)
    T = np.maximum(T, 0.0)
    ratio = np.where(np.abs(T_bottom) > 1e-12, T / T_bottom, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(
            np.abs(lapse) < 1e-12,
            P_bottom * np.exp(-9.80665 * (h - h_bottom) / (287.05 * T_bottom)),
            P_bottom * np.where(ratio > 0, ratio ** (-9.80665 / (287.05 * lapse)), 0.0),
        )
    return P / (287.05 * T)


def temperature(h):
    _init_layer_arrays()
    h = np.asarray(h, dtype=np.float64)
    idx = np.searchsorted(_USSA76_BOTTOMS, h, side='right') - 1
    idx = np.clip(idx, 0, 7)
    T_bottom = _USSA76_T_BOTTOMS[idx]
    lapse = _USSA76_LAPSES[idx]
    h_bottom = _USSA76_BOTTOMS[idx]
    return T_bottom + lapse * (h - h_bottom)


def pressure(h):
    _init_layer_arrays()
    h = np.asarray(h, dtype=np.float64)
    idx = np.searchsorted(_USSA76_BOTTOMS, h, side='right') - 1
    idx = np.clip(idx, 0, 7)
    h_bottom = _USSA76_BOTTOMS[idx]
    lapse = _USSA76_LAPSES[idx]
    T_bottom = _USSA76_T_BOTTOMS[idx]
    P_bottom = _USSA76_P_BOTTOMS[idx]
    T = T_bottom + lapse * (h - h_bottom)
    T = np.maximum(T, 0.0)
    ratio = np.where(np.abs(T_bottom) > 1e-12, T / T_bottom, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(
            np.abs(lapse) < 1e-12,
            P_bottom * np.exp(-9.80665 * (h - h_bottom) / (287.05 * T_bottom)),
            P_bottom * np.where(ratio > 0, ratio ** (-9.80665 / (287.05 * lapse)), 0.0),
        )
    return P


try:
    from .atmosphere_cython import density as _cy_density, temperature as _cy_temperature, pressure as _cy_pressure
    _HAVE_CYTHON = True
except ImportError:
    _HAVE_CYTHON = False


def fast_density(h):
    if _HAVE_CYTHON:
        return _cy_density(h)
    return density(h)


def fast_temperature(h):
    if _HAVE_CYTHON:
        return _cy_temperature(h)
    return temperature(h)


def fast_pressure(h):
    if _HAVE_CYTHON:
        return _cy_pressure(h)
    return pressure(h)
