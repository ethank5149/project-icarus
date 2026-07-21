# distutils: language = c
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt, exp, pow, log, fabs, M_PI


# Pre-computed layer arrays (filled at import time).
cdef np.ndarray _USSA76_BOTTOMS_arr
cdef np.ndarray _USSA76_TOPS_arr
cdef np.ndarray _USSA76_LAPSES_arr
cdef np.ndarray _USSA76_T_BOTTOMS_arr
cdef np.ndarray _USSA76_P_BOTTOMS_arr
cdef bint _initialized = 0


def _init_arrays():
    global _initialized, _USSA76_BOTTOMS_arr, _USSA76_TOPS_arr
    global _USSA76_LAPSES_arr, _USSA76_T_BOTTOMS_arr, _USSA76_P_BOTTOMS_arr
    if _initialized:
        return
    from project_icarus.dynamics.atmosphere import (
        _USSA76_BOTTOMS as b,
        _USSA76_TOPS as t,
        _USSA76_LAPSES as l,
        _USSA76_T_BOTTOMS as tb,
        _USSA76_P_BOTTOMS as pb,
    )
    _USSA76_BOTTOMS_arr = np.array(b, dtype=np.float64)
    _USSA76_TOPS_arr = np.array(t, dtype=np.float64)
    _USSA76_LAPSES_arr = np.array(l, dtype=np.float64)
    _USSA76_T_BOTTOMS_arr = np.array(tb, dtype=np.float64)
    _USSA76_P_BOTTOMS_arr = np.array(pb, dtype=np.float64)
    _initialized = 1


@cython.boundscheck(False)
@cython.wraparound(False)
def density(np.ndarray[np.float64_t, ndim=1] h):
    """Cython-accelerated atmospheric density calculation."""
    _init_arrays()
    cdef int n = h.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int i, idx
    cdef double h_val, h_bottom, lapse, T_bottom, P_bottom
    cdef double T, ratio, P
    
    for i in range(n):
        h_val = h[i]
        idx = _layer_index(h_val)
        h_bottom = _USSA76_BOTTOMS_arr[idx]
        lapse = _USSA76_LAPSES_arr[idx]
        T_bottom = _USSA76_T_BOTTOMS_arr[idx]
        P_bottom = _USSA76_P_BOTTOMS_arr[idx]
        T = T_bottom + lapse * (h_val - h_bottom)
        if T < 0.0:
            T = 0.0
        if fabs(T_bottom) > 1e-12:
            ratio = T / T_bottom
        else:
            ratio = 1.0
        if fabs(lapse) < 1e-12:
            P = P_bottom * exp(-9.80665 * (h_val - h_bottom) / (287.05 * T_bottom))
        else:
            if ratio > 0.0:
                P = P_bottom * pow(ratio, -9.80665 / (287.05 * lapse))
            else:
                P = 0.0
        result[i] = P / (287.05 * T)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def temperature(np.ndarray[np.float64_t, ndim=1] h):
    """Cython-accelerated atmospheric temperature calculation."""
    _init_arrays()
    cdef int n = h.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int i, idx
    cdef double h_val, h_bottom, lapse, T_bottom
    
    for i in range(n):
        h_val = h[i]
        idx = _layer_index(h_val)
        h_bottom = _USSA76_BOTTOMS_arr[idx]
        lapse = _USSA76_LAPSES_arr[idx]
        T_bottom = _USSA76_T_BOTTOMS_arr[idx]
        result[i] = T_bottom + lapse * (h_val - h_bottom)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def pressure(np.ndarray[np.float64_t, ndim=1] h):
    """Cython-accelerated atmospheric pressure calculation."""
    _init_arrays()
    cdef int n = h.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef int i, idx
    cdef double h_val, h_bottom, lapse, T_bottom, P_bottom
    cdef double T, ratio, P
    
    for i in range(n):
        h_val = h[i]
        idx = _layer_index(h_val)
        h_bottom = _USSA76_BOTTOMS_arr[idx]
        lapse = _USSA76_LAPSES_arr[idx]
        T_bottom = _USSA76_T_BOTTOMS_arr[idx]
        P_bottom = _USSA76_P_BOTTOMS_arr[idx]
        T = T_bottom + lapse * (h_val - h_bottom)
        if T < 0.0:
            T = 0.0
        if fabs(T_bottom) > 1e-12:
            ratio = T / T_bottom
        else:
            ratio = 1.0
        if fabs(lapse) < 1e-12:
            P = P_bottom * exp(-9.80665 * (h_val - h_bottom) / (287.05 * T_bottom))
        else:
            if ratio > 0.0:
                P = P_bottom * pow(ratio, -9.80665 / (287.05 * lapse))
            else:
                P = 0.0
        result[i] = P
    return result


cdef int _layer_index(double h):
    cdef int idx = 0
    if h >= _USSA76_BOTTOMS_arr[0]:
        idx = _binary_search(h)
    return idx


cdef int _binary_search(double h):
    cdef int left = 0
    cdef int right = 8  # 8 layers
    cdef int mid
    while left < right:
        mid = (left + right) // 2
        if h < _USSA76_BOTTOMS_arr[mid]:
            right = mid
        else:
            left = mid + 1
    return left - 1
