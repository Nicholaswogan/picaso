import numba as nb
import numpy as np
import os

_RAMAN_FILE = os.path.join(os.environ.get('picaso_refdata'), 'opacities', 'raman_fortran.txt')
_RAMAN_TABLE = np.ascontiguousarray(np.loadtxt(_RAMAN_FILE, dtype=np.float64))
_RAMAN_WAVELENGTH = np.ascontiguousarray(_RAMAN_TABLE[:, 0])
_RAMAN_FACTOR = np.ascontiguousarray(_RAMAN_TABLE[:, 1])
_RAMAN_BASELINE = 0.99999


@nb.njit(cache=True)
def _interp_raman_scalar(wavelength, wavelength_grid, factor_grid):
    if wavelength <= wavelength_grid[0]:
        return factor_grid[0]
    if wavelength >= wavelength_grid[wavelength_grid.shape[0] - 1]:
        return factor_grid[factor_grid.shape[0] - 1]

    left = 0
    right = wavelength_grid.shape[0] - 1
    while right - left > 1:
        mid = (left + right) // 2
        if wavelength_grid[mid] <= wavelength:
            left = mid
        else:
            right = mid

    x0 = wavelength_grid[left]
    x1 = wavelength_grid[right]
    y0 = factor_grid[left]
    y1 = factor_grid[right]
    frac = (wavelength - x0) / (x1 - x0)
    return y0 + frac * (y1 - y0)


@nb.njit
def _fill_raman_mode_1(wavelength, raman_factor):
    for i in range(wavelength.shape[0]):
        raman_factor[i] = _interp_raman_scalar(wavelength[i], _RAMAN_WAVELENGTH, _RAMAN_FACTOR)


@nb.njit
def _fill_raman_mode_2(raman_factor):
    for i in range(raman_factor.shape[0]):
        raman_factor[i] = _RAMAN_BASELINE


@nb.njit
def compute_raman(raman_mode, wavelength, raman_factor):
    """
    Fill ``raman_factor`` for the supported Raman modes.

    Parameters
    ----------
    raman_mode : int
        Supported values:
        - 1: interpolate the legacy lookup table from ``raman_fortran.txt``
        - 2: use the no-Raman baseline factor ``0.99999``
    wavelength : 1D numpy.ndarray
        Wavelength grid in micron.
    raman_factor : 1D numpy.ndarray
        Output buffer, same shape as ``wavelength``.
    """
    if raman_mode == 1:
        _fill_raman_mode_1(wavelength, raman_factor)
        return
    if raman_mode == 2:
        _fill_raman_mode_2(raman_factor)
        return
    raise ValueError("raman_mode must be 1 or 2")
