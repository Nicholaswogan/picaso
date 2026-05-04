# Comment below helps ignore linting false-positives.
# type: ignore

"""Experimental no-alloc thermal flux solvers.

This module mirrors the reflected-light no-alloc pattern in
``picaso.fluxes_noalloc`` but only implements the spectrum-mode thermal
source-function solver and only returns TOA fluxes in the first pass.
"""

import numba as nb
from numba.experimental import jitclass
from numba import types
from numba import typed
import numpy as np

from .fluxes_noalloc import setup_tri_diag_inplace, tri_diag_solve_inplace


@nb.njit(cache=True)
def _fill_blackbody_column(tlevel, wno, out):
    """Fill a wavelength-dependent blackbody source column in place.

    Parameters
    ----------
    tlevel : ndarray, shape (nlevel,)
        Temperatures at the atmospheric levels.
    wno : float
        Wavenumber in cm^-1 for the current wavelength.
    out : ndarray, shape (nlevel,)
        Output buffer, overwritten in place with the blackbody source in
        cgs units per unit wavelength.
    """
    h = 6.62607004e-27  # erg s
    c = 2.99792458e10  # cm/s
    k = 1.38064852e-16  # erg/K

    w = 1.0 / wno
    c1 = 2.0 * h * c * c
    c2 = h * c / k

    for i in range(tlevel.shape[0]):
        out[i] = (c1 / (w ** 5.0)) / (np.exp(c2 / (w * tlevel[i])) - 1.0)


@nb.njit(cache=True)
def _clamp_expterm(value):
    if value > 35.0:
        return 35.0
    return value


@nb.experimental.jitclass
class ThermalWorkspace:
    """Per-thread scratch space for the thermal 1D solver."""

    nlevel: nb.int64
    nlayer: nb.int64
    g1: nb.float64[:]
    g2: nb.float64[:]
    lamda: nb.float64[:]
    gama: nb.float64[:]
    c_plus_up: nb.float64[:]
    c_minus_up: nb.float64[:]
    c_plus_down: nb.float64[:]
    c_minus_down: nb.float64[:]
    exptrm: nb.float64[:]
    exptrm_positive: nb.float64[:]
    exptrm_minus: nb.float64[:]
    b1: nb.float64[:]
    A: nb.float64[:]
    B: nb.float64[:]
    C: nb.float64[:]
    D: nb.float64[:]
    positive: nb.float64[:]
    negative: nb.float64[:]
    flux_plus: nb.float64[:]
    bb: nb.float64[:]

    def __init__(self, nlevel):
        self._allocate(nlevel)

    def _allocate(self, nlevel):
        self.nlevel = nlevel
        self.nlayer = nlevel - 1
        self.g1 = np.empty(self.nlayer, dtype=np.float64)
        self.g2 = np.empty(self.nlayer, dtype=np.float64)
        self.lamda = np.empty(self.nlayer, dtype=np.float64)
        self.gama = np.empty(self.nlayer, dtype=np.float64)
        self.c_plus_up = np.empty(self.nlayer, dtype=np.float64)
        self.c_minus_up = np.empty(self.nlayer, dtype=np.float64)
        self.c_plus_down = np.empty(self.nlayer, dtype=np.float64)
        self.c_minus_down = np.empty(self.nlayer, dtype=np.float64)
        self.exptrm = np.empty(self.nlayer, dtype=np.float64)
        self.exptrm_positive = np.empty(self.nlayer, dtype=np.float64)
        self.exptrm_minus = np.empty(self.nlayer, dtype=np.float64)
        self.b1 = np.empty(self.nlayer, dtype=np.float64)
        self.A = np.empty(2 * self.nlayer, dtype=np.float64)
        self.B = np.empty(2 * self.nlayer, dtype=np.float64)
        self.C = np.empty(2 * self.nlayer, dtype=np.float64)
        self.D = np.empty(2 * self.nlayer, dtype=np.float64)
        self.positive = np.empty(self.nlayer, dtype=np.float64)
        self.negative = np.empty(self.nlayer, dtype=np.float64)
        self.flux_plus = np.empty(nlevel, dtype=np.float64)
        self.bb = np.empty(nlevel, dtype=np.float64)

    def _ensure(self, nlevel):
        if nlevel != self.nlevel:
            self._allocate(nlevel)

ThermalWorkspaceType = ThermalWorkspace.class_type.instance_type

@nb.experimental.jitclass
class ThermalResult:

    nwavelengths: nb.int64
    wavelength_um: nb.float64[:] # Wavelengths in microns
    thermal: nb.float64[:] # Disk-integrated TOA flux in CGS units

    def __init__(self):
        self._allocate(0)

    def _allocate(self, nwavelengths):
        self.nwavelengths = nwavelengths
        self.wavelength_um = np.empty(nwavelengths, dtype=np.float64)
        self.thermal = np.empty(nwavelengths, dtype=np.float64)

    def _ensure(self, nwavelengths):
        if nwavelengths != self.nwavelengths:
            self._allocate(nwavelengths)

@nb.experimental.jitclass
class ThermalSolver:
    """Persistent thermal solver state and TOA flux outputs."""

    nlevel: nb.int64
    nwno: nb.int64
    workspace: types.ListType(ThermalWorkspaceType)

    def __init__(self):
        self._allocate(0, 0)

    def _allocate(self, nlevel, nwno):
        self.nlevel = nlevel
        self.nwno = nwno
        nthreads = nb.get_num_threads()
        self.workspace = typed.List.empty_list(ThermalWorkspaceType)
        if nlevel <= 0:
            return
        for _ in range(nthreads):
            self.workspace.append(ThermalWorkspace(nlevel))

    def _ensure(self, nlevel, nwno):
        if nlevel != self.nlevel or nwno != self.nwno or len(self.workspace) != nb.get_num_threads():
            self._allocate(nlevel, nwno)

@nb.njit(parallel=True)
def get_thermal_1d(
    self,
    nlevel,
    nwno,
    numg,
    numt,
    gweight,
    tweight,
    wavelength_um,
    dtau,
    w0,
    cosb,
    tlevel,
    plevel,
    ubar1,
    surf_reflect,
    hard_surface,
    result,
):
    """Compute TOA thermal fluxes for a single atmosphere.

    This first pass only supports spectrum mode (``calc_type == 0``) and
    returns the top-of-atmosphere flux on the Gauss/Chebyshev grid.

    The opacity inputs are expected to be chunk-major, with shape
    ``(nwno, nlayer)`` for ``dtau``, ``w0``, and ``cosb``.
    """

    self._ensure(nlevel, nwno)
    result._ensure(nwno)

    for iw in nb.prange(nwno):
        result.wavelength_um[iw] = wavelength_um[iw]
        result.thermal[iw] = get_thermal_1d_w(
            self.workspace[nb.get_thread_id()],
            nlevel,
            numg,
            numt,
            gweight,
            tweight,
            wavelength_um[iw],
            dtau[iw, :],
            w0[iw, :],
            cosb[iw, :],
            tlevel,
            plevel,
            ubar1,
            surf_reflect[iw],
            hard_surface,
        )

    return result


@nb.njit
def get_thermal_1d_w(
    wrk,
    nlevel,
    numg,
    numt,
    gweight,
    tweight,
    wavelength_um,
    dtau,
    w0,
    cosb,
    tlevel,
    plevel,
    ubar1,
    surf_reflect,
    hard_surface,
):
    """Per-wavelength thermal solve used by :func:`get_thermal_1d`."""
    nlayer = nlevel - 1
    mu1 = 0.5
    twopi = 2.0 * np.pi
    wno = 1.0e4 / wavelength_um

    bb = wrk.bb
    b1 = wrk.b1
    g1 = wrk.g1
    g2 = wrk.g2
    lamda = wrk.lamda
    gama = wrk.gama
    c_plus_up = wrk.c_plus_up
    c_minus_up = wrk.c_minus_up
    c_plus_down = wrk.c_plus_down
    c_minus_down = wrk.c_minus_down
    exptrm = wrk.exptrm
    exptrm_positive = wrk.exptrm_positive
    exptrm_minus = wrk.exptrm_minus
    A = wrk.A
    B = wrk.B
    C = wrk.C
    D = wrk.D
    positive = wrk.positive
    negative = wrk.negative
    flux_plus = wrk.flux_plus

    _fill_blackbody_column(tlevel, wno, bb)

    for i in range(nlayer):
        b1[i] = (bb[i + 1] - bb[i]) / dtau[i]

    for i in range(nlayer):
        g1[i] = 2.0 - w0[i] * (1.0 + cosb[i])
        g2[i] = w0[i] * (1.0 - cosb[i])

    for i in range(nlayer):
        lamda_i = np.sqrt(g1[i] * g1[i] - g2[i] * g2[i])
        lamda[i] = lamda_i
        gama[i] = (g1[i] - lamda_i) / g2[i]

        exptrm_val = _clamp_expterm(lamda_i * dtau[i])
        exptrm[i] = exptrm_val
        exptrm_positive[i] = np.exp(exptrm_val)
        exptrm_minus[i] = 1.0 / exptrm_positive[i]

    for i in range(nlayer):
        inv_sum = 1.0 / (g1[i] + g2[i])
        c_plus_up[i] = twopi * mu1 * (bb[i] + b1[i] * inv_sum)
        c_minus_up[i] = twopi * mu1 * (bb[i] - b1[i] * inv_sum)
        c_plus_down[i] = twopi * mu1 * (bb[i] + b1[i] * dtau[i] + b1[i] * inv_sum)
        c_minus_down[i] = twopi * mu1 * (bb[i] + b1[i] * dtau[i] - b1[i] * inv_sum)

    tau_top = dtau[0] * plevel[0] / (plevel[1] - plevel[0])
    b_top = (1.0 - np.exp(-tau_top / mu1)) * bb[0] * np.pi
    if hard_surface:
        emissivity = 1.0 - surf_reflect
        b_surface = emissivity * bb[nlayer] * np.pi
    else:
        b_surface = (bb[nlayer] + b1[nlayer - 1] * mu1) * np.pi

    setup_tri_diag_inplace(
        A,
        B,
        C,
        D,
        nlayer,
        c_plus_up,
        c_minus_up,
        c_plus_down,
        c_minus_down,
        b_top,
        b_surface,
        surf_reflect,
        gama,
        exptrm_positive,
        exptrm_minus,
    )
    tri_diag_solve_inplace(2 * nlayer, A, B, C, D)

    for i in range(nlayer):
        positive[i] = D[2 * i] + D[2 * i + 1]
        negative[i] = D[2 * i] - D[2 * i + 1]

    thermal_sum = 0.0
    for nt in range(numt):
        for ng in range(numg):
            u1 = ubar1[ng, nt]

            if hard_surface:
                flux_plus[nlayer] = (1.0 - surf_reflect) * bb[nlayer] * twopi
            else:
                flux_plus[nlayer] = (bb[nlayer] + b1[nlayer - 1] * u1) * twopi

            for ibot in range(nlayer - 1, -1, -1):
                exptrm_angle = np.exp(-dtau[ibot] / u1)

                gcoef = (1.0 / mu1 - lamda[ibot]) * positive[ibot]
                hcoef = gama[ibot] * (lamda[ibot] + 1.0 / mu1) * negative[ibot]
                alpha1 = twopi * (bb[ibot] + b1[ibot] * (1.0 / (g1[ibot] + g2[ibot]) - mu1))
                alpha2 = twopi * b1[ibot]

                flux_plus[ibot] = (
                    flux_plus[ibot + 1] * exptrm_angle
                    + (gcoef / (lamda[ibot] * u1 - 1.0)) * (exptrm_positive[ibot] * exptrm_angle - 1.0)
                    + (hcoef / (lamda[ibot] * u1 + 1.0)) * (1.0 - exptrm_minus[ibot] * exptrm_angle)
                    + alpha1 * (1.0 - exptrm_angle)
                    + alpha2 * (u1 - (dtau[ibot] + u1) * exptrm_angle)
                )

            exptrm_angle_mdpt = np.exp(-0.5 * dtau[0] / u1)
            exptrm_positive_mdpt = np.exp(0.5 * exptrm[0])
            exptrm_minus_mdpt = 1.0 / exptrm_positive_mdpt
            gcoef = (1.0 / mu1 - lamda[0]) * positive[0]
            hcoef = gama[0] * (lamda[0] + 1.0 / mu1) * negative[0]
            alpha1 = twopi * (bb[0] + b1[0] * (1.0 / (g1[0] + g2[0]) - mu1))
            alpha2 = twopi * b1[0]

            flux_at_top = (
                flux_plus[1] * exptrm_angle_mdpt
                + (gcoef / (lamda[0] * u1 - 1.0)) * (exptrm_positive[0] * exptrm_angle_mdpt - exptrm_positive_mdpt)
                - (hcoef / (lamda[0] * u1 + 1.0)) * (exptrm_minus[0] * exptrm_angle_mdpt - exptrm_minus_mdpt)
                + alpha1 * (1.0 - exptrm_angle_mdpt)
                + alpha2 * (u1 + 0.5 * dtau[0] - (dtau[0] + u1) * exptrm_angle_mdpt)
            )
            thermal_sum += flux_at_top * gweight[ng] * tweight[nt]

    if numt == 1:
        sym_fac = 1.0
    else:
        sym_fac = 1.0 / (2.0 * np.pi)

    return thermal_sum * sym_fac
