# Comment below helps ignore linting false-positives.
# type: ignore

"""Experimental no-alloc thermal and reflected flux solvers.

This module mirrors the reflected-light no-alloc pattern in
``picaso.fluxes_noalloc`` but only implements the spectrum-mode thermal
source-function solver and a TOA-only reflected-light solver in the first
pass.
"""

import numba as nb
from numba import types
from numba import typed
import numpy as np

from ..fluxes_noalloc import setup_tri_diag_inplace, tri_diag_solve_inplace


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
    fpfs: nb.float64[:] 

    def __init__(self):
        self._allocate(0)

    def _allocate(self, nwavelengths):
        self.nwavelengths = nwavelengths
        self.wavelength_um = np.empty(nwavelengths, dtype=np.float64)
        self.thermal = np.empty(nwavelengths, dtype=np.float64)
        self.fpfs = np.empty(nwavelengths, dtype=np.float64)

    def _ensure(self, nwavelengths):
        if nwavelengths != self.nwavelengths:
            self._allocate(nwavelengths)

@nb.experimental.jitclass
class ThermalSolver:
    """Persistent thermal solver state and TOA flux outputs."""

    nlevel: nb.int64
    workspace: types.ListType(ThermalWorkspaceType)

    def __init__(self):
        self._allocate(0)

    def _allocate(self, nlevel):
        self.nlevel = nlevel
        nthreads = nb.get_num_threads()
        self.workspace = typed.List.empty_list(ThermalWorkspaceType)
        if nlevel <= 0:
            return
        for _ in range(nthreads):
            self.workspace.append(ThermalWorkspace(nlevel))

    def _ensure(self, nlevel):
        if nlevel != self.nlevel or len(self.workspace) != nb.get_num_threads():
            self._allocate(nlevel)

@nb.njit(parallel=True)
def get_thermal_1d(
    self,
    nlevel,
    nwavelengths_in_chunk,
    ngauss_ck,
    numg,
    numt,
    ck_weights,
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
    flux,
):
    """Compute TOA thermal fluxes for a single atmosphere.

    This first pass only supports spectrum mode (``calc_type == 0``) and
    returns the top-of-atmosphere flux on the Gauss/Chebyshev grid.

    The opacity inputs are expected to be chunk-major, with shape
    ``(nwavelengths_in_chunk, ngauss_ck, nlayer)`` for ``dtau``, ``w0``, and
    ``cosb``.
    """

    self._ensure(nlevel)

    for iw in nb.prange(nwavelengths_in_chunk):
        flux_iw = 0.0
        for igauss in range(ngauss_ck):
            flux_iw += ck_weights[igauss] * get_thermal_1d_w(
                self.workspace[nb.get_thread_id()],
                nlevel,
                numg,
                numt,
                gweight,
                tweight,
                wavelength_um[iw],
                dtau[iw, igauss, :],
                w0[iw, igauss, :],
                cosb[iw, igauss, :],
                tlevel,
                plevel,
                ubar1,
                surf_reflect[iw],
                hard_surface,
            )
        flux[iw] = flux_iw

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
    gcoef = wrk.c_plus_up
    hcoef = wrk.c_minus_up
    alpha1 = wrk.c_plus_down
    alpha2 = wrk.c_minus_down

    _fill_blackbody_column(tlevel, wno, bb)

    for i in range(nlayer):
        b1[i] = (bb[i + 1] - bb[i]) / dtau[i]

    for i in range(nlayer):
        g1[i] = 2.0 - w0[i] * (1.0 + cosb[i])
        g2[i] = w0[i] * (1.0 - cosb[i])

    for i in range(nlayer):
        if g2[i] == 0.0:
            lamda_i = g1[i]
            gama[i] = 0.0
        else:
            lamda_i = np.sqrt(g1[i] * g1[i] - g2[i] * g2[i])
            gama[i] = (g1[i] - lamda_i) / g2[i]
        lamda[i] = lamda_i

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

    for i in range(nlayer):
        gcoef[i] = (1.0 / mu1 - lamda[i]) * positive[i]
        hcoef[i] = gama[i] * (lamda[i] + 1.0 / mu1) * negative[i]
        alpha1[i] = twopi * (bb[i] + b1[i] * (1.0 / (g1[i] + g2[i]) - mu1))
        alpha2[i] = twopi * b1[i]

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

                flux_plus[ibot] = (
                    flux_plus[ibot + 1] * exptrm_angle
                    + (gcoef[ibot] / (lamda[ibot] * u1 - 1.0)) * (exptrm_positive[ibot] * exptrm_angle - 1.0)
                    + (hcoef[ibot] / (lamda[ibot] * u1 + 1.0)) * (1.0 - exptrm_minus[ibot] * exptrm_angle)
                    + alpha1[ibot] * (1.0 - exptrm_angle)
                    + alpha2[ibot] * (u1 - (dtau[ibot] + u1) * exptrm_angle)
                )

            exptrm_angle_mdpt = np.exp(-0.5 * dtau[0] / u1)
            exptrm_positive_mdpt = np.exp(0.5 * exptrm[0])
            exptrm_minus_mdpt = 1.0 / exptrm_positive_mdpt

            flux_at_top = (
                flux_plus[1] * exptrm_angle_mdpt
                + (gcoef[0] / (lamda[0] * u1 - 1.0)) * (exptrm_positive[0] * exptrm_angle_mdpt - exptrm_positive_mdpt)
                - (hcoef[0] / (lamda[0] * u1 + 1.0)) * (exptrm_minus[0] * exptrm_angle_mdpt - exptrm_minus_mdpt)
                + alpha1[0] * (1.0 - exptrm_angle_mdpt)
                + alpha2[0] * (u1 + 0.5 * dtau[0] - (dtau[0] + u1) * exptrm_angle_mdpt)
            )
            thermal_sum += flux_at_top * gweight[ng] * tweight[nt]

    if numt == 1:
        sym_fac = 1.0
    else:
        sym_fac = 1.0 / (2.0 * np.pi)

    return thermal_sum * sym_fac


@nb.experimental.jitclass
class ReflectedWorkspace:
    """Per-thread scratch space for the TOA-only reflected-light solver."""

    nlayer: nb.int64
    g1: nb.float64[:]
    g2: nb.float64[:]
    lamda: nb.float64[:]
    gama: nb.float64[:]
    g3: nb.float64[:]
    a_minus: nb.float64[:]
    a_plus: nb.float64[:]
    c_minus_up: nb.float64[:]
    c_plus_up: nb.float64[:]
    c_minus_down: nb.float64[:]
    c_plus_down: nb.float64[:]
    exptrm: nb.float64[:]
    exptrm_positive: nb.float64[:]
    exptrm_minus: nb.float64[:]
    p_single: nb.float64[:]
    A: nb.float64[:]
    B: nb.float64[:]
    C: nb.float64[:]
    D: nb.float64[:]
    positive: nb.float64[:]
    negative: nb.float64[:]
    xint: nb.float64[:]

    def __init__(self, nlayer):
        self._allocate(nlayer)

    def _allocate(self, nlayer):
        self.nlayer = nlayer
        self.g1 = np.empty(nlayer, dtype=np.float64)
        self.g2 = np.empty(nlayer, dtype=np.float64)
        self.lamda = np.empty(nlayer, dtype=np.float64)
        self.gama = np.empty(nlayer, dtype=np.float64)
        self.g3 = np.empty(nlayer, dtype=np.float64)
        self.a_minus = np.empty(nlayer, dtype=np.float64)
        self.a_plus = np.empty(nlayer, dtype=np.float64)
        self.c_minus_up = np.empty(nlayer, dtype=np.float64)
        self.c_plus_up = np.empty(nlayer, dtype=np.float64)
        self.c_minus_down = np.empty(nlayer, dtype=np.float64)
        self.c_plus_down = np.empty(nlayer, dtype=np.float64)
        self.exptrm = np.empty(nlayer, dtype=np.float64)
        self.exptrm_positive = np.empty(nlayer, dtype=np.float64)
        self.exptrm_minus = np.empty(nlayer, dtype=np.float64)
        self.p_single = np.empty(nlayer, dtype=np.float64)
        self.A = np.empty(2 * nlayer, dtype=np.float64)
        self.B = np.empty(2 * nlayer, dtype=np.float64)
        self.C = np.empty(2 * nlayer, dtype=np.float64)
        self.D = np.empty(2 * nlayer, dtype=np.float64)
        self.positive = np.empty(nlayer, dtype=np.float64)
        self.negative = np.empty(nlayer, dtype=np.float64)
        self.xint = np.empty(nlayer + 1, dtype=np.float64)

    def _ensure(self, nlayer):
        if nlayer != self.nlayer:
            self._allocate(nlayer)


ReflectedWorkspaceType = ReflectedWorkspace.class_type.instance_type


@nb.experimental.jitclass
class ReflectedResult:
    """Persistent reflected-light solver outputs."""

    nwavelengths: nb.int64
    wavelength_um: nb.float64[:]
    albedo: nb.float64[:]
    fpfs: nb.float64[:]

    def __init__(self):
        self._allocate(0)

    def _allocate(self, nwavelengths):
        self.nwavelengths = nwavelengths
        self.wavelength_um = np.empty(nwavelengths, dtype=np.float64)
        self.albedo = np.empty(nwavelengths, dtype=np.float64)
        self.fpfs = np.empty(nwavelengths, dtype=np.float64)

    def _ensure(self, nwavelengths):
        if nwavelengths != self.nwavelengths:
            self._allocate(nwavelengths)


@nb.experimental.jitclass
class ReflectedSolver:
    """Persistent reflected-light solver state."""

    nlevel: nb.int64
    workspace: types.ListType(ReflectedWorkspaceType)

    def __init__(self):
        self._allocate(0)

    def _allocate(self, nlevel):
        self.nlevel = nlevel
        nthreads = nb.get_num_threads()
        self.workspace = typed.List.empty_list(ReflectedWorkspaceType)
        if nlevel <= 0:
            return
        for _ in range(nthreads):
            self.workspace.append(ReflectedWorkspace(nlevel - 1))

    def _ensure(self, nlevel):
        if nlevel != self.nlevel or len(self.workspace) != nb.get_num_threads():
            self._allocate(nlevel)


@nb.njit(parallel=True)
def get_reflected_1d(
    self,
    nlevel,
    nwavelengths_in_chunk,
    ngauss_ck,
    numg,
    numt,
    ck_weights,
    gweight,
    tweight,
    dtau,
    tau,
    w0,
    cosb,
    gcos2,
    ftau_cld,
    ftau_ray,
    dtau_og,
    tau_og,
    w0_og,
    cosb_og,
    surf_reflect,
    ubar0,
    ubar1,
    cos_theta,
    single_phase,
    multi_phase,
    frac_a,
    frac_b,
    frac_c,
    constant_back,
    constant_forward,
    get_toa_intensity,
    get_lvl_flux,
    toon_coefficients,
    b_top,
    albedo,
):
    """Compute TOA reflected-light intensity for a single atmosphere."""

    if not get_toa_intensity:
        raise ValueError("TOA intensity output is required in the current reflected-light solver")
    if get_lvl_flux:
        raise ValueError("level-flux output is not implemented in the experimental reflected-light solver")

    self._ensure(nlevel)

    for iw in nb.prange(nwavelengths_in_chunk):
        albedo_iw = 0.0
        for igauss in range(ngauss_ck):
            albedo_iw += ck_weights[igauss] * get_reflected_1d_w(
                self.workspace[nb.get_thread_id()],
                nlevel,
                numg,
                numt,
                gweight,
                tweight,
                dtau[iw, igauss, :],
                tau[iw, igauss, :],
                w0[iw, igauss, :],
                cosb[iw, igauss, :],
                gcos2[iw, :],
                ftau_cld[iw, :],
                ftau_ray[iw, :],
                dtau_og[iw, igauss, :],
                tau_og[iw, igauss, :],
                w0_og[iw, igauss, :],
                cosb_og[iw, igauss, :],
                surf_reflect[iw],
                ubar0,
                ubar1,
                cos_theta,
                1.0,
                single_phase,
                multi_phase,
                frac_a,
                frac_b,
                frac_c,
                constant_back,
                constant_forward,
                toon_coefficients,
                b_top,
            )
        albedo[iw] = albedo_iw

@nb.njit
def get_reflected_1d_w(
    wrk,
    nlevel,
    numg,
    numt,
    gweight,
    tweight,
    dtau,
    tau,
    w0,
    cosb,
    gcos2,
    ftau_cld,
    ftau_ray,
    dtau_og,
    tau_og,
    w0_og,
    cosb_og,
    surf_reflect,
    ubar0,
    ubar1,
    cos_theta,
    F0PI,
    single_phase,
    multi_phase,
    frac_a,
    frac_b,
    frac_c,
    constant_back,
    constant_forward,
    toon_coefficients,
    b_top,
):
    """Per-wavelength TOA reflected-light solve used by :func:`get_reflected_1d`."""

    nlayer = nlevel - 1
    sq3 = np.sqrt(3.0)

    g1 = wrk.g1
    g2 = wrk.g2
    lamda = wrk.lamda
    gama = wrk.gama
    g3 = wrk.g3
    a_minus = wrk.a_minus
    a_plus = wrk.a_plus
    c_minus_up = wrk.c_minus_up
    c_plus_up = wrk.c_plus_up
    c_minus_down = wrk.c_minus_down
    c_plus_down = wrk.c_plus_down
    exptrm = wrk.exptrm
    exptrm_positive = wrk.exptrm_positive
    exptrm_minus = wrk.exptrm_minus
    p_single = wrk.p_single
    A = wrk.A
    B = wrk.B
    C = wrk.C
    D = wrk.D
    positive = wrk.positive
    negative = wrk.negative
    xint = wrk.xint

    f0pi_w = F0PI
    surf_reflect_w = surf_reflect
    albedo_sum = 0.0

    if toon_coefficients == 1:
        for i in range(nlayer):
            w0_iw = w0[i]
            ft_iw = ftau_cld[i]
            cb_iw = cosb[i]
            g1[i] = (7.0 - w0_iw * (4.0 + 3.0 * ft_iw * cb_iw)) / 4.0
            g2[i] = -(1.0 - w0_iw * (4.0 - 3.0 * ft_iw * cb_iw)) / 4.0
    elif toon_coefficients == 0:
        for i in range(nlayer):
            w0_iw = w0[i]
            ft_iw = ftau_cld[i]
            cb_iw = cosb[i]
            g1[i] = (sq3 * 0.5) * (2.0 - w0_iw * (1.0 + ft_iw * cb_iw))
            g2[i] = (sq3 * w0_iw * 0.5) * (1.0 - ft_iw * cb_iw)

    for i in range(nlayer):
        lamda_i = np.sqrt(g1[i] * g1[i] - g2[i] * g2[i])
        lamda[i] = lamda_i
        gama[i] = (g1[i] - lamda_i) / g2[i]

        exptrm_val = lamda[i] * dtau[i]
        if exptrm_val > 35.0:
            exptrm_val = 35.0
        exptrm[i] = exptrm_val
        exptrm_positive[i] = np.exp(exptrm_val)
        exptrm_minus[i] = 1.0 / exptrm_positive[i]

    for i in range(nlayer):
        g_forward = 0.0
        g_back = 0.0
        f = 0.0
        if single_phase != 1:
            g_forward = constant_forward * cosb_og[i]
            g_back = constant_back * cosb_og[i]
            f = frac_a + frac_b * g_back ** frac_c

        if single_phase == 0:
            HG_forward = (1.0 - g_forward * g_forward) / np.sqrt((1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) ** 3)
            HG_backward = (1.0 - g_back * g_back) / np.sqrt((1.0 + g_back * g_back + 2.0 * g_back * cos_theta) ** 3)
            p_single[i] = f * HG_forward + (1.0 - f) * HG_backward + gcos2[i]
        elif single_phase == 1:
            cb = cosb_og[i]
            p_single[i] = (1.0 - cb * cb) / np.sqrt((1.0 + cb * cb + 2.0 * cb * cos_theta) ** 3)
        elif single_phase == 2:
            HG_forward = (1.0 - g_forward * g_forward) / np.sqrt((1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) ** 3)
            HG_backward = (1.0 - g_back * g_back) / np.sqrt((1.0 + g_back * g_back + 2.0 * g_back * cos_theta) ** 3)
            p_single[i] = f * HG_forward + (1.0 - f) * HG_backward
        elif single_phase == 3:
            HG_forward = (1.0 - g_forward * g_forward) / np.sqrt((1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) ** 3)
            HG_back = (1.0 - g_back * g_back) / np.sqrt((1.0 + g_back * g_back + 2.0 * g_back * cos_theta) ** 3)
            p_single[i] = ftau_cld[i] * (f * HG_forward + (1.0 - f) * HG_back) + ftau_ray[i] * (0.75 * (1.0 + cos_theta * cos_theta))

    for nt in range(numt):
        for ng in range(numg):
            u1 = ubar1[ng, nt]
            u0 = ubar0[ng, nt]
            inv_u0 = 1.0 / u0
            inv_u0_sq = inv_u0 * inv_u0
            inv_u1 = 1.0 / u1
            sum_u = u0 + u1
            inv_sum_u = 1.0 / sum_u
            inv_u0u1 = inv_u0 * inv_u1

            if toon_coefficients == 1:
                for i in range(nlayer):
                    g3[i] = (2.0 - 3.0 * ftau_cld[i] * cosb[i] * u0) / 4.0
            elif toon_coefficients == 0:
                for i in range(nlayer):
                    g3[i] = 0.5 * (1.0 - sq3 * ftau_cld[i] * cosb[i] * u0)

            for i in range(nlayer):
                g4 = 1.0 - g3[i]
                denom = lamda[i] * lamda[i] - inv_u0_sq
                w0_iw = w0[i]
                a_minus[i] = f0pi_w * w0_iw * (g4 * (g1[i] + inv_u0) + g2[i] * g3[i]) / denom
                a_plus[i] = f0pi_w * w0_iw * (g3[i] * (g1[i] - inv_u0) + g2[i] * g4) / denom

                exp_up = np.exp(-tau[i] / u0)
                exp_down = np.exp(-tau[i + 1] / u0)
                c_minus_up[i] = a_minus[i] * exp_up
                c_plus_up[i] = a_plus[i] * exp_up
                c_minus_down[i] = a_minus[i] * exp_down
                c_plus_down[i] = a_plus[i] * exp_down

            b_surface = surf_reflect_w * u0 * f0pi_w * np.exp(-tau[nlevel - 1] * inv_u0)
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
                surf_reflect_w,
                gama,
                exptrm_positive,
                exptrm_minus,
            )
            tri_diag_solve_inplace(2 * nlayer, A, B, C, D)

            for i in range(nlayer):
                positive[i] = D[2 * i] + D[2 * i + 1]
                negative[i] = D[2 * i] - D[2 * i + 1]

            flux_zero = (
                positive[nlayer - 1] * exptrm_positive[nlayer - 1]
                + gama[nlayer - 1] * negative[nlayer - 1] * exptrm_minus[nlayer - 1]
                + c_plus_down[nlayer - 1]
            )
            xint[nlayer] = flux_zero / np.pi

            for i in range(nlayer - 1, -1, -1):
                if multi_phase == 0:
                    ubar2 = 0.767
                    phase_term = 3.0 * ubar2 * ubar2 * u1 * u1 - 1.0
                    multi_plus = 1.0 + 1.5 * ftau_cld[i] * cosb[i] * u1 + gcos2[i] * phase_term / 2.0
                    multi_minus = 1.0 - 1.5 * ftau_cld[i] * cosb[i] * u1 + gcos2[i] * phase_term / 2.0
                elif multi_phase == 1:
                    multi_plus = 1.0 + 1.5 * ftau_cld[i] * cosb[i] * u1
                    multi_minus = 1.0 - 1.5 * ftau_cld[i] * cosb[i] * u1
                else:
                    raise ValueError("multi_phase must be 0 or 1")

                G = positive[i] * (multi_plus + gama[i] * multi_minus) * w0[i] * 0.5 / np.pi
                H = negative[i] * (gama[i] * multi_plus + multi_minus) * w0[i] * 0.5 / np.pi
                source_A = (multi_plus * c_plus_up[i] + multi_minus * c_minus_up[i]) * w0[i] * 0.5 / np.pi

                xint[i] = (
                    xint[i + 1] * np.exp(-dtau[i] * inv_u1)
                    + (w0_og[i] * (f0pi_w * 0.25 / np.pi))
                    * p_single[i]
                    * np.exp(-tau_og[i] * inv_u0)
                    * (1.0 - np.exp(-dtau_og[i] * sum_u * inv_u0u1))
                    * (u0 * inv_sum_u)
                    + source_A * (1.0 - np.exp(-dtau[i] * sum_u * inv_u0u1))
                    * (u0 * inv_sum_u)
                    + G * (np.exp(exptrm[i] - dtau[i] * inv_u1) - 1.0) / (lamda[i] * u1 - 1.0)
                    + H * (1.0 - np.exp(-(exptrm[i] + dtau[i] * inv_u1))) / (lamda[i] * u1 + 1.0)
                )
            albedo_sum += xint[0] * gweight[ng] * tweight[nt]

    if numt == 1:
        sym_fac = 2.0 * np.pi
    else:
        sym_fac = 1.0

    return sym_fac * 0.5 * albedo_sum / F0PI * (cos_theta + 1.0)


@nb.experimental.jitclass
class TransmissionResult:
    """Persistent transmission-spectrum outputs."""

    nwavelengths: nb.int64
    wavelength_um: nb.float64[:]
    rprs2: nb.float64[:]

    def __init__(self):
        self._allocate(0)

    def _allocate(self, nwavelengths):
        self.nwavelengths = nwavelengths
        self.wavelength_um = np.empty(nwavelengths, dtype=np.float64)
        self.rprs2 = np.empty(nwavelengths, dtype=np.float64)

    def _ensure(self, nwavelengths):
        if nwavelengths != self.nwavelengths:
            self._allocate(nwavelengths)


@nb.njit(parallel=True)
def get_transit_1d(
    nlevel,
    nwavelengths_in_chunk,
    ngauss_ck,
    z,
    dz,
    rstar,
    mmw,
    k_b,
    amu,
    player,
    tlayer,
    colden,
    dtau,
    ck_weights,
    transit_depth,
):
    """Compute transmission spectra for a single atmosphere."""

    for iw in nb.prange(nwavelengths_in_chunk):
        transit_iw = 0.0
        for igauss in range(ngauss_ck):
            transit_iw += ck_weights[igauss] * get_transit_1d_w(
                nlevel,
                z,
                dz,
                rstar,
                mmw,
                k_b,
                amu,
                player,
                tlayer,
                colden,
                dtau[iw, igauss, :],
            )
        transit_depth[iw] = transit_iw


@nb.njit
def get_transit_1d_w(
    nlevel,
    z,
    dz,
    rstar,
    mmw,
    k_b,
    amu,
    player,
    tlayer,
    colden,
    dtau,
):
    """Per-wavelength transmission solve used by :func:`get_transit_1d`."""

    nlayer = nlevel - 1
    mmw_grams = mmw * amu
    total = 0.0
    zmin = z[0]
    for i in range(1, nlevel):
        if z[i] < zmin:
            zmin = z[i]

    for i in range(nlevel):
        tauall = 0.0
        reference_shell = z[i]
        for j in range(i):
            inner_shell = z[i - j]
            outer_shell = z[i - j - 1]
            if (inner_shell != reference_shell) and (outer_shell != reference_shell):
                integrate_segment = (
                    np.sqrt(outer_shell * outer_shell - reference_shell * reference_shell)
                    - np.sqrt(inner_shell * inner_shell - reference_shell * reference_shell)
                )
            else:
                integrate_segment = np.sqrt(outer_shell * outer_shell - reference_shell * reference_shell)

            layer_idx = i - j - 1
            tauall += (
                2.0
                * (dtau[layer_idx] / colden[layer_idx] * mmw_grams[layer_idx])
                * integrate_segment
                * player[layer_idx]
                / tlayer[layer_idx]
                / k_b
            )

        total += (1.0 - np.exp(-tauall)) * z[i] * dz[i]

    return (zmin / rstar) ** 2.0 + 2.0 / (rstar * rstar) * total
