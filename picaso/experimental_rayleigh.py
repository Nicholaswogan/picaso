import numpy as np
import numba as nb
from numba import types
import scipy.constants as sc

N_REF = (101325.0/(sc.k * 273.15)) * 1.0e-6 
SIGNATURE = types.UniTuple(types.float64, 2)(types.float64, types.float64)

###
### Helper functions
###

@nb.njit(types.float64(types.float64, types.float64, types.float64), cache=True)
def compute_sigma_from_eta_F(nu, eta, F):
    sigma_ray = (((24.0 * np.pi**3 * nu**4)/(N_REF**2)) * (((eta**2 - 1.0)/(eta**2 + 2.0))**2) * F)
    return sigma_ray * 6.02214086e+23

@nb.njit(types.float64(types.float64, types.float64, types.float64, types.float64, types.float64), cache=True)
def get_polarizability(nu, f_par, w_par_sq, f_perp, w_perp_sq):
    """Calculate polarisability from Hohm equation 

    Notes
    -----
    .. [1] Hohm, U. 1993. Mol. Phys., 78: 929
    """
    # Now calculate polarisability using formula from Hohm, 1993
    # Polarisability - Hohm, 1993
    alpha = ((1.0/3.0)*((f_par/(w_par_sq - (nu/219474.6305)**2)) +                 
                    2.0*(f_perp/(w_perp_sq - (nu/219474.6305)**2))))     
    return alpha

@nb.njit(types.float64(types.float64, types.float64, types.float64, types.float64, types.float64), cache=True)
def get_anisotropy(nu, f_par, w_par_sq, f_perp, w_perp_sq):
    """get polarisavility anisotropy from Hohm 1993""" 
    gamma = ((f_par/(w_par_sq - (nu/219474.6305)**2)) - 
                (f_perp/(w_perp_sq - (nu/219474.6305)**2)))    
    return gamma
        
@nb.njit(types.float64(types.float64), cache=True)
def get_Lorentz_Lorenz(alpha):
    """Lorentz-Lorenz relation""" 
    return np.sqrt((1.0 + (8.0*np.pi*N_REF*alpha/3.0))/(1.0 - (4.0*np.pi*N_REF*alpha/3.0)))  

@nb.njit(types.float64(types.float64, types.float64), cache=True)
def get_king_correction(alpha, gamma):
    return 1.0 + 2.0 * (gamma/(3.0*alpha))**2 

###
### Rayleigh function for specific species
###

@nb.njit(SIGNATURE, cache=True)
def rayleigh_CH4(wl, nu):
    """Returns polarisability and King correction factor using various sources 
    
    Notes 
    -----
    .. [1] Sneep, M., & Ubachs, W. 2005, JQSRT, 92, 293
    """
    # http://refractiveindex.info (Polyanskiy, 2016)
    if wl < 0.325:
        eta = 1.000504679
    elif wl > 0.633:
        eta = 1.000476653
    else:
        # Sneep & Ubachs, 2005 (sec 5.2) / Hohm, 1993
        eta = 1.0 + (46662.0e-8 + (4.02e-14 * nu**2))
    # scale to 0 C and 1 atm (1.01325 bar), for refractive indices 
    # defined at 15 C and 1013 hPa - Sneep & Ubachs, 2005 
    eta = ((eta-1.0) * (288.15/273.15)) + 1.0 
    # King correction 
    F = 1.0 # Negligable difference from unity - Sneep & Ubachs, (sec 5.2)  
    return eta , F

@nb.njit(SIGNATURE, cache=True)
def rayleigh_CO2(wl, nu):
    """Returns polarisability and King correction factur using formula from Hohm, 1993
    
    Notes
    -----
    .. [1] A. Bideau-Mehu, Y. Guern, R. Abjean and A. Johannin-Gilles. Interferometric determination of the refractive index of carbon dioxide in the ultraviolet region, Opt. Commun. 9, 432-434 (1973)
    """
    if wl < 0.10:
        # Clamp below the fit domain to avoid the unphysical UV pole.
        wl = 0.10
        nu = 1.0e4 / wl
    f_par = 6.00332
    w_par_sq = 0.22525399 
    f_perp = 8.54433 
    w_perp_sq = 0.66083749
    alpha = get_polarizability(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    # convert to cm^3 
    eta = get_Lorentz_Lorenz(alpha * 0.148184e-24)
    gamma = get_anisotropy(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    F = get_king_correction(alpha, gamma)
    return eta, F


@nb.njit(SIGNATURE, cache=True)
def rayleigh_H2(wl, nu):
    """Returns polarisability and King correction factur using formula from Hohm, 1993

    Notes
    -----
    .. [1] Peck, E. R., & Huang, S. 1977, JOSA, 67, 1550
    """
    if wl < 0.095:
        # Clamp below the fit domain to avoid the unphysical UV pole.
        wl = 0.095
        nu = 1.0e4 / wl
    f_par = 1.62632
    w_par_sq = 0.23940245
    f_perp = 1.40105
    w_perp_sq = 0.29486069
    alpha = get_polarizability(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    # convert to cm^3
    eta = get_Lorentz_Lorenz(alpha * 0.148184e-24)
    gamma = get_anisotropy(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    F = get_king_correction(alpha, gamma)
    return eta, F


@nb.njit(SIGNATURE, cache=True)
def rayleigh_H2O(wl, nu):
    """Returns polarisability and King correction factur using various sources

    Notes
    -----
    .. [1] Hill, R. J., & Lawrence, R. S. 1986, InfPh, 26, 371
    .. [2] Polyanskiy, M. N. 2016, Refractive index database, http://refractiveindex.info
    """
    # http://refractiveindex.info (Polyanskiy, 2016)
    eta = 1.0 + ((3.011e-2 / (124.40 - 1.0 / (wl**2.0))) +
                 (7.46e-3 * (0.203 - 1.0 / wl)) /
                 (1.03 - 1.98e3 / (wl**2.0) + 8.1e4 / (wl**4.0) - 1.7e8 / (wl**8.0)))
    if wl < 0.360:
        eta = 1.000258047
    elif wl > 17.60:
        eta = 1.000000000 # Technically formula goes to 19um, but can't have n<1.0
    F = 1.001005
    return eta, F


@nb.njit(SIGNATURE, cache=True)
def rayleigh_He(wl, nu):
    """Returns polarisability and King correction factur using various sources

    Notes
    -----
    .. [1] C. Cuthbertson and C. Cuthbertson. The refraction and dispersion of neon and helium. Proc. R. Soc. London A 135, 40-47 (1936)
    .. [2] Polyanskiy, M. N. 2016, Refractive index database, http://refractiveindex.info
    .. [3] Mansfield, C. R., & Peck, E. R. 1969, JOSA, 59, 199
    """
    # http://refractiveindex.info (Polyanskiy, 2016)
    eta = 1.0 + ((0.014755297 / (426.29740 - 1.0 / (wl**2.0))) * 1.0018141444038913) # Cuthbertson & Cuthbertson, 1936 (multiplicative factor for continuity)
    if wl < 0.2753:
        eta = 1.00003578
    elif wl > 0.4801:
        eta = 1.0 + (0.01470091 / (423.98 - 1.0 / (wl**2.0))) # Mansfield & Peck, 1969
    if wl > 2.0586:
        eta = 1.00003469
    F = 1.0 # Spherical atom, so King correction factor = 1
    return eta, F


@nb.njit(SIGNATURE, cache=True)
def rayleigh_N2(wl, nu):
    """Returns polarisability and King correction factur using various sources

    Notes
    -----
    .. [1] Sneep, M., & Ubachs, W. 2005, JQSRT, 92, 293
    .. [2] E. R. Peck and B. N. Khanna. Dispersion of nitrogen, J. Opt. Soc. Am. 56, 1059-1063 (1966)
    """
    eta = 1.0 + ((5677.465e-8 + (318.81874e4 / (14.4e9 - nu**2.0))) * 1.0001468057477378) # Sneep & Ubachs, 2005 (sec 4.2) / Bates, 1984  (fmultiplicative actor for continuity)
    if wl < 0.2540:
        eta = 1.00030493
    elif wl > 0.46816:
        eta = 1.0 + (6498.2e-8 + (307.43305e4 / (14.4e9 - nu**2.0))) # Sneep & Ubachs, 2005 (sec 4.2) / Peck & Khanna, 1966
    if wl > 2.0576:
        eta = 1.00027883
    # scale to 0 C and 1 atm (1.01325 bar), for refractive indices defined at 15 C and 1013 hPa - Sneep & Ubachs, 2005
    eta = ((eta - 1.0) * (288.15 / 273.15)) + 1.0
    F = 1.034 + 3.17e-12 * nu**2.0 # Sneep & Ubachs, 2005 (sec 4.2) / Bates, 1984
    return eta, F


@nb.njit(SIGNATURE, cache=True)
def rayleigh_N2O(wl, nu):
    """Returns polarisability and King correction factur using various sources

    Notes
    -----
    .. [1] C. Cuthbertson and C. Cuthbertson. On the refraction and dispersion of the halogens, halogen acids, ozone, steam, oxides of nitrogen and ammonia, Phil. Trans. R. Soc. Lond. A 213, 1-26 (1914)
    """
    f_par = 5.65126
    w_par_sq = 0.17424213
    f_perp = 9.72095
    w_perp_sq = 0.72904985
    alpha = get_polarizability(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    # convert to cm^3
    eta = get_Lorentz_Lorenz(alpha * 0.148184e-24)
    gamma = get_anisotropy(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    F = get_king_correction(alpha, gamma)
    return eta, F


@nb.njit(SIGNATURE, cache=True)
def rayleigh_NH3(wl, nu):
    """Returns polarisability and King correction factur using various sources

    Notes
    -----
    .. [1] C. Cuthbertson and C. Cuthbertson. On the refraction and dispersion of the halogens, halogen acids, ozone, steam, oxides of nitrogen and ammonia, Phil. Trans. R. Soc. Lond. A 213, 1-26 (1914)
    .. [2] Hohm, U. 1993. Mol. Phys., 78: 929
    """
    if wl < 0.16:
        # Clamp below the fit domain to avoid the unphysical UV pole.
        wl = 0.16
        nu = 1.0e4 / wl
    f_par = 1.28964
    w_par_sq = 0.08454599
    f_perp = 10.84943
    w_perp_sq = 0.76338846
    alpha = get_polarizability(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    # convert to cm^3
    eta = get_Lorentz_Lorenz(alpha * 0.148184e-24)
    gamma = get_anisotropy(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    F = get_king_correction(alpha, gamma)
    return eta, F


@nb.njit(SIGNATURE, cache=True)
def rayleigh_O2(wl, nu):
    """Returns polarisability and King correction factur using various sources

    Notes
    -----
    .. [1] P. L. Smith, M. C. E. Huber, W. H. Parkinson. Refractivities of H2, He, O2, CO, and Kr for 168≤λ≤288 nm Phys Rev. A 13, 199-203 (1976)
    .. [2] Hohm, U. 1993. Mol. Phys., 78: 929
    """
    if wl < 0.11:
        # Clamp below the fit domain to avoid the unphysical UV pole.
        wl = 0.11
        nu = 1.0e4 / wl
    f_par = 2.74876
    w_par_sq = 0.18095751
    f_perp = 4.86007
    w_perp_sq = 0.58545449
    alpha = get_polarizability(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    # convert to cm^3
    eta = get_Lorentz_Lorenz(alpha * 0.148184e-24)
    gamma = get_anisotropy(nu, f_par, w_par_sq, f_perp, w_perp_sq)
    F = get_king_correction(alpha, gamma)
    return eta, F

###
### Global dictionaries
###

RAYLEIGH_FCNS = {
    'CH4': rayleigh_CH4,
    'CO2': rayleigh_CO2,
    'H2': rayleigh_H2,
    'H2O': rayleigh_H2O,
    'He': rayleigh_He,
    'N2': rayleigh_N2,
    'N2O': rayleigh_N2O,
    'NH3': rayleigh_NH3,
    'O2': rayleigh_O2,
}


RAYLEIGH_POLARISABILITIES = {
    'H2':  0.80e-24,  'He':  0.21e-24, 'N2':   1.74e-24,  'O2':  1.58e-24, 
    'O3':  3.21e-24,  'H2O': 1.45e-24, 'CH4':  2.59e-24,  'CO':  1.95e-24,
    'CO2': 2.91e-24,  'NH3': 2.26e-24, 'HCN':  2.59e-24,  'PH3': 4.84e-24, 
    'SO2': 3.72e-24,  'SO3': 4.84e-24, 'C2H2': 3.33e-24,  'H2S': 3.78e-24,
    'NO':  1.70e-24,  'NO2': 3.02e-24, 'H3+':  0.385e-24, 'OH':  6.965e-24,   # H3+from Kawaoka & Borkman, 1971
    'Na':  24.11e-24, 'K':   42.9e-24, 'Li':   24.33e-24, 'Rb':  47.39e-24,     
    'Cs':  59.42e-24, 'TiO': 16.9e-24, 'VO':   14.4e-24,  'AlO': 8.22e-24,    # Without tabulated values for metal
    'SiO': 5.53e-24,  'CaO': 23.8e-24, 'TiH':  16.9e-24,  'MgH': 10.5e-24,    # oxides and hydrides, these are taken
    'NaH': 24.11e-24, 'AlH': 8.22e-24, 'CrH':  11.6e-24,  'FeH': 9.47e-24,    # to be metal atom polarisabilities
    'CaH': 23.8e-24,  'BeH': 5.60e-24, 'ScH':  21.2e-24
}

RAYLEIGH_KING_CORRECTION_NO_WAVE = {
    "O3": 1.060000,   "CO": 1.016995,  "C2H2": 1.064385,
    "C2H6": 1.006063, "OCS": 1.138786, "CH3Cl": 1.026042,
    "H2S": 1.001880,  "SO2": 1.062638
}

RAYLEIGH_MOLECULES = list(RAYLEIGH_POLARISABILITIES.keys())

###
### Drivers
###

@nb.njit(cache=True)
def _compute_sigma(fcn, wl, sigma):
    for i in range(len(wl)):
        nu = 1.0e4/wl[i]
        eta, F = fcn(wl[i], nu)
        sigma[i] = compute_sigma_from_eta_F(nu, eta, F)

@nb.njit(cache=True)
def _compute_sigma_default(wl, sigma, eta, F):
    for i in range(len(wl)):
        nu = 1.0e4 / wl[i]
        sigma[i] = compute_sigma_from_eta_F(nu, eta, F)

def compute_sigma(species, wl, sigma):

    # Ensure inputs
    if len(wl) != len(sigma):
        raise ValueError(
            f"wl and sigma must have the same length, got {len(wl)} and {len(sigma)}"
        )
    
    # Specialized case
    fcn = RAYLEIGH_FCNS.get(species)
    if fcn is not None:
        _compute_sigma(fcn, wl, sigma)
        return
    
    # Default case
    alpha = RAYLEIGH_POLARISABILITIES.get(species)
    if alpha is None:
        eta = 0.0
    else:
        eta = get_Lorentz_Lorenz(alpha)
    F = RAYLEIGH_KING_CORRECTION_NO_WAVE.get(species, 1.0)
    _compute_sigma_default(wl, sigma, eta, F)
