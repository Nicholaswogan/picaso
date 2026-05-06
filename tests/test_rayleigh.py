import numpy as np
from numpy.testing import assert_allclose

from picaso.experimental_rayleigh import compute_sigma as compute_sigma_new
from picaso.rayleigh import Rayleigh


WAVELENGTH_UM = np.array([0.25, 0.30, 0.50, 1.00, 5.00, 10.0], dtype=np.float64)
WAVENUMBER_CM1 = 1.0e4 / WAVELENGTH_UM


def _compute_sigma_old(species):
    return Rayleigh(WAVENUMBER_CM1).compute_sigma(species)


def _compute_sigma_new(species):
    sigma = np.empty_like(WAVELENGTH_UM)
    compute_sigma_new(species, WAVELENGTH_UM, sigma)
    return sigma


def test_rayleigh_specific_species_parity():
    species_list = ["CH4", "CO2", "H2", "H2O", "He", "N2", "N2O", "NH3", "O2"]
    for species in species_list:
        sigma_old = _compute_sigma_old(species)
        sigma_new = _compute_sigma_new(species)
        assert_allclose(sigma_new, sigma_old, rtol=1e-12, atol=1e-14, err_msg=species)


def test_rayleigh_generic_species_parity():
    species_list = ["O3", "CO", "C2H2", "H2S", "SO2", "HCCCN", "C2H6"]
    for species in species_list:
        sigma_old = _compute_sigma_old(species)
        sigma_new = _compute_sigma_new(species)
        assert_allclose(sigma_new, sigma_old, rtol=1e-12, atol=1e-14, err_msg=species)
