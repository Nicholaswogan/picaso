from pathlib import Path
import sys

import numpy as np
import pytest

# Get the root of the repo and prepend to path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from picaso import disco
from picaso import atmsetup
from picaso import fluxes
from picaso import experimental
from picaso import experimental_fluxes


def _make_thermal_case():
    return dict(
        nlevel=4,
        nwno=5,
        numg=2,
        numt=2,
        wavelength_um=np.array([25.0, 20.0, 13.333333333333334, 10.0, 8.333333333333334]),
        wno=np.array([400.0, 500.0, 750.0, 1000.0, 1200.0]),
        tlevel=np.array([900.0, 1000.0, 1100.0, 1200.0]),
        dtau=np.array(
            [
                [0.09, 0.10, 0.11],
                [0.11, 0.12, 0.13],
                [0.21, 0.22, 0.23],
                [0.31, 0.32, 0.33],
                [0.41, 0.42, 0.43],
            ]
        ),
        w0=np.array(
            [
                [0.09, 0.10, 0.11],
                [0.11, 0.12, 0.13],
                [0.21, 0.22, 0.23],
                [0.31, 0.32, 0.33],
                [0.41, 0.42, 0.43],
            ]
        ),
        cosb=np.array(
            [
                [0.09, 0.10, 0.11],
                [0.11, 0.12, 0.13],
                [0.21, 0.22, 0.23],
                [0.31, 0.32, 0.33],
                [0.41, 0.42, 0.43],
            ]
        ),
        plevel=np.array([1.0e2, 2.0e2, 4.0e2, 8.0e2]),
        ubar1=np.array(
            [
                [0.41, 0.42],
                [0.51, 0.52],
            ]
        ),
        surf_reflect=np.array([0.23, 0.24, 0.25, 0.26, 0.27]),
        dwno=250.0,
    )


def _call_legacy(case, hard_surface):
    flux_at_top, _ = fluxes.get_thermal_1d.py_func(
        case["nlevel"],
        case["wno"].copy(),
        case["nwno"],
        case["numg"],
        case["numt"],
        case["tlevel"].copy(),
        case["dtau"].T.copy(),
        case["w0"].T.copy(),
        case["cosb"].T.copy(),
        case["plevel"].copy(),
        case["ubar1"].copy(),
        case["surf_reflect"].copy(),
        hard_surface,
        case["dwno"],
        0,
    )
    _, gweight, _, tweight = disco.get_angles_3d(case["numg"], case["numt"])
    return disco.compress_thermal(case["nwno"], flux_at_top, gweight, tweight)


def _call_experimental(case, hard_surface):
    ind_wv0 = 1
    ind_wv1 = 4
    dtau = case["dtau"][ind_wv0:ind_wv1].copy()
    w0 = case["w0"][ind_wv0:ind_wv1].copy()
    cosb = case["cosb"][ind_wv0:ind_wv1].copy()
    _, gweight, _, tweight = disco.get_angles_3d(case["numg"], case["numt"])
    solver = experimental_fluxes.ThermalSolver()
    result = experimental_fluxes.ThermalResult()
    result._allocate(case["nwno"])
    result.wavelength_um[:] = np.nan
    result.thermal[:] = np.nan

    return experimental_fluxes.get_thermal_1d.py_func(
        solver,
        case["nlevel"],
        ind_wv1 - ind_wv0,
        ind_wv0,
        ind_wv1,
        case["nwno"],
        case["numg"],
        case["numt"],
        gweight,
        tweight,
        case["wavelength_um"][ind_wv0:ind_wv1].copy(),
        dtau,
        w0,
        cosb,
        case["tlevel"].copy(),
        case["plevel"].copy(),
        case["ubar1"].copy(),
        case["surf_reflect"][ind_wv0:ind_wv1].copy(),
        hard_surface,
        result,
    )


@pytest.mark.parametrize("hard_surface", [0, 1])
def test_experimental_thermal_toa_parity(hard_surface):
    case = _make_thermal_case()
    expected = _call_legacy(case, hard_surface)
    actual = _call_experimental(case, hard_surface)

    ind_wv0 = 1
    ind_wv1 = 4
    assert actual.thermal.shape == (case["nwno"],)
    np.testing.assert_allclose(actual.thermal[ind_wv0:ind_wv1], expected[ind_wv0:ind_wv1], rtol=1e-12, atol=1e-12)
    assert np.all(np.isnan(actual.thermal[:ind_wv0]))
    assert np.all(np.isnan(actual.thermal[ind_wv1:]))


def test_get_weights_parity():
    species = ["H2O", "CO2", "12C_16O2", "13C_16O2", "12C_O2", "HCCCN"]
    old_weights = atmsetup.ATMSETUP.get_weights(None, species)
    new_weights = experimental.get_weights(species)
    assert old_weights.keys() == set(species)
    np.testing.assert_allclose(
        new_weights,
        np.array([old_weights[name] for name in species], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_atmosphere_auto_species_mu():
    species = ["H2O", "CO2", "CH4"]
    pressures = np.array([1.0e-6, 1.0e-4, 1.0e-2], dtype=np.float64)
    temperatures = np.array([300.0, 250.0, 200.0], dtype=np.float64)
    mixing_ratios = np.array(
        [
            [1.0e-3, 2.0e-3, 3.0e-3],
            [4.0e-4, 5.0e-4, 6.0e-4],
            [0.9996, 0.9975, 0.9964],
        ],
        dtype=np.float64,
    )
    atm = experimental.Atmosphere(species, pressures, temperatures, mixing_ratios)
    np.testing.assert_allclose(atm._atm.species_mu, experimental.get_weights(species))
