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
from picaso import fluxes_noalloc
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
    _, gweight, _, tweight = disco.get_angles_3d(case["numg"], case["numt"])
    solver = experimental_fluxes.ThermalSolver()
    flux = np.full(case["nwno"], np.nan, dtype=np.float64)

    experimental_fluxes.get_thermal_1d.py_func(
        solver,
        case["nlevel"],
        case["nwno"],
        case["numg"],
        case["numt"],
        gweight,
        tweight,
        case["wavelength_um"].copy(),
        case["dtau"].copy(),
        case["w0"].copy(),
        case["cosb"].copy(),
        case["tlevel"].copy(),
        case["plevel"].copy(),
        case["ubar1"].copy(),
        case["surf_reflect"].copy(),
        hard_surface,
        flux,
    )

    return flux


def _make_reflected_case():
    return dict(
        nlevel=4,
        nwno=5,
        numg=2,
        numt=2,
        wavelength_um=np.array([25.0, 20.0, 13.333333333333334, 10.0, 8.333333333333334]),
        wno=np.array([400.0, 500.0, 750.0, 1000.0, 1200.0]),
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
        ftau_cld=np.array(
            [
                [0.19, 0.20, 0.21],
                [0.21, 0.22, 0.23],
                [0.31, 0.32, 0.33],
                [0.41, 0.42, 0.43],
                [0.51, 0.52, 0.53],
            ]
        ),
        ftau_ray=np.array(
            [
                [0.81, 0.80, 0.79],
                [0.79, 0.78, 0.77],
                [0.69, 0.68, 0.67],
                [0.59, 0.58, 0.57],
                [0.49, 0.48, 0.47],
            ]
        ),
        dtau_og=np.array(
            [
                [0.05, 0.06, 0.07],
                [0.07, 0.08, 0.09],
                [0.17, 0.18, 0.19],
                [0.27, 0.28, 0.29],
                [0.37, 0.38, 0.39],
            ]
        ),
        w0_og=np.array(
            [
                [0.51, 0.52, 0.53],
                [0.54, 0.55, 0.56],
                [0.57, 0.58, 0.59],
                [0.60, 0.61, 0.62],
                [0.63, 0.64, 0.65],
            ]
        ),
        cosb_og=np.array(
            [
                [0.14, 0.15, 0.16],
                [0.24, 0.25, 0.26],
                [0.34, 0.35, 0.36],
                [0.44, 0.45, 0.46],
                [0.54, 0.55, 0.56],
            ]
        ),
        surf_reflect=np.array([0.23, 0.24, 0.25, 0.26, 0.27]),
        phase_angle=0.0,
        cos_theta=0.37,
        F0PI=np.array([1.13, 1.14, 1.15, 1.16, 1.17]),
        single_phase=0,
        multi_phase=0,
        frac_a=0.17,
        frac_b=0.27,
        frac_c=1.3,
        constant_back=0.29,
        constant_forward=0.39,
        toon_coefficients=0,
        b_top=0.07,
    )


def _call_legacy_reflected(case):
    gangle, _, tangle, _ = disco.get_angles_3d(case["numg"], case["numt"])
    ubar0, ubar1, _, _, _ = disco.compute_disco(case["numg"], case["numt"], gangle, tangle, case["phase_angle"])
    xint_at_top, _ = fluxes_noalloc.get_reflected_1d.py_func(
        case["nlevel"],
        case["wno"].copy(),
        case["nwno"],
        case["numg"],
        case["numt"],
        case["dtau"].T.copy(),
        np.pad(np.cumsum(case["dtau"].T, axis=0), ((1, 0), (0, 0)), mode="constant"),
        case["w0"].T.copy(),
        case["cosb"].T.copy(),
        (0.5 * case["ftau_ray"]).T.copy(),
        case["ftau_cld"].T.copy(),
        case["ftau_ray"].T.copy(),
        case["dtau_og"].T.copy(),
        np.pad(np.cumsum(case["dtau_og"].T, axis=0), ((1, 0), (0, 0)), mode="constant"),
        case["w0_og"].T.copy(),
        case["cosb_og"].T.copy(),
        case["surf_reflect"].copy(),
        ubar0,
        ubar1,
        case["cos_theta"],
        case["F0PI"].copy(),
        case["single_phase"],
        case["multi_phase"],
        case["frac_a"],
        case["frac_b"],
        case["frac_c"],
        case["constant_back"],
        case["constant_forward"],
        1,
        1,
        case["toon_coefficients"],
        case["b_top"],
    )
    return xint_at_top


def _call_experimental_reflected(case):
    dtau = case["dtau"].copy()
    tau = np.zeros((case["nwno"], case["nlevel"]), dtype=np.float64)
    tau[:, 1:] = np.cumsum(dtau, axis=1)
    w0 = case["w0"].copy()
    cosb = case["cosb"].copy()
    gcos2 = 0.5 * case["ftau_ray"].copy()
    dtau_og = case["dtau_og"].copy()
    tau_og = np.zeros((case["nwno"], case["nlevel"]), dtype=np.float64)
    tau_og[:, 1:] = np.cumsum(dtau_og, axis=1)

    gangle, _, tangle, _ = disco.get_angles_3d(case["numg"], case["numt"])
    ubar0, ubar1, _, _, _ = disco.compute_disco(case["numg"], case["numt"], gangle, tangle, case["phase_angle"])
    _, gweight, _, tweight = disco.get_angles_3d(case["numg"], case["numt"])
    solver = experimental_fluxes.ReflectedSolver()
    albedo = np.full(case["nwno"], np.nan, dtype=np.float64)

    experimental_fluxes.get_reflected_1d.py_func(
        solver,
        case["nlevel"],
        case["nwno"],
        case["numg"],
        case["numt"],
        gweight,
        tweight,
        dtau,
        tau,
        w0,
        cosb,
        gcos2,
        case["ftau_cld"].copy(),
        case["ftau_ray"].copy(),
        dtau_og,
        tau_og,
        case["w0_og"].copy(),
        case["cosb_og"].copy(),
        case["surf_reflect"].copy(),
        ubar0,
        ubar1,
        case["cos_theta"],
        case["F0PI"].copy(),
        case["single_phase"],
        case["multi_phase"],
        case["frac_a"],
        case["frac_b"],
        case["frac_c"],
        case["constant_back"],
        case["constant_forward"],
        1,
        0,
        case["toon_coefficients"],
        case["b_top"],
        albedo,
    )
    return albedo


@pytest.mark.parametrize("hard_surface", [0, 1])
def test_experimental_thermal_toa_parity(hard_surface):
    case = _make_thermal_case()
    expected = _call_legacy(case, hard_surface)
    actual = _call_experimental(case, hard_surface)

    assert actual.shape == (case["nwno"],)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


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


def test_experimental_reflected_toa_parity():
    case = _make_reflected_case()
    gangle, gweight, tangle, tweight = disco.get_angles_3d(case["numg"], case["numt"])
    _, _, _, _, _ = disco.compute_disco(case["numg"], case["numt"], gangle, tangle, case["phase_angle"])
    expected = disco.compress_disco(
        case["nwno"],
        case["cos_theta"],
        _call_legacy_reflected(case),
        gweight,
        tweight,
        case["F0PI"].copy(),
    )
    actual = _call_experimental_reflected(case)

    assert actual.shape == (case["nwno"],)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
