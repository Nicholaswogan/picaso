from pathlib import Path
import sys

import numpy as np
import pytest

# Get the root of the repo and prepend to path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from picaso import disco
from picaso import fluxes
from picaso import experimental_fluxes


def _make_thermal_case():
    return dict(
        nlevel=4,
        nwno=3,
        numg=2,
        numt=2,
        wavelength_um=np.array([20.0, 13.333333333333334, 10.0]),
        wno=np.array([500.0, 750.0, 1000.0]),
        tlevel=np.array([900.0, 1000.0, 1100.0, 1200.0]),
        dtau=np.array(
            [
                [0.11, 0.12, 0.13],
                [0.21, 0.22, 0.23],
                [0.31, 0.32, 0.33],
            ]
        ),
        w0=np.array(
            [
                [0.11, 0.12, 0.13],
                [0.21, 0.22, 0.23],
                [0.31, 0.32, 0.33],
            ]
        ),
        cosb=np.array(
            [
                [0.11, 0.12, 0.13],
                [0.21, 0.22, 0.23],
                [0.31, 0.32, 0.33],
            ]
        ),
        plevel=np.array([1.0e2, 2.0e2, 4.0e2, 8.0e2]),
        ubar1=np.array(
            [
                [0.41, 0.42],
                [0.51, 0.52],
            ]
        ),
        surf_reflect=np.array([0.23, 0.24, 0.25]),
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
        case["dtau"].copy(),
        case["w0"].copy(),
        case["cosb"].copy(),
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
    dtau = case["dtau"].T.copy()
    w0 = case["w0"].T.copy()
    cosb = case["cosb"].T.copy()
    _, gweight, _, tweight = disco.get_angles_3d(case["numg"], case["numt"])
    solver = experimental_fluxes.ThermalSolver()
    result = experimental_fluxes.ThermalResult()

    return experimental_fluxes.get_thermal_1d.py_func(
        solver,
        case["nlevel"],
        case["nwno"],
        case["numg"],
        case["numt"],
        gweight,
        tweight,
        case["wavelength_um"].copy(),
        dtau,
        w0,
        cosb,
        case["tlevel"].copy(),
        case["plevel"].copy(),
        case["ubar1"].copy(),
        case["surf_reflect"].copy(),
        hard_surface,
        result,
    )


@pytest.mark.parametrize("hard_surface", [0, 1])
def test_experimental_thermal_toa_parity(hard_surface):
    case = _make_thermal_case()
    expected = _call_legacy(case, hard_surface)
    actual = _call_experimental(case, hard_surface)

    assert actual.thermal.shape == (case["nwno"],)
    np.testing.assert_allclose(actual.thermal, expected, rtol=1e-12, atol=1e-12)
