from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import astropy.units as u

# Get the root of the repo and prepend to path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from picaso import disco
from picaso import atmsetup
from picaso import justdoit as jdi
from picaso import optics
from picaso import fluxes
from picaso import fluxes_noalloc
from picaso import experimental
from picaso.experimental import fluxes as experimental_fluxes
from picaso.experimental import opacityfiles as experimental_opacityfiles
from picaso.experimental import raman as experimental_raman


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
    ck_weights = np.array([1.0], dtype=np.float64)

    experimental_fluxes.get_thermal_1d.py_func(
        solver,
        case["nlevel"],
        case["nwno"],
        1,
        case["numg"],
        case["numt"],
        ck_weights,
        gweight,
        tweight,
        case["wavelength_um"].copy(),
        case["dtau"][:, None, :].copy(),
        case["w0"][:, None, :].copy(),
        case["cosb"][:, None, :].copy(),
        case["tlevel"].copy(),
        case["plevel"].copy(),
        case["ubar1"].copy(),
        case["surf_reflect"].copy(),
        hard_surface,
        flux,
    )

    return flux


def _make_reflected_case(
    phase_angle=0.0,
    single_phase=0,
    multi_phase=0,
    toon_coefficients=0,
    b_top=0.0,
):
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
        phase_angle=phase_angle,
        cos_theta=float(np.cos(phase_angle)),
        F0PI=np.array([1.13, 1.14, 1.15, 1.16, 1.17]),
        single_phase=single_phase,
        multi_phase=multi_phase,
        frac_a=0.17,
        frac_b=0.27,
        frac_c=1.3,
        constant_back=0.29,
        constant_forward=0.39,
        toon_coefficients=toon_coefficients,
        b_top=b_top,
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
    dtau = case["dtau"][:, None, :].copy()
    tau = np.zeros((case["nwno"], 1, case["nlevel"]), dtype=np.float64)
    tau[:, 0, 1:] = np.cumsum(case["dtau"], axis=1)
    w0 = case["w0"][:, None, :].copy()
    cosb = case["cosb"][:, None, :].copy()
    gcos2 = 0.5 * case["ftau_ray"].copy()
    dtau_og = case["dtau_og"][:, None, :].copy()
    tau_og = np.zeros((case["nwno"], 1, case["nlevel"]), dtype=np.float64)
    tau_og[:, 0, 1:] = np.cumsum(case["dtau_og"], axis=1)
    w0_og = case["w0_og"][:, None, :].copy()
    cosb_og = case["cosb_og"][:, None, :].copy()

    gangle, _, tangle, _ = disco.get_angles_3d(case["numg"], case["numt"])
    ubar0, ubar1, _, _, _ = disco.compute_disco(case["numg"], case["numt"], gangle, tangle, case["phase_angle"])
    _, gweight, _, tweight = disco.get_angles_3d(case["numg"], case["numt"])
    solver = experimental_fluxes.ReflectedSolver()
    albedo = np.full(case["nwno"], np.nan, dtype=np.float64)
    ck_weights = np.array([1.0], dtype=np.float64)

    experimental_fluxes.get_reflected_1d.py_func(
        solver,
        case["nlevel"],
        case["nwno"],
        1,
        case["numg"],
        case["numt"],
        ck_weights,
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
        w0_og,
        cosb_og,
        case["surf_reflect"].copy(),
        ubar0,
        ubar1,
        case["cos_theta"],
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


def _make_transmission_case():
    return dict(
        nlevel=4,
        nwno=5,
        z=np.array([4.0e8, 3.0e8, 2.0e8, 1.0e8], dtype=np.float64),
        dz=np.array([1.0e8, 1.0e8, 1.0e8, 1.0e8], dtype=np.float64),
        rstar=6.957e10,
        mmw=np.array([2.3, 2.4, 2.5], dtype=np.float64),
        player=np.array([1.0e6, 5.0e5, 1.0e5], dtype=np.float64),
        tlayer=np.array([300.0, 250.0, 200.0], dtype=np.float64),
        colden=np.array([1.0e21, 1.1e21, 1.2e21], dtype=np.float64),
        dtau=np.array(
            [
                [0.05, 0.06, 0.07],
                [0.07, 0.08, 0.09],
                [0.17, 0.18, 0.19],
                [0.27, 0.28, 0.29],
                [0.37, 0.38, 0.39],
            ],
            dtype=np.float64,
        ),
    )


def _call_legacy_transmission(case):
    return fluxes.get_transit_1d.py_func(
        case["z"].copy(),
        case["dz"].copy(),
        case["nlevel"],
        case["nwno"],
        case["rstar"],
        case["mmw"].copy(),
        experimental.KB_CGS,
        experimental.AMU_CGS,
        case["player"].copy(),
        case["tlayer"].copy(),
        case["colden"].copy(),
        case["dtau"].T.copy(),
    )


def _call_experimental_transmission(case):
    transit_depth = np.full(case["nwno"], np.nan, dtype=np.float64)
    ck_weights = np.array([1.0], dtype=np.float64)
    experimental_fluxes.get_transit_1d.py_func(
        case["nlevel"],
        case["nwno"],
        1,
        case["z"].copy(),
        case["dz"].copy(),
        case["rstar"],
        case["mmw"].copy(),
        experimental.KB_CGS,
        experimental.AMU_CGS,
        case["player"].copy(),
        case["tlayer"].copy(),
        case["colden"].copy(),
        case["dtau"][:, None, :].copy(),
        ck_weights,
        transit_depth,
    )
    return transit_depth


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
    atm = experimental.Atmosphere(species, pressures, temperatures, mixing_ratios, reference_pressure=1.0e-3)
    np.testing.assert_allclose(atm._atm.species_mu, experimental.get_weights(species))


def _make_legacy_atmosphere(profile_df):
    inputs = jdi.inputs()
    inputs.phase_angle(0.0, num_gangle=10, num_tangle=1)
    inputs.gravity(
        radius=1.0,
        radius_unit=u.Unit("R_earth"),
        mass=1.0,
        mass_unit=u.Unit("M_earth"),
    )

    profile_df = profile_df.copy()
    species_names = [col for col in profile_df.columns if col not in ("pressure", "temperature")]
    mixing = profile_df[species_names].to_numpy(dtype=np.float64)
    mixing /= np.sum(mixing, axis=1, keepdims=True)
    profile_df.loc[:, species_names] = mixing

    inputs.atmosphere(df=profile_df, exclude_mol=None)

    legacy = atmsetup.ATMSETUP(inputs.inputs)
    legacy.planet.radius = inputs.inputs["planet"]["radius"]
    legacy.planet.mass = inputs.inputs["planet"]["mass"]
    legacy.planet.gravity = inputs.inputs["planet"]["gravity"]
    legacy.get_profile()
    legacy.get_mmw()
    legacy.get_density()
    legacy.get_altitude(p_reference=1.0)
    legacy.get_column_density()
    return legacy


def test_atmosphere_and_radtran_atmosphere_parity():
    profile_df = pd.read_csv(jdi.earth_icrccm_pt(), sep=r"\s+")
    species_names = [col for col in profile_df.columns if col not in ("pressure", "temperature")]
    pressures = profile_df["pressure"].to_numpy(dtype=np.float64)
    temperatures = profile_df["temperature"].to_numpy(dtype=np.float64)
    mixing_ratios = profile_df[species_names].to_numpy(dtype=np.float64).T

    legacy = _make_legacy_atmosphere(profile_df)

    new_atm = experimental.Atmosphere(
        species_names,
        pressures,
        temperatures,
        mixing_ratios,
        reference_pressure=1.0,
    )

    np.testing.assert_allclose(
        new_atm._atm.level_pressures,
        np.asarray(legacy.level["pressure_bar"], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        new_atm._atm.level_temperatures,
        np.asarray(legacy.level["temperature"], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        new_atm._atm.level_mixing_ratios,
        np.asarray(legacy.level["mixingratios"], dtype=np.float64).T,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        new_atm._atm.layer_pressures,
        np.asarray(legacy.layer["pressure"], dtype=np.float64) / 1.0e6,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        new_atm._atm.layer_temperatures,
        np.asarray(legacy.layer["temperature"], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        new_atm._atm.layer_mixing_ratios,
        np.asarray(legacy.layer["mixingratios"], dtype=np.float64).T,
        rtol=0.0,
        atol=0.0,
    )

    planet = experimental.Planet(radius=1.0, mass=1.0, semimajor=1.0)
    rad_atm = experimental.RadtranAtmosphere()
    rad_atm.setup(new_atm._atm, planet)

    np.testing.assert_allclose(
        rad_atm.level_pressures_cgs,
        np.asarray(legacy.level["pressure"], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rad_atm.level_temperatures,
        np.asarray(legacy.level["temperature"], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rad_atm.level_mixing_ratios,
        np.asarray(legacy.level["mixingratios"], dtype=np.float64).T,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rad_atm.layer_pressures_cgs,
        np.asarray(legacy.layer["pressure"], dtype=np.float64),
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        rad_atm.layer_temperatures,
        np.asarray(legacy.layer["temperature"], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rad_atm.layer_mixing_ratios,
        np.asarray(legacy.layer["mixingratios"], dtype=np.float64).T,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        rad_atm.level_mubar,
        np.asarray(legacy.level["mmw"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rad_atm.layer_mubar,
        np.asarray(legacy.layer["mmw"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rad_atm.level_z,
        np.asarray(legacy.level["z"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        rad_atm.level_dz,
        np.asarray(legacy.level["dz"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        rad_atm.level_gravity,
        legacy.c.G * rad_atm.mass / (np.asarray(legacy.level["z"], dtype=np.float64) ** 2),
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        rad_atm.level_density,
        np.asarray(legacy.level["den"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        rad_atm.layer_gravity,
        np.asarray(legacy.layer["gravity"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        rad_atm.layer_density,
        np.asarray(legacy.layer["pressure"], dtype=np.float64) / (
            legacy.c.k_b * np.asarray(legacy.layer["temperature"], dtype=np.float64)
        ),
        rtol=1e-12,
        atol=1e-8,
    )
    expected_layer_columns = (
        np.asarray(legacy.layer["mixingratios"], dtype=np.float64).T
        * (
            np.asarray(legacy.layer["colden"], dtype=np.float64)
            / (np.asarray(legacy.layer["mmw"], dtype=np.float64) * legacy.c.amu)
        )
    )
    np.testing.assert_allclose(
        rad_atm.layer_columns,
        expected_layer_columns,
        rtol=1e-12,
        atol=1e-8,
    )


REFLECTED_KERNEL_CASES = [
    pytest.param(
        dict(phase_angle=0.0, single_phase=0, multi_phase=0, toon_coefficients=0),
        id="phase0-single0-multi0-toon0",
    ),
    pytest.param(
        dict(phase_angle=0.0, single_phase=1, multi_phase=0, toon_coefficients=1),
        id="phase0-single1-multi0-toon1",
    ),
    pytest.param(
        dict(phase_angle=0.0, single_phase=2, multi_phase=1, toon_coefficients=0),
        id="phase0-single2-multi1-toon0",
    ),
    pytest.param(
        dict(phase_angle=np.pi / 2.0, single_phase=3, multi_phase=0, toon_coefficients=1),
        id="phase90-single3-multi0-toon1",
    ),
    pytest.param(
        dict(phase_angle=np.pi / 2.0, single_phase=0, multi_phase=1, toon_coefficients=0),
        id="phase90-single0-multi1-toon0",
    ),
]


@pytest.mark.parametrize("reflected_case_kwargs", REFLECTED_KERNEL_CASES)
def test_experimental_reflected_toa_parity(reflected_case_kwargs):
    case = _make_reflected_case(**reflected_case_kwargs)
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


def test_experimental_transmission_toa_parity():
    case = _make_transmission_case()
    expected = _call_legacy_transmission(case)
    actual = _call_experimental_transmission(case)

    assert actual.shape == (case["nwno"],)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_raman_mode_1_parity_with_legacy_pollack():
    wavelength = np.array(
        [0.25, 0.3, 0.355, 0.5123, 0.7777, 0.9818, 1.05],
        dtype=np.float64,
    )
    actual = np.empty_like(wavelength)
    experimental_raman.compute_raman(1, wavelength, actual)

    expected = optics.raman_pollack(2, wavelength)[0]

    assert actual.shape == wavelength.shape
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_hdf5_chunk_shape_adapts_molecular_chunks_for_continuum():
    chunks = (1, 1, 4096)

    assert experimental_opacityfiles._hdf5_chunk_shape(chunks, (20, 30, 60718)) == chunks
    assert experimental_opacityfiles._hdf5_chunk_shape(chunks, (30, 60718)) == (1, 4096)
