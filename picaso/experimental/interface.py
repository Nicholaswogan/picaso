import numpy as np
import pandas as pd

from . import driver

class ExperimentalRT:

    def __init__(self, opacity_filename, wavelength_range, nwavelengths_per_chunk):
        self.rad = driver.Radtran(opacity_filename, wavelength_range)
        self.wno = 1.0e4/self.rad.opacities.wavelength[::-1]
        self.nwavelengths_per_chunk = nwavelengths_per_chunk

def opannection(
    wave_range=None, 
    filename_db=None, 
    resample=1, 
    method='resampled',
    ck_db=None, 
    raman_db=None, 
    preload_gases='all',
    query_method='linear',
    verbose=False,
    nwavelengths_per_chunk=10_000,
):

    # Initial checks
    if resample != 1:
        raise ValueError(f"resample must be 1, got {resample}")
    if method != 'resampled':
        raise ValueError(f"method must be 'resampled', got {method!r}")
    if ck_db is not None:
        raise ValueError("ck_db is not supported by the experimental interface")
    if raman_db is not None:
        raise ValueError("raman_db is not supported by the experimental interface")
    if query_method != 'linear':
        raise ValueError(f"query_method must be 'linear', got {query_method!r}")
    
    opa = ExperimentalRT(
        opacity_filename=filename_db,
        wavelength_range=wave_range,
        nwavelengths_per_chunk=nwavelengths_per_chunk,
    )

    return opa

def bundle_to_atmosphere(bundle):
    """Convert a legacy PICASO bundle into a new experimental Atmosphere."""

    profile = bundle.inputs["atmosphere"]["profile"]
    approx = bundle.inputs["approx"]

    if not isinstance(profile, pd.DataFrame):
        raise ValueError(
            f"bundle.inputs['atmosphere']['profile'] must be a pandas DataFrame, got {type(profile)!r}"
        )
    if "pressure" not in profile.columns:
        raise ValueError("bundle atmosphere profile must contain a 'pressure' column")
    if "temperature" not in profile.columns:
        raise ValueError("bundle atmosphere profile must contain a 'temperature' column")

    profile = profile.sort_values("pressure").reset_index(drop=True)

    ignore_columns = {
        "pressure",
        "temperature",
        "kz",
        "kzz",
        "lat",
        "latitude",
        "lon",
        "longitude",
        "phase",
    }
    species_names = [name for name in profile.columns if name not in ignore_columns]
    if len(species_names) == 0:
        raise ValueError("bundle atmosphere profile must contain at least one species column")

    pressures = profile["pressure"].to_numpy(dtype=np.float64, copy=True)
    temperatures = profile["temperature"].to_numpy(dtype=np.float64, copy=True)
    mixing_ratios = profile.loc[:, species_names].to_numpy(dtype=np.float64, copy=True).T
    reference_pressure = approx["p_reference"]

    return driver.Atmosphere(
        species_names=species_names,
        pressures=pressures,
        temperatures=temperatures,
        mixing_ratios=mixing_ratios,
        reference_pressure=reference_pressure,
    )


def bundle_to_planet(bundle):
    """Convert a legacy PICASO bundle into a new experimental Planet."""

    if "planet" not in bundle.inputs:
        raise ValueError("bundle.inputs must contain a 'planet' section")
    if "star" not in bundle.inputs:
        raise ValueError("bundle.inputs must contain a 'star' section")

    planet = bundle.inputs["planet"]
    star = bundle.inputs["star"]
    if "radius" not in planet:
        raise ValueError("bundle.inputs['planet'] must contain a 'radius' field")
    if "mass" not in planet:
        raise ValueError("bundle.inputs['planet'] must contain a 'mass' field")
    if "semi_major" not in star:
        raise ValueError("bundle.inputs['star'] must contain a 'semi_major' field")

    radius = float(planet["radius"]) / driver.R_EARTH_CGS
    mass = float(planet["mass"]) / driver.M_EARTH_CGS
    semi_major = star["semi_major"]
    if isinstance(semi_major, str) and semi_major == 'nostar':
        semi_major = np.nan
    else:
        semi_major = float(semi_major) / 1.495978707e13

    return driver.Planet(radius=radius, mass=mass, semimajor=semi_major)


def bundle_to_clouds(bundle, opacityclass, atmosphere):
    """Convert a legacy PICASO bundle into a new experimental Clouds object."""
    if "clouds" not in bundle.inputs:
        raise ValueError("bundle.inputs must contain a 'clouds' section")

    clouds = bundle.inputs["clouds"]
    if not isinstance(clouds, dict):
        raise ValueError(f"bundle.inputs['clouds'] must be a dict, got {type(clouds)!r}")
    if "profile" not in clouds:
        raise ValueError("bundle.inputs['clouds'] must contain a 'profile' field")

    profile = clouds["profile"]
    if profile is None:
        return None
    if not isinstance(profile, pd.DataFrame):
        raise ValueError(
            f"bundle.inputs['clouds']['profile'] must be a pandas DataFrame, got {type(profile)!r}"
        )

    # Get each variable without copies.
    # Legacy cloud tables are stored pressure-major, then wavenumber-major.
    pressure = atmosphere._atm.layer_pressures
    nwavelength = opacityclass.rad.opacities.nwavelength
    nlayer = len(pressure)
    if profile.shape[0] != nlayer * nwavelength:
        raise ValueError(
            "bundle cloud profile must contain a complete pressure x wavenumber grid, "
            f"got {profile.shape[0]} rows for {nlayer} layers and {nwavelength} wavelengths"
        )
    shape = (nlayer, nwavelength)
    opd = profile['opd'].to_numpy(copy=False).reshape(shape).T[::-1, :]
    w0 = profile['w0'].to_numpy(copy=False).reshape(shape).T[::-1, :]
    g0 = profile['g0'].to_numpy(copy=False).reshape(shape).T[::-1, :]
    do_holes = clouds.get("do_holes", False)
    fthin_cld = clouds.get("fthin_cld", 1.0)
    fhole = clouds.get("fhole", 0.0)

    return driver.Clouds(
        wavelength=opacityclass.rad.opacities.wavelength,
        pressure=pressure,
        opd=opd,
        w0=w0,
        g0=g0,
        do_holes=do_holes,
        fthin_cld=fthin_cld,
        fhole=fhole,
    )

def bundle_to_surface(bundle, opacityclass):
    """Convert a legacy PICASO bundle into a new experimental Surface object."""
    if "surface_reflect" not in bundle.inputs:
        return None
    if "hard_surface" not in bundle.inputs:
        raise ValueError("bundle.inputs must contain a 'hard_surface' field when 'surface_reflect' is present")

    reflectance = bundle.inputs["surface_reflect"]
    hard_surface = bool(bundle.inputs["hard_surface"])
    if reflectance is None:
        return None
    if not isinstance(hard_surface, (bool, np.bool_)):
        raise ValueError(
            f"bundle.inputs['hard_surface'] must be a bool, got {type(hard_surface)!r}"
        )

    wavelength = opacityclass.rad.opacities.wavelength
    if np.isscalar(reflectance):
        return driver.Surface(
            hard_surface=bool(hard_surface),
            reflectance=float(reflectance),
        )

    reflectance = np.asarray(reflectance, dtype=np.float64)
    if reflectance.ndim != 1:
        raise ValueError(
            f"bundle.inputs['surface_reflect'] must be scalar or 1D, got shape {reflectance.shape}"
        )
    if reflectance.shape[0] != wavelength.shape[0]:
        raise ValueError(
            "bundle.inputs['surface_reflect'] must match the opacity wavelength grid length, "
            f"got {reflectance.shape[0]} and {wavelength.shape[0]}"
        )

    return driver.Surface(
        hard_surface=bool(hard_surface),
        reflectance=reflectance[::-1].copy(),
        wavelength=wavelength.copy(),
    )

def bundle_to_star(bundle):
    """Convert a legacy PICASO bundle into a new experimental Star object."""
    if "star" not in bundle.inputs:
        raise ValueError("bundle.inputs must contain a 'star' section")

    star = bundle.inputs["star"]
    if not isinstance(star, dict):
        raise ValueError(f"bundle.inputs['star'] must be a dict, got {type(star)!r}")

    radius = star.get("radius", None)
    if radius is None or (isinstance(radius, str) and radius == "nostar"):
        return None
    if "flux" not in star:
        raise ValueError("bundle.inputs['star'] must contain a 'flux' field when a star radius is present")
    if "wno" not in star:
        raise ValueError("bundle.inputs['star'] must contain a 'wno' field when a star radius is present")

    flux = star["flux"]
    wno = star["wno"]
    if isinstance(flux, str) or isinstance(wno, str):
        raise ValueError("bundle.inputs['star']['flux'] and ['wno'] must be numeric arrays when a star radius is present")

    flux = np.asarray(flux, dtype=np.float64)
    wno = np.asarray(wno, dtype=np.float64)
    if flux.ndim != 1:
        raise ValueError(f"bundle.inputs['star']['flux'] must be 1D, got shape {flux.shape}")
    if wno.ndim != 1:
        raise ValueError(f"bundle.inputs['star']['wno'] must be 1D, got shape {wno.shape}")
    if flux.shape[0] != wno.shape[0]:
        raise ValueError(
            "bundle.inputs['star']['flux'] and ['wno'] must have the same length, "
            f"got {flux.shape[0]} and {wno.shape[0]}"
        )

    radius = float(radius) / driver.R_SUN_CGS
    wavelength = np.asarray(1.0e4 / wno[::-1], dtype=np.float64)
    spectrum = flux[::-1].copy()

    return driver.Star(
        radius=radius,
        wavelength=wavelength,
        spectrum=spectrum,
    )


def bundle_to_settings(bundle):
    """Convert a legacy PICASO bundle into experimental RadtranSettings."""
    if "approx" not in bundle.inputs:
        raise ValueError("bundle.inputs must contain an 'approx' section")

    approx = bundle.inputs["approx"]
    if not isinstance(approx, dict):
        raise ValueError(f"bundle.inputs['approx'] must be a dict, got {type(approx)!r}")
    if "rt_params" not in approx:
        raise ValueError("bundle.inputs['approx'] must contain an 'rt_params' section")

    rt_params = approx["rt_params"]
    if not isinstance(rt_params, dict):
        raise ValueError(f"bundle.inputs['approx']['rt_params'] must be a dict, got {type(rt_params)!r}")
    if "common" not in rt_params:
        raise ValueError("bundle.inputs['approx']['rt_params'] must contain a 'common' section")
    if "toon" not in rt_params:
        raise ValueError("bundle.inputs['approx']['rt_params'] must contain a 'toon' section")

    common = rt_params["common"]
    toon = rt_params["toon"]
    if not isinstance(common, dict):
        raise ValueError(f"bundle.inputs['approx']['rt_params']['common'] must be a dict, got {type(common)!r}")
    if not isinstance(toon, dict):
        raise ValueError(f"bundle.inputs['approx']['rt_params']['toon'] must be a dict, got {type(toon)!r}")

    if "TTHG_params" not in common:
        raise ValueError("bundle.inputs['approx']['rt_params']['common'] must contain a 'TTHG_params' section")
    tthg = common["TTHG_params"]
    if not isinstance(tthg, dict):
        raise ValueError(f"bundle.inputs['approx']['rt_params']['common']['TTHG_params'] must be a dict, got {type(tthg)!r}")
    if "fraction" not in tthg:
        raise ValueError("bundle.inputs['approx']['rt_params']['common']['TTHG_params'] must contain a 'fraction' field")
    if "constant_back" not in tthg:
        raise ValueError("bundle.inputs['approx']['rt_params']['common']['TTHG_params'] must contain a 'constant_back' field")
    if "constant_forward" not in tthg:
        raise ValueError("bundle.inputs['approx']['rt_params']['common']['TTHG_params'] must contain a 'constant_forward' field")

    if "stream" not in common:
        raise ValueError("bundle.inputs['approx']['rt_params']['common'] must contain a 'stream' field")
    if "delta_eddington" not in common:
        raise ValueError("bundle.inputs['approx']['rt_params']['common'] must contain a 'delta_eddington' field")
    if "raman" not in common:
        raise ValueError("bundle.inputs['approx']['rt_params']['common'] must contain a 'raman' field")

    if "single_phase" not in toon:
        raise ValueError("bundle.inputs['approx']['rt_params']['toon'] must contain a 'single_phase' field")
    if "multi_phase" not in toon:
        raise ValueError("bundle.inputs['approx']['rt_params']['toon'] must contain a 'multi_phase' field")
    if "toon_coefficients" not in toon:
        raise ValueError("bundle.inputs['approx']['rt_params']['toon'] must contain a 'toon_coefficients' field")

    fraction = tthg["fraction"]
    if not isinstance(fraction, (list, tuple, np.ndarray)) or len(fraction) != 3:
        raise ValueError(
            "bundle.inputs['approx']['rt_params']['common']['TTHG_params']['fraction'] must have length 3"
        )

    return driver.RadtranSettings(
        single_phase=int(toon["single_phase"]),
        multi_phase=int(toon["multi_phase"]),
        frac_a=float(fraction[0]),
        frac_b=float(fraction[1]),
        frac_c=float(fraction[2]),
        constant_back=float(tthg["constant_back"]),
        constant_forward=float(tthg["constant_forward"]),
        toon_coefficients=int(toon["toon_coefficients"]),
        stream=float(common["stream"]),
        delta_eddington=bool(common["delta_eddington"]),
        raman=int(common["raman"]),
    )


def bundle_to_phase(bundle):
    """Convert a legacy PICASO bundle into experimental RadtranPhase."""
    if "phase_angle" not in bundle.inputs:
        raise ValueError("bundle.inputs must contain a 'phase_angle' field")
    if "disco" not in bundle.inputs:
        raise ValueError("bundle.inputs must contain a 'disco' section")

    phase_angle = bundle.inputs["phase_angle"]
    disco = bundle.inputs["disco"]
    if not np.isscalar(phase_angle):
        raise ValueError(f"bundle.inputs['phase_angle'] must be scalar, got {type(phase_angle)!r}")
    if not isinstance(disco, dict):
        raise ValueError(f"bundle.inputs['disco'] must be a dict, got {type(disco)!r}")
    if "num_gangle" not in disco:
        raise ValueError("bundle.inputs['disco'] must contain a 'num_gangle' field")
    if "num_tangle" not in disco:
        raise ValueError("bundle.inputs['disco'] must contain a 'num_tangle' field")

    return driver.RadtranPhase(
        numg=int(disco["num_gangle"]),
        numt=int(disco["num_tangle"]),
        phase_angle=float(phase_angle),
    )

def build_returns(opacityclass: ExperimentalRT, calculation: str):

    rad = opacityclass.rad
    result = rad._get_result(calculation)
    returns = {"wavenumber": opacityclass.wno}
    wno = opacityclass.wno
    flip = slice(None, None, -1)

    if calculation == "reflected":
        albedo = result.albedo[flip]
        returns["albedo"] = albedo

        if rad.star is not None and rad.star.spectrum is not None:
            star_flux = rad.star.spectrum[flip]
            returns["bond_albedo"] = (
                np.trapezoid(
                    x=1.0 / wno,
                    y=albedo * star_flux,
                )
                / np.trapezoid(x=1.0 / wno, y=star_flux)
            )
            if np.isfinite(rad.atmosphere.semimajor) and np.isfinite(rad.atmosphere.radius):
                returns["fpfs_reflected"] = result.fpfs[flip]
            else:
                returns["fpfs_reflected"] = []
                if not np.isfinite(rad.atmosphere.semimajor):
                    returns["fpfs_reflected"].append(
                        "Semi-major axis not supplied. If you want fpfs, add it to `star` function."
                    )
                if not np.isfinite(rad.atmosphere.radius):
                    returns["fpfs_reflected"].append(
                        "Planet Radius not supplied. If you want fpfs, add it to `gravity` function with a mass."
                    )
        else:
            returns["bond_albedo"] = []
            returns["bond_albedo"].append(
                "Stellar spectrum not supplied. If you want bond albedo, add a star."
            )
            returns["fpfs_reflected"] = result.fpfs

    if calculation == "thermal":
        thermal = result.thermal[flip]
        returns["thermal"] = thermal
        returns["thermal_unit"] = "erg/s/(cm^2)/(cm)"
        returns["effective_temperature"] = (
            np.trapezoid(x=1.0 / wno, y=thermal) / 5.67e-5
        ) ** 0.25

        if rad.star is None or rad.star.spectrum is None:
            returns["fpfs_thermal"] = ["No star mode for Brown Dwarfs was used"]
        elif np.isfinite(rad.atmosphere.radius) and np.isfinite(rad.star.radius):
            returns["fpfs_thermal"] = result.fpfs[flip]
        else:
            returns["fpfs_thermal"] = []
            if not np.isfinite(rad.atmosphere.radius):
                returns["fpfs_thermal"].append(
                    "Planet Radius not supplied. If you want fpfs, add it to `gravity` function with radius."
                )
            if not np.isfinite(rad.star.radius):
                returns["fpfs_thermal"].append(
                    "Stellar Radius not supplied. If you want fpfs, add it to `stellar` function."
                )

    if calculation == "transmission":
        returns["transit_depth"] = result.rprs2[flip]

    return returns

def picaso(
    bundle, 
    opacityclass: ExperimentalRT, 
    dimension: str, 
    calculation: str, 
    full_output: bool, 
    plot_opacity: bool, 
    as_dict: bool
):

    # Initial input check
    if not as_dict:
        raise ValueError("as_dict must be True")
    if plot_opacity:
        raise ValueError("plot_opacity is not supported by the experimental interface")
    if full_output:
        raise ValueError("full_output is not supported by the experimental interface")
    if dimension != '1d':
        raise ValueError(f"dimension must be '1d', got {dimension!r}")
    if not isinstance(opacityclass, ExperimentalRT):
        raise ValueError(f"opacityclass must be an ExperimentalRT, got {type(opacityclass)!r}")
    
    # Convert inputs
    atmosphere = bundle_to_atmosphere(bundle)
    planet = bundle_to_planet(bundle)
    clouds = bundle_to_clouds(bundle, opacityclass, atmosphere)
    surface = bundle_to_surface(bundle, opacityclass)
    star = bundle_to_star(bundle)
    settings = bundle_to_settings(bundle)
    phase = bundle_to_phase(bundle)

    # Settings
    opacityclass.rad.settings = settings
    opacityclass.rad.phase = phase

    # Calculation
    _ = opacityclass.rad.spectrum(
        atmosphere,
        planet,
        clouds,
        surface,
        star,
        calculation,
        opacityclass.nwavelengths_per_chunk,
    )

    # Return
    return build_returns(opacityclass, calculation)





    

    
