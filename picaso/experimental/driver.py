from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from collections import OrderedDict

import h5py
import numpy as np
import numba as nb
from numba import typed

from ..elements import ELEMENTS
from ..disco import compute_disco, get_angles_1d, get_angles_3d
from .fluxes import (
    ReflectedResult,
    ReflectedSolver,
    ThermalResult,
    ThermalSolver,
    TransmissionResult,
    get_transit_1d,
    get_reflected_1d,
    get_thermal_1d,
)
from .rayleigh import (
    compute_sigma as compute_rayleigh_sigma,
    RAYLEIGH_MOLECULES,
)
from .raman import compute_raman
from .opacityfiles import _bin_centers_from_edges, _decode_hdf5_string

# Comment below helps ignore type checking false-positives.
# pyright: reportInvalidTypeForm=false

# cgs constants for the compiled hydrostatic setup
KB_CGS = 1.380649e-16
AMU_CGS = 1.66053906660e-24
G_CGS = 6.67430e-8
M_EARTH_CGS = 5.972167867791379e27
R_EARTH_CGS = 6.3781e8
R_SUN_CGS = 6.957e10
CIA_AMAGAT_TO_MOLECULE_CM = 1.385277e-39
# Convert number column density to molar column density for legacy Rayleigh parity.
AVOGADRO = 6.02214076e23
LOG10 = np.log(10.0)


def separate_molecule_name(molecule_name):
    """Separate a molecule string into element/isotope tokens."""
    return re.findall(r'[A-Z][a-z]?\d*|\d+', molecule_name)


def separate_string_number(string):
    """Separate a token into alphabetic and numeric parts."""
    return re.findall(r'[A-Za-z]+|\d+', string)


def get_weights(molecule):
    """Return molecular weights for one species or a list of species.

    This matches the legacy `atmsetup.py:get_weights` behavior, including
    isotope parsing and defaulting to the most abundant isotope when a species
    is not explicitly isotopically labeled.
    """
    separator = '_'
    if isinstance(molecule, str):
        molecule = [molecule]
    elif not isinstance(molecule, list):
        molecule = list(molecule)

    weights = np.empty(len(molecule), dtype=np.float64)
    for i, species in enumerate(molecule):
        totmass = 0.0
        if separator in species:
            elements = [separate_molecule_name(j) for j in species.split(separator)]
        else:
            elements = separate_molecule_name(species)

        for iele in elements:
            if isinstance(iele, list):
                if len(iele) == 1:
                    iele = iele[0]
                    iso_num = 'main'
                else:
                    iso_num = int(iele[0])
                    iele = iele[1]
            else:
                iso_num = 'main'

            sep = separate_string_number(iele)
            if len(sep) == 1:
                el, num = sep[0], 1
            else:
                el, num = sep

            if iso_num == 'main':
                main_iso = np.argmax([ELEMENTS[el].isotopes[j].abundance for j in ELEMENTS[el].isotopes.keys()])
                iso_num = list(ELEMENTS[el].isotopes.keys())[main_iso]
            totmass += ELEMENTS[el].isotopes[iso_num].mass * float(num)

        weights[i] = totmass

    return weights

@nb.experimental.jitclass
class Atmosphere_:

    nspecies: nb.int64
    nlevels: nb.int64
    nlayers: nb.int64

    species_names: nb.types.ListType(nb.types.unicode_type)
    species_mu: nb.float64[:]
    level_pressures: nb.float64[:]
    level_temperatures: nb.float64[:]
    level_mixing_ratios: nb.float64[:,:]
    layer_pressures: nb.float64[:]
    layer_temperatures: nb.float64[:]
    layer_mixing_ratios: nb.float64[:,:]
    reference_pressure: nb.float64

    def __init__(self, species_names, species_mu, pressures, temperatures, mixing_ratios, reference_pressure):
        
        # Check dimensions
        nspecies, nlevels = mixing_ratios.shape
        if nlevels <= 1:
            raise ValueError("mixing_ratios must have at least two levels")
        if len(species_names) != nspecies:
            raise ValueError(
                "species_names length must match mixing_ratios.shape[0] "
                f"({len(species_names)} != {nspecies})"
            )
        if species_mu.shape[0] != nspecies:
            raise ValueError(
                "species_mu length must match mixing_ratios.shape[0] "
                f"({species_mu.shape[0]} != {nspecies})"
            )
        if pressures.shape[0] != nlevels:
            raise ValueError(
                "pressures length must match mixing_ratios.shape[1] "
                f"({pressures.shape[0]} != {nlevels})"
            )
        if temperatures.shape[0] != nlevels:
            raise ValueError(
                "temperatures length must match mixing_ratios.shape[1] "
                f"({temperatures.shape[0]} != {nlevels})"
            )
        
        # Check for physical values.
        if not np.all(np.isfinite(pressures)):
            raise ValueError("pressures must contain only finite values")
        if np.any(pressures <= 0.0):
            raise ValueError(
                f"pressures must be strictly positive, got minimum {float(np.min(pressures))}"
            )
        if not np.all(np.isfinite(temperatures)):
            raise ValueError("temperatures must contain only finite values")
        if np.any(temperatures <= 0.0):
            raise ValueError(
                f"temperatures must be strictly positive, got minimum {float(np.min(temperatures))}"
            )
        if not np.all(np.isfinite(mixing_ratios)):
            raise ValueError("mixing_ratios must contain only finite values")
        if np.any(mixing_ratios < 0.0):
            raise ValueError("mixing_ratios must be nonnegative volume mixing ratios")
        if not np.isfinite(reference_pressure):
            raise ValueError("reference_pressure must be a finite value")
        if reference_pressure < 0.0:
            raise ValueError("reference_pressure must be nonnegative")
        
        # Check that pressures are increasing
        for i in range(len(pressures) - 1):
            if pressures[i+1] <= pressures[i]:
                raise ValueError(
                    "pressures must be strictly increasing with layer index "
                    f"({pressures[i+1]} <= {pressures[i]} at indices {i+1} and {i})"
                )

        # Normalize mixing ratios in place so each level sums to one.
        # Do this before deriving layer quantities to keep the layer state consistent.
        for i in range(nlevels):
            level_sum = 0.0
            for j in range(nspecies):
                level_sum += mixing_ratios[j, i]
            if level_sum <= 0.0:
                raise ValueError(
                    f"mixing_ratios at level {i} must sum to a positive value, got {level_sum}"
                )
            for j in range(nspecies):
                mixing_ratios[j, i] /= level_sum

        # Check that reference pressure is in pressures
        if reference_pressure < pressures[0] or reference_pressure > pressures[-1]:
            raise ValueError(
                "reference_pressure must lie within the pressure grid "
                f"[{pressures[0]}, {pressures[-1]}], got {reference_pressure}"
            )

        # Set attributes
        self.nspecies = nspecies
        self.nlevels = nlevels
        self.nlayers = nlevels - 1
        self.species_names = species_names
        self.species_mu = species_mu
        self.level_pressures = pressures
        self.level_temperatures = temperatures
        self.level_mixing_ratios = mixing_ratios
        self.layer_pressures = np.sqrt(pressures[:-1] * pressures[1:])
        self.layer_temperatures = 0.5 * (temperatures[:-1] + temperatures[1:])
        self.layer_mixing_ratios = 0.5 * (mixing_ratios[:, :-1] + mixing_ratios[:, 1:])
        self.reference_pressure = reference_pressure

class Atmosphere:

    def __init__(self, species_names, pressures, temperatures, mixing_ratios, species_mu=None, reference_pressure=1.0):
        # Check inputs.
        if not isinstance(species_names, list):
            raise TypeError(f"species_names must be a list, got {type(species_names)!r}")

        if species_mu is None:
            species_mu = get_weights(species_names)

        # Initialize most terms
        atm = Atmosphere_(
            typed.List(species_names),
            species_mu,
            pressures,
            temperatures,
            mixing_ratios,
            reference_pressure,
        )

        self._atm = atm


@nb.experimental.jitclass
class Clouds:
    
    nwavelengths: nb.int64
    nlayers: nb.int64
    wavelength: nb.float64[:]
    pressure: nb.float64[:]
    opd: nb.float64[:,:]
    w0: nb.float64[:,:]
    g0: nb.float64[:,:]

    do_holes: nb.bool
    fthin_cld: nb.float64
    fhole: nb.float64

    interpolate: nb.bool

    def __init__(self, wavelength, pressure, opd, w0, g0, do_holes=False, fthin_cld=1.0, fhole=0.0, interpolate=True):
        
        # Check shape
        nwavelengths = len(wavelength)
        nlayers = len(pressure)
        if opd.shape[0] != nwavelengths or opd.shape[1] != nlayers:
            raise ValueError(
                "opd shape must be (nwavelengths, nlayers), got "
                f"{opd.shape} for nwavelengths={nwavelengths}, nlayers={nlayers}"
            )
        if w0.shape[0] != nwavelengths or w0.shape[1] != nlayers:
            raise ValueError(
                "w0 shape must be (nwavelengths, nlayers), got "
                f"{w0.shape} for nwavelengths={nwavelengths}, nlayers={nlayers}"
            )
        if g0.shape[0] != nwavelengths or g0.shape[1] != nlayers:
            raise ValueError(
                "g0 shape must be (nwavelengths, nlayers), got "
                f"{g0.shape} for nwavelengths={nwavelengths}, nlayers={nlayers}"
            )

        # Check physical
        for i in range(nwavelengths):
            for j in range(nlayers):
                if opd[i,j] < 0.0:
                    raise ValueError(f"opd must be nonnegative, got opd[{i},{j}]={opd[i, j]}")
                if w0[i,j] < 0.0 or w0[i,j] > 1.0:
                    raise ValueError(f"w0 must lie in [0, 1], got w0[{i},{j}]={w0[i, j]}")
                if g0[i,j] < -1.0 or g0[i,j] > 1.0:
                    raise ValueError(f"g0 must lie in [-1, 1], got g0[{i},{j}]={g0[i, j]}")
        if fthin_cld < 0.0 or fthin_cld > 1.0:
            raise ValueError(f"fthin_cld must lie in [0, 1], got {fthin_cld}")
        if fhole < 0.0 or fhole > 1.0:
            raise ValueError(f"fhole must lie in [0, 1], got {fhole}")
        
        # Assume wavelength and pressure are OK.
        # They are checked later.

        # Set values
        self.nwavelengths = nwavelengths
        self.nlayers = nlayers
        self.wavelength = wavelength
        self.pressure = pressure
        self.opd = opd
        self.w0 = w0
        self.g0 = g0

        self.do_holes = do_holes
        self.fthin_cld = fthin_cld
        self.fhole = fhole

        self.interpolate = interpolate


@dataclass(frozen=True, slots=True)
class Surface:
    """Surface boundary condition for reflected and thermal solves."""

    hard_surface: bool = False
    reflectance: np.ndarray | float = 0.0
    wavelength: np.ndarray | None = None

    def __post_init__(self):

        if not isinstance(self.hard_surface, bool):
            raise TypeError(f"hard_surface must be a bool, got {type(self.hard_surface)!r}")

        if np.isscalar(self.reflectance):
            if not np.isfinite(self.reflectance):
                raise ValueError("reflectance must be finite")
            if self.reflectance < 0.0 or self.reflectance > 1.0:
                raise ValueError(f"reflectance must lie in [0, 1], got {self.reflectance}")
            return

        if not isinstance(self.reflectance, np.ndarray):
            raise TypeError(
                f"reflectance must be a scalar or 1D numpy.ndarray, got {type(self.reflectance)!r}"
            )
        if not isinstance(self.wavelength, np.ndarray):
            raise TypeError(
                f"wavelength must be a scalar or 1D numpy.ndarray, got {type(self.wavelength)!r}"
            )
        if self.reflectance.ndim != 1:
            raise ValueError(f"reflectance must be scalar or 1D, got shape {self.reflectance.shape}")
        if self.wavelength.ndim != 1:
            raise ValueError(f"wavelength must be scalar or 1D, got shape {self.wavelength.shape}")
        if self.reflectance.shape[0] != self.wavelength.shape[0]:
            raise ValueError(
                "reflectance and wavelength must have the same length, "
                f"got {self.reflectance.shape[0]} and {self.wavelength.shape[0]}"
            )
        _check_reflectance(self.reflectance)


@nb.njit
def _check_reflectance(reflectance):
    for i in range(reflectance.shape[0]):
        value = reflectance[i]
        if not np.isfinite(value):
            raise ValueError(f"reflectance must contain only finite values, got {value} at index {i}")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"reflectance must lie in [0, 1], got {value} at index {i}")

@dataclass(frozen=True, slots=True)
class Star:

    radius: float
    wavelength: np.ndarray | None = None
    spectrum: np.ndarray | None = None

    def __post_init__(self):

        if self.radius < 0.0:
            raise ValueError(f"radius must be nonnegative, got {self.radius}")
        object.__setattr__(self, "radius", self.radius * R_SUN_CGS)

        if self.wavelength is None:
            if self.spectrum is not None:
                raise ValueError("wavelength and spectrum must either both be provided or both be None")
            return
        if self.spectrum is None:
            if self.wavelength is not None:
                raise ValueError("wavelength and spectrum must either both be provided or both be None")
            return
        
        if not isinstance(self.wavelength, np.ndarray):
            raise ValueError(f"wavelength must be a numpy.ndarray or None, got {type(self.wavelength)!r}")
        if not isinstance(self.spectrum, np.ndarray):
            raise ValueError(f"spectrum must be a numpy.ndarray or None, got {type(self.spectrum)!r}")
        _check_spectrum(self.spectrum)
        
@nb.njit
def _check_spectrum(spectrum):
    for i in range(spectrum.shape[0]):
        value = spectrum[i]
        if not np.isfinite(value):
            raise ValueError(f"spectrum must contain only finite values, got {value} at index {i}")

@nb.experimental.jitclass
class Planet:

    radius: nb.float64
    mass: nb.float64
    semimajor: nb.float64

    def __init__(self, radius, mass, semimajor=np.nan):

        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"radius must be finite and > 0, got {radius}")
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"mass must be finite and > 0, got {mass}")
        if not np.isnan(semimajor) and (not np.isfinite(semimajor) or semimajor <= 0.0):
            raise ValueError(f"semimajor must be finite and > 0, got {semimajor}")

        self.radius = radius
        self.mass = mass
        self.semimajor = semimajor
        
@dataclass(frozen=True, slots=True)
class RadtranSettings:
    single_phase: int = 3
    multi_phase: int = 0
    frac_a: float = 1.0
    frac_b: float = -1.0
    frac_c: float = 2.0
    constant_back: float = -0.5
    constant_forward: float = 1.0
    toon_coefficients: int = 0
    stream: float = 2.0
    delta_eddington: bool = True
    raman: int = 1

    def __post_init__(self):
        def _check_int(name, value, allowed):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            if int(value) not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value}")

        def _check_finite(name, value):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

        _check_int("single_phase", self.single_phase, {0, 1, 2, 3})
        _check_int("multi_phase", self.multi_phase, {0, 1})
        _check_int("toon_coefficients", self.toon_coefficients, {0, 1})
        _check_int("raman", self.raman, {1, 2})

        for name, value in (
            ("frac_a", self.frac_a),
            ("frac_b", self.frac_b),
            ("frac_c", self.frac_c),
            ("constant_back", self.constant_back),
            ("constant_forward", self.constant_forward),
            ("stream", self.stream),
        ):
            _check_finite(name, value)

        if self.stream <= 0.0:
            raise ValueError(f"stream must be > 0, got {self.stream}")
        if not isinstance(self.delta_eddington, bool):
            raise ValueError(f"delta_eddington must be a bool, got {self.delta_eddington!r}")

@dataclass
class RadtranPhase:
    numg: int = 10
    numt: int = 1
    phase_angle: float = 0.0
    effective_numg: int = 10
    effective_numt: int = 1

    gangle: np.ndarray = None
    gweight: np.ndarray = None
    tangle: np.ndarray = None
    tweight: np.ndarray = None
    ubar0: np.ndarray = None
    ubar1: np.ndarray = None
    cos_theta: float = np.nan
    latitude: np.ndarray = None
    longitude: np.ndarray = None

    def __post_init__(self):
        self._validate()
        self._refresh_geometry()

    def _validate(self):
        if self.numg <= 0:
            raise ValueError(f"numg must be positive, got {self.numg}")
        if self.numt <= 0:
            raise ValueError(f"numt must be positive, got {self.numt}")
        if self.phase_angle < 0.0 or self.phase_angle > 2.0 * np.pi:
            raise ValueError(
                f"phase_angle must be between 0 and 2*pi radians, got {self.phase_angle}"
            )
        if self.numt == 1 and self.numg == 1:
            raise ValueError("numg cannot be 1 when using the 1D symmetry geometry")

    def _refresh_geometry(self):
        if self.numt == 1:
            if self.phase_angle != 0.0:
                raise ValueError(
                    "1D symmetry geometry only supports phase_angle == 0.0; "
                    f"got {self.phase_angle}"
                )
            self.effective_numg = min(max(int(self.numg / 2), 5), 8)
            self.effective_numt = 1
            self.gangle, self.gweight, self.tangle, self.tweight = get_angles_1d(self.effective_numg)
        else:
            self.effective_numg = int(self.numg)
            self.effective_numt = int(self.numt)
            self.gangle, self.gweight, self.tangle, self.tweight = get_angles_3d(
                self.effective_numg, self.effective_numt
            )
        self.ubar0, self.ubar1, self.cos_theta, self.latitude, self.longitude = compute_disco(
            self.effective_numg, self.effective_numt, self.gangle, self.tangle, self.phase_angle
        )

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"RadtranSettings has no attribute {key!r}")
            setattr(self, key, value)
        self._validate()
        self._refresh_geometry()
        return self


@nb.experimental.jitclass
class RadtranOpacitiesWorkspace:
    nlayers: nb.int64
    npressure: nb.int64
    ntemperature: nb.int64
    ncontinuum_temperature: nb.int64
    nwavelengths: nb.int64
    nwavelengths_per_chunk: nb.int64
    molecular_npairs: nb.int64
    continuum_nrows: nb.int64
    molecular_pressure_ind0: nb.int64[:]
    molecular_pressure_ind1: nb.int64[:]
    molecular_pressure_weight: nb.float64[:]
    molecular_temperature_ind0: nb.int64[:]
    molecular_temperature_ind1: nb.int64[:]
    molecular_temperature_weight: nb.float64[:]
    continuum_temperature_ind0: nb.int64[:]
    continuum_temperature_ind1: nb.int64[:]
    continuum_temperature_weight: nb.float64[:]
    continuum_scale: nb.float64[:]
    molecular_pair_map: nb.int64[:,:]
    molecular_pair_pindex: nb.int64[:]
    molecular_pair_tindex: nb.int64[:]
    continuum_temperature_map: nb.int64[:]
    continuum_temperature_load_idx: nb.int64[:]
    molecular_block: nb.float64[:,:]
    continuum_block: nb.float64[:,:]
    raw_u16: nb.uint16[:]
    raw_f32: nb.float32[:]
    raw_full_u16: nb.uint16[:]
    raw_full_f32: nb.float32[:]
    rayleigh_sigma: nb.float64[:]
    cloud_wavelength_ind0: nb.int64[:]
    cloud_wavelength_ind1: nb.int64[:]
    cloud_wavelength_weight: nb.float64[:]

    def __init__(self):
        self._allocate(0, 0, 0, 0, 0, 0)

    def _allocate(
        self,
        nlayers,
        npressure,
        ntemperature,
        ncontinuum_temperature,
        nwavelengths,
        nwavelengths_per_chunk,
    ):
        self.nlayers = nlayers
        self.npressure = npressure
        self.ntemperature = ntemperature
        self.ncontinuum_temperature = ncontinuum_temperature
        self.nwavelengths = nwavelengths
        self.nwavelengths_per_chunk = nwavelengths_per_chunk
        self.molecular_npairs = 0
        self.continuum_nrows = 0
        self.molecular_pressure_ind0 = np.empty(nlayers, dtype=np.int64)
        self.molecular_pressure_ind1 = np.empty(nlayers, dtype=np.int64)
        self.molecular_pressure_weight = np.empty(nlayers, dtype=np.float64)
        self.molecular_temperature_ind0 = np.empty(nlayers, dtype=np.int64)
        self.molecular_temperature_ind1 = np.empty(nlayers, dtype=np.int64)
        self.molecular_temperature_weight = np.empty(nlayers, dtype=np.float64)
        self.continuum_temperature_ind0 = np.empty(nlayers, dtype=np.int64)
        self.continuum_temperature_ind1 = np.empty(nlayers, dtype=np.int64)
        self.continuum_temperature_weight = np.empty(nlayers, dtype=np.float64)
        self.continuum_scale = np.empty(nlayers, dtype=np.float64)
        self.molecular_pair_map = np.full((npressure, ntemperature), -1, dtype=np.int64)
        self.molecular_pair_pindex = np.empty(npressure * ntemperature, dtype=np.int64)
        self.molecular_pair_tindex = np.empty(npressure * ntemperature, dtype=np.int64)
        self.continuum_temperature_map = np.full(ncontinuum_temperature, -1, dtype=np.int64)
        self.continuum_temperature_load_idx = np.empty(ncontinuum_temperature, dtype=np.int64)
        self.molecular_block = np.empty((npressure * ntemperature, nwavelengths_per_chunk), dtype=np.float64)
        self.continuum_block = np.empty((ncontinuum_temperature, nwavelengths_per_chunk), dtype=np.float64)
        self.raw_u16 = np.empty(nwavelengths_per_chunk, dtype=np.uint16)
        self.raw_f32 = np.empty(nwavelengths_per_chunk, dtype=np.float32)
        self.raw_full_u16 = np.empty(nwavelengths, dtype=np.uint16)
        self.raw_full_f32 = np.empty(nwavelengths, dtype=np.float32)
        self.rayleigh_sigma = np.empty(nwavelengths_per_chunk, dtype=np.float64)
        self.cloud_wavelength_ind0 = np.empty(nwavelengths, dtype=np.int64)
        self.cloud_wavelength_ind1 = np.empty(nwavelengths, dtype=np.int64)
        self.cloud_wavelength_weight = np.empty(nwavelengths, dtype=np.float64)

    def _ensure(
        self,
        nlayers,
        npressure,
        ntemperature,
        ncontinuum_temperature,
        nwavelengths,
        nwavelengths_per_chunk,
    ):
        if (
            nlayers != self.nlayers
            or npressure != self.npressure
            or ntemperature != self.ntemperature
            or ncontinuum_temperature != self.ncontinuum_temperature
            or nwavelengths != self.nwavelengths
            or nwavelengths_per_chunk != self.nwavelengths_per_chunk
        ):
            self._allocate(
                nlayers,
                npressure,
                ntemperature,
                ncontinuum_temperature,
                nwavelengths,
                nwavelengths_per_chunk,
            )


class RadtranOpacitiesCache:
    def __init__(self, size_limit_bytes, row_length, raw_dtype):
        self.size_limit_bytes = int(size_limit_bytes)
        self.row_length = int(row_length)
        self.raw_dtype = np.dtype(raw_dtype)
        self.row_nbytes = int(self.row_length * self.raw_dtype.itemsize)
        self.capacity = 0 if self.row_nbytes <= 0 else int(self.size_limit_bytes // self.row_nbytes)
        if self.capacity <= 0:
            raise ValueError(
                "opacity_cache_size_limit is too small to cache even one row; "
                f"need at least {self.row_nbytes} bytes for capacity > 0, got {self.size_limit_bytes}"
            )
        self.entries = OrderedDict()
        self.pool = np.empty((self.capacity, self.row_length), dtype=self.raw_dtype)
        self.free_slots = list(range(self.capacity - 1, -1, -1))

    def get(self, key):
        """Get cached data for `key`, or `None` if the key is absent."""
        slot = self.entries.get(key)
        if slot is None:
            return None
        self.entries.move_to_end(key)
        return self.pool[slot]

    def put(self, key, value):
        """Insert `value` under `key`, updating LRU recency."""
        if key in self.entries:
            self.entries.move_to_end(key)
            return

        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            _, slot = self.entries.popitem(last=False)

        self.pool[slot, :] = value
        self.entries[key] = slot
        self.entries.move_to_end(key)


def _read_hdf5_scalar(group, name, filename, default=None):
    if name in group:
        value = group[name][()]
    elif default is None:
        raise ValueError(f"{filename!r} is missing required metadata {name!r}")
    else:
        return default

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"metadata field {name!r} must be scalar, got shape {value.shape}")
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value

_MISSING = object()

def _read_hdf5_attr_scalar(obj, name, filename, default=_MISSING):
    if name in obj.attrs:
        value = obj.attrs[name]
    elif default is _MISSING:
        raise ValueError(f"{filename!r} is missing required attribute {name!r} on {obj.name!r}")
    else:
        return default

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"attribute {name!r} on {obj.name!r} must be scalar, got shape {value.shape}")
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_ck_log10_opacity_block(raw_block, dataset):
    encoding = _decode_hdf5_string(dataset.attrs["encoding"]).lower()
    raw_block = np.asarray(raw_block)

    if encoding == "log10_uint16":
        y_min = float(dataset.attrs["y_min"])
        y_max = float(dataset.attrs["y_max"])
        if y_max == y_min:
            return np.full(raw_block.shape, y_min, dtype=np.float64)
        scale = (y_max - y_min) / float(np.iinfo(np.uint16).max)
        return y_min + raw_block.astype(np.float64) * scale

    if encoding == "log10_float32":
        return raw_block.astype(np.float64)

    raise ValueError(
        f"unsupported source encoding {encoding!r}; expected 'log10_uint16' or 'log10_float32'"
    )


def _infer_bin_edges_from_centers(wavelength):
    if not isinstance(wavelength, np.ndarray):
        wavelength = np.asarray(wavelength, dtype=np.float64)
    if wavelength.ndim != 1:
        raise ValueError(f"wavelength must be 1D to infer bin edges, got shape {wavelength.shape}")
    if wavelength.size < 2:
        raise ValueError("at least two wavelength centers are required to infer bin edges")
    if not np.all(np.isfinite(wavelength)):
        raise ValueError("wavelength must contain only finite values")
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength must be strictly increasing to infer bin edges")

    bin_edges = np.empty((wavelength.size, 2), dtype=np.float64)
    midpoints = 0.5 * (wavelength[1:] + wavelength[:-1])
    bin_edges[1:, 0] = midpoints
    bin_edges[:-1, 1] = midpoints
    bin_edges[0, 0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    bin_edges[-1, 1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    return bin_edges


def _make_radtran_opacities(opacity_filename, wavelength_range, opacity_cache_size_limit):
    if not isinstance(opacity_filename, (str, Path)):
        raise TypeError(
            f"opacity_filename must be a string path or Path, got {type(opacity_filename)!r}"
        )

    opacity_filename = str(opacity_filename)
    if not h5py.is_hdf5(opacity_filename):
        raise ValueError(f"{opacity_filename!r} is not a valid HDF5 file")

    with h5py.File(opacity_filename, "r") as file:
        opacity_type = _read_hdf5_scalar(file["header"], "opacity_type", opacity_filename)

    opacity_type = str(opacity_type).lower()
    if opacity_type == "correlated-k":
        return RadtranOpacitiesCK(opacity_filename, wavelength_range, opacity_cache_size_limit)
    if opacity_type == "molecular+continuum":
        return RadtranOpacities(opacity_filename, wavelength_range, opacity_cache_size_limit)

    raise ValueError(
        f"{opacity_filename!r} has unsupported opacity_type {opacity_type!r}; "
        "expected 'molecular+continuum' or 'correlated-k'"
    )



class RadtranOpacitiesCK:

    def __init__(self, opacity_filename, wavelength_range, opacity_cache_size_limit):
        if not isinstance(opacity_filename, (str, Path)):
            raise TypeError(
                f"opacity_filename must be a string path or Path, got {type(opacity_filename)!r}"
            )

        self.opacity_filename = str(opacity_filename)
        if not h5py.is_hdf5(self.opacity_filename):
            raise ValueError(f"{self.opacity_filename!r} is not a valid HDF5 file")

        self.opacity_cache_size_limit = opacity_cache_size_limit

        with h5py.File(self.opacity_filename, "r") as file:
            header = file["header"]
            molecular_group = file["molecular"]
            continuum_group = file["continuum"]

            opacity_type = _read_hdf5_scalar(header, "opacity_type", self.opacity_filename)
            if str(opacity_type).lower() != "correlated-k":
                raise ValueError(
                    f"{self.opacity_filename!r} has opacity_type {opacity_type!r}; "
                    "expected 'correlated-k'"
                )

            self.storage_format = str(_read_hdf5_scalar(header, "storage_format", self.opacity_filename))
            if self.storage_format not in {"log10_uint16", "log10_float32"}:
                raise ValueError(
                    f"unsupported storage_format {self.storage_format!r}; "
                    "expected 'log10_uint16' or 'log10_float32'"
                )

            pressure = np.asarray(header["pressure"][:], dtype=np.float64)
            temperature = np.asarray(header["temperature"][:], dtype=np.float64)
            continuum_temperatures = np.asarray(header["continuum_temperatures"][:], dtype=np.float64)
            wavelength = np.asarray(header["wavelength"][:], dtype=np.float64)
            bin_edges = np.asarray(header["bin_edges"][:], dtype=np.float64)
            g_points = np.asarray(header["g_points"][:], dtype=np.float64)
            g_weights = np.asarray(header["g_weights"][:], dtype=np.float64)

            if wavelength.ndim != 1:
                raise ValueError(f"header/wavelength must be 1D, got shape {wavelength.shape}")
            if bin_edges.shape != (wavelength.size, 2):
                raise ValueError(
                    f"header/bin_edges must have shape ({wavelength.size}, 2), got {bin_edges.shape}"
                )
            if not np.allclose(_bin_centers_from_edges(bin_edges), wavelength):
                raise ValueError("header/bin_edges are not consistent with header/wavelength")
            if g_points.ndim != 1 or g_weights.ndim != 1 or g_points.size != g_weights.size:
                raise ValueError(
                    "header/g_points and header/g_weights must be 1D arrays with the same length"
                )
            if not np.all(np.isfinite(g_points)) or not np.all(np.isfinite(g_weights)):
                raise ValueError("header/g_points and header/g_weights must contain only finite values")

            molecular_names = [str(name) for name in _decode_hdf5_string(header["molecular_names"][:])]
            continuum_names = [str(name) for name in _decode_hdf5_string(header["continuum_names"][:])]

            if wavelength_range is not None:
                if len(wavelength_range) != 2:
                    raise ValueError(
                        "wavelength_range must be a (min_wavelength, max_wavelength) pair"
                    )
                wmin = float(wavelength_range[0])
                wmax = float(wavelength_range[1])
                if not np.isfinite(wmin) or not np.isfinite(wmax):
                    raise ValueError("wavelength_range must contain finite values")
                if wmin > wmax:
                    raise ValueError(
                        f"wavelength_range minimum must not exceed maximum, got {wmin} > {wmax}"
                    )
                selected = np.flatnonzero((wavelength >= wmin) & (wavelength <= wmax))
                if selected.size == 0:
                    raise ValueError(
                        f"wavelength_range {wavelength_range!r} selects no wavelengths from the file"
                    )
                if selected.size > 1 and np.any(np.diff(selected) != 1):
                    raise ValueError("wavelength_range must select a contiguous block of wavelengths")
            else:
                selected = np.arange(wavelength.size, dtype=np.int64)

            self.wavelength = wavelength[selected]
            self.bin_edges = bin_edges[selected]
            self.nwavelength = int(self.wavelength.size)
            self.ngauss = int(g_points.size)

            self.pressure = pressure
            self.temperature = temperature
            self.continuum_temperatures = continuum_temperatures
            self.npressure = int(pressure.size)
            self.ntemperature = int(temperature.size)
            self.ncontinuum_temperature = int(continuum_temperatures.size)

            self.molecular_names = molecular_names
            self.continuum_names = continuum_names
            self.nmolecular = int(len(molecular_names))
            self.ncontinuum = int(len(continuum_names))
            self.molecular_name_to_index = {name: i for i, name in enumerate(molecular_names)}
            self.continuum_name_to_index = {name: i for i, name in enumerate(continuum_names)}

            self.g_points = g_points
            self.g_weights = g_weights

            self.molecular_unit = str(_read_hdf5_scalar(header, "molecular_unit", self.opacity_filename))
            self.continuum_unit = str(_read_hdf5_scalar(header, "continuum_unit", self.opacity_filename))

            if self.nmolecular:
                self.molecular_tables = np.empty(
                    (self.nmolecular, self.npressure, self.ntemperature, self.nwavelength, self.ngauss),
                    dtype=np.float64,
                )
            else:
                self.molecular_tables = np.empty((0, self.npressure, self.ntemperature, self.nwavelength, self.ngauss), dtype=np.float64)

            self.continuum_tables = np.empty(
                (self.ncontinuum, self.ncontinuum_temperature, self.nwavelength),
                dtype=np.float64,
            )

            self.continuum_types = []
            self.continuum_primary_species = []
            self.continuum_secondary_species = []
            self.continuum_opacity_units = []

            for i, name in enumerate(self.molecular_names):
                dataset = molecular_group[name]
                if dataset.ndim != 4:
                    raise ValueError(f"molecular dataset {name!r} must have rank 4, got shape {dataset.shape}")
                if dataset.shape != (self.npressure, self.ntemperature, wavelength.size, self.ngauss):
                    raise ValueError(
                        f"molecular dataset {name!r} has shape {dataset.shape}, "
                        f"expected {(self.npressure, self.ntemperature, wavelength.size, self.ngauss)}"
                    )
                raw = dataset[:, :, selected, :]
                self.molecular_tables[i] = np.asarray(
                    _decode_ck_log10_opacity_block(raw, dataset),
                    dtype=np.float64,
                )

            for i, name in enumerate(self.continuum_names):
                dataset = continuum_group[name]
                if dataset.ndim != 2:
                    raise ValueError(f"continuum dataset {name!r} must have rank 2, got shape {dataset.shape}")
                if dataset.shape != (self.ncontinuum_temperature, wavelength.size):
                    raise ValueError(
                        f"continuum dataset {name!r} has shape {dataset.shape}, "
                        f"expected {(self.ncontinuum_temperature, wavelength.size)}"
                    )
                raw = dataset[:, selected]
                self.continuum_tables[i] = np.asarray(
                    _decode_ck_log10_opacity_block(raw, dataset),
                    dtype=np.float64,
                )

                continuum_type = str(_read_hdf5_attr_scalar(dataset, "continuum_type", self.opacity_filename)).lower()
                primary_species = str(_read_hdf5_attr_scalar(dataset, "primary_species", self.opacity_filename))
                secondary_species = _read_hdf5_attr_scalar(
                    dataset,
                    "secondary_species",
                    self.opacity_filename,
                    default=None,
                )
                opacity_unit = str(_read_hdf5_attr_scalar(dataset, "opacity_unit", self.opacity_filename))

                if continuum_type not in {"cia", "cross_section"}:
                    raise ValueError(
                        f"continuum dataset {name!r} has unsupported continuum_type {continuum_type!r}; "
                        "expected 'cia' or 'cross_section'"
                    )

                self.continuum_types.append(continuum_type)
                self.continuum_primary_species.append(primary_species)
                self.continuum_secondary_species.append(None if secondary_species is None else str(secondary_species))
                self.continuum_opacity_units.append(opacity_unit)

        if self.nmolecular and self.molecular_unit != "cm2/molecule":
            raise ValueError(
                f"unsupported molecular_unit {self.molecular_unit!r}; expected 'cm2/molecule'"
            )
        if self.ncontinuum and self.continuum_unit == "":
            raise ValueError("continuum_unit must not be empty when continuum data are present")

        self.opacity_type = "correlated-k"

        self.workspace = RadtranOpacitiesWorkspace()

    def _prepare_cloud_interpolation(self, clouds: Clouds):
        if clouds is None or not clouds.interpolate:
            return

        _fill_cloud_interpolation_workspace(
            clouds.wavelength,
            self.wavelength,
            self.workspace.cloud_wavelength_ind0,
            self.workspace.cloud_wavelength_ind1,
            self.workspace.cloud_wavelength_weight,
        )

    def prepare_interpolation(self, atmosphere: RadtranAtmosphere, clouds: Clouds, nwavelengths_per_chunk: int):
        self.workspace._ensure(
            atmosphere.nlayers,
            self.npressure,
            self.ntemperature,
            self.ncontinuum_temperature,
            self.nwavelength,
            nwavelengths_per_chunk,
        )

        _fill_molecular_interpolation_workspace(
            atmosphere,
            self.pressure,
            self.temperature,
            self.workspace,
        )
        _fill_continuum_interpolation_workspace(
            atmosphere,
            self.continuum_temperatures,
            self.workspace,
        )
        self._prepare_cloud_interpolation(clouds)

    def compute_opacity(
        self, 
        atmosphere: RadtranAtmosphere, 
        settings: RadtranSettings, 
        clouds: Clouds, 
        surface: Surface, 
        ind_wv0: int, 
        ind_wv1: int, 
        opacities_result: RadtranOpacitiesResult
    ):  
        # Width of the wavelength chunk
        chunk_width = ind_wv1 - ind_wv0

        # Ensure we have the right allocated workspace.
        opacities_result._ensure(atmosphere.nlayers, self.ngauss, self.workspace.nwavelengths_per_chunk)

        # Gauss weights
        opacities_result.ck_weights[: self.ngauss] = self.g_weights

        # Wavelengths and surface reflectance
        opacities_result.wavelength_um[:chunk_width] = self.wavelength[ind_wv0:ind_wv1]
        if np.isscalar(surface.reflectance):
            opacities_result.surf_reflect[:chunk_width] = float(surface.reflectance)
        else:
            opacities_result.surf_reflect[:chunk_width] = surface.reflectance[ind_wv0:ind_wv1]

        #~~ Molecular opacities ~~#
        active_species = []
        active_tables = []
        for i_species in range(atmosphere.nspecies):
            species_name = str(atmosphere.species_names[i_species])
            table_index = self.molecular_name_to_index.get(species_name)
            if table_index is None:
                continue
            active_species.append(i_species)
            active_tables.append(table_index)
        active_species_indices = np.asarray(active_species, dtype=np.int64)
        active_table_indices = np.asarray(active_tables, dtype=np.int64)

        _compute_ck_molecular_taugas(
            self.molecular_tables,
            active_species_indices,
            active_table_indices,
            atmosphere.layer_mixing_ratios,
            atmosphere.layer_colden,
            atmosphere.layer_mubar,
            self.workspace.molecular_pressure_ind0,
            self.workspace.molecular_pressure_ind1,
            self.workspace.molecular_pressure_weight,
            self.workspace.molecular_temperature_ind0,
            self.workspace.molecular_temperature_ind1,
            self.workspace.molecular_temperature_weight,
            self.workspace.molecular_pair_pindex,
            self.workspace.molecular_pair_tindex,
            self.g_points,
            self.g_weights,
            opacities_result.taugas[:chunk_width, :, :],
        )

        #~~ CIA & continuum ~~#
        atmosphere_name_to_index = {str(name): i for i, name in enumerate(atmosphere.species_names)}
        for i_continuum, continuum_name in enumerate(self.continuum_names):
            continuum_type = self.continuum_types[i_continuum]
            primary_species = self.continuum_primary_species[i_continuum]
            secondary_species = self.continuum_secondary_species[i_continuum]
            if primary_species not in atmosphere_name_to_index:
                continue
            i_primary = atmosphere_name_to_index[primary_species]

            if continuum_type == "cia":
                if secondary_species is None or secondary_species not in atmosphere_name_to_index:
                    continue
                i_secondary = atmosphere_name_to_index[secondary_species]
                _fill_continuum_scale_workspace(atmosphere, i_primary, i_secondary, self.workspace)
                continuum_scale_constant = CIA_AMAGAT_TO_MOLECULE_CM
            elif continuum_type == "cross_section":
                _fill_cross_section_scale_workspace(atmosphere, i_primary, self.workspace)
                continuum_scale_constant = 1.0
            else:
                raise ValueError(
                    f"unsupported continuum_type {continuum_type!r} for continuum {continuum_name!r}"
                )

            _add_ck_continuum_species_taugas(
                self.continuum_tables[i_continuum],
                continuum_scale_constant,
                self.workspace.continuum_scale,
                self.workspace.continuum_temperature_ind0,
                self.workspace.continuum_temperature_ind1,
                self.workspace.continuum_temperature_weight,
                self.workspace.continuum_temperature_load_idx,
                opacities_result.taugas[:chunk_width, :, :],
            )

        #~~ Rayleigh ~~#
        tauray = opacities_result.tauray[:chunk_width, :]
        tauray[:,:] = 0.0
        rayleigh_sigma = self.workspace.rayleigh_sigma[:chunk_width]
        wavelength_chunk = self.wavelength[ind_wv0:ind_wv1]
        for i_species in range(atmosphere.nspecies):
            species_name = str(atmosphere.species_names[i_species])
            if species_name not in RAYLEIGH_MOLECULES:
                continue
            compute_rayleigh_sigma(species_name, wavelength_chunk, rayleigh_sigma)
            _accumulate_rayleigh_tau(rayleigh_sigma, atmosphere.layer_columns[i_species], tauray)

        #~~ Raman ~~#
        compute_raman(
            settings.raman,
            opacities_result.wavelength_um[:chunk_width],
            opacities_result.raman_factor[:chunk_width],
        )

        #~~ Clouds ~~#
        taucld = opacities_result.taucld[:chunk_width, :]
        w0_cld = opacities_result.w0_cld[:chunk_width, :]
        g0_cld = opacities_result.g0_cld[:chunk_width, :]
        if clouds is not None:
            _set_clouds(
                clouds,
                ind_wv0,
                ind_wv1,
                taucld,
                w0_cld,
                g0_cld,
                self.workspace.cloud_wavelength_ind0[ind_wv0:ind_wv1],
                self.workspace.cloud_wavelength_ind1[ind_wv0:ind_wv1],
                self.workspace.cloud_wavelength_weight[ind_wv0:ind_wv1],
            )
        else:
            taucld[:, :] = 0.0
            w0_cld[:, :] = 0.0
            g0_cld[:, :] = 0.0

        # Finish up the calculation
        _finish_compute_opacity(
            opacities_result,
            chunk_width,
            self.ngauss,
            settings.stream,
            settings.delta_eddington,
            fthin_cld=1.0,
        )

    def adjust_opacity_for_clearsky(self, clouds: Clouds, settings: RadtranSettings, ind_wv0: int, ind_wv1: int, opacities_result: RadtranOpacitiesResult):
        if clouds is None or not clouds.do_holes:
            raise ValueError("adjust_opacity_for_clearsky requires a cloud object with do_holes=True")

        chunk_width = ind_wv1 - ind_wv0

        _finish_compute_opacity(
            opacities_result,
            chunk_width,
            self.ngauss,
            settings.stream,
            settings.delta_eddington,
            clouds.fthin_cld,
        )

    

class RadtranOpacities:

    def __init__(self, opacity_filename, wavelength_range, opacity_cache_size_limit):

        if not isinstance(opacity_filename, (str, Path)):
            raise TypeError(
                f"opacity_filename must be a string path or Path, got {type(opacity_filename)!r}"
            )

        self.opacity_filename = str(opacity_filename)
        if not h5py.is_hdf5(self.opacity_filename):
            raise ValueError(f"{self.opacity_filename!r} is not a valid HDF5 file")

        self.file = h5py.File(self.opacity_filename, "r")
        self._header = self.file["header"]
        self._molecular_group = self.file["molecular"]
        self._continuum_group = self.file["continuum"]
        self.storage_format = _read_hdf5_scalar(self._header, "storage_format", self.opacity_filename)
        if self.storage_format not in {"log10_uint16", "log10_float32"}:
            raise ValueError(
                f"unsupported storage_format {self.storage_format!r}; "
                "expected 'log10_uint16' or 'log10_float32'"
            )

        self.pressure = np.asarray(self._header["pressure"][:], dtype=np.float64)
        self.temperature = np.asarray(self._header["temperature"][:], dtype=np.float64)
        wavelength = np.asarray(self._header["wavelength"][:], dtype=np.float64)
        self.continuum_temperatures = np.asarray(self._header["continuum_temperatures"][:], dtype=np.float64)
        self.molecular_names = [str(name) for name in _decode_hdf5_string(self._header["molecular_names"][:])]
        self.continuum_names = [str(name) for name in _decode_hdf5_string(self._header["continuum_names"][:])]

        if wavelength_range is not None:
            if len(wavelength_range) != 2:
                raise ValueError(
                    "wavelength_range must be a (min_wavelength, max_wavelength) pair"
                )
            wmin = float(wavelength_range[0])
            wmax = float(wavelength_range[1])
            if not np.isfinite(wmin) or not np.isfinite(wmax):
                raise ValueError("wavelength_range must contain finite values")
            if wmin > wmax:
                raise ValueError(
                    f"wavelength_range minimum must not exceed maximum, got {wmin} > {wmax}"
                )
            selected = np.flatnonzero((wavelength >= wmin) & (wavelength <= wmax))
            if selected.size == 0:
                raise ValueError(
                    f"wavelength_range {wavelength_range!r} selects no wavelengths from the file"
                )
            if selected.size > 1 and np.any(np.diff(selected) != 1):
                raise ValueError(
                    "wavelength_range must select a contiguous block of wavelengths"
                )
            self.wavelength_source_indices = selected.astype(np.int64)
            self.wavelength = wavelength[self.wavelength_source_indices]
        else:
            self.wavelength_source_indices = np.arange(wavelength.size, dtype=np.int64)
            self.wavelength = wavelength
        self.bin_edges = _infer_bin_edges_from_centers(self.wavelength)

        self.npressure = int(self.pressure.size)
        self.ntemperature = int(self.temperature.size)
        self.nwavelength = int(self.wavelength.size)
        self.ncontinuum_temperature = int(self.continuum_temperatures.size)
        self.nmolecular = int(len(self.molecular_names))
        self.ncontinuum = int(len(self.continuum_names))
        self.molecular_name_to_index = {name: i for i, name in enumerate(self.molecular_names)}
        self.molecular_y_min = np.empty(self.nmolecular, dtype=np.float64)
        self.molecular_y_max = np.empty(self.nmolecular, dtype=np.float64)
        for i, name in enumerate(self.molecular_names):
            dataset = self._molecular_group[name]
            storage_format = self.storage_format
            if storage_format == "log10_uint16":
                if "y_min" not in dataset.attrs or "y_max" not in dataset.attrs:
                    raise ValueError(f"molecular dataset {name!r} is missing required y_min/y_max attrs")
                self.molecular_y_min[i] = float(dataset.attrs["y_min"])
                self.molecular_y_max[i] = float(dataset.attrs["y_max"])
            else:
                self.molecular_y_min[i] = np.nan
                self.molecular_y_max[i] = np.nan

        self.continuum_y_min = np.empty(self.ncontinuum, dtype=np.float64)
        self.continuum_y_max = np.empty(self.ncontinuum, dtype=np.float64)
        self.continuum_types = []
        self.continuum_primary_species = []
        self.continuum_secondary_species = []
        self.continuum_opacity_units = []
        for i, name in enumerate(self.continuum_names):
            dataset = self._continuum_group[name]
            continuum_type = str(_read_hdf5_attr_scalar(dataset, "continuum_type", self.opacity_filename)).lower()
            primary_species = str(_read_hdf5_attr_scalar(dataset, "primary_species", self.opacity_filename))
            secondary_species = _read_hdf5_attr_scalar(
                dataset,
                "secondary_species",
                self.opacity_filename,
                default=None,
            )
            opacity_unit = str(_read_hdf5_attr_scalar(dataset, "opacity_unit", self.opacity_filename))

            if continuum_type not in {"cia", "cross_section"}:
                raise ValueError(
                    f"continuum dataset {name!r} has unsupported continuum_type {continuum_type!r}; "
                    "expected 'cia' or 'cross_section'"
                )
            if continuum_type == "cia":
                if secondary_species is None or str(secondary_species) == "":
                    raise ValueError(f"continuum dataset {name!r} requires secondary_species for CIA")
                if opacity_unit != "cm-1 amagat-2":
                    raise ValueError(
                        f"continuum dataset {name!r} with continuum_type='cia' must use opacity_unit 'cm-1 amagat-2', "
                        f"got {opacity_unit!r}"
                    )
            else:
                if secondary_species not in (None, ""):
                    raise ValueError(
                        f"continuum dataset {name!r} with continuum_type='cross_section' must not define secondary_species"
                    )
                if opacity_unit != "cm2/molecule":
                    raise ValueError(
                        f"continuum dataset {name!r} with continuum_type='cross_section' must use opacity_unit 'cm2/molecule', "
                        f"got {opacity_unit!r}"
                    )

            if self.storage_format == "log10_uint16":
                if "y_min" not in dataset.attrs or "y_max" not in dataset.attrs:
                    raise ValueError(f"continuum dataset {name!r} is missing required y_min/y_max attrs")
                self.continuum_y_min[i] = float(dataset.attrs["y_min"])
                self.continuum_y_max[i] = float(dataset.attrs["y_max"])
            else:
                self.continuum_y_min[i] = np.nan
                self.continuum_y_max[i] = np.nan

            self.continuum_types.append(continuum_type)
            self.continuum_primary_species.append(primary_species)
            self.continuum_secondary_species.append(None if secondary_species in (None, "") else str(secondary_species))
            self.continuum_opacity_units.append(opacity_unit)

        self.pressure.flags.writeable = False
        self.temperature.flags.writeable = False
        self.wavelength.flags.writeable = False
        self.bin_edges.flags.writeable = False
        self.continuum_temperatures.flags.writeable = False

        self.workspace = RadtranOpacitiesWorkspace()
        if opacity_cache_size_limit is None:
            self.cache = None
        else:
            cache_dtype = np.uint16 if self.storage_format == "log10_uint16" else np.float32
            self.cache = RadtranOpacitiesCache(opacity_cache_size_limit, self.nwavelength, cache_dtype)

    def _prepare_cloud_interpolation(self, clouds: Clouds):
        if clouds is None or not clouds.interpolate:
            return

        _fill_cloud_interpolation_workspace(
            clouds.wavelength,
            self.wavelength,
            self.workspace.cloud_wavelength_ind0,
            self.workspace.cloud_wavelength_ind1,
            self.workspace.cloud_wavelength_weight,
        )

    def prepare_interpolation(self, atmosphere: RadtranAtmosphere, clouds: Clouds, nwavelengths_per_chunk: int):
        self.workspace._ensure(
            atmosphere.nlayers,
            self.npressure,
            self.ntemperature,
            self.ncontinuum_temperature,
            self.nwavelength,
            nwavelengths_per_chunk,
        )

        _fill_molecular_interpolation_workspace(
            atmosphere,
            self.pressure,
            self.temperature,
            self.workspace,
        )
        _fill_continuum_interpolation_workspace(
            atmosphere,
            self.continuum_temperatures,
            self.workspace,
        )
        self._prepare_cloud_interpolation(clouds)

    def _read_and_decode_opacity_row(
        self,
        dataset,
        source_sel,
        storage_code,
        y_min,
        y_max,
        post_decode_log10_factor,
        raw_out,
        out_row,
    ):
        dataset.read_direct(
            raw_out,
            source_sel=source_sel,
            dest_sel=np.s_[: raw_out.shape[0]],
        )
        if storage_code == 0:
            if y_max == y_min:
                out_row[:] = y_min
            else:
                out_row[:] = raw_out
                out_row *= (y_max - y_min) / np.iinfo(np.uint16).max
                out_row += y_min
        else:
            out_row[:] = raw_out
        out_row += post_decode_log10_factor

    def _read_and_decode_opacity_row_cached(
        self,
        dataset,
        source_sel,
        source_slice,
        cache_key,
        storage_code,
        y_min,
        y_max,
        post_decode_log10_factor,
        raw_full_buffer,
        out_row,
    ):
        raw_row = self.cache.get(cache_key)
        if raw_row is None:
            dataset.read_direct(
                raw_full_buffer,
                source_sel=source_sel,
                dest_sel=np.s_[: raw_full_buffer.shape[0]],
            )
            raw_row = raw_full_buffer
            self.cache.put(cache_key, raw_row)

        raw_view = raw_row[source_slice]
        if storage_code == 0:
            if y_max == y_min:
                out_row[:] = y_min
            else:
                out_row[:] = raw_view
                out_row *= (y_max - y_min) / np.iinfo(np.uint16).max
                out_row += y_min
        else:
            out_row[:] = raw_view
        out_row += post_decode_log10_factor

    def _read_and_decode_opacity(
        self,
        dataset,
        source_sel,
        source_slice,
        cache_key,
        storage_code,
        y_min,
        y_max,
        post_decode_log10_factor,
        raw_chunk_buffer,
        raw_full_buffer,
        out_row,
    ):
        if self.cache is None:
            self._read_and_decode_opacity_row(
                dataset,
                source_sel,
                storage_code,
                y_min,
                y_max,
                post_decode_log10_factor,
                raw_chunk_buffer,
                out_row,
            )
        else:
            self._read_and_decode_opacity_row_cached(
                dataset,
                source_sel,
                source_slice,
                cache_key,
                storage_code,
                y_min,
                y_max,
                post_decode_log10_factor,
                raw_full_buffer,
                out_row,
            )

    def _load_molecular_block(
        self,
        dataset,
        i_molecular,
        source_sel,
        source_slice,
        chunk_width,
        storage_code,
        raw_buffer,
        raw_full_buffer,
        block,
    ):
        for row_id in range(self.workspace.molecular_npairs):
            ip = self.workspace.molecular_pair_pindex[row_id]
            it = self.workspace.molecular_pair_tindex[row_id]
            self._read_and_decode_opacity(
                dataset,
                np.s_[ip, it, source_sel],
                source_slice,
                (dataset.name, ip, it),
                storage_code,
                self.molecular_y_min[i_molecular],
                self.molecular_y_max[i_molecular],
                0.0,
                raw_buffer[:chunk_width],
                raw_full_buffer,
                block[row_id, :chunk_width],
            )

    def _load_continuum_block(
        self,
        dataset,
        i_continuum,
        source_sel,
        source_slice,
        chunk_width,
        storage_code,
        post_decode_log10_factor,
        raw_buffer,
        raw_full_buffer,
        block,
    ):
        for row_id in range(self.workspace.continuum_nrows):
            it = self.workspace.continuum_temperature_load_idx[row_id]
            self._read_and_decode_opacity(
                dataset,
                np.s_[it, source_sel],
                source_slice,
                (dataset.name, it),
                storage_code,
                self.continuum_y_min[i_continuum],
                self.continuum_y_max[i_continuum],
                post_decode_log10_factor,
                raw_buffer[:chunk_width],
                raw_full_buffer,
                block[row_id, :chunk_width],
            )

    def compute_opacity(
        self, 
        atmosphere: RadtranAtmosphere, 
        settings: RadtranSettings, 
        clouds: Clouds, 
        surface: Surface, 
        ind_wv0: int, 
        ind_wv1: int, 
        opacities_result: RadtranOpacitiesResult
    ):  
        # Width of the wavelength chunk
        chunk_width = ind_wv1 - ind_wv0

        # Ensure we have the right allocated workspace.
        opacities_result._ensure(atmosphere.nlayers, 1, self.workspace.nwavelengths_per_chunk)
        
        # Get some workspace buffers, which depends on how opacities are encoded.
        storage_code = 0 if self.storage_format == "log10_uint16" else 1
        if storage_code == 0:
            raw_buffer = self.workspace.raw_u16
            raw_full_buffer = self.workspace.raw_full_u16
        else:
            raw_buffer = self.workspace.raw_f32
            raw_full_buffer = self.workspace.raw_full_f32

        # Gauss weights
        opacities_result.ck_weights[:] = 1.0

        # Wavelengths and surface reflectance
        opacities_result.wavelength_um[:chunk_width] = self.wavelength[ind_wv0:ind_wv1]
        if np.isscalar(surface.reflectance):
            opacities_result.surf_reflect[:chunk_width] = float(surface.reflectance)
        else:
            opacities_result.surf_reflect[:chunk_width] = surface.reflectance[ind_wv0:ind_wv1]

        # Get the needed slices.
        source_wv0 = self.wavelength_source_indices[ind_wv0]
        source_wv1 = self.wavelength_source_indices[ind_wv1 - 1] + 1
        full_source_sel = np.s_[self.wavelength_source_indices[0]: self.wavelength_source_indices[-1] + 1]
        source_slice = slice(ind_wv0, ind_wv1)
        source_sel = np.s_[source_wv0:source_wv1] if self.cache is None else np.s_[full_source_sel]

        # Counters
        nmolecular_active = 0
        ncontinuum_active = 0

        #~~ Molecular opacities ~~#
        opacities_result.taugas[:chunk_width, :, :] = 0.0
        taugas = opacities_result.taugas[:chunk_width, 0, :]
        for i_species in range(atmosphere.nspecies):
            species_name = str(atmosphere.species_names[i_species])
            if species_name not in self.molecular_name_to_index:
                continue

            i_molecular = self.molecular_name_to_index[species_name]
            block = self.workspace.molecular_block
            dataset = self._molecular_group[species_name]
            self._load_molecular_block(
                dataset,
                i_molecular,
                source_sel,
                source_slice,
                chunk_width,
                storage_code,
                raw_buffer,
                raw_full_buffer,
                block,
            )
            _accumulate_molecular_tau(
                block[:self.workspace.molecular_npairs, :chunk_width],
                atmosphere.layer_columns[i_species],
                self.workspace.molecular_pressure_ind0,
                self.workspace.molecular_pressure_ind1,
                self.workspace.molecular_pressure_weight,
                self.workspace.molecular_temperature_ind0,
                self.workspace.molecular_temperature_ind1,
                self.workspace.molecular_temperature_weight,
                taugas,
            )

            nmolecular_active += 1

        #~~ CIA & continuum ~~#
        atmosphere_name_to_index = {str(name): i for i, name in enumerate(atmosphere.species_names)}

        for i_continuum, continuum_name in enumerate(self.continuum_names):
            continuum_type = self.continuum_types[i_continuum]
            primary_species = self.continuum_primary_species[i_continuum]
            secondary_species = self.continuum_secondary_species[i_continuum]

            if primary_species not in atmosphere_name_to_index:
                continue
            i_primary = atmosphere_name_to_index[primary_species]
            block = self.workspace.continuum_block
            dataset = self._continuum_group[continuum_name]
            if continuum_type == "cia":
                if secondary_species is None or secondary_species not in atmosphere_name_to_index:
                    continue
                i_secondary = atmosphere_name_to_index[secondary_species]
                _fill_continuum_scale_workspace(atmosphere, i_primary, i_secondary, self.workspace)
                log10_scale = np.log10(CIA_AMAGAT_TO_MOLECULE_CM)
            elif continuum_type == "cross_section":
                _fill_cross_section_scale_workspace(atmosphere, i_primary, self.workspace)
                log10_scale = 0.0
            else:
                raise ValueError(
                    f"unsupported continuum_type {continuum_type!r} for continuum {continuum_name!r}"
                )
            self._load_continuum_block(
                dataset,
                i_continuum,
                source_sel,
                source_slice,
                chunk_width,
                storage_code,
                log10_scale,
                raw_buffer,
                raw_full_buffer,
                block,
            )
            _accumulate_continuum_tau(
                block[:self.workspace.continuum_nrows, :chunk_width],
                self.workspace.continuum_scale,
                self.workspace.continuum_temperature_ind0,
                self.workspace.continuum_temperature_ind1,
                self.workspace.continuum_temperature_weight,
                taugas,
            )

            ncontinuum_active += 1

        # Check to make sure the cache is big enough to be useful.
        if self.cache is not None:
            max_molecular_npairs = min(4 * atmosphere.nlayers, self.npressure * self.ntemperature)
            max_continuum_npairs = min(2 * atmosphere.nlayers, self.ncontinuum_temperature)
            min_cache_capacity = nmolecular_active*max_molecular_npairs
            min_cache_capacity += ncontinuum_active*max_continuum_npairs
            if self.cache.capacity < min_cache_capacity:
                min_cache_size = min_cache_capacity * self.cache.row_nbytes
                raise ValueError(
                    "opacity_cache_size_limit is too small for the active opacity working set; "
                    f"need at least {min_cache_size} bytes to cache {min_cache_capacity} rows "
                    f"of {self.cache.row_nbytes} bytes each, got {self.cache.size_limit_bytes} bytes "
                    f"(capacity {self.cache.capacity} rows)"
                )

        #~~ Rayleigh ~~#
        tauray = opacities_result.tauray[:chunk_width, :]
        tauray[:,:] = 0.0
        rayleigh_sigma = self.workspace.rayleigh_sigma[:chunk_width]
        wavelength_chunk = self.wavelength[ind_wv0:ind_wv1]
        for i_species in range(atmosphere.nspecies):
            species_name = str(atmosphere.species_names[i_species])
            if species_name not in RAYLEIGH_MOLECULES:
                continue
            compute_rayleigh_sigma(species_name, wavelength_chunk, rayleigh_sigma)
            _accumulate_rayleigh_tau(rayleigh_sigma, atmosphere.layer_columns[i_species], tauray)

        #~~ Raman ~~#
        compute_raman(
            settings.raman,
            opacities_result.wavelength_um[:chunk_width],
            opacities_result.raman_factor[:chunk_width],
        )

        #~~ Clouds ~~#
        taucld = opacities_result.taucld[:chunk_width, :]
        w0_cld = opacities_result.w0_cld[:chunk_width, :]
        g0_cld = opacities_result.g0_cld[:chunk_width, :]
        if clouds is not None:
            _set_clouds(
                clouds,
                ind_wv0,
                ind_wv1,
                taucld,
                w0_cld,
                g0_cld,
                self.workspace.cloud_wavelength_ind0[ind_wv0:ind_wv1],
                self.workspace.cloud_wavelength_ind1[ind_wv0:ind_wv1],
                self.workspace.cloud_wavelength_weight[ind_wv0:ind_wv1],
            )
        else:
            taucld[:,:] = 0.0
            w0_cld[:,:] = 0.0
            g0_cld[:,:] = 0.0

        # Finish up the calculation
        _finish_compute_opacity(
            opacities_result,
            chunk_width,
            1,
            settings.stream,
            settings.delta_eddington,
            fthin_cld=1.0,
        )
    
    def adjust_opacity_for_clearsky(self, clouds: Clouds, settings: RadtranSettings, ind_wv0: int, ind_wv1: int, opacities_result: RadtranOpacitiesResult):
        
        if clouds is None or not clouds.do_holes:
            raise ValueError("adjust_opacity_for_clearsky requires a cloud object with do_holes=True")
        
        chunk_width = ind_wv1 - ind_wv0

        _finish_compute_opacity(
            opacities_result,
            chunk_width,
            1,
            settings.stream,
            settings.delta_eddington,
            clouds.fthin_cld,
        )

    def close(self) -> None:
        if getattr(self, "file", None) is not None:
            self.file.close()
            self.file = None
            self._header = None
            self._molecular_group = None
            self._continuum_group = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@nb.njit(nb.float64(nb.float64), fastmath=True, inline="always")
def fast_pow10(x):
    return np.exp(LOG10 * x)


@nb.njit
def _bracket_1d(grid, value):
    ngrid = grid.shape[0]
    if value <= grid[0]:
        return 0, 0, 0.0
    if value >= grid[ngrid - 1]:
        return ngrid - 1, ngrid - 1, 0.0

    lo = 0
    hi = ngrid - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if value < grid[mid]:
            hi = mid
        else:
            lo = mid

    weight = (value - grid[lo]) / (grid[hi] - grid[lo])
    return lo, hi, weight


@nb.njit
def _get_or_create_molecular_pair(ip, it, workspace):
    row_id = workspace.molecular_pair_map[ip, it]
    if row_id == -1:
        row_id = workspace.molecular_npairs
        workspace.molecular_pair_map[ip, it] = row_id
        workspace.molecular_pair_pindex[row_id] = ip
        workspace.molecular_pair_tindex[row_id] = it
        workspace.molecular_npairs += 1
    return row_id


@nb.njit
def _get_or_create_continuum_temp(it, workspace):
    row_id = workspace.continuum_temperature_map[it]
    if row_id == -1:
        row_id = workspace.continuum_nrows
        workspace.continuum_temperature_map[it] = row_id
        workspace.continuum_temperature_load_idx[row_id] = it
        workspace.continuum_nrows += 1
    return row_id


@nb.njit
def _fill_molecular_interpolation_workspace(atmosphere, pressure_grid, temperature_grid, workspace):
    nlayers = atmosphere.nlayers
    workspace.molecular_npairs = 0
    for ip in range(workspace.npressure):
        for it in range(workspace.ntemperature):
            workspace.molecular_pair_map[ip, it] = -1
    for i in range(nlayers):
        ip0, ip1, pw = _bracket_1d(pressure_grid, atmosphere.layer_pressures_cgs[i]/1.0e6)
        it0, it1, tw = _bracket_1d(temperature_grid, atmosphere.layer_temperatures[i])

        workspace.molecular_pressure_ind0[i] = _get_or_create_molecular_pair(ip0, it0, workspace)
        workspace.molecular_pressure_ind1[i] = _get_or_create_molecular_pair(ip1, it0, workspace)
        workspace.molecular_temperature_ind0[i] = _get_or_create_molecular_pair(ip0, it1, workspace)
        workspace.molecular_temperature_ind1[i] = _get_or_create_molecular_pair(ip1, it1, workspace)
        workspace.molecular_pressure_weight[i] = pw
        workspace.molecular_temperature_weight[i] = tw


@nb.njit
def _fill_continuum_interpolation_workspace(atmosphere, temperature_grid, workspace):
    nlayers = atmosphere.nlayers
    workspace.continuum_nrows = 0
    for it in range(workspace.ncontinuum_temperature):
        workspace.continuum_temperature_map[it] = -1
    for i in range(nlayers):
        it0, it1, tw = _bracket_1d(temperature_grid, atmosphere.layer_temperatures[i])
        workspace.continuum_temperature_ind0[i] = _get_or_create_continuum_temp(it0, workspace)
        workspace.continuum_temperature_ind1[i] = _get_or_create_continuum_temp(it1, workspace)
        workspace.continuum_temperature_weight[i] = tw


@nb.njit
def _fill_continuum_scale_workspace(atmosphere, i_primary_species, i_secondary_species, workspace):
    nlayers = atmosphere.nlayers
    for i in range(nlayers):
        workspace.continuum_scale[i] = (
            atmosphere.layer_densities[i_primary_species, i]
            * atmosphere.layer_densities[i_secondary_species, i]
            * atmosphere.layer_dz[i]
        )


@nb.njit
def _fill_cross_section_scale_workspace(atmosphere, i_primary_species, workspace):
    nlayers = atmosphere.nlayers
    for i in range(nlayers):
        workspace.continuum_scale[i] = atmosphere.layer_densities[i_primary_species, i] * atmosphere.layer_dz[i]


@nb.njit(cache=True)
def _ck_interp_log_table_row(table, p0, p1, t0, t1, pw, tw, iw, out):
    c00 = (1.0 - pw) * (1.0 - tw)
    c10 = pw * (1.0 - tw)
    c01 = (1.0 - pw) * tw
    c11 = pw * tw
    ng = out.shape[0]
    for ig in range(ng):
        out[ig] = fast_pow10(
            c00 * table[p0, t0, iw, ig]
            + c10 * table[p1, t0, iw, ig]
            + c01 * table[p0, t1, iw, ig]
            + c11 * table[p1, t1, iw, ig]
        )


@nb.njit(cache=True)
def _ck_mix_2_gases(k1, k2, mix1, mix2, gauss_pts, gauss_wts, kmix, wtsmix):
    mix_t = mix1 + mix2
    ng = gauss_wts.shape[0]
    if mix_t <= 0.0:
        for i in range(ng):
            k1[i] = 0.0
        return 0.0

    for i in range(ng):
        for j in range(ng):
            idx = i * ng + j
            kmix[idx] = (mix1 * k1[i] + mix2 * k2[j]) / mix_t
            wtsmix[idx] = gauss_wts[i] * gauss_wts[j]

    sort_indices = np.argsort(kmix, kind="mergesort")
    kmix_sort = np.maximum(kmix[sort_indices], 1.0e-300)
    wtsmix_sort = wtsmix[sort_indices]
    cumulative = np.cumsum(wtsmix_sort)
    x = cumulative / cumulative[-1]
    k1[:] = fast_pow10(np.interp(gauss_pts, x, np.log10(kmix_sort)))
    return mix_t


@nb.njit(cache=True)
def _compute_ck_molecular_taugas(
    molecular_tables,
    active_species_indices,
    active_table_indices,
    layer_mixing_ratios,
    layer_colden,
    layer_mubar,
    p_ind0,
    p_ind1,
    p_weight,
    t_ind0,
    t_ind1,
    t_weight,
    pair_pindex,
    pair_tindex,
    g_points,
    g_weights,
    taugas_out,
):
    nlayers = layer_mubar.shape[0]
    nwavelengths = taugas_out.shape[0]
    ngauss = g_points.shape[0]
    nspecies = active_species_indices.shape[0]
    mixed = np.empty(ngauss, dtype=np.float64)
    tmp = np.empty(ngauss, dtype=np.float64)
    kmix = np.empty(ngauss * ngauss, dtype=np.float64)
    wtsmix = np.empty(ngauss * ngauss, dtype=np.float64)

    for il in range(nlayers):
        total_column = layer_colden[il] / (layer_mubar[il] * AMU_CGS)

        row00 = p_ind0[il]
        row10 = p_ind1[il]
        row01 = t_ind0[il]
        row11 = t_ind1[il]
        p0 = pair_pindex[row00]
        p1 = pair_pindex[row10]
        t0 = pair_tindex[row01]
        t1 = pair_tindex[row11]
        pw = p_weight[il]
        tw = t_weight[il]

        if nspecies == 0:
            for iw in range(nwavelengths):
                for ig in range(ngauss):
                    taugas_out[iw, ig, il] = 0.0
            continue

        if nspecies == 1:
            table_index = active_table_indices[0]
            for iw in range(nwavelengths):
                _ck_interp_log_table_row(
                    molecular_tables[table_index],
                    p0,
                    p1,
                    t0,
                    t1,
                    pw,
                    tw,
                    iw,
                    tmp,
                )
                for ig in range(ngauss):
                    taugas_out[iw, ig, il] = tmp[ig] * total_column
            continue

        for iw in range(nwavelengths):
            first = True
            mix_total = 0.0
            for ispecies in range(nspecies):
                species_index = active_species_indices[ispecies]
                table_index = active_table_indices[ispecies]
                mix = layer_mixing_ratios[species_index, il]
                _ck_interp_log_table_row(
                    molecular_tables[table_index],
                    p0,
                    p1,
                    t0,
                    t1,
                    pw,
                    tw,
                    iw,
                    tmp,
                )

                if first:
                    for ig in range(ngauss):
                        mixed[ig] = tmp[ig]
                    mix_total = mix
                    first = False
                else:
                    mix_total = _ck_mix_2_gases(
                        mixed,
                        tmp,
                        mix_total,
                        mix,
                        g_points,
                        g_weights,
                        kmix,
                        wtsmix,
                    )

            for ig in range(ngauss):
                taugas_out[iw, ig, il] = mixed[ig] * total_column


@nb.njit(cache=True)
def _add_ck_continuum_species_taugas(
    continuum_table,
    continuum_scale_constant,
    continuum_scale_row,
    t_ind0,
    t_ind1,
    t_weight,
    load_index,
    taugas_out,
):
    nwavelengths = taugas_out.shape[0]
    ngauss = taugas_out.shape[1]
    nlayers = taugas_out.shape[2]

    for il in range(nlayers):
        it0 = load_index[t_ind0[il]]
        it1 = load_index[t_ind1[il]]
        tw = t_weight[il]
        c0 = 1.0 - tw
        c1 = tw
        add_scale = continuum_scale_constant * continuum_scale_row[il]
        for iw in range(nwavelengths):
            coeff = fast_pow10(c0 * continuum_table[it0, iw] + c1 * continuum_table[it1, iw]) * add_scale
            for ig in range(ngauss):
                taugas_out[iw, ig, il] += coeff

@nb.njit
def _accumulate_molecular_tau(block, columns_row, p_ind0, p_ind1, p_weight, t_ind0, t_ind1, t_weight, tau_out):
    nwavelengths = block.shape[1]
    nlayers = columns_row.shape[0]

    for i in range(nlayers):
        column = columns_row[i]
        i00 = p_ind0[i]
        i10 = p_ind1[i]
        i01 = t_ind0[i]
        i11 = t_ind1[i]
        pw = p_weight[i]
        tw = t_weight[i]
        c00 = (1.0 - pw) * (1.0 - tw)
        c10 = pw * (1.0 - tw)
        c01 = (1.0 - pw) * tw
        c11 = pw * tw

        for iw in range(nwavelengths):
            log_opacity = (
                c00 * block[i00, iw]
                + c10 * block[i10, iw]
                + c01 * block[i01, iw]
                + c11 * block[i11, iw]
            )
            tau_out[iw, i] += fast_pow10(log_opacity) * column


@nb.njit
def _accumulate_continuum_tau(block, continuum_scale_row, t_ind0, t_ind1, t_weight, tau_out):
    nwavelengths = block.shape[1]
    nlayers = continuum_scale_row.shape[0]

    for i in range(nlayers):
        scale = continuum_scale_row[i]
        it0 = t_ind0[i]
        it1 = t_ind1[i]
        tw = t_weight[i]
        c0 = 1.0 - tw
        c1 = tw
        for iw in range(nwavelengths):
            log_opacity = c0 * block[it0, iw] + c1 * block[it1, iw]
            tau_out[iw, i] += fast_pow10(log_opacity) * scale


@nb.njit
def _accumulate_rayleigh_tau(sigma_row, columns_row, tau_out):
    nwavelengths = sigma_row.shape[0]
    nlayers = columns_row.shape[0]

    for iw in range(nwavelengths):
        sigma = sigma_row[iw]
        for i in range(nlayers):
            tau_out[iw, i] += sigma * (columns_row[i] / AVOGADRO)

@nb.njit
def _fill_cloud_interpolation_workspace(source_wavelength, target_wavelength, ind0, ind1, weight):
    for i in range(target_wavelength.shape[0]):
        lo, hi, w = _bracket_1d(source_wavelength, target_wavelength[i])
        ind0[i] = lo
        ind1[i] = hi
        weight[i] = w


@nb.njit
def _set_clouds(
    clouds,
    ind_wv0,
    ind_wv1,
    taucld,
    w0_cld,
    g0_cld,
    cloud_ind0,
    cloud_ind1,
    cloud_weight,
):
    if not clouds.interpolate:
        taucld[:, :] = clouds.opd[ind_wv0:ind_wv1, :]
        w0_cld[:, :] = clouds.w0[ind_wv0:ind_wv1, :]
        g0_cld[:, :] = clouds.g0[ind_wv0:ind_wv1, :]
        return

    # Interpolate the clouds in wavelength only. Pressure is already aligned.
    chunk_width = ind_wv1 - ind_wv0
    nlayers = clouds.nlayers

    for iw in range(chunk_width):
        lo = cloud_ind0[iw]
        hi = cloud_ind1[iw]
        weight = cloud_weight[iw]
        c0 = 1.0 - weight
        c1 = weight
        for il in range(nlayers):
            taucld[iw, il] = c0 * clouds.opd[lo, il] + c1 * clouds.opd[hi, il]
            w0_cld[iw, il] = c0 * clouds.w0[lo, il] + c1 * clouds.w0[hi, il]
            g0_cld[iw, il] = c0 * clouds.g0[lo, il] + c1 * clouds.g0[hi, il]


@nb.njit
def _finish_compute_opacity(result: RadtranOpacitiesResult, chunk_width, ngauss, stream, delta_eddington, fthin_cld):

    for iw in range(chunk_width):
        raman_factor = result.raman_factor[iw]

        for igauss in range(ngauss):

            running_tau = 0.0
            running_tau_dedd = 0.0
            result.tau[iw, igauss, 0] = 0.0
            result.tau_dedd[iw, igauss, 0] = 0.0

            for i in range(result.nlayers):
                # Unpack to scalars
                taugas = result.taugas[iw, igauss, i]
                tauray = result.tauray[iw, i]
                taucld = result.taucld[iw, i]
                w0_cld = result.w0_cld[iw, i]
                g0_cld = result.g0_cld[iw, i]

                # Apply thinning to cloud
                taucld *= fthin_cld

                # Total opacity
                dtau = taugas + tauray + taucld

                tauscat_cld = w0_cld*taucld
                tauscat = tauscat_cld + tauray
                if tauscat > 0.0:
                    # Fraction of total scattering due to clouds.
                    ftau_cld = tauscat_cld/tauscat
                    # Fraction of total scattering due to Rayleigh.
                    ftau_ray = tauray/tauscat
                    # Hansen & Travis 1974 for Rayleigh scattering 
                    gcos2 = 0.5 * ftau_ray
                else:
                    ftau_cld = 0.0
                    ftau_ray = 0.0
                    gcos2 = 0.0

                # Asymmetry
                cosb = g0_cld

                # Single scattering albedo
                if dtau > 0:
                    w0 = (tauray*raman_factor + taucld*w0_cld)/dtau
                    w0 = np.minimum(np.maximum(w0, 1.0e-8), 1.0 - 1.0e-8)

                    w0_no_raman = (tauray*0.99999 + taucld*w0_cld)/dtau
                    w0_no_raman = np.minimum(np.maximum(w0_no_raman, 1.0e-8), 1.0 - 1.0e-8)
                else:
                    w0 = 1.0e-8
                    w0_no_raman = 1.0e-8

                # Cumulative total opacity
                running_tau += dtau

                # Delta eddington
                if delta_eddington:
                    f_deltaM = cosb**stream
                    w0_dedd = w0*(1.0 - f_deltaM)/(1.0 - w0*f_deltaM)
                    cosb_dedd = (cosb - f_deltaM)/(1.0 - f_deltaM)
                    dtau_dedd = dtau*(1.0 - w0*f_deltaM)
                    running_tau_dedd += dtau_dedd
                else:
                    w0_dedd = w0
                    cosb_dedd = cosb
                    dtau_dedd = dtau
                    running_tau_dedd += dtau_dedd

                # Save results
                result.dtau[iw, igauss, i] = dtau
                result.ftau_cld[iw, i] = ftau_cld
                result.ftau_ray[iw, i] = ftau_ray
                result.gcos2[iw, i] = gcos2
                result.cosb[iw, igauss, i] = cosb
                result.w0[iw, igauss, i] = w0
                result.w0_no_raman[iw, igauss, i] = w0_no_raman
                result.tau[iw, igauss, i + 1] = running_tau
                result.w0_dedd[iw, igauss, i] = w0_dedd
                result.cosb_dedd[iw, igauss, i] = cosb_dedd
                result.dtau_dedd[iw, igauss, i] = dtau_dedd
                result.tau_dedd[iw, igauss, i + 1] = running_tau_dedd
        
@nb.experimental.jitclass
class RadtranOpacitiesResult:

    # Dimensions
    nlayers : nb.int64
    nwavelengths_per_chunk : nb.int64
    ngauss : nb.int64

    wavelength_um : nb.float64[:]
    surf_reflect : nb.float64[:]
    ck_weights : nb.float64[:]
    taugas : nb.float64[:,:,:]
    tauray : nb.float64[:,:]
    taucld : nb.float64[:,:]
    w0_cld : nb.float64[:,:]
    g0_cld : nb.float64[:,:]

    dtau_dedd : nb.float64[:,:,:]
    tau_dedd : nb.float64[:,:,:]
    w0_dedd : nb.float64[:,:,:]
    cosb_dedd : nb.float64[:,:,:]

    ftau_cld : nb.float64[:,:]
    ftau_ray : nb.float64[:,:]
    gcos2 : nb.float64[:,:]

    dtau : nb.float64[:,:,:]
    tau : nb.float64[:,:,:]
    w0 : nb.float64[:,:,:]
    w0_no_raman : nb.float64[:,:,:]
    cosb : nb.float64[:,:,:]
    raman_factor : nb.float64[:]

    spectrum : nb.float64[:]

    def __init__(self):
        self._allocate(0, 1, 0)

    def _allocate(self, nlayers, ngauss, nwavelengths_per_chunk):
        self.nlayers = nlayers
        self.ngauss = ngauss
        self.nwavelengths_per_chunk = nwavelengths_per_chunk

        self.wavelength_um = np.empty(nwavelengths_per_chunk, dtype=np.float64)
        self.surf_reflect = np.empty(nwavelengths_per_chunk, dtype=np.float64)
        self.ck_weights = np.ones(ngauss, dtype=np.float64)
        self.taugas = np.empty((nwavelengths_per_chunk, ngauss, nlayers), dtype=np.float64)
        self.tauray = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.taucld = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.w0_cld = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.g0_cld = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)

        self.dtau_dedd = np.empty((nwavelengths_per_chunk, ngauss, nlayers), dtype=np.float64)
        self.tau_dedd = np.empty((nwavelengths_per_chunk, ngauss, nlayers+1), dtype=np.float64)
        self.w0_dedd = np.empty((nwavelengths_per_chunk, ngauss, nlayers), dtype=np.float64)
        self.cosb_dedd = np.empty((nwavelengths_per_chunk, ngauss, nlayers), dtype=np.float64)

        self.ftau_cld = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.ftau_ray = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.gcos2 = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)

        self.dtau = np.empty((nwavelengths_per_chunk, ngauss, nlayers), dtype=np.float64)
        self.tau = np.empty((nwavelengths_per_chunk, ngauss, nlayers+1), dtype=np.float64)
        self.w0 = np.empty((nwavelengths_per_chunk, ngauss, nlayers), dtype=np.float64)
        self.w0_no_raman = np.empty((nwavelengths_per_chunk, ngauss, nlayers), dtype=np.float64)
        self.cosb = np.empty((nwavelengths_per_chunk, ngauss, nlayers), dtype=np.float64)
        self.raman_factor = np.empty(nwavelengths_per_chunk, dtype=np.float64)

        self.spectrum = np.empty(nwavelengths_per_chunk, dtype=np.float64)

    def _ensure(self, nlayers, ngauss, nwavelengths_per_chunk):
        if (
            nlayers != self.nlayers
            or ngauss != self.ngauss
            or nwavelengths_per_chunk != self.nwavelengths_per_chunk
        ):
            self._allocate(nlayers, ngauss, nwavelengths_per_chunk)

@nb.experimental.jitclass
class RadtranAtmosphere:

    radius: nb.float64
    mass: nb.float64
    semimajor: nb.float64

    nlayers: nb.int64
    nspecies: nb.int64

    species_names: nb.types.ListType(nb.types.unicode_type)
    species_mu: nb.float64[:]
    level_pressures_cgs: nb.float64[:]
    level_temperatures: nb.float64[:]
    level_mixing_ratios: nb.float64[:,:]
    level_mubar: nb.float64[:]
    level_z: nb.float64[:]
    level_dz: nb.float64[:]
    level_gravity: nb.float64[:]
    level_scale_height: nb.float64[:]
    level_density: nb.float64[:]
    layer_pressures_cgs: nb.float64[:]
    layer_temperatures: nb.float64[:]
    layer_mixing_ratios: nb.float64[:,:]
    layer_mubar: nb.float64[:]
    layer_gravity: nb.float64[:]
    layer_dz: nb.float64[:]
    layer_density: nb.float64[:]
    layer_densities: nb.float64[:,:]
    layer_columns: nb.float64[:,:]
    layer_colden: nb.float64[:]
    reference_pressure: nb.float64

    def __init__(self):
        self._allocate(0, 0)

    def setup(self, atm: Atmosphere_, planet: Planet):
        # Ensure size
        self._ensure(atm.nlayers, atm.nspecies)

        # Copy over information in Atmosphere_ and Planet.
        # Planet inputs are stored in Earth/AU units; keep CGS internally here.
        self.radius = planet.radius * R_EARTH_CGS
        self.mass = planet.mass * M_EARTH_CGS
        self.semimajor = planet.semimajor * 1.495978707e13

        self.nlayers = atm.nlayers
        self.nspecies = atm.nspecies
        self.species_names = atm.species_names
        self.species_mu[:] = atm.species_mu[:]
        self.level_pressures_cgs[:] = atm.level_pressures[:] * 1.0e6
        self.level_temperatures[:] = atm.level_temperatures[:]
        self.level_mixing_ratios[:, :] = atm.level_mixing_ratios[:, :]
        self.layer_pressures_cgs[:] = atm.layer_pressures[:] * 1.0e6
        self.layer_temperatures[:] = atm.layer_temperatures[:]
        self.layer_mixing_ratios[:, :] = atm.layer_mixing_ratios[:, :]
        self.reference_pressure = atm.reference_pressure

        # Mean molecular weight on each level.
        for i in range(self.nlayers + 1):
            mu = 0.0
            for j in range(self.nspecies):
                mu += self.species_mu[j] * self.level_mixing_ratios[j, i]
            self.level_mubar[i] = mu

        # Mean molecular weight on each layer.
        for i in range(self.nlayers):
            mu = 0.0
            for j in range(self.nspecies):
                mu += self.species_mu[j] * self.layer_mixing_ratios[j, i]
            self.layer_mubar[i] = mu

        # Build the hydrostatic solution on the legacy level grid.
        # The level grid is anchored at the nearest level pressure to the
        # requested reference pressure, mirroring atmsetup.get_altitude().
        nlevels = self.nlayers + 1
        p_reference = self.reference_pressure * 1.0e6
        max_pressure = np.max(self.level_pressures_cgs)
        if p_reference >= max_pressure:
            p_reference = max_pressure
        else:
            p_reference = self.level_pressures_cgs[self.level_pressures_cgs >= p_reference][0]

        planet_radius = self.radius
        planet_mass = self.mass
        gravity_work = np.zeros(nlevels, dtype=np.float64)

        self.level_z[:] = planet_radius
        self.level_gravity[:] = 0.0
        self.level_dz[:] = 0.0
        self.level_scale_height[:] = 0.0
        self.level_density[:] = 0.0
        self.layer_dz[:] = 0.0
        self.layer_colden[:] = 0.0

        iref = 0
        while iref < nlevels and self.level_pressures_cgs[iref] < p_reference:
            iref += 1
        if iref == nlevels:
            iref = nlevels - 1

        for i in range(iref, nlevels - 1):
            gravity_work[i] = G_CGS * planet_mass / (self.level_z[i] * self.level_z[i])
            self.level_scale_height[i] = KB_CGS * self.level_temperatures[i] / (self.level_mubar[i] * AMU_CGS * gravity_work[i])
            delta_logp = np.log(self.level_pressures_cgs[i + 1] / self.level_pressures_cgs[i])
            self.level_dz[i] = self.level_scale_height[i] * delta_logp
            self.level_z[i + 1] = self.level_z[i] - self.level_dz[i]

        for i in range(iref, 0, -1):
            gravity_work[i] = G_CGS * planet_mass / (self.level_z[i] * self.level_z[i])
            self.level_scale_height[i] = KB_CGS * self.level_temperatures[i] / (self.level_mubar[i] * AMU_CGS * gravity_work[i])
            delta_logp = np.log(self.level_pressures_cgs[i] / self.level_pressures_cgs[i - 1])
            self.level_dz[i] = self.level_scale_height[i] * delta_logp
            self.level_z[i - 1] = self.level_z[i] + self.level_dz[i]

        # Populate the layer gravity using the same ordering as the legacy code:
        # it is computed before the endpoint gravity values are filled.
        self.layer_gravity[:] = 0.5 * (gravity_work[:-1] + gravity_work[1:])

        # Populate the endpoint gravity values.
        for i in range(nlevels):
            self.level_gravity[i] = G_CGS * planet_mass / (self.level_z[i] * self.level_z[i])
            self.level_density[i] = self.level_pressures_cgs[i] / (KB_CGS * self.level_temperatures[i])

        self.level_scale_height[:] = (
            KB_CGS * self.level_temperatures[:] / (self.level_mubar[:] * AMU_CGS * self.level_gravity[:])
        )
        if nlevels > 1:
            self.level_dz[0] = self.level_dz[1]
            self.level_dz[nlevels - 1] = self.level_dz[nlevels - 2]

        # Layer quantities derived from the legacy level grid.
        for i in range(self.nlayers):
            self.layer_dz[i] = self.level_z[i] - self.level_z[i + 1]
            self.layer_density[i] = self.layer_pressures_cgs[i] / (KB_CGS * self.layer_temperatures[i])
            self.layer_colden[i] = (self.level_pressures_cgs[i + 1] - self.level_pressures_cgs[i]) / self.layer_gravity[i]
            for j in range(self.nspecies):
                self.layer_densities[j, i] = self.layer_mixing_ratios[j, i] * self.layer_density[i]
                self.layer_columns[j, i] = (
                    self.layer_mixing_ratios[j, i] * self.layer_colden[i] / (self.layer_mubar[i] * AMU_CGS)
                )

    def _allocate(self, nlayers, nspecies):
        self.nlayers = nlayers
        self.nspecies = nspecies
        self.radius = np.nan
        self.mass = np.nan
        self.semimajor = np.nan
        self.species_names = nb.typed.List.empty_list(nb.types.unicode_type)
        self.species_mu = np.empty(nspecies, dtype=np.float64)
        self.level_pressures_cgs = np.empty(nlayers + 1, dtype=np.float64)
        self.level_temperatures = np.empty(nlayers + 1, dtype=np.float64)
        self.level_mixing_ratios = np.empty((nspecies, nlayers + 1), dtype=np.float64)
        self.level_mubar = np.empty(nlayers + 1, dtype=np.float64)
        self.level_z = np.empty(nlayers + 1, dtype=np.float64)
        self.level_dz = np.empty(nlayers + 1, dtype=np.float64)
        self.level_gravity = np.empty(nlayers + 1, dtype=np.float64)
        self.level_scale_height = np.empty(nlayers + 1, dtype=np.float64)
        self.level_density = np.empty(nlayers + 1, dtype=np.float64)
        self.layer_pressures_cgs = np.empty(nlayers, dtype=np.float64)
        self.layer_temperatures = np.empty(nlayers, dtype=np.float64)
        self.layer_mixing_ratios = np.empty((nspecies, nlayers), dtype=np.float64)
        self.layer_mubar = np.empty(nlayers, dtype=np.float64)
        self.layer_gravity = np.empty(nlayers, dtype=np.float64)
        self.layer_dz = np.empty(nlayers, dtype=np.float64)
        self.layer_density = np.empty(nlayers, dtype=np.float64)
        self.layer_densities = np.empty((nspecies, nlayers), dtype=np.float64)
        self.layer_columns = np.empty((nspecies, nlayers), dtype=np.float64)
        self.layer_colden = np.empty(nlayers, dtype=np.float64)
        self.reference_pressure = np.nan

    def _ensure(self, nlayers, nspecies):
        if nlayers != self.nlayers or nspecies != self.nspecies:
            self._allocate(nlayers, nspecies)


@nb.njit
def _validate_clouds(clouds: Clouds, pressures, wavelength):

    if clouds.nlayers != len(pressures):
        raise ValueError(
            f"clouds.nlayers must match len(pressures), got {clouds.nlayers} and {len(pressures)}"
        )
    for i in range(clouds.nlayers):
        if not np.isclose(clouds.pressure[i] * 1.0e6, pressures[i]):
            raise ValueError(
                f"cloud pressure grid must match atmosphere pressures at index {i}, "
                f"got {clouds.pressure[i]} and {pressures[i]}"
            )

    # If we are going to interpolate, then return
    if clouds.interpolate:
        for i in range(clouds.nwavelengths):
            if not np.isfinite(clouds.wavelength[i]):
                raise ValueError(
                    f"cloud wavelength grid must contain only finite values, got {clouds.wavelength[i]} at index {i}"
                )
        for i in range(clouds.nwavelengths - 1):
            if clouds.wavelength[i+1] <= clouds.wavelength[i]:
                raise ValueError(
                    "cloud wavelength grid must be strictly increasing when interpolate=True, "
                    f"got {clouds.wavelength[i+1]} <= {clouds.wavelength[i]} at indices {i+1} and {i}"
                )
    else:
        if clouds.nwavelengths != len(wavelength):
            raise ValueError(
                f"clouds.nwavelengths must match len(wavelength), got {clouds.nwavelengths} and {len(wavelength)}"
            )
        for i in range(clouds.nwavelengths):
            if not np.isclose(clouds.wavelength[i], wavelength[i]):
                raise ValueError(
                    f"cloud wavelength grid must match opacity wavelength grid at index {i}, "
                    f"got {clouds.wavelength[i]} and {wavelength[i]}"
                )

@nb.njit
def _validate_wavelength(wavelength1, wavelength2):

    if len(wavelength1) != len(wavelength2):
        raise ValueError("External wavelength grid must match the Radtran wavelength grid")
    for i in range(len(wavelength1)):
        if not np.isclose(wavelength1[i], wavelength2[i]):
            raise ValueError(f"External wavelength grid must match the Radtran wavelength grid at index {i}")

class Radtran:
    "Radiative-transfer driver."

    def __init__(
        self,
        opacity_filename: str,
        wavelength_range=None,
        settings_kwargs=None,
        phase_kwargs=None,
        opacity_cache_size_limit=None,
    ):

        # Opacities
        self.opacities = _make_radtran_opacities(opacity_filename, wavelength_range, opacity_cache_size_limit)
        self.opacities_result = RadtranOpacitiesResult()

        # Atmosphere
        self.atmosphere = RadtranAtmosphere()
        self.clouds = None
        self.surface = None
        self.star = None

        # Solvers
        self.thermal = ThermalSolver()
        self.thermal_result = ThermalResult()

        self.reflected = ReflectedSolver()
        self.reflected_result = ReflectedResult()

        self.transmission_result = TransmissionResult()

        # Phase
        if phase_kwargs is None:
            phase_kwargs = {}
        self.phase = RadtranPhase(**phase_kwargs)

        # Various settings
        if settings_kwargs is None:
            settings_kwargs = {}
        self.settings = RadtranSettings(**settings_kwargs)

        # Variables that will be set later
        self.nwavelengths_per_chunk = None
        self.nwavelength_chunks = None

    def _set_wavelength_chunks(self, nwavelengths_per_chunk):

        # Work out the wavelength chunking
        self.nwavelengths_per_chunk = nwavelengths_per_chunk
        if self.nwavelengths_per_chunk is None:
            self.nwavelengths_per_chunk = self.opacities.nwavelength
        if self.nwavelengths_per_chunk <= 0:
            raise ValueError("nwavelengths_per_chunk must be positive")
        # Number of wavelength chunks
        self.nwavelength_chunks = (self.opacities.nwavelength + self.nwavelengths_per_chunk - 1) // self.nwavelengths_per_chunk

    def _setup_atmosphere(self, atm: Atmosphere, planet: Planet):
        "Setup atmospheric grid."
        self.atmosphere.setup(atm._atm, planet)

    def _setup_clouds(self, clouds: Clouds):
        "Validate then set cloud properties"

        scale_factor_cloudy = 1.0
        scale_factor_clear = np.nan

        if clouds is None:
            self.clouds = None
            return scale_factor_cloudy, scale_factor_clear

        # Validate
        _validate_clouds(clouds, self.atmosphere.layer_pressures_cgs, self.opacities.wavelength)

        # Set clouds
        self.clouds = clouds

        # Determine some scale factors for patchy clouds.
        if self.clouds.do_holes:
            scale_factor_cloudy = 1.0 - self.clouds.fhole
            scale_factor_clear = self.clouds.fhole

        return scale_factor_cloudy, scale_factor_clear

    def _setup_surface(self, surface: Surface):
        "Validate then set the surface boundary condition"

        if surface is None:
            self.surface = Surface(hard_surface=False) # default
            return

        if not isinstance(surface, Surface):
            raise TypeError(f"surface must be a Surface or None, got {type(surface)!r}")

        if not np.isscalar(surface.reflectance):
            _validate_wavelength(surface.wavelength, self.opacities.wavelength)

        self.surface = surface

    def _setup_star(self, star: Star):

        if star is None:
            self.star = None
            return
        
        if not isinstance(star, Star):
            raise TypeError(f"star must be a Star or None, got {type(star)!r}")
        
        if star.wavelength is not None:
            _validate_wavelength(star.wavelength, self.opacities.wavelength)

        self.star = star

    def _prepare_interpolation(self):
        "Prepared interpolation for computing opacities"
        self.opacities.prepare_interpolation(self.atmosphere, self.clouds, self.nwavelengths_per_chunk)

    def _compute_opacity(self, ind_wv0, ind_wv1):
        "Compute the opacity of the atmosphere."
        self.opacities.compute_opacity(
            self.atmosphere,
            self.settings,
            self.clouds,
            self.surface,
            ind_wv0,
            ind_wv1,
            self.opacities_result,
        )

    def _adjust_opacity_for_clearsky(self, ind_wv0, ind_wv1):
        "Adjust opacity for clear-sky portin of atmosphere"
        self.opacities.adjust_opacity_for_clearsky(self.clouds, self.settings, ind_wv0, ind_wv1, self.opacities_result)

    def _radiate_thermal(self, ind_wv0, ind_wv1, scale_factor):

        chunk_width = ind_wv1 - ind_wv0

        # RT
        get_thermal_1d(
            self.thermal,
            self.atmosphere.nlayers,
            chunk_width,
            self.opacities_result.ngauss,
            self.phase.ubar1.shape[0],
            self.phase.ubar1.shape[1],
            self.opacities_result.ck_weights,
            self.phase.gweight,
            self.phase.tweight,
            self.opacities_result.wavelength_um[:chunk_width],
            self.opacities_result.dtau[:chunk_width, :, :],
            self.opacities_result.w0[:chunk_width, :, :],
            self.opacities_result.cosb[:chunk_width, :, :],
            self.atmosphere.level_temperatures,
            self.atmosphere.level_pressures_cgs,
            self.phase.ubar1,
            self.opacities_result.surf_reflect[:chunk_width],
            self.surface.hard_surface,
            self.opacities_result.spectrum[:chunk_width],
        )

        # Save chunk
        self.thermal_result.wavelength_um[ind_wv0:ind_wv1] = self.opacities_result.wavelength_um[:chunk_width]
        spectrum = self.opacities_result.spectrum[:chunk_width]
        spectrum *= scale_factor
        self.thermal_result.thermal[ind_wv0:ind_wv1] += spectrum
    
    def _radiate_reflected(self, ind_wv0, ind_wv1, scale_factor):

        if not np.isfinite(self.atmosphere.semimajor) or self.atmosphere.semimajor <= 0.0:
            raise ValueError(
                f"reflected light requires a finite positive semimajor axis, got {self.atmosphere.semimajor}"
            )
        
        chunk_width = ind_wv1 - ind_wv0

        # RT
        get_reflected_1d(
            self.reflected,
            self.atmosphere.nlayers,
            chunk_width,
            self.opacities_result.ngauss,
            self.phase.effective_numg,
            self.phase.effective_numt,
            self.opacities_result.ck_weights,
            self.phase.gweight,
            self.phase.tweight,
            self.opacities_result.dtau_dedd[:chunk_width, :, :],
            self.opacities_result.tau_dedd[:chunk_width, :, :],
            self.opacities_result.w0_dedd[:chunk_width, :, :],
            self.opacities_result.cosb_dedd[:chunk_width, :, :],
            self.opacities_result.gcos2[:chunk_width, :],
            self.opacities_result.ftau_cld[:chunk_width, :],
            self.opacities_result.ftau_ray[:chunk_width, :],
            self.opacities_result.dtau[:chunk_width, :, :],
            self.opacities_result.tau[:chunk_width, :, :],
            self.opacities_result.w0_no_raman[:chunk_width, :, :],
            self.opacities_result.cosb[:chunk_width, :, :],
            self.opacities_result.surf_reflect[:chunk_width],
            self.phase.ubar0,
            self.phase.ubar1,
            self.phase.cos_theta,
            self.settings.single_phase,
            self.settings.multi_phase,
            self.settings.frac_a,
            self.settings.frac_b,
            self.settings.frac_c,
            self.settings.constant_back,
            self.settings.constant_forward,
            1,
            0,
            self.settings.toon_coefficients,
            0.0,
            self.opacities_result.spectrum[:chunk_width],
        )

        # Save chunk
        self.reflected_result.wavelength_um[ind_wv0:ind_wv1] = self.opacities_result.wavelength_um[:chunk_width]
        spectrum = self.opacities_result.spectrum[:chunk_width]
        spectrum *= scale_factor
        self.reflected_result.albedo[ind_wv0:ind_wv1] += spectrum

    def _radiate_transmission(self, ind_wv0, ind_wv1, scale_factor):

        if self.star is None:
            raise ValueError("transmission requires a star with a finite radius")
        if not np.isfinite(self.star.radius) or self.star.radius <= 0.0:
            raise ValueError(
                f"transmission requires a finite positive stellar radius, got {self.star.radius}"
            )

        chunk_width = ind_wv1 - ind_wv0

        get_transit_1d(
            self.atmosphere.nlayers + 1,
            chunk_width,
            self.opacities_result.ngauss,
            self.atmosphere.level_z,
            self.atmosphere.level_dz,
            self.star.radius,
            self.atmosphere.layer_mubar,
            KB_CGS,
            AMU_CGS,
            self.atmosphere.level_pressures_cgs,
            self.atmosphere.level_temperatures,
            self.atmosphere.layer_colden,
            self.opacities_result.dtau[:chunk_width, :, :],
            self.opacities_result.ck_weights,
            self.opacities_result.spectrum[:chunk_width],
        )

        # Save chunk
        self.transmission_result.wavelength_um[ind_wv0:ind_wv1] = self.opacities_result.wavelength_um[:chunk_width]
        spectrum = self.opacities_result.spectrum[:chunk_width]
        spectrum *= scale_factor
        self.transmission_result.rprs2[ind_wv0:ind_wv1] += spectrum

    def _zero_result(self, calculation):
        if calculation == 'thermal':
            self.thermal_result._ensure(self.opacities.nwavelength)
            self.thermal_result.thermal[:] = 0.0
        elif calculation == 'reflected':
            self.reflected_result._ensure(self.opacities.nwavelength)
            self.reflected_result.albedo[:] = 0.0
        elif calculation == 'transmission':
            self.transmission_result._ensure(self.opacities.nwavelength)
            self.transmission_result.rprs2[:] = 0.0

    def _radiate(self, ind_wv0, ind_wv1, calculation, scale_factor):
        if calculation == 'thermal':
            self._radiate_thermal(ind_wv0, ind_wv1, scale_factor)
        elif calculation == 'reflected':
            self._radiate_reflected(ind_wv0, ind_wv1, scale_factor)
        elif calculation == 'transmission':
            self._radiate_transmission(ind_wv0, ind_wv1, scale_factor)

    def _post_process(self, calculation):
        if calculation == 'thermal':
            if self.star is None or self.star.wavelength is None or self.star.spectrum is None:
                # No star provided
                self.thermal_result.fpfs[:] = np.nan
                return
            fpfs_scale = (self.atmosphere.radius / self.star.radius) ** 2.0
            self.thermal_result.fpfs[:] = self.thermal_result.thermal[:]
            self.thermal_result.fpfs[:] /= self.star.spectrum
            self.thermal_result.fpfs[:] *= fpfs_scale
        elif calculation == 'reflected':
            fpfs_scale = (self.atmosphere.radius / self.atmosphere.semimajor) ** 2.0
            self.reflected_result.fpfs[:] = self.reflected_result.albedo[:] * fpfs_scale
        elif calculation == 'transmission':
            pass
            
    def _get_result(self, calculation):
        if calculation == 'thermal':
            return self.thermal_result
        elif calculation == 'reflected':
            return self.reflected_result
        elif calculation == 'transmission':
            return self.transmission_result
    
    def spectrum(
        self,
        atm: Atmosphere,
        planet: Planet,
        clouds: Clouds = None,
        surface: Surface = None,
        star: Star = None,
        calculation='thermal',
        nwavelengths_per_chunk=10_000,
    ):

        if calculation not in ['thermal', 'reflected', 'transmission']:
            raise ValueError(
                f"calculation must be 'thermal', 'reflected', or 'transmission', got {calculation!r}"
            )
        if calculation == 'transmission' and star is None:
            raise ValueError("transmission calculations require a Star with a finite radius")
        
        # Set wavelength chunking
        self._set_wavelength_chunks(nwavelengths_per_chunk)

        # Setup the atmospheric grid.
        self._setup_atmosphere(atm, planet)

        # Setup clouds
        scale_factor_cloudy, scale_factor_clear = self._setup_clouds(clouds)

        # Setup surface boundary condition
        self._setup_surface(surface)

        # Setup star
        self._setup_star(star)

        # Prepare interpolation
        self._prepare_interpolation()

        # Allocate and zero-out result
        self._zero_result(calculation)

        # Loop over each wavelength chunk
        for i in range(self.nwavelength_chunks):
            ind_wv0 = i * self.nwavelengths_per_chunk
            ind_wv1 = min(ind_wv0 + self.nwavelengths_per_chunk, self.opacities.nwavelength)
            
            # Compute opacity for the wavelength chunk
            self._compute_opacity(ind_wv0, ind_wv1)

            # Do the RT for the wavelength chunk
            self._radiate(ind_wv0, ind_wv1, calculation, scale_factor_cloudy)

            # If patchy clouds
            if self.clouds is not None and self.clouds.do_holes:

                # Adjust opacities for clear-sky portion
                self._adjust_opacity_for_clearsky(ind_wv0, ind_wv1)

                # Do RT for clear-sky portion
                self._radiate(ind_wv0, ind_wv1, calculation, scale_factor_clear)

        # Do any needed post-processing
        self._post_process(calculation)

        return self._get_result(calculation)
