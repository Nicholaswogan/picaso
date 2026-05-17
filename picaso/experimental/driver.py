# Comment below helps ignore linting false-positives.
# type: ignore

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import sqlite3
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
        if self.raman not in (1, 2):
            raise ValueError(f"raman must be 1 or 2, got {self.raman}")

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
    cia_scale: nb.float64[:]
    molecular_pair_map: nb.int64[:,:]
    molecular_pair_pindex: nb.int64[:]
    molecular_pair_tindex: nb.int64[:]
    continuum_temperature_map: nb.int64[:]
    continuum_temperature_load_idx: nb.int64[:]
    molecular_block: nb.float64[:,:]
    continuum_block: nb.float64[:,:]
    molecular_raw_u16: nb.uint16[:]
    molecular_raw_f32: nb.float32[:]
    continuum_raw_u16: nb.uint16[:]
    continuum_raw_f32: nb.float32[:]
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
        self.cia_scale = np.empty(nlayers, dtype=np.float64)
        self.molecular_pair_map = np.full((npressure, ntemperature), -1, dtype=np.int64)
        self.molecular_pair_pindex = np.empty(npressure * ntemperature, dtype=np.int64)
        self.molecular_pair_tindex = np.empty(npressure * ntemperature, dtype=np.int64)
        self.continuum_temperature_map = np.full(ncontinuum_temperature, -1, dtype=np.int64)
        self.continuum_temperature_load_idx = np.empty(ncontinuum_temperature, dtype=np.int64)
        self.molecular_block = np.empty((npressure * ntemperature, nwavelengths_per_chunk), dtype=np.float64)
        self.continuum_block = np.empty((ncontinuum_temperature, nwavelengths_per_chunk), dtype=np.float64)
        self.molecular_raw_u16 = np.empty(nwavelengths_per_chunk, dtype=np.uint16)
        self.molecular_raw_f32 = np.empty(nwavelengths_per_chunk, dtype=np.float32)
        self.continuum_raw_u16 = np.empty(nwavelengths_per_chunk, dtype=np.uint16)
        self.continuum_raw_f32 = np.empty(nwavelengths_per_chunk, dtype=np.float32)
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
        if "storage_format" not in self.file.attrs:
            raise ValueError(f"{self.opacity_filename!r} is missing required file-level storage_format attr")
        self.storage_format = _decode_hdf5_string(self.file.attrs["storage_format"])
        if self.storage_format not in {"log10_uint16", "log10_float32"}:
            raise ValueError(
                f"unsupported file-level storage_format {self.storage_format!r}; "
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
        for i, name in enumerate(self.continuum_names):
            dataset = self._continuum_group[name]
            storage_format = self.storage_format
            if storage_format == "log10_uint16":
                if "y_min" not in dataset.attrs or "y_max" not in dataset.attrs:
                    raise ValueError(f"continuum dataset {name!r} is missing required y_min/y_max attrs")
                self.continuum_y_min[i] = float(dataset.attrs["y_min"])
                self.continuum_y_max[i] = float(dataset.attrs["y_max"])
            else:
                self.continuum_y_min[i] = np.nan
                self.continuum_y_max[i] = np.nan

        self.pressure.flags.writeable = False
        self.temperature.flags.writeable = False
        self.wavelength.flags.writeable = False
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
        cache_key,
        storage_code,
        y_min,
        y_max,
        post_decode_log10_factor,
        raw_chunk_buffer,
        raw_full_buffer,
        out_row,
        source_slice=None,
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
            if source_slice is None:
                raise ValueError("source_slice must be provided when opacity caching is enabled")
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
        species_name,
        i_molecular,
        source_wv0,
        source_wv1,
        full_source_sel,
        source_slice,
        chunk_width,
        storage_code,
        molecular_raw_buffer,
        molecular_full_raw_buffer,
        block,
    ):
        source_sel = np.s_[source_wv0:source_wv1] if self.cache is None else np.s_[full_source_sel]
        for row_id in range(self.workspace.molecular_npairs):
            ip = self.workspace.molecular_pair_pindex[row_id]
            it = self.workspace.molecular_pair_tindex[row_id]
            self._read_and_decode_opacity(
                dataset,
                np.s_[ip, it, source_sel],
                (dataset.name, ip, it),
                storage_code,
                self.molecular_y_min[i_molecular],
                self.molecular_y_max[i_molecular],
                0.0,
                molecular_raw_buffer[:chunk_width],
                molecular_full_raw_buffer,
                block[row_id, :chunk_width],
                source_slice=source_slice,
            )

    def _load_continuum_block(
        self,
        dataset,
        continuum_name,
        i_continuum,
        source_wv0,
        source_wv1,
        full_source_sel,
        source_slice,
        chunk_width,
        storage_code,
        continuum_raw_buffer,
        continuum_full_raw_buffer,
        block,
    ):
        source_sel = np.s_[source_wv0:source_wv1] if self.cache is None else np.s_[full_source_sel]
        for row_id in range(self.workspace.continuum_nrows):
            it = self.workspace.continuum_temperature_load_idx[row_id]
            self._read_and_decode_opacity(
                dataset,
                np.s_[it, source_sel],
                (dataset.name, it),
                storage_code,
                self.continuum_y_min[i_continuum],
                self.continuum_y_max[i_continuum],
                np.log10(CIA_AMAGAT_TO_MOLECULE_CM),
                continuum_raw_buffer[:chunk_width],
                continuum_full_raw_buffer,
                block[row_id, :chunk_width],
                source_slice=source_slice,
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
        chunk_width = ind_wv1 - ind_wv0
        opacities_result._ensure(atmosphere.nlayers, self.workspace.nwavelengths_per_chunk)
        storage_code = 0 if self.storage_format == "log10_uint16" else 1
        if storage_code == 0:
            molecular_raw_buffer = self.workspace.molecular_raw_u16
            continuum_raw_buffer = self.workspace.continuum_raw_u16
            molecular_full_raw_buffer = self.workspace.raw_full_u16
            continuum_full_raw_buffer = self.workspace.raw_full_u16
        else:
            molecular_raw_buffer = self.workspace.molecular_raw_f32
            continuum_raw_buffer = self.workspace.continuum_raw_f32
            molecular_full_raw_buffer = self.workspace.raw_full_f32
            continuum_full_raw_buffer = self.workspace.raw_full_f32

        # wavelengths and surface
        opacities_result.wavelength_um[:chunk_width] = self.wavelength[ind_wv0:ind_wv1]
        if np.isscalar(surface.reflectance):
            opacities_result.surf_reflect[:chunk_width] = float(surface.reflectance)
        else:
            opacities_result.surf_reflect[:chunk_width] = surface.reflectance[ind_wv0:ind_wv1]

        # Line by line
        taugas = opacities_result.taugas[:chunk_width, :]
        taugas[:] = 0.0
        source_wv0 = self.wavelength_source_indices[ind_wv0]
        source_wv1 = self.wavelength_source_indices[ind_wv1 - 1] + 1
        full_source_sel = np.s_[self.wavelength_source_indices[0]: self.wavelength_source_indices[-1] + 1]
        source_slice = slice(ind_wv0, ind_wv1)
        for i_species in range(atmosphere.nspecies):
            species_name = str(atmosphere.species_names[i_species])
            if species_name not in self.molecular_name_to_index:
                continue

            i_molecular = self.molecular_name_to_index[species_name]
            block = self.workspace.molecular_block
            dataset = self._molecular_group[species_name]
            self._load_molecular_block(
                dataset,
                species_name,
                i_molecular,
                source_wv0,
                source_wv1,
                full_source_sel,
                source_slice,
                chunk_width,
                storage_code,
                molecular_raw_buffer,
                molecular_full_raw_buffer,
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

        # CIA & continuum
        atmosphere_name_to_index = {str(name): i for i, name in enumerate(atmosphere.species_names)}

        for i_continuum, continuum_name in enumerate(self.continuum_names):
            if "-" not in continuum_name:
                continue

            species_left, species_right = continuum_name.split("-", 1)
            if species_left not in atmosphere_name_to_index or species_right not in atmosphere_name_to_index:
                continue

            i_left = atmosphere_name_to_index[species_left]
            i_right = atmosphere_name_to_index[species_right]
            block = self.workspace.continuum_block
            dataset = self._continuum_group[continuum_name]
            _fill_cia_scale_workspace(
                atmosphere,
                i_left,
                i_right,
                self.workspace,
            )
            self._load_continuum_block(
                dataset,
                continuum_name,
                i_continuum,
                source_wv0,
                source_wv1,
                full_source_sel,
                source_slice,
                chunk_width,
                storage_code,
                continuum_raw_buffer,
                continuum_full_raw_buffer,
                block,
            )
            _accumulate_cia_tau(
                block[:self.workspace.continuum_nrows, :chunk_width],
                self.workspace.cia_scale,
                self.workspace.continuum_temperature_ind0,
                self.workspace.continuum_temperature_ind1,
                self.workspace.continuum_temperature_weight,
                taugas,
            )

        # Rayleigh
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

        # Raman
        compute_raman(
            settings.raman,
            opacities_result.wavelength_um[:chunk_width],
            opacities_result.raman_factor[:chunk_width],
        )

        # Clouds
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

        # All of these will ultimately be inputs
        _finish_compute_opacity(
            opacities_result,
            chunk_width,
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
def _fill_cia_scale_workspace(atmosphere, i_left_species, i_right_species, workspace):
    nlayers = atmosphere.nlayers
    for i in range(nlayers):
        workspace.cia_scale[i] = (
            atmosphere.layer_densities[i_left_species, i]
            * atmosphere.layer_densities[i_right_species, i]
            * atmosphere.layer_dz[i]
        )


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
def _accumulate_cia_tau(block, continuum_scale_row, t_ind0, t_ind1, t_weight, tau_out):
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
def _finish_compute_opacity(result: RadtranOpacitiesResult, chunk_width, stream, delta_eddington, fthin_cld):
    
    for iw in range(chunk_width):
        running_tau = 0.0
        running_tau_dedd = 0.0
        result.tau[iw, 0] = 0.0
        result.tau_dedd[iw, 0] = 0.0

        for i in range(result.nlayers):
            # Unpack to scalars
            taugas = result.taugas[iw,i]
            tauray = result.tauray[iw,i]
            taucld = result.taucld[iw,i]
            w0_cld = result.w0_cld[iw,i]
            g0_cld = result.g0_cld[iw,i]
            raman_factor = result.raman_factor[iw]

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
            result.dtau[iw,i] = dtau
            result.ftau_cld[iw,i] = ftau_cld
            result.ftau_ray[iw,i] = ftau_ray
            result.gcos2[iw,i] = gcos2
            result.cosb[iw,i] = cosb
            result.w0[iw,i] = w0
            result.w0_no_raman[iw,i] = w0_no_raman
            result.tau[iw,i+1] = running_tau
            result.w0_dedd[iw,i] = w0_dedd
            result.cosb_dedd[iw,i] = cosb_dedd
            result.dtau_dedd[iw,i] = dtau_dedd
            result.tau_dedd[iw, i+1] = running_tau_dedd
        
@nb.experimental.jitclass
class RadtranOpacitiesResult:

    # Dimensions
    nlayers : nb.int64
    nwavelengths_per_chunk : nb.int64

    wavelength_um : nb.float64[:]
    surf_reflect : nb.float64[:]
    taugas : nb.float64[:,:]
    tauray : nb.float64[:,:]
    taucld : nb.float64[:,:]
    w0_cld : nb.float64[:,:]
    g0_cld : nb.float64[:,:]

    dtau_dedd : nb.float64[:,:]
    tau_dedd : nb.float64[:,:]
    w0_dedd : nb.float64[:,:]
    cosb_dedd : nb.float64[:,:]

    ftau_cld : nb.float64[:,:]
    ftau_ray : nb.float64[:,:]
    gcos2 : nb.float64[:,:]

    dtau : nb.float64[:,:]
    tau : nb.float64[:,:]
    w0 : nb.float64[:,:]
    w0_no_raman : nb.float64[:,:]
    cosb : nb.float64[:,:]
    raman_factor : nb.float64[:]

    spectrum : nb.float64[:]

    def __init__(self):
        self._allocate(0, 0)

    def _allocate(self, nlayers, nwavelengths_per_chunk):
        self.nlayers = nlayers
        self.nwavelengths_per_chunk = nwavelengths_per_chunk

        self.wavelength_um = np.empty(nwavelengths_per_chunk, dtype=np.float64)
        self.surf_reflect = np.empty(nwavelengths_per_chunk, dtype=np.float64)
        self.taugas = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.tauray = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.taucld = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.w0_cld = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.g0_cld = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)

        self.dtau_dedd = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.tau_dedd = np.empty((nwavelengths_per_chunk, nlayers+1), dtype=np.float64)
        self.w0_dedd = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.cosb_dedd = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)

        self.ftau_cld = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.ftau_ray = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.gcos2 = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)

        self.dtau = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.tau = np.empty((nwavelengths_per_chunk, nlayers+1), dtype=np.float64)
        self.w0 = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.w0_no_raman = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.cosb = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.raman_factor = np.empty(nwavelengths_per_chunk, dtype=np.float64)

        self.spectrum = np.empty(nwavelengths_per_chunk, dtype=np.float64)

    def _ensure(self, nlayers, nwavelengths_per_chunk):
        if nlayers != self.nlayers or nwavelengths_per_chunk != self.nwavelengths_per_chunk:
            self._allocate(nlayers, nwavelengths_per_chunk)

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
        self.opacities = RadtranOpacities(opacity_filename, wavelength_range, opacity_cache_size_limit)
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

        if clouds is None:
            self.clouds = None
            return

        _validate_clouds(clouds, self.atmosphere.layer_pressures_cgs, self.opacities.wavelength)
        self.clouds = clouds

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
            self.phase.ubar1.shape[0],
            self.phase.ubar1.shape[1],
            self.phase.gweight,
            self.phase.tweight,
            self.opacities_result.wavelength_um[:chunk_width],
            self.opacities_result.dtau[:chunk_width, :],
            self.opacities_result.w0[:chunk_width, :],
            self.opacities_result.cosb[:chunk_width, :],
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
            self.phase.effective_numg,
            self.phase.effective_numt,
            self.phase.gweight,
            self.phase.tweight,
            self.opacities_result.dtau_dedd[:chunk_width, :],
            self.opacities_result.tau_dedd[:chunk_width, :],
            self.opacities_result.w0_dedd[:chunk_width, :],
            self.opacities_result.cosb_dedd[:chunk_width, :],
            self.opacities_result.gcos2[:chunk_width, :],
            self.opacities_result.ftau_cld[:chunk_width, :],
            self.opacities_result.ftau_ray[:chunk_width, :],
            self.opacities_result.dtau[:chunk_width, :],
            self.opacities_result.tau[:chunk_width, :],
            self.opacities_result.w0[:chunk_width, :],
            self.opacities_result.cosb[:chunk_width, :],
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
            self.atmosphere.level_z,
            self.atmosphere.level_dz,
            self.star.radius,
            self.atmosphere.layer_mubar,
            KB_CGS,
            AMU_CGS,
            self.atmosphere.level_pressures_cgs,
            self.atmosphere.level_temperatures,
            self.atmosphere.layer_colden,
            self.opacities_result.dtau[:chunk_width, :],
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
        self._setup_clouds(clouds)

        # Setup surface boundary condition
        self._setup_surface(surface)

        # Setup star
        self._setup_star(star)

        # Determine some scale factors for patchy clouds, if needed
        if self.clouds is not None and self.clouds.do_holes:
            scale_factor_cloudy = 1.0 - self.clouds.fhole
            scale_factor_clear = self.clouds.fhole
        else:
            scale_factor_cloudy = 1.0
            scale_factor_clear = np.nan

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

def _decode_sqlite_array(cell):
    if isinstance(cell, np.ndarray):
        return np.asarray(cell)
    if isinstance(cell, (bytes, bytearray, memoryview)):
        with io.BytesIO(cell) as bio:
            bio.seek(0)
            return np.load(bio, allow_pickle=False)
    return np.asarray(cell)


def _encode_log10_uint16_block(opacity, floor):
    log_arr = np.log10(np.maximum(np.asarray(opacity, dtype=np.float64), floor))
    y_min = float(np.min(log_arr))
    y_max = float(np.max(log_arr))
    if y_max == y_min:
        encoded = np.zeros_like(log_arr, dtype=np.uint16)
    else:
        scaled = (log_arr - y_min) / (y_max - y_min)
        encoded = np.round(scaled * np.iinfo(np.uint16).max).astype(np.uint16)
    return encoded, y_min, y_max


def _encode_log10_float32_block(opacity, floor):
    log_arr = np.log10(np.maximum(np.asarray(opacity, dtype=np.float64), floor))
    return log_arr.astype(np.float32)


def _decode_hdf5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_hdf5_string(value.item())
        return [_decode_hdf5_string(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_decode_hdf5_string(item) for item in value]
    return str(value)


def _continuum_name(name):
    if "-" in name:
        return name

    # Common CIA species tokens that appear in PICASO continuum tables.
    # We prefer explicit `A-B` names in the new file format.
    tokens = (
        "C2H6",
        "C2H2",
        "CH4",
        "CO2",
        "H2O",
        "NH3",
        "H2",
        "He",
        "CO",
        "N2",
        "O2",
        "O3",
        "H",
        "bf",
        "ff",
        "e-",
    )
    token_set = set(tokens)

    for split in range(1, len(name)):
        left = name[:split]
        right = name[split:]
        if left in token_set and right in token_set:
            return f"{left}-{right}"

    raise ValueError(
        f"Could not infer a delimited continuum name from {name!r}; "
        "please update the parser with an explicit mapping."
    )


def convert_sqlite_to_hdf5(
    input_db,
    output_hdf5,
    compression='lzf',
    shuffle=True,
    storage_format="log10_uint16",
    chunks=(1, 1, 4096),
    molecular_log10_floor=1e-50,
    continuum_log10_floor=1e-100,
    verbose=True,
):
    """Convert a square SQLite opacity database into the new shared-grid HDF5 layout.

    The SQLite database must define a single shared wavelength grid in ``header``
    and square molecular / continuum opacity blocks:

    - molecular rows must form a complete ``(nP, nT, nW)`` cube per species
    - continuum rows must form a complete ``(nT, nW)`` block per species
      on a shared continuum temperature grid, which may differ from the
      molecular temperature grid

    Parameters
    ----------
    input_db : str or Path
        Path to the SQLite database.
    output_hdf5 : str or Path
        Destination HDF5 file path.
    compression : str or None, optional
        HDF5 compression filter for opacity datasets.
    shuffle : bool, optional
        Whether to enable HDF5 shuffle filtering.
    storage_format : {'log10_uint16', 'log10_float32'}, optional
        Storage encoding for the opacity blocks.
    chunks : tuple or int, optional
        HDF5 chunk shape. For molecular blocks the expected dimensionality is
        three; for continuum blocks two.
    molecular_log10_floor : float, optional
        Floor applied before taking log10 for molecular opacities.
    continuum_log10_floor : float, optional
        Floor applied before taking log10 for continuum opacities.
    verbose : bool, optional
        Print progress messages.
    """
    input_db = Path(input_db)
    output_hdf5 = Path(output_hdf5)

    if storage_format not in {"log10_uint16", "log10_float32"}:
        raise ValueError(f"Unsupported storage_format: {storage_format!r}")

    sqlite3.register_converter("array", lambda text: np.load(io.BytesIO(text), allow_pickle=False))

    def _get_chunks(ndim, data_shape):
        if isinstance(chunks, int):
            return (1,) * (ndim - 1) + (min(int(chunks), data_shape[-1]),)
        if len(chunks) == ndim:
            return tuple(min(int(c), int(s)) for c, s in zip(chunks, data_shape))
        if ndim == 3 and len(chunks) == 2:
            return (min(1, data_shape[0]), min(int(chunks[0]), data_shape[1]), min(int(chunks[1]), data_shape[2]))
        if ndim == 2 and len(chunks) == 2:
            return tuple(min(int(c), int(s)) for c, s in zip(chunks, data_shape))
        raise ValueError(f"Unsupported chunks specification {chunks!r} for array with ndim={ndim}")

    conn = sqlite3.connect(str(input_db), detect_types=sqlite3.PARSE_DECLTYPES)
    try:
        cur = conn.cursor()

        cur.execute(
            "SELECT pressure_unit, temperature_unit, wavenumber_grid, continuum_unit, molecular_unit FROM header LIMIT 1"
        )
        header_row = cur.fetchone()
        if header_row is None:
            raise RuntimeError(f"{input_db} does not contain a header row.")

        pressure_unit, temperature_unit, wavenumber_grid, continuum_unit, molecular_unit = header_row
        wavenumber_grid = np.asarray(_decode_sqlite_array(wavenumber_grid), dtype=np.float64)
        wavelength_grid = 1.0e4 / wavenumber_grid
        wl_sort = np.argsort(wavelength_grid)
        wavelength_grid = wavelength_grid[wl_sort]

        cur.execute("SELECT DISTINCT molecule FROM molecular ORDER BY molecule")
        molecular_names = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT DISTINCT molecule FROM continuum ORDER BY molecule")
        continuum_source_names = [row[0] for row in cur.fetchall()]
        continuum_names = [_continuum_name(name) for name in continuum_source_names]

        if not molecular_names:
            raise RuntimeError("No molecular species found in SQLite database.")

        if verbose:
            print(
                f"Writing {output_hdf5} with {len(molecular_names)} molecular and "
                f"{len(continuum_names)} continuum species"
            )

        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output_hdf5, "w") as f:
            f.attrs["format_version"] = "1.0"
            f.attrs["opacity_type"] = "molecular+continuum"
            f.attrs["storage_format"] = storage_format

            header = f.create_group("header")
            header.create_dataset("molecular_names", data=np.asarray(molecular_names, dtype=object), dtype=string_dtype)
            header.create_dataset("continuum_names", data=np.asarray(continuum_names, dtype=object), dtype=string_dtype)
            header.attrs["pressure_unit"] = str(pressure_unit)
            header.attrs["temperature_unit"] = str(temperature_unit)
            header.attrs["wavelength_unit"] = "micron"
            header.attrs["molecular_unit"] = str(molecular_unit)
            header.attrs["continuum_unit"] = str(continuum_unit)
            header.attrs["molecular_log10_floor"] = float(molecular_log10_floor)
            header.attrs["continuum_log10_floor"] = float(continuum_log10_floor)

            molecular_group = f.create_group("molecular")
            base_pressures = None
            base_temperatures = None
            base_nw = None
            molecular_chunks = None
            total_molecules = len(molecular_names)
            for i_molecule, name in enumerate(molecular_names, start=1):
                if verbose:
                    print(f"[molecular {i_molecule}/{total_molecules}] Writing {name}")
                cur.execute(
                    "SELECT ptid, pressure, temperature, opacity FROM molecular WHERE molecule = ? ORDER BY ptid",
                    (name,),
                )
                rows = cur.fetchall()
                if not rows:
                    raise RuntimeError(f"Molecular species {name!r} has no rows.")

                pressures = np.asarray([r[1] for r in rows], dtype=np.float64)
                temperatures = np.asarray([r[2] for r in rows], dtype=np.float64)
                row_arrays = [np.asarray(_decode_sqlite_array(r[3]), dtype=np.float64).ravel() for r in rows]
                row_lengths = {arr.size for arr in row_arrays}
                if len(row_lengths) != 1:
                    raise RuntimeError(
                        f"Molecular species {name!r} has non-square opacity rows: {sorted(row_lengths)}"
                    )
                nw = row_lengths.pop()

                unique_pressures = np.unique(pressures)
                unique_temperatures = np.unique(temperatures)
                nP = unique_pressures.size
                nT = unique_temperatures.size
                if nP * nT != len(rows):
                    raise RuntimeError(
                        f"Molecular species {name!r} does not fill a square grid: "
                        f"{nP} pressures x {nT} temperatures != {len(rows)} rows."
                    )

                if base_pressures is None:
                    base_pressures = unique_pressures
                    base_temperatures = unique_temperatures
                    base_nw = nw
                    header.create_dataset("pressure", data=base_pressures.astype(np.float64))
                    header.create_dataset("temperature", data=base_temperatures.astype(np.float64))
                    header.create_dataset("wavelength", data=(1.0e4 / wavenumber_grid)[wl_sort].astype(np.float64))
                    molecular_chunks = _get_chunks(3, (base_pressures.size, base_temperatures.size, base_nw))
                else:
                    if nw != base_nw:
                        raise RuntimeError(
                            f"Molecular species {name!r} has wavelength length {nw}, expected {base_nw}."
                        )
                    if not np.array_equal(unique_pressures, base_pressures):
                        raise RuntimeError(
                            f"Molecular species {name!r} uses a different pressure grid than the first species."
                        )
                    if not np.array_equal(unique_temperatures, base_temperatures):
                        raise RuntimeError(
                            f"Molecular species {name!r} uses a different temperature grid than the first species."
                        )

                p_index = {float(p): i for i, p in enumerate(base_pressures)}
                t_index = {float(t): i for i, t in enumerate(base_temperatures)}
                cube = np.empty((base_pressures.size, base_temperatures.size, base_nw), dtype=np.float64)
                seen = set()
                for (_, p, t, arr) in rows:
                    key = (float(p), float(t))
                    if key in seen:
                        raise RuntimeError(f"Duplicate molecular row for species {name!r} at P={p}, T={t}.")
                    seen.add(key)
                    if key[0] not in p_index or key[1] not in t_index:
                        raise RuntimeError(
                            f"Row for species {name!r} has values outside the shared grid: P={p}, T={t}."
                        )
                    arr = np.asarray(_decode_sqlite_array(arr), dtype=np.float64).ravel()
                    if arr.size != base_nw:
                        raise RuntimeError(
                            f"Molecular species {name!r} has inconsistent wavelength length {arr.size} != {base_nw}."
                        )
                    cube[p_index[key[0]], t_index[key[1]], :] = arr[wl_sort]
                if len(seen) != base_pressures.size * base_temperatures.size:
                    raise RuntimeError(f"Molecular species {name!r} does not contain every P/T combination.")

                if storage_format == "log10_uint16":
                    encoded, y_min, y_max = _encode_log10_uint16_block(cube, molecular_log10_floor)
                else:
                    encoded = _encode_log10_float32_block(cube, molecular_log10_floor)
                dataset = molecular_group.create_dataset(
                    name,
                    data=encoded,
                    compression=compression,
                    shuffle=shuffle,
                    chunks=molecular_chunks,
                )
                dataset.attrs["log10_floor"] = float(molecular_log10_floor)
                if storage_format == "log10_uint16":
                    dataset.attrs["y_min"] = np.float64(y_min)
                    dataset.attrs["y_max"] = np.float64(y_max)

            continuum_group = f.create_group("continuum")
            continuum_temperature_grid = None
            continuum_chunks = None
            total_continuum = len(continuum_names)
            for i_continuum, (source_name, continuum_name) in enumerate(
                zip(continuum_source_names, continuum_names), start=1
            ):
                if verbose:
                    print(f"[continuum {i_continuum}/{total_continuum}] Writing {continuum_name}")
                cur.execute(
                    "SELECT temperature, opacity FROM continuum WHERE molecule = ? ORDER BY temperature",
                    (source_name,),
                )
                rows = cur.fetchall()
                if not rows:
                    raise RuntimeError(f"Continuum species {continuum_name!r} has no rows.")

                temps = np.asarray([r[0] for r in rows], dtype=np.float64)
                row_arrays = [np.asarray(_decode_sqlite_array(r[1]), dtype=np.float64).ravel() for r in rows]
                row_lengths = {arr.size for arr in row_arrays}
                if len(row_lengths) != 1:
                    raise RuntimeError(
                        f"Continuum species {continuum_name!r} has non-square opacity rows: {sorted(row_lengths)}"
                    )
                nw = row_lengths.pop()
                if nw != base_nw:
                    raise RuntimeError(
                        f"Continuum species {continuum_name!r} has wavelength length {nw}, expected {base_nw}."
                    )

                unique_temperatures = np.unique(temps)
                if unique_temperatures.size != len(rows):
                    raise RuntimeError(
                        f"Continuum species {continuum_name!r} does not fill a square temperature grid: "
                        f"{unique_temperatures.size} temperatures != {len(rows)} rows."
                    )
                if continuum_temperature_grid is None:
                    continuum_temperature_grid = unique_temperatures
                    header.create_dataset(
                        "continuum_temperatures", data=continuum_temperature_grid.astype(np.float64)
                    )
                    continuum_chunks = _get_chunks(2, (continuum_temperature_grid.size, base_nw))
                elif not np.array_equal(unique_temperatures, continuum_temperature_grid):
                    raise RuntimeError(
                        f"Continuum species {continuum_name!r} uses a different temperature grid than the other continuum tables."
                    )

                t_index = {float(t): i for i, t in enumerate(continuum_temperature_grid)}
                cube = np.empty((continuum_temperature_grid.size, base_nw), dtype=np.float64)
                seen = set()
                for t, arr in rows:
                    key = float(t)
                    if key in seen:
                        raise RuntimeError(f"Duplicate continuum row for species {continuum_name!r} at T={t}.")
                    seen.add(key)
                    if key not in t_index:
                        raise RuntimeError(
                            f"Row for continuum species {continuum_name!r} has T={t} outside the shared grid."
                        )
                    arr = np.asarray(_decode_sqlite_array(arr), dtype=np.float64).ravel()
                    if arr.size != base_nw:
                        raise RuntimeError(
                            f"Continuum species {continuum_name!r} has inconsistent wavelength length {arr.size} != {base_nw}."
                        )
                    cube[t_index[key], :] = arr[wl_sort]
                if len(seen) != continuum_temperature_grid.size:
                    raise RuntimeError(
                        f"Continuum species {continuum_name!r} does not contain every temperature in the shared grid."
                    )

                if storage_format == "log10_uint16":
                    encoded, y_min, y_max = _encode_log10_uint16_block(cube, continuum_log10_floor)
                else:
                    encoded = _encode_log10_float32_block(cube, continuum_log10_floor)
                dataset = continuum_group.create_dataset(
                    continuum_name,
                    data=encoded,
                    compression=compression,
                    shuffle=shuffle,
                    chunks=continuum_chunks,
                )
                dataset.attrs["log10_floor"] = float(continuum_log10_floor)
                if storage_format == "log10_uint16":
                    dataset.attrs["y_min"] = np.float64(y_min)
                    dataset.attrs["y_max"] = np.float64(y_max)

            if continuum_temperature_grid is None:
                header.create_dataset("continuum_temperatures", data=np.asarray([], dtype=np.float64))

    finally:
        conn.close()
