# Comment below helps ignore linting false-positives.
# type: ignore

from dataclasses import dataclass
from pathlib import Path
import io
import sqlite3

import h5py
import numpy as np
import numba as nb
from numba import typed

from .experimental_fluxes import ThermalResult, ThermalSolver

# cgs constants for the compiled hydrostatic setup
KB_CGS = 1.380649e-16
AMU_CGS = 1.66053906660e-24
G_CGS = 6.67430e-8
M_EARTH_CGS = 5.9722e27
R_EARTH_CGS = 6.371e8
CIA_AMAGAT_TO_MOLECULE_CM = 1.385277e-39

@nb.experimental.jitclass
class Atmosphere_:

    nspecies: nb.int64
    nlayers: nb.int64

    species_names: nb.types.ListType(nb.types.unicode_type)
    species_mu: nb.float64[:]
    pressures: nb.float64[:]
    temperatures: nb.float64[:]
    mixing_ratios: nb.float64[:,:]
    reference_pressure: np.float64

    def __init__(self, species_names, pressures, temperatures, mixing_ratios, reference_pressure):
        
        # Check dimensions
        nspecies, nlayers = mixing_ratios.shape
        if nlayers <= 1:
            raise ValueError("mixing_ratios must have at least two layers")
        if len(species_names) != nspecies:
            raise ValueError(
                "species_names length must match mixing_ratios.shape[0] "
                f"({len(species_names)} != {nspecies})"
            )
        if pressures.shape[0] != nlayers:
            raise ValueError(
                "pressures length must match mixing_ratios.shape[1] "
                f"({pressures.shape[0]} != {nlayers})"
            )
        if temperatures.shape[0] != nlayers:
            raise ValueError(
                "temperatures length must match mixing_ratios.shape[1] "
                f"({temperatures.shape[0]} != {nlayers})"
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
        
        # Check that reference pressure is in pressures
        if reference_pressure < pressures[0] or reference_pressure > pressures[-1]:
            raise ValueError(
                "reference_pressure must lie within the pressure grid "
                f"[{pressures[0]}, {pressures[-1]}], got {reference_pressure}"
            )

        # Normalize mixing ratios so they sum to 1 in each layer.
        for i in range(nlayers):
            layer_sum = np.sum(mixing_ratios[:, i])
            if layer_sum <= 0.0:
                raise ValueError(
                    "each layer must contain at least one nonzero volume mixing ratio; "
                    f"empty layer at index {i}"
                )
            mixing_ratios[:, i] /= layer_sum
            
        # Set attributes
        self.nspecies = nspecies
        self.nlayers = nlayers
        self.species_names = species_names
        self.pressures = pressures
        self.temperatures = temperatures
        self.mixing_ratios = mixing_ratios
        self.reference_pressure = reference_pressure

class Atmosphere:

    def __init__(self, species_names, pressures, temperatures, mixing_ratios, species_mu=None, reference_pressure=1.0e-3):
        # Check inputs.
        if not isinstance(species_names, list):
            raise TypeError(f"species_names must be a list, got {type(species_names)!r}")

        # Initialize most terms
        atm = Atmosphere_(typed.List(species_names), pressures, temperatures, mixing_ratios)

        if species_mu is None:
            # Get mu for each species
            atm.species_mu = np.ones(atm.nspecies) * 2.0  # g/mol
        else:
            atm.species_mu = species_mu

        self._atm = atm
    
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
        
        
class Clouds:
    pass


class Star:
    pass


@nb.experimental.jitclass
class RadtranOpacitiesWorkspace:
    nlayers: nb.int64
    npressure: nb.int64
    ntemperature: nb.int64
    ncontinuum_temperature: nb.int64
    nwavelengths_per_chunk: nb.int64
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
    molecular_block: nb.float64[:,:,:]
    continuum_block: nb.float64[:,:]

    def __init__(self):
        self._allocate(0, 0, 0, 0, 0)

    def _allocate(self, nlayers, npressure, ntemperature, ncontinuum_temperature, nwavelengths_per_chunk):
        self.nlayers = nlayers
        self.npressure = npressure
        self.ntemperature = ntemperature
        self.ncontinuum_temperature = ncontinuum_temperature
        self.nwavelengths_per_chunk = nwavelengths_per_chunk
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
        self.molecular_block = np.empty((npressure, ntemperature, nwavelengths_per_chunk), dtype=np.float64)
        self.continuum_block = np.empty((ncontinuum_temperature, nwavelengths_per_chunk), dtype=np.float64)

    def _ensure(self, nlayers, npressure, ntemperature, ncontinuum_temperature, nwavelengths_per_chunk):
        if (
            nlayers != self.nlayers
            or npressure != self.npressure
            or ntemperature != self.ntemperature
            or ncontinuum_temperature != self.ncontinuum_temperature
            or nwavelengths_per_chunk != self.nwavelengths_per_chunk
        ):
            self._allocate(nlayers, npressure, ntemperature, ncontinuum_temperature, nwavelengths_per_chunk)

class RadtranOpacities:

    def __init__(self, opacity_filename):

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

        self.format_version = _decode_hdf5_string(self.file.attrs.get("format_version", "unknown"))
        self.opacity_type = _decode_hdf5_string(self.file.attrs.get("opacity_type", "unknown"))
        self.storage_format = _decode_hdf5_string(self.file.attrs.get("storage_format", "unknown"))

        self.pressure = np.asarray(self._header["pressure"][:], dtype=np.float64)
        self.temperature = np.asarray(self._header["temperature"][:], dtype=np.float64)
        self.wavelength = np.asarray(self._header["wavelength"][:], dtype=np.float64)
        self.continuum_temperatures = np.asarray(self._header["continuum_temperatures"][:], dtype=np.float64)
        self.molecular_names = [str(name) for name in _decode_hdf5_string(self._header["molecular_names"][:])]
        self.continuum_names = [str(name) for name in _decode_hdf5_string(self._header["continuum_names"][:])]

        self.pressure_unit = _decode_hdf5_string(self._header.attrs.get("pressure_unit", ""))
        self.temperature_unit = _decode_hdf5_string(self._header.attrs.get("temperature_unit", ""))
        self.wavelength_unit = _decode_hdf5_string(self._header.attrs.get("wavelength_unit", ""))
        self.molecular_unit = _decode_hdf5_string(self._header.attrs.get("molecular_unit", ""))
        self.continuum_unit = _decode_hdf5_string(self._header.attrs.get("continuum_unit", ""))
        self.molecular_log10_floor = float(self._header.attrs.get("molecular_log10_floor", 0.0))
        self.continuum_log10_floor = float(self._header.attrs.get("continuum_log10_floor", 0.0))

        self.npressure = int(self.pressure.size)
        self.ntemperature = int(self.temperature.size)
        self.nwavelength = int(self.wavelength.size)
        self.ncontinuum_temperature = int(self.continuum_temperatures.size)
        self.nmolecular = int(len(self.molecular_names))
        self.ncontinuum = int(len(self.continuum_names))
        self.molecular_name_to_index = {name: i for i, name in enumerate(self.molecular_names)}
        self.continuum_name_to_index = {name: i for i, name in enumerate(self.continuum_names)}
        self.molecular_storage_format = np.empty(self.nmolecular, dtype=np.int64)
        self.molecular_log10_floor = np.empty(self.nmolecular, dtype=np.float64)
        self.molecular_y_min = np.empty(self.nmolecular, dtype=np.float64)
        self.molecular_y_max = np.empty(self.nmolecular, dtype=np.float64)
        for i, name in enumerate(self.molecular_names):
            dataset = self._molecular_group[name]
            storage_format = _decode_hdf5_string(dataset.attrs.get("storage_format", self.storage_format))
            self.molecular_storage_format[i] = 0 if storage_format == "log10_uint16" else 1
            self.molecular_log10_floor[i] = float(dataset.attrs.get("log10_floor", self.molecular_log10_floor))
            if storage_format == "log10_uint16":
                self.molecular_y_min[i] = float(dataset.attrs["y_min"])
                self.molecular_y_max[i] = float(dataset.attrs["y_max"])
            else:
                self.molecular_y_min[i] = np.nan
                self.molecular_y_max[i] = np.nan

        self.continuum_storage_format = np.empty(self.ncontinuum, dtype=np.int64)
        self.continuum_log10_floor = np.empty(self.ncontinuum, dtype=np.float64)
        self.continuum_y_min = np.empty(self.ncontinuum, dtype=np.float64)
        self.continuum_y_max = np.empty(self.ncontinuum, dtype=np.float64)
        for i, name in enumerate(self.continuum_names):
            dataset = self._continuum_group[name]
            storage_format = _decode_hdf5_string(dataset.attrs.get("storage_format", self.storage_format))
            self.continuum_storage_format[i] = 0 if storage_format == "log10_uint16" else 1
            self.continuum_log10_floor[i] = float(dataset.attrs.get("log10_floor", self.continuum_log10_floor))
            if storage_format == "log10_uint16":
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

    def prepare_interpolation(self, atmosphere: RadtranAtmosphere, nwavelengths_per_chunk: int):
        self.workspace._ensure(
            atmosphere.nlayers,
            self.npressure,
            self.ntemperature,
            self.ncontinuum_temperature,
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

    def _read_and_decode_opacity_block(
        self,
        dataset,
        ind_wv0,
        ind_wv1,
        storage_code,
        log10_floor,
        y_min,
        y_max,
        post_decode_factor,
        out,
    ):
        dataset.read_direct(out, source_sel=np.s_[:, :, ind_wv0:ind_wv1])
        if storage_code == 0:
            if y_max == y_min:
                out[:] = y_min
            else:
                out *= (y_max - y_min) / np.iinfo(np.uint16).max
                out += y_min
        np.power(10.0, out, out=out)
        out *= post_decode_factor

    def compute_opacity(self, atmosphere: RadtranAtmosphere, ind_wv0: int, ind_wv1: int, opacities_result: RadtranOpacitiesResult):
        chunk_width = ind_wv1 - ind_wv0
        opacities_result._ensure(atmosphere.nlayers, self.workspace.nwavelengths_per_chunk)
        dtau_out = opacities_result.dtau[:chunk_width, :]
        dtau_out[:] = 0.0

        # Set nwavelengths and wavelengths
        opacities_result.nwavelengths = chunk_width
        opacities_result.wavelength_um[:chunk_width] = self.wavelength_um[ind_wv0:ind_wv1]

        # Loop over atmospheric species and accumulate molecular opacities.
        for i_species in range(atmosphere.nspecies):
            species_name = str(atmosphere.species_names[i_species])
            if species_name not in self.molecular_name_to_index:
                continue

            i_molecular = self.molecular_name_to_index[species_name]
            block = self.workspace.molecular_block[:, :, :chunk_width]

            self._read_and_decode_opacity_block(
                self._molecular_group[species_name],
                ind_wv0,
                ind_wv1,
                self.molecular_storage_format[i_molecular],
                self.molecular_log10_floor[i_molecular],
                self.molecular_y_min[i_molecular],
                self.molecular_y_max[i_molecular],
                1.0,
                block,
            )

            _accumulate_molecular_tau(
                block,
                atmosphere.columns[i_species],
                self.workspace.molecular_pressure_ind0,
                self.workspace.molecular_pressure_ind1,
                self.workspace.molecular_pressure_weight,
                self.workspace.molecular_temperature_ind0,
                self.workspace.molecular_temperature_ind1,
                self.workspace.molecular_temperature_weight,
                dtau_out,
            )

        # Loop over CIA continuum opacities.
        atmosphere_name_to_index = {str(name): i for i, name in enumerate(atmosphere.species_names)}

        for i_continuum, continuum_name in enumerate(self.continuum_names):
            if "-" not in continuum_name:
                continue

            species_left, species_right = continuum_name.split("-", 1)
            if species_left not in atmosphere_name_to_index or species_right not in atmosphere_name_to_index:
                continue

            i_left = atmosphere_name_to_index[species_left]
            i_right = atmosphere_name_to_index[species_right]
            block = self.workspace.continuum_block[:, :chunk_width]

            self._read_and_decode_opacity_block(
                self._continuum_group[continuum_name],
                ind_wv0,
                ind_wv1,
                self.continuum_storage_format[i_continuum],
                self.continuum_log10_floor[i_continuum],
                self.continuum_y_min[i_continuum],
                self.continuum_y_max[i_continuum],
                CIA_AMAGAT_TO_MOLECULE_CM,
                block,
            )

            _fill_cia_scale_workspace(
                atmosphere,
                i_left,
                i_right,
                self.workspace,
            )

            _accumulate_cia_tau(
                block,
                self.workspace.cia_scale,
                self.workspace.continuum_temperature_ind0,
                self.workspace.continuum_temperature_ind1,
                self.workspace.continuum_temperature_weight,
                dtau_out,
            )

        # For now w0 and cosb are zero.
        opacities_result.w0[:chunk_width, :] = 0.0
        opacities_result.cosb[:chunk_width, :] = 0.0


    def close(self) -> None:
        if getattr(self, "file", None) is not None:
            self.file.close()
            self.file = None
            self._header = None
            self._molecular_group = None
            self._continuum_group = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


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
def _fill_molecular_interpolation_workspace(atmosphere, pressure_grid, temperature_grid, workspace):
    nlayers = atmosphere.nlayers
    for i in range(nlayers):
        ip0, ip1, pw = _bracket_1d(pressure_grid, atmosphere.pressures[i])
        it0, it1, tw = _bracket_1d(temperature_grid, atmosphere.temperatures[i])

        workspace.molecular_pressure_ind0[i] = ip0
        workspace.molecular_pressure_ind1[i] = ip1
        workspace.molecular_pressure_weight[i] = pw
        workspace.molecular_temperature_ind0[i] = it0
        workspace.molecular_temperature_ind1[i] = it1
        workspace.molecular_temperature_weight[i] = tw


@nb.njit
def _fill_continuum_interpolation_workspace(atmosphere, temperature_grid, workspace):
    nlayers = atmosphere.nlayers
    for i in range(nlayers):
        it0, it1, tw = _bracket_1d(temperature_grid, atmosphere.temperatures[i])
        workspace.continuum_temperature_ind0[i] = it0
        workspace.continuum_temperature_ind1[i] = it1
        workspace.continuum_temperature_weight[i] = tw


@nb.njit
def _fill_cia_scale_workspace(atmosphere, i_left_species, i_right_species, workspace):
    nlayers = atmosphere.nlayers
    for i in range(nlayers):
        workspace.cia_scale[i] = (
            atmosphere.densities[i_left_species, i]
            * atmosphere.densities[i_right_species, i]
            * atmosphere.dz[i]
        )


@nb.njit
def _interp_molecular_opacity(block, i_layer, i_wavelength, p_ind0, p_ind1, p_weight, t_ind0, t_ind1, t_weight):
    ip0 = p_ind0[i_layer]
    ip1 = p_ind1[i_layer]
    pw = p_weight[i_layer]
    it0 = t_ind0[i_layer]
    it1 = t_ind1[i_layer]
    tw = t_weight[i_layer]

    v00 = block[ip0, it0, i_wavelength]
    v10 = block[ip1, it0, i_wavelength]
    v01 = block[ip0, it1, i_wavelength]
    v11 = block[ip1, it1, i_wavelength]

    v0 = (1.0 - pw) * v00 + pw * v10
    v1 = (1.0 - pw) * v01 + pw * v11
    return (1.0 - tw) * v0 + tw * v1


@nb.njit
def _accumulate_molecular_tau(block, columns_row, p_ind0, p_ind1, p_weight, t_ind0, t_ind1, t_weight, tau_out):
    nwavelengths = block.shape[2]
    nlayers = columns_row.shape[0]

    for i in range(nlayers):
        column = columns_row[i]

        for iw in range(nwavelengths):
            tau_out[iw, i] += _interp_molecular_opacity(
                block, i, iw, p_ind0, p_ind1, p_weight, t_ind0, t_ind1, t_weight
            ) * column


@nb.njit
def _interp_continuum_opacity(block, i_layer, i_wavelength, t_ind0, t_ind1, t_weight):
    it0 = t_ind0[i_layer]
    it1 = t_ind1[i_layer]
    tw = t_weight[i_layer]

    v0 = block[it0, i_wavelength]
    v1 = block[it1, i_wavelength]
    return (1.0 - tw) * v0 + tw * v1


@nb.njit
def _accumulate_cia_tau(block, continuum_scale_row, t_ind0, t_ind1, t_weight, tau_out):
    nwavelengths = block.shape[1]
    nlayers = continuum_scale_row.shape[0]

    for i in range(nlayers):
        scale = continuum_scale_row[i]
        for iw in range(nwavelengths):
            tau_out[iw, i] += _interp_continuum_opacity(block, i, iw, t_ind0, t_ind1, t_weight) * scale

@nb.experimental.jitclass
class RadtranOpacitiesResult:

    # Dimensions
    nlayers : nb.int64
    nwavelengths_per_chunk : nb.int64
    nwavelengths : nb.int64
    wavelength_um : nb.float64[:]

    # Layer optical depth, chunk-major.
    dtau : nb.float64[:,:]
    w0 : nb.float64[:,:]
    cosb : nb.float64[:,:]

    def __init__(self):
        self._allocate(0, 0)

    def _allocate(self, nlayers, nwavelengths_per_chunk):
        self.nlayers = nlayers
        self.nwavelengths_per_chunk = nwavelengths_per_chunk
        self.nwavelengths = 0
        self.wavelength_um = np.empty(nwavelengths_per_chunk, dtype=np.float64)
        self.dtau = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.w0 = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)
        self.cosb = np.empty((nwavelengths_per_chunk, nlayers), dtype=np.float64)

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
    pressures: nb.float64[:]
    temperatures: nb.float64[:]
    mixing_ratios: nb.float64[:,:]
    reference_pressure: np.float64

    z: nb.float64[:]
    z_edge: nb.float64[:]
    dz: nb.float64[:]
    gravity: nb.float64[:]
    mubar: nb.float64[:]
    density: nb.float64[:] # total molecules/cm^3
    densities: nb.float64[:,:] # molecules/cm^3 of each species
    columns: nb.float64[:,:]

    def __init__(self):
        self._allocate(0, 0)

    def setup(self, atm: Atmosphere_, planet: Planet):
        # Ensure size
        self._ensure(atm.nlayers, atm.nspecies)

        # Copy over information in Atmosphere_ and Planet.
        self.radius = planet.radius
        self.mass = planet.mass
        self.semimajor = planet.semimajor

        self.nlayers = atm.nlayers
        self.nspecies = atm.nspecies
        self.species_names = atm.species_names
        self.species_mu[:] = atm.species_mu[:]
        self.pressures[:] = atm.pressures[:]
        self.temperatures[:] = atm.temperatures[:]
        self.mixing_ratios[:, :] = atm.mixing_ratios[:, :]
        self.reference_pressure = atm.reference_pressure

        # Mean molecular weight in each cell.
        for i in range(self.nlayers):
            mu = 0.0
            for j in range(self.nspecies):
                mu += self.species_mu[j] * self.mixing_ratios[j, i]
            self.mubar[i] = mu

        # Find the layer just above the reference pressure.
        iref = 0
        while iref < self.nlayers and self.pressures[iref] < self.reference_pressure:
            iref += 1
        if iref == self.nlayers:
            iref = self.nlayers - 1

        # Get planet radius and mass in CGS units.
        planet_radius = self.radius * R_EARTH_CGS
        planet_mass = self.mass * M_EARTH_CGS

        # z is the altitude at the midpoint of each cell, and should decrease
        # with increasing index because pressure increases with index.
        # Start from the layer nearest the reference pressure and integrate
        # hydrostatic balance upward and downward.
        gravity_ref = G_CGS * planet_mass / (planet_radius * planet_radius)
        scale_height_ref = KB_CGS * self.temperatures[iref] / (self.mubar[iref] * AMU_CGS * gravity_ref)
        self.z[iref] = -scale_height_ref * np.log(self.pressures[iref] / self.reference_pressure)

        for i in range(iref - 1, -1, -1):
            gravity_here = G_CGS * planet_mass / ((planet_radius + self.z[i + 1]) * (planet_radius + self.z[i + 1]))
            scale_height = KB_CGS * self.temperatures[i + 1] / (self.mubar[i + 1] * AMU_CGS * gravity_here)
            delta_logp = np.log(self.pressures[i + 1] / self.pressures[i])
            self.z[i] = self.z[i + 1] + scale_height * delta_logp

        for i in range(iref, self.nlayers - 1):
            gravity_here = G_CGS * planet_mass / ((planet_radius + self.z[i]) * (planet_radius + self.z[i]))
            scale_height = KB_CGS * self.temperatures[i] / (self.mubar[i] * AMU_CGS * gravity_here)
            delta_logp = np.log(self.pressures[i + 1] / self.pressures[i])
            self.z[i + 1] = self.z[i] - scale_height * delta_logp

        # Build edges from the midpoint altitude grid.
        self.z_edge[0] = self.z[0] + 0.5 * (self.z[0] - self.z[1])
        for i in range(1, self.nlayers):
            self.z_edge[i] = 0.5 * (self.z[i - 1] + self.z[i])
        self.z_edge[self.nlayers] = self.z[self.nlayers - 1] - 0.5 * (self.z[self.nlayers - 2] - self.z[self.nlayers - 1])

        # Per-cell thickness from adjacent edges.
        for i in range(self.nlayers):
            self.dz[i] = self.z_edge[i] - self.z_edge[i + 1]

        # Gravity at each cell midpoint.
        for i in range(self.nlayers):
            self.gravity[i] = G_CGS * planet_mass / ((planet_radius + self.z[i]) * (planet_radius + self.z[i]))

        # Get densities and columns.
        for i in range(self.nlayers):
            self.density[i] = (self.pressures[i] * 1.0e6) / (KB_CGS * self.temperatures[i])
            for j in range(self.nspecies):
                self.densities[j, i] = self.mixing_ratios[j, i] * self.density[i]
                self.columns[j, i] = self.densities[j, i] * self.dz[i]

    def _allocate(self, nlayers, nspecies):
        self.nlayers = nlayers
        self.nspecies = nspecies
        self.radius = np.nan
        self.mass = np.nan
        self.semimajor = np.nan
        self.species_names = nb.typed.List.empty_list(nb.types.unicode_type)
        self.species_mu = np.empty(nspecies, dtype=np.float64)
        self.pressures = np.empty(nlayers, dtype=np.float64)
        self.temperatures = np.empty(nlayers, dtype=np.float64)
        self.mixing_ratios = np.empty((nspecies, nlayers), dtype=np.float64)
        self.reference_pressure = np.nan
        self.z = np.empty(nlayers, dtype=np.float64)
        self.z_edge = np.empty(nlayers + 1, dtype=np.float64)
        self.dz = np.empty(nlayers, dtype=np.float64)
        self.gravity = np.empty(nlayers, dtype=np.float64)
        self.mubar = np.empty(nlayers, dtype=np.float64)
        self.density = np.empty(nlayers, dtype=np.float64)
        self.densities = np.empty((nspecies, nlayers), dtype=np.float64)
        self.columns = np.empty((nspecies, nlayers), dtype=np.float64)

    def _ensure(self, nlayers, nspecies):
        if nlayers != self.nlayers or nspecies != self.nspecies:
            self._allocate(nlayers, nspecies)

class Radtran:
    "Radiative-transfer driver."

    def __init__(self, opacity_filename: str, nwavelengths_per_chunk=4096):

        # Opacities
        self.opacities = RadtranOpacities(opacity_filename)
        self.opacities_result = RadtranOpacitiesResult()

        # Work out the wavelength chunking
        self.nwavelengths_per_chunk = nwavelengths_per_chunk
        if self.nwavelengths_per_chunk <= 0:
            raise ValueError("nwavelengths_per_chunk must be positive")
        # Number of wavelength chunks
        self.nwavelength_chunks = (self.opacities.nwavelength + self.nwavelengths_per_chunk - 1) // self.nwavelengths_per_chunk

        # Atmosphere
        self.atmosphere = RadtranAtmosphere()

        # Solvers
        self.ThermalSolver = ThermalSolver()
        self.ThermalResult = ThermalResult()

    def _setup_atmosphere(self, atm: Atmosphere, planet: Planet):
        "Setup atmospheric grid."
        self.atmosphere.setup(atm._atm, planet)

    def _prepare_interpolation(self):
        "Prepared interpolation for computing opacities"
        self.opacities.prepare_interpolation(self.atmosphere, self.nwavelengths_per_chunk)

    def _compute_opacity(self, ind_wv0, ind_wv1):
        "Compute the opacity of the atmosphere."
        self.opacities.compute_opacity(self.atmosphere, ind_wv0, ind_wv1, self.opacities_result)

    def _radiate(self, ind_wv0, ind_wv1, calculation):
        "Do the radiative transfer."

        if calculation != 'thermal':
            raise ValueError

    def spectrum(self, atm: Atmosphere, planet: Planet, clouds: Clouds=None, star: Star=None, calculation='thermal'):

        if calculation != 'thermal':
            raise ValueError()
        
        # Setup the atmospheric grid.
        self._setup_atmosphere(atm, planet)

        # Prepare interpolation
        self._prepare_interpolation()

        for i in range(self.nwavelength_chunks):
            ind_wv0 = i * self.nwavelengths_per_chunk
            ind_wv1 = min(ind_wv0 + self.nwavelengths_per_chunk, self.opacities.nwavelength)
            
            # Compute opacity for the wavelength chunk
            self._compute_opacity(ind_wv0, ind_wv1)

            # Do the RT for the wavelength chunk
            self._radiate(ind_wv0, ind_wv1, calculation)
    

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
                dataset.attrs["storage_format"] = storage_format
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
                dataset.attrs["storage_format"] = storage_format
                dataset.attrs["log10_floor"] = float(continuum_log10_floor)
                if storage_format == "log10_uint16":
                    dataset.attrs["y_min"] = np.float64(y_min)
                    dataset.attrs["y_max"] = np.float64(y_max)

            if continuum_temperature_grid is None:
                header.create_dataset("continuum_temperatures", data=np.asarray([], dtype=np.float64))

    finally:
        conn.close()
