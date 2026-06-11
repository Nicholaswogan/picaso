from pathlib import Path
import io
import sqlite3

import h5py
import numpy as np

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


def _continuum_metadata_from_source_name(source_name, continuum_unit):
    if source_name == "H-bf":
        return {
            "continuum_name": source_name,
            "continuum_type": "cross_section",
            "primary_species": "H-",
            "secondary_species": None,
            "opacity_unit": "cm2/molecule",
        }

    if source_name == "H-ff":
        return {
            "continuum_name": source_name,
            "continuum_type": "cia",
            "primary_species": "H",
            "secondary_species": "e-",
            "opacity_unit": str(continuum_unit),
        }

    if source_name == "H2-":
        return {
            "continuum_name": source_name,
            "continuum_type": "cia",
            "primary_species": "H2",
            "secondary_species": "e-",
            "opacity_unit": str(continuum_unit),
        }

    continuum_name = _continuum_name(source_name)
    if "-" not in continuum_name:
        raise ValueError(
            f"Could not infer continuum metadata from {source_name!r}; "
            "please add an explicit special-case mapping."
        )
    primary_species, secondary_species = continuum_name.split("-", 1)
    return {
        "continuum_name": continuum_name,
        "continuum_type": "cia",
        "primary_species": primary_species,
        "secondary_species": secondary_species,
        "opacity_unit": str(continuum_unit),
    }


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
        continuum_metadata = [_continuum_metadata_from_source_name(name, continuum_unit) for name in continuum_source_names]
        continuum_names = [meta["continuum_name"] for meta in continuum_metadata]

        if not molecular_names:
            raise RuntimeError("No molecular species found in SQLite database.")

        if verbose:
            print(
                f"Writing {output_hdf5} with {len(molecular_names)} molecular and "
                f"{len(continuum_names)} continuum species"
            )

        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output_hdf5, "w") as f:
            header = f.create_group("header")
            header.create_dataset("format_version", data=np.asarray("1.0", dtype=string_dtype), dtype=string_dtype)
            header.create_dataset(
                "opacity_type", data=np.asarray("molecular+continuum", dtype=string_dtype), dtype=string_dtype
            )
            header.create_dataset("storage_format", data=np.asarray(storage_format, dtype=string_dtype), dtype=string_dtype)
            header.create_dataset("molecular_names", data=np.asarray(molecular_names, dtype=object), dtype=string_dtype)
            header.create_dataset("continuum_names", data=np.asarray(continuum_names, dtype=object), dtype=string_dtype)
            header.create_dataset("pressure_unit", data=np.asarray(str(pressure_unit), dtype=string_dtype), dtype=string_dtype)
            header.create_dataset("temperature_unit", data=np.asarray(str(temperature_unit), dtype=string_dtype), dtype=string_dtype)
            header.create_dataset("wavelength_unit", data=np.asarray("micron", dtype=string_dtype), dtype=string_dtype)
            header.create_dataset("molecular_unit", data=np.asarray(str(molecular_unit), dtype=string_dtype), dtype=string_dtype)
            header.create_dataset("molecular_log10_floor", data=np.float64(molecular_log10_floor))
            header.create_dataset("continuum_log10_floor", data=np.float64(continuum_log10_floor))

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
            for i_continuum, (source_name, meta) in enumerate(zip(continuum_source_names, continuum_metadata), start=1):
                continuum_name = meta["continuum_name"]
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
                dataset.attrs["continuum_type"] = meta["continuum_type"]
                dataset.attrs["primary_species"] = meta["primary_species"]
                if meta["secondary_species"] is not None:
                    dataset.attrs["secondary_species"] = meta["secondary_species"]
                dataset.attrs["opacity_unit"] = meta["opacity_unit"]
                if storage_format == "log10_uint16":
                    dataset.attrs["y_min"] = np.float64(y_min)
                    dataset.attrs["y_max"] = np.float64(y_max)

            if continuum_temperature_grid is None:
                header.create_dataset("continuum_temperatures", data=np.asarray([], dtype=np.float64))

    finally:
        conn.close()
