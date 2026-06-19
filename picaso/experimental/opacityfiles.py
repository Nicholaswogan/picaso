from pathlib import Path
import io
import sqlite3

import h5py
import numpy as np
from tqdm.auto import tqdm

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


def _build_constant_r_wavelength_grid(min_wavelength, max_wavelength, constant_r):
    min_wavelength = float(min_wavelength)
    max_wavelength = float(max_wavelength)
    constant_r = float(constant_r)
    if not np.isfinite(min_wavelength) or not np.isfinite(max_wavelength):
        raise ValueError("wavelength_range must contain finite values")
    if min_wavelength <= 0.0 or max_wavelength <= 0.0:
        raise ValueError("wavelength_range values must be positive")
    if min_wavelength >= max_wavelength:
        raise ValueError(
            f"wavelength_range minimum must be smaller than maximum, got {min_wavelength} >= {max_wavelength}"
        )
    if not np.isfinite(constant_r) or constant_r <= 0.0:
        raise ValueError(f"R must be positive and finite, got {constant_r!r}")
    if constant_r <= 0.5:
        raise ValueError("R must be larger than 0.5")

    spacing = (2.0 * constant_r + 1.0) / (2.0 * constant_r - 1.0)
    npts = np.log(max_wavelength / min_wavelength) / np.log(spacing)
    size = int(np.ceil(npts)) + 1
    wavelength = np.empty(size, dtype=np.float64)
    wavelength[0] = min_wavelength
    for i in range(1, size):
        wavelength[i] = wavelength[i - 1] * spacing
    return wavelength


def _build_bin_edges_from_centers(wavelength):
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


def _validate_bin_edges(bin_edges):
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    if bin_edges.ndim != 2 or bin_edges.shape[1] != 2:
        raise ValueError(f"bin_edges must have shape (nbin, 2), got {bin_edges.shape}")
    if bin_edges.shape[0] == 0:
        raise ValueError("bin_edges must contain at least one bin")
    if not np.all(np.isfinite(bin_edges)):
        raise ValueError("bin_edges must contain only finite values")
    if np.any(bin_edges[:, 0] >= bin_edges[:, 1]):
        raise ValueError("each bin edge pair must satisfy low < high")
    if np.any(np.diff(bin_edges[:, 0]) <= 0.0):
        raise ValueError("bin_edges must be ordered by increasing lower edge")
    if np.any(bin_edges[:-1, 1] > bin_edges[1:, 0]):
        raise ValueError("bin_edges must not overlap")
    return bin_edges


def _bin_centers_from_edges(bin_edges):
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    return 0.5 * (bin_edges[:, 0] + bin_edges[:, 1])


def _double_gauss_points_weights(order=4, gfrac=0.95):
    order = int(order)
    gfrac = float(gfrac)
    if order <= 0:
        raise ValueError("order must be a positive integer")
    if not np.isfinite(gfrac) or not (0.0 < gfrac < 1.0):
        raise ValueError("gfrac must be finite and strictly between 0 and 1")

    g, w = np.polynomial.legendre.leggauss(order)
    wnew1 = gfrac * w * 0.5
    gnew1 = gfrac * 0.5 * (g + 1.0)
    wnew2 = (1.0 - gfrac) * w * 0.5
    gnew2 = gfrac + (1.0 - gfrac) * 0.5 * (g + 1.0)
    return np.concatenate((gnew1, gnew2)), np.concatenate((wnew1, wnew2))


def _normalize_species_selection(selection):
    if selection is None:
        return None
    if isinstance(selection, (str, bytes)):
        return {str(selection)}
    return {str(name) for name in selection}


def _select_range_indices(values, value_range, label):
    values = np.asarray(values, dtype=np.float64)
    if value_range is None:
        return np.arange(values.size, dtype=np.int64)
    if not isinstance(value_range, (tuple, list)) or len(value_range) != 2:
        raise ValueError(f"{label}_range must be a tuple or list of two values")
    low = float(value_range[0])
    high = float(value_range[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{label}_range must contain finite values")
    if low > high:
        raise ValueError(f"{label}_range minimum must not exceed maximum, got {low} > {high}")
    selected = np.flatnonzero((values >= low) & (values <= high))
    if selected.size == 0:
        raise ValueError(f"{label}_range {value_range!r} selects no {label} values from the source grid")
    return selected.astype(np.int64)


def _decode_log10_opacity_block(raw_block, dataset):
    encoding = _decode_hdf5_string(dataset.attrs["encoding"]).lower()
    raw_block = np.asarray(raw_block)

    if encoding == "log10_uint16":
        y_min = float(dataset.attrs["y_min"])
        y_max = float(dataset.attrs["y_max"])
        if y_max == y_min:
            log_block = np.full(raw_block.shape, y_min, dtype=np.float64)
        else:
            scale = (y_max - y_min) / float(np.iinfo(np.uint16).max)
            log_block = y_min + raw_block.astype(np.float64) * scale
        return 10.0 ** log_block

    if encoding == "log10_float32":
        return 10.0 ** raw_block.astype(np.float64)

    raise ValueError(
        f"unsupported source encoding {encoding!r}; expected 'log10_uint16' or 'log10_float32'"
    )


def _infer_source_kind(dataset):
    if dataset.ndim == 3:
        return "molecular"
    if dataset.ndim == 2:
        return "continuum"
    raise ValueError(f"Unsupported opacity dataset rank {dataset.ndim}; expected 2 or 3")


def _select_molecular_row_specs(pressure_indices, temperature_indices):
    return [((ip_out, it_out), (int(it_src), int(ip_src)))
            for ip_out, ip_src in enumerate(pressure_indices)
            for it_out, it_src in enumerate(temperature_indices)]


def _select_continuum_row_specs(temperature_indices):
    return [((it_out,), (int(it_src),)) for it_out, it_src in enumerate(temperature_indices)]


def _finalize_binned_row(sum_row, count_row, wavelengths, log10_floor):
    values = np.zeros_like(sum_row, dtype=np.float64)
    valid = count_row > 0
    if np.any(valid):
        np.divide(sum_row, count_row, out=values, where=valid)
    values = np.maximum(values, float(log10_floor))

    empty = ~valid
    if np.any(empty) and np.any(valid):
        values[empty] = np.interp(wavelengths[empty], wavelengths[valid], values[valid])
        values = np.maximum(values, float(log10_floor))
    elif not np.any(valid):
        values.fill(float(log10_floor))

    return values


def _prepare_wavelength_chunks(source_wavelengths, target_bin_edges, target_wavelengths, source_chunk_wavelengths):
    chunks = []
    target_lows = target_bin_edges[:, 0]
    target_highs = target_bin_edges[:, 1]
    for w0 in range(0, source_wavelengths.size, source_chunk_wavelengths):
        w1 = min(w0 + source_chunk_wavelengths, source_wavelengths.size)
        wavelength_chunk = source_wavelengths[w0:w1]
        if wavelength_chunk.size == 0:
            continue
        bin_ids = np.searchsorted(target_lows, wavelength_chunk, side="right") - 1
        valid = (
            (bin_ids >= 0)
            & (bin_ids < target_wavelengths.size)
            & (wavelength_chunk >= target_lows[bin_ids])
            & (wavelength_chunk <= target_highs[bin_ids])
        )
        if not np.any(valid):
            continue
        bin_ids = bin_ids[valid]
        bin_counts = np.bincount(bin_ids, minlength=target_wavelengths.size).astype(np.uint32)
        chunks.append((w0, w1, np.flatnonzero(valid), bin_ids, bin_counts))
    return chunks


def _accumulate_interp_row(dataset, row_index, source_wavelengths, target_wavelengths, log10_floor):
    raw_row = dataset[(slice(None),) + row_index]
    row_values = _decode_log10_opacity_block(raw_row, dataset)
    row_values = np.interp(
        target_wavelengths,
        source_wavelengths,
        row_values,
        left=float(log10_floor),
        right=float(log10_floor),
    )
    return np.maximum(row_values, float(log10_floor))


def _accumulate_binned_row(dataset, row_index, wavelength_chunks, target_wavelengths, log10_floor):
    target_size = target_wavelengths.size
    sum_row = np.zeros(target_size, dtype=np.float64)
    count_row = np.zeros(target_size, dtype=np.uint32)
    for w0, w1, valid_indices, bin_ids, bin_counts in wavelength_chunks:
        raw_block = dataset[(slice(w0, w1),) + row_index]
        row_values = _decode_log10_opacity_block(raw_block[valid_indices], dataset)
        sum_row += np.bincount(bin_ids, weights=row_values, minlength=target_size)
        count_row += bin_counts
    return _finalize_binned_row(sum_row, count_row, target_wavelengths, log10_floor)


def _process_row_block(
    dataset,
    out,
    row_specs,
    source_wavelengths,
    target_wavelengths,
    log10_floor,
    storage_format,
    verbose,
    desc,
    source_y_min=None,
    source_y_max=None,
):
    if storage_format == "log10_uint16":
        if source_y_min is None or source_y_max is None:
            raise ValueError(f"{desc} is missing source y_min/y_max attrs needed for quantized output")
        y_min = float(source_y_min)
        y_max = float(source_y_max)
        scale = 0.0 if y_max == y_min else float(np.iinfo(np.uint16).max) / (y_max - y_min)
        out.attrs["y_min"] = np.float64(y_min)
        out.attrs["y_max"] = np.float64(y_max)
        with tqdm(
            total=len(row_specs),
            disable=not verbose,
            desc=desc,
            unit="row",
            leave=False,
        ) as bar:
            for output_index, source_index in row_specs:
                row_values = _accumulate_interp_row(
                    dataset,
                    source_index,
                    source_wavelengths,
                    target_wavelengths,
                    log10_floor,
                )
                log_row = np.log10(row_values)
                if scale == 0.0:
                    encoded = np.zeros_like(log_row, dtype=np.uint16)
                else:
                    encoded = np.rint((log_row - y_min) * scale).astype(np.uint16)
                out[output_index] = encoded
                bar.update(1)
        return

    y_min = np.inf
    y_max = -np.inf
    with tqdm(
        total=len(row_specs),
        disable=not verbose,
        desc=desc,
        unit="row",
        leave=False,
    ) as bar:
        for output_index, source_index in row_specs:
            row_values = _accumulate_interp_row(
                dataset,
                source_index,
                source_wavelengths,
                target_wavelengths,
                log10_floor,
            )
            log_row = np.log10(row_values)
            y_min = min(y_min, float(np.min(log_row)))
            y_max = max(y_max, float(np.max(log_row)))
            out[output_index] = log_row.astype(np.float32)
            bar.update(1)
    out.attrs["y_min"] = np.float64(y_min)
    out.attrs["y_max"] = np.float64(y_max)


def _finalize_ck_row(row_values, count_row, wavelengths, log10_floor):
    values = np.asarray(row_values, dtype=np.float64)
    valid = count_row > 0
    if np.any(valid):
        if np.any(~valid):
            for ig in range(values.shape[1]):
                values[~valid, ig] = np.interp(wavelengths[~valid], wavelengths[valid], values[valid, ig])
        values = np.maximum(values, float(log10_floor))
        return values

    values.fill(float(log10_floor))
    return values


def _accumulate_ck_row(dataset, row_index, wavelength_chunks, target_wavelengths, g_points, log10_floor):
    target_size = target_wavelengths.size
    ng = g_points.size
    sample_chunks = [[] for _ in range(target_size)]
    count_row = np.zeros(target_size, dtype=np.uint32)

    for w0, w1, valid_indices, bin_ids, bin_counts in wavelength_chunks:
        raw_block = dataset[(slice(w0, w1),) + row_index]
        row_values = _decode_log10_opacity_block(raw_block[valid_indices], dataset)
        if row_values.size == 0:
            continue

        count_row += bin_counts
        if row_values.size == 1:
            bin_id = int(bin_ids[0])
            sample_chunks[bin_id].append(
                np.asarray([np.log10(max(float(row_values[0]), float(log10_floor)))], dtype=np.float64)
            )
            continue

        split_points = np.flatnonzero(np.diff(bin_ids)) + 1
        unique_bins = bin_ids[np.r_[0, split_points]]
        for bin_id, values in zip(unique_bins, np.split(row_values, split_points)):
            values = np.log10(np.maximum(np.asarray(values, dtype=np.float64), float(log10_floor)))
            sample_chunks[int(bin_id)].append(values)

    out_row = np.full((target_size, ng), np.log10(float(log10_floor)), dtype=np.float64)
    for i_bin, chunks in enumerate(sample_chunks):
        if not chunks:
            continue
        data = np.concatenate(chunks)
        data.sort()
        if data.size == 1:
            out_row[i_bin, :] = data[0]
            continue
        x = np.linspace(0.0, 1.0, data.size)
        out_row[i_bin, :] = np.interp(g_points, x, data)

    return _finalize_ck_row(out_row, count_row, target_wavelengths, np.log10(float(log10_floor)))


def _process_ck_row_block(
    dataset,
    out,
    row_specs,
    wavelength_chunks,
    target_wavelengths,
    g_points,
    log10_floor,
    storage_format,
    verbose,
    desc,
):
    if storage_format == "log10_uint16":
        y_min = np.inf
        y_max = -np.inf
        with tqdm(
            total=len(row_specs),
            disable=not verbose,
            desc=desc,
            unit="row",
            leave=False,
        ) as bar:
            for _, source_index in row_specs:
                row_values = _accumulate_ck_row(
                    dataset,
                    source_index,
                    wavelength_chunks,
                    target_wavelengths,
                    g_points,
                    log10_floor,
                )
                y_min = min(y_min, float(np.min(row_values)))
                y_max = max(y_max, float(np.max(row_values)))
                bar.update(1)

        scale = 0.0 if y_max == y_min else float(np.iinfo(np.uint16).max) / (y_max - y_min)
        out.attrs["y_min"] = np.float64(y_min)
        out.attrs["y_max"] = np.float64(y_max)
        with tqdm(
            total=len(row_specs),
            disable=not verbose,
            desc=desc,
            unit="row",
            leave=False,
        ) as bar:
            for output_index, source_index in row_specs:
                row_values = _accumulate_ck_row(
                    dataset,
                    source_index,
                    wavelength_chunks,
                    target_wavelengths,
                    g_points,
                    log10_floor,
                )
                if scale == 0.0:
                    encoded = np.zeros_like(row_values, dtype=np.uint16)
                else:
                    encoded = np.rint((row_values - y_min) * scale).astype(np.uint16)
                out[output_index] = encoded
                bar.update(1)
        return

    y_min = np.inf
    y_max = -np.inf
    with tqdm(
        total=len(row_specs),
        disable=not verbose,
        desc=desc,
        unit="row",
        leave=False,
    ) as bar:
        for output_index, source_index in row_specs:
            row_values = _accumulate_ck_row(
                dataset,
                source_index,
                wavelength_chunks,
                target_wavelengths,
                g_points,
                log10_floor,
            )
            y_min = min(y_min, float(np.min(row_values)))
            y_max = max(y_max, float(np.max(row_values)))
            out[output_index] = row_values.astype(np.float32)
            bar.update(1)
    out.attrs["y_min"] = np.float64(y_min)
    out.attrs["y_max"] = np.float64(y_max)


def _read_opacity_grid(grid_path):
    with h5py.File(grid_path, "r") as f:
        if "wavelengths" not in f or "pressures" not in f:
            raise ValueError(f"{grid_path} is missing required wavelength or pressure grids")
        wavelengths = np.asarray(f["wavelengths"][:], dtype=np.float64)
        pressures = np.asarray(f["pressures"][:], dtype=np.float64)
        molecular_temperatures = np.asarray(f["molecular_temperatures"][:], dtype=np.float64)
        continuum_temperatures = np.asarray(f["continuum_temperatures"][:], dtype=np.float64)
        pressure_unit = _decode_hdf5_string(f["pressures"].attrs.get("units", "bar"))
        temperature_unit = _decode_hdf5_string(f["molecular_temperatures"].attrs.get("units", "K"))
        wavelength_unit = _decode_hdf5_string(f["wavelengths"].attrs.get("units", "micron"))
    return {
        "wavelengths": wavelengths,
        "pressures": pressures,
        "molecular_temperatures": molecular_temperatures,
        "continuum_temperatures": continuum_temperatures,
        "pressure_unit": pressure_unit,
        "temperature_unit": temperature_unit,
        "wavelength_unit": wavelength_unit,
    }


def _discover_opacity_species(opacity_dir):
    opacity_dir = Path(opacity_dir)
    if not opacity_dir.is_dir():
        raise ValueError(f"{opacity_dir} is not a directory")

    species = []
    for path in sorted(opacity_dir.glob("*.h5")):
        if path.name == "grid.h5":
            continue
        with h5py.File(path, "r") as f:
            if "opacity" not in f:
                raise ValueError(f"{path} does not contain an 'opacity' dataset")
            dataset = f["opacity"]
            kind = _infer_source_kind(dataset)
            species_name = _decode_hdf5_string(dataset.attrs.get("species", path.stem))
            species.append(
                {
                    "path": path,
                    "kind": kind,
                    "species_name": species_name,
                    "dataset_shape": tuple(dataset.shape),
                    "encoding": _decode_hdf5_string(dataset.attrs["encoding"]).lower(),
                    "opacity_unit": _decode_hdf5_string(dataset.attrs.get("opacity_unit", "")),
                    "primary_species": _decode_hdf5_string(dataset.attrs.get("primary_species", species_name)),
                    "secondary_species": _decode_hdf5_string(dataset.attrs["secondary_species"])
                    if "secondary_species" in dataset.attrs
                    else None,
                }
            )
    return species


def opacity_dir_to_hdf5(
    opacity_dir,
    output_hdf5,
    wavelength_range,
    R,
    compression="lzf",
    shuffle=True,
    storage_format="log10_uint16",
    molecular_species=None,
    continuum_species=None,
    temperature_range=None,
    pressure_range=None,
    source_chunk_wavelengths=65536,
    molecular_log10_floor=1e-50,
    continuum_log10_floor=1e-100,
    verbose=True,
):
    """Convert an opacity directory to the new HDF5 layout.

    The input directory must contain a ``grid.h5`` file and one ``.h5`` file
    per opacity species. Molecular files are expected to contain 3D opacity
    cubes with axes ``(wavelength, temperature, pressure)``. Continuum files
    are expected to contain 2D opacity tables with axes
    ``(wavelength, temperature)``.

    Parameters
    ----------
    opacity_dir : str or Path
        Directory containing the opacity source files.
    output_hdf5 : str or Path
        Destination HDF5 file path.
    wavelength_range : tuple of float
        Inclusive minimum and maximum wavelength to write to the output file.
    R : float
        Target resolving power used to build the output wavelength grid.
    compression : str or None, optional
        HDF5 compression filter for opacity datasets.
    shuffle : bool, optional
        Whether to enable the HDF5 shuffle filter.
    storage_format : {'log10_uint16', 'log10_float32'}, optional
        Storage encoding for the opacity datasets.
    molecular_species : sequence of str or str, optional
        Explicit molecular species to include. ``None`` includes all available
        molecular species.
    continuum_species : sequence of str or str, optional
        Explicit continuum species to include. ``None`` includes all available
        continuum species.
    temperature_range : tuple of float or None, optional
        Inclusive minimum and maximum temperature to include. ``None`` keeps
        all temperature points.
    pressure_range : tuple of float or None, optional
        Inclusive minimum and maximum pressure to include. ``None`` keeps all
        pressure points.
    source_chunk_wavelengths : int, optional
        Number of source wavelengths to process per chunk.
    molecular_log10_floor : float, optional
        Floor applied before taking ``log10`` for molecular opacities.
    continuum_log10_floor : float, optional
        Floor applied before taking ``log10`` for continuum opacities.
    verbose : bool, optional
        Whether to show per-species progress bars.
    """
    opacity_dir = Path(opacity_dir)
    output_hdf5 = Path(output_hdf5)

    if storage_format not in {"log10_uint16", "log10_float32"}:
        raise ValueError(f"Unsupported storage_format: {storage_format!r}")
    if compression not in {"lzf", None}:
        raise ValueError(f"Unsupported compression: {compression!r}")
    if not isinstance(source_chunk_wavelengths, int) or source_chunk_wavelengths <= 0:
        raise ValueError("source_chunk_wavelengths must be a positive integer")
    if not isinstance(wavelength_range, (tuple, list)) or len(wavelength_range) != 2:
        raise ValueError("wavelength_range must be a tuple or list of two values")

    grid = _read_opacity_grid(opacity_dir / "grid.h5")
    source_wavelengths = grid["wavelengths"]
    source_min = float(np.min(source_wavelengths))
    source_max = float(np.max(source_wavelengths))
    molecular_temperature_indices = _select_range_indices(
        grid["molecular_temperatures"], temperature_range, "temperature"
    )
    continuum_temperature_indices = _select_range_indices(
        grid["continuum_temperatures"], temperature_range, "temperature"
    )
    pressure_indices = _select_range_indices(grid["pressures"], pressure_range, "pressure")
    target_wavelengths = _build_constant_r_wavelength_grid(wavelength_range[0], wavelength_range[1], R)
    target_bin_edges = np.empty((target_wavelengths.size, 2), dtype=np.float64)
    if target_wavelengths.size < 2:
        raise ValueError("target wavelength grid must contain at least two points")
    target_midpoints = 0.5 * (target_wavelengths[1:] + target_wavelengths[:-1])
    target_bin_edges[1:, 0] = target_midpoints
    target_bin_edges[:-1, 1] = target_midpoints
    target_bin_edges[0, 0] = target_wavelengths[0] - 0.5 * (target_wavelengths[1] - target_wavelengths[0])
    target_bin_edges[-1, 1] = target_wavelengths[-1] + 0.5 * (target_wavelengths[-1] - target_wavelengths[-2])

    if target_bin_edges[0, 0] < source_min or target_bin_edges[-1, 1] > source_max:
        if verbose:
            print(
                "Requested wavelength range extends beyond the source grid edges; "
                "bins will be clipped to the available source opacities."
            )

    available = _discover_opacity_species(opacity_dir)
    molecular_requested = _normalize_species_selection(molecular_species)
    continuum_requested = _normalize_species_selection(continuum_species)

    molecular_sources = [item for item in available if item["kind"] == "molecular"]
    continuum_sources = [item for item in available if item["kind"] == "continuum"]

    if molecular_requested is not None:
        molecular_sources = [item for item in molecular_sources if item["species_name"] in molecular_requested]
        missing = molecular_requested - {item["species_name"] for item in available if item["kind"] == "molecular"}
        if missing:
            raise ValueError(f"Requested molecular species not found: {sorted(missing)!r}")
    if continuum_requested is not None:
        continuum_sources = [item for item in continuum_sources if item["species_name"] in continuum_requested]
        missing = continuum_requested - {item["species_name"] for item in available if item["kind"] == "continuum"}
        if missing:
            raise ValueError(f"Requested continuum species not found: {sorted(missing)!r}")

    if not molecular_sources and not continuum_sources:
        raise ValueError("No molecular or continuum opacities were selected")

    molecular_sources = sorted(molecular_sources, key=lambda item: item["species_name"])
    continuum_sources = sorted(continuum_sources, key=lambda item: item["species_name"])
    continuum_metadata = [
        _continuum_metadata_from_source_name(item["species_name"], item["opacity_unit"] or "cm-1 amagat^-2")
        for item in continuum_sources
    ]

    molecular_unit = molecular_sources[0]["opacity_unit"] if molecular_sources else "cm2/molecule"
    continuum_unit = continuum_sources[0]["opacity_unit"] if continuum_sources else ""
    if molecular_sources and any(item["opacity_unit"] != molecular_unit for item in molecular_sources):
        raise ValueError("All selected molecular files must use the same opacity_unit")
    if continuum_sources and any(item["opacity_unit"] != continuum_unit for item in continuum_sources):
        raise ValueError("All selected continuum files must use the same opacity_unit")

    if verbose:
        print(
            f"Writing {output_hdf5} with {len(molecular_sources)} molecular and "
            f"{len(continuum_sources)} continuum species"
        )

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_hdf5, "w") as f:
        header = f.create_group("header")
        header.create_dataset("format_version", data=np.asarray("1.0", dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("opacity_type", data=np.asarray("molecular+continuum", dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("storage_format", data=np.asarray(storage_format, dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("molecular_names", data=np.asarray([item["species_name"] for item in molecular_sources], dtype=object), dtype=string_dtype)
        header.create_dataset("continuum_names", data=np.asarray([item["continuum_name"] for item in continuum_metadata], dtype=object), dtype=string_dtype)
        header.create_dataset("pressure_unit", data=np.asarray(grid["pressure_unit"], dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("temperature_unit", data=np.asarray(grid["temperature_unit"], dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("wavelength_unit", data=np.asarray(grid["wavelength_unit"], dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("molecular_unit", data=np.asarray(molecular_unit, dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("continuum_unit", data=np.asarray(continuum_unit, dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("molecular_log10_floor", data=np.float64(molecular_log10_floor))
        header.create_dataset("continuum_log10_floor", data=np.float64(continuum_log10_floor))
        header.create_dataset("pressure", data=np.asarray(grid["pressures"][pressure_indices], dtype=np.float64))
        header.create_dataset(
            "temperature", data=np.asarray(grid["molecular_temperatures"][molecular_temperature_indices], dtype=np.float64)
        )
        header.create_dataset(
            "continuum_temperatures",
            data=np.asarray(grid["continuum_temperatures"][continuum_temperature_indices], dtype=np.float64),
        )
        header.create_dataset("wavelength", data=np.asarray(target_wavelengths, dtype=np.float64))

        molecular_group = f.create_group("molecular")
        total_molecules = len(molecular_sources)
        for index, item in enumerate(molecular_sources, start=1):
            species_name = item["species_name"]
            if verbose:
                print(f"[molecular {index}/{total_molecules}] Writing {species_name}")
            with h5py.File(item["path"], "r") as source:
                dataset = source["opacity"]
                if dataset.ndim != 3:
                    raise ValueError(f"Molecular file {item['path']} must be 3D")
                nw_src, nt_src, np_src = dataset.shape
                if nw_src != source_wavelengths.size:
                    raise ValueError(
                        f"Molecular file {item['path']} has {nw_src} wavelengths, expected {source_wavelengths.size}"
                    )
                if nt_src != grid["molecular_temperatures"].size or np_src != grid["pressures"].size:
                    raise ValueError(
                        f"Molecular file {item['path']} has unexpected PT dimensions {dataset.shape[1:]}"
                    )

                out_shape = (pressure_indices.size, molecular_temperature_indices.size, target_wavelengths.size)
                dataset_kwargs = {}
                if compression is not None:
                    dataset_kwargs["compression"] = compression
                    dataset_kwargs["shuffle"] = shuffle
                    dataset_kwargs["chunks"] = (1, 1, min(target_wavelengths.size, 4096))
                elif shuffle:
                    dataset_kwargs["shuffle"] = True

                out = molecular_group.create_dataset(
                    species_name,
                    shape=out_shape,
                    dtype=np.uint16 if storage_format == "log10_uint16" else np.float32,
                    **dataset_kwargs,
                )
                out.attrs["axes"] = "pressure,temperature,wavelength"
                out.attrs["encoding"] = storage_format
                out.attrs["log10_floor"] = float(molecular_log10_floor)
                out.attrs["opacity_unit"] = "cm2/molecule"
                out.attrs["species"] = species_name
                row_specs = _select_molecular_row_specs(pressure_indices, molecular_temperature_indices)
                source_y_min = dataset.attrs.get("y_min")
                source_y_max = dataset.attrs.get("y_max")
                _process_row_block(
                    dataset,
                    out,
                    row_specs,
                    source_wavelengths,
                    target_wavelengths,
                    molecular_log10_floor,
                    storage_format,
                    verbose,
                    species_name,
                    source_y_min=source_y_min,
                    source_y_max=source_y_max,
                )

        continuum_group = f.create_group("continuum")
        total_continuum = len(continuum_sources)
        for index, (item, meta) in enumerate(zip(continuum_sources, continuum_metadata), start=1):
            species_name = item["species_name"]
            if verbose:
                print(f"[continuum {index}/{total_continuum}] Writing {meta['continuum_name']}")
            with h5py.File(item["path"], "r") as source:
                dataset = source["opacity"]
                if dataset.ndim != 2:
                    raise ValueError(f"Continuum file {item['path']} must be 2D")
                nw_src, nt_src = dataset.shape
                if nw_src != source_wavelengths.size:
                    raise ValueError(
                        f"Continuum file {item['path']} has {nw_src} wavelengths, expected {source_wavelengths.size}"
                    )
                if nt_src != grid["continuum_temperatures"].size:
                    raise ValueError(
                        f"Continuum file {item['path']} has unexpected temperature dimension {dataset.shape[1]}"
                    )

                out_shape = (continuum_temperature_indices.size, target_wavelengths.size)
                dataset_kwargs = {}
                if compression is not None:
                    dataset_kwargs["compression"] = compression
                    dataset_kwargs["shuffle"] = shuffle
                    dataset_kwargs["chunks"] = (1, min(target_wavelengths.size, 4096))
                elif shuffle:
                    dataset_kwargs["shuffle"] = True

                out = continuum_group.create_dataset(
                    meta["continuum_name"],
                    shape=out_shape,
                    dtype=np.uint16 if storage_format == "log10_uint16" else np.float32,
                    **dataset_kwargs,
                )
                out.attrs["axes"] = "temperature,wavelength"
                out.attrs["encoding"] = storage_format
                out.attrs["log10_floor"] = float(continuum_log10_floor)
                out.attrs["continuum_type"] = meta["continuum_type"]
                out.attrs["primary_species"] = meta["primary_species"]
                if meta["secondary_species"] is not None:
                    out.attrs["secondary_species"] = meta["secondary_species"]
                out.attrs["opacity_unit"] = meta["opacity_unit"]
                out.attrs["species"] = meta["continuum_name"]
                row_specs = _select_continuum_row_specs(continuum_temperature_indices)
                source_y_min = dataset.attrs.get("y_min")
                source_y_max = dataset.attrs.get("y_max")
                _process_row_block(
                    dataset,
                    out,
                    row_specs,
                    source_wavelengths,
                    target_wavelengths,
                    continuum_log10_floor,
                    storage_format,
                    verbose,
                    meta["continuum_name"],
                    source_y_min=source_y_min,
                    source_y_max=source_y_max,
                )


def opacity_dir_to_correlated_k_hdf5(
    opacity_dir,
    output_hdf5,
    bin_edges,
    compression="lzf",
    shuffle=True,
    storage_format="log10_float32",
    molecular_species=None,
    continuum_species=None,
    temperature_range=None,
    pressure_range=None,
    source_chunk_wavelengths=65536,
    molecular_log10_floor=1e-50,
    continuum_log10_floor=1e-100,
    g_order=4,
    gfrac=0.95,
    verbose=True,
):
    """Convert an opacity directory to a correlated-k HDF5 layout.

    The source directory must contain a ``grid.h5`` file and one ``.h5`` file
    per opacity species. Molecular files are expected to contain 3D opacity
    cubes with axes ``(wavelength, temperature, pressure)``. Continuum files
    are expected to contain 2D opacity tables with axes
    ``(wavelength, temperature)``.

    The output keeps the same header / group organization as the resampled
    writer, but adds explicit ``bin_edges`` together with shared ``g_points``
    and ``g_weights`` arrays.
    """
    opacity_dir = Path(opacity_dir)
    output_hdf5 = Path(output_hdf5)

    if storage_format not in {"log10_uint16", "log10_float32"}:
        raise ValueError(f"Unsupported storage_format: {storage_format!r}")
    if compression not in {"lzf", None}:
        raise ValueError(f"Unsupported compression: {compression!r}")
    if not isinstance(source_chunk_wavelengths, int) or source_chunk_wavelengths <= 0:
        raise ValueError("source_chunk_wavelengths must be a positive integer")
    grid = _read_opacity_grid(opacity_dir / "grid.h5")
    source_wavelengths = grid["wavelengths"]
    source_min = float(np.min(source_wavelengths))
    source_max = float(np.max(source_wavelengths))
    molecular_temperature_indices = _select_range_indices(
        grid["molecular_temperatures"], temperature_range, "temperature"
    )
    continuum_temperature_indices = _select_range_indices(
        grid["continuum_temperatures"], temperature_range, "temperature"
    )
    pressure_indices = _select_range_indices(grid["pressures"], pressure_range, "pressure")
    target_bin_edges = _validate_bin_edges(bin_edges)
    target_wavelengths = _bin_centers_from_edges(target_bin_edges)
    g_points, g_weights = _double_gauss_points_weights(g_order, gfrac)

    if target_bin_edges[0, 0] < source_min or target_bin_edges[-1, 1] > source_max:
        raise ValueError(
            "bin_edges extend beyond the source wavelength grid: "
            f"source range is [{source_min}, {source_max}], "
            f"but bin_edges span [{target_bin_edges[0, 0]}, {target_bin_edges[-1, 1]}]"
        )
    wavelength_chunks = _prepare_wavelength_chunks(
        source_wavelengths, target_bin_edges, target_wavelengths, source_chunk_wavelengths
    )

    available = _discover_opacity_species(opacity_dir)
    molecular_requested = _normalize_species_selection(molecular_species)
    continuum_requested = _normalize_species_selection(continuum_species)

    molecular_sources = [item for item in available if item["kind"] == "molecular"]
    continuum_sources = [item for item in available if item["kind"] == "continuum"]

    if molecular_requested is not None:
        molecular_sources = [item for item in molecular_sources if item["species_name"] in molecular_requested]
        missing = molecular_requested - {item["species_name"] for item in available if item["kind"] == "molecular"}
        if missing:
            raise ValueError(f"Requested molecular species not found: {sorted(missing)!r}")
    if continuum_requested is not None:
        continuum_sources = [item for item in continuum_sources if item["species_name"] in continuum_requested]
        missing = continuum_requested - {item["species_name"] for item in available if item["kind"] == "continuum"}
        if missing:
            raise ValueError(f"Requested continuum species not found: {sorted(missing)!r}")

    if not molecular_sources and not continuum_sources:
        raise ValueError("No molecular or continuum opacities were selected")

    molecular_sources = sorted(molecular_sources, key=lambda item: item["species_name"])
    continuum_sources = sorted(continuum_sources, key=lambda item: item["species_name"])
    continuum_metadata = [
        _continuum_metadata_from_source_name(item["species_name"], item["opacity_unit"] or "cm-1 amagat^-2")
        for item in continuum_sources
    ]

    molecular_unit = molecular_sources[0]["opacity_unit"] if molecular_sources else "cm2/molecule"
    continuum_unit = continuum_sources[0]["opacity_unit"] if continuum_sources else ""
    if molecular_sources and any(item["opacity_unit"] != molecular_unit for item in molecular_sources):
        raise ValueError("All selected molecular files must use the same opacity_unit")
    if continuum_sources and any(item["opacity_unit"] != continuum_unit for item in continuum_sources):
        raise ValueError("All selected continuum files must use the same opacity_unit")

    if verbose:
        print(
            f"Writing {output_hdf5} with {len(molecular_sources)} molecular and "
            f"{len(continuum_sources)} continuum species"
        )

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_hdf5, "w") as f:
        header = f.create_group("header")
        header.create_dataset("format_version", data=np.asarray("1.0", dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("opacity_type", data=np.asarray("correlated-k", dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("storage_format", data=np.asarray(storage_format, dtype=string_dtype), dtype=string_dtype)
        header.create_dataset(
            "molecular_names",
            data=np.asarray([item["species_name"] for item in molecular_sources], dtype=object),
            dtype=string_dtype,
        )
        header.create_dataset(
            "continuum_names",
            data=np.asarray([item["continuum_name"] for item in continuum_metadata], dtype=object),
            dtype=string_dtype,
        )
        header.create_dataset("pressure_unit", data=np.asarray(grid["pressure_unit"], dtype=string_dtype), dtype=string_dtype)
        header.create_dataset(
            "temperature_unit", data=np.asarray(grid["temperature_unit"], dtype=string_dtype), dtype=string_dtype
        )
        header.create_dataset("wavelength_unit", data=np.asarray(grid["wavelength_unit"], dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("molecular_unit", data=np.asarray(molecular_unit, dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("continuum_unit", data=np.asarray(continuum_unit, dtype=string_dtype), dtype=string_dtype)
        header.create_dataset("molecular_log10_floor", data=np.float64(molecular_log10_floor))
        header.create_dataset("continuum_log10_floor", data=np.float64(continuum_log10_floor))
        header.create_dataset("pressure", data=np.asarray(grid["pressures"][pressure_indices], dtype=np.float64))
        header.create_dataset(
            "temperature", data=np.asarray(grid["molecular_temperatures"][molecular_temperature_indices], dtype=np.float64)
        )
        header.create_dataset(
            "continuum_temperatures",
            data=np.asarray(grid["continuum_temperatures"][continuum_temperature_indices], dtype=np.float64),
        )
        header.create_dataset("wavelength", data=np.asarray(target_wavelengths, dtype=np.float64))
        header.create_dataset("bin_edges", data=np.asarray(target_bin_edges, dtype=np.float64))
        header.create_dataset("g_points", data=np.asarray(g_points, dtype=np.float64))
        header.create_dataset("g_weights", data=np.asarray(g_weights, dtype=np.float64))

        molecular_group = f.create_group("molecular")
        total_molecules = len(molecular_sources)
        for index, item in enumerate(molecular_sources, start=1):
            species_name = item["species_name"]
            if verbose:
                print(f"[molecular {index}/{total_molecules}] Writing {species_name}")
            with h5py.File(item["path"], "r") as source:
                dataset = source["opacity"]
                if dataset.ndim != 3:
                    raise ValueError(f"Molecular file {item['path']} must be 3D")
                nw_src, nt_src, np_src = dataset.shape
                if nw_src != source_wavelengths.size:
                    raise ValueError(
                        f"Molecular file {item['path']} has {nw_src} wavelengths, expected {source_wavelengths.size}"
                    )
                if nt_src != grid["molecular_temperatures"].size or np_src != grid["pressures"].size:
                    raise ValueError(
                        f"Molecular file {item['path']} has unexpected PT dimensions {dataset.shape[1:]}"
                    )

                out_shape = (
                    pressure_indices.size,
                    molecular_temperature_indices.size,
                    target_wavelengths.size,
                    g_points.size,
                )
                dataset_kwargs = {}
                if compression is not None:
                    dataset_kwargs["compression"] = compression
                    dataset_kwargs["shuffle"] = shuffle
                    dataset_kwargs["chunks"] = (1, 1, min(target_wavelengths.size, 4096), g_points.size)
                elif shuffle:
                    dataset_kwargs["shuffle"] = True

                out = molecular_group.create_dataset(
                    species_name,
                    shape=out_shape,
                    dtype=np.uint16 if storage_format == "log10_uint16" else np.float32,
                    **dataset_kwargs,
                )
                out.attrs["axes"] = "pressure,temperature,wavelength,g"
                out.attrs["encoding"] = storage_format
                out.attrs["log10_floor"] = float(molecular_log10_floor)
                out.attrs["opacity_unit"] = "cm2/molecule"
                out.attrs["species"] = species_name
                row_specs = _select_molecular_row_specs(pressure_indices, molecular_temperature_indices)
                _process_ck_row_block(
                    dataset,
                    out,
                    row_specs,
                    wavelength_chunks,
                    target_wavelengths,
                    g_points,
                    molecular_log10_floor,
                    storage_format,
                    verbose,
                    species_name,
                )

        continuum_group = f.create_group("continuum")
        total_continuum = len(continuum_sources)
        for index, (item, meta) in enumerate(zip(continuum_sources, continuum_metadata), start=1):
            species_name = item["species_name"]
            if verbose:
                print(f"[continuum {index}/{total_continuum}] Writing {meta['continuum_name']}")
            with h5py.File(item["path"], "r") as source:
                dataset = source["opacity"]
                if dataset.ndim != 2:
                    raise ValueError(f"Continuum file {item['path']} must be 2D")
                nw_src, nt_src = dataset.shape
                if nw_src != source_wavelengths.size:
                    raise ValueError(
                        f"Continuum file {item['path']} has {nw_src} wavelengths, expected {source_wavelengths.size}"
                    )
                if nt_src != grid["continuum_temperatures"].size:
                    raise ValueError(
                        f"Continuum file {item['path']} has unexpected temperature dimension {dataset.shape[1]}"
                    )

                out_shape = (continuum_temperature_indices.size, target_wavelengths.size)
                dataset_kwargs = {}
                if compression is not None:
                    dataset_kwargs["compression"] = compression
                    dataset_kwargs["shuffle"] = shuffle
                    dataset_kwargs["chunks"] = (1, min(target_wavelengths.size, 4096))
                elif shuffle:
                    dataset_kwargs["shuffle"] = True

                out = continuum_group.create_dataset(
                    meta["continuum_name"],
                    shape=out_shape,
                    dtype=np.uint16 if storage_format == "log10_uint16" else np.float32,
                    **dataset_kwargs,
                )
                out.attrs["axes"] = "temperature,wavelength"
                out.attrs["encoding"] = storage_format
                out.attrs["log10_floor"] = float(continuum_log10_floor)
                out.attrs["continuum_type"] = meta["continuum_type"]
                out.attrs["primary_species"] = meta["primary_species"]
                if meta["secondary_species"] is not None:
                    out.attrs["secondary_species"] = meta["secondary_species"]
                out.attrs["opacity_unit"] = meta["opacity_unit"]
                out.attrs["species"] = meta["continuum_name"]
                row_specs = _select_continuum_row_specs(continuum_temperature_indices)

                if storage_format == "log10_uint16":
                    cont_y_min = np.inf
                    cont_y_max = -np.inf
                    for _, source_index in row_specs:
                        row_values = _accumulate_interp_row(
                            dataset,
                            source_index,
                            source_wavelengths,
                            target_wavelengths,
                            continuum_log10_floor,
                        )
                        log_row = np.log10(row_values)
                        cont_y_min = min(cont_y_min, float(np.min(log_row)))
                        cont_y_max = max(cont_y_max, float(np.max(log_row)))
                    source_y_min = np.float64(cont_y_min)
                    source_y_max = np.float64(cont_y_max)
                else:
                    source_y_min = None
                    source_y_max = None

                _process_row_block(
                    dataset,
                    out,
                    row_specs,
                    source_wavelengths,
                    target_wavelengths,
                    continuum_log10_floor,
                    storage_format,
                    verbose,
                    meta["continuum_name"],
                    source_y_min=source_y_min,
                    source_y_max=source_y_max,
                )

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
            "opacity_unit": "cm-1 amagat-2",
        }

    if source_name == "H2-":
        return {
            "continuum_name": source_name,
            "continuum_type": "cia",
            "primary_species": "H2",
            "secondary_species": "e-",
            "opacity_unit": "cm-1 amagat-2",
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
        "opacity_unit": "cm-1 amagat-2",
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
