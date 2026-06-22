import numba as nb
import numpy as np

def grid_near_resolution(wv_min, wv_max, R):
    """
    Build wavelength bin edges spanning ``[wv_min, wv_max]`` at a resolving
    power as close as practical to ``R``.

    The grid is log-spaced so the bins have approximately constant resolving
    power across the bandpass, but the final bin is forced to land exactly on
    ``wv_max`` rather than leaving a tiny leftover bin.

    Parameters
    ----------
    wv_min : float
        Lower wavelength edge.
    wv_max : float
        Upper wavelength edge.
    R : float
        Target resolving power.

    Returns
    -------
    numpy.ndarray
        Wavelength bin edges.
    """
    wv_min = float(wv_min)
    wv_max = float(wv_max)
    R = float(R)

    if not np.isfinite(wv_min) or not np.isfinite(wv_max) or not np.isfinite(R):
        raise ValueError("wv_min, wv_max, and R must be finite")
    if wv_min <= 0.0 or wv_max <= 0.0:
        raise ValueError("wv_min and wv_max must be positive")
    if wv_max <= wv_min:
        raise ValueError("wv_max must be larger than wv_min")
    if R <= 0.0:
        raise ValueError("R must be positive")

    # For a log-spaced grid, the number of bins needed to achieve a resolution
    # near R is approximately R * ln(wv_max / wv_min). Rounding keeps the grid
    # close to the requested resolving power while ensuring the last bin is not
    # artificially tiny.
    nbin = max(1, int(np.round(R * np.log(wv_max / wv_min))))
    return np.geomspace(wv_min, wv_max, nbin + 1)


def bin_edges_from_wavelength_edges(edges):
    """
    Convert a 1D array of wavelength edges into a 2D ``(nbin, 2)`` array.

    Parameters
    ----------
    edges : array_like, shape (nbin + 1,)
        Monotonically increasing wavelength edges.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(nbin, 2)`` where each row contains the low and high
        edge of a bin.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1:
        raise ValueError(f"edges must be 1D, got shape {edges.shape}")
    if edges.size < 2:
        raise ValueError("at least two wavelength edges are required to build bin_edges")
    if not np.all(np.isfinite(edges)):
        raise ValueError("edges must contain only finite values")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges must be strictly increasing")

    bin_edges = np.empty((edges.size - 1, 2), dtype=np.float64)
    bin_edges[:, 0] = edges[:-1]
    bin_edges[:, 1] = edges[1:]
    return bin_edges


def wavelength_edges_from_bin_edges(bin_edges):
    """
    Convert a contiguous 2D ``(nbin, 2)`` bin-edge array into 1D edges.

    Parameters
    ----------
    bin_edges : array_like, shape (nbin, 2)
        Bin edges where each row is ``(low, high)`` and consecutive bins are
        contiguous.

    Returns
    -------
    numpy.ndarray
        Monotonically increasing wavelength edges with shape ``(nbin + 1,)``.
    """
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    if bin_edges.ndim != 2 or bin_edges.shape[1] != 2:
        raise ValueError(f"bin_edges must have shape (nbin, 2), got {bin_edges.shape}")
    if bin_edges.shape[0] < 1:
        raise ValueError("bin_edges must contain at least one bin")
    if not np.all(np.isfinite(bin_edges)):
        raise ValueError("bin_edges must contain only finite values")
    if np.any(bin_edges[:, 0] >= bin_edges[:, 1]):
        raise ValueError("each bin_edges row must satisfy low < high")
    if np.any(np.diff(bin_edges[:, 0]) <= 0.0):
        raise ValueError("bin_edges must be ordered by increasing lower edge")
    if bin_edges.shape[0] > 1 and not np.allclose(bin_edges[:-1, 1], bin_edges[1:, 0]):
        raise ValueError("bin_edges must be contiguous to convert to 1D edges")

    edges = np.empty(bin_edges.shape[0] + 1, dtype=np.float64)
    edges[:-1] = bin_edges[:, 0]
    edges[-1] = bin_edges[-1, 1]
    return edges


@nb.njit
def rebin(old_bins, old_vals, new_bins, check_inputs=True, force_subset=True):
    """
    Rebin values defined on contiguous 1D bin edges onto a new grid.

    Parameters
    ----------
    old_bins : array_like, shape (n_old + 1,)
        Source bin edges.
    old_vals : array_like, shape (n_old,)
        Values defined on ``old_bins``.
    new_bins : array_like, shape (n_new + 1,)
        Target bin edges.
    check_inputs : bool, optional
        If True, validate the shapes and ordering of the inputs.
    force_subset : bool, optional
        If True, raise an error when ``new_bins`` extend outside the domain
        covered by ``old_bins``.

    Returns
    -------
    numpy.ndarray
        Rebinned values with shape ``(n_new,)``.
    """
    new_vals = np.empty(new_bins.shape[0] - 1)
    _rebin(old_bins, old_vals, new_bins, new_vals, check_inputs, force_subset)
    return new_vals


@nb.njit
def _rebin(old_bins, old_vals, new_bins, new_vals, check_inputs, force_subset):
    n_old = old_vals.shape[0]
    n_new = new_vals.shape[0]

    if check_inputs:
        if old_bins.ndim != 1:
            raise ValueError(f"old_bins must be 1D, got shape {old_bins.shape}")
        if old_vals.ndim != 1:
            raise ValueError(f"old_vals must be 1D, got shape {old_vals.shape}")
        if new_bins.ndim != 1:
            raise ValueError(f"new_bins must be 1D, got shape {new_bins.shape}")
        if new_vals.ndim != 1:
            raise ValueError(f"new_vals must be 1D, got shape {new_vals.shape}")
        if old_bins.shape[0] != n_old + 1:
            raise ValueError("old_bins must have a length of size(old_vals) + 1")
        if new_bins.shape[0] != n_new + 1:
            raise ValueError("new_bins must have a length of size(new_vals) + 1")
        if not np.all(np.isfinite(old_bins)):
            raise ValueError("old_bins must contain only finite values")
        if not np.all(np.isfinite(new_bins)):
            raise ValueError("new_bins must contain only finite values")
        if not np.all(np.isfinite(old_vals)):
            raise ValueError("old_vals must contain only finite values")
        for i in range(n_old):
            if old_bins[i + 1] <= old_bins[i]:
                raise ValueError("old_bins must be strictly increasing")
        for i in range(n_new):
            if new_bins[i + 1] <= new_bins[i]:
                raise ValueError("new_bins must be strictly increasing")

    if force_subset:
        if new_bins[0] < old_bins[0] or new_bins[new_bins.shape[0] - 1] > old_bins[old_bins.shape[0] - 1]:
            raise ValueError("new_bins must be fully contained within old_bins when force_subset=True")

    l = 0

    for i in range(n_new):
        b_new0 = new_bins[i]
        b_new1 = new_bins[i + 1]
        b1_inv = 1.0 / (b_new1 - b_new0)
        v_new = 0.0

        for j in range(l, n_old):
            b_old0 = old_bins[j]
            b_old1 = old_bins[j + 1]
            v_old = old_vals[j]

            if b_old0 > b_new0 and b_old1 < b_new1:
                b2 = b_old1 - b_old0
                v_new += (b2 * b1_inv) * v_old
            elif b_old0 <= b_new0 and b_old1 > b_new0 and b_old1 <= b_new1:
                b2 = b_old1 - b_new0
                v_new += (b2 * b1_inv) * v_old
            elif b_old0 >= b_new0 and b_old0 < b_new1 and b_old1 >= b_new1:
                b2 = b_new1 - b_old0
                v_new += (b2 * b1_inv) * v_old
                l = j
                break
            elif b_old0 <= b_new0 and b_old1 >= b_new1:
                v_new += v_old
                l = j
                break

        new_vals[i] = v_new

    return new_vals


@nb.njit
def rebin_edges(old_bin_edges, old_vals, new_bin_edges, check_inputs=True, force_subset=True):
    """
    Rebin values defined on explicit ``(low, high)`` bin pairs onto a new grid.

    Parameters
    ----------
    old_bin_edges : array_like, shape (n_old, 2)
        Source bin edges.
    old_vals : array_like, shape (n_old,)
        Values defined on ``old_bin_edges``.
    new_bin_edges : array_like, shape (n_new, 2)
        Target bin edges.
    check_inputs : bool, optional
        If True, validate the shapes, ordering, and finiteness of the inputs.
    force_subset : bool, optional
        If True, raise an error when ``new_bin_edges`` extend outside the domain
        covered by ``old_bin_edges``.

    Returns
    -------
    numpy.ndarray
        Rebinned values with shape ``(n_new,)``.
    """
    new_vals = np.empty(new_bin_edges.shape[0])
    _rebin_edges(old_bin_edges, old_vals, new_bin_edges, new_vals, check_inputs, force_subset)
    return new_vals


@nb.njit
def _rebin_edges(old_bin_edges, old_vals, new_bin_edges, new_vals, check_inputs, force_subset):
    n_old = old_vals.shape[0]
    n_new = new_vals.shape[0]

    if check_inputs:
        if old_bin_edges.ndim != 2:
            raise ValueError(f"old_bin_edges must be 2D, got shape {old_bin_edges.shape}")
        if old_vals.ndim != 1:
            raise ValueError(f"old_vals must be 1D, got shape {old_vals.shape}")
        if new_bin_edges.ndim != 2:
            raise ValueError(f"new_bin_edges must be 2D, got shape {new_bin_edges.shape}")
        if new_vals.ndim != 1:
            raise ValueError(f"new_vals must be 1D, got shape {new_vals.shape}")
        if old_bin_edges.shape[0] != n_old:
            raise ValueError(
                "old_bin_edges must have one row per old value, "
                f"got {old_bin_edges.shape[0]} rows for {n_old} values"
            )
        if old_bin_edges.shape[1] != 2:
            raise ValueError(f"old_bin_edges must have shape (nbin, 2), got {old_bin_edges.shape}")
        if new_bin_edges.shape[0] != n_new:
            raise ValueError(
                "new_bin_edges must have one row per new value, "
                f"got {new_bin_edges.shape[0]} rows for {n_new} values"
            )
        if new_bin_edges.shape[1] != 2:
            raise ValueError(f"new_bin_edges must have shape (nbin, 2), got {new_bin_edges.shape}")
        if not np.all(np.isfinite(old_bin_edges)):
            raise ValueError("old_bin_edges must contain only finite values")
        if not np.all(np.isfinite(new_bin_edges)):
            raise ValueError("new_bin_edges must contain only finite values")
        if not np.all(np.isfinite(old_vals)):
            raise ValueError("old_vals must contain only finite values")
        for i in range(n_old):
            if old_bin_edges[i, 0] >= old_bin_edges[i, 1]:
                raise ValueError("each old_bin_edges row must satisfy low < high")
        for i in range(n_new):
            if new_bin_edges[i, 0] >= new_bin_edges[i, 1]:
                raise ValueError("each new_bin_edges row must satisfy low < high")
        for i in range(n_old - 1):
            if old_bin_edges[i + 1, 0] <= old_bin_edges[i, 0]:
                raise ValueError("old_bin_edges must be ordered by increasing lower edge")
            if old_bin_edges[i, 1] > old_bin_edges[i + 1, 0]:
                raise ValueError("old_bin_edges must not overlap")
        for i in range(n_new - 1):
            if new_bin_edges[i + 1, 0] <= new_bin_edges[i, 0]:
                raise ValueError("new_bin_edges must be ordered by increasing lower edge")
            if new_bin_edges[i, 1] > new_bin_edges[i + 1, 0]:
                raise ValueError("new_bin_edges must not overlap")
    
    if force_subset:
        old_idx = 0
        for new_idx in range(n_new):
            new0 = new_bin_edges[new_idx, 0]
            new1 = new_bin_edges[new_idx, 1]

            while old_idx < n_old and old_bin_edges[old_idx, 1] <= new0:
                old_idx += 1

            if old_idx >= n_old:
                raise ValueError("new_bin_edges must be fully contained within old_bin_edges when force_subset=True")

            if old_bin_edges[old_idx, 0] > new0:
                raise ValueError("new_bin_edges must be fully contained within old_bin_edges when force_subset=True")

            cursor = new0
            idx = old_idx
            while cursor < new1:
                if idx >= n_old:
                    raise ValueError("new_bin_edges must be fully contained within old_bin_edges when force_subset=True")
                old0 = old_bin_edges[idx, 0]
                old1 = old_bin_edges[idx, 1]
                if old0 > cursor:
                    raise ValueError("new_bin_edges must be fully contained within old_bin_edges when force_subset=True")
                if old1 >= new1:
                    cursor = new1
                    break
                cursor = old1
                idx += 1

            old_idx = idx

    l = 0

    for i in range(n_new):
        b_new0 = new_bin_edges[i, 0]
        b_new1 = new_bin_edges[i, 1]
        b1_inv = 1.0 / (b_new1 - b_new0)
        v_new = 0.0

        for j in range(l, n_old):
            b_old0 = old_bin_edges[j, 0]
            b_old1 = old_bin_edges[j, 1]
            v_old = old_vals[j]

            if b_old0 > b_new0 and b_old1 < b_new1:
                b2 = b_old1 - b_old0
                v_new += (b2 * b1_inv) * v_old
            elif b_old0 <= b_new0 and b_old1 > b_new0 and b_old1 <= b_new1:
                b2 = b_old1 - b_new0
                v_new += (b2 * b1_inv) * v_old
            elif b_old0 >= b_new0 and b_old0 < b_new1 and b_old1 >= b_new1:
                b2 = b_new1 - b_old0
                v_new += (b2 * b1_inv) * v_old
                l = j
                break
            elif b_old0 <= b_new0 and b_old1 >= b_new1:
                v_new += v_old
                l = j
                break

        new_vals[i] = v_new

    return new_vals
