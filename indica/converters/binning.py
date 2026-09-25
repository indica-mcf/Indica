"""Utilities for binning xarray DataArrays with custom boundaries."""

from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from xarray import DataArray


def bin_dataarray(
    data: DataArray,
    bins: Union[np.ndarray, Tuple[Tuple[float, float], ...], List[Tuple[float, float]]],
    coord_name: str = "t",
    labels: np.ndarray = None,
    reduce_func: str = "mean",
) -> DataArray:
    """
    Bin a DataArray with automatic detection of bin type.

    Automatically dispatches to `bin_by_boundaries` for (left, right) tuples
    or to `bin_by_edges` for monotonically increasing bin edge arrays.
    Works with multi-dimensional data, preserving dimensions except binned coordinate.

    Parameters
    ----------
    data : DataArray
        The xarray DataArray to bin. Can be multi-dimensional.
    bins : np.ndarray or List[Tuple[float, float]]
        Either:
        - Monotonically increasing array of bin edges [e0, e1, e2, ...]
        - List/tuple of (left, right) tuples for arbitrary bins
    coord_name : str, optional
        Name of the coordinate to bin along (default: "t" for time).
    labels : np.ndarray, optional
        Labels for the bins. If None, uses bin centers.
    reduce_func : str, optional
        Reduction function to apply: "mean", "sum", "std", "var", "count", etc.

    Returns
    -------
    DataArray
        Binned data with the binned coordinate as the first dimension,
        followed by all other dimensions from the input data.

    Examples
    --------
    >>> # Monotonically increasing bin edges (non-overlapping)
    >>> bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    >>> binned = bin_dataarray(data, bin_edges, coord_name="t")

    >>> # (left, right) tuples (supports overlapping)
    >>> bins = [(0.5, 1.5), (1.3, 2.3), (2.1, 3.1)]
    >>>    or
    >>> bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    >>> binned = bin_dataarray(data, bins, coord_name="t")

    >>> # Multi-dimensional data supported (e.g., time and radial coord)
    >>> bins = [(0.5, 1.5), (1.3, 2.3)]
    >>> binned = bin_dataarray(data_2d, bins, coord_name="t")
    >>> # Result has shape (n_bins, n_radial_points)
    """
    # Check if bins are (left, right) tuples
    is_tuple_format = isinstance(bins[0], (tuple, list))

    if is_tuple_format:
        # Use bin_by_boundaries for (left, right) tuples
        return bin_by_boundaries(
            data, bins, coord_name=coord_name, labels=labels, reduce_func=reduce_func
        )
    else:
        # Use bin_by_edges for monotonically increasing edges
        return bin_by_edges(
            data, bins, coord_name=coord_name, labels=labels, reduce_func=reduce_func
        )


def bin_by_boundaries(
    data: DataArray,
    bins: Union[Tuple[Tuple[float, float], ...], List[Tuple[float, float]]],
    coord_name: str = "t",
    labels: np.ndarray = None,
    reduce_func: str = "mean",
) -> DataArray:
    """
    Bin a DataArray using custom left and right boundaries for each bin.

    Supports arbitrary and overlapping bins by manually selecting and reducing
    data within each bin's boundaries. Works with multi-dimensional data,
    preserving all dimensions except the binned coordinate.

    Parameters
    ----------
    data : DataArray
        The xarray DataArray to bin. Can be multi-dimensional.
    bins : Tuple[Tuple[float, float], ...] or List[Tuple[float, float]]
        Sequence of (left, right) tuples specifying bin boundaries.
        Bins can be overlapping or non-contiguous.
    coord_name : str, optional
        Name of the coordinate to bin along (default: "t" for time).
    labels : np.ndarray, optional
        Labels for the bins. If None, uses bin centers.
    reduce_func : str, optional
        Reduction function to apply: "mean", "sum", "std", "var", etc.

    Returns
    -------
    DataArray
        Binned data with the binned coordinate as the first dimension,
        followed by all other dimensions from the input data.

    Examples
    --------
    >>> # 1D data
    >>> bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    >>> binned = bin_by_boundaries(data, bins, coord_name="t")

    >>> # 2D data (e.g., time and radial coordinates)
    >>> bins = [(0.5, 1.5), (1.3, 2.3), (2.1, 3.1)]
    >>> binned = bin_by_boundaries(data, bins, coord_name="t")
    >>> # Result has shape (n_bins, n_radial_points)

    >>> # Overlapping bins are also supported
    >>> bins = [(0.5, 1.5), (1.3, 2.3), (2.1, 3.1)]
    >>> binned = bin_by_boundaries(data, bins, coord_name="t")
    """

    if labels is None:
        labels = np.array([0.5 * (left + right) for left, right in bins])

    # Manually bin the data by iterating through each bin
    binned_results = []
    coord_vals = data[coord_name].values
    for left, right in bins:
        # Select data within bin boundaries using boolean indexing
        # This ensures we capture all data points where left <= coord <= right
        mask = (coord_vals >= left) & (coord_vals <= right)
        in_bin = data.isel({coord_name: mask})

        # Apply reduction function along the binned coordinate
        if reduce_func == "mean":
            binned_val = in_bin.mean(dim=coord_name, keep_attrs=True, skipna=True)
        elif reduce_func == "sum":
            binned_val = in_bin.sum(dim=coord_name, keep_attrs=True, skipna=True)
        elif reduce_func == "std":
            binned_val = in_bin.std(dim=coord_name, keep_attrs=True, skipna=True)
        elif reduce_func == "var":
            binned_val = in_bin.var(dim=coord_name, keep_attrs=True, skipna=True)
        elif reduce_func == "count":
            binned_val = in_bin.count(dim=coord_name)
        else:
            binned_val = in_bin.reduce(reduce_func, dim=coord_name)

        binned_results.append(binned_val)

    # Determine the dimensions for the result
    # First dimension is the binned coordinate, others are preserved
    other_dims = [d for d in data.dims if d != coord_name]
    all_dims = [coord_name] + other_dims

    # Create coordinates for the result
    coords = {coord_name: labels}
    for dim in other_dims:
        coords[dim] = data.coords[dim]

    # Stack results along the binned coordinate
    result = DataArray(binned_results, coords=coords, dims=all_dims)

    # Preserve attributes from original data
    if data.attrs:
        result.attrs = data.attrs

    return result


def bin_by_edges(
    data: DataArray,
    bin_edges: np.ndarray,
    coord_name: str = "t",
    labels: np.ndarray = None,
    reduce_func: str = "mean",
) -> DataArray:
    """
    Bin a DataArray using monotonically increasing bin edge values.

    Handles non-overlapping bins using xarray's efficient groupby_bins.

    Parameters
    ----------
    data : DataArray
        The xarray DataArray to bin.
    bin_edges : np.ndarray
        Monotonically increasing array of bin edges. If length is n, creates n-1 bins.
    coord_name : str, optional
        Name of the coordinate to bin along (default: "t" for time).
    labels : np.ndarray, optional
        Labels for the bins. If None, uses bin centers.
    reduce_func : str, optional
        Reduction function to apply: "mean", "sum", "std", "var", "count", etc.

    Returns
    -------
    DataArray
        Binned data with bins as the new coordinate.

    Examples
    --------
    >>> bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    >>> binned = bin_by_edges(data, bin_edges, coord_name="t")
    """

    if labels is None:
        labels = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    grouped = data.groupby_bins(coord_name, bins=bin_edges, labels=labels)

    # Apply reduction function
    if reduce_func == "mean":
        result = grouped.mean(coord_name, keep_attrs=True, skipna=True)
    elif reduce_func == "sum":
        result = grouped.sum(coord_name, keep_attrs=True, skipna=True)
    elif reduce_func == "std":
        result = grouped.std(coord_name, keep_attrs=True, skipna=True)
    elif reduce_func == "var":
        result = grouped.var(coord_name, keep_attrs=True, skipna=True)
    elif reduce_func == "count":
        result = grouped.count()
    else:
        result = grouped.reduce(reduce_func)

    # Rename the binned coordinate back to original name
    result = result.rename({f"{coord_name}_bins": coord_name})

    return result
