"""Routines for averaging or interpolate along an axis given start, end and bin size"""

import numpy as np
from xarray import DataArray


def bin_dimension(
    start: float,
    end: float,
    width: float,
    data: DataArray,
    dim: str,
    method: str = "linear",
) -> DataArray:
    """Bin given data along specified dimension, discarding data before or after
    the limits.

    Parameters
    ----------
    start
        The lower limit along the desired dimension to determine which data to retain.
    end
        The upper limit along the desired dimension to determine which data to retain.
    width
        Bin width
    data
        Data to be binned.
    dim
        Dimension along which data is to be binned

    Returns
    -------
    :
        Array like the input, but binned along the desired dimension

    """

    coords = data.coords[dim]
    data_width = coords[1] - coords[0]
    if data_width <= width / 2:
        return bin_on_dim(start, end, width, data, dim)
    else:
        return interpolate_on_dim(start, end, width, data, dim, method=method)


def interpolate_to_dim_labels(
    dim_labels: np.ndarray,
    data: DataArray,
    dim: str,
    method: str = "linear",
) -> DataArray:
    """Interpolate data to sit on the specified dimension labels.

    Parameters
    ----------
    dim_labels
        The values at which the data should be binned.
    data
        Data to be binned.

    Returns
    -------
    :
        Array like the input, but binned onto the dimension labels.

    """
    if data.coords[dim].shape == dim_labels.shape and np.all(
        data.coords[dim] == dim_labels
    ):
        return data

    interpolated = data.interp(dict([(dim, dim_labels)]), method=method)
    if "error" in data.attrs:
        interpolated.attrs["error"] = interpolated.attrs["error"].interp(
            dict([(dim, dim_labels)]), method=method
        )
    if "dropped" in data.attrs:
        dropped = interpolated.attrs["dropped"].interp(
            dict([(dim, dim_labels)]), method=method
        )
        if "error" in dropped.attrs:
            dropped.attrs["error"] = dropped.attrs["error"].interp(
                dict([(dim, dim_labels)]), method=method
            )
        interpolated.attrs["dropped"] = dropped
    if "provenance" in data.attrs:
        del interpolated.attrs["partial_provenance"]
        del interpolated.attrs["provenance"]

    return interpolated


def bin_to_dim_labels(dim_labels: np.ndarray, data: DataArray, dim: str) -> DataArray:
    """Bin data to sit on the specified dimension labels.

    Parameters
    ----------
    dim_labels
        The values at which the data should be binned.
    data
        Data to be binned.

    Returns
    -------
    :
        Array like the input, but binned onto the dimension labels.

    """
    if data.coords[dim].shape == dim_labels.shape and np.all(
        data.coords[dim] == dim_labels
    ):
        return data
    npoints = len(dim_labels)
    half_interval = 0.5 * (dim_labels[1] - dim_labels[0])
    bins = np.empty(npoints + 1)
    bins[0] = dim_labels[0] - half_interval
    bins[1:] = dim_labels + half_interval
    grouped = data.sel(dict([(dim, slice(bins[0], bins[-1]))])).groupby_bins(
        dim, bins, labels=dim_labels
    )
    averaged = grouped.mean(dim, keep_attrs=True)
    stdev = grouped.std(dim, keep_attrs=True)

    if "error" in data.attrs:
        grouped = (
            data.attrs["error"]
            .sel(dict([(dim, slice(bins[0], bins[-1]))]))
            .groupby_bins(dim, bins, labels=dim_labels)
        )
        uncertainty = np.sqrt(
            grouped.reduce(
                lambda x, axis: np.sum(x**2, axis) / np.size(x, axis) ** 2, dim
            )
        )
        error = np.sqrt(uncertainty**2 + stdev**2)
        averaged.attrs["error"] = error.rename(dict([(f"{dim}_bins", dim)]))
    if "dropped" in data.attrs:
        grouped = (
            data.attrs["dropped"]
            .sel(dict([(dim, slice(bins[0], bins[-1]))]))
            .groupby_bins(dim, bins, labels=dim_labels)
        )
        dropped = grouped.mean(dim)
        stdev = grouped.std(dim)
        averaged.attrs["dropped"] = dropped.rename(dict([(f"{dim}_bins", dim)]))
        if "error" in data.attrs["dropped"].attrs:
            grouped = (
                data.attrs["dropped"]
                .attrs["error"]
                .sel(dict([(dim, slice(bins[0], bins[-1]))]))
                .groupby_bins(dim, bins, labels=dim_labels)
            )
            uncertainty = np.sqrt(
                grouped.reduce(
                    lambda x, axis: np.sum(x**2, axis) / np.size(x, axis) ** 2, dim
                )
            )
            error = np.sqrt(uncertainty**2 + stdev**2)
            averaged.attrs["dropped"].attrs["error"] = error.rename(
                dict([(f"{dim}_bins", dim)])
            )
    if "provenance" in data.attrs:
        del averaged.attrs["partial_provenance"]
        del averaged.attrs["provenance"]

    return averaged.rename(dict([(f"{dim}_bins", dim)]))


def interpolate_on_dim(
    start: float,
    end: float,
    width: float,
    data: DataArray,
    dim: str,
    method: str = "linear",
) -> DataArray:
    """Bin given data along specified dimension, discarding data before or after
    the limits.

    Parameters
    ----------
    start
        The lower limit along the desired dimension to determine which data to retain.
    end
        The upper limit along the desired dimension to determine which data to retain.
    width
        Bin width
    data
        Data to be binned.
    dim
        Dimension along which data is to be binned
    method
        Interpolation method to use. Must be a value accepted by
        :py:class:`scipy.interpolate.interp1d`.

    Returns
    -------
    :
        Array like the input, but binned along the desired dimension

    """

    check_bounds_interp(start, end, data, dim)
    dim_labels = get_dim_labels(start, end, width)

    return interpolate_to_dim_labels(dim_labels, data, dim, method=method)


def bin_on_dim(
    start: float, end: float, width: float, data: DataArray, dim: str
) -> DataArray:
    """Bin given data along the dim axis, discarding data before or after
    the limits.

    Parameters
    ----------
    start
        The lower limit along the desired dimension to determine which data to retain.
    end
        The upper limit along the desired dimension to determine which data to retain.
    width
        Bin width
    data
        Data to be binned.
    dim
        Dimension along which data is to be binned

    Returns
    -------
    :
        Array like the input, but binned along the dim axis.

    """
    check_bounds_bin(start, end, width, data, dim)
    dim_labels = get_dim_labels(start, end, width)
    return bin_to_dim_labels(dim_labels, data, dim)


def get_dim_labels(start: float, end: float, width: float):
    """
    Build array given start, end and bin width

    Parameters
    ----------
    start
        The lower limit along the desired dimension to determine which data to retain.
    end
        The upper limit along the desired dimension to determine which data to retain.
    width
        Bin width

    Returns
    -------
    dim_labels
        Binned dimension array

    """
    dim_labels = np.arange(start, end + width, width)
    return dim_labels


def check_bounds_bin(start: float, end: float, width: float, data: DataArray, dim: str):
    """
    Check necessary bounds for binning data in time

    Parameters
    ----------
    start
        The lower limit along the desired dimension to determine which data to retain.
    end
        The upper limit along the desired dimension to determine which data to retain.
    width
        Bin width
    data
        Data to be binned.
    dim
        Dimension along which data is to be binned
    """
    coords = data.coords[dim]
    half_interval = width / 2
    if coords[0] > start + half_interval:
        raise ValueError(
            "No data falls within first bin {}.".format(
                (start - half_interval, start + half_interval)
            )
        )
    if coords[-1] < end - half_interval:
        raise ValueError(
            "No data falls within last bin {}.".format(
                (end - half_interval, end + half_interval)
            )
        )


def check_bounds_interp(start: float, end: float, data: DataArray, dim: str):
    """
    Check necessary bounds for binning data in time

    Parameters
    ----------
    start
        The lower limit along the desired dimension to determine which data to retain.
    end
        The upper limit along the desired dimension to determine which data to retain.
    data
        Data to be binned.
    dim
        Dimension along which data is to be binned
    """
    coords = data.coords[dim]
    _start = np.argmax((coords > start).data) - 1
    if _start < 0:
        raise ValueError("Start {} not in range of provided data.".format(start))
    _end = np.argmax((coords >= end).data)
    if _end < 1:
        raise ValueError("End {} not in range of provided data.".format(end))

    return
