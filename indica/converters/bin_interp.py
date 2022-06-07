"""Routines for averaging or interpolate along an axis given start, stop and bin size"""

import numpy as np
from xarray import DataArray


def convert(
    start: float,
    stop: float,
    step: float,
    data: DataArray,
    dim: str,
    method: str = "linear",
) -> DataArray:
    """Bin or interpolate data along specified dimension, discarding data before
    or after the limits.

    Parameters
    ----------
    start
        Start of interval.  The interval includes this value.
    stop
        End of interval.  The interval includes this value.
    step
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.
    data
        Data to be binned.
    dim
        Dimension along which data is to be binned

    Returns
    -------
    :
        Array like the input, but binned/interpolated along the desired dimension

    """

    coords = data.coords[dim]
    data_step = coords[1] - coords[0]
    if data_step <= step / 2:
        return bin_to_dim(start, stop, step, data, dim)
    else:
        return interpolate_to_dim(start, stop, step, data, dim, method=method)


def interpolate_to_labels(
    labels: np.ndarray,
    data: DataArray,
    dim: str,
    method: str = "linear",
) -> DataArray:
    """Interpolate data to sit on the specified dimension labels.

    Parameters
    ----------
    labels
        The values at which the data should be binned.
    data
        Data to be binned.

    Returns
    -------
    :
        Array like the input, but binned onto the dimension labels.

    """
    if data.coords[dim].shape == labels.shape and np.all(data.coords[dim] == labels):
        return data

    interpolated = data.interp(dict([(dim, labels)]), method=method)
    if "error" in data.attrs:
        interpolated.attrs["error"] = interpolated.attrs["error"].interp(
            dict([(dim, labels)]), method=method
        )
    if "dropped" in data.attrs:
        dropped = interpolated.attrs["dropped"].interp(
            dict([(dim, labels)]), method=method
        )
        if "error" in dropped.attrs:
            dropped.attrs["error"] = dropped.attrs["error"].interp(
                dict([(dim, labels)]), method=method
            )
        interpolated.attrs["dropped"] = dropped
    if "provenance" in data.attrs:
        del interpolated.attrs["partial_provenance"]
        del interpolated.attrs["provenance"]

    return interpolated


def bin_to_labels(labels: np.ndarray, data: DataArray, dim: str) -> DataArray:
    """Bin data to sit on the specified dimension labels.

    Parameters
    ----------
    labels
        The values at which the data should be binned.
    data
        Data to be binned.

    Returns
    -------
    :
        Array like the input, but binned onto the dimension labels.

    """
    if data.coords[dim].shape == labels.shape and np.all(data.coords[dim] == labels):
        return data
    npoints = len(labels)
    half_interval = 0.5 * (labels[1] - labels[0])
    bins = np.empty(npoints + 1)
    bins[0] = labels[0] - half_interval
    bins[1:] = labels + half_interval
    grouped = data.sel(dict([(dim, slice(bins[0], bins[-1]))])).groupby_bins(
        dim, bins, labels=labels
    )
    averaged = grouped.mean(dim, keep_attrs=True)
    stdev = grouped.std(dim, keep_attrs=True)

    if "error" in data.attrs:
        grouped = (
            data.attrs["error"]
            .sel(dict([(dim, slice(bins[0], bins[-1]))]))
            .groupby_bins(dim, bins, labels=labels)
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
            .groupby_bins(dim, bins, labels=labels)
        )
        dropped = grouped.mean(dim)
        stdev = grouped.std(dim)
        averaged.attrs["dropped"] = dropped.rename(dict([(f"{dim}_bins", dim)]))
        if "error" in data.attrs["dropped"].attrs:
            grouped = (
                data.attrs["dropped"]
                .attrs["error"]
                .sel(dict([(dim, slice(bins[0], bins[-1]))]))
                .groupby_bins(dim, bins, labels=labels)
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


def interpolate_to_dim(
    start: float,
    stop: float,
    step: float,
    data: DataArray,
    dim: str,
    method: str = "linear",
) -> DataArray:
    """Bin given data along specified dimension, discarding data before or after
    the limits.

    Parameters
    ----------
    start
        Start of interval.  The interval includes this value.
    stop
        End of interval.  The interval includes this value.
    step
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.
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

    check_bounds(start, stop, step, data, dim)
    labels = get_labels(start, stop, step)

    return interpolate_to_labels(labels, data, dim, method=method)


def bin_to_dim(
    start: float, stop: float, step: float, data: DataArray, dim: str
) -> DataArray:
    """Bin given data along the dim axis, discarding data before or after
    the limits.

    Parameters
    ----------
    start
        Start of interval.  The interval includes this value.
    stop
        End of interval.  The interval includes this value.
    step
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.
    data
        Data to be binned.
    dim
        Dimension along which data is to be binned

    Returns
    -------
    :
        Array like the input, but binned along the dim axis.

    """
    check_bounds(start, stop, step, data, dim)
    labels = get_labels(start, stop, step)
    return bin_to_labels(labels, data, dim)


def get_labels(start: float, stop: float, step: float):
    """
    Build array given start, stop and bin step

    Parameters
    ----------
    start
        Start of interval.  The interval includes this value.
    stop
        End of interval.  The interval includes this value.
    step
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.

    Returns
    -------
    labels
        Binned dimension array

    """
    labels = np.arange(start, stop + step, step)
    return labels


def check_bounds(start: float, stop: float, step: float, data: DataArray, dim: str):
    """
    Check necessary bounds for binning/interpolating data in time

    Parameters
    ----------
    start
        Start of interval.  The interval includes this value.
    stop
        End of interval.  The interval includes this value.
    step
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.
    data
        Data to be binned.
    dim
        Dimension along which data is to be binned
    """

    coords = data.coords[dim]
    data_step = coords[1] - coords[0]
    half_interval = step / 2

    # For both binning and interpolating
    if start < coords.min():
        raise ValueError("Start {} not in range of provided data.".format(start))
    if stop > coords.max():
        raise ValueError("End {} not in range of provided data.".format(stop))

    # For binning only
    if data_step <= half_interval:
        if coords[0] > start + half_interval:
            raise ValueError(
                "No data falls within first bin {}.".format(
                    (start - half_interval, start + half_interval)
                )
            )
        if coords[-1] < stop - half_interval:
            raise ValueError(
                "No data falls within last bin {}.".format(
                    (stop - half_interval, stop + half_interval)
                )
            )

    return
