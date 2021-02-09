"""Routines for averaging or interpolate along the time axis."""

import numpy as np
from xarray import DataArray


def convert_in_time(
    tstart: float,
    tend: float,
    frequency: float,
    data: DataArray,
    method: str = "linear",
) -> DataArray:
    """Interpolate or bin (as appropriate) the given data along the time
    axis, discarding data before or after the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tup
        The upper limit in time for determining which data to retain.
    frequency
        Frequency of sampling on the time axis.
    data
        Data to be interpolated/binned.
    method
        Interpolation method to use. Must be a value accepted by
        :py:class:`scipy.interpolate.interp1d`.

    Returns
    -------
    :
        Array like the input, but interpolated or binned along the time axis.

    """
    original_freq = 1 / (data.coords["t"][1] - data.coords["t"][0])
    if frequency / original_freq <= 0.2:
        return bin_in_time(tstart, tend, frequency, data)
    else:
        return interpolate_in_time(tstart, tend, frequency, data, method)


def interpolate_in_time(
    tstart: float,
    tend: float,
    frequency: float,
    data: DataArray,
    method: str = "linear",
) -> DataArray:
    """Interpolate the given data along the time axis, discarding data
    before or after the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tup
        The upper limit in time for determining which data to retain.
    frequency
        Frequency of sampling on the time axis.
    data
        Data to be interpolated.
    method
        Interpolation method to use. Must be a value accepted by
        :py:class:`scipy.interpolate.interp1d`.

    Returns
    -------
    :
        Array like the input, but interpolated along the time axis.

    """
    # Old implementation from abstract reader. Will likely need to be changed.
    tcoords = data.coords["t"]
    start = np.argmax((tcoords > tstart).data) - 1
    if start < 0:
        raise ValueError("Start time {} not in range of provided data.".format(tstart))
    end = np.argmax((tcoords >= tend).data)
    if end < 1:
        raise ValueError("End time {} not in range of provided data.".format(tend))
    npoints = round((tend - tstart) * frequency) + 1
    tvals = np.linspace(tstart, tend, npoints)
    cleaned_data = data.indica.with_ignored_data
    result = cleaned_data.interp(t=tvals, method=method)
    if "error" in data.attrs:
        result.attrs["error"] = cleaned_data.attrs["error"].interp(
            t=tvals, method=method
        )
    if "dropped" in data.attrs:
        ddim = data.indica.drop_dim
        result = result.indica.ignore_data(data.attrs["dropped"].coords[ddim], ddim)

    return result


def bin_to_time_labels(tlabels: np.ndarray, data: DataArray) -> DataArray:
    """Bin data to sit on the specified time labels.

    Parameters
    ----------
    tlabels
        The times at which the data should be binned.
    data
        Data to be binned.

    Returns
    -------
    :
        Array like the input, but binned onto the time labels.

    """
    if data.coords["t"].shape == tlabels.shape and np.all(data.coords["t"] == tlabels):
        return data
    npoints = len(tlabels)
    half_interval = 0.5 * (tlabels[1] - tlabels[0])
    tbins = np.empty(npoints + 1)
    tbins[0] = tlabels[0] - half_interval
    tbins[1:] = tlabels + half_interval
    grouped = data.sel(t=slice(tbins[0], tbins[-1])).groupby_bins(
        "t", tbins, labels=tlabels
    )
    averaged = grouped.mean("t", keep_attrs=True)
    if "error" in data.attrs:
        grouped = (
            data.attrs["error"]
            .sel(t=slice(tbins[0], tbins[-1]))
            .groupby_bins("t", tbins, labels=tlabels)
        )
        uncertainty = np.sqrt(
            grouped.reduce(
                lambda x, axis: np.sum(x ** 2, axis) / np.size(x, axis) ** 2, "t"
            )
        )
        averaged.attrs["error"] = uncertainty.rename(t_bins="t")
    if "dropped" in data.attrs:
        grouped = (
            data.attrs["dropped"]
            .sel(t=slice(tbins[0], tbins[-1]))
            .groupby_bins("t", tbins, labels=tlabels)
        )
        dropped = grouped.mean("t")
        averaged.attrs["dropped"] = dropped.rename(t_bins="t")
        if "error" in data.attrs:
            grouped = (
                data.attrs["dropped"]
                .attrs["error"]
                .sel(t=slice(tbins[0], tbins[-1]))
                .groupby_bins("t", tbins, labels=tlabels)
            )
            uncertainty = np.sqrt(
                grouped.reduce(
                    lambda x, axis: np.sum(x ** 2, axis) / np.size(x, axis) ** 2, "t"
                )
            )
            averaged.attrs["dropped"].attrs["error"] = uncertainty.rename(t_bins="t")

    return averaged.rename(t_bins="t")


def bin_in_time(
    tstart: float, tend: float, frequency: float, data: DataArray
) -> DataArray:
    """Bin given data along the time axis, discarding data before or after
    the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tup
        The upper limit in time for determining which data to retain.
    frequency
        Frequency of sampling on the time axis.
    data
        Data to be binned.

    Returns
    -------
    :
        Array like the input, but binned along the time axis.

    """
    npoints = round(abs(tend - tstart) * frequency) + 1
    half_interval = 0.5 * (tend - tstart) / (npoints - 1)
    tcoords = data.coords["t"]
    if tcoords[0] > tstart + half_interval:
        raise ValueError(
            "No data falls within first bin {}.".format(
                (tstart - half_interval, tstart + half_interval)
            )
        )
    if tcoords[-1] < tend - half_interval:
        raise ValueError(
            "No data falls within last bin {}.".format(
                (tend - half_interval, tend + half_interval)
            )
        )
    tlabels = np.linspace(tstart, tend, npoints)
    return bin_to_time_labels(tlabels, data)
