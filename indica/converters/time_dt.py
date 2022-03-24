"""Routines for averaging or interpolate along the time axis given start and end times
and a desired time resolution"""

import numpy as np
from xarray import DataArray


def convert_in_time(
    tstart: float,
    tend: float,
    dt: float,
    data: DataArray,
    method="linear",
) -> DataArray:
    """Bin given data along the time axis, discarding data before or after
    the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
        The upper limit in time for determining which data to retain.
    dt
        Time resolution of new time axis.
    data
        Data to be binned.

    Returns
    -------
    :
        Array like the input, but binned along the time axis.

    """
    tlabels = np.arange(tstart, tend, dt)

    tcoords = data.coords["t"]
    data_dt = tcoords[1] - tcoords[0]

    if data_dt <= dt / 2:
        return bin_to_time_labels(tlabels, data)
    else:
        return interpolate_to_time_labels(tlabels, data, method=method)


def interpolate_to_time_labels(
    tlabels: np.ndarray, data: DataArray, method: str = "linear"
) -> DataArray:
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

    interpolated = data.interp(t=tlabels, method=method)
    if "error" in data.attrs:
        interpolated.attrs["error"] = interpolated.attrs["error"].interp(
            t=tlabels, method=method
        )
    if "dropped" in data.attrs:
        dropped = interpolated.attrs["dropped"].interp(t=tlabels, method=method)
        if "error" in dropped.attrs:
            dropped.attrs["error"] = dropped.attrs["error"].interp(
                t=tlabels, method=method
            )
        interpolated.attrs["dropped"] = dropped
    if "provenance" in data.attrs:
        del interpolated.attrs["partial_provenance"]
        del interpolated.attrs["provenance"]

    return interpolated


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
    stdev = grouped.std("t", keep_attrs=True)

    if "error" in data.attrs:
        grouped = (
            data.attrs["error"]
            .sel(t=slice(tbins[0], tbins[-1]))
            .groupby_bins("t", tbins, labels=tlabels)
        )
        uncertainty = np.sqrt(
            grouped.reduce(
                lambda x, axis: np.sum(x**2, axis) / np.size(x, axis) ** 2, "t"
            )
        )
        error = np.sqrt(uncertainty**2 + stdev**2)
        averaged.attrs["error"] = error.rename(t_bins="t")
    if "dropped" in data.attrs:
        grouped = (
            data.attrs["dropped"]
            .sel(t=slice(tbins[0], tbins[-1]))
            .groupby_bins("t", tbins, labels=tlabels)
        )
        dropped = grouped.mean("t")
        stdev = grouped.std("t")
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
                    lambda x, axis: np.sum(x**2, axis) / np.size(x, axis) ** 2, "t"
                )
            )
            error = np.sqrt(uncertainty**2 + stdev**2)
            averaged.attrs["dropped"].attrs["error"] = error.rename(t_bins="t")
    if "provenance" in data.attrs:
        del averaged.attrs["partial_provenance"]
        del averaged.attrs["provenance"]

    return averaged.rename(t_bins="t")


def interpolate_in_time(
    tstart: float,
    tend: float,
    dt: float,
    data: DataArray,
    method: str = "linear",
) -> DataArray:
    """Interpolate the given data along the time axis, discarding data
    before or after the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
        The upper limit in time for determining which data to retain.
    dt
        Time resolution of new time axis.
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

    tlabels = np.arange(tstart, tend, dt)
    return interpolate_to_time_labels(tlabels, data, method=method)


def bin_in_time(tstart: float, tend: float, dt: float, data: DataArray) -> DataArray:
    """Bin given data along the time axis, discarding data before or after
    the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
        The upper limit in time for determining which data to retain.
    dt
        Time resolution of new time axis.
    data
        Data to be binned.

    Returns
    -------
    :
        Array like the input, but binned along the time axis.

    """

    tcoords = data.coords["t"]
    half_interval = dt / 2
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
    tlabels = np.arange(tstart, tend, dt)
    return bin_to_time_labels(tlabels, data)
