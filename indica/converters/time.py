"""
Routines for averaging or interpolating along the time axis.
Takes start and end times and a desired time resolution.
"""

import numpy as np
from xarray import DataArray
from xarray.core.types import InterpOptions


def strip_provenance(arr: DataArray):
    """
    Remove provenance information from a DataArray if present.

    Parameters
    ----------
    arr
        DataArray to strip provenance from.

    """
    if "provenance" in arr.attrs:
        del arr.attrs["partial_provenance"]
        del arr.attrs["provenance"]


def convert_in_time(
    tstart: float,
    tend: float,
    frequency: float,
    data: DataArray,
    method: InterpOptions = "linear",
) -> DataArray:
    """
    Interpolate or bin (as appropriate) the given data along the time
    axis, discarding data before or after the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
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
    # TODO: Generate PROV information
    original_freq = 1 / (data.coords["t"][1] - data.coords["t"][0])
    if frequency / original_freq <= 0.2:
        return bin_in_time(tstart, tend, frequency, data)
    else:
        return interpolate_in_time(tstart, tend, frequency, data, method)


def convert_in_time_dt(
    tstart: float,
    tend: float,
    dt: float,
    data: DataArray,
    method: InterpOptions = "linear",
) -> DataArray:
    """
    Interpolate or bin given data along the time axis, discarding data before
    or after the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
        The upper limit in time for determining which data to retain.
    dt
        Time resolution of new time axis.
    data
        Data to be interpolated/binned.
    method
        Interpolation method to use. Default "linear".

    Returns
    -------
    :
        Array like the input, but interpolated/binned along the time axis.

    """
    tcoords = data.coords["t"]
    data_dt = tcoords[1] - tcoords[0]
    if data_dt <= dt / 2:
        return bin_in_time_dt(tstart, tend, dt, data)
    else:
        return interpolate_in_time_dt(tstart, tend, dt, data, method=method)


def interpolate_to_time_labels(
    tlabels: np.ndarray, data: DataArray, method: InterpOptions = "linear"
) -> DataArray:
    """
    Interpolate data to sit on the specified time labels.

    Parameters
    ----------
    tlabels
        The times at which the data should be interpolated.
    data
        Data to be interpolated.
    method
        Interpolation method to use. Default "linear".

    Returns
    -------
    :
        Array like the input, but interpolated onto the time labels.

    """
    # No interpolation required if the current t coordinates already match the desired
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

    if "transform" in data.attrs:
        interpolated.attrs["transform"] = data.attrs["transform"]

    strip_provenance(interpolated)

    return interpolated


def bin_to_time_labels(tlabels: np.ndarray, data: DataArray) -> DataArray:
    """
    Bin data to sit on the specified time labels.

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
    # No binning required if the current t coordinates already match the desired
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
            (data.attrs["error"] ** 2)
            .sel(t=slice(tbins[0], tbins[-1]))
            .groupby_bins("t", tbins, labels=tlabels)
        )
        uncertainty_square = grouped.sum("t") / grouped.count("t")
        error = (uncertainty_square + stdev**2) ** 0.5
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
        if "error" in data.attrs["dropped"].attrs:
            grouped = (
                data.attrs["dropped"]
                .attrs["error"]
                .sel(t=slice(tbins[0], tbins[-1]))
                .groupby_bins("t", tbins, labels=tlabels)
            )
            uncertainty_square = grouped.sum("t") / grouped.count("t")
            error = (uncertainty_square + stdev**2) ** 0.5
            averaged.attrs["dropped"].attrs["error"] = error.rename(t_bins="t")

    if "transform" in data.attrs:
        averaged.attrs["transform"] = data.attrs["transform"]

    strip_provenance(averaged)

    return averaged.rename(t_bins="t")


def interpolate_in_time(
    tstart: float,
    tend: float,
    frequency: float,
    data: DataArray,
    method: InterpOptions = "linear",
) -> DataArray:
    """
    Interpolate the given data along the time axis, discarding data
    before or after the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
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
    check_bounds_interp(tstart, tend, data)

    tlabels = get_tlabels(tstart, tend, frequency)

    return interpolate_to_time_labels(tlabels, data, method=method)


def interpolate_in_time_dt(
    tstart: float,
    tend: float,
    dt: float,
    data: DataArray,
    method: InterpOptions = "linear",
) -> DataArray:
    """
    Interpolate the given data along the time axis, discarding data
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
    check_bounds_interp(tstart, tend, data)
    tlabels = get_tlabels_dt(tstart, tend, dt)

    return interpolate_to_time_labels(tlabels, data, method=method)


def bin_in_time(
    tstart: float, tend: float, frequency: float, data: DataArray
) -> DataArray:
    """
    Bin given data along the time axis, discarding data before or after
    the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
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
    check_bounds_bin(tstart, tend, 1.0 / frequency, data)
    tlabels = get_tlabels(tstart, tend, frequency)

    return bin_to_time_labels(tlabels, data)


def bin_in_time_dt(tstart: float, tend: float, dt: float, data: DataArray) -> DataArray:
    """
    Bin given data along the time axis, discarding data before or after
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
    check_bounds_bin(tstart, tend, dt, data)
    tlabels = get_tlabels_dt(tstart, tend, dt)
    return bin_to_time_labels(tlabels, data)


def get_tlabels(tstart: float, tend: float, frequency: float):
    """
    Build time array given start, end and frequency.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
        The upper limit in time for determining which data to retain.
    frequency
        Frequency of sampling on the time axis.

    Returns
    -------
    tlabels
        Time array

    """
    npoints = round((tend - tstart) * frequency) + 1
    return np.linspace(tstart, tend, npoints)


def get_tlabels_dt(tstart: float, tend: float, dt: float):
    """
    Build time array given start, end and frequency.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
        The upper limit in time for determining which data to retain.
    dt
        Time resolution of new time axis.

    Returns
    -------
    tlabels
        Time array

    """
    tlabels = np.arange(tstart, tend + dt, dt)
    return tlabels


def check_bounds_bin(tstart: float, tend: float, dt: float, data: DataArray):
    """
    Check necessary bounds for binning data in time.

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

    Raises
    ------
    ValueError
        If not data falls within either the first or last bin.

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


def check_bounds_interp(tstart: float, tend: float, data: DataArray):
    """
    Check necessary bounds for interpolating in time.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tend
        The upper limit in time for determining which data to retain.
    data
        Data to be interpolated.

    Raises
    ------
    ValueError
        If either ``tstart`` or ``tend`` is not in range of provided data.

    """
    tcoords = data.coords["t"]
    start = np.argmax((tcoords > tstart).data) - 1
    if start < 0:
        raise ValueError("Start time {} not in range of provided data.".format(tstart))
    end = np.argmax((tcoords >= tend).data)
    if end < 1:
        raise ValueError("End time {} not in range of provided data.".format(tend))


#
# def run_example(nt=50, plot=False):
#     values = np.sin(np.linspace(0, np.pi * 3, nt)) + np.random.random(nt) - 0.5
#     time = np.linspace(0, 0.1, nt)
#     data = DataArray(values, coords=[("t", time)])
#
#     dt = time[1] - time[0]
#     dt_binned = dt * 4
#     dt_interp = dt / 4
#
#     tstart = time[0] + 5 * dt
#     tend = time[-1] - 10 * dt
#     data_interp = convert_in_time_dt(tstart, tend, dt_interp, data)
#     data_binned = convert_in_time_dt(tstart, tend, dt_binned, data)
#
#     if plot:
#         import matplotlib.pylab as plt
#
#         plt.figure()
#         data_interp.plot(marker="x", label="Interpolated")
#         data.plot(marker="o", label="Original data")
#         data_binned.plot(marker="x", label="Binned")
#         plt.legend()
#         plt.show()
