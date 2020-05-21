"""Routines for averaging or interpolate along the time axis."""

from utilities import sum_squares

import numpy as np
from xarray import DataArray, sqrt


def convert_in_time(tstart: float, tend: float, interval: float,
                    data: DataArray) -> DataArray:
    """Interpolate or bin (as appropriate) the given data along the time
    axis, discarding data before or after the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tup
        The upper limit in time for determining which data to retain.
    interval
        Width of bins on the time axis.
    data
        Data to be interpolated/binned.

    Returns
    -------
    :
        Array like the input, but interpolated or binned along the time axis.

    """
    pass


def interpolate_in_time(tstart: float, tend: float, interval: float,
                        data: DataArray) -> DataArray:
    """Interpolate the given data along the time axis, discarding data
    before or after the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tup
        The upper limit in time for determining which data to retain.
    interval
        Width of bins on the time axis.
    data
        Data to be interpolated.

    Returns
    -------
    :
        Array like the input, but interpolated along the time axis.

    """
    # Old implementation from abstract reader. Will likely need to be changed.
    # start = np.argmax(t > self.times[0]) - 1
    # if start < 0:
    #     raise ValueError("Start time {} not in range of provided "
    #                      "data.".format(self.times[0]))
    # end = np.argmax(t >= self.times[-1])
    # if end < 1:
    #     raise ValueError("End time {} not in range of provided "
    #                      "data.".format(self.times[-1]))
    # slc = [slice(None)] * data.ndim
    # slc[axis] = slice(start, end)
    # f = interp1d(t[start:end], data[slc], axis=axis, copy=False,
    #              assume_sorted=True)
    # return f(self.times)


def bin_in_time(tstart: float, tend: float, interval: float,
                data: DataArray) -> DataArray:
    """Bin given data along the time axis, discarding data before or after
    the limits.

    Parameters
    ----------
    tstart
        The lower limit in time for determining which data to retain.
    tup
        The upper limit in time for determining which data to retain.
    interval
        Width of bins on the time axis.
    data
        Data to be binned.

    Returns
    -------
    :
        Array like the input, but binned along the time axis.

    """
    npoints = round((tend - tstart)/interval)
    tlabels = np.linspace(tstart, tend, npoints)
    tbins = np.empty(npoints + 1)
    tbins[0] = tstart - 0.5*interval
    tbins[1:] = tlabels + 0.5*interval

    tcoords = data.coords["t"]
    nstart = np.argmax(tcoords > tbins[0])
    if tcoords[nstart] < tbins[0] or tcoords[nstart] > tbins[1]:
        raise ValueError("Start time {} not in range of provided "
                         "data.".format(tstart))
    nend = np.argmax(tcoords > tbins[-1])
    if tcoords[nend] < tbins[-1]:
        raise ValueError("End time {} not in range of provided "
                         "data.".format(tend))
    grouped = data.isel(t=slice(nstart, nend)).groupby_bins(
        "t", tbins, tlabels)
    averaged = grouped.mean("t", keep_attrs=True)
    # TODO: determine appropriate value of DDOF (Delta Degrees of Freedom)
    variance = grouped.var("t")
    grouped = data.attrs["error"].isel(t=slice(nstart, nend)).groupby_bins(
        "t", tbins, tlabels)
    uncertainty = sqrt(grouped.reduce(sum_squares, "t") + variance)
    averaged.attrs["error"] = uncertainty.rename(t_bins="t")
    # TODO: Update provenance with this operation
    return averaged.rename(t_bins="t")
