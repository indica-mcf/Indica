"""Routines for averaging or interpolate along the time axis given start and end time
and a desired time resolution"""

import numpy as np
from xarray import DataArray

from indica.converters.binning import bin_dataarray


def convert_in_time_dt(
    tstart: float,
    tend: float,
    dt: float,
    overlap: float,
    data: DataArray,
    method: str = "linear",
    check_bounds: bool = True,
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

    Returns
    -------
    :
        Array like the input, but interpolated/binned along the time axis.

    """

    tcoords = data.coords["t"]
    data_dt = tcoords[1] - tcoords[0]
    if data_dt <= dt / 2 and tstart != tend:
        _result = bin_in_time_dt(
            tstart, tend, dt, overlap, data, check_bounds=check_bounds
        )
    else:
        _result = interpolate_in_time_dt(
            tstart, tend, dt, data, method=method, check_bounds=check_bounds
        )
    return _result


def bin_in_time_dt(
    tstart: float,
    tend: float,
    dt: float,
    overlap: float,
    data: DataArray,
    check_bounds: bool = True,
) -> DataArray:
    """Bin given data along the time axis, discarding data before or after
    the limits.
    """
    if check_bounds:
        check_bounds_bin(tstart, tend, dt, data)
    tlabels = get_tlabels_dt(tstart, tend, dt)
    return bin_to_time_labels(tlabels, overlap, data)


def interpolate_in_time_dt(
    tstart: float,
    tend: float,
    dt: float,
    data: DataArray,
    method: str = "linear",
    check_bounds: bool = True,
) -> DataArray:
    """Interpolate the given data along the time axis, discarding data
    before or after the limits.
    """

    if check_bounds:
        check_bounds_interp(tstart, tend, data)
    tlabels = get_tlabels_dt(tstart, tend, dt)

    return interpolate_to_time_labels(tlabels, data, method=method)


def interpolate_to_time_labels(
    tlabels: np.ndarray, data: DataArray, method: str = "linear"
) -> DataArray:
    """
    Interpolate data to sit on the specified time labels.
    """
    if data.coords["t"].shape == tlabels.shape and np.all(data.coords["t"] == tlabels):
        return data

    interpolated = data.interp(t=tlabels, method=method)
    dims = interpolated.dims
    if "error" not in data.coords:
        error = np.full_like(interpolated.data, 0)
        interpolated = interpolated.assign_coords(error=(dims, error))
    interpolated = interpolated.assign_coords(
        stdev=(dims, np.full_like(interpolated.data, 0))
    )

    return interpolated


def bin_to_time_labels(
    tlabels: np.ndarray, overlap: float, data: DataArray
) -> DataArray:
    """Bin data to sit on the specified time labels."""
    if data.coords["t"].shape == tlabels.shape and np.all(data.coords["t"] == tlabels):
        return data

    if overlap > 0:
        dt = tlabels[1] - tlabels[0]
        half_interval = 0.5 * dt + overlap * dt / 2
        tbins = []
        for _t in tlabels:
            tbins.append((_t - half_interval, _t + half_interval))
    else:
        npoints = len(tlabels)
        half_interval = 0.5 * (tlabels[1] - tlabels[0])
        tbins = np.empty(npoints + 1)
        tbins[0] = tlabels[0] - half_interval
        tbins[1:] = tlabels + half_interval

    averaged = bin_dataarray(data, tbins, coord_name="t", labels=tlabels)
    dims = averaged.dims

    stdev = bin_dataarray(
        data, tbins, coord_name="t", labels=tlabels, reduce_func="std"
    ).data
    averaged = averaged.assign_coords(stdev=(dims, stdev))

    if "error" in data.coords:
        # Propagate error by summing in quadrature and dividing by count
        _error = bin_dataarray(
            data.error**2, tbins, coord_name="t", labels=tlabels, reduce_func="sum"
        )
        _count = bin_dataarray(
            data.error**2, tbins, coord_name="t", labels=tlabels, reduce_func="count"
        )
        error = (np.sqrt(_error) / _count).data
    else:
        error = np.full_like(averaged.data, 0)
    averaged = averaged.assign_coords(error=(dims, error))

    return averaged


def get_tlabels_dt(tstart: float, tend: float, dt: float):
    """
    Build time array given start, end and frequency
    """
    _tlabels = np.arange(tstart, tend + dt, dt)
    tlabels = _tlabels[np.logical_or(_tlabels < tend, np.isclose(_tlabels, tend))]
    return np.array(tlabels, ndmin=1)


def check_bounds_bin(tstart: float, tend: float, dt: float, data: DataArray):
    """
    Check necessary bounds for binning data in time
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
    Check necessary bounds for interpolating in time
    """
    tcoords = data.coords["t"]
    start = np.argmax((tcoords > tstart).data) - 1
    if start < 0:
        raise ValueError("Start time {} not in range of provided data.".format(tstart))
    end = np.argmax((tcoords >= tend).data)
    if end < 1:
        raise ValueError("End time {} not in range of provided data.".format(tend))

    return
