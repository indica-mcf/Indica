"""Test interpolation/downsampling of data in time."""

from unittest.mock import patch

from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import builds
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import sampled_from
import numpy as np
from pytest import approx
from pytest import mark
from pytest import raises
from xarray import DataArray

from indica.converters.time import convert_in_time
from indica.utilities import coord_array
from .test_abstract_transform import coordinate_transforms_and_axes
from ..data_strategies import data_arrays_from_coords
from ..strategies import sane_floats

start_times = floats(50.0, 120.0)
end_times = start_times.map(lambda t: 120.0 - (t - 50.0))
samples = integers(2, 100)
methods = sampled_from(["nearest", "zero", "linear", "slinear", "quadratic", "cubic"])
t_axes = builds(np.linspace, just(50.0), just(120.0), integers(4, 50)).map(
    lambda t: coord_array(t, "t")
)


@composite
def useful_data_arrays(draw, rel_sigma=0.02, abs_sigma=1e-3):
    times = draw(t_axes)
    transform, x1, x2, _ = draw(coordinate_transforms_and_axes())
    return draw(
        data_arrays_from_coords(
            coordinates=transform,
            axes=(x1, x2, times),
            abs_sigma=abs_sigma,
            rel_sigma=rel_sigma,
        )
    )


useful_smooth_data = useful_data_arrays(
    0.0,
    0.0,
)

# Ignore warnings when averaging over a bin containign only NaNs
pytestmark = mark.filterwarnings("ignore:Mean of empty slice")


def linear_data_array(a, b, times, abs_err):
    """Create a DataArray where values are ``a*times + b``."""
    result = DataArray(a * times + b, coords=[("t", times)])
    result.attrs["error"] = DataArray(
        np.ones_like(times) * abs_err, coords=[("t", times)]
    )
    return result


@given(start_times, end_times, samples, useful_data_arrays(), methods)
def test_correct_time_values(tstart, tend, n, data, method):
    """Check always have requested time values."""
    if tstart > tend:
        tstart, tend = tend, tstart
    frequency = (n - 1) / 70.0
    assume((tend - tstart) * frequency >= 2.0)
    result = convert_in_time(tstart, tend, frequency, data, method)
    time_arrays = [result.coords["t"]]
    if "error" in data.attrs:
        time_arrays.append(result.attrs["error"].coords["t"])
    if "dropped" in data.attrs:
        time_arrays.append(result.attrs["dropped"].coords["t"])
        if "error" in data.attrs:
            time_arrays.append(result.attrs["dropped"].attrs["error"].coords["t"])
    for times in time_arrays:
        tdiffs = times[1:] - times[:-1]
        length = len(times)
        actual_frequency = (length - 1) / (tend - tstart)
        assert times[0] == approx(tstart)
        assert times[-1] == approx(tend)
        if length >= 2:
            assert np.all(tdiffs == approx(float(tdiffs[0])))
        assert abs(frequency - actual_frequency) < abs(
            frequency - length / (tend - tstart)
        )
        assert abs(frequency - actual_frequency) < abs(
            frequency - (length - 2) / (tend - tstart)
        )


@settings(report_multiple_bugs=False)
@given(start_times, end_times, samples, useful_data_arrays(), methods)
def test_unchanged_axes(tstart, tend, n, data, method):
    """Check other axes unchanged"""
    if tstart > tend:
        tstart, tend = tend, tstart
    frequency = (n - 1) / 70.0
    assume((tend - tstart) * frequency >= 2.0)
    result = convert_in_time(tstart, tend, frequency, data, method)
    all_results = [(result, data)]
    if "error" in data.attrs:
        all_results.append(
            (result.attrs["error"].coords["t"], data.attrs["error"].coords["t"])
        )
    if "dropped" in data.attrs:
        all_results.append(
            (result.attrs["dropped"].coords["t"], data.attrs["dropped"].coords["t"])
        )
        if "error" in data.attrs:
            all_results.append(
                (
                    result.attrs["dropped"].attrs["error"].coords["t"],
                    data.attrs["dropped"].attrs["error"].coords["t"],
                )
            )
    for res, orig in all_results:
        assert res.dims == orig.dims
        for dim, coords in res.coords.items():
            if dim == "t":
                continue
            assert np.all(coords == orig.coords[dim])


@given(start_times, end_times, samples, useful_data_arrays(), methods)
def test_unchanged_attrs(tstart, tend, n, data, method):
    """Check other axes unchanged"""
    data_attrs = data.attrs
    if tstart > tend:
        tstart, tend = tend, tstart
    frequency = (n - 1) / 70.0
    assume((tend - tstart) * frequency >= 2.0)
    result = convert_in_time(tstart, tend, frequency, data, method)
    assert set(result.attrs) == set(data_attrs)
    for key, val in data_attrs.items():
        if key == "error" or key == "dropped":
            continue
        assert result.attrs[key] == val


@given(
    floats(max_value=50.0, exclude_max=True, allow_infinity=False, allow_nan=False),
    end_times,
    samples,
    useful_data_arrays(),
    methods,
)
def test_invalid_start_time(tstart, tend, n, data, method):
    """Test an exception is raised when tstart falls outside of the available
    data."""
    frequency = (n - 1) / 70.0
    # This frequency is not exactly the one which will actually be
    # used when binning, so make extra offset to ensure outside of
    # range.
    tstart -= 1.0 / frequency + 1e-8
    assume(abs(tend - tstart) * frequency >= 2.0)
    with raises(ValueError, match="Start time|first bin"):
        convert_in_time(tstart, tend, frequency, data, method)


@given(
    start_times,
    floats(min_value=120.0, exclude_min=True, allow_infinity=False, allow_nan=False),
    samples,
    useful_data_arrays(),
    methods,
)
def test_invalid_end_time(tstart, tend, n, data, method):
    """Test an exception is raised when tstart falls outside of the available
    data."""
    frequency = (n - 1) / 70.0
    # This frequency is not exactly the one which will actually be
    # used when binning, so make extra offset to ensure outside of
    # range.
    tend += 1.0 / frequency + 1e-8
    assume(abs(tend - tstart) * frequency >= 2.0)
    with raises(ValueError, match="End time|last bin"):
        convert_in_time(tstart, tend, frequency, data, method)


@given(start_times, end_times, integers(400, 1000), useful_smooth_data, methods)
def test_interpolate_downsample(tstart, tend, n, data, method):
    """Check interpolate then downsamples gives sensible results"""
    dmax = data.max()
    dmin = data.min()
    if dmax + dmin < 0:
        data -= 10 * np.abs(dmax)
        if "dropped" in data.attrs:
            data.attrs["dropped"] -= 10 * np.abs(dmax)
    else:
        data += 10 * np.abs(dmin)
        if "dropped" in data.attrs:
            data.attrs["dropped"] += 10 * np.abs(dmin)
    if tstart > tend:
        tstart, tend = tend, tstart
    times = data.coords["t"]
    original_freq = (len(times) - 1) / float(times[-1] - times[0])
    new_times = times.sel(
        t=slice(tstart + 0.5 / original_freq, tend - 0.5 / original_freq)
    )
    assume(len(new_times) > 1)
    new_tstart = float(new_times[0])
    new_tend = float(new_times[-1])
    frequency = (n - 1) / 70.0
    result = convert_in_time(tstart, tend, frequency, data, "cubic")
    result2 = convert_in_time(new_tstart, new_tend, original_freq, result, method)
    assert np.all(
        result2.values
        == approx(data.sel(t=new_times).values, rel=1e-1, abs=1e-2, nan_ok=True)
    )
    if "dropped" in data.attrs:
        assert np.all(
            result2.attrs["dropped"].values
            == approx(data.attrs["dropped"].sel(t=new_times).values, rel=1e-1, abs=1e-2)
        )


@given(
    floats(51.0, 119.0),
    floats(51.0, 119.0).map(lambda t: 119.0 - (t - 51.0)),
    samples,
    t_axes,
    sane_floats(),
    sane_floats(),
    floats(0.0, 0.2).map(lambda x: 0.2 - x),
    sampled_from(["linear", "slinear", "quadratic", "cubic"]),
)
def test_interpolate_linear_data(tstart, tend, n, times, a, b, abs_err, method):
    """Test interpolate/downsample on linear data"""

    def count_floor(count):
        floor = np.floor(count)
        if np.isclose(floor, count, 1e-12, 1e-12):
            floor -= 1
        return floor

    def count_ceil(count):
        ceil = np.ceil(count)
        if np.isclose(ceil, count, 1e-12, 1e-12):
            ceil += 1
        return ceil

    print(a, b)
    data = linear_data_array(a, b, times, abs_err)
    if tstart > tend:
        tstart, tend = tend, tstart
    original_frequency = 1 / (times[1] - times[0])
    frequency = (n - 1) / 70.0
    tstart += 0.5 / frequency
    tend -= 0.5 / frequency
    assume(tstart < tend)
    assume((tend - tstart) * frequency >= 2.0)
    result = convert_in_time(tstart, tend, frequency, data, method)
    new_times = result.coords["t"]
    expected = a * new_times + b
    new_times = result.coords["t"]
    count = ((new_times[1] - new_times[0]) / (times[1] - times[0])).values
    if frequency / original_frequency <= 0.2:
        assert np.all(
            result.values
            == approx(
                expected.values, abs=max(1e-12, 0.5 * np.abs(a) / original_frequency)
            )
        )
        assert np.all(
            abs_err / np.sqrt(count_ceil(count) + 1) - 1e-12
            <= result.attrs["error"][1:-1].values
        )
        assert np.all(
            result.attrs["error"].values
            <= abs_err / np.sqrt(count_floor(count) - 1) + 1e-12
        )
    else:
        assert np.all(result.values == approx(expected.values))
        assert np.all(result.attrs["error"].values == approx(abs_err))


@given(
    start_times,
    end_times,
    floats(0.200001, 10.0),
    useful_data_arrays(),
    methods,
)
def test_call_interpolate(tstart, tend, frequency_factor, data, method):
    """Check interpolation is used when a high frequency is specified."""
    if tstart > tend:
        tstart, tend = tend, tstart
    times = data.coords["t"]
    original_freq = (len(times) - 1) / float(times[-1] - times[0])
    frequency = frequency_factor * original_freq
    with patch("indica.converters.time.interpolate_in_time"):
        from indica.converters.time import interpolate_in_time

        convert_in_time(tstart, tend, frequency, data, method)
        interpolate_in_time.assert_called_with(tstart, tend, frequency, data, method)


@given(start_times, end_times, floats(0.01, 0.2), useful_data_arrays(), methods)
def test_call_bin(tstart, tend, frequency_factor, data, method):
    """Check binning is used when a low frequency is specified."""
    if tstart > tend:
        tstart, tend = tend, tstart
    times = data.coords["t"]
    original_freq = (len(times) - 1) / float(times[-1] - times[0])
    frequency = frequency_factor * original_freq
    with patch("indica.converters.time.bin_in_time"):
        convert_in_time(tstart, tend, frequency, data, method)
        from indica.converters.time import bin_in_time

        bin_in_time.assert_called_with(tstart, tend, frequency, data)
