"""Test interpolation/downsampling of data in time."""

from unittest.mock import patch

from hypothesis import given
from hypothesis.strategies import builds
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import sampled_from
import numpy as np
from xarray import DataArray

from indica.converters.time import convert_in_time
from ..data_strategies import data_arrays

start_times = floats(50.0, 120.0)
end_times = start_times.map(lambda t: 120.0 - (t - 50.0))
samples = integers(2, 350)
methods = sampled_from(
    ["linear", "nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next"]
)
t_axes = builds(np.linspace, just(50.0), just(120.0), integers(2, 200))
useful_data_arrays = t_axes.flatmap(
    lambda x: data_arrays(override_coords=(None, None, np.expand_dims(x, (1, 2))))
)


def linear_data_array(a, b, times, abs_err):
    """Create a DataArray where values are ``a*times + b``."""
    result = DataArray(a * times + b, coords=[("t", times)])
    result.attrs["error"] = DataArray(
        np.ones_like(times) * abs_err, coords=[("t", times)]
    )
    return result


# @ given(start_times, end_times, samples, useful_data_arrays, methods)
# def test_correct_time_values(tstart, tend, n, data, method):
#     """Check always have requested time values."""
#     if tstart > tend:
#         tstart, tend = tend, tstart
#     frequency = n/70.0
#     assume((tend - tstart) * frequency >= 2.0)
#     result = convert_in_time(tstart, tend, frequency, data, method)
#     for times in [result.coords["t"], result.attrs["error"].coords["t"]]:
#         tdiffs = times[1:] - times[:-1]
#         n = len(times)
#         actual_frequency = (tend - tstart) / n
#         assert times[0] == approx(tstart)
#         assert times[-1] == approx(tend)
#         assert np.all(tdiffs == approx(tdiffs[0]))
#         assert abs(frequency - actual_frequency) < abs(
#             frequency - (tend - tstart) / (n + 1)
#         )
#         assert abs(frequency - actual_frequency) < abs(
#             frequency - (tend - tstart) / (n - 1)
#         )


# @given(start_times, end_times, samples, useful_data_arrays, methods)
# def test_unchanged_axes(tstart, tend, n, data, method):
#     """Check other axes unchanged"""
#     if tstart > tend:
#         tstart, tend = tend, tstart
#     frequency = n/70.0
#     assume((tend - tstart) * frequency >= 2.0)
#     result = convert_in_time(tstart, tend, frequency, data, method)
#     assert result.dims == data.dims
#     for dim, coords in result.coords.items():
#         if dim == "t":
#             continue
#         assert np.all(coords == data.coords[dim])


# @given(start_times, end_times, samples, useful_data_arrays, methods)
# def test_unchanged_attrs(tstart, tend, n, data, method):
#     """Check other axes unchanged"""
#     data_attrs = data.attrs
#     if tstart > tend:
#         tstart, tend = tend, tstart
#     frequency = n/70.0
#     assume((tend - tstart) * frequency >= 2.0)
#     result = convert_in_time(tstart, tend, frequency, data, method)
#     assert set(result.attrs) == set(data_attrs)
#     for key, val in data_attrs.items():
#         if key == "error":
#             continue
#         assert result.attrs[key] == data_attrs


# @given(start_times, end_times, integers(400, 1000), useful_data_arrays, methods)
# def test_interpolate_downsample(tstart, tend, n, data, method):
#     """Check interpolate then downsamples gives sensible results"""
#     if tstart > tend:
#         tstart, tend = tend, tstart
#     frequency = n/70.0
#     result = convert_in_time(tstart, tend, frequency, data, "cubic")
#     times = data.coords["t"]
#     original_freq = (times[-1] - times[0]) / len(times)
#     result2 = convert_in_time(tstart, tend, original_freq, result, method)
#     assert np.all(result2 == approx(data.sel(t=slice(tstart, tend))))


# @given(
#     start_times,
#     end_times,
#     samples,
#     t_axes,
#     sane_floats,
#     sane_floats,
#     floats(0.0, 0.2),
#     sampled_from(["linear", "slinear", "quadratric", "cubic"]),
# )
# def test_interpolate_linear_data(tstart, tend, n, times, a, b, abs_err, method):
#     """Test interpolate/downsample on linear data"""
#     data = linear_data_array(a, b, times, abs_err)
#     if tstart > tend:
#         tstart, tend = tend, tstart
#     frequency = n/70.0
#     assume((tend - tstart) * frequency >= 2.0)
#     result = convert_in_time(tstart, tend, frequency, data, method)
#     new_times = result.coords["t"]
#     expected = a * new_times + b
#     times = data.coords["t"]
#     ratio = (times[-1] - times[0]) / len(times) / frequency
#     if ratio <= 5:
#         # This should fail, whereas line below should pass, but I want
#         # to make sure.  Once I've confirmit fails with the smaller
#         # value I can delete that line as I'll be confident it isn't
#         # just a matter of massively overestimating the error when it
#         # passes.
#         assert np.all(
#             result.values == approx(expected, abs=0.4 * np.abs(a) / frequency)
#         )
#         assert np.all(
#             result.values == approx(expected, abs=0.5 * np.abs(a) / frequency)
#         )
#         assert np.all(
#             result.attrs["error"].values == approx(abs_err / np.sqrt(len(times)))
#         )
#     else:
#         assert np.all(result.values == approx(expected))
#         assert np.all(result.attrs["error"].values == approx(abs_err))


@given(
    start_times,
    end_times,
    floats(0.2, 10.0, exclude_min=True),
    useful_data_arrays,
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


@given(start_times, end_times, floats(0.01, 0.2), useful_data_arrays, methods)
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
