"""Test the contents of the utilities module."""

import re
import unittest

from hypothesis import assume
from hypothesis import example
from hypothesis import given
from hypothesis.extra.numpy import array_shapes
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import dictionaries
from hypothesis.strategies import from_regex
from hypothesis.strategies import integers
from hypothesis.strategies import none
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
import numpy as np
import pytest
import scipy
from xarray import DataArray
from xarray.testing import assert_allclose

from indica import utilities

VALID_FILENAME = re.compile(r"^[a-zA-Z0-9_\-().]+$")


def test_positional_parameters1():
    def example(a, b, c=None, d=5):
        pass

    params, varpos = utilities.positional_parameters(example)
    assert params == ["a", "b", "c", "d"]
    assert varpos is None


def test_positional_parameters2():
    def example():
        pass

    params, varpos = utilities.positional_parameters(example)
    assert params == []
    assert varpos is None


def test_positional_parameters3():
    def example(a, /, b, c, *, d):
        pass

    params, varpos = utilities.positional_parameters(example)
    assert params == ["a", "b", "c"]
    assert varpos is None


def test_positional_parameters4():
    def example(a, b, c, *args, **kwargs):
        pass

    params, varpos = utilities.positional_parameters(example)
    assert params == ["a", "b", "c"]
    assert varpos == "args"


@given(arrays(np.float64, array_shapes(), elements=sampled_from([1.0, -1.0])))
def test_sum_squares_ones(a):
    """Test summing arrays made up only of +/- 1."""
    print(a)
    print(a.shape)
    for i, l in enumerate(a.shape):
        assert np.all(utilities.sum_squares(a, i) == l)


@given(dictionaries(from_regex("[_a-zA-Z0-9]+", fullmatch=True), none()))
def test_sum_squares_known(kwargs):
    assume("x" not in kwargs)
    assume("axis" not in kwargs)
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert utilities.sum_squares(a, 0, **kwargs) == 55


@given(from_regex(VALID_FILENAME.pattern[1:-1]))
@example("this/ is \\ the 'n@stiest' èxample_I-can think of")
def test_to_filename(name_in):
    # Check there are some valid characthers in the filename to start with
    assume(VALID_FILENAME.match(name_in))
    name_out = utilities.to_filename(name_in)
    print(name_in, name_out, VALID_FILENAME.match(name_out))
    assert VALID_FILENAME.match(name_out) is not None


def test_to_filename_known_result():
    assert utilities.to_filename("a/b/C\\d-e(f, g)") == "a-b-C-d-e(f_g)"


# There appears to be a bug in the Hypothesis type annotation for the
# arrays() strategy
@given(arrays(float, integers(0, 100)), text())  # type: ignore
def test_coord_array(vals, name):
    coords = utilities.coord_array(vals, name)
    np.testing.assert_array_equal(coords.data, vals)
    np.testing.assert_array_equal(coords.coords[name], vals)
    assert len(coords.dims) == 1
    datatype = coords.attrs["datatype"]
    if name == "R":
        assert datatype[0] == "major_rad"
    elif name == "t":
        assert datatype[0] == "time"
    elif name == "rho_poloidal":
        assert datatype[0] == "norm_flux_pol"
    elif name == "rho_toroidal":
        assert datatype[0] == "norm_flux_tor"
    else:
        assert datatype[0] == name
    assert datatype[1] == "plasma"


def func_3d(x, y, t):
    return 2 * x + 3 * y - 0.5 * t


x = utilities.coord_array(np.linspace(0.0, 1.0, 5), "x")
y = utilities.coord_array(np.linspace(0.0, 10.0, 4), "y")
t = utilities.coord_array(np.linspace(50.0, 55.0, 6), "t")
spline_dims = ("y", "t")
spline_coords = {"y": y, "t": t}
interp_data = func_3d(x, y, t)
spline = scipy.interpolate.CubicSpline(x, interp_data, 0, "natural")


def test_broadcast_spline_new_coord():
    x_interp = utilities.coord_array(np.linspace(0.1, 0.4, 4), "x_new")
    result = utilities.broadcast_spline(spline, spline_dims, spline_coords, x_interp)
    assert_allclose(result, func_3d(x_interp, y, t))
    assert result.dims == ("x_new", "y", "t")


def test_broadcast_spline_t():
    x_interp = DataArray(
        np.linspace(0.1, 0.4, 4), coords=[("t", np.linspace(52.0, 53.0, 4))]
    )
    result = utilities.broadcast_spline(spline, spline_dims, spline_coords, x_interp)
    assert_allclose(result, func_3d(x_interp, y, x_interp.t))
    assert result.dims == ("t", "y")


def test_broadcast_spline_2d():
    x_interp = utilities.coord_array(
        np.linspace(0.2, 0.5, 3), "alpha"
    ) * utilities.coord_array(np.linspace(0.5, 1.0, 4), "beta")
    result = utilities.broadcast_spline(spline, spline_dims, spline_coords, x_interp)
    assert_allclose(result, func_3d(x_interp, y, t))
    assert result.dims == ("alpha", "beta", "y", "t")


@pytest.mark.xfail(reason="Feature not implemented.")
def test_broadcast_spline_old_coord():
    x_interp = DataArray(
        np.linspace(0.1, 0.4, 4), coords=[("y", np.linspace(2.5, 7.5, 4))]
    )
    result = utilities.broadcast_spline(spline, spline_dims, spline_coords, x_interp)
    assert_allclose(result, func_3d(x_interp, x_interp.y, t))
    assert result.dims == ("y", "t")


class Compatible_Input_Type_Test_Case(unittest.TestCase):
    def __init__(self):
        self.Ne = np.logspace(19.0, 16.0, 10)

    def type_check(self):
        with self.assertRaises(TypeError):
            utilities.input_check("Ne", self.Ne, str)

    def value_check(self):
        Ne = 0 * self.Ne
        with self.assertRaises(ValueError):
            utilities.input_check(
                "Ne", Ne, np.ndarray, greater_than_or_equal_zero=False
            )

        Ne = -1 * self.Ne
        with self.assertRaises(ValueError):
            utilities.input_check("Ne", Ne, np.ndarray, greater_than_or_equal_zero=True)

        Ne = np.nan * self.Ne
        with self.assertRaises(ValueError):
            utilities.input_check("Ne", Ne, np.ndarray)

        Ne = np.inf * self.Ne
        with self.assertRaises(ValueError):
            utilities.input_check("Ne", Ne, np.ndarray)

        Ne = -np.inf * self.Ne
        with self.assertRaises(ValueError):
            utilities.input_check("Ne", Ne, np.ndarray)

        Ne = self.Ne[:, np.newaxis]
        with self.assertRaises(ValueError):
            utilities.input_check("Ne", Ne, np.ndarray, ndim_to_check=1)

        # Check dropped channel handling
        t = np.array([78.5, 80.5, 82.5])
        rho = np.linspace(0, 1, 11)
        Ne = np.logspace(19.0, 16.0, 11)
        Ne = np.tile(Ne, [3, 1])
        Ne[1, :] /= 10.0
        Ne[2, :] *= 10.0

        dropped_t_coord = np.array([80.5])
        dropped_rho_coord = np.array([rho[3], rho[7]])

        Ne = DataArray(
            data=Ne,
            coords=[("t", t), ("rho_poloidal", rho)],
            dims=["t", "rho_poloidal"],
        )

        dropped = Ne.sel({"t": dropped_t_coord})
        dropped = dropped.sel({"rho_poloidal": dropped_rho_coord})

        Ne.loc[{"t": dropped_t_coord, "rho_poloidal": dropped_rho_coord}] = np.nan

        Ne.attrs["dropped"] = dropped

        try:
            utilities.input_check("Ne", Ne, DataArray, ndim_to_check=2)
        except Exception as e:
            raise e


def test_compatible_input_type():
    compatible_input_type = Compatible_Input_Type_Test_Case()
    compatible_input_type.type_check()
    compatible_input_type.value_check()
