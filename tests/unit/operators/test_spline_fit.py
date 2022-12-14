from unittest.mock import MagicMock

import numpy as np
import xarray as xr
from xarray import DataArray
from xarray.testing import assert_allclose

from indica.converters import FluxSurfaceCoordinates
from indica.converters import TransectCoordinates
from indica.converters import TrivialTransform
from indica.operators.spline_fit import Spline
from indica.operators.spline_fit import SplineFit
from indica.utilities import coord_array
from ..fake_equilibrium import FakeEquilibrium


# Test the spline class

fake_equilib = FakeEquilibrium(0.6, 0.0)

x_positions = DataArray(
    np.linspace(1.0, 0.1, 10), coords=[("alpha", np.arange(10))]
).assign_attrs(datatype=("major_rad", "plasma"))
y_positions = xr.zeros_like(x_positions)
z_positions = DataArray(
    np.linspace(-0.1, 0.2, 10), coords=[("alpha", np.arange(10))]
).assign_attrs(datatype=("z", "plasma"))
coords = TransectCoordinates(x_positions, y_positions, z_positions, "")
coords.set_equilibrium(fake_equilib)

trivial = TrivialTransform()
R_grid = coord_array(np.linspace(0.0, 1.2, 7), "R")
z_grid = coord_array(np.linspace(-1.0, 1.0, 5), "z")
t_grid = coord_array(np.linspace(0.0, 10.0, 11), "t")


def func(R, z, t):
    return R - z + 0.5 * t


def test_spline_trivial_coords_R():
    spline = Spline(func(R_grid, z_grid, t_grid), "R", trivial, "natural")
    R = coord_array(np.linspace(0.2, 0.8, 4), "R")
    result = spline(trivial, R, z_grid, t_grid)
    assert_allclose(result, func(R, z_grid, t_grid))


def test_spline_trivial_coords_z():
    spline = Spline(func(R_grid, z_grid, t_grid), "z", trivial, "natural")
    z = coord_array(np.linspace(-0.2, 0.2, 5), "z")
    result = spline(trivial, R_grid, z, t_grid)
    assert_allclose(result, func(R_grid, z, t_grid).transpose(*result.dims))


def test_spline_transect_coords_x1():
    spline = Spline(func(R_grid, 0, t_grid), "R", trivial, "natural")
    indices = [1, 3, 5]
    alpha = coord_array(indices, coords.x1_name)
    alpha_z_offset = coord_array([0.2, 0.1], coords.x2_name)
    result = spline(coords, alpha, alpha_z_offset, t_grid)
    assert_allclose(result, func(x_positions[indices], 0.0, t_grid))


def test_spline_transect_coords_x2():
    spline = Spline(func(0, z_grid, t_grid), "z", trivial, "natural")
    indices = [1, 3, 5]
    alpha = coord_array(indices, coords.x1_name)
    alpha_z_offset = coord_array([0.2, 0.1], coords.x2_name)
    result = spline(coords, alpha, alpha_z_offset, t_grid)
    assert_allclose(result, func(0, z_positions[indices] + alpha_z_offset, t_grid))


# Test the fit itself


def test_spline_fit():
    flux_coords = FluxSurfaceCoordinates("poloidal")
    flux_coords.set_equilibrium(fake_equilib)
    knot_locations = [0.0, 0.5, 0.8, 1.05]
    knot_vals = (
        DataArray([1.0, 0.9, 0.6, 0.0], coords=[("rho_poloidal", knot_locations)])
        + 0.05 * t_grid
    )
    knot_vals.loc[knot_locations[-1]] = 0.0
    expected_spline = Spline(knot_vals, "rho_poloidal", flux_coords)
    input_vals = expected_spline(
        coords, x_positions.coords["alpha"], 0.0, t_grid
    ).assign_coords(alpha_z_offset=0)
    input_vals.attrs["datatype"] = ("temperature", "electrons")
    fitter = SplineFit(knot_locations, sess=MagicMock())
    result_locations = coord_array(np.linspace(0, 1.05, 10), "rho_poloidal")
    result, spline_fit, binned_input = fitter(result_locations, t_grid, input_vals)
    assert_allclose(input_vals, binned_input)
    assert_allclose(spline_fit, knot_vals.transpose(*spline_fit.dims))
    assert_allclose(result, expected_spline(flux_coords, result_locations, 0.0, t_grid))
    assert result.attrs["datatype"] == input_vals.attrs["datatype"]
    assert isinstance(result.attrs["splines"], Spline)
    assert result.attrs["transform"] == flux_coords
    assert "provenance" in result.attrs
    assert "provenance" in spline_fit.attrs
    assert "provenance" in binned_input.attrs
