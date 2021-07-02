import numpy as np
from xarray import DataArray
from xarray.testing import assert_allclose

from indica.converters import FluxSurfaceCoordinates
from indica.converters import TransectCoordinates
from indica.operators.invert_radiation import EmissivityProfile
from indica.utilities import coord_array
from ..fake_equilibrium import FakeEquilibrium


fake_equilib = FakeEquilibrium(0.6, 0.0)
flux_coords = FluxSurfaceCoordinates("poloidal")
flux_coords.set_equilibrium(fake_equilib)

R_positions = DataArray(
    np.linspace(0.6, 0.6, 10), coords=[("alpha", np.arange(10))]
).assign_attrs(datatype=("major_rad", "plasma"))
z_positions = DataArray(
    np.linspace(-0.3, 0.3, 10), coords=[("alpha", np.arange(10))]
).assign_attrs(datatype=("z", "plasma"))
coords = TransectCoordinates(R_positions, z_positions)
coords.set_equilibrium(fake_equilib)

knot_locs = coord_array([0.0, 0.5, 0.75, 0.9, 1.0], "rho_poloidal")
knot_times = coord_array(np.linspace(0.0, 10.0, 6), "t")


def sym_func(rho, t):
    # Need this form of polynomial in order to satisfy boundary conditions
    return rho ** 3 - 3 * rho ** 2 + 0.2 * t + 2


def asym_func(rho, t):
    return 2 * rho + 0.1 * t


def test_emissivity_profile_sym():
    sym_emiss = sym_func(knot_locs, knot_times)
    asym_par = 0 * knot_locs
    profile = EmissivityProfile(sym_emiss, asym_par, flux_coords)
    rho_grid = coord_array(np.linspace(0.3, 0.7, 9), "rho_poloidal")
    theta_grid = coord_array(np.linspace(0, 2 * np.pi, 8, False), "theta")
    t_grid = coord_array(np.linspace(0.5, 9.5, 10), "t")
    expected_emissiv = sym_func(rho_grid, t_grid) + 0 * theta_grid
    result = profile(flux_coords, rho_grid, theta_grid, t_grid)
    assert_allclose(result, expected_emissiv.transpose(*result.dims))
    assert result.attrs["transform"] == flux_coords


def test_emissivity_profile_asym():
    sym_emiss = 0 * knot_locs * knot_times + 1
    asym_par = asym_func(knot_locs, knot_times)
    profile = EmissivityProfile(sym_emiss, asym_par, flux_coords)
    alpha_grid = R_positions.coords["alpha"]
    alpha_z_off_grid = 0 * z_positions
    t_grid = coord_array(np.linspace(0.5, 9.5, 4), "t")
    rho_grid, theta_grid = coords.convert_to(
        flux_coords, alpha_grid, alpha_z_off_grid, t_grid
    )
    R_0 = fake_equilib.R_hfs(rho_grid, t_grid, "poloidal")[0]
    expected_emissiv = np.exp(asym_func(rho_grid, t_grid) * (0.36 - R_0 ** 2))
    result = profile(coords, alpha_grid, alpha_z_off_grid, t_grid)
    assert_allclose(result, expected_emissiv.transpose(*result.dims))
    assert result.attrs["transform"] == coords


def test_emissivity_profile_evaluate1():
    sym_emiss = sym_func(knot_locs, knot_times)
    asym_par = asym_func(knot_locs, knot_times)
    profile = EmissivityProfile(sym_emiss, asym_par, flux_coords)
    R_grid = coord_array(np.linspace(0.3, 0.9, 7), "R")
    z_grid = coord_array(np.linspace(-0.2, 0.2, 5), "z")
    rho_grid, theta_grid = flux_coords.convert_from_Rz(R_grid, z_grid, 0.0)
    del rho_grid.coords["t"]
    expected_emissiv = sym_func(rho_grid, knot_times)
    result = profile.evaluate(rho_grid, R_grid, R_0=R_grid)
    assert_allclose(result, expected_emissiv.transpose(*result.dims))


def test_emissivity_profile_evaluate2():
    sym_emiss = sym_func(knot_locs, knot_times)
    asym_par = asym_func(knot_locs, knot_times)
    profile = EmissivityProfile(sym_emiss, asym_par, flux_coords)
    R_grid = coord_array(np.linspace(0.3, 0.9, 7), "R")
    z_grid = coord_array(np.linspace(-0.2, 0.2, 5), "z")
    t_grid = coord_array(np.linspace(0.5, 9.5, 4), "t")
    rho_grid, theta_grid = flux_coords.convert_from_Rz(R_grid, z_grid, t_grid)
    expected_emissiv = sym_func(rho_grid, t_grid) * np.exp(
        0.01 * asym_func(rho_grid, t_grid)
    )
    result = profile.evaluate(rho_grid, R_grid, R_0=np.sqrt(R_grid ** 2 - 0.01))
    assert_allclose(result, expected_emissiv.transpose(*result.dims))
