from unittest.mock import MagicMock

import numpy as np
from pytest import mark
from scipy.integrate import romb
from xarray import DataArray
from xarray.testing import assert_allclose

from indica.converters import FluxSurfaceCoordinates
from indica.converters import LinesOfSightTransform
from indica.converters import TransectCoordinates
from indica.converters import TrivialTransform
from indica.operators.invert_radiation import EmissivityProfile
from indica.operators.invert_radiation import InvertRadiation
from indica.utilities import coord_array
from ..fake_equilibrium import FakeEquilibrium

pytestmark = mark.filterwarnings("ignore:invalid value encountered in true_divide")

fake_equilib = FakeEquilibrium(0.6, 0.0, poloidal_alpha=0.002)
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
    return rho**3 - 3 * rho**2 + 0.2 * t + 2


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
    expected_emissiv = np.exp(asym_func(rho_grid, t_grid) * (0.36 - R_0**2))
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
    result = profile.evaluate(rho_grid, R_grid, R_0=np.sqrt(R_grid**2 - 0.01))
    assert_allclose(result, expected_emissiv.transpose(*result.dims))


# TODO: Consider placing checks of metadata in separate test and only
# checking numerics in this one
def test_invert_radiation():
    n_los = 15
    n_int = 33
    times = coord_array(np.linspace(50.0, 53.0, 4), "t")
    los_transform = LinesOfSightTransform(
        np.linspace(1.0, 0.2, n_los),
        np.ones(n_los),
        np.zeros(n_los),
        np.linspace(1.0, 0.2, n_los),
        -np.ones(n_los),
        np.zeros(n_los),
        "alpha",
        ((0.1, 1.2), (-1.5, 1.5)),
    )
    los_transform.set_equilibrium(fake_equilib)
    rho_max, _, _ = fake_equilib.flux_coords(1.0, 0.0, times, "poloidal")
    knot_locs = coord_array(
        InvertRadiation.knot_positions(6, rho_max.mean("t")), "rho_poloidal"
    )
    expected_sym = (
        DataArray(
            [1.0, 0.9, 0.8, 0.2, 0.1, 0.0], coords=[("rho_poloidal", knot_locs.data)]
        )
        * (1 + 0.01 * times)
        * 3e3
    )
    expected_asym = DataArray(
        [0.0, 0.1, 0.2, 0.22, 0.05, 0.0], coords=[("rho_poloidal", knot_locs.data)]
    ) * (1 - 0.002 * times)
    expected_profile = EmissivityProfile(expected_sym, expected_asym, flux_coords)
    los_x1_grid = coord_array(np.arange(n_los), "alpha_coords")
    los_x2_grid = coord_array(np.linspace(0.0, 1.0, n_int), "alpha_los_position")
    x2 = "alpha_los_position"
    expected_emiss = expected_profile(los_transform, los_x1_grid, los_x2_grid, times)
    flux = DataArray(
        romb(expected_emiss, 2.5 / (n_int - 1), expected_emiss.dims.index(x2)),
        dims=[d for d in expected_emiss.dims if d != x2],
        coords={k: v for k, v in expected_emiss.coords.items() if k != x2},
    )
    flux.attrs["datatype"] = ("luminous_flux", "sxr")
    flux.attrs["partial_provenance"] = MagicMock()
    flux.attrs["provenance"] = MagicMock()
    flux.attrs["transform"] = los_transform
    inverter = InvertRadiation(1, "sxr", len(knot_locs.data), n_int, MagicMock())
    R_grid = coord_array(np.linspace(0.1, 1.1, 6), "R")
    z_grid = coord_array(np.linspace(-1.0, 1.0, 5), "z")
    emissivity, fit_params, cam_data = inverter(R_grid, z_grid, times, flux)
    assert_allclose(
        cam_data.camera.drop_vars("alpha_rho_poloidal").transpose(*flux.dims), flux
    )
    assert_allclose(
        cam_data.back_integral.drop_vars("alpha_rho_poloidal").transpose(*flux.dims),
        flux,
        rtol=1e-3,
    )
    assert_allclose(
        emissivity,
        expected_profile(TrivialTransform(), R_grid, z_grid, times),
        rtol=2e-2,
    )
    assert isinstance(emissivity.attrs["emissivity_model"], EmissivityProfile)
    assert_allclose(
        emissivity,
        emissivity.attrs["emissivity_model"](TrivialTransform(), R_grid, z_grid, times),
    )
    assert_allclose(
        fit_params.symmetric_emissivity.transpose(*expected_sym.dims),
        expected_sym,
        rtol=2e-3,
    )
    assert_allclose(
        fit_params.asymmetry_parameter.transpose(*expected_asym.dims),
        expected_asym,
        rtol=0.1,
        atol=0.05,
    )
    assert isinstance(cam_data.weights, DataArray)
    assert "provenance" in emissivity.attrs
    assert "provenance" in fit_params.attrs
    assert "provenance" in cam_data.attrs
    assert fit_params.symmetric_emissivity.attrs["transform"] == flux_coords
    assert fit_params.asymmetry_parameter.attrs["transform"] == flux_coords
    assert cam_data.camera.attrs["transform"] == los_transform
    assert cam_data.back_integral.attrs["transform"] == los_transform
    assert cam_data.weights.attrs["transform"] == los_transform
