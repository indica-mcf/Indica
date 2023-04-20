"""Tests for line-of-sight coordinate transforms."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import xarray as xr
from xarray import DataArray

from indica import equilibrium
from indica.converters import FluxSurfaceCoordinates
from indica.converters import line_of_sight
from indica.converters import TrivialTransform


class Exception_Line_Of_Sight_Test_Case(unittest.TestCase):
    pass


def default_inputs():
    """Default inputs for a single line of sight, no time dependence"""
    x1 = 0.0
    x2 = DataArray(np.linspace(0.0, 1.0, 350, dtype=float))
    t = 0.0

    return x1, x2, t


def load_line_of_sight_default():
    origin = np.array([[3.8, -1.0, 0.5], [3.8, -1.5, 0.0]])
    direction = np.array([[-1.0, 1.0, 0.0], [-1.0, 2.0, 0.0]])
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))
    name = "los_test"
    los = line_of_sight.LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        machine_dimensions=machine_dims,
        name=name,
    )
    return los, machine_dims


def load_equilibrium_default():
    data = equilibrium_dat()
    equil = equilibrium.Equilibrium(
        data,
        sess=MagicMock(),
    )
    return equil


def test_missing_los():
    """
    Test whether a line-of-sight that misses the vessel is handled correctly
    """
    # Line of sight origin tuple
    origin = np.array(
        [
            [4.0, -2.0, 0.5],
        ]
    )  # [xyz]

    # Line of sight direction
    direction = np.array(
        [
            [0.0, 1.0, 0.0],
        ]
    )  # [xyz]

    # machine dimensions
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))

    # name
    name = "los_test"

    # Set-up line of sight class
    los_test_obj = Exception_Line_Of_Sight_Test_Case()
    with los_test_obj.assertRaises(ValueError):
        line_of_sight.LineOfSightTransform(
            origin[:, 0],
            origin[:, 1],
            origin[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            machine_dimensions=machine_dims,
            name=name,
        )


def test_convert_to_xy(debug=False):
    """
    Test for the method convert_to_xy()
    """
    # Load line-of-sight default
    los, machine_dims = load_line_of_sight_default()
    x1 = 0
    x2 = los.x2[0]
    t = 0

    # Test method
    x, y = los.convert_to_xy(x1, x2, t)

    assert np.all(x <= np.max([los.x_start, los.x_end]))
    assert np.all(x >= np.min([los.x_start, los.x_end]))
    assert np.all(y <= np.max([los.y_start, los.y_end]))
    assert np.all(y >= np.min([los.y_start, los.y_end]))


def test_convert_to_Rz(debug=False):
    """
    Test for the method convert_to_Rz()
    """
    # Load line-of-sight default
    los, machine_dims = load_line_of_sight_default()
    x1 = 0
    x2 = los.x2[0]
    t = 0

    # Test method
    R, z = los.convert_to_Rz(x1, x2, t)

    R_min = np.sqrt(
        np.min([los.x_start**2, los.x_end**2])
        + np.min([los.y_start**2, los.y_end**2])
    )
    R_max = np.sqrt(
        np.max([los.x_start**2, los.x_end**2])
        + np.max([los.y_start**2, los.y_end**2])
    )

    # R and z are as expected=
    assert np.all(R <= R_max)
    assert np.all(R >= R_min)
    assert np.all(z <= np.max([los.z_start, los.z_end]))
    assert np.all(z >= np.min([los.z_start, los.z_end]))


def test_distance(debug=False):
    """
    Test for the method distance()
    """
    # Load line-of-sight default
    los, machine_dims = load_line_of_sight_default()
    x1 = 0
    x2 = los.x2[0:-1]
    t = 0

    # Test method
    dist = los.distance("los_position", x1, x2, t)
    dls = np.diff(dist)

    # dl is identical along the line of sight up to 1 per million
    assert all(np.abs(dls - dls[0]) < (dls[0] * 1.0e-6))


def test_set_dl(debug=False):
    """
    Test for the method set_dl()
    """
    # Load line-of-sight default
    los, machine_dims = load_line_of_sight_default()

    # Test inputs
    dl = 0.002

    # Test method
    los.set_dl(dl)

    assert np.abs(dl - los.dl) < 1.0e-6


def test_map_profile_to_los():
    """
    Test for the method map_profile_to_los()
    """
    los, machine_dims = load_line_of_sight_default()
    equil = load_equilibrium_default()

    los.set_equilibrium(equil)
    R_ = np.linspace(machine_dims[0][0], machine_dims[0][1], 30)
    z_ = np.linspace(machine_dims[1][0], machine_dims[1][1], 30)

    R_ = DataArray(R_, coords={"R": R_}, dims=["R"])
    z_ = DataArray(z_, coords={"z": z_}, dims=["z"])

    t_ = np.linspace(74.5, 80.5, 5)
    t_ = DataArray(t_, coords={"t": t_}, dims=["t"])

    R_ = R_.expand_dims(dim={"t": t_})
    z_ = z_.expand_dims(dim={"t": t_})

    rho_equil, theta_equil, _ = equil.flux_coords(R_, z_, t_)
    rho_max = rho_equil.max(dim=["t", "R", "z"], skipna=True).data[()]

    rho_ = np.linspace(0.0, rho_max, 30)
    theta_ = np.linspace(-np.pi, np.pi, 30)
    rho_ = DataArray(rho_, coords={"rho_poloidal": rho_}, dims=["rho_poloidal"])
    theta_ = DataArray(theta_, coords={"theta": theta_}, dims=["theta"])

    Ne = DataArray(
        np.tile(np.logspace(19.0, 16.0, 30), (5, 30, 1)).transpose([2, 1, 0]),
        coords={"rho_poloidal": rho_, "theta": theta_, "t": t_},
        dims=["rho_poloidal", "theta", "t"],
    )

    Ne = xr.where(Ne.rho_poloidal > 1, 0, Ne)

    Ne_Rz = Ne.interp(rho_poloidal=rho_equil, theta=theta_equil).drop_vars(
        ["rho_poloidal", "theta"]
    )

    Ne_los_rho_theta = los.map_profile_to_los(Ne, t_)
    Ne_los_R_z = los.map_profile_to_los(Ne_Rz, t_)

    Ne_los_rho_theta = Ne_los_rho_theta.assign_attrs(name="Ne_los_rho_theta")
    Ne_los_R_z = Ne_los_R_z.assign_attrs(name="Ne_los_R_z")

    assert Ne.min() <= Ne_los_rho_theta.all() <= Ne.max()
    assert Ne.min() <= Ne_los_R_z.all() <= Ne.max()


def test_check_rho_and_profile():
    """
    Test for the method check_rho_and_profile()
    """

    los, machine_dims = load_line_of_sight_default()
    equil = load_equilibrium_default()

    los.set_equilibrium(equil)
    R_ = np.linspace(machine_dims[0][0], machine_dims[0][1], 30)
    z_ = np.linspace(machine_dims[1][0], machine_dims[1][1], 30)

    R_ = DataArray(R_, coords={"R": R_}, dims=["R"])
    z_ = DataArray(z_, coords={"z": z_}, dims=["z"])

    t_ = np.linspace(74.5, 80.5, 5)
    t_ = DataArray(t_, coords={"t": t_}, dims=["t"])

    R_ = R_.expand_dims(dim={"t": t_})
    z_ = z_.expand_dims(dim={"t": t_})

    rho_equil, theta_equil, _ = equil.flux_coords(R_, z_, t_)
    rho_max = rho_equil.max(dim=["t", "R", "z"], skipna=True).data[()]

    rho_ = np.linspace(0.0, rho_max, 30)
    theta_ = np.linspace(-np.pi, np.pi, 30)
    rho_ = DataArray(rho_, coords={"rho_poloidal": rho_}, dims=["rho_poloidal"])
    theta_ = DataArray(theta_, coords={"theta": theta_}, dims=["theta"])

    Ne = DataArray(
        np.tile(np.logspace(19.0, 16.0, 30), (5, 30, 1)).transpose([2, 1, 0]),
        coords={"rho_poloidal": rho_, "theta": theta_, "t": t_},
        dims=["rho_poloidal", "theta", "t"],
    )

    Ne = xr.where(Ne.rho_poloidal > 1, 0, Ne)

    error_t = equil.rho.t.min() - 1

    los_test_obj = Exception_Line_Of_Sight_Test_Case()
    with los_test_obj.assertRaises(ValueError):
        los.check_rho_and_profile(Ne, error_t)

    error_t = equil.rho.t.max() + 1

    los_test_obj = Exception_Line_Of_Sight_Test_Case()
    with los_test_obj.assertRaises(ValueError):
        los.check_rho_and_profile(Ne, error_t)

    error_t = -10.0

    los_test_obj = Exception_Line_Of_Sight_Test_Case()
    with los_test_obj.assertRaises(ValueError):
        los.check_rho_and_profile(Ne.isel(t=0), error_t)

    error_t = np.append(Ne.t.data, 90.0)

    los_test_obj = Exception_Line_Of_Sight_Test_Case()
    with los_test_obj.assertRaises(ValueError):
        los.check_rho_and_profile(Ne.isel, error_t)

    error_t = np.insert(Ne.t.data, 0, 40.0)

    los_test_obj = Exception_Line_Of_Sight_Test_Case()
    with los_test_obj.assertRaises(ValueError):
        los.check_rho_and_profile(Ne.isel, error_t)


def test_integrate_on_los():
    """
    Test for the method integrate_on_los()
    """
    los, machine_dims = load_line_of_sight_default()
    equil = load_equilibrium_default()

    los.set_equilibrium(equil)
    R_ = np.linspace(machine_dims[0][0], machine_dims[0][1], 30)
    z_ = np.linspace(machine_dims[1][0], machine_dims[1][1], 30)

    R_ = DataArray(R_, coords={"R": R_}, dims=["R"])
    z_ = DataArray(z_, coords={"z": z_}, dims=["z"])

    t_ = np.linspace(74.5, 80.5, 5)
    t_ = DataArray(t_, coords={"t": t_}, dims=["t"])

    R_ = R_.expand_dims(dim={"t": t_})
    z_ = z_.expand_dims(dim={"t": t_})

    rho_equil, theta_equil, _ = equil.flux_coords(R_, z_, t_)
    rho_max = rho_equil.max(dim=["t", "R", "z"], skipna=True).data[()]

    rho_ = np.linspace(0.0, rho_max, 30)
    theta_ = np.linspace(-np.pi, np.pi, 30)
    rho_ = DataArray(rho_, coords={"rho_poloidal": rho_}, dims=["rho_poloidal"])
    theta_ = DataArray(theta_, coords={"theta": theta_}, dims=["theta"])

    Ne = DataArray(
        np.tile(np.logspace(19.0, 16.0, 30), (5, 30, 1)).transpose([2, 1, 0]),
        coords={"rho_poloidal": rho_, "theta": theta_, "t": t_},
        dims=["rho_poloidal", "theta", "t"],
    )

    Ne = xr.where(Ne.rho_poloidal > 1, 0, Ne)

    Ne_Rz = Ne.interp(rho_poloidal=rho_equil, theta=theta_equil).drop_vars(
        ["rho_poloidal", "theta"]
    )

    Ne_los_integral_rho_theta = los.integrate_on_los(Ne, t_)
    Ne_los_integral_R_z = los.integrate_on_los(Ne_Rz, t_)

    assert 0 <= Ne_los_integral_rho_theta.all() <= np.inf
    assert 0 <= Ne_los_integral_R_z.all() <= np.inf


# Function for defining equilibrium
def equilibrium_dat():
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))
    start_time, end_time = 75.0, 80.0
    Btot_factor = None

    result = {}
    nspace = 8
    ntime = 3
    times = np.linspace(start_time - 0.5, end_time + 0.5, ntime)

    tfuncs = smooth_funcs((start_time, end_time), 0.01)
    r_centre = (machine_dims[0][0] + machine_dims[0][1]) / 2
    z_centre = (machine_dims[1][0] + machine_dims[1][1]) / 2
    raw_result = {}
    attrs = {
        "transform": TrivialTransform(),
        "provenance": MagicMock(),
        "partial_provenance": MagicMock(),
    }
    result["rmag"] = xr.DataArray(
        r_centre + tfuncs(times), coords=[("t", times)], name="rmag", attrs=attrs
    )
    result["rmag"].attrs["datatype"] = ("major_rad", "mag_axis")

    result["zmag"] = xr.DataArray(
        z_centre + tfuncs(times), coords=[("t", times)], name="zmag", attrs=attrs
    )
    result["zmag"].attrs["datatype"] = ("z", "mag_axis")

    fmin = 0.1
    result["faxs"] = xr.DataArray(
        fmin + np.abs(tfuncs(times)),
        {"t": times, "R": result["rmag"], "z": result["zmag"]},
        ["t"],
        name="faxs",
        attrs=attrs,
    )
    result["faxs"].attrs["datatype"] = ("magnetic_flux", "mag_axis")

    a_coeff = xr.DataArray(
        np.vectorize(lambda x: 0.8 * x)(
            np.minimum(
                np.abs(machine_dims[0][0] - result["rmag"]),
                np.abs(machine_dims[0][1] - result["rmag"]),
            ),
        ),
        coords=[("t", times)],
    )

    if Btot_factor is None:
        b_coeff = xr.DataArray(
            np.vectorize(lambda x: 0.8 * x)(
                np.minimum(
                    np.abs(machine_dims[1][0] - result["zmag"].data),
                    np.abs(machine_dims[1][1] - result["zmag"].data),
                ),
            ),
            coords=[("t", times)],
        )
        n_exp = 0.5
        fmax = 5.0
        result["fbnd"] = xr.DataArray(
            fmax - np.abs(tfuncs(times)),
            coords=[("t", times)],
            name="fbnd",
            attrs=attrs,
        )
    else:
        b_coeff = a_coeff
        n_exp = 1
        fdiff_max = Btot_factor * a_coeff
        result["fbnd"] = xr.DataArray(
            np.vectorize(lambda axs, diff: axs + 0.03 * diff)(
                result["faxs"], fdiff_max.values
            ),
            coords=[("t", times)],
            name="fbnd",
            attrs=attrs,
        )

    result["fbnd"].attrs["datatype"] = ("magnetic_flux", "separtrix")

    thetas = xr.DataArray(
        np.linspace(0.0, 2 * np.pi, nspace, endpoint=False), dims=["arbitrary_index"]
    )
    result["rbnd"] = (
        result["rmag"]
        + a_coeff * b_coeff / np.sqrt(a_coeff**2 * np.tan(thetas) ** 2 + b_coeff**2)
    ).assign_attrs(**attrs)
    result["rbnd"].name = "rbnd"
    result["rbnd"].attrs["datatype"] = ("major_rad", "separatrix")

    result["zbnd"] = (
        result["zmag"]
        + a_coeff
        * b_coeff
        / np.sqrt(a_coeff**2 + b_coeff**2 * np.tan(thetas) ** -2)
    ).assign_attrs(**attrs)
    result["zbnd"].name = "zbnd"
    result["zbnd"].attrs["datatype"] = ("z", "separatrix")

    r = np.linspace(machine_dims[0][0], machine_dims[0][1], nspace)
    z = np.linspace(machine_dims[1][0], machine_dims[1][1], nspace)
    rgrid = xr.DataArray(r, coords=[("R", r)])
    zgrid = xr.DataArray(z, coords=[("z", z)])
    psin = (
        (-result["zmag"] + zgrid) ** 2 / b_coeff**2
        + (-result["rmag"] + rgrid) ** 2 / a_coeff**2
    ) ** (0.5 / n_exp)
    result["psin"] = psin

    psi = psin * (result["fbnd"] - result["faxs"]) + result["faxs"]
    psi.name = "psi"
    psi.attrs["transform"] = attrs["transform"]
    psi.attrs["provenance"] = MagicMock()
    psi.attrs["partial_provenance"] = MagicMock()
    psi.attrs["datatype"] = ("magnetic_flux", "plasma")
    result["psi"] = psi

    psin_coords = np.linspace(0.0, 1.0, nspace)
    rho = np.sqrt(psin_coords)
    psin_data = xr.DataArray(psin_coords, coords=[("rho_poloidal", rho)])
    attrs["transform"] = FluxSurfaceCoordinates(
        "poloidal",
    )

    def monotonic_series(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
        return np.linspace(start, stop, num, endpoint, retstep, dtype)

    ftor_min = 0.1
    ftor_max = 5.0
    result["ftor"] = xr.DataArray(
        np.outer(1 + tfuncs(times), monotonic_series(ftor_min, ftor_max, nspace)),
        coords=[("t", times), ("rho_poloidal", rho)],
        name="ftor",
        attrs=attrs,
    )
    result["ftor"].attrs["datatype"] = ("toroidal_flux", "plasma")

    if Btot_factor is None:
        f_min = 0.1
        f_max = 5.0
        time_vals = tfuncs(times)
        space_vals = monotonic_series(f_min, f_max, nspace)
        f_raw = np.outer(abs(1 + time_vals), space_vals)
    else:
        f_raw = np.outer(
            np.sqrt(
                Btot_factor**2
                - (raw_result["fbnd"] - raw_result["faxs"]) ** 2 / a_coeff**2
            ),
            np.ones_like(rho),
        )
        f_raw[:, 0] = Btot_factor

    result["f"] = xr.DataArray(
        f_raw, coords=[("t", times), ("rho_poloidal", rho)], name="f", attrs=attrs
    )
    result["f"].attrs["datatype"] = ("f_value", "plasma")
    result["rmjo"] = (result["rmag"] + a_coeff * psin_data**n_exp).assign_attrs(
        **attrs
    )
    result["rmjo"].name = "rmjo"
    result["rmjo"].attrs["datatype"] = ("major_rad", "lfs")
    result["rmjo"].coords["z"] = result["zmag"]
    result["rmji"] = (result["rmag"] - a_coeff * psin_data**n_exp).assign_attrs(
        **attrs
    )
    result["rmji"].name = "rmji"
    result["rmji"].attrs["datatype"] = ("major_rad", "hfs")
    result["rmji"].coords["z"] = result["zmag"]
    result["ajac"] = (
        np.pi * result["rmag"] * b_coeff * psin_data ** (2 * n_exp - 1)
    ).assign_attrs(**attrs)
    result["ajac"].name = "ajac"
    result["ajac"].attrs["datatype"] = ("area_jacobian", "plasma")
    result["vjac"] = (
        4
        * n_exp
        * np.pi**2
        * result["rmag"]
        * a_coeff
        * b_coeff
        * psin_data ** (2 * n_exp - 1)
    ).assign_attrs(**attrs)
    result["vjac"].name = "vjac"
    result["vjac"].attrs["datatype"] = ("volume_jacobian", "plasma")
    return result


def smooth_funcs(domain=(0.0, 1.0), max_val=None, min_terms=1, max_terms=11):
    if not max_val:
        max_val = 0.01
    min_val = -max_val
    nterms = 6
    coeffs = np.linspace(min_val, max_val, nterms)

    def f(x):
        x = (x - domain[0]) / (domain[1] - domain[0])
        term = 1
        y = xr.zeros_like(x) if isinstance(x, xr.DataArray) else np.zeros_like(x)
        for coeff in coeffs:
            y += coeff * term
            term *= x
        return y

    return f
