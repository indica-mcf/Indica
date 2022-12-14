"""Tests for line-of-sight coordinate transforms."""

from unittest.mock import MagicMock

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica import equilibrium
from indica.converters import flux_surfaces
from indica.converters import FluxSurfaceCoordinates
from indica.converters import line_of_sight
from indica.converters import line_of_sight_multi
from indica.converters import TrivialTransform


def default_inputs():
    x1 = 0.0  # does nothing
    x2 = DataArray(
        np.linspace(0.0, 1.0, 350, dtype=float)
    )  # index along line of sight, must be a DataArray
    t = 0.0  # does nothing

    return x1, x2, t


def load_los_default():
    # Line of sight origin tuple
    origin = (3.8, -2.0, 0.5)  # [xyz]

    # Line of sight direction
    direction = (-1.0, 0.0, 0.0)  # [xyz]

    # machine dimensions
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))

    # name
    name = "los_test"

    # Set-up line of sight class
    los = line_of_sight_multi.LineOfSightTransform(
        [origin[0]],
        [origin[1]],
        [origin[2]],
        [direction[0]],
        [direction[1]],
        [direction[2]],
        machine_dimensions=machine_dims,
        name=name,
    )

    return los, machine_dims


def load_los_default_multi():
    # Line of sight origin tuple
    origin = np.array([[3.8, -2.0, 0.5], [3.8, -2.0, 0.0]])  # [xyz]

    # Line of sight direction
    direction = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])  # [xyz]

    # machine dimensions
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))

    # name
    name = "los_test"

    # Set-up line of sight class
    los = line_of_sight_multi.LineOfSightTransform(
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


def convert_to_rho(plot=False):
    # Load line-of-sight default
    los, machine_dims = load_los_default()

    # Equilibrium
    # data, Te = equilibrium_dat_and_te()
    data = equilibrium_dat()
    Te = None
    offset = MagicMock(side_effect=[(0.02, False), (0.02, True)])
    equil = equilibrium.Equilibrium(
        data,
        Te,
        sess=MagicMock(),
        offset_picker=offset,
    )

    # Flux Transform
    flux_coord = flux_surfaces.FluxSurfaceCoordinates("poloidal")
    flux_coord.set_equilibrium(equil)

    # Assign flux transform
    los.assign_flux_transform(flux_coord)

    # Convert_to_rho method
    los.convert_to_rho(t=77.0)

    if plot:
        # centre column
        th = np.linspace(0.0, 2 * np.pi, 1000)
        x_cc = machine_dims[0][0] * np.cos(th)
        y_cc = machine_dims[0][0] * np.sin(th)

        # IVC
        x_ivc = machine_dims[0][1] * np.cos(th)
        y_ivc = machine_dims[0][1] * np.sin(th)

        plt.figure()
        plt.plot(los.x2, los.rho[0], "b")
        plt.ylabel("rho")

        plt.figure()
        plt.plot(x_cc, y_cc, "k--")
        plt.plot(x_ivc, y_ivc, "k--")
        plt.plot(los.x_start, los.y_start, "ro", label="start")
        plt.plot(los.x_end, los.y_end, "bo", label="end")
        plt.plot(los.x, los.y, "g", label="los")
        plt.legend()
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show(block=True)


# Test convert_to_xy method
def test_convert_to_xy(debug=False):
    # Load line-of-sight default
    los, machine_dims = load_los_default()
    x1, x2, t = default_inputs()

    # Test method
    x, y = los.convert_to_xy(x1, x2, t)
    _, z = los.convert_to_Rz(x1, x2, t)

    assert np.all(x.values <= np.max([los.x_start, los.x_end]))
    assert np.all(x >= np.min([los.x_start, los.x_end]))
    assert np.all(y <= np.max([los.y_start, los.y_end]))
    assert np.all(y >= np.min([los.y_start, los.y_end]))
    assert np.all(z <= np.max([los.z_start, los.z_end]))
    assert np.all(z >= np.min([los.z_start, los.z_end]))

    if debug:
        print(f"x = {x}")
        print(f"y = {y}")
        print(f"z = {z}")


# Test convert_to_Rz method
def test_convert_to_Rz(debug=False):
    # Load line-of-sight default
    los, machine_dims = load_los_default()
    x1, x2, t = default_inputs()

    # Test method
    R_, z_ = los.convert_to_Rz(x1, x2, t)

    x, y = los.convert_to_xy(x1, x2, t)
    R = np.sign(x) * np.sqrt(x**2 + y**2)

    # R and z are as expected=
    assert all(R == R_)

    if debug:
        print(f"R = {R}")


# Test convert_from_Rz method
def test_convert_from_Rz(debug=False):
    # Load line-of-sight default
    los, machine_dims = load_los_default()

    # Test inputs
    R_test = DataArray(2.5)  # Does not work as an array
    Z_test = DataArray(0.5)  # Does not work as an array
    t = 0.0

    # Test method
    _, x2_out = los.convert_from_Rz(R_test, Z_test, t)

    # x2 is within specified range
    assert x2_out <= 1 and x2_out >= 0

    if debug:
        print(f"x2_out2 = {x2_out}")
    return


# Test distance method
def test_distance(debug=False):
    # Load line-of-sight default
    los, machine_dims = load_los_default()
    x1, x2, t = default_inputs()

    # Test method
    dist = los.distance("dim_0", x1, x2, t)
    dls = [dist[i + 1] - dist[i] for i in range(len(dist) - 1)]

    # dl is identical along the line of sight up to 1 per million
    assert all(np.abs(dls - dls[0]) < (dls[0] * 1.0e-6))

    if debug:
        print(f"dist = {dist}")
    return


# Test distance method
def test_set_dl(debug=False):
    # Load line-of-sight default
    los, machine_dims = load_los_default()

    # Test inputs
    dl = 0.002

    # Test method
    x2, dl_out = los.set_dl(dl)

    assert np.abs(dl - dl_out) < 1.0e-6

    if debug:
        print(f"x2 = {x2}")
        print(f"dl_out = {dl_out}")
    return


# Test script for intersections
def test_intersections(debug=False):
    """Test script for intersections"""

    # Test parallel lines -> should return an empty list
    line_1_x = np.array([0.0, 1.0])
    line_1_y = np.array([1.0, 2.0])
    line_2_x = np.array([0.0, 1.0])
    line_2_y = np.array([2.0, 3.0])

    rx, zx, _, _ = line_of_sight.intersection(line_1_x, line_1_y, line_2_x, line_2_y)
    assert len(rx) == 0
    assert len(zx) == 0

    if debug:
        print(rx)
        print(zx)

    # Test intersecting lines - should return list of len=1
    line_3_x = np.array([0.0, 1.0])
    line_3_y = np.array([2.0, 1.0])
    rx, zx, _, _ = line_of_sight.intersection(line_1_x, line_1_y, line_3_x, line_3_y)
    assert len(rx) != 0
    assert len(zx) != 0

    if debug:
        print(rx)
        print(zx)


# Test LOS missing vessel
def test_missing_los():
    # Line of sight origin tuple
    origin = (4.0, -2.0, 0.5)  # [xyz]

    # Line of sight direction
    direction = (0.0, 1.0, 0.0)  # [xyz]

    # machine dimensions
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))

    # name
    name = "los_test"

    # Set-up line of sight class
    try:
        _ = line_of_sight.LineOfSightTransform(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            machine_dimensions=machine_dims,
            name=name,
        )
    except ValueError:
        # Value Error since the LOS does not intersect with machine dimensions
        print("LOS initialisation failed with ValueError as expected")


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
