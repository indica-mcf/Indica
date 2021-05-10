from unittest.mock import MagicMock
from functools import reduce

import numpy as np
from pytest import approx
from pytest import mark
from scipy.integrate import quad

import xarray as xr
from xarray import broadcast
from xarray import DataArray
from xarray import where

from indica.converters import FluxSurfaceCoordinates
from indica.converters import TransectCoordinates
from indica.converters import TrivialTransform
from indica.converters import LinesOfSightTransform
from indica.equilibrium import Equilibrium

from scipy.optimize import fmin


def smooth_funcs(
    domain=(0.0, 1.0),
    max_val=None,
    min_terms=1,
    max_terms=11
):
    if not max_val:
        max_val = 0.01
    min_val = -max_val
    nterms = 6
    coeffs = np.linspace(min_val, max_val, nterms)

    def f(x):
        x = (x - domain[0]) / (domain[1] - domain[0])
        term = 1
        y = xr.zeros_like(x) \
            if isinstance(x, xr.DataArray) else np.zeros_like(x)
        for coeff in coeffs:
            y += coeff * term
            term *= x
        return y

    return f


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
        r_centre + tfuncs(times),
        coords=[("t", times)], name="rmag", attrs=attrs
    )
    result["rmag"].attrs["datatype"] = ("major_rad", "mag_axis")

    result["zmag"] = xr.DataArray(
        z_centre + tfuncs(times),
        coords=[("t", times)], name="zmag", attrs=attrs
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
        np.vectorize(lambda x: 0.8*x)(
            np.minimum(
                np.abs(machine_dims[0][0] - result["rmag"]),
                np.abs(machine_dims[0][1] - result["rmag"]),
            ),
        ),
        coords=[("t", times)],
    )
    if Btot_factor is None:
        b_coeff = xr.DataArray(
            np.vectorize(lambda x: 0.8*x)(
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
            np.vectorize(lambda axs, diff: axs + 0.03*diff)(
                result["faxs"], fdiff_max.values
            ),
            coords=[("t", times)],
            name="fbnd",
            attrs=attrs,
        )
    result["fbnd"].attrs["datatype"] = ("magnetic_flux", "separtrix")

    thetas = xr.DataArray(
        np.linspace(0.0, 2 * np.pi, nspace, endpoint=False),
        dims=["arbitrary_index"]
    )
    result["rbnd"] = (
        result["rmag"] + a_coeff * b_coeff /
        np.sqrt(a_coeff ** 2 * np.tan(thetas) ** 2 + b_coeff ** 2)
    ).assign_attrs(**attrs)
    result["rbnd"].name = "rbnd"
    result["rbnd"].attrs["datatype"] = ("major_rad", "separatrix")

    result["zbnd"] = (
        result["zmag"]
        + a_coeff
        * b_coeff
        / np.sqrt(a_coeff ** 2 + b_coeff ** 2 * np.tan(thetas) ** -2)
    ).assign_attrs(**attrs)
    result["zbnd"].name = "zbnd"
    result["zbnd"].attrs["datatype"] = ("z", "separatrix")

    r = np.linspace(machine_dims[0][0], machine_dims[0][1], nspace)
    z = np.linspace(machine_dims[1][0], machine_dims[1][1], nspace)
    rgrid = xr.DataArray(r, coords=[("R", r)])
    zgrid = xr.DataArray(z, coords=[("z", z)])
    psin = (
        (-result["zmag"] + zgrid) ** 2 / b_coeff ** 2
        + (-result["rmag"] + rgrid) ** 2 / a_coeff ** 2
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

    def monotonic_series(
        start, stop, num=50, endpoint=True, retstep=False, dtype=None
    ):
        return np.linspace(start, stop, num, endpoint, retstep, dtype)

    ftor_min = 0.1
    ftor_max = 5.0
    result["ftor"] = xr.DataArray(
        np.outer(
            1 + tfuncs(times), monotonic_series(ftor_min, ftor_max, nspace)
        ),
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
                Btot_factor ** 2 -
                (raw_result["fbnd"] - raw_result["faxs"]) ** 2 /
                a_coeff ** 2
            ),
            np.ones_like(rho),
        )
        f_raw[:, 0] = Btot_factor
    result["f"] = xr.DataArray(
        f_raw,
        coords=[("t", times), ("rho_poloidal", rho)], name="f", attrs=attrs
    )
    result["f"].attrs["datatype"] = ("f_value", "plasma")
    result["rmjo"] = (
        result["rmag"] + a_coeff * psin_data ** n_exp
    ).assign_attrs(**attrs)
    result["rmjo"].name = "rmjo"
    result["rmjo"].attrs["datatype"] = ("major_rad", "lfs")
    result["rmjo"].coords["z"] = result["zmag"]
    result["rmji"] = (
        result["rmag"] - a_coeff * psin_data ** n_exp
    ).assign_attrs(**attrs)
    result["rmji"].name = "rmji"
    result["rmji"].attrs["datatype"] = ("major_rad", "hfs")
    result["rmji"].coords["z"] = result["zmag"]
    result["vjac"] = (
        4
        * n_exp
        * np.pi ** 2
        * result["rmag"]
        * a_coeff
        * b_coeff
        * psin_data ** (2 * n_exp - 1)
    ).assign_attrs(**attrs)
    result["vjac"].name = "vjac"
    result["vjac"].attrs["datatype"] = ("volume_jacobian", "plasma")
    return result


def separable_funcs(*args):
    funcs = [args]

    def func(*coords):
        assert len(coords) == len(funcs)
        tmp = [f(c) for f, c in zip(funcs, coords)]
        return reduce(lambda x, y: x * y, tmp)

    return func


def noisy_funcs(func, rel_sigma=0.02, abs_sigma=1e-3):

    def noisy(*args):
        y = func(*args)
        max_std = 5
        variance = max_std * rel_sigma * np.max(np.abs(y)) + \
            max_std * abs_sigma
        elements = np.random.random_sample(y.shape)
        elements = (2.0 * elements) - 1.0
        elements *= variance
        return y + elements

    return noisy


def dropped_chnls(size, max_dropped=0.1):
    return np.random.randint(
        low=0,
        high=size if size > 0 else 1,
        size=int(max_dropped * size)
    ).tolist()


def data_arrays_from_coord(
    data_type=(None, None),
    coordinates=TrivialTransform(),
    axes=(0.0, 0.0, 0.0),
    data=separable_funcs(
        smooth_funcs(max_val=1e3),
        smooth_funcs(max_val=1e3),
        smooth_funcs(max_val=1e3),
    ),
    rel_sigma=0.02,
    abs_sigma=1e-3,
    uncertainty=True,
    max_dropped=0.1,
    require_dropped=False,
):
    general_type = (data_type[0])  # To-do: error-checking
    specific_type = (data_type[1])  # To-do: error-checking

    x1, x2, t = axes
    func = noisy_funcs(data, rel_sigma, abs_sigma) \
        if rel_sigma or abs_sigma else data
    coords = {}
    dims = []
    for n, c in [
        ("t", t), (coordinates.x1_name, x1), (coordinates.x2_name, x2)
    ]:
        if isinstance(c, np.ndarray) and c.ndim > 0:
            coords[n] = c.flatten()
            dims.append(n)
        elif isinstance(c, xr.DataArray) and c.ndim > 0:
            coords[n] = c.values
            dims.append(n)
        elif not isinstance(coordinates, LinesOfSightTransform):
            coords[n] = c if c is not None else 0.0
    shape = tuple(len(coords[dim]) for dim in dims)
    if isinstance(t, (np.ndarray, xr.DataArray)) and t.ndim > 0:
        min_val = np.min(t)
        width = np.abs(np.max(t) - min_val)
        t_scaled = (t - min_val) / (width if width else 1.0)
    else:
        t_scaled = np.array([0.0])
    if isinstance(x1, (np.ndarray, xr.DataArray)) and x1.ndim > 0:
        min_val = np.min(x1)
        width = np.abs(np.max(x1) - min_val)
        x1_scaled = (x1 - min_val) / (width if width else 1.0)
    else:
        x1_scaled = np.array([0.0])
    if isinstance(x2, (np.ndarray, xr.DataArray)) and x2.ndim > 0:
        min_val = np.min(x2)
        width = np.abs(np.max(x2) - min_val)
        x2_scaled = (x2 - min_val) / (width if width else 1.0)
    else:
        x2_scaled = np.array([0.0])
    tmp = func(x1_scaled, x2_scaled, t_scaled)
    result = xr.DataArray(np.reshape(tmp, shape), coords, dims)

    if isinstance(x1, np.ndarray):
        flat_x1 = x1.flatten()
    else:
        flat_x1 = x1
    dropped = (
        [
            flat_x1[i] for i in dropped_chnls(
                len(x1) if x1.ndim else 0, max_dropped
            )
        ]
        if isinstance(x1, np.ndarray)
        else []
    )
    if require_dropped and len(dropped) == 0:
        dropped = [flat_x1[0]]
    if uncertainty and (rel_sigma and abs_sigma):
        error = rel_sigma * result + abs_sigma
        result.attrs["error"] = error
    if dropped and flat_x1[0] != flat_x1[-1]:
        to_keep = np.logical_not(
            xr.DataArray(flat_x1, coords=[("x1", flat_x1)]).isin(dropped)
        )
        dropped_result = result.sel({coordinates.x1_name: dropped})
        result = result.where(to_keep)
        if uncertainty and (rel_sigma or abs_sigma):
            dropped_result.attrs["error"] = \
                result.attrs["error"].sel({coordinates.x1_name: dropped})
            result.attrs["error"] = result.attrs["error"].where(to_keep)
        result.attrs["dropped"] = dropped_result
    result.attrs["datatype"] = (general_type, specific_type)
    result.attrs["provenance"] = MagicMock()
    result.attrs["partial_provenance"] = MagicMock()
    result.attrs["transform"] = coordinates
    if hasattr(coordinates, "equilibrium"):
        result.indica.equilibrium = coordinates.equilibrium
    else:
        result.indica.equilibrium = MagicMock()
    return result


def electron_temp(rho, zmag):
    machine_dimensions = ((1.83, 3.9), (-1.75, 2.0))
    start_time = 75.0
    end_time = 80.0

    zmin = float(zmag.min())
    zmax = float(zmag.max())
    nspace = 15
    R_vals = np.linspace(*machine_dimensions[0], nspace)
    R_scaled = np.linspace(0.0, 1.0, nspace)
    zstart = 0.5 * (zmin + machine_dimensions[1][0])
    zend = 0.5 * (zmax + machine_dimensions[1][1])
    z_vals = np.linspace(zstart, zend, nspace)
    z_scaled = np.linspace(0.0, 1.0, nspace)
    ntime = 3
    times = np.linspace(start_time, end_time, ntime)
    times_scaled = np.linspace(0.0, 1.0, ntime)
    R_array = xr.DataArray(R_vals, dims="index")
    z_array = xr.DataArray(z_vals, dims="index")
    transform = TransectCoordinates(R_array, z_array)
    rhos = rho.interp(R=R_array, z=z_array)
    indices = rhos.indica.invert_root(
        1.0, "index", nspace - 1, method="cubic"
    ).assign_coords(t=(rhos.t - start_time) / (end_time - start_time))

    m = xr.DataArray(
        np.zeros((ntime, nspace, nspace), dtype=float) + -1e3 + -1e1,
        coords=[
            ("t", times_scaled),
            ("index", R_scaled),
            ("index_z_offset", z_scaled)
        ],
    )
    b = 1e2 - m * indices.interp(t=times_scaled, method="nearest") / nspace

    return data_arrays_from_coord(
        ("temperature", "electrons"),
        transform,
        [
            np.expand_dims(R_array, 1),
            np.expand_dims(z_array, 1),
            np.expand_dims(times, 1)
        ],
        lambda x1, x2, t: np.reshape(
            m.interp(
                t=np.ravel(t),
                index=np.ravel(x1),
                index_z_offset=np.ravel(x2)
            ).data,
            (len(t), len(x1), len(x2))
        ) * x1
        + np.reshape(
            b.interp(
                t=np.ravel(t),
                index=np.ravel(x1),
                index_z_offset=np.ravel(x2)
            ).data,
            (len(t), len(x1), len(x2))
        ),
        rel_sigma=0.001,
    )


def equilibrium_dat_and_te():
    data = equilibrium_dat()

    if False:
        rho = \
            np.sqrt(
                (data["psi"] - data["faxs"]) /
                (data["fbnd"] - data["faxs"])
            )
        Te = electron_temp(
            rho,
            data["zmag"],
        )
    else:
        Te = None
    return data, Te


def test_enclosed_volume():
    offset = MagicMock(return_value=0.02)
    """Generate equilibrium data
    """

    equilib_dat, Te = equilibrium_dat_and_te()
    
    """Check enclosed volume falls within reasonable bounds for the flux surface
    chosen. This is done by comparing the result to two different volumes which are
    constructed by assuming circular cross-sections with radii of the minimum
    minor radius and the maximum minor radius.

    """
    equilib = Equilibrium(equilib_dat, Te, sess=MagicMock(), offset_picker=offset)

    rho = np.array([0.5])
    time = np.array([77.5])

    Rmag = equilib.rmag.interp(
        t=time,
        method="linear",
        assume_sorted=True,
    )

    min_minor_radius = fmin(
        func=lambda th: equilib.minor_radius(rho, th, time)[0],
        x0=0.0,
        disp=False,
        full_output=True
    )[1]

    max_minor_radius = -1 * fmin(
        func=lambda th: -1 * equilib.minor_radius(rho, th, time)[0],
        x0=0.5 * np.pi,
        disp=False,
        full_output=True
    )[1]

    lower_limit_vol = (np.pi * min_minor_radius ** 2) * (2.0 * np.pi * Rmag)
    upper_limit_vol = (np.pi * max_minor_radius ** 2) * (2.0 * np.pi * Rmag)

    actual, _ = equilib.enclosed_volume(rho, time)

    assert (actual <= upper_limit_vol) and (actual >= lower_limit_vol)

    """Same as above but with multiple time values
    """

    Rmag = equilib.rmag.interp(
        t=equilib.rho.coords["t"],
        method="linear",
        assume_sorted=True,
    )

    min_minor_radius = np.array([])
    max_minor_radius = np.array([])
    for t_ in equilib.rho.coords["t"].data:
        t_ = np.array([t_])
        min_minor_radius = np.append(min_minor_radius,
            fmin(
                func=lambda th: equilib.minor_radius(rho, th, t_)[0],
                x0=0.0,
                disp=False,
                full_output=True
            )[1]
        )
        
        max_minor_radius = np.append(max_minor_radius,
            -1 * fmin(
                func=lambda th: -1 * equilib.minor_radius(rho, th, t_)[0],
                x0=0.5 * np.pi,
                disp=False,
                full_output=True
            )[1]
        )

    lower_limit_vol = (np.pi * min_minor_radius ** 2) * (2.0 * np.pi * Rmag)
    upper_limit_vol = (np.pi * max_minor_radius ** 2) * (2.0 * np.pi * Rmag)

    actual, _ = equilib.enclosed_volume(rho)

    assert np.all(actual <= upper_limit_vol) and np.all(actual >= lower_limit_vol)

    

def test_Btot():
    time = np.array([76.5])
    offset = MagicMock(return_value=0.02)
    """Generate equilibrium data
    """

    equilib_dat, Te = equilibrium_dat_and_te()

    equilib = Equilibrium(equilib_dat, Te, sess=MagicMock(), offset_picker=offset)

    """Arbitrary test data
    """

    R_input = equilib.rmag.interp(
        t=time,
        method="linear",
    )
    z_input = equilib.zmag.interp(
        t=time,
        method="linear",
    )

    max_rho_inboard = 1.0
    max_rho_outboard = max_rho_inboard
    
    max_R_inboard, _ = equilib.R_hfs(max_rho_inboard, time)

    max_R_outboard, _ = equilib.R_lfs(max_rho_outboard, time)

    max_height_R, max_height_z, _ = equilib.spatial_coords(1.0, 0.5*np.pi, time)

    min_height_R, min_height_z, _ = equilib.spatial_coords(1.0, -0.5*np.pi, time)

    R_multi_input = equilib.rmag

    z_multi_input = equilib.zmag

    def test_nan_B(Total_B, rho_):
        Total_B = Total_B.transpose(*rho_.dims)
        rho_ = rho_.data.flatten()
        Total_B = Total_B.data.flatten()

        """Check whether all Total_B values for rho > 1 are NaN
        """
        assert np.all(np.isnan(Total_B[np.where(np.abs(rho_)>1)[0]]))

        """Check whether all Total_B values for rho <= 1 are positive
        """
        assert np.all(Total_B[np.where(np.abs(rho_)<=1)[0]] >= 0)

    """Magnetic field strength at magnetic axis
    """
    Total_B, _ = equilib.Btot(R_input, z_input, time)

    rho_, theta_, t_ = equilib.flux_coords(R_input, z_input, time)
    test_nan_B(Total_B, rho_)

    """Magnetic field strength at magnetic axis at all times
    """
    Total_B, _ = equilib.Btot(R_multi_input, z_multi_input)

    rho_, theta_, t_ = equilib.flux_coords(R_multi_input, z_multi_input)
    test_nan_B(Total_B, rho_)

    """Magnetic field strength at inboard(high-field) side
    """
    Total_B, _ = equilib.Btot(max_R_inboard, z_input, time)

    rho_, theta_, t_ = equilib.flux_coords(max_R_inboard, z_input, time)
    test_nan_B(Total_B, rho_)

    """Magnetic field strength at outboard(low-field) side
    """
    Total_B, _ = equilib.Btot(max_R_outboard, z_input, time)

    rho_, theta_, t_ = equilib.flux_coords(max_R_outboard, z_input, time)
    test_nan_B(Total_B, rho_)

    """Magnetic field strength at maximum height of the device (whilst inside LCFS)
    """
    Total_B, _ = equilib.Btot(max_height_R, max_height_z, time)

    rho_, theta_, t_ = equilib.flux_coords(max_height_R, max_height_z, time)
    test_nan_B(Total_B, rho_)

    """Magnetic field strength at minimum height of the device (whilst inside LCFS)
    """
    Total_B, _ = equilib.Btot(min_height_R, min_height_z, time)

    rho_, theta_, t_ = equilib.flux_coords(min_height_R, min_height_z, time)
    test_nan_B(Total_B, rho_)
    
    """Magnetic field strength at all locations on the (R, z) grid at a given time
    """
    Total_B, _ = equilib.Btot(equilib.psi.coords["R"], equilib.psi.coords["z"], time)

    rho_, theta_, t_ = equilib.flux_coords(equilib.psi.coords["R"], equilib.psi.coords["z"], time)    
    test_nan_B(Total_B, rho_)

    """Magnetic field strength at all locations on the (R, z) grid at all times
    """
    Total_B, _ = equilib.Btot(equilib.psi.coords["R"], equilib.psi.coords["z"])
    
    rho_, theta_, t_ = equilib.flux_coords(equilib.psi.coords["R"], equilib.psi.coords["z"])
    test_nan_B(Total_B, rho_)

