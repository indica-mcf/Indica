import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from indica.converters import FluxSurfaceCoordinates
from indica.converters import TrivialTransform
from indica.converters.time import get_tlabels_dt
from indica.equilibrium import Equilibrium

MACHINE_DIMS = ((0.15, 0.85), (-0.75, 0.75))
DEFAULT_PARAMS = {
    "poloidal_a": 0.5,
    "poloidal_b": 1.0,
    "poloidal_n": 1,
    "poloidal_alpha": 0.01,
    "toroidal_a": 0.7,
    "toroidal_b": 1.4,
    "toroidal_n": 1,
    "toroidal_alpha": -0.00005,
    "Btot_a": 1.0,
    "Btot_b": 1.0,
    "Btot_alpha": 0.001,
}


def smooth_funcs(domain=(0.0, 1.0), max_val=None):
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


def fake_equilibrium(
    tstart: float = 0,
    tend: float = 0.1,
    dt: float = 0.01,
    machine_dims=None,
    times=None,
):
    equilibrium_data = fake_equilibrium_data(
        tstart=tstart,
        tend=tend,
        dt=dt,
        machine_dims=machine_dims,
        times=times,
    )
    return Equilibrium(equilibrium_data)


def fake_equilibrium_data(
    tstart: float = 0,
    tend: float = 0.1,
    dt: float = 0.01,
    machine_dims=None,
    times=None,
):
    def monotonic_series(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
        return np.linspace(start, stop, num, endpoint, retstep, dtype)

    if machine_dims is None:
        machine_dims = MACHINE_DIMS

    if times is None:
        get_tlabels_dt(tstart, tend, dt)
        times = np.arange(tstart, tend + dt, dt)
    # ntime = times.size
    Btot_factor = None

    result = {}
    nspace = 100

    tfuncs = smooth_funcs((tstart, tend), 0.01)
    r_centre = (machine_dims[0][0] + machine_dims[0][1]) / 2
    z_centre = (machine_dims[1][0] + machine_dims[1][1]) / 2
    raw_result = {}
    attrs = {
        "transform": TrivialTransform(),
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
    # psi.attrs["provenance"] = MagicMock()
    # psi.attrs["partial_provenance"] = MagicMock()
    psi.attrs["datatype"] = ("magnetic_flux", "plasma")
    result["psi"] = psi

    psin_coords = np.linspace(0.0, 1.0, nspace)
    rho1d = np.sqrt(psin_coords)
    psin_data = xr.DataArray(psin_coords, coords=[("rho_poloidal", rho1d)])
    attrs["transform"] = FluxSurfaceCoordinates("poloidal")
    result["psin"] = psin_data

    ftor_min = 0.1
    ftor_max = 5.0
    result["ftor"] = xr.DataArray(
        np.outer(1 + tfuncs(times), monotonic_series(ftor_min, ftor_max, nspace)),
        coords=[("t", times), ("rho_poloidal", rho1d)],
        name="ftor",
        attrs=attrs,
    )
    result["ftor"].attrs["datatype"] = ("toroidal_flux", "plasma")

    # It should be noted during this calculation that the full extent of theta
    # isn't represented in the resultant rbnd and zbnd values.
    # This is because for rbnd: 1/sqrt(tan(x)^2) and for zbnd: 1/sqrt(tan(x)^-2)
    # are periodic functions which span a fixed 0 to +inf range on the y-axis
    # between 0 and 2pi, with f(x) = f(x+pi) and f(x) = f(pi-x)
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

    # Indices of thetas for,
    # 90 <= thetas < 180
    # 180 <= thetas < 270
    # 270 <= thetas < 360
    arcs = {
        "90to180": np.flatnonzero((0.5 * np.pi <= thetas) & (thetas < 1.0 * np.pi)),
        "180to270": np.flatnonzero((1.0 * np.pi <= thetas) & (thetas < 1.5 * np.pi)),
        "270to360": np.flatnonzero((1.5 * np.pi <= thetas) & (thetas < 2.0 * np.pi)),
    }

    # Transforms rbnd appropriately to represent the values when
    # 90 <= theta < 180 and 180 <= theta < 270
    result["rbnd"][:, arcs["90to180"]] = (
        -result["rbnd"][:, arcs["90to180"]] + 2 * result["rmag"]
    )
    result["rbnd"][:, arcs["180to270"]] = (
        -result["rbnd"][:, arcs["180to270"]] + 2 * result["rmag"]
    )

    # Transforms zbnd appropriately to represent the values when
    # 180 <= theta < 270 and 270 <= theta < 360
    result["zbnd"][:, arcs["180to270"]] = (
        -result["zbnd"][:, arcs["180to270"]] + 2 * result["zmag"]
    )
    result["zbnd"][:, arcs["270to360"]] = (
        -result["zbnd"][:, arcs["270to360"]] + 2 * result["zmag"]
    )

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
            np.ones_like(rho1d),
        )
        f_raw[:, 0] = Btot_factor

    result["f"] = xr.DataArray(
        f_raw, coords=[("t", times), ("rho_poloidal", rho1d)], name="f", attrs=attrs
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

    result["ajac"] = (
        2 * n_exp * np.pi * a_coeff * b_coeff * psin_data ** (2 * n_exp - 1)
    ).assign_attrs(**attrs)
    result["ajac"].name = "ajac"
    result["ajac"].attrs["datatype"] = ("area_jacobian", "plasma")

    return result


if __name__ == "__main__":
    # Diagnostic example to show that now the full theta extent is
    # represented in the rbnd and zbnd values.
    test_equilibrium = fake_equilibrium()

    plt.plot(test_equilibrium.rbnd.isel(t=0), test_equilibrium.zbnd.isel(t=0))
    plt.xlabel("R")
    plt.ylabel("z")
    plt.axis("scaled")
    plt.show()
