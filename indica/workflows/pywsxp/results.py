"""
Process output data into netCDF files
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import xarray as xr
from indica.converters import LineOfSightTransform
from indica.converters.transect import TransectCoordinates
from indica.models.pinhole_camera import PinholeCamera
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_parameter
from indica.plasma import LabeledArray, PlasmaProfiler
from indica.utilities import assign_datatype
from xarray import DataArray, Dataset

from indica.workflows.pywsxp import optimise
from indica.workflows.pywsxp.diagnostic import calc_closest_approach, calc_weighted_rho
from indica.workflows.pywsxp.plasma import _ion_density_2d, make_plasma_2d
from indica.workflows.pywsxp.types import Config, Diagnostics, History, Inputs, Results


def convert_to_dataset(
    config: Config,
    profiler: PlasmaProfiler,
    diagnostics: Diagnostics,
    inputs: Inputs,
    results: Results,
    history: History,
    xknots: tuple[float, ...],
    concentrations: dict[str, tuple[float, int, int]],
    diag_to_scale: list[str],
    rho_2d: DataArray,
    R_0: DataArray,
    R_lfs: LabeledArray,
    t: float,
    avrg: float,
) -> Dataset:
    residual = optimise.residual(
        profiler=profiler,
        diagnostics=diagnostics,
        diag_to_scale=diag_to_scale,
        rho_2d=rho_2d,
        R_0=R_0,
        R_lfs=R_lfs,
        t=t,
        avrg=avrg,
    )
    plasma = profiler.plasma
    initial_density = xr.concat(
        [
            concentrations.get(elem, [0.0])[0] * plasma.electron_density.interp(t=t)
            for elem in plasma.elements
        ],
        "element",
    ).assign_coords({"element": ("element", list(plasma.elements))})
    fzt = xr.concat(
        [plasma.fz[imp].sel(t=t) for imp in plasma.elements], "element"
    ).assign_coords({"element": ("element", list(plasma.elements))})
    _dens_rho: list[DataArray] = []
    for elem in plasma.elements:
        try:
            _, n_lower, n_upper = concentrations[elem]
            _fz = fzt.sel(element=elem, ion_charge=range(n_lower, n_upper + 1)).sum(
                "ion_charge"
            )
            _fz_rho = float(_fz.rhop[_fz.argmax("rhop")].data)
        except (KeyError, TypeError):
            _fz_rho = 0.0
        _dens_rho.append(xr.DataArray(_fz_rho, coords={"t": t, "element": elem}))
    initial_density_rho = xr.concat(_dens_rho, "element")
    zeff = (
        plasma.ion_density * plasma.meanz * (plasma.meanz - 1) / plasma.electron_density
    )
    zeff.loc[{"element": plasma.main_ion}] = xr.ones_like(
        zeff.sel(element=plasma.main_ion)
    )
    assign_datatype(zeff, "effective_charge")
    asymmetry_parameter = centrifugal_asymmetry_parameter(
        plasma.ion_density,
        plasma.ion_temperature,
        plasma.electron_temperature,
        (plasma.toroidal_rotation / R_lfs),
        plasma.meanz,
        zeff,
        plasma.main_ion,
    )
    asymmetry_parameter.loc[plasma.main_ion] = (
        asymmetry_parameter.sel(element=plasma.main_ion) * 0.0
    )
    plasma_2d = make_plasma_2d(
        plasma,
        asymmetry_parameter=asymmetry_parameter.interp(t=t, rhop=rho_2d).drop_vars("t"),
        R_0=R_0,
    )
    conc = plasma.ion_density / plasma.electron_density
    assign_datatype(conc, "concentration")
    ion_density_2d = _ion_density_2d(
        plasma.ion_density.interp(t=t),
        asymmetry_parameter.interp(rhop=rho_2d, t=t),
        R_0,
    )
    assign_datatype(ion_density_2d, "ion_density")
    conc_2d = ion_density_2d / plasma.electron_density.interp(rhop=rho_2d, t=t)
    assign_datatype(conc_2d, "concentration")
    psin = plasma.rhop**2
    assign_datatype(psin, "psin")
    data_vars = {
        "results": DataArray(results, dims=("parameter",)),
        "rhop": plasma.rhop,
        "psin": psin,
        "rmag": plasma.equilibrium.rmag.interp(t=t),
        "zmag": plasma.equilibrium.zmag.interp(t=t),
        "xknots": np.asarray(xknots),
        "R_0": R_0,
        "rho_2d": rho_2d,
        "fzt": fzt,
        "electron_temperature": plasma.electron_temperature.interp(t=t),
        "electron_density": plasma.electron_density.interp(t=t),
        "ion_temperature": plasma.ion_temperature.interp(t=t),
        "ion_density": plasma.ion_density.interp(t=t),
        "ion_density_2d": ion_density_2d,
        "concentration": conc.interp(t=t),
        "concentration_2d": conc_2d,
        "toroidal_rotation": plasma.toroidal_rotation.interp(t=t),
        "asymmetry_parameter": asymmetry_parameter.interp(t=t),
        "total_radiation": plasma.total_radiation.interp(t=t),
        "total_radiation_2d": plasma.total_radiation.interp(t=t, rhop=rho_2d),
        "volume": plasma.volume.sel(t=t),
        "prad_tot": plasma.prad_tot.interp(t=t),
        "initial_density": initial_density,
        "initial_density_rho": initial_density_rho,
        "effective_charge": zeff.interp(t=t),
    }
    data_vars.update(
        {
            key: (
                ("iteration", "particle", "parameter")[: np.asarray(val).ndim],
                np.asarray(val),
            )
            for key, val in history.items()
        }
    )
    for name, prof in inputs.items():
        data = deepcopy(prof)
        try:
            rhop, _ = data.transform.convert_to_rho_theta(t=t)
            data_vars.update({f"input_{name}": data.interp(t=t), f"rhop_{name}": rhop})
        except AttributeError:
            pass
    for name, diag in diagnostics.items():
        diag.model.set_plasma(
            (plasma_2d if isinstance(diag.model, PinholeCamera) else plasma)
        )
        diag.model(t=t)
        if isinstance(diag.measurement.transform, TransectCoordinates):
            rhop = diag.measurement.transform.rhop
            theta = diag.measurement.transform.theta
        elif isinstance(diag.measurement.transform, LineOfSightTransform):
            closest_approach = calc_closest_approach(diag.measurement.transform, t=t)
            rhop, theta = closest_approach["rhop"], closest_approach["theta"]
        else:
            raise UserWarning(
                f"Unsupported type of transform: {type(diag.measurement.transform)}"
            )
        if diag.instrument.endswith("h"):
            pos = theta >= 0
        else:
            pos = (theta <= np.pi / 2) & (theta >= -np.pi / 2)
        impact_parameter = rhop.where(pos, other=-1.0 * rhop)
        if (emissivity := getattr(diag.model, "emissivity_element", None)) is not None:
            (
                weighted_rho,
                delta_weighted_rho,
                rho_los,
            ) = calc_weighted_rho(diag.model, t=t)
            assign_datatype(weighted_rho, "rhop")
            assign_datatype(delta_weighted_rho, "rhop")
            assign_datatype(rho_los, "rhop")
            delta_weighted_rho.attrs["long_name"] = r"$\delta\rho_{pol}$"
        else:
            weighted_rho, delta_weighted_rho, rho_los = None, None, None
        error = diag.measurement.error.where(
            (diag.measurement.t >= (t - avrg)) & (diag.measurement.t <= (t + avrg)),
            drop=True,
        ).mean("t")
        measurement = (
            diag.measurement.where(
                (diag.measurement.t >= (t - avrg)) & (diag.measurement.t <= (t + avrg)),
                drop=True,
            ).mean("t")
            * diag.rescale_factor
        )
        if "channel" not in measurement.dims:
            measurement = measurement.expand_dims({"channel": [0]}, axis=0)
        if "channel" not in error.dims:
            error = error.expand_dims({"channel": [0]}, axis=0)
        model = diag.model.bckc[diag.quantity]
        if "channel" not in model.dims:
            model = model.expand_dims({"channel": [0]}, axis=0)
        del model.attrs["transform"]
        data_vars.update(
            {
                f"{name}_{key}": quant.drop_vars(
                    [
                        val
                        for val in quant.coords
                        if val not in quant.dims and val != "t"
                    ]
                )
                for key, quant in (
                    ("measurement", measurement),
                    ("error", error),
                    ("model", model),
                    ("impact_parameter", impact_parameter),
                    ("emissivity", emissivity),
                    ("weighted_rho", weighted_rho),
                    ("delta_weighted_rho", delta_weighted_rho),
                    ("rho_line_of_sight", rho_los),
                )
                if quant is not None
            }
        )
        data_vars.update(
            {f"{name}_fit_weight": (diag.weight if diag.weight is not None else -1)}
        )
        if diag.channels is not None:
            data_vars.update({f"{name}_channels": diag.channels})
    data_vars.update(
        {
            key: val.drop_vars([var for var in val.coords if var not in val.dims])
            for key, val in data_vars.items()
            if isinstance(val, DataArray)
        }
    )
    for val in data_vars.values():
        if hasattr(val, "attrs"):
            val.attrs.pop("transform", None)
    diag_names = list(diagnostics.keys())
    data_vars["weights"] = xr.DataArray(
        [diagnostics[key].weight for key in diag_names],
        dims=("diagnostic",),
        coords={
            "diagnostic": diag_names,
            "t": float(t),
        },
    )
    data_vars["rescale_factors"] = xr.DataArray(
        [diagnostics[key].rescale_factor for key in diag_names],
        dims=("diagnostic",),
        coords={
            "diagnostic": diag_names,
            "t": float(t),
        },
    )
    data_vars["residual"] = xr.DataArray(
        residual,
        dims=("residual_index",),
        coords={"t": float(t)},
    )
    return Dataset(
        data_vars,
        coords={"pulse": config["pulse"], "t": float(t)},
        attrs={
            "config": json.dumps(config),
            "impurities": list(plasma.impurities),
            "diagnostics": diag_names,
            "diagnostics_to_scale": diag_to_scale,
            "avrg": avrg,
            "highz": str(config.get("highz")),
            "midz": str(config.get("midz")),
            "lowz": str(config.get("lowz")),
        },
    )


def main(datadir: Path):
    """Combine netCDF files in given directory"""
    datafiles = list(datadir.glob("results_*.nc"))
    if len(datafiles) == 0:
        raise UserWarning(f"No netCDF files found in {datadir}")
    combined = xr.concat([xr.load_dataset(f) for f in datafiles], dim="t")
    combined.sortby("t").to_netcdf(datadir / "results.nc")
    for df in datafiles:
        df.unlink()
