"""Common JET workflow boilerplate code

Instantiates a Plasma class with JET-relevant conditions
"""

from typing import Optional
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
import xarray as xr
from xarray import DataArray

from indica.models.pinhole_camera import PinholeCamera
from indica.numpy_typing import LabeledArray
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_parameter
from indica.plasma import PlasmaProfiler
from indica.utilities import assign_datatype
from indica.workflows.pywsxp.utilities import Diagnostic
from indica.workflows.pywsxp.utilities import make_plasma_2d

DEFAULT_OPTIONS: dict[str, float] = {"c1": 0.65, "c2": 0.35, "w": 0.9}


def _profile_parameters(parameters: list[float]) -> dict[str, float]:
    y0 = float(10 ** parameters[0])
    shape = [float(val) for val in parameters[1:-1]]
    y1 = float(parameters[-1])
    return {
        "y0": y0,
        **{f"shape{i:02}": val for i, val in enumerate(shape)},
        "y1": y1,
    }


def recover_profiles(
    profiler: PlasmaProfiler,
    parameters: NDArray[np.float32],
    impurities: tuple[str, ...],
    xknots: tuple[float, ...],
    time: Union[int, float],
) -> PlasmaProfiler:
    ne_core = float(profiler.plasma.electron_density.interp(t=time, rhop=0.0).data)
    n_knots = len(xknots) - 1
    params: dict[str, float] = {}
    idx = 0
    for imp in impurities:
        if profiler.plasma.element_z.sel(element=imp).data <= 20:
            profiler.plasma.set_impurity_concentration(
                imp,
                ((10 ** parameters[idx]) / ne_core),
                time,
            )
            idx += 1
        else:
            params.update(
                **{
                    f"impurity_density:{imp}.{key}": p
                    for key, p in _profile_parameters(
                        [*(parameters[idx : idx + n_knots]), 0.0]
                    ).items()
                }
            )
            idx += n_knots
    params.update(
        {
            f"toroidal_rotation.{key}": p
            for key, p in _profile_parameters([*(parameters[-4:]), 0.0]).items()
        }
    )
    profiler(params, t=time)
    return profiler


def residual(
    profiler: PlasmaProfiler,
    diagnostics: dict[str, Diagnostic],
    diag_to_scale: list[str],
    rho_2d: DataArray,
    R_0: DataArray,
    R_lfs: LabeledArray,
    t: float,
    avrg: float = 0.01,
) -> list[float]:
    res: list[float] = []
    plasma = profiler.plasma
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
    ).interp(t=t)
    asymmetry_parameter.loc[plasma.main_ion] = (
        asymmetry_parameter.sel(element=plasma.main_ion) * 0.0
    )
    plasma_2d = make_plasma_2d(
        plasma,
        asymmetry_parameter=asymmetry_parameter.interp(rhop=rho_2d).drop_vars("t"),
        R_0=R_0,
    )
    for key in sorted(diagnostics.keys()):
        if diagnostics[key].weight is None:
            continue
        diagnostics[key].model.set_plasma(
            (plasma_2d if isinstance(diagnostics[key].model, PinholeCamera) else plasma)
        )
        data_slice: DataArray = diagnostics[key].measurement
        if diagnostics[key].channels is not None and "channel" in data_slice.dims:
            data_slice = data_slice.sel(channel=diagnostics[key].channels)
        error_slice = data_slice.error.where(
            (data_slice.t >= (t - avrg)) & (data_slice.t <= (t + avrg)), drop=True
        ).mean("t")
        data_slice = data_slice.where(
            (data_slice.t >= (t - avrg)) & (data_slice.t <= (t + avrg)), drop=True
        ).mean("t")
        model_result: DataArray = (
            diagnostics[key].model(t=[t])[diagnostics[key].quantity].sel(t=t)
        )
        model_slice: DataArray = model_result.sel(
            {
                dim: data_slice.coords[dim]
                for dim in data_slice.dims
                if dim in model_result.dims
            },
            method="nearest",
        )
        model_slice = model_slice.where(model_slice >= 0, other=0.0)
        if np.any(plasma_2d.ion_density < 0):
            model_slice = xr.ones_like(data_slice) * 100
        if "channel" not in data_slice.coords:
            data_slice = data_slice.expand_dims({"channel": [0]})
        nchan = len(data_slice.channel)
        weight = diagnostics[key].weight / nchan
        _locmean = data_slice.channel[
            np.abs(data_slice.mean("channel") - data_slice).argmax("channel")
        ]
        if diagnostics[key].instrument in diag_to_scale:
            _sfact = float((model_slice / data_slice).sel(channel=_locmean))
            diagnostics[key].rescale_factor = _sfact
            (_scalelow, _scale, _scalehigh) = (
                (0.5, 2.35, 3.5) if "sxr" in key else (0.5, 1.0, 1.5)
            )
            if _sfact < _scalelow:
                res.append(0.01 * (_sfact - _scalelow))
            elif _sfact > _scalehigh:
                res.append(0.01 * (_sfact - _scalehigh))
            else:
                res.append(0.0)
        data_slice = data_slice * diagnostics[key].rescale_factor
        if np.all((error_slice == 0.0) | error_slice.isnull()):
            error_slice = xr.zeros_like(data_slice) + float(0.05 * data_slice.mean())
        _res = np.array(
            np.sqrt(weight) * ((model_slice - data_slice) / error_slice).to_numpy(),
            ndmin=1,
        )
        res.extend(_res.flatten().tolist())

    for key, prf in profiler.profilers.items():
        if all(
            (
                ("impurity_density" in key or "toroidal_rotation" in key),
                (x := getattr(prf, "xspl")) is not None,
                isinstance((spline := getattr(prf, "spline")), CubicSpline),
            )
        ):
            d2spline = np.abs(spline(x, 2))
            d2spline[d2spline == 0] = d2spline[d2spline > 0].min()
            res.append(0.1 * np.trapz((np.log10(d2spline)), x))

    return np.asarray(res).tolist()


def costfn(
    parameters: NDArray[np.float32],
    profiler: PlasmaProfiler,
    diagnostics: dict[str, Diagnostic],
    impurities: tuple[str, ...],
    xknots: tuple[float, ...],
    diag_to_scale: list[str],
    rho_2d: DataArray,
    R_0: DataArray,
    R_lfs: LabeledArray,
    t: float,
    avrg: float = 0.01,
) -> NDArray[np.float32]:
    cost = []
    for p_params in parameters:
        recover_profiles(
            profiler,
            p_params,
            impurities,
            xknots,
            t,
        )
        res = residual(
            profiler=profiler,
            diagnostics=diagnostics,
            diag_to_scale=diag_to_scale,
            rho_2d=rho_2d,
            R_0=R_0,
            R_lfs=R_lfs,
            t=t,
            avrg=avrg,
        )
        cost.append(np.sum(np.asarray(res).flatten() ** 2))
    return np.asarray(cost)


def parse_residual(
    resid: NDArray[np.float32],
    diagnostics: dict[str, DataArray],
    diag_to_scale: Optional[list[str]] = None,
) -> dict[str, list[float]]:
    if diag_to_scale is None:
        diag_to_scale = []
    outp: dict[str, list[float]] = {}
    for key, diagnostic in sorted(diagnostics.items()):
        if "channel" in diagnostic.coords:
            idx = len(diagnostic.channel)
        else:
            idx = 1
        if any([key.startswith(diag) for diag in diag_to_scale]):
            resid = resid[1:]
        outp[key] = resid.tolist()[:idx]
        resid = resid[idx:]
    return outp
