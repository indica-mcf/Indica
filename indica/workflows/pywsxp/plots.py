"""
Functions to quickly generate plots for SXR analysis
"""

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from indica.profilers.profiler_spline import ProfilerCubicSpline
from indica.utilities import assign_datatype, get_element_info
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pyswarms.utils.plotters import plot_cost_history
from xarray import DataArray, Dataset

from indica.workflows.pywsxp.optimise import _profile_parameters, parse_residual


def _chisqr(data: Dataset) -> tuple[float, float]:
    """
    Calculate $chi^2$ and reduced $chi^2$
    """
    chisqr = float(np.sum(data.residual**2))
    nfree = len(data.residual) - len(data.results)
    redchisqr = float(chisqr / nfree) if nfree >= 1 else np.nan
    return chisqr, redchisqr


def _standard_title(data: Dataset) -> str:
    """Common title for all plots"""
    chisqr, redchisqr = _chisqr(data)
    return (
        f"JPN {int(data.pulse)}, t={float(data.t):.3f}s, "
        f"$\\chi^2$={chisqr:.3g}, $\\chi_\\nu^2$={redchisqr:.3g}"
    )


def _make_figure(dlist: Iterable[Any]) -> tuple[Figure, int, int]:
    ncols = 4 if len(dlist) > 4 else len(dlist)
    nrows = math.ceil(len(dlist) / ncols)
    return (
        plt.figure(figsize=((5 * ncols), (5 * nrows)), layout="constrained"),
        nrows,
        ncols,
    )


def _annotate(ax, label, x, y, xytext):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=8,
        # arrowprops={"arrowstyle": "-|>", "color": "black"},
    )


def _plot_input_profiles(data: Dataset) -> Figure:
    if "t" in data.dims:
        data = data.isel(t=0)
    _profs = [
        "electron_density",
        "electron_temperature",
        "ion_temperature",
        "toroidal_rotation",
        "effective_charge",
    ]
    fig, nrows, ncols = _make_figure(_profs)
    axes = fig.subplots(nrows, ncols, sharey=False, sharex=True)
    for ax, pr in zip(axes.flatten(), _profs):
        if (_pr := getattr(data, pr, None)) is None:
            continue
        if "element" in _pr.dims:
            _attrs = deepcopy(_pr.attrs)
            _pr = _pr.sum("element")
            _pr.attrs.update(_attrs)
        _pr.plot(x="rhop", ax=ax)
        if all(
            (
                (input_pr := getattr(data, f"input_{pr}", None)) is not None,
                (input_rhop := getattr(data, f"rhop_{pr}", None)) is not None,
            )
        ):
            (
                input_pr.assign_coords({"rhop": ("channel", input_rhop.data)})
                .swap_dims({"channel": "rhop"})
                .plot(
                    x="rhop",
                    linestyle=" ",
                    marker="x",
                    ax=ax,
                    label="/".join(
                        [str(val) for val in json.loads(data.config).get(pr, [])]
                    ),
                )
            )
            ax.legend()
        ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
        ax.set_title(None)
        if pr == "effective_charge":
            ax.set_ylim(bottom=1.0)
    fig.suptitle("Inputs: " + _standard_title(data))
    return fig


def _plot_concentration_profiles(data: Dataset, fig: Optional[Figure] = None) -> Figure:
    if "t" in data.dims:
        data = data.isel(t=0)
    if fig is None:
        fig = plt.figure(figsize=(15, 10), layout="constrained")
    ncols = 3
    axes = fig.subplots(
        math.ceil(len(data.ion_density_2d.element) / ncols),
        ncols,
        sharey=False,
        sharex=True,
    )
    conc_2d = data.concentration_2d.assign_coords(
        {
            "rhop": (
                ("z", "R"),
                data.rho_2d.where(
                    data.ion_density_2d.R >= data.rmag,
                    other=-1.0 * data.rho_2d,
                ).data,
            )
        }
    ).interp(z=data.zmag)
    for ax, element in zip(axes.flatten(), conc_2d.element):
        conc_2d.sel(element=element).plot(x="rhop", ax=ax)
        if element in [data.highz, data.midz]:
            for xknot in data.xknots:
                ax.plot(
                    xknot,
                    float(
                        data.concentration.sel(element=element).interp(rhop=xknot).data
                    ),
                    c="red",
                    marker="+",
                )
        if all(
            (
                (
                    input_pr := getattr(
                        data,
                        f"input_impurity_density:{str(element.data)}",
                        None,
                    )
                )
                is not None,
                (
                    input_rhop := getattr(
                        data,
                        f"rhop_impurity_density:{str(element.data)}",
                        None,
                    )
                )
                is not None,
            )
        ):
            input_dens = (
                input_pr.assign_coords({"rhop": ("channel", input_rhop.data)})
                .swap_dims({"channel": "rhop"})
                .dropna("rhop")
            )
            input_conc = input_dens / data.electron_density.interp(rhop=input_dens.rhop)
            assign_datatype(input_conc, "concentration")
            (
                input_conc.plot(
                    x="rhop",
                    linestyle=" ",
                    marker="x",
                    ax=ax,
                    # label="/".join(
                    #     [str(val) for val in json.loads(data.config).get(pr, [])]
                    # ),
                )
            )
            # ax.legend()
        ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
        ax.set_title(f"{str(element.data).capitalize()}")
    fig.suptitle("Concentrations: " + _standard_title(data))
    return fig


def _plot_radiation_profiles(data: Dataset, fig: Optional[Figure] = None) -> Figure:
    if "t" in data.dims:
        data = data.isel(t=0)
    if fig is None:
        fig = plt.figure(figsize=(15, 10), layout="constrained")
    ncols = 3
    axes = fig.subplots(
        math.ceil(len(data.ion_density_2d.element) / ncols),
        ncols,
        sharey=False,
        sharex=True,
    )
    rad_2d = data.total_radiation_2d.assign_coords(
        {
            "rhop": (
                ("z", "R"),
                data.rho_2d.where(
                    data.total_radiation_2d.R >= data.rmag,
                    other=-1.0 * data.rho_2d,
                ).data,
            )
        }
    ).interp(z=data.zmag)
    for ax, element in zip(axes.flatten(), rad_2d.element):
        rad_2d.sel(element=element).plot(x="rhop", ax=ax)
        ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
        ax.set_title(f"{str(element.data).capitalize()}")
    fig.suptitle("Radiation: " + _standard_title(data))
    return fig


def _plot_profiles(data: Dataset, fig: Optional[Figure] = None) -> list[Figure]:
    return [
        _plot_input_profiles(data=data),
        _plot_concentration_profiles(data=data, fig=fig),
        _plot_radiation_profiles(data=data, fig=fig),
    ]


def _plot_radiation(data: Dataset, fig: Optional[Figure] = None) -> Figure:
    if "t" in data.dims:
        data = data.isel(t=0)
    if fig is None:
        fig = plt.figure(figsize=(16, 10), layout="constrained")
    radiation_diagnostics = [
        diag
        for diag in (
            [data.attrs["diagnostics"]]
            if isinstance(data.attrs["diagnostics"], str)
            else data.attrs["diagnostics"]
        )
        if getattr(data, f"{diag}_emissivity", None) is not None
    ]
    axes = fig.subplots(
        nrows=len(radiation_diagnostics),
        ncols=len(data.attrs["impurities"]) + 2,
        sharex=True,
        sharey=True,
        squeeze=False,
        subplot_kw={"aspect": "equal"},
    )
    for row, diag in zip(axes, radiation_diagnostics):
        emissivity = getattr(data, f"{diag}_emissivity")
        for col, elem in zip(row[:-1], data.element):
            qmesh = emissivity.sel(element=elem).plot(x="R", y="z", ax=col)
            qmesh.colorbar.ax.ticklabel_format(
                style="sci", scilimits=(-2, 2), useMathText=True
            )
            qcs = data.rho_2d.plot.contour(
                levels=np.linspace(0.3, 0.9, 3), ax=col, cmap="plasma"
            )
            col.clabel(qcs)
            col.set_title(
                f"{diag.split('_')[0].upper()} ({str(elem.data).capitalize()})"
            )
        qmesh = emissivity.sum("element").plot(x="R", y="z", ax=row[-1])
        qmesh.colorbar.ax.ticklabel_format(
            style="sci", scilimits=(-2, 2), useMathText=True
        )
        qcs = data.rho_2d.plot.contour(
            levels=np.linspace(0.3, 0.9, 3), ax=row[-1], cmap="plasma"
        )
        row[-1].clabel(qcs)
        row[-1].set_title(f"{diag.split('_')[0].upper()}, Total")
    fig.suptitle(_standard_title(data))
    return fig


def _plot_history(data: Dataset, fig: Optional[Figure] = None) -> Figure:
    if "t" in data.dims:
        data = data.isel(t=0)
    if fig is None:
        fig = plt.figure(figsize=(10, 10), layout="tight")
    impurities: tuple[str, ...] = tuple(
        val
        for val in (str(data.highz), str(data.midz), str(data.lowz))
        if val != "None"
    )
    n_elem = len(impurities)
    n_knots = len(data.xknots) - 1
    pos = data.pos
    asym_fitted = True
    ax_cost = fig.add_subplot(
        1 + math.ceil(n_elem / 2), (2 if asym_fitted is True else 1), 1
    )
    plot_cost_history(data.cost, ax=ax_cost)
    ax_cost.axvline(data.cost.argmin(), color="green", linestyle="--", alpha=0.5)
    if (data.cost.max() / data.cost.min()) >= 100:
        ax_cost.set_yscale("log")
    ax_vtor = None  # type: ignore
    if asym_fitted is True:
        ax_vtor = fig.add_subplot(1 + math.ceil(n_elem / 2), 2, 2)
        vtor_profiler = ProfilerCubicSpline(
            datatype="toroidal_rotation",
            xspl=data.rhop,
            parameters={
                "xknots": [0.0, 0.3, 0.6, 0.9, 1.05],
                **_profile_parameters(np.array([0.0] * 5, ndmin=1)),
            },
        )
        p = []
        for part in (
            pos.isel(
                parameter=range(len(pos.parameter) - 4, len(pos.parameter)),
                # iteration=range(
                #     0, len(pos.iteration), max(20, len(pos.iteration) // 20)
                # ),
            )
            .stack(point=["iteration", "particle"], create_index=False)
            .transpose(..., "parameter")
        ):
            vtor_profiler.set_parameters(
                **_profile_parameters(np.asarray([*part.data.tolist(), 0.0]))
            )
            p.append(xr.ones_like(data.toroidal_rotation) * vtor_profiler())
        xr.concat(p, dim="point").plot(
            x="rhop",
            hue="point",
            yscale="log",
            c="grey",
            alpha=0.1,
            add_legend=False,
        )
        data.toroidal_rotation.plot(
            x="rhop",
            color="green",
            label="Final Toroidal Rotation",
            ax=ax_vtor,
        )
        ax_vtor.legend()
        ax_vtor.set_title("")
        ax_vtor.set_ylabel("Toroidal Rotation")
        ax_vtor.set_yscale("log")
    ax = None  # type: ignore
    dens_profiler = ProfilerCubicSpline(
        "ion_density",
        xspl=data.rhop,
        parameters={
            "xknots": data.xknots,
            **_profile_parameters(np.array([0.0] * (n_knots + 1), ndmin=1)),
        },
    )
    idx = 0
    for i, imp in enumerate(impurities):
        ax = fig.add_subplot(
            1 + math.ceil(n_elem / 2),
            2,
            i + 3,
            sharex=ax_vtor if ax_vtor is not None else ax,
            sharey=ax,
        )
        if get_element_info(imp)[0] <= 20:
            p = (
                10
                ** pos.isel(
                    parameter=idx,
                    iteration=range(
                        0, len(pos.iteration), max(20, len(pos.iteration) // 20)
                    ),
                ).stack(
                    point=["iteration", "particle"],
                    create_index=False,
                )
            ) / data.electron_density
            idx += 1
        else:
            p = []
            for part in (
                pos.isel(
                    parameter=range(idx, idx + n_knots),
                    iteration=range(0, len(pos.iteration), 10),
                )
                .stack(point=["iteration", "particle"], create_index=False)
                .transpose(..., "parameter")
            ):
                dens_profiler.set_parameters(**_profile_parameters([*part, 0.0]))
                conc = dens_profiler() / data.electron_density
                if np.all(conc > 0):
                    p.append(conc)
            p = xr.concat(p, dim="point")
            idx += n_knots
        assign_datatype(p, "concentration")
        p.plot(
            x="rhop",
            hue="point",
            yscale="log",
            c="grey",
            alpha=0.1,
            add_legend=False,
        )
        (data.ion_density / data.electron_density).sel(element=imp).plot(
            x="rhop",
            color="green",
            label="Final Concentration",
            ax=ax,
        )
        if np.all(data.initial_density.sel(element=imp) > 0):
            (data.initial_density / data.electron_density).sel(element=imp).plot(
                x="rhop",
                color="red",
                linestyle="--",
                label="Initial Concentration",
                ax=ax,
            )
        ax.legend()
        ax.set_title("")
        ax.set_ylabel(f"{str(imp).capitalize()} concentration")
        ax.set_yscale("log")

    fig.suptitle(_standard_title(data))
    return fig


def _plot_diagnostics(data: Dataset, fig: Optional[Figure] = None) -> Figure:
    if "t" in data.dims:
        data = data.isel(t=0)
    diagnostics = (
        [data.diagnostics] if isinstance(data.diagnostics, str) else data.diagnostics
    )
    if fig is None:
        fig, nrows, ncols = _make_figure(diagnostics)
    subfigs = np.array(fig.subfigures(nrows, ncols), ndmin=1)
    ax_prev = None
    residual = parse_residual(
        data.residual.data,
        {
            name: getattr(data, f"{name}_measurement").sel(
                channel=getattr(
                    data,
                    f"{name}_channels",
                    getattr(data, f"{name}_measurement").dropna("channel").channel,
                )
            )
            for name in diagnostics
            if getattr(data, f"{name}_fit_weight", -1) > 0
        },
        (
            [data.diagnostics_to_scale]
            if isinstance(data.diagnostics_to_scale, str)
            else data.diagnostics_to_scale
        ),
    )
    for subfig, weight, rescale_factor, diagnostic in zip(
        subfigs.flatten(),
        data.weights.data,
        data.rescale_factors.data,
        data.rescale_factors.diagnostic.data,
    ):
        gs = subfig.add_gridspec(3, 1)
        ax = subfig.add_subplot(gs[:2, 0], sharex=ax_prev)
        ax_res = subfig.add_subplot(gs[2, 0], sharex=ax)
        ax_prev = ax
        channels: Optional[DataArray] = getattr(data, f"{diagnostic}_channels", None)
        impact_parameter: DataArray = getattr(data, f"{diagnostic}_impact_parameter")
        error = getattr(data, f"{diagnostic}_error")
        if "wavelength" in error.coords:
            error = error.mean("wavelength", skipna=True)
        measurement: DataArray = getattr(data, f"{diagnostic}_measurement")
        if "wavelength" in measurement.coords:
            measurement = measurement.mean("wavelength", skipna=True)
        if np.all((error == 0.0) | error.isnull()):
            error = xr.zeros_like(measurement) + float(0.05 * measurement.mean())
        measurement = (
            measurement.assign_coords(
                {
                    "error": ("channel", error.data),
                    "impact_parameter": (
                        "channel",
                        impact_parameter.sel(channel=measurement.channel).data,
                    ),
                }
            )
            .swap_dims({"channel": "impact_parameter"})
            .dropna("impact_parameter")
        )
        _measurement = measurement.sortby("impact_parameter")
        if channels is not None:
            ax.plot(
                _measurement.impact_parameter,
                _measurement.values,
                marker="+",
                alpha=0.25,
                color="grey",
            )
            _measurement = _measurement.sel(
                impact_parameter=impact_parameter.sel(channel=channels).data
            )
        ax.errorbar(
            x=_measurement.impact_parameter,
            y=_measurement.values,
            yerr=_measurement.error,
            marker="+"
            if np.all((_measurement.error == 0.0) | _measurement.error.isnull())
            else "_",
            linestyle=" ",
            capsize=0.0
            if np.all((_measurement.error == 0.0) | _measurement.error.isnull())
            else 5.0,
            label="Measurement",
        )
        model: DataArray = getattr(data, f"{diagnostic}_model")
        if "wavelength" in model.coords:
            model = model.mean("wavelength", skipna=True)
        model = (
            model.assign_coords(
                {
                    "impact_parameter": (
                        "channel",
                        impact_parameter.sel(channel=model.channel).data,
                    )
                }
            )
            .swap_dims({"channel": "impact_parameter"})
            .sortby("impact_parameter")
        )
        model.sortby("impact_parameter").plot.line(
            x="impact_parameter",
            marker="x",
            linestyle="--",
            label="Model",
            ax=ax,
        )
        for channel, x, y in zip(
            _measurement.sortby("impact_parameter").channel,
            _measurement.sortby("impact_parameter").impact_parameter,
            _measurement.sortby("impact_parameter").data,
        ):
            _annotate(ax, int(channel) + 1, x, y, (0, -30))
        _min = min(_measurement.min(), model.min())
        _max = max(_measurement.max(), model.max())
        ax.set_ylim(_min - (0.1 * _max), _max + (0.1 * _max))
        ax.set_xlabel(None)
        ax.set_title(
            diagnostic.split("_")[0].upper()
            + f" $\\chi^2$: {np.sum(np.array(residual.get(diagnostic, 0), ndmin=1) ** 2):.3g}"
            + f" (weight: {weight:.2f}, scale: {rescale_factor:.2f})"
        )
        ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
        ax.legend()
        if "zeff" in diagnostic.lower():
            ax.set_ylim(bottom=1.0)
        (
            xr.ones_like(_measurement.dropna("impact_parameter"))
            * residual.get(diagnostic, 0)
        ).sortby("impact_parameter").plot.line(
            x="impact_parameter",
            marker="x",
            linestyle=" ",
            color="black",
            ax=ax_res,
        )
        ax_res.axhline(0.0, color="grey", linestyle="--", alpha=0.5)
        ax_res.set_xlabel(r"$\rho_{impact,pol}$")
        ax_res.set_ylabel(r"$\chi^{2}$")
        ax_res.set_title(None)
        ax_res.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)

    fig.suptitle(_standard_title(data))
    return fig


def _plot_diagnostic_contribution(
    data: Dataset, diagnostic: str, fig: Optional[Figure] = None
) -> Figure:
    if "t" in data.dims:
        data = data.isel(t=0)
    diagnostics = [
        val
        for val in (
            [data.diagnostics]
            if isinstance(data.diagnostics, str)
            else data.diagnostics
        )
        if hasattr(data, f"{val}_origin") and hasattr(data, f"{val}_direction")
    ]
    if fig is None:
        fig, nrows, ncols = _make_figure(diagnostics)
    subfigs = np.array(fig.subfigures(nrows, ncols), ndmin=1)
    ax_prev = None
    for subfig, weight, rescale_factor, diagnostic in zip(
        subfigs.flatten(),
        data.weights.data,
        data.rescale_factors.data,
        data.rescale_factors.diagnostic.data,
    ):
        gs = subfig.add_gridspec(3, 1)
        ax = subfig.add_subplot(gs[:2, 0], sharex=ax_prev)
        ax_res = subfig.add_subplot(gs[2, 0], sharex=ax)
        ax_prev = ax
        channels: Optional[DataArray] = getattr(data, f"{diagnostic}_channels", None)
        impact_parameter: DataArray = getattr(data, f"{diagnostic}_impact_parameter")
        error = getattr(data, f"{diagnostic}_error")
        if "wavelength" in error.coords:
            error = error.mean("wavelength", skipna=True)
        measurement: DataArray = getattr(data, f"{diagnostic}_measurement")
        if "wavelength" in measurement.coords:
            measurement = measurement.mean("wavelength", skipna=True)
        if np.all((error == 0.0) | error.isnull()):
            error = xr.zeros_like(measurement) + float(0.05 * measurement.mean())
        measurement = (
            measurement.assign_coords(
                {
                    "error": ("channel", error.data),
                    "impact_parameter": (
                        "channel",
                        impact_parameter.sel(channel=measurement.channel).data,
                    ),
                }
            )
            .swap_dims({"channel": "impact_parameter"})
            .dropna("impact_parameter")
        )
        _measurement = measurement.sortby("impact_parameter")
        if channels is not None:
            ax.plot(
                _measurement.impact_parameter,
                _measurement.values,
                marker="+",
                alpha=0.2,
            )
            _measurement = _measurement.sel(
                impact_parameter=impact_parameter.sel(channel=channels).data
            )
        ax.errorbar(
            x=_measurement.impact_parameter,
            y=_measurement.values,
            yerr=_measurement.error,
            marker="+"
            if np.all((_measurement.error == 0.0) | _measurement.error.isnull())
            else "_",
            linestyle=" ",
            capsize=0.0
            if np.all((_measurement.error == 0.0) | _measurement.error.isnull())
            else 5.0,
            label="Measurement",
        )
        model: DataArray = getattr(data, f"{diagnostic}_model")
        if "wavelength" in model.coords:
            model = model.mean("wavelength", skipna=True)
        model = (
            model.assign_coords(
                {
                    "impact_parameter": (
                        "channel",
                        impact_parameter.sel(channel=model.channel).data,
                    )
                }
            )
            .swap_dims({"channel": "impact_parameter"})
            .sortby("impact_parameter")
        )
        model.sortby("impact_parameter").plot.line(
            x="impact_parameter",
            marker="x",
            linestyle="--",
            label="Model",
            ax=ax,
        )
        for channel, x, y in zip(
            _measurement.sortby("impact_parameter").channel,
            _measurement.sortby("impact_parameter").impact_parameter,
            _measurement.sortby("impact_parameter").data,
        ):
            _annotate(ax, int(channel) + 1, x, y, (0, -30))
        ax.set_xlabel(None)
        ax.set_title(
            diagnostic.split("_")[0].upper()
            + f" $\\chi^2$: {np.sum(np.array(residual.get(diagnostic, 0), ndmin=1) ** 2):.3g}"
            + f" (weight: {weight:.2f}, scale: {rescale_factor:.2f})"
        )
        ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
        ax.legend()
        if "zeff" in diagnostic.lower():
            ax.set_ylim(bottom=1.0)
        (
            xr.ones_like(_measurement.dropna("impact_parameter"))
            * residual.get(diagnostic, 0)
        ).sortby("impact_parameter").plot.line(
            x="impact_parameter",
            marker="x",
            linestyle=" ",
            color="black",
            ax=ax_res,
        )
        ax_res.axhline(0.0, color="grey", linestyle="--", alpha=0.5)
        ax_res.set_xlabel(r"$\rho_{impact,pol}$")
        ax_res.set_ylabel(r"$\chi^{2}$")
        ax_res.set_title(None)
        ax_res.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    return fig


def _plot_diagnostic_contributions(data: Dataset) -> list[Figure]:
    return [
        _plot_diagnostic_contribution(data=data, diagnostic=diagnostic)
        for diagnostic in [
            val
            for val in (
                [data.diagnostics]
                if isinstance(data.diagnostics, str)
                else data.diagnostics
            )
            if hasattr(data, f"{val}_origin") and hasattr(data, f"{val}_direction")
        ]
    ]


def _plot_channel_contribution(
    data: Dataset, diagnostic: str, fig: Optional[Figure] = None
) -> Figure:
    if "t" in data.dims:
        data = data.isel(t=0)
    if fig is None:
        fig = plt.figure(figsize=(16, 10), layout="constrained")
    ncols = 3
    axes = fig.subplots(
        math.ceil((len(data.impurities) + 2) / ncols),
        ncols,
        sharey=False,
        sharex=False,
    )
    wrho: DataArray = getattr(data, f"{diagnostic}_weighted_rho")
    wrho["channel"] = wrho["channel"] + 1
    wrho.dropna("channel").plot(
        x="channel",
        hue="element",
        marker="x",
        linestyle="--",
        ax=axes.flatten()[0],
    )
    axes.flatten()[0].set_title(None)
    axes.flatten()[0].set_ylabel(r"$\langle\rho\rangle$")
    axes.flatten()[0].ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    dwrho: DataArray = getattr(data, f"{diagnostic}_delta_weighted_rho")
    dwrho["channel"] = dwrho["channel"] + 1
    dwrho.dropna("channel").plot(
        x="channel",
        hue="element",
        marker="x",
        linestyle="--",
        ax=axes.flatten()[1],
        add_legend=False,
    )
    axes.flatten()[1].set_title(None)
    axes.flatten()[1].set_ylabel(r"$\delta\langle\rho\rangle$")
    axes.flatten()[1].ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    emissivity: DataArray = getattr(data, f"{diagnostic}_emissivity").assign_coords(
        {"rhop": data.rho_2d}
    )
    rho_los: DataArray = getattr(data, f"{diagnostic}_rho_line_of_sight")
    emissivity = (
        emissivity.where(emissivity.R >= data.rmag, drop=True)
        .interp(z=data.zmag)
        .swap_dims({"R": "rhop"})
        .drop_vars("R")
    )
    channels = getattr(
        data,
        f"{diagnostic}_channels",
        getattr(data, f"{diagnostic}_measurement").dropna("channel").channel.data,
    )
    for ax, element in zip(axes.flatten()[2:], data.impurities):
        for channel, colour in zip(
            channels, mpl.colormaps["plasma"](np.linspace(0.0, 0.8, len(channels)))
        ):
            counts, bins = np.histogram(
                rho_los.where(rho_los > 0, drop=True).sel(channel=channel),
                bins=np.arange(0, 1, 0.01),
            )
            ax.stairs(
                (counts * emissivity.sel(element=element).interp(rhop=bins[1:])),
                bins,
                color=colour,
                alpha=0.9,
                label=f"Channel: {int(channel) + 1}",
            )
        ax.set_title(f"{str(element.capitalize())} weighted $\\rho$")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel("Count")
        ax.legend()
        ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
        ax.set_xlim(0.0, 1.0)

    fig.suptitle(diagnostic.split("_")[0].upper() + ": " + _standard_title(data))
    return fig


def _plot_channel_contributions(data: Dataset) -> list[Figure]:
    return [
        _plot_channel_contribution(data=data, diagnostic=diagnostic)
        for diagnostic in [
            diag
            for diag in (
                [data.diagnostics]
                if isinstance(data.diagnostics, str)
                else data.diagnostics
            )
            if any(
                [diag in val for val in data.data_vars.keys() if "weighted_rho" in val]
            )
        ]
    ]


kinds: Dict[str, Callable[..., Figure]] = {
    "profiles": _plot_profiles,
    "radiation": _plot_radiation,
    "history": _plot_history,
    "diagnostics": _plot_diagnostics,
    "channels": _plot_channel_contributions,
}


def main(
    savefile: Union[str, Path],
    kind: Union[str, List[str]] = "all",
    outpath: Optional[Union[str, Path]] = None,
) -> None:
    if kind == "all":
        kind = list(kinds.keys())
    if isinstance(kind, str):
        kind = [kind]
    assert all([k in kinds.keys() for k in kind])
    savefile = Path(savefile).expanduser().resolve()
    assert savefile.exists()
    data = xr.load_dataset(savefile)
    times: NDArray = data.t.values
    if times.size == 1:
        times = np.expand_dims(times, 0)
    plt.close("all")
    for t in times:
        for k in kind:
            figures: Union[Figure, list[Figure]] = kinds[k](
                data.sel(t=t) if "t" in data.dims else data
            )
            if outpath is not None:
                Path(outpath).mkdir(exist_ok=True)
                if isinstance(figures, Figure):
                    figures = [figures]
                for i, fig in enumerate(figures):
                    fig.savefig(Path(outpath) / f"{k}_{i}_{float(t):.3f}.png")
        if outpath is None:
            plt.show()
        plt.close("all")
