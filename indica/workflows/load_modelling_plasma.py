from copy import deepcopy
from typing import Dict

from matplotlib import rcParams
import matplotlib.pylab as plt
from MDSplus.mdsExceptions import TreeNODATA
import numpy as np
import xarray as xr

from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters.transect import TransectCoordinates
from indica.equilibrium import Equilibrium
from indica.models.bolometer_camera import Bolometer
from indica.models.charge_exchange import ChargeExchange
from indica.models.diode_filters import BremsstrahlungDiode
from indica.models.helike_spectroscopy import HelikeSpectrometer
from indica.models.interferometry import Interferometry
from indica.models.plasma import Plasma
from indica.models.sxr_camera import SXRcamera
from indica.models.thomson_scattering import ThomsonScattering
from indica.readers.read_st40 import ReadST40
from indica.utilities import FIG_PATH
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.utilities import set_plot_rcparams

# from indica.readers.read_gacode import get_gacode_data

CMAP, COLORS = set_plot_colors()

DIAGNOSTIC_MODELS = {
    "smmh": Interferometry,
    "smmh1": Interferometry,
    "nirh1": Interferometry,
    "xrcs": HelikeSpectrometer,
    "ts": ThomsonScattering,
    "cxff_pi": ChargeExchange,
    "cxff_tws_c": ChargeExchange,
    "cxqf_tws_c": ChargeExchange,
    "brems": BremsstrahlungDiode,
    "sxrc_xy1": Bolometer,
    "sxrc_xy2": SXRcamera,
    "sxr_spd": SXRcamera,
    "blom_xy1": Bolometer,
}
INSTRUMENTS: list = [
    "smmh",
    "smmh1",
    "nirh1",
    "xrcs",
    "blom_xy1",
    "sxrc_xy1",
    "sxrc_xy2",
    "efit",
    "sxr_spd",
]
REVISIONS: dict = {instr: 0 for instr in INSTRUMENTS}
plt.ion()


def plasma_code(
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    data: dict,
    equilibrium: dict = None,
    verbose: bool = False,
) -> Dict[str, Plasma]:
    """
    Assign code data to new Plasma class

    Parameters
    ----------
    pulse
        MDS+ pulse number to read from
    revision
        Tree revision/run number
    tstart, tend, dt
        Times axis of the new plasma object

    Returns
    -------
        Plasma class with code data and the data

    """
    plasma: Dict[str, Plasma] = {}
    runs = list(data.keys())

    print("Initializing Plasma class and building default atomic data")
    n_rad = len(data[runs[0]]["ne"].rho_poloidal)
    main_ion = "h"
    impurities = ("ar", "c", "he")
    impurity_concentration = (0.001, 0.03, 0.01)
    _plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
        impurities=impurities,
        main_ion=main_ion,
        impurity_concentration=impurity_concentration,
        full_run=False,
        n_rad=n_rad,
    )
    _plasma.build_atomic_data()
    for run in runs:
        plasma[run] = deepcopy(_plasma)

    for run in runs:
        _plasma = plasma[run]
        if equilibrium is not None:
            _plasma.set_equilibrium(equilibrium[run])

        _data = data[run]
        Te = _data["te"].interp(rho_poloidal=_plasma.rho, t=_plasma.t) * 1.0e3
        _plasma.electron_temperature.values = Te.values

        Ne = _data["ne"].interp(rho_poloidal=_plasma.rho, t=_plasma.t) * 1.0e19
        _plasma.electron_density.values = Ne.values

        Ti = _data["ti"].interp(rho_poloidal=_plasma.rho, t=_plasma.t) * 1.0e3
        for element in _plasma.elements:
            _plasma.ion_temperature.loc[dict(element=element)] = Ti.values
        for i, impurity in enumerate(_plasma.impurities):
            Nimp = (
                _data[f"niz{i+1}"].interp(rho_poloidal=_plasma.rho, t=_plasma.t)
                * 1.0e19
            )
            _plasma.impurity_density.loc[dict(element=impurity)] = Nimp.values

        Nf = _data["nf"].interp(rho_poloidal=_plasma.rho, t=_plasma.t) * 1.0e19
        _plasma.fast_density.values = Nf.values

        Nn = _data["nn"].interp(rho_poloidal=_plasma.rho, t=_plasma.t) * 1.0e19
        _plasma.neutral_density.values = Nn.values

        Pblon = _data["pblon"].interp(rho_poloidal=_plasma.rho, t=_plasma.t)
        _plasma.pressure_fast_parallel.values = Pblon.values

        Pbper = _data["pbper"].interp(rho_poloidal=_plasma.rho, t=_plasma.t)
        _plasma.pressure_fast_perpendicular.values = Pbper.values

    return plasma


def read_modelling_runs(
    code: str,
    pulse: int,
    runs: list,
    tstart: float = 0.0,
    tend: float = 0.2,
    dt: float = 0.01,
):
    print(f"Reading {code} data")

    code_raw_data: dict = {}
    code_binned_data: dict = {}
    code_reader = ReadST40(pulse, tstart, tend, dt=dt, tree=code)
    for run in runs:
        try:
            code_raw_data[run] = code_reader.get_raw_data("", code, run)
            code_reader.bin_data_in_time([code], tstart, tend, dt)
            code_binned_data[run] = code_reader.binned_data[code]
        except TreeNODATA:
            print(f"   no data for {run}")

    data = code_binned_data[run]
    for quantity in data.keys():
        if "t" in data[quantity].dims:
            tstart = np.min(data[quantity].t.values)
            tend = np.max(data[quantity].t.values)
            break

    return code_raw_data, code_binned_data, tstart, tend


def initialize_diagnostic_models(
    diagnostic_data: dict,
):
    """
    Initialize diagnostic models

    Parameters
    ----------
    data
        Dictionary of data with instrument names as keys

    Returns
    -------
    Dictionary of models with
    """
    models: dict = {}
    for instrument, data in diagnostic_data.items():
        if instrument in DIAGNOSTIC_MODELS.keys():
            models[instrument] = DIAGNOSTIC_MODELS[instrument](instrument)

            transform = data[list(data)[0]].transform
            models[instrument].set_transform(transform)

    if "xrcs" in models.keys():
        models["xrcs"].calibration = 0.2e-16

    return models


def example_run(
    dt: float = 0.01,
    verbose: bool = True,
    example: str = "alsu_11314",
    all_runs: bool = False,
    plot: bool = True,
    save_fig: bool = False,
    fig_style="profiles",
    alpha: float = 1.0,
):
    """
    Run all diagnostic models using profiles and equilibrium from ASTRA modelling
    """

    (
        pulse_code,
        pulse,
        equil,
        equil_run,
        code,
        runs,
        comment,
        tstart,
        tend,
        tplot,
    ) = example_params(example, all_runs=all_runs)

    code_raw_data, code_binned_data, tstart, tend = read_modelling_runs(
        code,
        pulse_code,
        runs,
        tstart=tstart,
        tend=tend,
        dt=dt,
    )
    runs = list(code_raw_data.keys())

    if pulse is not None:
        print("Reading ST40 data")
        st40 = ReadST40(pulse, tstart, tend, dt=dt, tree="st40")
        REVISIONS["efit"] = equil_run
        st40(instruments=INSTRUMENTS, map_diagnostics=False, revisions=REVISIONS)

    if equil != code:
        equilibrium = {run: Equilibrium(st40.raw_data[equil]) for run in runs}
    else:
        equilibrium = {run: Equilibrium(code_raw_data[run]) for run in runs}

    plasma = plasma_code(
        pulse_code,
        tstart,
        tend,
        dt,
        code_raw_data,
        verbose=verbose,
        equilibrium=equilibrium,
    )

    if example == "ga_code":
        raise NotImplementedError
        # if verbose:
        #     print("Reading GA-code data corresponding to ASTRA run")
        # filename = "/home/marco.sertoli/python/Indica/indica/data/input.gacode.new"
        # data_ga = get_gacode_data(filename)
        # add_gacode_data(plasma, equilibrium, data_ga, tplot)

    print("Initializing diagnostic models")
    models = initialize_diagnostic_models(st40.binned_data)

    bckc: dict = {}
    print("Running diagnostic models")
    for run in runs:
        print(run)
        bckc[run] = {}
        _plasma = plasma[run]
        for instrument in models.keys():
            if hasattr(models[instrument], "los_transform"):
                models[instrument].los_transform.set_equilibrium(
                    equilibrium[run], force=True
                )
            if hasattr(models[instrument], "transect_transform"):
                models[instrument].transect_transform.set_equilibrium(
                    equilibrium[run], force=True
                )

            models[instrument].set_plasma(_plasma)

            if instrument == "xrcs":
                _bckc = models[instrument](moment_analysis=True)
            else:
                _bckc = models[instrument]()
            bckc[run][instrument] = deepcopy(_bckc)

    if plot or save_fig:
        plot_plasmas(
            plasma,
            tplot,
            code=code,
            save_fig=save_fig,
            fig_style=fig_style,
            alpha=alpha,
        )

        plot_data_bckc_comparison(
            st40,
            bckc,
            plasma,
            tplot,
            code=code,
            save_fig=save_fig,
            fig_style=fig_style,
            alpha=alpha,
        )

        # for instrument in models.keys():
        #     models[instrument].los_transform.plot_los(
        #         t=tplot, save_fig=save_fig, fig_path=fig_path
        #     )

    return st40, bckc, plasma


def plot_plasma_quantity(
    plasma: Plasma,
    quantity: str,
    time: float = None,
    element: str = None,
    alpha: float = 1.0,
    color=None,
    add_label: bool = True,
    label: str = None,
    ylabel: str = None,
):
    if element is None:
        element = plasma.elements[0]

    if not hasattr(plasma, quantity):
        return

    to_plot = getattr(plasma, quantity)
    if time is not None:
        to_plot = to_plot.sel(t=time, method="nearest")
    else:
        to_plot = to_plot.T
    if "element" in to_plot.dims:
        to_plot = to_plot.sel(element=element)
    # if add_label and "datatype" in to_plot.attrs:
    #     label = to_plot.datatype[1]
    # if "datatype" in to_plot.attrs and color is None and len(np.shape(to_plot)) == 1:
    #     color_label = to_plot.datatype[1]
    #     if color_label in COLORS.keys():
    #         color = COLORS[color_label]
    #     ylabel = f"{to_plot.datatype[0]} [{to_plot.units}]"
    #     ylabel = ylabel[0].upper() + ylabel[1:]

    to_plot.plot(
        label=label,
        color=color,
        alpha=alpha,
    )
    if ylabel is not None:
        plt.ylabel(ylabel)


def plot_plasmas(
    plasma: dict,
    time: float,
    code: str = "",
    save_fig: bool = False,
    fig_path: str = None,
    fig_style: str = "profiles",
    alpha: float = 1.0,
):
    xr.set_options(keep_attrs=True)
    runs = list(plasma.keys())
    if fig_path is None:
        run_str = runs[0]
        pulse_code = plasma[runs[0]].pulse
        if len(runs) > 1:
            run_str += f"-{runs[1]}"
        fig_path = f"{FIG_PATH}{pulse_code}_{time}_{code}_{run_str}/"

    # imp_colors = {
    #     "ar": COLORS["fast"],
    #     "c": COLORS["ion"],
    #     "he": COLORS["electron"],
    # }

    set_plot_rcparams(fig_style)

    plt.close("all")

    plt.figure()
    levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for run in runs:
        plasma[run].equilibrium.rho.sel(t=time, method="nearest").plot.contour(
            levels=levels,
            alpha=alpha,
        )
    plt.axis("scaled")
    plt.title(f"t = {time:.3f} s")
    save_figure(fig_path, f"equilibrium_{int(time*1.e3)}ms", save_fig=save_fig)

    plt.figure()
    quantities = ["electron_density", "ion_density", "fast_density"]
    add_label = False
    for run in runs:
        if run == runs[-1]:
            add_label = True
        for quantity in quantities:
            plot_plasma_quantity(
                plasma[run], quantity, time, alpha=alpha, add_label=add_label
            )
    plt.title(f"t = @ {time:.3f} s")
    set_axis_sci()
    plt.legend()
    save_figure(fig_path, f"density_{time:.3f}_s", save_fig=save_fig)

    plt.figure()
    quantities = ["electron_temperature", "ion_temperature"]
    add_label = False
    for run in runs:
        if run == runs[-1]:
            add_label = True
        for quantity in quantities:
            plot_plasma_quantity(
                plasma[run], quantity, time, alpha=alpha, add_label=add_label
            )
    plt.title(f"t = {time:.3f} s")
    set_axis_sci()
    plt.legend()
    save_figure(fig_path, f"temperature_{time:.3f}_s", save_fig=save_fig)

    plt.figure()
    quantities = ["pressure_fast", "pressure_th", "pressure_tot"]
    add_label = False
    for run in runs:
        if run == runs[-1]:
            add_label = True
        for quantity in quantities:
            plot_plasma_quantity(
                plasma[run], quantity, time, alpha=alpha, add_label=add_label
            )
    plt.title(f"t = {time:.3f} s")
    set_axis_sci()
    plt.legend()
    save_figure(fig_path, f"pressure_{time:.3f}_s", save_fig=save_fig)

    if len(runs) == 1:
        run = runs[0]

        quantities = [
            "electron_density",
            "ion_density",
            "fast_density",
            "electron_temperature",
            "ion_temperature",
            "pressure_fast",
            "pressure_th",
            "pressure_tot",
        ]
        for quantity in quantities:
            plt.figure()
            plot_plasma_quantity(
                plasma[run],
                quantity,
            )
            # plt.title(f"")
            # set_axis_sci()
            # plt.legend()
            save_figure(fig_path, f"{quantity}_time_evolution", save_fig=save_fig)

    # plt.figure()
    # impurity_conc = plasma[run].ion_density / plasma[run].electron_density
    #
    # for run in runs:
    #     for element in plasma[run].impurities:
    #         impurity_conc.sel(element=element).sel(t=time, method="nearest").plot(
    #             color=imp_colors[element], label=element,
    #         )
    # plt.title(f"Impurity concentrations @ {time:.3f} s")
    # plt.ylabel("%")
    # plt.yscale("log")
    # plt.legend()
    # save_figure(fig_path, f"Impurity_concentration_{time:.3f}_s", save_fig=save_fig)


def plot_data_bckc_comparison(
    st40: ReadST40,
    bckc: dict,
    plasma: dict,
    time: float,
    code: str = "",
    save_fig: bool = False,
    fig_path: str = None,
    fig_style: str = "profiles",
    alpha: float = 1.0,
):
    set_plot_rcparams(fig_style)
    xr.set_options(keep_attrs=True)
    runs = list(plasma.keys())
    if fig_path is None:
        run_str = runs[0]
        pulse_code = plasma[runs[0]].pulse
        if len(runs) > 1:
            run_str += f"-{runs[1]}"
        fig_path = f"{FIG_PATH}{pulse_code}_{time}_{code}_{run_str}/"

    norm: dict = {}
    norm["xrcs"] = {}
    norm["xrcs"]["spectra"] = True
    # norm["brems"] = True
    # norm["sxr_camera_4"] = True
    # norm["sxrc_xy2"] = {}
    # norm["sxrc_xy2"]["brightness"] = True
    y0 = {}
    y0["nirh1"] = True
    y0["smmh1"] = True
    y0["xrcs"] = True
    y0["sxr_diode_1"] = True
    y0["brems"] = True
    y0["efit"] = True
    binned_marker = "o"

    for run in runs:
        bckc[run]["efit"] = {"wp": plasma[run].wp}

    instruments = st40.raw_data.keys()
    for instrument in instruments:
        quantities = bckc[runs[0]][instrument].keys()
        for quantity in quantities:
            if (
                quantity not in st40.binned_data[instrument].keys()
                or quantity not in st40.raw_data[instrument].keys()
            ):
                continue

            print(instrument)
            print(f"  {quantity}")
            _raw = st40.raw_data[instrument][quantity]
            _binned = st40.binned_data[instrument][quantity]
            _bckc = bckc[runs[0]][instrument][quantity]

            tslice = slice(_bckc.t.min().values, _bckc.t.max().values)
            str_to_add = ""
            tslice_raw = tslice
            tslice_binned = tslice

            if "error" not in _binned.attrs:
                _binned.attrs["error"] = xr.full_like(_binned, 0.0)
            if "stdev" not in _binned.attrs:
                _binned.attrs["stdev"] = xr.full_like(_binned, 0.0)
            err = np.sqrt(_binned.error**2 + _binned.stdev**2)
            err = xr.where(err / _binned.values < 1.0, err, 0.0)

            if len(_binned.dims) > 1:
                str_to_add = f" @ {time:.3f} s"
                tslice_binned = _binned.t.sel(t=time, method="nearest")
                tslice_raw = _raw.t.sel(t=time, method="nearest")

            _raw = _raw.sel(t=tslice_raw)
            _binned = _binned.sel(t=tslice_binned)
            _err = err.sel(t=tslice_binned)
            markersize = deepcopy(rcParams["lines.markersize"])
            if instrument in "xrcs" and quantity == "spectra":
                markersize /= 2
                bgnd = _binned.sel(wavelength=slice(0.393, 0.388)).mean("wavelength")
                _binned -= bgnd
                _raw -= bgnd

            plt.figure()
            _raw.plot(
                label="Raw",
                color=COLORS["raw"],
                linestyle="dashed",
            )
            if "t" in _binned.dims:
                _t = _binned.t.sel(t=tslice_binned)
                plt.fill_between(
                    _t,
                    _binned.values - _err.values,
                    _binned.values + _err.values,
                    color=COLORS["binned"],
                    alpha=0.7,
                )
            _binned.plot(
                label="Binned",
                color=COLORS["binned"],
                marker=binned_marker,
                markersize=markersize,
            )

            label: str = "Model"
            for run in runs:
                _bckc = bckc[run][instrument][quantity].sel(t=tslice_binned)
                if instrument in norm.keys():
                    if quantity in norm[instrument].keys():
                        _bckc = _bckc / _bckc.max() * _binned.max()
                    if label is not None:
                        label += " (scaled)"

                (_bckc).plot(
                    label=label,
                    color=COLORS["bckc"],
                    linewidth=rcParams["lines.linewidth"],
                    alpha=alpha,
                )
                del label

            set_axis_sci()
            plt.title(f"{instrument.upper()} {quantity}" + str_to_add)
            if instrument in y0.keys():
                plt.ylim(
                    0,
                )
            if quantity == "spectra":
                plt.xlim(_bckc.wavelength.min(), _bckc.wavelength.max())

            plt.legend()
            save_figure(fig_path, f"{instrument}_{quantity}", save_fig=save_fig)


def example_params(example: str, all_runs: bool = False):
    runs_all: list = [f"RUN{run}" for run in (500 + np.arange(61, 77))]

    params = {
        "predictive": dict(
            comment="Tests using fixed-boundary predictive ASTRA",
            pulse_code=13110009,
            pulse=10009,
            equil="astra",
            equil_run=0,
            code="astra",
            runs=["RUN2621"],
            tstart=0.02,
            tend=0.08,
            tplot=0.06,
        ),
        "interpretative_10009": dict(
            comment="interpretative ASTRA using HDA/EFIT",
            pulse_code=13110009,
            pulse=10009,
            equil="efit",
            equil_run=0,
            code="astra",
            runs=["RUN564"],
            tstart=0.03,
            tend=0.1,
            tplot=0.06,
        ),
        "interpretative_9850": dict(
            comment="ASTRA interpretative using HDA/EFIT",
            pulse=9850,
            pulse_code=13109850,
            equil="efit",
            equil_run=0,
            code="astra",
            runs=["RUN564"],
            tstart=0.02,
            tend=0.1,
            tplot=0.08,
        ),
        "interpretative_9229": dict(
            comment="ASTRA interpretative using HDA/EFIT",
            pulse=9229,
            pulse_code=13109229,
            equil="efit",
            equil_run=0,
            code="astra",
            runs=["RUN572"],
            tstart=0.03,
            tend=0.11,
            tplot=0.06,
        ),
        "diverted": dict(
            comment="predictive ASTRA using for diverted scenario",
            pulse_code=13000040,
            pulse=10009,
            equil="astra",
            equil_run=0,
            code="astra",
            runs=["RUN292"],
            tstart=0.03,
            tend=0.11,
            tplot=0.1,
        ),
        "michail_10014": dict(
            comment="predictive ASTRA",
            pulse_code=36010014,
            pulse=10014,
            equil="efit",
            equil_run=0,
            code="astra",
            runs=["RUN24"],
            tstart=0.03,
            tend=0.1,
            tplot=0.07,
        ),
        "aleksei_11228": dict(
            comment="ASTRA using TS and invented Ti shapes",
            pulse=11228,
            pulse_code=13011228,
            equil="efit",  # "astra"
            equil_run=1,
            code="astra",
            runs=["RUN610", "RUN611", "RUN612"],
            tstart=0.03,
            tend=0.11,
            tplot=0.08,
        ),
        "alsu_11312": dict(
            comment="ASTRA using TS and peaked Ti scaled to CXRS",
            pulse=11312,
            pulse_code=33011312,
            equil="astra",
            equil_run=0,
            code="astra",
            runs=["RUN21"],
            tstart=0.065,
            tend=0.095,
            tplot=0.075,
        ),
        "alsu_11314": dict(
            comment="ASTRA using TS and peaked Ti scaled to CXRS",
            pulse=11314,
            pulse_code=33011314,
            equil="astra",
            equil_run=0,
            code="astra",
            runs=["RUN12"],
            tstart=0.065,
            tend=0.095,
            tplot=0.075,
        ),
        "alsu_11317": dict(
            comment="ASTRA using TS and peaked Ti scaled to CXRS",
            pulse=11317,
            pulse_code=33011317,
            equil="astra",
            equil_run=0,
            code="astra",
            runs=["RUN9"],
            tstart=0.065,
            tend=0.095,
            tplot=0.075,
        ),
    }

    _params = params[example]
    if all_runs:
        _params["runs"] = runs_all

    return (
        _params["pulse_code"],
        _params["pulse"],
        _params["equil"],
        _params["equil_run"],
        _params["code"],
        _params["runs"],
        _params["comment"],
        _params["tstart"],
        _params["tend"],
        _params["tplot"],
    )


# def add_gacode_data(
#     plasma: Plasma, equilibrium: Equilibrium, data_ga: dict, time: float,
# ):
#     """
#     Assign gacode data to Plasma class foir specified time-point
#     """
#     # Kinetic quantities (only Ne, Te, Ti)
#     t = plasma.t.sel(t=time, method="nearest")
#     Te = np.interp(plasma.rho, data_ga["rho_pol"], data_ga["t_e"] * 1.0e3)
#     plasma.electron_temperature.loc[dict(t=t)] = Te
#
#     Ne = np.interp(plasma.rho, data_ga["rho_pol"], data_ga["n_e"] * 1.0e19)
#     plasma.electron_density.loc[dict(t=t)] = Ne
#
#     Ti = np.interp(plasma.rho, data_ga["rho_pol"], data_ga["t_ion"][0, :] * 1.0e3)
#     for element in plasma.elements:
#         plasma.ion_temperature.loc[dict(element=element, t=t)] = Ti
#
#     # Equilibrium quantities (only rho, Rmag, zmag)
#     t = equilibrium.rho.t.sel(t=t, method="nearest")
#     _rho_ga = DataArray(
#         data_ga["rho_xy"], coords=[("z", data_ga["Z_xy"]), ("R", data_ga["R_xy"])]
#     )
#     rho_ga = _rho_ga.interp(R=equilibrium.rho.R, z=equilibrium.rho.z)
#     rho_ga = xr.where(np.isfinite(rho_ga), rho_ga, 1.4)
#     equilibrium.rho.loc[dict(t=t)] = rho_ga
#
#     zmag = (rho_ga.idxmin("z").dropna("R")).mean().values
#     rmag = rho_ga.idxmin("R").dropna("z").interp(z=zmag).values
#
#     equilibrium.rmag.loc[dict(t=t)] = rmag
#     equilibrium.zmag.loc[dict(t=t)] = zmag


if __name__ == "__main__":
    plt.ioff()
    example_run(example="alsu_11314")
    plt.show()
