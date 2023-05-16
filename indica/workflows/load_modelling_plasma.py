import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters.transect import TransectCoordinates
from indica.equilibrium import Equilibrium
from indica.models.charge_exchange import ChargeExchange
from indica.models.diode_filters import BremsstrahlungDiode
from indica.models.helike_spectroscopy import Helike_spectroscopy
from indica.models.interferometry import Interferometry
from indica.models.plasma import Plasma
from indica.models.sxr_camera import SXRcamera
from indica.models.thomson_scattering import ThomsonScattering
from indica.numpy_typing import RevisionLike
from indica.readers.read_gacode import get_gacode_data
from indica.readers.read_st40 import ReadST40
from indica.utilities import save_figure

DIAGNOSTIC_MODELS = {
    "smmh1": Interferometry,
    "nirh1": Interferometry,
    "xrcs": Helike_spectroscopy,
    "ts": ThomsonScattering,
    "cxff_pi": ChargeExchange,
    "cxff_tws_c": ChargeExchange,
    "cxqf_tws_c": ChargeExchange,
    "brems": BremsstrahlungDiode,
    "sxr_diode_1": SXRcamera,
    "sxr_camera_4": SXRcamera,
}
FIG_PATH = "./results/load_modelling/"
plt.ion()


def plasma_code(
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    data: dict,
    verbose: bool = False,
):
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
    if verbose:
        print("Assigning code data to Plasma class")

    n_rad = len(data["ne"].rho_poloidal)
    main_ion = "h"
    impurities = ("ar", "c", "he")
    impurity_concentration = (0.001, 0.03, 0.01)

    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
        impurities=impurities,
        main_ion=main_ion,
        impurity_concentration=impurity_concentration,
        pulse=pulse,
        full_run=False,
        n_rad=n_rad,
    )

    Te = data["te"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e3
    plasma.electron_temperature.values = Te.values

    Ne = data["ne"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
    plasma.electron_density.values = Ne.values

    Ti = data["ti"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e3
    for element in plasma.elements:
        plasma.ion_temperature.loc[dict(element=element)] = Ti.values
    for i, impurity in enumerate(plasma.impurities):
        Nimp = data[f"niz{i+1}"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
        plasma.impurity_density.loc[dict(element=impurity)] = Nimp.values

    Nf = data["nf"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
    plasma.fast_density.values = Nf.values

    Nn = data["nn"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
    plasma.neutral_density.values = Nn.values

    Pblon = data["pblon"].interp(rho_poloidal=plasma.rho, t=plasma.t)
    plasma.pressure_fast_parallel.values = Pblon.values

    Pbper = data["pbper"].interp(rho_poloidal=plasma.rho, t=plasma.t)
    plasma.pressure_fast_perpendicular.values = Pbper.values

    plasma.build_atomic_data(default=True)

    return plasma


def add_gacode_data(
    plasma: Plasma,
    equilibrium: Equilibrium,
    data_ga: dict,
    time: float,
):
    """
    Assign gacode data to Plasma class foir specified time-point
    """
    # Kinetic quantities (only Ne, Te, Ti)
    t = plasma.t.sel(t=time, method="nearest")
    Te = np.interp(plasma.rho, data_ga["rho_pol"], data_ga["t_e"] * 1.0e3)
    plasma.electron_temperature.loc[dict(t=t)] = Te

    Ne = np.interp(plasma.rho, data_ga["rho_pol"], data_ga["n_e"] * 1.0e19)
    plasma.electron_density.loc[dict(t=t)] = Ne

    Ti = np.interp(plasma.rho, data_ga["rho_pol"], data_ga["t_ion"][0, :] * 1.0e3)
    for element in plasma.elements:
        plasma.ion_temperature.loc[dict(element=element, t=t)] = Ti

    # Equilibrium quantities (only rho, Rmag, zmag)
    t = equilibrium.rho.t.sel(t=t, method="nearest")
    _rho_ga = DataArray(
        data_ga["rho_xy"], coords=[("z", data_ga["Z_xy"]), ("R", data_ga["R_xy"])]
    )
    rho_ga = _rho_ga.interp(R=equilibrium.rho.R, z=equilibrium.rho.z)
    rho_ga = xr.where(np.isfinite(rho_ga), rho_ga, 1.4)
    equilibrium.rho.loc[dict(t=t)] = rho_ga

    zmag = (rho_ga.idxmin("z").dropna("R")).mean().values
    rmag = rho_ga.idxmin("R").dropna("z").interp(z=zmag).values

    equilibrium.rmag.loc[dict(t=t)] = rmag
    equilibrium.zmag.loc[dict(t=t)] = zmag


def initialize_diagnostic_models(
    diagnostic_data: dict, plasma: Plasma = None, equilibrium: Equilibrium = None
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
            if hasattr(transform, "set_equilibrium") and equilibrium is not None:
                transform.set_equilibrium(equilibrium, force=True)

            if type(transform) is LineOfSightTransform:
                models[instrument].set_los_transform(transform)
            elif type(transform) is TransectCoordinates:
                models[instrument].set_transect_transform(transform)
            else:
                raise ValueError("Transform not recognized...")

            if plasma is not None:
                models[instrument].set_plasma(plasma)

    return models


def example_params(example: str):
    comment: str
    pulse_code: int
    pulse: int
    equil: str
    code: str
    run_code: RevisionLike
    if example == "predictive":
        comment = "Tests using fixed-boundary predictive ASTRA"
        pulse_code = 13110009
        pulse = 10009
        equil = "astra"
        code = "astra"
        run_code = "RUN2621"
        tplot = 0.11
    elif example == "interpretative_10009":
        comment = "interpretative ASTRA using HDA/EFIT"
        pulse_code = 13110009
        pulse = 10009
        equil = "efit"
        code = "astra"
        run_code = "RUN564" # "RUN573"
        tplot = 0.08
    elif example == "interpretative_9850":
        comment = "ASTRA interpretative using HDA/EFIT"
        pulse = 9850
        pulse_code = 13109850
        equil = "efit"
        code = "astra"
        run_code = "RUN564" #61
        tplot = 0.08
    elif example == "interpretative_9229":
        comment = "ASTRA interpretative using HDA/EFIT"
        pulse = 9229
        pulse_code = 13109229
        equil = "efit"
        code = "astra"
        run_code = "RUN574" #"RUN567" #"RUN573"
        tplot = 0.06
    elif example == "diverted":
        comment = "predictive ASTRA using for diverted scenario"
        pulse_code = 13000040
        pulse = 10009
        equil = "astra"
        code = "astra"
        run_code = "RUN292"
        tplot = 0.11
    # elif example == "ga_code":
    #     comment = "GaCODE + ASTRA interpretative using HDA/EFIT"
    #     pulse = 9850
    #     pulse_code = 13109850
    #     run_code = 61
    #     code = "astra"
    #     equil = "efit"
    #     tplot = 0.11
    else:
        raise ValueError(f"No parameters for example {example}")

    return pulse_code, pulse, equil, code, run_code, comment, tplot


def example_run(
    tstart: float = 0.05,
    tend: float = 0.10,
    dt: float = 0.01,
    verbose: bool = True,
    tplot: float = None,
    plot: bool = True,
    example: str = "predictive",
    save_fig:bool=False,
):
    """
    Run all diagnostic models using profiles and equilibrium from ASTRA modelling
    """

    plasma: Plasma
    code_data: dict

    pulse_code, pulse, equil, code, run_code, comment, _tplot = example_params(example)
    if tplot is None:
        tplot = _tplot

    instruments = ["smmh1", "nirh1", "xrcs", "sxr_diode_1", "efit", "brems"]

    # Read code data
    st40_code = ReadST40(pulse_code, tstart, tend, dt=dt, tree=code)
    st40_code.get_raw_data("", code, run_code)
    st40_code.bin_data_in_time([code], tstart, tend, dt)

    # Assign code data in plasma class
    data_code = st40_code.binned_data[code]
    for quantity in data_code.keys():
        if "t" in data_code[quantity].dims:
            tstart = np.min(data_code[quantity].t.values)
            tend = np.max(data_code[quantity].t.values)
            break

    plasma = plasma_code(pulse_code, tstart, tend, dt, data_code, verbose=verbose)

    # Read experimental data
    if verbose:
        print(f"Reading ST40 data for pulse={pulse} t=[{tstart}, {tend}]")

    st40 = ReadST40(pulse, tstart, tend, dt=dt, tree="st40")
    st40(instruments=instruments, map_diagnostics=False)
    # Initialize Equilibrium
    equilibrium: Equilibrium
    if equil == code:
        equilibrium = Equilibrium(data_code)
    else:
        equilibrium = Equilibrium(st40.binned_data[equil])

    plasma.set_equilibrium(equilibrium)

    if example == "ga_code":
        if verbose:
            print("Reading GA-code data corresponding to ASTRA run")
        filename: str = "/home/marco.sertoli/python/Indica/indica/data/input.gacode.new"
        data_ga = get_gacode_data(filename)
        add_gacode_data(plasma, equilibrium, data_ga, tplot)

    # Load and run the diagnostic forward models
    raw = st40.raw_data
    binned = st40.binned_data
    bckc: dict = {}
    models = initialize_diagnostic_models(
        binned, plasma=plasma, equilibrium=equilibrium
    )

    for instrument in models.keys():
        if verbose:
            print(f"Running {instrument} model")
        if instrument == "xrcs":
            models[instrument].calibration = 0.2e-16
            bckc[instrument] = models[instrument](
                calc_spectra=True,
                moment_analysis=False,
            )
        else:
            bckc[instrument] = models[instrument]()

    if plot:
        plot_modelling_results(raw, binned, bckc, plasma, tplot, save_fig=save_fig)

    return raw, binned, bckc, models, plasma


def plot_modelling_results(
    raw: dict, binned: dict, bckc: dict, plasma: Plasma, time: float, save_fig:bool=False,
):

    raw_color = "black"
    binned_color = "blue"
    bckc_color = "red"
    linewidth = 2

    pressure_tot = plasma.pressure_tot
    pressure_th = plasma.pressure_th
    ion_density = plasma.ion_density
    fast_density = plasma.fast_density
    impurity_conc = ion_density / plasma.electron_density

    # Example plots
    plt.close("all")

    plt.figure()
    levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    plasma.equilibrium.rho.sel(t=time, method="nearest").plot.contour(levels=levels)
    plt.axis("scaled")
    plt.title(f"Equilibrium @ {int(time*1.e3)} ms")
    if save_fig:
        save_figure(FIG_PATH, f"Equilibrium_{int(time*1.e3)}ms")

    plt.figure()
    plasma.electron_density.sel(t=time, method="nearest").plot(label="electrons")
    ion_density.sel(element=plasma.main_ion).sel(t=time, method="nearest").plot(
        label="main ion"
    )
    fast_density.sel(t=time, method="nearest").plot(label="fast ions")
    plt.title(f"Electron/Ion densities @ {int(time*1.e3)} ms")
    plt.legend()
    if save_fig:
        save_figure(FIG_PATH, f"Electron_and_Ion_densities_{int(time*1.e3)}_ms")

    plt.figure()
    plasma.electron_temperature.sel(t=time, method="nearest").plot(label="electrons")
    plasma.ion_temperature.sel(element=plasma.main_ion).sel(
        t=time, method="nearest"
    ).plot(label="ion")
    plt.title(f"Electron/Ion temperatures @ {int(time*1.e3)} ms")
    plt.legend()
    if save_fig:
        save_figure(FIG_PATH, f"Electron_and_Ion_temperatures_{int(time*1.e3)}_ms")

    plt.figure()
    plasma.pressure_fast.sel(t=time, method="nearest").plot(label="Pfast")
    pressure_th.sel(t=time, method="nearest").plot(label="Pth")
    pressure_tot.sel(t=time, method="nearest").plot(label="Ptot")
    plt.title(f"Pressure @ {int(time*1.e3)} ms")
    plt.legend()
    if save_fig:
        save_figure(FIG_PATH, f"Pressure_{int(time*1.e3)}_ms")

    plt.figure()
    for element in plasma.impurities:
        impurity_conc.sel(element=element).sel(t=time, method="nearest").plot(
            label=element
        )
    plt.title(f"Impurity concentrations @ {int(time*1.e3)} ms")
    plt.ylabel("(%)")
    plt.yscale("log")
    plt.legend()
    if save_fig:
        save_figure(FIG_PATH, f"Impurity_concentration_{int(time*1.e3)}_ms")

    # Plot time evolution of raw data vs back-calculated values
    norm = {}
    norm["brems"] = True
    norm["sxr_diode_1"] = True
    bckc["efit"] = {"wp": plasma.wp}
    for instrument in bckc.keys():
        for quantity in bckc[instrument].keys():
            print(instrument)
            print(f"  {quantity}")
            if (
                quantity not in binned[instrument].keys()
                or quantity not in raw[instrument].keys()
            ):
                continue

            if len(bckc[instrument][quantity].dims) > 1:
                continue

            plt.figure()
            _raw = raw[instrument][quantity]
            _binned = binned[instrument][quantity]
            _bckc = bckc[instrument][quantity]
            tslice = slice(_bckc.t.min().values, _bckc.t.max().values)
            if "error" not in _binned.attrs:
                _binned.attrs["error"] = xr.full_like(_binned, 0.0)
            if "stdev" not in _binned.attrs:
                _binned.attrs["stdev"] = xr.full_like(_binned, 0.0)

            err = np.sqrt(_binned.error**2 + _binned.stdev**2)
            err = xr.where(err / _binned.values < 1.0, err, 0.0)

            plt.fill_between(
                _binned.t.sel(t=tslice),
                _binned.sel(t=tslice).values - err.sel(t=tslice).values,
                _binned.sel(t=tslice).values + err.sel(t=tslice).values,
                color=binned_color,
                alpha=0.7,
            )
            _binned.sel(t=tslice).plot(
                label="Binned",
                color=binned_color,
                marker="o",
            )
            _raw.sel(t=tslice).plot(
                label="Raw",
                color=raw_color,
            )
            mult = 1.0
            label="Model"
            if instrument in norm.keys():
                mult = (_binned.sel(t=tslice).max()/_bckc.sel(t=tslice).max())
                label += " (scaled)"

            (_bckc * mult).plot(label=label, color=bckc_color, linewidth=linewidth)
            plt.title(f"{instrument} {quantity}")
            plt.legend()
            if save_fig:
                save_figure(FIG_PATH, f"{instrument}_{quantity}")

    if "spectra" in bckc["xrcs"].keys():
        plt.figure()
        _raw = raw["xrcs"]["spectra"].sel(t=time, method="nearest")
        _binned = binned["xrcs"]["spectra"].sel(t=time, method="nearest")
        _bckc = bckc["xrcs"]["spectra"].sel(t=time, method="nearest")
        _binned -= _binned.sel(wavelength=slice(0.393, 0.388)).mean("wavelength")
        _raw -= _raw.sel(wavelength=slice(0.393, 0.388)).mean("wavelength")

        (_raw / _raw.max()).plot(color=raw_color, label="Raw")
        (_binned / _binned.max()).plot(color=binned_color, label="Binned")
        (_bckc / _bckc.max()).plot(color=bckc_color, label="Model")
        plt.xlim(_bckc.wavelength.min(), _bckc.wavelength.max())
        plt.title(f"XRCS spectra @ {int(time*1.e3)} ms")
        plt.legend()
        if save_fig:
            save_figure(FIG_PATH, f"XRCS_spectra_{int(time*1.e3)}_ms")


if __name__ == "__main__":
    example_run()
