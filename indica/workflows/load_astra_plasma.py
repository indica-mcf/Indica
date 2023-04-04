import matplotlib.pylab as plt
import numpy as np
import xarray as xr

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
from indica.readers import ST40Reader
from indica.readers.read_st40 import ReadST40

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

plt.ion()


def astra_plasma(astra: dict, pulse: int, tstart: float, tend: float, dt: float):
    """
    Assign ASTRA to new Plasma class

    Parameters
    ----------
    astra
        dictionary of astra data from AbstractReader
    pulse
        Pulse number identifier

    Returns
    -------
        Plasma class with ASTRA data

    TODO: missing info in ASTRA tree
        - Impurity and main ion element identifiers (what element are NIZ1-3?)
        - parallel and perpendicular fast particle pressures (PBLON and PBPER)

    """

    n_rad = len(astra["ne"].rho_poloidal)
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

    Te = astra["te"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e3
    plasma.electron_temperature.values = Te.values

    Ne = astra["ne"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
    plasma.electron_density.values = Ne.values

    Ti = astra["ti"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e3
    for element in plasma.elements:
        plasma.ion_temperature.loc[dict(element=element)] = Ti.values
    for i, impurity in enumerate(plasma.impurities):
        Nimp = astra[f"niz{i+1}"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
        plasma.impurity_density.loc[dict(element=impurity)] = Nimp.values

    Nf = astra["nf"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
    plasma.fast_density.values = Nf.values

    Nn = astra["nn"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
    plasma.neutral_density.values = Nn.values

    Pblon = astra["pblon"].interp(rho_poloidal=plasma.rho, t=plasma.t)
    plasma.pressure_fast_parallel.values = Pblon.values

    Pbper = astra["pbper"].interp(rho_poloidal=plasma.rho, t=plasma.t)
    plasma.pressure_fast_perpendicular.values = Pbper.values

    plasma.build_atomic_data(default=True)

    return plasma


def initialize_diagnostic_models(diagnostic_data: dict, plasma: Plasma = None):
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
            if type(transform) is LineOfSightTransform:
                models[instrument].set_los_transform(transform)
            elif type(transform) is TransectCoordinates:
                models[instrument].set_transect_transform(transform)
            else:
                raise ValueError("Transform not recognized...")

            if plasma is not None:
                models[instrument].set_plasma(plasma)

    return models


def example_run(verbose: bool = True):
    """
    Compare ASTRA equilibrium vs EFIT, check Plasma class is correctly set up
    """
    pulse = 10009
    pulse_astra = int(pulse + 13.1e6)
    tstart = 0.02
    tend = 0.08
    dt = 0.005
    tplot = 0.04

    equil = "astra"
    astra_run = "2621"  # "2721"

    equil = "efit"
    # astra_run = "573"

    calc_spectra = True
    moment_analysis = True
    calibration = 0.2e-16
    instruements = ["smmh1", "nirh1", "xrcs", "sxr_diode_1"]
    linewidth = 2

    # Read experimental data
    if verbose:
        print("Reading ST40 data")
    st40 = ReadST40(pulse, tstart, tend, dt=dt, tree="st40")
    st40(instruments=instruements, map_diagnostics=False)

    # Read ASTRA simulation
    if verbose:
        print("Reading ASTRA data")
    reader = ST40Reader(pulse_astra, tstart - dt, tend + dt, tree="ASTRA")
    astra = reader.get("", "astra", astra_run)

    if verbose:
        print("Initializing Plasma class with ASTRA data")
    plasma = astra_plasma(astra, pulse, tstart, tend, dt)

    # Initialize Equilibria
    equilibrium_astra = Equilibrium(astra)
    equilibrium_efit = Equilibrium(st40.raw_data["efit"])
    if equil == "astra":
        equilibrium = equilibrium_astra
    elif equil == "efit":
        equilibrium = equilibrium_efit
    else:
        raise ValueError(f"..{equil} equilibrium not available..")
    plasma.set_equilibrium(equilibrium)

    # Load and run the diagnostic forward models
    raw = st40.raw_data
    binned = st40.binned_data
    bckc: dict = {}
    models = initialize_diagnostic_models(
        binned,
        plasma=plasma,
    )
    for instrument in models.keys():
        if verbose:
            print(f"Running {instrument} model")
        if instrument == "xrcs":
            models[instrument].calibration = calibration
            bckc[instrument] = models[instrument](
                calc_spectra=calc_spectra,
                moment_analysis=moment_analysis,
            )
        else:
            bckc[instrument] = models[instrument]()

    # return raw, binned, bckc, models

    # Plot
    pressure_tot = plasma.pressure_tot
    pressure_th = plasma.pressure_th
    ion_density = plasma.ion_density
    fast_density = plasma.fast_density
    impurity_conc = ion_density / plasma.electron_density
    # wth = plasma.wth
    # wp = plasma.wp

    raw_color = "black"
    binned_color = "blue"
    bckc_color = "red"

    # Example plots
    plt.close("all")
    plt.figure()
    levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    equilibrium_efit.rho.sel(t=tplot, method="nearest").plot.contour(levels=levels)
    plasma.equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
        levels=levels, linestyles="dashed"
    )
    plt.axis("scaled")
    plt.title("Equilibrium: EFIT=solid, MODEL=dashed")

    plt.figure()
    plasma.electron_density.sel(t=tplot, method="nearest").plot(label="electrons")
    ion_density.sel(element=plasma.main_ion).sel(t=tplot, method="nearest").plot(
        label="main ion"
    )
    fast_density.sel(t=tplot, method="nearest").plot(label="fast ions")
    plt.title("Electron/Ion densities")
    plt.legend()

    plt.figure()
    plasma.electron_temperature.sel(t=tplot, method="nearest").plot(label="electrons")
    plasma.ion_temperature.sel(element=plasma.main_ion).sel(
        t=tplot, method="nearest"
    ).plot(label="ion")
    plt.title("Electron/Ion temperatures")
    plt.legend()

    plt.figure()
    plasma.pressure_fast.sel(t=tplot, method="nearest").plot(label="Pfast")
    pressure_th.sel(t=tplot, method="nearest").plot(label="Pth")
    pressure_tot.sel(t=tplot, method="nearest").plot(label="Ptot")
    plt.title("Pressure")
    plt.legend()

    plt.figure()
    for element in plasma.impurities:
        impurity_conc.sel(element=element).sel(t=tplot, method="nearest").plot(
            label=element
        )
    plt.title("Impurity concentrations")
    plt.ylabel("(%)")
    plt.yscale("log")
    plt.legend()

    # Plot time evolution of raw data vs back-calculated values
    scaling = {}
    scaling["brems"] = {"brightness": 0.07}
    scaling["sxr_diode_1"] = {"brightness": 30}
    bckc["efit"] = {"wp": plasma.wp}
    # scaling["xrcs"] = {"int_w": 2.e-17}
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
            if instrument in scaling.keys():
                if quantity in scaling[instrument].keys():
                    mult = scaling[instrument][quantity]

            (_bckc * mult).plot(label="Model", color=bckc_color, linewidth=linewidth)
            plt.title(f"{instrument} {quantity}")
            plt.legend()

    return raw, binned, bckc, models, astra


if __name__ == "__main__":
    example_run()
