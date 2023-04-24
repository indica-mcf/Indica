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
from indica.readers.read_gacode import get_gacode_data
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


def example_run(
    pulse: int = 9850,
    pulse_code: int = 13109850,
    code: str = "astra",
    equil: str = "efit",
    run_code: str = "RUN61",
    tstart: float = 0.05,
    tend: float = 0.12,
    dt: float = 0.01,
    time: float = 0.11,
    add_gacode: bool = True,
    verbose: bool = True,
):
    """
    Compare ASTRA equilibrium vs EFIT, check Plasma class is correctly set up

    Tests using fixed-boundary predictive ASTRA:
        pulse = 10009
        equil = "astra"
        code = "astra"
        run_code = "2621"  # "2721"

    interpretative ASTRA using HDA/EFIT:
        pulse = 10009
        equil = "efit"
        code = "astra"
        run_code = "573"

    GaCODE + ASTRA interpretative using HDA/EFIT:
        pulse = 9850
        pulse_code = 13109850
        run_code = 61
        code = "astra"
        equil = "efit"
        add_gacode = True
    """
    plasma: Plasma
    code_data: dict

    calc_spectra = True
    moment_analysis = False
    calibration = 0.2e-16
    linewidth = 2
    instruments = ["smmh1", "nirh1", "xrcs", "sxr_diode_1", "efit"]

    # Read code data and assign to plasma
    st40 = ReadST40(pulse_code, tstart, tend, dt=dt, tree=code)
    st40.get_raw_data("", code, run_code)
    st40.bin_data_in_time([code], tstart, tend, dt)

    data_code = st40.binned_data[code]

    plasma = plasma_code(pulse_code, tstart, tend, dt, data_code, verbose=verbose)

    # Read experimental data
    if verbose:
        print(f"Reading ST40 data for pulse={pulse} t=[{tstart}, {tend}]")
    st40 = ReadST40(pulse, tstart, tend, dt=dt, tree="st40")
    st40(instruments=instruments, map_diagnostics=False)

    # Initialize Equilibria
    if equil == code:
        equilibrium = Equilibrium(data_code)
    else:
        equilibrium = Equilibrium(st40.binned_data[equil])

    if add_gacode:
        if pulse_code == 13109850 and run_code == "RUN61" and time == 0.11:
            filename: str = (
                "/home/marco.sertoli/python/Indica/indica/data/input.gacode.new"
            )
        else:
            raise ValueError

        if verbose:
            print(f"Reading GA-code data corresponding to ASTRA run")
        data_ga = get_gacode_data(filename)
        add_gacode_data(plasma, equilibrium, data_ga, time)

    plasma.set_equilibrium(equilibrium)

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
    wth = plasma.wth
    wp = plasma.wp

    raw_color = "black"
    binned_color = "blue"
    bckc_color = "red"

    # Example plots
    plt.close("all")
    plt.figure()
    levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    equilibrium.rho.sel(t=time, method="nearest").plot.contour(levels=levels)
    plt.axis("scaled")
    plt.title("Equilibrium")

    plt.figure()
    plasma.electron_density.sel(t=time, method="nearest").plot(label="electrons")
    ion_density.sel(element=plasma.main_ion).sel(t=time, method="nearest").plot(
        label="main ion"
    )
    fast_density.sel(t=time, method="nearest").plot(label="fast ions")
    plt.title("Electron/Ion densities")
    plt.legend()

    plt.figure()
    plasma.electron_temperature.sel(t=time, method="nearest").plot(label="electrons")
    plasma.ion_temperature.sel(element=plasma.main_ion).sel(
        t=time, method="nearest"
    ).plot(label="ion")
    plt.title("Electron/Ion temperatures")
    plt.legend()

    plt.figure()
    plasma.pressure_fast.sel(t=time, method="nearest").plot(label="Pfast")
    pressure_th.sel(t=time, method="nearest").plot(label="Pth")
    pressure_tot.sel(t=time, method="nearest").plot(label="Ptot")
    plt.title("Pressure")
    plt.legend()

    plt.figure()
    for element in plasma.impurities:
        impurity_conc.sel(element=element).sel(t=time, method="nearest").plot(
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

    if "spectra" in bckc["xrcs"].keys():
        plt.figure()
        _raw = raw["xrcs"]["spectra"].sel(t=time, method="nearest")
        _binned = binned["xrcs"]["spectra"].sel(t=time, method="nearest")
        _bckc = bckc["xrcs"]["spectra"].sel(t=time, method="nearest")
        (_raw / _raw.max()).plot(color=raw_color, label="Raw")
        (_binned / _binned.max()).plot(color=binned_color, label="Binned")
        (_bckc / _bckc.max()).plot(color=bckc_color, label="Model")
        plt.xlim(_bckc.wavelength.min(), _bckc.wavelength.max())
        plt.title(f"XRCS spectra at {time:.3f} s")
        plt.legend()

    return raw, binned, bckc, models, plasma


if __name__ == "__main__":
    example_run()
