import matplotlib.pylab as plt
import numpy as np
import xarray as xr

from indica.equilibrium import Equilibrium
from indica.models.plasma import Plasma
from indica.readers.read_st40 import ReadST40
from indica.workflows.load_modelling_plasma import initialize_diagnostic_models

# from indica.converters.line_of_sight import LineOfSightTransform
# from indica.converters.transect import TransectCoordinates
# from indica.models.charge_exchange import ChargeExchange
# from indica.models.diode_filters import BremsstrahlungDiode
# from indica.models.helike_spectroscopy import Helike_spectroscopy
# from indica.models.interferometry import Interferometry
# from indica.models.sxr_camera import SXRcamera
# from indica.models.thomson_scattering import ThomsonScattering
# from indica.readers.read_gacode import get_gacode_data
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
    # impurities = ("ar", "c", "he")
    impurities = ("ar","c")
    # impurity_concentration = (0.001, 0.03, 0.01)
    impurity_concentration = ([0.00001,0.06])
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

    Te = data["te"].interp(rho_poloidal=plasma.rho, t=plasma.t)  # * 1.0e3
    plasma.electron_temperature.values = Te.values

    Ne = data["ne"].interp(rho_poloidal=plasma.rho, t=plasma.t)  # * 1.0e19
    plasma.electron_density.values = Ne.values

    Ti = data["ti"].interp(rho_poloidal=plasma.rho, t=plasma.t)  # * 1.0e3
    normalized_to_c_concentration=1.0 / 0.03 * np.array(impurity_concentration)
    for element in plasma.elements:
        plasma.ion_temperature.loc[dict(element=element)] = Ti.values

    for i, impurity in enumerate(plasma.impurities):
        # todo fix hack to add mutiple impurities
        Nimp = data[f"niz{1}"].interp(
            rho_poloidal=plasma.rho, t=plasma.t
        )  # * 1.0e19 #i+

        Nimp = normalized_to_c_concentration[i]*Nimp
        plasma.impurity_density.loc[dict(element=impurity)] = Nimp.values

    Nf = data["nf"].interp(rho_poloidal=plasma.rho, t=plasma.t)  # * 1.0e19
    plasma.fast_density.values = Nf.values

    Nn = data["nn"].interp(rho_poloidal=plasma.rho, t=plasma.t)  # * 1.0e19
    plasma.neutral_density.values = Nn.values

    Pblon = data["pblon"].interp(rho_poloidal=plasma.rho, t=plasma.t)
    plasma.pressure_fast_parallel.values = Pblon.values

    Pbper = data["pbper"].interp(rho_poloidal=plasma.rho, t=plasma.t)
    plasma.pressure_fast_perpendicular.values = Pbper.values

    plasma.build_atomic_data(default=True)

    return plasma


def plot_modelling_results(
    raw: dict, binned: dict, bckc: dict, plasma: Plasma, time: float
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
    plt.title("Equilibrium")

    plt.figure()
    plasma.electron_density.sel(t=time, method="nearest").plot(label="electrons")
    ion_density.sel(element=plasma.main_ion).sel(t=time, method="nearest").plot(
        label="main ion"
    )
    fast_density.sel(t=time, method="nearest").plot(label="fast ions")

    # ne_c=6*plasma.impurity_density.sel(element='c').sel(t=time, method="nearest")
    # ne_he = 2*plasma.impurity_density.sel(element='he').sel(t=time, method="nearest")
    # ne_ar = 18 * plasma.impurity_density.sel(element='ar').sel(t=time, method="nearest")
    # ne_t = fast_density.sel(t=time, method="nearest") + ion_density.sel(element=plasma.main_ion).sel(t=time, method="nearest")+ne_c+ne_he+ne_ar
    # ne_c.plot(label="carbon e-")
    # ne_he.plot(label="he e-")
    # ne_ar.plot(label="ar e-")
    # ne_t.plot(label="tot e- from ions")

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
    scaling["xrcs"] = {"int_w": 2.e-17}
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
        _binned -= _binned.sel(wavelength=slice(0.393, 0.388)).mean("wavelength")
        _raw -= _raw.sel(wavelength=slice(0.393, 0.388)).mean("wavelength")

        (_raw / _raw.max()).plot(color=raw_color, label="Raw")
        (_binned / _binned.max()).plot(color=binned_color, label="Binned")
        (_bckc / _bckc.max()).plot(color=bckc_color, label="Model")
        plt.xlim(_bckc.wavelength.min(), _bckc.wavelength.max())
        plt.title(f"XRCS spectra at {time:.3f} s")
        plt.legend()


# GaCODE + ASTRA interpretative using HDA/EFIT:
pulse = 10009
pulse_code = 34010009  # 13109850
run_code = "V50"  # 61
code = "transp_test"  # "astra"

# pulse_code1 = 13109850
# run_code1 = 61
# code1 = "astra"

equil = "transp_test"
tstart = 0.02
tend = 0.06
dt = 0.01
verbose = True
instruments = ["smmh1", "nirh1", "xrcs", "sxr_diode_1", "efit"]
st40_code = ReadST40(pulse_code, tstart, tend, dt=dt, tree=code)

st40_code.get_raw_data("", code, run_code)
st40_code.bin_data_in_time([code], tstart, tend, dt)
data_code = st40_code.binned_data[code]

st40 = ReadST40(pulse, tstart, tend, dt=dt, tree="st40")
st40(instruments=instruments, map_diagnostics=False)

# st40_a = ReadST40(pulse_code1, tstart, tend, dt=dt, tree=code1)
# st40_a.get_raw_data("", code1, run_code1)
# st40_a.bin_data_in_time([code1], tstart, tend, dt)
# data_code_a = st40_a.binned_data[code1]
equilibrium = Equilibrium(data_code)
plasma = plasma_code(pulse_code, tstart, tend, dt, data_code, verbose=verbose)
plasma.set_equilibrium(equilibrium)
# Load and run the diagnostic forward models
# time=0.05
# plt.figure()
# plasma.electron_density.sel(t=time, method="nearest").plot(label="electrons")
# plasma.ion_density.sel(element=plasma.main_ion).sel(t=time, method="nearest").plot(
#     label="main ion"
# )
# plasma.fast_density.sel(t=time, method="nearest").plot(label="fast ions")
# plt.title("Electron/Ion densities")
# plt.legend()

raw = st40.raw_data
binned = st40.binned_data
bckc: dict = {}
models = initialize_diagnostic_models(binned, plasma=plasma, equilibrium=equilibrium)
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
plot = True
tplot = 0.05
if plot:
    plot_modelling_results(raw, binned, bckc, plasma, tplot)

# time=0.05
# plt.figure()
# levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
# plasma.equilibrium.rho.sel(t=time, method="nearest").plot.contour(levels=levels)
# plt.axis("scaled")
# plt.title("Equilibrium")

help(st40)
