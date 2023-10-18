from copy import deepcopy

import emcee
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import xarray as xr
from xarray import DataArray

from indica.bayesmodels import BayesModels
from indica.bayesmodels import get_uniform
from indica.equilibrium import Equilibrium
from indica.models.diode_filters import example_run as example_diode
from indica.models.plasma import example_run as example_plasma
from indica.models.plasma import Plasma
from indica.operators import tomo_1D
from indica.operators.gpr_fit import run_gpr_fit
import indica.physics as ph
from indica.readers.read_st40 import ReadST40
from indica.utilities import set_plot_colors
from indica.workflows.bayes_workflow import plot_bayes_result
from indica.workflows.bayes_workflow import sample_with_autocorr

PATHNAME = "./plots/"

PULSE = 11085
TIME = 0.07
MAIN_ION = "h"
IMPURITIES = ("c",)
IMPURITY_CONCENTRATION = (0.03,)
FULL_RUN = False
N_RAD = 10

CM, COLS = set_plot_colors()

PRIORS = {
    "Ne_prof.y0": get_uniform(1e19, 8e19),
    "Ne_prof.y1": get_uniform(1e18, 5e18),
    "Ne_prof.y0/Ne_prof.y1": lambda x1, x2: np.where(((x1 > x2 * 2)), 1, 0),
    "Ne_prof.wped": get_uniform(1, 5),
    "Ne_prof.wcenter": get_uniform(0.1, 0.8),
    "Ne_prof.peaking": get_uniform(1, 5),
    "Nimp_prof.y0": get_uniform(8e16, 2e19),
    "Nimp_prof.y1": get_uniform(1e16, 2e19),
    "Nimp_prof.yend": get_uniform(8e16, 2e19),
    "Nimp_prof.peaking": get_uniform(0.05, 12),
    "Nimp_prof.wcenter": get_uniform(0.1, 0.5),
    "Nimp_prof.wped": get_uniform(1, 5),
    "Nimp_prof.y0/Nimp_prof.y1": lambda x1, x2: np.where((x1 >= x2), 1, 0),
    "Nimp_prof.y1/Nimp_prof.yend": lambda x1, x2: np.where((x1 >= x2), 1, 0),
    "Te_prof.y0": get_uniform(1000, 6000),
    "Te_prof.peaking": get_uniform(1, 4),
    "Ti_prof.y0": get_uniform(2000, 10000),
    "Ti_prof.peaking": get_uniform(1, 4),

    # "Nimp_prof.y0": get_uniform(1e16, 1e19),
    # "Nimp_prof.y1": get_uniform(1e15, 1e19),
    # "Nimp_prof.yend": get_uniform(1e15, 1e19),
}

PHANTOM_PROFILE_PARAMS = {
    "Ne_prof.y0": 5e19,
    "Ne_prof.wcenter": 0.4,
    "Ne_prof.peaking": 2,
    "Ne_prof.y1": 2e18,
    "Ne_prof.yend": 1e18,
    "Ne_prof.wped": 2,
    "Te_prof.y0": 3000,
    "Te_prof.peaking": 2,
    "Ti_prof.y0": 5000,
    "Ti_prof.peaking": 2,

    "Nimp_prof.y0": 1e18,
    "Nimp_prof.y1": 1e16,
    "Nimp_prof.yend": 1e16,
    "Nimp_prof.peaking": 5,
    "Nimp_prof.wcenter": 0.4,
    "Nimp_prof.wped":6,


}
PARAM_NAMES = [
    "Nimp_prof.y0",
    "Nimp_prof.y1",
    "Nimp_prof.peaking",
    "Nimp_prof.wcenter",
    "Nimp_prof.wped",
]


# TODO: allow conditional prior usage even when only
#  one param is being optimisied i.e. 1 is constant


def prepare_data_ts(
    plasma: Plasma,
    models: dict,
    st40: ReadST40,
    xdim: str = "R",
    ts_side: str ="LFS",
    map_to_rho: bool = True,
    err_bounds: tuple = (0, 0),
    flat_data: dict = {},
):
    if "ts" not in st40.binned_data.keys():
        raise ValueError

    quantities = ["te", "ne"]
    for quantity in quantities:
        if "ts" not in st40.binned_data.keys():
            continue

        _data = st40.binned_data["ts"][quantity]
        if hasattr(_data.transform, "equilibrium"):
            _data.transform.convert_to_rho_theta(t=_data.t)

        flat_data[f"ts.{quantity}"] = _data

        # TODO: normalizing data due to issues with fit convergence
        const = 1.0
        if quantity == "ne":
            const = 1.0e-16
        xr.set_options(keep_attrs=True)
        data = deepcopy(st40.binned_data["ts"][quantity]) * const
        data.attrs["error"] = deepcopy(st40.binned_data["ts"][quantity].error) * const

        y_bounds = (1, 1)
        if xdim == "R":
            x_bounds = plasma.machine_dimensions[0]
        else:
            x_bounds = (0, 1)

        if xdim not in data.dims and hasattr(data, xdim):
            data = data.swap_dims({"channel": xdim})

        fit, fit_err = run_gpr_fit(
            data,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            err_bounds=err_bounds,
            xdim=xdim,
        )
        fit /= const
        fit_err /= const

        if not map_to_rho:
            st40.binned_data["ts"][f"{quantity}_fit"] = fit
            # _data["ts"][f"{quantity}_fit"].attrs["error"] = fit_err
            flat_data[f"ts.{quantity}_fit"] = fit
        else:
            fit_rho, _, _ = plasma.equilibrium.flux_coords(fit.R, fit.R * 0, fit.t)
            exp_rho = fit_rho.interp(R=data.R)
            rmag = plasma.equilibrium.rmag.interp(t=fit.t)

            fit_lfs = []
            for t in fit.t:
                Rmag = rmag.sel(t=t)
                #_x = exp_rho.sel(t=t).where(data.R >= Rmag, drop=True)
                if ts_side == "LFS":
                    _x = exp_rho.sel(t=t).where(exp_rho.R >= Rmag, drop=True)
                else:
                    _x = exp_rho.sel(t=t).where(exp_rho.R <= Rmag, drop=True)
                _y = fit.sel(t=t).interp(R=_x.R)
                ind = np.argsort(_x.values)
                x = _x.values[ind]
                y = _y.values[ind]
                cubicspline = CubicSpline(
                    np.append(0, x[1:]), np.append(y[0], y[1:]), bc_type="clamped"
                )
                _fit_lfs = cubicspline(plasma.rho)
                fit_lfs.append(
                    DataArray(_fit_lfs, coords=[("rho_poloidal", plasma.rho)])
                )
            fit_lfs = xr.concat(fit_lfs, "t").assign_coords(t=fit.t)
            st40.binned_data["ts"][f"{quantity}_fit"] = fit_lfs
            # _data["ts"][f"{quantity}_fit"].attrs["error"] = fit_lfs_err
            flat_data[f"ts.{quantity}_fit"] = fit_lfs

    if "ts" in models.keys():
        models["ts"].set_los_transform(data["te"].transform)
        models["ts"].set_plasma(plasma)

    return flat_data


def prepare_data_cxrs(
    plasma: Plasma,
    models: dict,
    st40: ReadST40,
    flat_data: dict = {},
):
    """
    Read CXRS data from experiment, assign transform to model and return flat_data
    dictionary with either experimental or phantom data built using the models.
    """
    instruments = ["pi", "tws_c"]
    for instrument in instruments:
        if instrument not in st40.binned_data.keys():
            continue
        data = st40.binned_data[instrument]
        attrs = data["spectra"].attrs
        (data["background"], data["brightness"],) = models[
            instrument
        ].integrate_spectra(data["spectra"])
        data["background"].attrs = attrs
        data["brightness"].attrs = attrs

        for quantity in data.keys():
            flat_data[f"{instrument}.{quantity}"] = data[quantity]

        if instrument in models.keys():
            models[instrument].set_los_transform(data["spectra"].transform)
            models[instrument].set_plasma(plasma)

    return flat_data


def prepare_inputs(
    pulse: int,
    tstart=0.01,
    tend=0.1,
    dt=0.01,
    time: float = None,
    phantom_profile_params: dict = None,
    phantom_data: bool = True,
    ts_side: str ="LFS",
):

    flat_data: dict = {}
    models: dict = {}

    print("Generating plasma")
    plasma = example_plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=MAIN_ION,
        impurities=IMPURITIES,
        impurity_concentration=IMPURITY_CONCENTRATION,
        full_run=FULL_RUN,
        n_rad=N_RAD,
    )
    if not phantom_data:
        plasma.initialize_variables()

    if time is None:
        time = plasma.t

    time = plasma.t.sel(t=time, method="nearest")
    plasma.time_to_calculate = time

    if phantom_profile_params is not None:
        plasma.update_profiles(phantom_profile_params)

    print("Generating diagnostic models")
    _, pi_model, bckc = example_diode(plasma=plasma)
    models["pi"] = pi_model
    models["pi"].name = "pi"
    models["tws_c"] = deepcopy(pi_model)
    models["tws_c"].name = "pi"

    if pulse is not None:
        print("Reading experimental data")
        st40 = ReadST40(pulse, tstart, tend, dt)
        st40(["pi", "tws_c", "ts", "efit"])
        plasma.set_equilibrium(Equilibrium(st40.raw_data["efit"]))

        if not phantom_data and "ts" in st40.binned_data.keys():
            print("Fitting TS data")
            prepare_data_ts(
                plasma,
                models,
                st40,
                flat_data=flat_data,
                ts_side=ts_side,
            )
            plasma.electron_density.loc[dict(t=time)] = (
                flat_data["ts.ne_fit"].sel(t=time).interp(rho_poloidal=plasma.rho)
            )
            plasma.electron_temperature.loc[dict(t=time)] = (
                flat_data["ts.te_fit"].sel(t=time).interp(rho_poloidal=plasma.rho)
            )
            plasma.Ne_prof = lambda: plasma.electron_density.loc[dict(t=time)]
            plasma.Te_prof = lambda: plasma.electron_temperature.loc[dict(t=time)]
        print("Fitting Bremsstrahlung PI/TWS_C spectra")
        prepare_data_cxrs(
            plasma,
            models,
            st40,
            flat_data=flat_data,
        )

    if phantom_data:
        plasma.impurity_density *= plasma.t * 100.0
        print("Generating phantom data using Plasma and Models")
        for instrument in ["pi", "tws_c"]:
            bckc = models[f"{instrument}"]()
            flat_data[f"{instrument}.brightness"] = bckc["brightness"]
            flat_data[f"{instrument}.emissivity"] = models[f"{instrument}"].emissivity

            transform = models[f"{instrument}"].los_transform
            flat_data[f"{instrument}.brightness"].attrs["transform"] = transform
            flat_data[f"{instrument}.emissivity"].attrs["transform"] = transform

    if phantom_data:
        zeff = plasma.zeff.sum("element").sel(t=time)
        impurity_density = plasma.impurity_density.sel(t=time, element=IMPURITIES[0])
    else:
        zeff, impurity_density = None, None

    input_profiles = {
        "electron_density": deepcopy(plasma.electron_density.sel(t=time)),
        "electron_temperature": deepcopy(plasma.electron_temperature.sel(t=time)),
        "ion_temperature": deepcopy(
            plasma.ion_temperature.sel(t=time, element=IMPURITIES[0])
        ),
        "impurity_density": deepcopy(impurity_density),
        "zeff": deepcopy(zeff),
    }
    #plasma.Ne_prof=lambda: plasma.electron_density.loc[dict(t=time)]
    #plasma.Te_prof = lambda: plasma.electron_temperature.loc[dict(t=time)]
    for key in flat_data.keys():
        if "t" not in flat_data[key].dims:
            flat_data[key] = flat_data[key].expand_dims(
                dim={"t": [plasma.time_to_calculate]}
            )
        else:
            if np.size(time) == 1:
                flat_data[key] = flat_data[key].sel(t=[time])

        if "brightness" in key:
           print(f"Reorganising {key} channel range starting at 0")
           t = flat_data[key].t
           channel = np.arange(flat_data[key].channel.size)
           flat_data[key] = DataArray(
                flat_data[key].values,
                coords=[("t", t), ("channel", channel)],
                attrs=flat_data[key].attrs,
            )

    return plasma, models, flat_data, input_profiles


def run_bayes(
    pulse: int,
    time: float,
    phantom_profile_params,
    iterations,
    result_path,
    tstart=0.03,
    tend=0.1,
    dt=0.01,
    burn_in=0,
    nwalkers=10,
    phantom_data: bool = True,
    ts_side: str="LFS",
):

    plasma, models, flat_data, input_profiles = prepare_inputs(
        pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        time=time,
        phantom_profile_params=phantom_profile_params,
        phantom_data=phantom_data,
        ts_side=ts_side,
    )
    phantom_plasma = deepcopy(plasma)

    print("Instatiating Bayes model")
    diagnostic_models = [models["pi"]]
    quant_to_optimise = [
        "pi.brightness",
    ]
    bm = BayesModels(
        plasma=plasma,
        data=flat_data,
        diagnostic_models=diagnostic_models,
        quant_to_optimise=quant_to_optimise,
        priors=PRIORS,
    )
    ndim = PARAM_NAMES.__len__()
    start_points = bm.sample_from_priors(PARAM_NAMES, size=nwalkers)
    move = [(emcee.moves.StretchMove(), 1.0), (emcee.moves.DEMove(), 0.0)]
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_fn=bm.ln_posterior,
        parameter_names=PARAM_NAMES,
        moves=move,
    )

    print("Sampling")
    autocorr = sample_with_autocorr(
        sampler, start_points, iterations=iterations, auto_sample=5
    )

    blobs = sampler.get_blobs(discard=burn_in, flat=True)
    blob_names = sampler.get_blobs().flatten()[0].keys()
    blob_dict = {
        blob_name: xr.concat(
            [data[blob_name] for data in blobs],
            dim=pd.Index(np.arange(0, blobs.__len__()), name="index"),
        )
        for blob_name in blob_names
    }

    samples = sampler.get_chain(flat=True)
    prior_samples = bm.sample_from_priors(PARAM_NAMES, size=int(1e5))
    result = {
        "blobs": blob_dict,
        "diag_data": flat_data,
        "samples": samples,
        "prior_samples": prior_samples,
        "param_names": PARAM_NAMES,
        "phantom_profiles": input_profiles,
        "plasma": plasma,
        "autocorr": autocorr,
    }
    print(sampler.acceptance_fraction.sum())
    plot_bayes_result(**result, figheader=result_path)

    #if not phantom_data and pulse is not None:
       # plot_ts(phantom_plasma, flat_data, tplot=[time])
    if not phantom_data:
        plot_ts(plasma, flat_data, ts_side=ts_side)

    return result


def run_inversion(
    pulse,
    #phantom_profile_params,
    tstart=0.03,
    tend=0.1,
    dt=0.01,
    reg_level_guess: float = 0.3,
    phantom_data: bool = True,
    ts_side: str = "LFS",
):

    plasma, models, flat_data, input_profiles = prepare_inputs(
        pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        #phantom_profile_params=phantom_profile_params,
        phantom_data=phantom_data,
        ts_side=ts_side,
    )

    data = flat_data["pi.brightness"]
    has_data = np.isfinite(data) #* (data > 0)
    data_to_invert = data.where(has_data, drop=True)
    channels = data_to_invert.channel
    has_data = [True] * len(channels)

    los_transform = data.transform
    z = los_transform.z.sel(channel=channels)
    R = los_transform.R.sel(channel=channels)
    dl = los_transform.dl

    rho_equil = los_transform.equilibrium.rho.interp(t=data.t)
    input_dict = dict(
        brightness=data_to_invert.data,
        brightness_error=data_to_invert.data * 0.1,
        dl=dl,
        t=data_to_invert.t.data,
        R=R,
        z=z,
        rho_equil=dict(
            R=rho_equil.R.data,
            z=rho_equil.z.data,
            t=rho_equil.t.data,
            rho=rho_equil.data,
        ),
        has_data=has_data,
        debug=False,
    )

    if "pi.emissivity" in flat_data.keys() is not None:
        input_dict["emissivity"] = flat_data["pi.emissivity"]

    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=reg_level_guess)
    tomo()

    #models["pi"].los_transform.plot()
    tomo.show_reconstruction()

    inverted_emissivity = DataArray(
        tomo.emiss, coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)]
    )
    inverted_error = DataArray(
        tomo.emiss_err,
        coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)],
    )
    inverted_emissivity.attrs["error"] = inverted_error


    zeff = ph.zeff_bremsstrahlung(
        plasma.electron_temperature,
        plasma.electron_density,
        models["pi"].filter_wavelength,
        bremsstrahlung=inverted_emissivity,
        gaunt_approx="callahan",
    )
    zeff_up = ph.zeff_bremsstrahlung(
        plasma.electron_temperature,
        plasma.electron_density,
        models["pi"].filter_wavelength,
        bremsstrahlung=inverted_emissivity - inverted_error,
        gaunt_approx="callahan",
    )
    zeff_down = ph.zeff_bremsstrahlung(
        plasma.electron_temperature,
        plasma.electron_density,
        models["pi"].filter_wavelength,
        bremsstrahlung=inverted_emissivity + inverted_error,
        gaunt_approx="callahan",
    )
    #zeff = zeff.where(zeff < 10, np.nan)

    cols = CM(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
    plt.figure()
    lines = []
    for i, t in enumerate(zeff.t):
        if i % 2:
            line3, = zeff.sel(t=t).plot(color=cols[i], label=f"{t.values:.3f}")
            plt.fill_between(
                zeff.rho_poloidal,
                zeff_up.sel(t=t),
                zeff_down.sel(t=t),
                color=cols[i],
                alpha=0.1,
            )
            if phantom_data:
                plasma.zeff.sum("element").sel(t=t).plot(
                    color=cols[i],
                    alpha=0.25,
                    linestyle="--",
                    label=f"{t.values:.3f}",
                )
            lines.append([line3])
    if phantom_data:
        line1, = plasma.zeff.sum("element").sel(t=t).plot(
            color=cols[i], alpha=0.5,
            linestyle="--", label="Phantom"
        )
    line2, = zeff.sel(t=t).plot(label="Recalculated", color=cols[i])
    plt.ylim(0, 10)
    plt.grid(alpha=0.25)
    plt.ylabel("$Z_{eff}$")
    plt.xlabel("$ρ_{pol}$")
    plt.title("")
    if phantom_data:
        first_legend = plt.legend(handles=[line1, line2], loc="upper left")
    else:
        first_legend = plt.legend(handles=[line2], loc="upper left")
    ax = plt.gca().add_artist(first_legend)

    times = ["0.04 s", "0.06 s", "0.08 s", "0.1 s"]
    plt.legend([i[0] for i in lines], times, loc="upper right")

    if not phantom_data:
        plot_ts(plasma, flat_data, cols=cols, ts_side=ts_side)
    plt.figure()
    return zeff


def plot_ts(plasma: Plasma, flat_data: dict, cols=None, ts_side: str="LFS"):
    if cols is None:
        cols = CM(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))

   # if tplot is None:
    #    tplot = list(plasma.t.values)
    #else:
    #    cols = [cols[int(np.size(plasma.t) / 2.0), :]]

    plt.figure()
    Te=flat_data["ts.te"]
    Te_err=flat_data["ts.te"].error
    Ne=flat_data["ts.ne"]
    Ne_err=flat_data["ts.ne"].error
    rho=Te.transform.rho
    rmag=plasma.equilibrium.rmag

    for i, t in enumerate(Te.t):
        plasma.electron_temperature.sel(t=t).plot(color=cols[i], label=f"{t.values:.3f}")

        #plt.legend()
        #plt.title(title)
        if ts_side == "LFS":
            channels = np.where(Te.R >= rmag.sel(t=t,
                                        method="nearest"))[0]
        else:
            channels = np.where(Te.R <= rmag.sel(t=t,
                                        method="nearest"))[0]
        x = rho.sel(channel=channels).sel(t=t,
                                          method="nearest")
        y = Te.sel(channel=channels).sel(t=t,
                                         method="nearest")
        err = Te_err.sel(channel=channels).sel(t=t,
                                               method="nearest")
        plt.errorbar(x, y, err, color=cols[i],
                     marker="o", linestyle="")
    plasma.electron_temperature.sel(t=t).plot(color=cols[i], label="Fit")
    plt.errorbar(x, y, err, color=cols[i], marker="o", linestyle="", label="Data")
    plt.ylim(
        0,
        np.max([plasma.electron_temperature.max(),
                flat_data["ts.te"].max()]) * 1.1
    )
    plt.legend()
    plt.title("TS electron temperature")

    plt.figure()
    for i, t in enumerate(Ne.t):
        plasma.electron_density.sel(t=t).plot(color=cols[i], label=f"{t.values:.3f}")
        if ts_side == "LFS":
            channels = np.where(Ne.R >= rmag.sel(t=t, method="nearest"))[0]
        else:
            channels = np.where(Ne.R <= rmag.sel(t=t, method="nearest"))[0]
        x = rho.sel(channel=channels).sel(t=t, method="nearest")
        y = Ne.sel(channel=channels).sel(t=t, method="nearest")
        err = Ne_err.sel(channel=channels).sel(t=t, method="nearest")
        plt.errorbar(x, y, err, color=cols[i], marker="o", linestyle="")
    plasma.electron_density.sel(t=t).plot(color=cols[i], label="Fit")
    plt.errorbar(x, y, err, color=cols[i], marker="o", linestyle="", label="Data")
    plt.legend()
    plt.title("TS electron density")


def inversion_example(
    pulse: int = 11085, phantom_data: bool = True,
        ts_side: str = "LFS"
):
    ff = run_inversion(pulse,
                       phantom_data=phantom_data,
                       #phantom_profile_params=PHANTOM_PROFILE_PARAMS,
                       ts_side=ts_side
                       )
    return ff


def bayesian_example(
    pulse: int = 11085, #11228, 11227, 11226, 11225, 11224
    time: float = 0.06,
    iterations=200,
    nwalkers=50,
    phantom_data: bool = False,
    ts_side: str="LFS",
):
    ff = run_bayes(
        pulse,
        time,
        PHANTOM_PROFILE_PARAMS,
        iterations,
        PATHNAME,
        burn_in=0,
        nwalkers=nwalkers,
        phantom_data=phantom_data,
        ts_side=ts_side,
    )

    return ff
