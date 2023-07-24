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
from indica.operators import tomo_1D
from indica.operators.gpr_fit import run_gpr_fit
import indica.physics as ph
from indica.readers.read_st40 import ReadST40
from indica.utilities import set_plot_colors
from indica.workflows.bayes_workflow import plot_bayes_result
from indica.workflows.bayes_workflow import sample_with_autocorr

PATHNAME = "./plots/"

PULSE = 11085  # 11088, 11086, 11089, 11092, 11093
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
    "Nimp_prof.peaking": get_uniform(1, 8),
    "Nimp_prof.wcenter": get_uniform(0.1, 0.4),
    "Nimp_prof.y0": get_uniform(1e16, 1e19),
    "Nimp_prof.y1": get_uniform(1e16, 1e19),
    # "Ne_prof.y0/Nimp_prof.y0": lambda x1, x2: np.where(
    #     (x1 > x2 * 100) & (x1 < x2 * 1e4), 1, 0
    # ),
    "Nimp_prof.y0/Nimp_prof.y1": lambda x1, x2: np.where((x1 >= x2), 1, 0),
    "Te_prof.y0": get_uniform(1000, 6000),
    "Te_prof.peaking": get_uniform(1, 4),
    "Ti_prof.y0": get_uniform(2000, 10000),
    "Ti_prof.peaking": get_uniform(1, 4),
}

PHANTOM_PROFILE_PARAMS = {
    "Ne_prof.y0": 5e19,
    "Ne_prof.wcenter": 0.4,
    "Ne_prof.peaking": 2,
    "Ne_prof.y1": 2e18,
    "Ne_prof.yend": 1e18,
    "Ne_prof.wped": 2,
    "Nimp_prof.y0": 2e18,
    "Nimp_prof.y1": 5e16,
    "Nimp_prof.peaking": 2,
    "Te_prof.y0": 3000,
    "Te_prof.peaking": 2,
    "Ti_prof.y0": 5000,
    "Ti_prof.peaking": 2,
}
PARAM_NAMES = [
    "Nimp_prof.y0",
    "Nimp_prof.y1",
    "Nimp_prof.peaking",
]


# TODO: allow conditional prior usage even when only
#  one param is being optimisied i.e. 1 is constant


def prepare_data_ts(
    plasma,
    models,
    st40: ReadST40 = None,
    phantom_data: bool = True,
    xdim: str = "R",
    map_to_rho: bool = True,
    err_bounds: tuple = (0, 0),
    flat_data: dict = {},
):
    quantities = ["te", "ne"]
    if st40 is not None and not phantom_data:
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
            data.attrs["error"] = (
                deepcopy(st40.binned_data["ts"][quantity].error) * const
            )

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
                    _x = exp_rho.sel(t=t).where(data.R >= Rmag, drop=True)
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
    # else:
    #     for quantity in quantities:
    #         flat_data[f"ts.{quantity}"] = models["ts"]()[quantity]

    return flat_data


def prepare_data_cxrs(
    plasma,
    models,
    st40: ReadST40 = None,
    phantom_data: bool = True,
    flat_data: dict = {},
):
    instruments = ["pi", "tws_c"]
    if st40 is not None:
        for instrument in instruments:
            if instrument not in models.keys() or instrument not in st40.binned_data:
                continue
            data = st40.binned_data[instrument]
            attrs = data["spectra"].attrs
            (data["background"], data["brightness"],) = models[
                instrument
            ].integrate_spectra(data["spectra"])
            data["background"].attrs = attrs
            data["brightness"].attrs = attrs

            models[instrument].set_los_transform(data["spectra"].transform)
            models[instrument].set_plasma(plasma)

            for quantity in data.keys():
                flat_data[f"{instrument}.{quantity}"] = data[quantity]

    if phantom_data or st40 is None:
        if "pi" in models.keys():
            bckc = models["pi"]()
            flat_data["pi.brightness"] = bckc["brightness"]
            flat_data["pi.emissivity"] = models["pi"].emissivity

        if "tws_c" in models.keys():
            bckc = models["tws_c"]()
            flat_data["tws_c.brightness"] = bckc["brightness"]
            flat_data["tws_c.emissivity"] = models["tws_c"].emissivity

    return flat_data


def run_bayesian_analysis(
    pulse: int,
    time: float,
    phantom_profile_params,
    iterations,
    result_path,
    tstart=0.01,
    tend=0.1,
    dt=0.01,
    burn_in=0,
    nwalkers=10,
    phantom_data: bool = True,
):
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
    models = {}
    if not phantom_data:
        plasma.initialize_variables()

    print("Reading data")
    st40 = None
    if pulse is not None:
        st40 = ReadST40(pulse, tstart, tend, dt)
        st40(["pi", "tws_c", "ts", "efit"])
        plasma.set_equilibrium(Equilibrium(st40.raw_data["efit"]))

    print("Generating model")
    _, pi_model, bckc = example_diode(plasma=plasma)
    models["pi"] = pi_model
    models["pi"].name = "pi"
    models["tws_c"] = deepcopy(pi_model)
    models["tws_c"].name = "pi"

    flat_data: dict = {}
    print("Preparing experimental data")
    prepare_data_ts(
        plasma,
        models,
        st40=st40,
        phantom_data=phantom_data,
        flat_data=flat_data,
    )
    prepare_data_cxrs(
        plasma,
        models,
        st40=st40,
        phantom_data=phantom_data,
        flat_data=flat_data,
    )

    for key in flat_data.keys():
        if "t" not in flat_data[key].dims:
            flat_data[key] = flat_data[key].sel(t=[time])

    # Reorganize channel names so they start at 0
    t = flat_data["pi.brightness"].t
    channel = np.arange(flat_data["pi.brightness"].channel.size)
    flat_data["pi.brightness"] = DataArray(
        flat_data["pi.brightness"].values,
        coords=[("t", t), ("channel", channel)],
    )

    time = plasma.t.sel(t=time, method="nearest")
    plasma.time_to_calculate = time
    plasma.update_profiles(phantom_profile_params)
    if pulse is not None and not phantom_data:
        # Assign experimental data to plasma class
        plasma.electron_density.loc[dict(t=time)] = (
            flat_data["ts.ne_fit"].sel(t=time).interp(rho_poloidal=plasma.rho)
        )
        plasma.electron_temperature.loc[dict(t=time)] = (
            flat_data["ts.te_fit"].sel(t=time).interp(rho_poloidal=plasma.rho)
        )

    phantom_profiles = {
        "electron_density": plasma.electron_density.sel(t=time),
        "electron_temperature": plasma.electron_temperature.sel(t=time),
        "ion_temperature": plasma.ion_temperature.sel(t=time, element=IMPURITIES[0]),
        "impurity_density": plasma.impurity_density.sel(t=time, element=IMPURITIES[0]),
        "zeff": plasma.zeff.sum("element").sel(t=time),
    }
    if not pulse:
        for key in phantom_profiles.keys():
            phantom_profiles[key] = None

    print("Instatiating Bayes model")
    bm = BayesModels(
        plasma=plasma,
        data=flat_data,
        diagnostic_models=[pi_model],
        quant_to_optimise=[
            "pi.brightness",
        ],
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
        "phantom_profiles": phantom_profiles,
        "plasma": plasma,
        "autocorr": autocorr,
    }
    print(sampler.acceptance_fraction.sum())
    plot_bayes_result(**result, figheader=result_path)

    if not phantom_data and pulse is not None:
        plt.figure()
        Te = flat_data["ts.te"].sel(t=time)
        rho = Te.transform.rho.sel(t=time)
        plt.plot(rho, Te, "o")
        plasma.electron_temperature.sel(t=time).plot()

        plt.figure()
        Ne = flat_data["ts.ne"].sel(t=time)
        rho = Ne.transform.rho.sel(t=time)
        plt.plot(rho, Ne, "o")
        plasma.electron_density.sel(t=time).plot()

    return plasma, st40, flat_data


def run_inversion(
    pulse,
    tstart=0.01,
    tend=0.1,
    dt=0.01,
    reg_level_guess: float = 0.3,
    phantom_data: bool = True,
):
    models = {}
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
    print("Reading data")
    st40 = None
    if pulse is not None:
        st40 = ReadST40(pulse, tstart, tend, dt)
        st40(["pi", "tws_c", "ts", "efit"])
        plasma.set_equilibrium(Equilibrium(st40.raw_data["efit"]))

    print("Generating model")
    _, pi_model, bckc = example_diode(plasma=plasma)
    models["pi"] = pi_model
    models["pi"].name = "pi"
    models["tws_c"] = deepcopy(pi_model)
    models["tws_c"].name = "pi"

    # return plasma, models, st40
    print("Preparing the data")
    flat_data: dict = {}
    prepare_data_ts(
        plasma, models, st40=st40, phantom_data=phantom_data, flat_data=flat_data
    )
    prepare_data_cxrs(
        plasma, models, st40=st40, phantom_data=phantom_data, flat_data=flat_data
    )

    if pulse is not None and not phantom_data:
        # Assign experimental data to plasma class
        plasma.electron_density = flat_data["ts.ne_fit"].interp(
            t=plasma.t, rho_poloidal=plasma.rho
        )
        plasma.electron_temperature = flat_data["ts.te_fit"].interp(
            t=plasma.t, rho_poloidal=plasma.rho
        )
        plasma.ion_temperature = plasma.ion_temperature * 0.0
        plasma.impurity_density = plasma.impurity_density * 0.0

    t = flat_data["pi.brightness"].t
    channel = np.arange(flat_data["pi.brightness"].channel.size)
    los_transform = flat_data["pi.brightness"].transform
    data = DataArray(
        flat_data["pi.brightness"].values,
        coords=[("t", t), ("channel", channel)],
    )

    has_data = np.isfinite(data) * (data > 0)
    data_to_invert = data.where(has_data, drop=True)
    channels = data_to_invert.channel

    has_data = [True] * len(channels)
    z = los_transform.z.sel(channel=channels)
    R = los_transform.R.sel(channel=channels)
    dl = los_transform.dl

    rho_equil = los_transform.equilibrium.rho.interp(t=data.t)
    input_dict = dict(
        brightness=data_to_invert.data,
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

    pi_model.los_transform.plot()
    tomo.show_reconstruction()

    inverted_emissivity = DataArray(
        tomo.emiss, coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)]
    )
    inverted_error = DataArray(
        tomo.emiss_err,
        coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)],
    )
    inverted_emissivity.attrs["error"] = inverted_error
    # data_tomo = data
    # bckc_tomo = DataArray(tomo.backprojection, coords=data_tomo.coords)

    zeff = ph.zeff_bremsstrahlung(
        plasma.electron_temperature,
        plasma.electron_density,
        pi_model.filter_wavelength,
        bremsstrahlung=inverted_emissivity,
        gaunt_approx="callahan",
    )
    # zeff = zeff.where(zeff < 10, drop=True)

    cols = CM(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
    plt.figure()
    for i, t in enumerate(zeff.t):
        if i % 2:
            zeff.sel(t=t).plot(color=cols[i])
            if phantom_data:
                plasma.zeff.sum("element").sel(t=t).plot(
                    marker="o", color=cols[i], alpha=0.5, linestyle=""
                )
    if phantom_data:
        plasma.zeff.sum("element").sel(t=t).plot(
            marker="o", color=cols[i], alpha=0.5, linestyle="", label="Phantom"
        )
    zeff.sel(t=t).plot(label="Recalculated", color=cols[i])
    plt.ylim(0, 10)
    plt.ylabel("Zeff")
    plt.legend()

    if not phantom_data:
        plt.figure()
        Te = flat_data["ts.te"]
        rho = Te.transform.rho
        for i, t in enumerate(Te.t):
            if i % 2:
                plasma.electron_temperature.sel(t=t).plot(color=cols[i])
                plt.plot(rho.sel(t=t), Te.sel(t=t), color=cols[i], marker="o")

        plt.figure()
        Ne = flat_data["ts.ne"]
        rho = Ne.transform.rho
        for i, t in enumerate(Ne.t):
            if i % 2:
                plasma.electron_density.sel(t=t).plot(color=cols[i])
                plt.plot(rho.sel(t=t), Ne.sel(t=t), color=cols[i], marker="o")

    return plasma, st40, flat_data, zeff


def inversion_phantom_example(pulse: int = 11085):
    ff = run_inversion(pulse, phantom_data=True)

    return ff


def inversion_data_example(pulse: int = 11085):
    ff = run_inversion(pulse, phantom_data=False)

    return ff


def bayesian_phantom_example(
    pulse: int = 11085, time: float = 0.04, iterations=200, nwalkers=30
):
    ff = run_bayesian_analysis(
        pulse,
        time,
        PHANTOM_PROFILE_PARAMS,
        iterations,
        PATHNAME,
        burn_in=0,
        nwalkers=nwalkers,
        phantom_data=True,
    )

    return ff


def bayesian_data_example(
    pulse: int = 11085, time: float = 0.04, iterations=200, nwalkers=30
):
    ff = run_bayesian_analysis(
        pulse,
        time,
        PHANTOM_PROFILE_PARAMS,
        iterations,
        PATHNAME,
        burn_in=0,
        nwalkers=nwalkers,
        phantom_data=False,
    )

    return ff
