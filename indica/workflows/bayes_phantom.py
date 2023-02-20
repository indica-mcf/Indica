import pickle
from pathlib import Path

import corner
import emcee
import flatdict
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.bayesmodels import get_uniform, BayesModels
from indica.models.interferometry import Interferometry
from indica.models.helike_spectroscopy import Helike_spectroscopy
from indica.models.plasma import Plasma
from indica.readers.manage_data import bin_data_in_time
from indica.readers.read_st40 import ST40data


def plot_bayes_phantom(figheader="./results/test/", blobs=None, diag_data=None, samples=None, prior_samples=None,
                       param_names=None, phantom_profiles=None, plasma=None, autocorr=None, ):

    Path(figheader).mkdir(parents=True, exist_ok=True)

    with open(figheader + "result.pkl", "wb") as handle:
        pickle.dump({"blobs":blobs, "diag_data":diag_data, "samples":samples, "phantom_profiles":phantom_profiles,
                     "plasma":plasma, "autocorr":autocorr}, handle, )

    plt.figure()
    mask = np.isfinite(autocorr)
    plt.plot(np.arange(0, autocorr.__len__())[mask], autocorr[mask], label="average tau")
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("tau")
    plt.savefig(figheader + "average_tau.png")

    if "xrcs.spectra" in blobs.keys():
        plt.figure()
        temp_data = blobs["xrcs.spectra"]
        plt.fill_between(temp_data.wavelength, temp_data.quantile(0.05, dim="index"), temp_data.quantile(0.95, dim="index"),
                         label="XRCS spectrum, 90% Confidence", zorder=3, color="blue")
        plt.fill_between(temp_data.wavelength, temp_data.quantile(0.01, dim="index"),
                         temp_data.quantile(0.99, dim="index"),
                         label="XRCS spectrum, 98% Confidence", zorder=2, color="grey")
        plt.plot(diag_data["xrcs.spectra"].wavelength,
                diag_data["xrcs.spectra"].sel(t=plasma.time_to_calculate).values,
                 linestyle="-", color="black", label="xrcs.spectra data", zorder=4)
        plt.legend()
        plt.savefig(figheader + "xrcs_spectra.png")


    if "smmh1.ne" in blobs.keys():
        plt.figure()
        temp_data = blobs["smmh1.ne"]
        plt.xlabel("samples ()")
        plt.ylabel("ne_int (m^-2)")
        plt.plot(temp_data, label="smmh1.ne_int model")
        plt.axhline(
            y=diag_data["smmh1.ne"].sel(t=plasma.time_to_calculate).values,
            color="red",
            linestyle="-",
            label="smmh1.ne_int data"
        )
        plt.legend()
        plt.savefig(figheader + "smmh1_ne.png")

    if "xrcs.te_kw" in blobs.keys():
        plt.figure()
        temp_data = blobs["xrcs.te_kw"][:, 0, 0]
        plt.ylabel("temperature (eV)")
        plt.plot(temp_data, label="xrcs.te_kw model", color="blue")
        plt.axhline(
            y=diag_data["xrcs.te_kw"][0,].sel(t=plasma.time_to_calculate).values,
            color="blue",
            linestyle="-",
            label="xrcs.te_kw data",
        )
        plt.legend()
        plt.savefig(figheader + "xrcs_te_kw.png")

    if "xrcs.ti_w" in blobs.keys():
        plt.figure()
        temp_data = blobs["xrcs.ti_w"][:, 0, 0]
        plt.plot(temp_data, label="xrcs.ti_w model", color="red")
        plt.axhline(
            y=diag_data["xrcs.ti_w"][0,].sel(t=plasma.time_to_calculate).values,
            color="red",
            linestyle="-",
            label="xrcs.ti_w data",
        )
        plt.legend()
        plt.savefig(figheader + "xrcs_ti_w.png")

    plt.figure()
    prof = blobs["electron_density"]
    plt.fill_between(prof.rho_poloidal, prof.quantile(0.05, dim="index"), prof.quantile(0.95, dim="index"),
        label="Ne, 90% Confidence", zorder=3, color="blue")
    plt.fill_between(prof.rho_poloidal, prof.quantile(0.01, dim="index"), prof.quantile(0.99, dim="index"),
        label="Ne, 98% Confidence", zorder=2, color="grey")
    if phantom_profiles:
        phantom_profiles["electron_density"].plot(label="phantom_profile", linestyle="--", color="black", zorder=4)
    plt.legend()
    plt.savefig(figheader + "electron_density.png")

    plt.figure()
    prof = blobs["electron_temperature"]
    plt.fill_between(prof.rho_poloidal, prof.quantile(0.05, dim="index"), prof.quantile(0.95, dim="index"),
        label="Te, 90% Confidence", zorder=3, color="blue")
    plt.fill_between(prof.rho_poloidal, prof.quantile(0.01, dim="index"), prof.quantile(0.99, dim="index"),
        label="Te, 98% Confidence", color="grey", zorder=2, alpha=0.7)
    if phantom_profiles:
        phantom_profiles["electron_temperature"].plot(label="Te, phantom_profile", linestyle="--", color="black", zorder=4)

    prof = blobs["ion_temperature"].sel(element="ar")
    plt.fill_between(prof.rho_poloidal, prof.quantile(0.05, dim="index"), prof.quantile(0.95, dim="index"),
        label="Ti, 90% Confidence", zorder=3, color="red")
    plt.fill_between(prof.rho_poloidal, prof.quantile(0.01, dim="index"), prof.quantile(0.99, dim="index", ),
        label="Ti, 98% Confidence", color="grey", zorder=2, alpha=0.7)
    if phantom_profiles:
        phantom_profiles["ion_temperature"].plot(label="Ti, phantom_profile", linestyle="-.", color="black", zorder=4)
    plt.legend()
    plt.ylabel("temperature (eV)")
    plt.savefig(figheader + "temperature.png")

    plt.figure()
    prof = blobs["impurity_density"].sel(element="ar")
    plt.fill_between(prof.rho_poloidal, prof.quantile(0.05, dim="index"), prof.quantile(0.95, dim="index"),
        label="Nimp, 90% Confidence", zorder=3, color="red")
    plt.fill_between(prof.rho_poloidal, prof.quantile(0.01, dim="index"), prof.quantile(0.99, dim="index", ),
        label="Nimp, 98% Confidence", color="grey")
    if phantom_profiles:
        phantom_profiles["impurity_density"].plot(label="phantom_profile", linestyle="--", color="black", zorder=4)
    plt.legend()
    plt.savefig(figheader + "impurity_density.png")

    corner.corner(samples, labels=param_names)
    plt.savefig(figheader + "posterior.png")

    corner.corner(prior_samples, labels=param_names)
    plt.savefig(figheader + "prior.png")


def sample_with_autocorr(sampler, start_points, iterations=10, auto_sample=5):
    autocorr = np.ones((iterations,)) * np.nan
    old_tau = np.inf
    for sample in sampler.sample(start_points, iterations=iterations, progress=True, ):
        if sampler.iteration % auto_sample:
            continue
        new_tau = sampler.get_autocorr_time(tol=0)
        autocorr[sampler.iteration - 1] = np.mean(new_tau)
        converged = np.all(new_tau * 50 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - new_tau) / new_tau < 0.01)
        if converged:
            break
        old_tau = new_tau
    autocorr = autocorr[:sampler.iteration]
    return autocorr

def initialise_diag_data(plasma, pulse, tstart=0.02, tend=0.10, dt=0.01):
    raw = ST40data(pulse, tstart - dt * 4, tend + dt * 4)
    raw_data = raw.get_all()
    equilibrium_data = raw_data["efit"]
    equilibrium = Equilibrium(equilibrium_data)
    flux_transform = FluxSurfaceCoordinates("poloidal")
    flux_transform.set_equilibrium(equilibrium)
    plasma.set_equilibrium(equilibrium)
    plasma.set_flux_transform(flux_transform)

    data = {}
    for instrument in raw_data.keys():
        quantities = list(raw_data[instrument])
        data[instrument] = bin_data_in_time(
            raw_data[instrument],
            plasma.tstart,
            plasma.tend,
            plasma.dt,
        )

        transform = data[instrument][quantities[0]].attrs["transform"]
        transform.set_equilibrium(equilibrium, force=True)
        for quantity in quantities:
            data[instrument][quantity].attrs["transform"] = transform

    # Get data as flat dict
    flat_data = flatdict.FlatDict(data, delimiter=".")
    return flat_data


if __name__ == "__main__":
    pulse = 9229
    tstart = 0.02
    tend = 0.10
    dt = 0.01

    # Initialise Plasma
    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion="h",
        impurities=("ar",),
        impurity_concentration=(0.001,),
        full_run=False,
    )
    plasma.time_to_calculate = plasma.t[3]
    plasma.update_profiles({"Ne_prof.y0": 5e19, "Ne_prof.wcenter": 0.4, "Ne_prof.peaking": 2,
                            "Ne_prof.y1": 1e19, "Ne_prof.yend": 1e18, "Ne_prof.wped": 5})
    plasma.build_atomic_data()

    flat_data = initialise_diag_data(plasma, pulse, tstart=tstart, tend=tend, dt=dt)

    # Initialise Diagnostic Models
    los_transform = flat_data["smmh1.ne"].transform
    smmh1 = Interferometry(name="smmh1", )
    smmh1.set_los_transform(los_transform)
    smmh1.plasma = plasma
    los_transform = flat_data["xrcs.te_kw"].transform
    xrcs = Helike_spectroscopy(name="xrcs", )
    xrcs.plasma = plasma
    xrcs.set_los_transform(los_transform)

    # Make phantom profiles
    phantom_profiles = {"electron_density": plasma.Ne_prof.yspl,
                        "electron_temperature": plasma.Te_prof.yspl,
                        "ion_temperature": plasma.Ti_prof.yspl,
                        "impurity_density": plasma.Nimp_prof.yspl}

    # Use phantom data instead of ST40 data
    flat_data["smmh1.ne"] = smmh1().pop("ne").expand_dims(dim={"t": [plasma.time_to_calculate]})
    flat_data["xrcs.te_kw"] = xrcs().pop("te_kw")
    flat_data["xrcs.ti_w"] = xrcs().pop("ti_w")
    # TODO: Add conditional priors e.g. y1 < y0
    priors = {
        "Ne_prof.y0": get_uniform(1e19, 1e20),
        "Ne_prof.y1": get_uniform(1e18, 5e18),
        # "Ne_prof.y0/Ne_prof.y1": lambda x1, x2: np.where(((x1>x2*1)), 1, 0),
        "Ne_prof.wped": get_uniform(1, 5),
        "Ne_prof.wcenter": get_uniform(0.1, 0.8),
        "Ne_prof.peaking": get_uniform(1, 5),

        "Nimp_prof.peaking": get_uniform(1, 8),
        "Nimp_prof.wcenter": get_uniform(0.1, 0.4),
        "Nimp_prof.y0": get_uniform(1e15, 1e17),
        "Nimp_prof.y1": get_uniform(1e15, 1e17),
        "Ne_prof.y0/Nimp_prof.y0": lambda x1, x2: np.where((x1 > x2 * 100) & (x1 < x2 * 1e4), 1, 0),
        "Nimp_prof.y0/Nimp_prof.y1": lambda x1, x2: np.where((x1 > x2), 1, 0),

        "Te_prof.y0": get_uniform(1000, 1e4),
        "Te_prof.peaking": get_uniform(1, 6),
        "Ti_prof.y0": get_uniform(1000, 1e4),
        "Ti_prof.peaking": get_uniform(1, 6),
    }
    param_names = [
        "Ne_prof.y0",
        "Ne_prof.y1",
        "Ne_prof.peaking",
        "Ne_prof.wped",
        "Ne_prof.wcenter",
    ]

    bm = BayesModels(
        plasma=plasma,
        data=flat_data,
        diagnostic_models=[
            smmh1,
            # xrcs
        ],
        quant_to_optimise=[
            "smmh1.ne",
            # "xrcs.te_kw",
            # "xrcs.ti_w",
        ],
        priors=priors,
    )

    ndim = param_names.__len__()
    nwalkers = ndim * 2
    start_points = bm.sample_from_priors(param_names, size=nwalkers)

    move = [emcee.moves.StretchMove()]
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_fn=bm.ln_posterior,
        parameter_names=param_names,
        moves=move,
        kwargs=dict(minimum_lines=True)
    )

    autocorr = sample_with_autocorr(sampler, start_points, iterations=1000, auto_sample=10)

    burn_in = 0
    blobs = sampler.get_blobs(discard=burn_in)
    blobs = blobs.flatten()
    blob_names = blobs[0].keys()
    blob_dict = {blob_name: xr.concat([data[blob_name] for data in blobs],
                                      dim=pd.Index(np.arange(0, blobs.__len__()), name="index")) for blob_name in
                 blob_names}
    samples = sampler.get_chain(flat=True)
    prior_samples = bm.sample_from_priors(param_names, size=int(1e5))

    # TODO make sure xrcs bckc doesn't have dims t and channels
    # save result
    result = {
        "blobs": blob_dict,
        "diag_data": flat_data,
        "samples": samples,
        "prior_samples": prior_samples,
        "param_names": param_names,
        "phantom_profiles": phantom_profiles,
        "plasma": plasma,
        "autocorr": autocorr,
    }
    plot_bayes_phantom(**result, figheader="./results/example/")
