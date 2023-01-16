from copy import deepcopy
import pickle

import corner
import emcee
import flatdict
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.bayesmodels import uniform, gaussian, BayesModels
from indica.models.interferometry import Interferometry
from indica.models.helike_spectroscopy import Helike_spectroscopy
from indica.models.plasma import Plasma
from indica.readers.manage_data import bin_data_in_time
from indica.readers.read_st40 import ST40data



if __name__ == "__main__":
    # First example to optimise the ne_int for the smm_interferom
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
        impurity_concentration=(0.001, ),
        full_run=False,
    )

    plasma.assign_profiles("electron_density", plasma.t)
    plasma.assign_profiles("electron_temperature", plasma.t)
    plasma.assign_profiles("impurity_density", plasma.t)
    plasma.assign_profiles("ion_temperature", plasma.t)
    plasma.build_atomic_data()
    plasma.time_to_calculate = plasma.t[3]

    # Initialise Data
    raw = ST40data(9229, tstart - dt * 4, tend + dt * 4)
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

    # Initialise Diagnostic Models
    los_transform = flat_data["smmh1.ne"].transform
    smmh1 = Interferometry(name="smmh1", )
    smmh1.set_los_transform(los_transform)
    smmh1.plasma=plasma
    los_transform = flat_data["xrcs.te_kw"].transform
    xrcs = Helike_spectroscopy(name="xrcs", )
    xrcs.plasma = plasma
    xrcs.set_los_transform(los_transform)

    # Make phantom profiles
    phantom_profiles = {"electron_density":plasma.Ne_prof.yspl,
                        "electron_temperature":plasma.Te_prof.yspl,
                        "ion_temperature":plasma.Ti_prof.yspl,
                        "impurity_density":plasma.Nimp_prof.yspl}


    # Use phantom data instead of ST40 data
    flat_data["smmh1.ne"] = smmh1().pop("ne").expand_dims(dim={"t":[plasma.time_to_calculate]})
    flat_data["xrcs.te_kw"] = xrcs().pop("te_kw")
    flat_data["xrcs.ti_w"] = xrcs().pop("ti_w")
    # TODO: Add conditional priors e.g. y1 < y0
    priors = {
        "Ne_prof_y0": lambda x: uniform(x, 5e18, 1e20),
        "Ne_prof_y1": lambda x: uniform(x, 1e17, 1e19),
        "Ne_prof_peaking": lambda x: uniform(x, 2, 4),
        "Ne_prof_wcenter": lambda x: uniform(x, 0.1, 0.9),

        "Nimp_prof_y0": lambda x: uniform(x, 5e14, 0.33e19),
        "Nimp_prof_peaking": lambda x: uniform(x, 2, 8),
        "Nimp_prof_y1": lambda x: uniform(x, 5e13, 1e18),

        "Te_prof_y0": lambda x: uniform(x, 500, 1.3e4),
        "Te_prof_peaking": lambda x: uniform(x, 1, 5),

        "Ti_prof_y0": lambda x: uniform(x, 1000, 2e4),
        "Ti_prof_peaking": lambda x: uniform(x, 1, 5),
    }

    bm = BayesModels(
        plasma=plasma,
        data=flat_data,
        diagnostic_models=[smmh1, xrcs],
        quant_to_optimise=[
            "smmh1.ne",
            "xrcs.te_kw",
            "xrcs.ti_w",
        ],
        priors=priors,
    )

    # Setup Optimiser
    params_names = [
        "Ne_prof_y0",
        "Ne_prof_peaking",
        "Ne_prof_y1",
        "Nimp_prof_y0",
        "Nimp_prof_peaking",
        "Nimp_prof_y1",
        "Te_prof_y0",
        "Te_prof_peaking",
        "Ti_prof_y0",
        "Ti_prof_peaking",
    ]
    nwalk = 20

    Ne_y0 = np.random.normal(5e19, 1e18, size=(nwalk, 1,), )
    Ne_peaking = np.random.normal(3, 0.2, size=(nwalk, 1, ), )
    Ne_y1 = np.random.normal(1e18, 1e17, size=(nwalk, 1,), )

    Nimp_y0 = np.random.normal(5e16, 1e15, size=(nwalk, 1, ), )
    Nimp_peaking = np.random.normal(4, 0.5, size=(nwalk, 1, ), )
    Nimp_y1 = np.random.normal(1e16, 1e15, size=(nwalk, 1, ), )

    Te_y0 = np.random.normal(3e3, 1e3, size=(nwalk, 1, ), )
    Te_peaking = np.random.normal(3, 0.5, size=(nwalk, 1, ), )

    Ti_y0 = np.random.normal(6e3, 1e3, size=(nwalk, 1, ), )
    Ti_peaking = np.random.normal(3, 0.5, size=(nwalk, 1, ), )

    start_points = np.concatenate(
        [
            Ne_y0,
            Ne_peaking,
            Ne_y1,
            Nimp_y0,
            Nimp_peaking,
            Nimp_y1,
            Te_y0,
            Te_peaking,
            Ti_y0,
            Ti_peaking,
        ],
        axis=1,
    )

    nwalkers, ndim = start_points.shape

    move = [emcee.moves.StretchMove()]
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_fn=bm.ln_posterior,
        parameter_names=params_names,
        moves=move,
        kwargs=dict(minimal_lines=True)
    )

    iterations = 10
    sampler.run_mcmc(start_points, iterations, progress=True, )
    burn_in = 0

    blobs = sampler.get_blobs(discard=burn_in)
    blobs = blobs.flatten()
    blob_names = blobs[0].keys()
    blob_dict = {blob_name: xr.concat([data[blob_name] for data in blobs],
                                      dim=pd.Index(np.arange(0, blobs.__len__()), name="index")) for blob_name in
                 blob_names}

    # TODO make sure xrcs bckc doesn't have dims t and channels
    # save result
    with open("bayesresult.pkl", "wb") as handle:
        pickle.dump(
            {
                "blobs": blob_dict,
                "diag_data": flat_data,
                "param_samples": sampler.get_chain(flat=True, discard=burn_in),
                "param_names": params_names,
                "phantom_profiles":phantom_profiles,
                "time":plasma.time_to_calculate,
            },
            handle,
        )

    # ------------- plotting --------------
    plt.figure()
    temp_data = blob_dict["smmh1.ne"]
    plt.ylabel("ne_int (m^-2)")
    plt.plot(temp_data, label="smmh1.ne_int model")
    plt.axhline(
        y=flat_data["smmh1.ne"].sel(t=plasma.time_to_calculate).values,
        color="red",
        linestyle="-",
        label="smmh1.ne_int data"
    )
    plt.legend()

    plt.figure()
    temp_data = blob_dict["xrcs.te_kw"][:,0,0]
    plt.ylabel("temperature (eV)")
    plt.plot(
        temp_data, label="xrcs.te_kw model", color="blue"
    )
    plt.axhline(
        y=flat_data["xrcs.te_kw"][0,].sel(t=plasma.time_to_calculate).values,
        color="blue",
        linestyle="-",
        label="xrcs.te_kw data",
    )
    temp_data = blob_dict["xrcs.ti_w"][:,0,0]
    plt.plot(
        temp_data, label="xrcs.ti_w model", color="red"
    )
    plt.axhline(
        y=flat_data["xrcs.ti_w"][0,].sel(t=plasma.time_to_calculate).values,
        color="red",
        linestyle="-",
        label="xrcs.ti_w data",
    )
    plt.legend()

    plt.figure()
    prof = blob_dict["electron_density"]
    plt.fill_between(
        prof.rho_poloidal, prof.quantile( 0.05, dim ="index"), prof.quantile( 0.95, dim ="index"),
                 label="Ne, 90% Confidence", zorder=2)
    if phantom_profiles:
        phantom_profiles["electron_density"].plot(label = "phantom_profile", linestyle="--", color="black")
    plt.legend()

    plt.figure()
    prof = blob_dict["electron_temperature"]
    plt.fill_between(
        prof.rho_poloidal, prof.quantile(0.05, dim="index"), prof.quantile(0.95, dim="index"),
        label="Te, 90% Confidence", color="blue", zorder=2, alpha=0.7)
    if phantom_profiles:
        phantom_profiles["electron_temperature"].plot(label = "Te, phantom_profile", linestyle="--", color="black")

    prof = blob_dict["ion_temperature"].sel(element="ar")
    plt.fill_between(
        prof.rho_poloidal, prof.quantile(0.05, dim="index"), prof.quantile(0.95, dim="index", ),
        label="Ti, 90% Confidence", color="red", zorder=2, alpha=0.7)
    if phantom_profiles:
        phantom_profiles["ion_temperature"].plot(label = "Ti, phantom_profile", linestyle="-.", color="black")
    plt.legend()

    plt.figure()
    prof = blob_dict["impurity_density"].sel(element="ar")
    plt.fill_between(
        prof.rho_poloidal, prof.quantile(0.05, dim="index"), prof.quantile(0.95, dim="index", ),
        label="Nimp, 90% Confidence", color="red")
    if phantom_profiles:
        phantom_profiles["impurity_density"].plot(label="phantom_profile", linestyle="--", color="black")
    plt.legend()

    samples = sampler.get_chain(flat=True)
    fig = corner.corner(samples, labels=params_names)

    print(sampler.acceptance_fraction)
    print(np.mean(sampler.get_autocorr_time(quiet=True)))
    plt.show(block=True)