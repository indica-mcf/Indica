import emcee
import numpy as np
import pandas as pd
import xarray as xr

from indica.bayesmodels import BayesModels
from indica.bayesmodels import get_uniform
from indica.models.diode_filters import BremsstrahlungDiode
from indica.models.plasma import Plasma
import indica.readers.read_st40 as read_st40
from indica.workflows.bayes_workflow import plot_bayes_result
from indica.workflows.bayes_workflow import sample_with_autocorr

# TODO: allow conditional prior usage even when only
#  one param is being optimisied i.e. 1 is constant


def bayesian_inference(
    pulse,
    phantom_profile_params,
    iterations,
    result_path,
    burn_in=0,
    tstart=0.02,
    tend=0.10,
    dt=0.01,
    time_to_calculate=0.04,
    nwalkers=10,
):
    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion="h",
        impurities=("c",),
        impurity_concentration=(0.2,),
        full_run=False,
        n_rad=10,
    )
    plasma.build_atomic_data()
    plasma.time_to_calculate = plasma.t.sel(t=time_to_calculate, method="nearest")

    ST40 = read_st40.ReadST40(pulse)
    ST40(["pi"])

    data_to_read = ST40.binned_data["pi"]["spectra"]
    los_transform = data_to_read.transform
    # data_to_read.transform.set_equilibrium(data_to_read.transform.equilibrium)
    pi = BremsstrahlungDiode(name="pi", channel_mask=slice(18, 28))
    pi.set_los_transform(los_transform)
    pi.plasma = plasma

    # Make phantom profiles
    plasma.update_profiles(phantom_profile_params)
    phantom_profiles = {
        "electron_density": plasma.Ne_prof.yspl,
        "electron_temperature": plasma.Te_prof.yspl,
        "ion_temperature": plasma.Ti_prof.yspl,
        "impurity_density": plasma.Nimp_prof.yspl,
    }

    pi_data = pi()["brightness"]

    # from indica.models.background_fit import Bremsstrahlung
    # pi_data = Bremsstrahlung(pulse)[1]
    # data_measured = Bremsstrahlung(pulse)[1].sel(channel=channels)
    # data_modelled = example_run(pulse)[2]["brightness"].sel(channel=channels)

    flat_data = {}
    flat_data["pi.brightness"] = pi_data.expand_dims(
        dim={"t": [plasma.time_to_calculate]}
    )

    priors = {
        "Ne_prof.y0": get_uniform(1e19, 8e19),
        "Ne_prof.y1": get_uniform(1e18, 5e18),
        "Ne_prof.y0/Ne_prof.y1": lambda x1, x2: np.where(((x1 > x2 * 2)), 1, 0),
        "Ne_prof.wped": get_uniform(1, 5),
        "Ne_prof.wcenter": get_uniform(0.1, 0.8),
        "Ne_prof.peaking": get_uniform(1, 5),
        "Nimp_prof.peaking": get_uniform(1, 8),
        "Nimp_prof.wcenter": get_uniform(0.1, 0.4),
        "Nimp_prof.y0": get_uniform(1e16, 5e18),
        "Nimp_prof.y1": get_uniform(1e15, 1e18),
        "Ne_prof.y0/Nimp_prof.y0": lambda x1, x2: np.where(
            (x1 > x2 * 100) & (x1 < x2 * 1e4), 1, 0
        ),
        "Nimp_prof.y0/Nimp_prof.y1": lambda x1, x2: np.where((x1 > x2), 1, 0),
        "Te_prof.y0": get_uniform(1000, 6000),
        "Te_prof.peaking": get_uniform(1, 4),
        "Ti_prof.y0": get_uniform(2000, 10000),
        "Ti_prof.peaking": get_uniform(1, 4),
    }

    param_names = [
        # "Ne_prof.y0",
        # "Ne_prof.y1",
        # "Ne_prof.peaking",
        "Nimp_prof.y0",
        "Nimp_prof.y1",
        "Nimp_prof.peaking",
        # "Te_prof.y0",
        #  "Te_prof.peaking",
        #  "Ti_prof.y0",
        # "Ti_prof.peaking",
    ]

    bm = BayesModels(
        plasma=plasma,
        data=flat_data,
        diagnostic_models=[pi],
        quant_to_optimise=[
            "pi.brightness",
        ],
        priors=priors,
    )
    ndim = param_names.__len__()
    start_points = bm.sample_from_priors(param_names, size=nwalkers)
    move = [(emcee.moves.StretchMove(), 1.0), (emcee.moves.DEMove(), 0.0)]
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_fn=bm.ln_posterior,
        parameter_names=param_names,
        moves=move,
        kwargs={"moment_analysis": False, "calc_spectra": True},
    )

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
    prior_samples = bm.sample_from_priors(param_names, size=int(1e5))
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
    print(sampler.acceptance_fraction.sum())
    plot_bayes_result(**result, figheader=result_path)


def run_bayesian_inference():
    phantom_profile_params = {
        "Ne_prof.y0": 5e19,
        "Ne_prof.wcenter": 0.4,
        "Ne_prof.peaking": 2,
        "Ne_prof.y1": 2e18,
        "Ne_prof.yend": 1e18,
        "Ne_prof.wped": 2,
        "Nimp_prof.y0": 1e18,
        "Nimp_prof.y1": 1e17,
        "Nimp_prof.peaking": 7,
        "Te_prof.y0": 3000,
        "Te_prof.peaking": 2,
        "Ti_prof.y0": 5000,
        "Ti_prof.peaking": 2,
    }
    nwalkers = 10  # 50 #
    iterations = 50  # 200 #

    pathname = "./plots/"

    bayesian_inference(
        10607,
        phantom_profile_params,
        iterations,
        pathname,
        burn_in=0,
        nwalkers=nwalkers,
    )


if __name__ == "__main__":
    run_bayesian_inference()
