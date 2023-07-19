import emcee
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

from indica.bayesmodels import BayesModels
from indica.bayesmodels import get_uniform
from indica.equilibrium import Equilibrium
from indica.models.diode_filters import example_run as example_diode
from indica.models.plasma import example_run as example_plasma
from indica.operators import tomo_1D
import indica.physics as ph
from indica.readers.read_st40 import ReadST40
from indica.workflows.bayes_workflow import plot_bayes_result
from indica.workflows.bayes_workflow import sample_with_autocorr

PATHNAME = "./plots/"

MAIN_ION = "h"
IMPURITIES = ("c",)
IMPURITY_CONCENTRATION = (0.03,)
FULL_RUN = False
N_RAD = 10

PATHNAME = "./plots/"

PRIORS = {
    "Ne_prof.y0": get_uniform(1e19, 8e19),
    "Ne_prof.y1": get_uniform(1e18, 5e18),
    "Ne_prof.y0/Ne_prof.y1": lambda x1, x2: np.where(((x1 > x2 * 2)), 1, 0),
    "Ne_prof.wped": get_uniform(1, 5),
    "Ne_prof.wcenter": get_uniform(0.1, 0.8),
    "Ne_prof.peaking": get_uniform(1, 5),
    "Nimp_prof.peaking": get_uniform(1, 8),
    "Nimp_prof.wcenter": get_uniform(0.1, 0.4),
    "Nimp_prof.y0": get_uniform(1e16, 5e18),
    "Nimp_prof.y1": get_uniform(1e16, 5e18),
    "Ne_prof.y0/Nimp_prof.y0": lambda x1, x2: np.where(
        (x1 > x2 * 100) & (x1 < x2 * 1e4), 1, 0
    ),
    "Nimp_prof.y0/Nimp_prof.y1": lambda x1, x2: np.where(
        (x1 >= x2) & (x1 < x2 * 5), 1, 0
    ),
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
    "Nimp_prof.y0": 1e18,
    "Nimp_prof.y1": 1e17,
    "Nimp_prof.peaking": 7,
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


def prepare_data(pulse, plasma, model, phantom_data: bool = True):
    if pulse is not None:
        st40 = ReadST40(pulse, tstart=plasma.tstart, tend=plasma.tend, dt=plasma.dt)
        st40(["pi", "efit"])
        attrs = st40.binned_data["pi"]["spectra"].attrs
        (
            st40.binned_data["pi"]["background"],
            st40.binned_data["pi"]["brightness"],
        ) = model.integrate_spectra(st40.binned_data["pi"]["spectra"])
        st40.binned_data["pi"]["background"].attrs = attrs
        st40.binned_data["pi"]["brightness"].attrs = attrs

        plasma.initialize_variables()
        plasma.set_equilibrium(Equilibrium(st40.raw_data["efit"]))

        model.set_los_transform(st40.binned_data["pi"]["spectra"].transform)
        model.set_plasma(plasma)

        data = st40.binned_data["pi"]["brightness"].sel(t=plasma.time_to_calculate)

    if phantom_data:
        data = model()["brightness"]

    return data


def run_bayesian_analysis(
    pulse,
    phantom_profile_params,
    iterations,
    result_path,
    tstart=0.01,
    tend=0.1,
    dt=0.01,
    burn_in=0,
    tsample=3,
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
    plasma.time_to_calculate = plasma.t[tsample]
    plasma.update_profiles(phantom_profile_params)

    print("Generating model")
    _, pi_model, bckc = example_diode(plasma=plasma)
    pi_model.name = "pi"

    print("Preparing data")
    data = prepare_data(pulse, plasma, pi_model, phantom_data=phantom_data)

    phantom_profiles = {
        "electron_density": plasma.electron_density.sel(t=plasma.time_to_calculate),
        "electron_temperature": plasma.electron_temperature.sel(
            t=plasma.time_to_calculate
        ),
        "ion_temperature": plasma.ion_temperature.sel(
            t=plasma.time_to_calculate, element=IMPURITIES[0]
        ),
        "impurity_density": plasma.impurity_density.sel(
            t=plasma.time_to_calculate, element=IMPURITIES[0]
        ),
        "zeff": plasma.zeff.sel(t=plasma.time_to_calculate, element=IMPURITIES[0]),
    }

    flat_data = {}
    flat_data["pi.brightness"] = data.expand_dims(dim={"t": [plasma.time_to_calculate]})

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


def bayes(pulse: int = 10821, nwalkers: int = 50, iterations: int = 200):
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
    ff = run_bayesian_analysis(
        pulse,
        phantom_profile_params,
        iterations,
        PATHNAME,
        burn_in=0,
        nwalkers=nwalkers,
    )

    return ff


def inversion(
    pulse,
    tstart=0.01,
    tend=0.1,
    dt=0.01,
    reg_level_guess: float = 0.3,
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

    print("Generating model")
    _, pi_model, bckc = example_diode(plasma=plasma)
    pi_model.name = "pi"

    print("Preparing data")
    data = prepare_data(pulse, plasma, pi_model, phantom_data=phantom_data)
    if phantom_data:
        emissivity = pi_model.emissivity
    else:
        emissivity = None

    los_transform = data.transform
    equilibrium = los_transform.equilibrium
    z = los_transform.z
    R = los_transform.R
    dl = los_transform.dl

    data_t0 = data.isel(t=0).data
    has_data = np.logical_not(np.isnan(data.isel(t=0).data)) & (data_t0 > 0)
    rho_equil = equilibrium.rho.interp(t=data.t)
    input_dict = dict(
        brightness=data.data,
        dl=dl,
        t=data.t.data,
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
    if emissivity is not None:
        input_dict["emissivity"] = emissivity

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
    data_tomo = data
    # bckc_tomo = DataArray(tomo.backprojection, coords=data_tomo.coords)

    zeff = ph.zeff_bremsstrahlung(
        plasma.electron_temperature,
        plasma.electron_density,
        pi_model.filter_wavelength,
        bremsstrahlung=inverted_emissivity,
        gaunt_approx="callahan",
    )

    plt.figure()
    plasma.zeff.sum("element").sel(t=0.03).plot(label="Phantom")
    zeff.sel(t=0.03).plot(marker="o", label="Recalculated")
    plt.ylabel("Zeff")
    plt.legend()

if __name__ == "__main__":
    bayes()
