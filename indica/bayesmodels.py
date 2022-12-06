import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as stats
from copy import deepcopy

from indica.readers.read_st40 import ST40data
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates

from indica.models.plasma import Plasma
from indica.models.interferometry import Interferometry

from indica.readers.manage_data import bin_data_in_time

import emcee
np.seterr(divide="ignore")


def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)


def uniform(x, lower, upper):
    if (x > lower) & (x < upper):
        return 1
    else:
        return 0


class BayesModels:
    """
    Class which operates with Plasma class to create ln_posterior method

    Parameters
    ----------
    plasma
        Plasma object needed for the optimisation
    data
        processed diagnostic data of format [diagnostic]_[quantity]
    quant_to_optimise
        quantity from data which will be optimised with bkcv from diagnostic_models
    priors
        prior functions to apply to parameters for ln_posterior
    diagnostic_models
        model objects to be called inside of ln_posterior
    """
    def __init__(self,
                 plasma=None,
                 data: dict = {},
                 quant_to_optimise: list = [],
                 priors: dict = {},
                 diagnostic_models: list = [],
                 ):

        self.plasma = plasma
        self.data = data
        self.quant_to_optimise = quant_to_optimise
        self.diagnostic_models = diagnostic_models
        self.priors = priors

        for diag_model in self.diagnostic_models:
            diag_model.plasma = self.plasma

    def _build_bkcv(self, params={}):
        self.bkcv = {}
        for model in self.diagnostic_models:
            self.bkcv = dict(self.bkcv, **model(params=params))
        return

    def _ln_likelihood(self):
        ln_likelihood = 0
        for key in self.quant_to_optimise:
            # TODO: What to use as error?  std as placeholder
            ln_likelihood += np.log(gaussian(self.bkcv[key], self.data[key].sel(t=self.plasma.time_to_calculate),
                                             self.data[key].stdev.sel(t=self.plasma.time_to_calculate) * 1e2))
        return ln_likelihood

    def _ln_prior(self, parameters: dict):
        ln_prior = 0
        for param_name, param_value in parameters.items():
            if param_name in self.priors:  # if no prior is defined then ignore
                ln_prior += np.log(self.priors[param_name](param_value))
            else:
                print(f"No prior assigned for {param_name}")
        return ln_prior

    def ln_posterior(self, parameters: dict):
        """
        Posterior probability given to optimisers

        Parameters
        ----------
        parameters
            inputs to optimise
        Returns
        -------
        ln_posterior
            log of posterior probability
        blob
            model outputs from bkcv and kinetic profiles
        """
        plasma.update_profiles(parameters)
        self._build_bkcv(parameters)  # model calls
        ln_likelihood = self._ln_likelihood()  # compare results to data
        ln_prior = self._ln_prior(parameters)
        ln_posterior = ln_likelihood + ln_prior

        kin_profs = {"electron_density": self.plasma.electron_density.sel(t=self.plasma.time_to_calculate),
                     }
        blob = deepcopy({**self.bkcv, **kin_profs})
        return ln_posterior, blob


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
        impurities=("c", "ar", "he"),
        impurity_concentration=(0.03, 0.001, 0.01),
        full_run=False,
    )
    plasma.time_to_calculate = plasma.t[1]
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
    data["smmh1"] = bin_data_in_time(
        raw_data["smmh1"], plasma.tstart, plasma.tend, plasma.dt,
    )
    transform = data["smmh1"]["ne"].attrs["transform"]
    data = data["smmh1"]
    # TODO: Get data as flat dict
    # Initialise Diagnostic Models
    smmh1 = Interferometry(name="smmh1")
    smmh1.set_transform(transform)
    smmh1.set_flux_transform(flux_transform)

    priors = {
        "Ne_prof_y0": lambda x: gaussian(x, 5e19, 5e19) *
                                uniform(x, 1e18, 5e20),
        "Ne_prof_peaking": lambda x: gaussian(x, 2, 3) *
                                     uniform(x, 0, 10),
        "Ne_prof_wcenter": lambda x: gaussian(x, 0.4, 0.2) *
                                     uniform(x, 0.1, 0.9),
        "Ne_prof_y1": lambda x: gaussian(x, 1e19, 1e19) *
                                uniform(x, 1e16, 1e20),
    }

    bm = BayesModels(plasma=plasma, data=data, diagnostic_models=[smmh1],
                     quant_to_optimise=["ne"], priors=priors)

    # ln_post, blob = bm.ln_posterior(parameters={"Ne_prof_y0":5e19})
    # print(ln_post,)

    params_names = ["Ne_prof_y0",
                    "Ne_prof_peaking",
                    "Ne_prof_wcenter",
                    "Ne_prof_y1"
                    ]
    nwalk = 8
    y0 = np.random.normal(5e19, 1e19, nwalk).reshape((nwalk, 1,))
    peaking = np.random.normal(3, 1, nwalk).reshape((nwalk, 1,))
    wcenter = np.random.normal(0.4, 0.1, nwalk).reshape((nwalk, 1,))
    y1 = np.random.normal(1e19, 1e18, nwalk).reshape((nwalk, 1,))
    start_points = np.concatenate([y0,
                                   peaking,
                                   wcenter,
                                   y1
                                   ], axis=1)

    nwalkers, ndim = start_points.shape

    move = [emcee.moves.StretchMove()]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn=bm.ln_posterior, parameter_names=params_names,
                                    moves=move)
    sampler.run_mcmc(start_points, 1000, progress=True)

    blobs = sampler.get_blobs()
    blobs = blobs.flatten()

    # ------------- plotting --------------
    ne_data = np.array([data["ne"].values for data in blobs])
    ne_data_std = np.std(ne_data)
    plt.ylabel("smmh1_ne_int (m^-2)")
    plt.plot(ne_data, )
    plt.axhline(y=data["ne"].sel(t=plasma.time_to_calculate).values, color="red", linestyle="-")

    plt.figure()

    ne_prof = xr.DataArray([data["electron_density"] for data in blobs])
    plt.errorbar(ne_prof.dim_1, ne_prof.mean("dim_0"), yerr=ne_prof.std("dim_0"))

    samples = sampler.get_chain(flat=True)

    import corner
    fig = corner.corner(samples, labels=params_names)

    print(sampler.acceptance_fraction)
    print(np.mean(sampler.get_autocorr_time(quiet=True)))
    plt.show(block=True)
