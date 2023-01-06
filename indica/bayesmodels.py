from copy import deepcopy
import pickle

import corner
import emcee
import flatdict
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.models.interferometry import Interferometry
from indica.models.plasma import Plasma
from indica.readers.manage_data import bin_data_in_time
from indica.readers.read_st40 import ST40data

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
        quantity from data which will be optimised with bckc from diagnostic_models
    priors
        prior functions to apply to parameters for ln_posterior
    diagnostic_models
        model objects to be called inside of ln_posterior
    """

    def __init__(
        self,
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

        missing_data = list(
            set(quant_to_optimise).difference(data.keys())
        )  # list of keys in quant_to_optimise but not data
        if missing_data:
            raise ValueError(f"{missing_data} not found in data given")

    def _outside_bounds(self, parameters):
        for param_name, param_value in parameters.items():
            if param_name in self.priors:  # if no prior is defined then ignore
                prior = np.log(self.priors[param_name](param_value))
                if prior == -np.inf:
                    return True
        return False

    def _build_bckc(self, params={}):
        self.bckc = {}
        for model in self.diagnostic_models:
            self.bckc = dict(self.bckc, **{model.name:{**model(params=params)}})
        self.bckc = flatdict.FlatDict(self.bckc, delimiter=".")
        return

    def _ln_likelihood(self):
        ln_likelihood = 0
        for key in self.quant_to_optimise:
            # TODO: What to use as error?  Assume percentage error if none given...
            ln_likelihood += np.log(
                gaussian(
                    self.bckc[key].values,
                    self.data[key].sel(t=self.plasma.time_to_calculate).values,
                    self.data[key].sel(t=self.plasma.time_to_calculate).values * 0.10,
                )
            )
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
            model outputs from bckc and kinetic profiles
        """
        outside_bounds = self._outside_bounds(parameters)
        if outside_bounds:
            return -np.inf, {}

        plasma.update_profiles(parameters)
        self._build_bckc(parameters)  # model calls
        ln_likelihood = self._ln_likelihood()  # compare results to data
        ln_prior = self._ln_prior(parameters)
        ln_posterior = ln_likelihood + ln_prior

        # Add better way of handling time array
        kin_profs = {
            "electron_density": self.plasma.electron_density.sel(
                t=self.plasma.time_to_calculate
            ),
            "electron_temperature": self.plasma.electron_temperature.sel(
                t=self.plasma.time_to_calculate
            ),
            "ion_temperature": self.plasma.ion_temperature.sel(
                t=self.plasma.time_to_calculate
            ),
            "impurity_density": self.plasma.impurity_density.sel(
                t=self.plasma.time_to_calculate
            ),
            # TODO: add Ni/Nh/Nimp when fz property works 1 timepoint
        }
        blob = deepcopy({**self.bckc, **kin_profs})
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

    priors = {
        "Ne_prof_y0": lambda x:
        # gaussian(x, 5e19, 5e19) *
        uniform(x, 1e19, 5e20),
        "Ne_prof_peaking": lambda x: uniform(x, 1, 5),
        "Ne_prof_wcenter": lambda x: uniform(x, 0.1, 0.9),
        "Ne_prof_y1": lambda x: uniform(x, 0, 1e20),
        "Nimp_prof_y0": lambda x: gaussian(x, 1e16, 1e15) * uniform(x, 1e15, 1e17),
        "Nimp_prof_peaking": lambda x: uniform(x, 1, 5),
        "Nimp_prof_y1": lambda x: uniform(x, 1e15, 1e17),
        "Te_prof_y0": lambda x: uniform(x, 1000, 1e4),
        "Te_prof_peaking": lambda x: uniform(x, 1, 5),
        "Ti_prof_y0": lambda x: uniform(x, 1000, 2e4),
        "Ti_prof_peaking": lambda x: uniform(x, 1, 5),
    }

    bm = BayesModels(
        plasma=plasma,
        data=flat_data,
        diagnostic_models=[smmh1],
        quant_to_optimise=[
            "smmh1.ne",
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

    Ne_y0 = np.random.normal(
        5e19,
        1e19,
        size=(
            nwalk,
            1,
        ),
    )
    Ne_peaking = np.random.normal(
        3,
        0.5,
        size=(
            nwalk,
            1,
        ),
    )
    Ne_y1 = np.random.normal(
        1e19,
        1e18,
        size=(
            nwalk,
            1,
        ),
    )
    Nimp_y0 = np.random.normal(
        1e16,
        1e15,
        size=(
            nwalk,
            1,
        ),
    )
    Nimp_peaking = np.random.normal(
        3,
        0.5,
        size=(
            nwalk,
            1,
        ),
    )
    Nimp_y1 = np.random.normal(
        1e16,
        1e15,
        size=(
            nwalk,
            1,
        ),
    )
    Te_y0 = np.random.normal(
        2e3,
        2e2,
        size=(
            nwalk,
            1,
        ),
    )
    Te_peaking = np.random.normal(
        3,
        0.5,
        size=(
            nwalk,
            1,
        ),
    )
    Ti_y0 = np.random.normal(
        6e3,
        2e3,
        size=(
            nwalk,
            1,
        ),
    )
    Ti_peaking = np.random.normal(
        3,
        0.5,
        size=(
            nwalk,
            1,
        ),
    )

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
    )
    sampler.run_mcmc(start_points, 10, progress=True)

    blobs = sampler.get_blobs()
    blobs = blobs.flatten()

    # save result
    with open("bayesresult.pkl", "wb") as handle:
        pickle.dump(
            {
                "blobs": blobs,
                "flat_data": flat_data,
                "samples": sampler.get_chain(flat=True),
                "param_names": params_names,
            },
            handle,
        )

    # ------------- plotting --------------
    plt.figure()
    ne_data = np.array([data["smmh1.ne"].values for data in blobs])
    plt.ylabel("smmh1_ne_int (m^-2)")
    plt.plot(
        ne_data,
    )
    plt.axhline(
        y=flat_data["smmh1.ne"].sel(t=plasma.time_to_calculate).values,
        color="red",
        linestyle="-",
    )
    # plt.figure()
    # Te_data = np.array([data["xrcs.te_kw"].values for data in blobs])
    # plt.ylabel("electron temperature (eV)")
    # plt.plot(
    #     Te_data,
    # )
    # plt.axhline(
    #     y=flat_data["xrcs.te_kw"].sel(t=plasma.time_to_calculate).values,
    #     color="red",
    #     linestyle="-",
    # )
    # plt.figure()
    # Ti_data = np.array([data["xrcs.ti_w"].values for data in blobs])
    # plt.ylabel("ion temperature (eV)")
    # plt.plot(
    #     Ti_data,
    # )
    # plt.axhline(
    #     y=flat_data["xrcs.ti_w"].sel(t=plasma.time_to_calculate).values,
    #     color="red",
    #     linestyle="-",
    # )

    plt.figure()
    ne_prof = xr.DataArray([data["electron_density"] for data in blobs])
    plt.errorbar(
        ne_prof.dim_1, ne_prof.mean("dim_0"), yerr=ne_prof.std("dim_0"), label="ne"
    )
    plt.legend()
    plt.figure()
    Te_prof = xr.DataArray([data["electron_temperature"] for data in blobs])
    plt.errorbar(
        Te_prof.dim_1, Te_prof.mean("dim_0"), yerr=Te_prof.std("dim_0"), label="te"
    )
    Ti_prof = xr.DataArray([data["ion_temperature"] for data in blobs])
    plt.errorbar(
        Ti_prof.dim_2,
        Ti_prof.mean(["dim_0", "dim_1"]),
        yerr=Ti_prof.std(["dim_0", "dim_1"]),
        label="ti",
    )
    plt.legend()

    plt.figure()
    nimp_prof = xr.DataArray([data["impurity_density"] for data in blobs])
    plt.errorbar(
        nimp_prof.dim_2,
        nimp_prof.mean(["dim_0", "dim_1"]),
        yerr=nimp_prof.std(["dim_0", "dim_1"]),
        label="NAr",
    )
    plt.legend()

    samples = sampler.get_chain(flat=True)
    fig = corner.corner(samples, labels=params_names)

    print(sampler.acceptance_fraction)
    print(np.mean(sampler.get_autocorr_time(quiet=True)))
    plt.show(block=True)
