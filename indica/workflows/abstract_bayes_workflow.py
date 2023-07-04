from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import pickle

class AbstractBayesWorkflow(ABC):
    @abstractmethod
    def __init__(self,
                 phantoms = None,
                 diagnostics = None,
                 param_names=None,
                 opt_quantity=None,
                 priors=None,
                 ):

        self.phantoms = phantoms
        self.diagnostics = diagnostics
        self.param_names = param_names
        self.opt_quantity = opt_quantity
        self.priors = priors

        self.setup_plasma()
        self.save_phantom_profiles()
        self.read_data(self.diagnostics)
        self.setup_opt_data(self.phantoms)
        self.setup_models(self.diagnostics)
        self.setup_optimiser()

    @abstractmethod
    def setup_plasma(self):
        """
        Contains all methods and settings for plasma object to be used in optimisation
        """
        self.plasma = None

    @abstractmethod
    def read_data(self, diagnostics: list):
        """
        Reads data from server

        Returns

        nested dictionary of data
        """
        self.data = {}


    @abstractmethod
    def setup_opt_data(self, phantom: bool = False):
        """
        Prepare the data in necessary format for optimiser i.e. flat dictionary
        """
        self.opt_data = {}

    @abstractmethod
    def setup_models(self, diagnostics: list):
        """
        Initialising models normally requires data to be read so transforms can be set

        """
        self.models = {}

    @abstractmethod
    def setup_optimiser(self):
        """
        Initialise and provide settings for optimiser
        """
        self.bayesopt = None

    def save_phantom_profiles(self, kinetic_profiles=None):
        if kinetic_profiles is None:
            kinetic_profiles = ["electron_density", "impurity_density", "electron_temperature",
                                "ion_temperature", "ion_density", "fast_density", "neutral_density"]
        if self.phantoms:
            phantom_profiles = {profile_key: getattr(self.plasma, profile_key).sel(
                                    t=self.plasma.time_to_calculate).copy()
                                    for profile_key in kinetic_profiles}
        else:
            phantom_profiles = {profile_key: getattr(self.plasma, profile_key).sel(
                                    t=self.plasma.time_to_calculate) * 0
                                    for profile_key in kinetic_profiles}

            self.phantom_profiles = phantom_profiles

    def _build_result_dict(self):

        """
        TODO: xarray to numpy for MDSPlus writing (plotting / dimensions / units have to be figured out)


        """


        result = {}
        result["TIME"] = self.plasma.t
        result["TIME_OPT"] = self.plasma.time_to_calculate

        result["INPUT"] = {
            "BURN_FRAC": self.burn_frac,
            "ITER": self.iterations,
            "MODEL_KWARGS": "KWARGS",
            "OPT_KWARGS": "KWARGS",
            "NWALKERS": self.nwalkers,
            "PARAM_NAMES": self.param_names,
            "PULSE": self.pulse,
        }

        result["GLOBAL"] = {
            "TI0": 0,
            "TE0": 0,
            "NE0": 0,
            "NI0": 0,
            "TI0_ERR": 0,
            "TE0_ERR": 0,
            "NE0_ERR": 0,
            "NI0_ERR": 0,
        }

        result["PHANTOMS"] = {
            "FLAG": self.phantoms,
            "NE": self.phantom_profiles["electron_density"],
            "TE": self.phantom_profiles["electron_temperature"],
            "TI": self.phantom_profiles["ion_temperature"],
            "NI": self.phantom_profiles["ion_density"],
            "NNEUTR": self.phantom_profiles["neutral_density"],
            "NFAST": self.phantom_profiles["fast_density"],
            "NIMP1": self.phantom_profiles["impurity_density"].sel(element="ar"),  # TODO: generalise
            "NIMP2": self.phantom_profiles["impurity_density"].sel(element="c")
        }

        result["PROFILE_STAT"] = {
            "SAMPLES": self.samples,
            "RHO_POLOIDAL": self.plasma.rho,
            "NE": self.blobs["electron_density"],
            "NI": self.blobs["ion_density"],
            "TE": self.blobs["electron_temperature"],
            "TI": self.blobs["ion_temperature"],
            "NFAST": self.blobs["fast_density"],
            "NNEUTR": self.blobs["neutral_density"],
            "NIMP1": self.blobs["impurity_density"].sel(element="ar"),  # TODO: generalise
            "NIMP2": self.blobs["impurity_density"].sel(element="c")
        }

        result["OPTIMISATION"] = {
            "OPT_QUANTITY": self.opt_quantity,
            "ACCEPT_FRAC": self.accept_frac,
            "PRIOR_SAMPLE": self.prior_sample,
            "POST_SAMPLE": self.post_sample,
            "AUTOCORR": self.autocorr,
        }

        quant_list = [item.split(".") for item in self.opt_quantity]
        result["MODEL_DATA"] = {diag_name.upper():
                                    {quantity[1].upper(): self.blobs[f"{quantity[0]}.{quantity[1]}"]
                                     for quantity in quant_list if quantity[0] == diag_name}
                                for diag_name in self.diagnostics
                                }
        result["MODEL_DATA"]["SAMPLES"] = self.samples

        result["DIAG_DATA"] = {diag_name.upper():
                                   {quantity[1].upper(): self.opt_data[f"{quantity[0]}.{quantity[1]}"]
                                    for quantity in quant_list if quantity[0] == diag_name}
                               for diag_name in self.diagnostics
                               }
        self.result = result
        return self.result

    def run_sampler(self):

        """
        TODO: unsure if keeping in abstract class is best practice

        Runs the sampler and saves certain attributes from the sampler

        Returns

        result in MDSPlus node formatting

        """
        self.autocorr = sample_with_autocorr(
            self.sampler, self.start_points, self.iterations, self.param_names.__len__(), auto_sample=10
        )
        blobs = self.sampler.get_blobs(
            discard=int(self.iterations * self.burn_frac), flat=True
        )
        blob_names = self.sampler.get_blobs().flatten()[0].keys()
        self.samples = np.arange(0, blobs.__len__())

        self.blobs = {
            blob_name: xr.concat(
                [data[blob_name] for data in blobs],
                dim=pd.Index(self.samples, name="index"),
            )
            for blob_name in blob_names
        }
        self.accept_frac = self.sampler.acceptance_fraction.sum()
        self.prior_sample = self.bayesopt.sample_from_priors(
            self.param_names, size=int(1e4)
        )
        self.post_sample = self.sampler.get_chain(flat=True)
        self.result = self._build_result_dict()


    def save_pickle(self, filepath):
        if filepath:
            Path(filepath).mkdir(parents=True, exist_ok=True)
            with open(filepath + "results.pkl", "wb") as handle:
                pickle.dump(self.result, handle)

    @abstractmethod
    def __call__(self, filepath="./results/test/", **kwargs):
        self.run_sampler()
        self.save_pickle(filepath=filepath)
        return self.result


def sample_with_autocorr(sampler, start_points, iterations, n_params, auto_sample=5):
    autocorr = np.ones(shape=(iterations, n_params)) * np.nan
    old_tau = np.inf
    for sample in sampler.sample(
        start_points,
        iterations=iterations,
        progress=True,
    ):
        if sampler.iteration % auto_sample:
            continue
        new_tau = sampler.get_autocorr_time(tol=0)
        autocorr[sampler.iteration - 1, ] = new_tau
        converged = np.all(new_tau * 50 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - new_tau) / new_tau < 0.01)
        if converged:
            break
        old_tau = new_tau
    autocorr = autocorr[: sampler.iteration, ]
    return autocorr