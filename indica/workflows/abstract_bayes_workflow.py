from abc import ABC
from abc import abstractmethod
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from indica.equilibrium import fake_equilibrium
from indica.readers.read_st40 import ReadST40


class AbstractBayesWorkflow(ABC):
    @abstractmethod
    def __init__(
        self,
        pulse=None,
        phantoms=None,
        diagnostics=None,
        param_names=None,
        opt_quantity=None,
        priors=None,
    ):
        self.pulse = pulse
        self.phantoms = phantoms
        self.diagnostics = diagnostics
        self.param_names = param_names
        self.opt_quantity = opt_quantity
        self.priors = priors
        self.read_data(self.diagnostics)
        self.setup_models(self.diagnostics)

    def read_test_data(
        self, diagnostic_transforms: dict, tstart=None, tend=None, dt=None
    ):
        # Used with phantom data for purposes of tests
        print("Reading fake data")
        self.equilibrium = fake_equilibrium(
            tstart,
            tend,
            dt,
        )
        self.transforms = diagnostic_transforms
        self.data: dict = {}

    def read_data(self, diagnostics: list, tstart=None, tend=None, dt=None):
        self.reader = ReadST40(self.pulse, tstart=tstart, tend=tend, dt=dt)
        self.reader(diagnostics)

        missing_keys = set(diagnostics) - set(self.reader.binned_data.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"missing data: {missing_keys}")

        self.equilibrium = self.reader.equilibrium
        self.transforms = self.reader.transforms
        self.data = self.reader.binned_data

    @abstractmethod
    def setup_plasma(self):
        """
        Contains all methods and settings for setting up / initialising plasma object
        """
        self.plasma = None
        self.plasma.set_equilibrium(self.reader.equilibrium)
        self.save_phantom_profiles()

    @abstractmethod
    def setup_models(self, diagnostics: list):
        """
        Initialising models normally requires data to be read so transforms can be set

        """
        self.models: dict = {}

    @abstractmethod
    def _phantom_data(self):
        opt_data = {}
        return opt_data

    @abstractmethod
    def _exp_data(self):
        opt_data = {}
        return opt_data

    @abstractmethod
    def setup_opt_data(self, phantom: bool = False):
        """
        Get and prepare the data in necessary format for optimiser
        """
        for model in self.models:
            model.plasma = self.plasma

        if phantom:
            self.opt_data = self._phantom_data()
        else:
            self.opt_data = self._exp_data()

    @abstractmethod
    def setup_optimiser(self, model_kwargs):
        """
        Initialise and provide settings for optimiser
        """
        self.bayesopt = None

    def save_phantom_profiles(self, kinetic_profiles=None):
        if kinetic_profiles is None:
            kinetic_profiles = [
                "electron_density",
                "impurity_density",
                "electron_temperature",
                "ion_temperature",
                "ion_density",
                "fast_density",
                "neutral_density",
            ]
        if self.phantoms:
            phantom_profiles = {
                profile_key: getattr(self.plasma, profile_key)
                .sel(t=self.plasma.time_to_calculate)
                .copy()
                for profile_key in kinetic_profiles
            }
        else:
            phantom_profiles = {
                profile_key: getattr(self.plasma, profile_key).sel(
                    t=self.plasma.time_to_calculate
                )
                * 0
                for profile_key in kinetic_profiles
            }

        self.phantom_profiles = phantom_profiles

    def _build_result_dict(self):
        """

        Returns
        -------

        dictionary of results in MDS+ structure

        """

        result = {}
        quant_list = [item.split(".") for item in self.opt_quantity]

        result["TIME_BINS"] = self.plasma.t
        result["TIME"] = self.plasma.time_to_calculate

        result["METADATA"] = {
            "GITCOMMIT": "PLACEHOLDER",
            "USER": "PLACEHOLDER",
            "EQUIL": "PLACEHOLDER",
        }

        result["INPUT"] = {
            "BURN_FRAC": self.burn_frac,
            "ITER": self.iterations,
            "NWALKERS": self.nwalkers,
            "MODEL_KWARGS": self.model_kwargs,
            "OPT_QUANTITY": self.opt_quantity,
            "PARAM_NAMES": self.param_names,
            "PULSE": self.pulse,
            "IMPURITIES": self.plasma.impurities,
            "MAIN_ION": self.plasma.main_ion,
            "TSTART":self.tstart,
            "TEND": self.tend,
            "DT": self.dt,
            "TSAMPLE": self.tsample,

        }
        result["INPUT"]["WORKFLOW"] = {
            diag_name.upper(): {
                "PULSE": self.pulse,  # Change this if different pulses used
                "USAGE": "".join(
                    [quantity[1] for quantity in quant_list if quantity[0] == diag_name]
                ),
                "RUN": "PLACEHOLDER",
            }
            for diag_name in self.diagnostics
        }

        result["MODEL_DATA"] = {
            diag_name.upper(): {
                quantity[1].upper(): self.blobs[f"{quantity[0]}.{quantity[1]}"]
                for quantity in quant_list
                if quantity[0] == diag_name
            }
            for diag_name in self.diagnostics
        }
        result["MODEL_DATA"]["SAMPLES"] = self.samples

        result["DIAG_DATA"] = {
            diag_name.upper(): {
                quantity[1].upper(): self.opt_data[f"{quantity[0]}.{quantity[1]}"]
                for quantity in quant_list
                if quantity[0] == diag_name
            }
            for diag_name in self.diagnostics
        }

        result["PHANTOMS"] = {
            "FLAG": self.phantoms,
            "NE": self.phantom_profiles["electron_density"],
            "TE": self.phantom_profiles["electron_temperature"],
            "TI": self.phantom_profiles["ion_temperature"].sel(
                element=self.plasma.main_ion
            ),
            "NI": self.phantom_profiles["ion_density"].sel(
                element=self.plasma.main_ion
            ),
            "NNEUTR": self.phantom_profiles["neutral_density"],
            "NFAST": self.phantom_profiles["fast_density"],
        }
        result["PHANTOMS"].update(
            {
                f"NIZ{num_imp + 1}": self.phantom_profiles["impurity_density"].sel(
                    element=imp
                )
                for num_imp, imp in enumerate(self.plasma.impurities)
            }
        )
        result["PHANTOMS"].update(
            {
                f"TIZ{num_imp + 1}": self.phantom_profiles["ion_temperature"].sel(
                    element=imp
                )
                for num_imp, imp in enumerate(self.plasma.impurities)
            }
        )

        result["PROFILES"] = {
            "RHO_POLOIDAL": self.plasma.rho,
            "RHO_TOR": self.plasma.equilibrium.rhotor.interp(t=self.plasma.t),
            "NE": self.blobs["electron_density"].median(dim="index"),
            "NI": self.blobs["ion_density"]
            .sel(element=self.plasma.main_ion)
            .median(dim="index"),
            "TE": self.blobs["electron_temperature"].median(dim="index"),
            "TI": self.blobs["ion_temperature"]
            .sel(element=self.plasma.main_ion)
            .median(dim="index"),
            "NFAST": self.blobs["fast_density"].median(dim="index"),
            "NNEUTR": self.blobs["neutral_density"].median(dim="index"),
            "NE_ERR": self.blobs["electron_density"].std(dim="index"),
            "NI_ERR": self.blobs["ion_density"]
            .sel(element=self.plasma.main_ion)
            .std(dim="index"),
            "TE_ERR": self.blobs["electron_temperature"].std(dim="index"),
            "TI_ERR": self.blobs["ion_temperature"]
            .sel(element=self.plasma.main_ion)
            .std(dim="index"),
            "NFAST_ERR": self.blobs["fast_density"].std(dim="index"),
            "NNEUTR_ERR": self.blobs["neutral_density"].std(dim="index"),
            "ZEFF": self.blobs["zeff"].sum("element").median(dim="index"),
            "ZEFF_ERR": self.blobs["zeff"].sum("element").std(dim="index"),
        }
        result["PROFILES"] = {
            **result["PROFILES"],
            **{
                f"NIZ{num_imp + 1}": self.blobs["impurity_density"]
                .sel(element=imp)
                .median(dim="index")
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }
        result["PROFILES"] = {
            **result["PROFILES"],
            **{
                f"NIZ{num_imp + 1}_ERR": self.blobs["impurity_density"]
                .sel(element=imp)
                .std(dim="index")
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }
        result["PROFILES"] = {
            **result["PROFILES"],
            **{
                f"TIZ{num_imp + 1}": self.blobs["ion_temperature"]
                .sel(element=imp)
                .median(dim="index")
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }
        result["PROFILES"] = {
            **result["PROFILES"],
            **{
                f"TIZ{num_imp + 1}_ERR": self.blobs["ion_temperature"]
                .sel(element=imp)
                .std(dim="index")
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }

        result["PROFILE_STAT"] = {
            "SAMPLES": self.samples,
            "RHO_POLOIDAL": self.plasma.rho,
            "NE": self.blobs["electron_density"],
            "NI": self.blobs["ion_density"].sel(element=self.plasma.main_ion),
            "TE": self.blobs["electron_temperature"],
            "TI": self.blobs["ion_temperature"].sel(element=self.plasma.main_ion),
            "NFAST": self.blobs["fast_density"],
            "NNEUTR": self.blobs["neutral_density"],
        }
        result["PROFILE_STAT"] = {
            **result["PROFILE_STAT"],
            **{
                f"NIZ{num_imp + 1}": self.blobs["impurity_density"].sel(element=imp)
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }
        result["PROFILE_STAT"] = {
            **result["PROFILE_STAT"],
            **{
                f"TIZ{num_imp + 1}": self.blobs["ion_temperature"].sel(element=imp)
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }

        result["OPTIMISATION"] = {
            "ACCEPT_FRAC": self.accept_frac,
            "PRIOR_SAMPLE": self.prior_sample,
            "POST_SAMPLE": self.post_sample,
            "AUTOCORR": self.autocorr,
        }

        result["GLOBAL"] = {
            "TI0": self.blobs["ion_temperature"]
            .sel(element=self.plasma.main_ion)
            .sel(rho_poloidal=0, method="nearest")
            .median(dim="index"),
            "TE0": self.blobs["electron_temperature"]
            .sel(rho_poloidal=0, method="nearest")
            .median(dim="index"),
            "NE0": self.blobs["electron_density"]
            .sel(rho_poloidal=0, method="nearest")
            .median(dim="index"),
            "NI0": self.blobs["ion_density"]
            .sel(element=self.plasma.main_ion)
            .sel(rho_poloidal=0, method="nearest")
            .median(dim="index"),
            "WP": self.blobs["wp"]
            .median(dim="index"),
            "WP_ERR": self.blobs["wp"]
            .std(dim="index"),
            "WTH": self.blobs["wth"]
            .median(dim="index"),
            "WTH_ERR": self.blobs["wth"]
            .std(dim="index"),
            "PTOT": self.blobs["ptot"]
                .median(dim="index"),
            "PTOT_ERR": self.blobs["ptot"]
                .std(dim="index"),
            "PTH": self.blobs["pth"]
                .median(dim="index"),
            "PTH_ERR": self.blobs["pth"]
                .std(dim="index"),

        }
        result["GLOBAL"] = {
            **result["GLOBAL"],
            **{
                f"TI0Z{num_imp + 1}": self.blobs["ion_temperature"]
                .sel(element=imp)
                .sel(rho_poloidal=0, method="nearest")
                .median(dim="index")
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }
        result["GLOBAL"] = {
            **result["GLOBAL"],
            **{
                f"TI0Z{num_imp + 1}_ERR": self.blobs["ion_temperature"]
                .sel(element=imp)
                .sel(rho_poloidal=0, method="nearest")
                .std(dim="index")
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }
        result["GLOBAL"] = {
            **result["GLOBAL"],
            **{
                f"NI0Z{num_imp + 1}": self.blobs["impurity_density"]
                .sel(element=imp)
                .sel(rho_poloidal=0, method="nearest")
                .median(dim="index")
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }
        result["GLOBAL"] = {
            **result["GLOBAL"],
            **{
                f"NI0Z{num_imp + 1}_ERR": self.blobs["impurity_density"]
                .sel(element=imp)
                .sel(rho_poloidal=0, method="nearest")
                .std(dim="index")
                for num_imp, imp in enumerate(self.plasma.impurities)
            },
        }

        self.result = result
        return self.result

    def run_sampler(self, iterations, burn_frac):
        """
        TODO: unsure if keeping in abstract class is best practice

        Runs the sampler and saves certain attributes from the sampler

        Returns

        result in MDSPlus node formatting

        """
        self.burn_frac = burn_frac
        self.iterations = iterations

        self.autocorr = sample_with_autocorr(
            self.sampler,
            self.start_points,
            iterations,
            self.param_names.__len__(),
            auto_sample=10,
        )
        blobs = self.sampler.get_blobs(discard=int(iterations * burn_frac), flat=True)
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
        self.prior_sample = self.bayesmodel.sample_from_priors(
            self.param_names, size=int(1e4)
        )
        self.post_sample = self.sampler.get_chain(flat=True)
        self.result = self._build_result_dict()

    def save_pickle(self, filepath):
        if filepath:
            Path(filepath).mkdir(parents=True, exist_ok=True)
            with open(filepath + "results.pkl", "wb") as handle:
                pickle.dump(self.result, handle)

    def dict_of_dataarray_to_numpy(self, dict_of_dataarray):
        """
        Mutates input dictionary to change xr.DataArray objects to np.array

        """
        for key, value in dict_of_dataarray.items():
            if isinstance(value, dict):
                self.dict_of_dataarray_to_numpy(value)
            elif isinstance(value, xr.DataArray):
                dict_of_dataarray[key] = dict_of_dataarray[key].values
        return dict_of_dataarray

    @abstractmethod
    def __call__(self, filepath="./results/test/", **kwargs):
        self.run_sampler()
        self.save_pickle(filepath=filepath)
        self.result = self.dict_of_dataarray_to_numpy(self.result)

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
        autocorr[
            sampler.iteration - 1,
        ] = new_tau
        converged = np.all(new_tau * 50 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - new_tau) / new_tau < 0.01)
        if converged:
            break
        old_tau = new_tau
    autocorr = autocorr[
        : sampler.iteration,
    ]
    return autocorr
