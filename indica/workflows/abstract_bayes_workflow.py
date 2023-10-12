from abc import ABC
from abc import abstractmethod
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from indica.equilibrium import fake_equilibrium
from indica.readers.read_st40 import ReadST40
import scipy.stats as stats

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


    def _build_inputs_dict(self):
        """

        Returns
        -------

        dictionary of inputs in MDS+ structure

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

        result["DIAG_DATA"] = {
            diag_name.upper(): {
                quantity[1].upper(): self.opt_data[f"{quantity[0]}.{quantity[1]}"]
                for quantity in quant_list
                if quantity[0] == diag_name
            }
            for diag_name in self.diagnostics
        }


        return result


    def _build_result_dict(self,  ):
        result = {}
        quant_list = [item.split(".") for item in self.opt_quantity]

        result["MODEL_DATA"] = {
            diag_name.upper(): {
                quantity[1].upper(): self.blobs[f"{quantity[0]}.{quantity[1]}"]
                for quantity in quant_list
                if quantity[0] == diag_name
            }
            for diag_name in self.diagnostics
        }
        result["MODEL_DATA"]["SAMPLES"] = self.samples


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
            # "GELMANRUBIN": gelman_rubin(self.sampler.get_chain(flat=False))
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
        return result


    def save_pickle(self, result, filepath):
        if filepath:
            Path(filepath).mkdir(parents=True, exist_ok=True)
            with open(filepath + "results.pkl", "wb") as handle:
                pickle.dump(result, handle)


    @abstractmethod
    def __call__(self, filepath="./results/test/", **kwargs):
        self.run_sampler()
        self.save_pickle(filepath=filepath)
        self.result = self.dict_of_dataarray_to_numpy(self.result)

        return self.result

