from abc import ABC
from abc import abstractmethod
from pathlib import Path
import pickle

import numpy as np

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
        quant_list = [item.split(".") for item in self.blackbox_settings.opt_quantity]

        result["TIME"] = self.plasma_context.plasma.t

        result["METADATA"] = {
            "GITCOMMIT": "PLACEHOLDER",
            "USER": "PLACEHOLDER",
            "EQUIL": "PLACEHOLDER",
        }

        result["INPUT"] = {
            "BURN_FRAC": self.optimiser_context.optimiser_settings.burn_frac,
            "ITER": self.optimiser_context.optimiser_settings.iterations,
            "NWALKERS": self.optimiser_context.optimiser_settings.nwalkers,
            "MODEL_KWARGS": self.model_context.model_settings.init_kwargs,
            "OPT_QUANTITY": self.blackbox_settings.opt_quantity,
            "PARAM_NAMES": self.blackbox_settings.param_names,
            "PULSE": self.data_context.pulse,
            "IMPURITIES": self.plasma_context.plasma_settings.impurities,
            "MAIN_ION": self.plasma_context.plasma_settings.main_ion,
            "TSTART":self.tstart,
            "TEND": self.tend,
            "DT": self.dt,

        }
        result["INPUT"]["WORKFLOW"] = {
            diag_name.upper(): {
                "PULSE": self.data_context.pulse,  # Change this if different pulses used
                "USAGE": "".join(
                    [quantity[1] for quantity in quant_list if quantity[0] == diag_name]
                ),
                "RUN": "PLACEHOLDER",
            }
            for diag_name in self.blackbox_settings.diagnostics
        }

        result["DIAG_DATA"] = {
            diag_name.upper(): {
                quantity[1].upper(): self.data_context.opt_data[f"{quantity[0]}.{quantity[1]}"]
                for quantity in quant_list
                if quantity[0] == diag_name
            }
            for diag_name in self.blackbox_settings.diagnostics
        }

        return result


    def _build_result_dict(self,  ):
        result = {}
        quant_list = [item.split(".") for item in self.blackbox_settings.opt_quantity]

        result["MODEL_DATA"] = {
            diag_name.upper(): {
                quantity[1].upper(): self.blobs[f"{quantity[0]}.{quantity[1]}"]
                for quantity in quant_list
                if quantity[0] == diag_name
            }
            for diag_name in self.blackbox_settings.diagnostics
        }
        result["MODEL_DATA"]["SAMPLE_IDX"] = np.arange(self.optimiser_context.optimiser_settings.iterations*
                                                       (1-self.optimiser_context.optimiser_settings.burn_frac))

        result["PHANTOMS"] = {
            "FLAG": self.data_context.phantoms,
            "NE": self.plasma_context.phantom_profiles["electron_density"],
            "TE": self.plasma_context.phantom_profiles["electron_temperature"],
            "TI": self.plasma_context.phantom_profiles["ion_temperature"].sel(
                element=self.plasma_context.plasma_settings.main_ion
            ),
            "NI": self.plasma_context.phantom_profiles["ion_density"].sel(
                element=self.plasma_context.plasma_settings.main_ion
            ),
            "NNEUTR": self.plasma_context.phantom_profiles["neutral_density"],
            "NFAST": self.plasma_context.phantom_profiles["fast_density"],
        }
        result["PHANTOMS"].update(
            {
                f"NIZ{num_imp + 1}": self.plasma_context.phantom_profiles["impurity_density"].sel(
                    element=imp
                )
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            }
        )
        result["PHANTOMS"].update(
            {
                f"TIZ{num_imp + 1}": self.plasma_context.phantom_profiles["ion_temperature"].sel(
                    element=imp
                )
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            }
        )

        result["PROFILES"] = {
            "PSI_NORM":{
                "RHOP": self.plasma_context.plasma.rho,
                "RHOT": self.plasma_context.plasma.equilibrium.rhotor.interp(t=self.plasma_context.plasma.t),
                "NE": self.blobs["electron_density"].median(dim="index"),
                "NI": self.blobs["ion_density"]
                    .sel(element=self.plasma_context.plasma_settings.main_ion)
                    .median(dim="index"),
                "TE": self.blobs["electron_temperature"].median(dim="index"),
                "TI": self.blobs["ion_temperature"]
                    .sel(element=self.plasma_context.plasma_settings.main_ion)
                    .median(dim="index"),
                "NFAST": self.blobs["fast_density"].median(dim="index"),
                "NNEUTR": self.blobs["neutral_density"].median(dim="index"),
                "NE_ERR": self.blobs["electron_density"].std(dim="index"),
                "NI_ERR": self.blobs["ion_density"]
                    .sel(element=self.plasma_context.plasma_settings.main_ion)
                    .std(dim="index"),
                "TE_ERR": self.blobs["electron_temperature"].std(dim="index"),
                "TI_ERR": self.blobs["ion_temperature"]
                    .sel(element=self.plasma_context.plasma_settings.main_ion)
                    .std(dim="index"),
                "NFAST_ERR": self.blobs["fast_density"].std(dim="index"),
                "NNEUTR_ERR": self.blobs["neutral_density"].std(dim="index"),
                "ZEFF": self.blobs["zeff"].sum("element").median(dim="index"),
                "ZEFF_ERR": self.blobs["zeff"].sum("element").std(dim="index"),
                "ZI": self.blobs["zeff"].sel(element=self.plasma_context.plasma_settings.main_ion).median(dim="index"),
                "ZI_ERR": self.blobs["zeff"].sel(element=self.plasma_context.plasma_settings.main_ion).std(dim="index"),
            },
            "R_MIDPLANE": {
                "RPOS": self.midplane_blobs["electron_temperature"].R,
                "ZPOS": self.midplane_blobs["electron_temperature"].z,
                "NE": self.midplane_blobs["electron_density"].median(dim="index"),
                "NI": self.midplane_blobs["ion_density"]
                    .sel(element=self.plasma_context.plasma_settings.main_ion)
                    .median(dim="index"),
                "TE": self.midplane_blobs["electron_temperature"].median(dim="index"),
                "TI": self.midplane_blobs["ion_temperature"]
                    .sel(element=self.plasma_context.plasma_settings.main_ion)
                    .median(dim="index"),
                "NFAST": self.midplane_blobs["fast_density"].median(dim="index"),
                "NNEUTR": self.midplane_blobs["neutral_density"].median(dim="index"),
                "NE_ERR": self.midplane_blobs["electron_density"].std(dim="index"),
                "NI_ERR": self.midplane_blobs["ion_density"]
                    .sel(element=self.plasma_context.plasma_settings.main_ion)
                    .std(dim="index"),
                "TE_ERR": self.midplane_blobs["electron_temperature"].std(dim="index"),
                "TI_ERR": self.midplane_blobs["ion_temperature"]
                    .sel(element=self.plasma_context.plasma_settings.main_ion)
                    .std(dim="index"),
                "NFAST_ERR": self.midplane_blobs["fast_density"].std(dim="index"),
                "NNEUTR_ERR": self.midplane_blobs["neutral_density"].std(dim="index"),
                "ZEFF": self.midplane_blobs["zeff"].sum("element").median(dim="index"),
                "ZEFF_ERR": self.midplane_blobs["zeff"].sum("element").std(dim="index"),
                "ZI": self.midplane_blobs["zeff"].sel(element=self.plasma_context.plasma_settings.main_ion).median(dim="index"),
                "ZI_ERR": self.midplane_blobs["zeff"].sel(element=self.plasma_context.plasma_settings.main_ion).std(dim="index"),

        },
        }
        result["PROFILES"]["PSI_NORM"] = {
            **result["PROFILES"]["PSI_NORM"],
            **{
                f"NIZ{num_imp + 1}": self.blobs["impurity_density"]
                    .sel(element=imp)
                    .median(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILES"]["PSI_NORM"] = {
            **result["PROFILES"]["PSI_NORM"],
            **{
                f"NIZ{num_imp + 1}_ERR": self.blobs["impurity_density"]
                    .sel(element=imp)
                    .std(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILES"]["PSI_NORM"] = {
            **result["PROFILES"]["PSI_NORM"],
            **{
                f"TIZ{num_imp + 1}": self.blobs["ion_temperature"]
                    .sel(element=imp)
                    .median(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILES"]["PSI_NORM"] = {
            **result["PROFILES"]["PSI_NORM"],
            **{
                f"TIZ{num_imp + 1}_ERR": self.blobs["ion_temperature"]
                    .sel(element=imp)
                    .std(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }

        result["PROFILES"]["PSI_NORM"] = {
            **result["PROFILES"]["PSI_NORM"],
            **{
                f"ZIM{num_imp + 1}": self.blobs["zeff"]
                    .sel(element=imp)
                    .median(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILES"]["PSI_NORM"] = {
            **result["PROFILES"]["PSI_NORM"],
            **{
                f"ZIM{num_imp + 1}_ERR": self.blobs["zeff"]
                    .sel(element=imp)
                    .std(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }

        result["PROFILES"]["R_MIDPLANE"] = {
            **result["PROFILES"]["R_MIDPLANE"],
            **{
                f"NIZ{num_imp + 1}": self.midplane_blobs["impurity_density"]
                    .sel(element=imp)
                    .median(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILES"]["R_MIDPLANE"] = {
            **result["PROFILES"]["R_MIDPLANE"],
            **{
                f"NIZ{num_imp + 1}_ERR": self.midplane_blobs["impurity_density"]
                    .sel(element=imp)
                    .std(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILES"]["R_MIDPLANE"] = {
            **result["PROFILES"]["R_MIDPLANE"],
            **{
                f"TIZ{num_imp + 1}": self.midplane_blobs["ion_temperature"]
                    .sel(element=imp)
                    .median(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILES"]["R_MIDPLANE"] = {
            **result["PROFILES"]["R_MIDPLANE"],
            **{
                f"TIZ{num_imp + 1}_ERR": self.midplane_blobs["ion_temperature"]
                    .sel(element=imp)
                    .std(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILES"]["R_MIDPLANE"] = {
            **result["PROFILES"]["R_MIDPLANE"],
            **{
                f"ZIM{num_imp + 1}": self.midplane_blobs["zeff"]
                    .sel(element=imp)
                    .median(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILES"]["R_MIDPLANE"] = {
            **result["PROFILES"]["R_MIDPLANE"],
            **{
                f"ZIM{num_imp + 1}_ERR": self.midplane_blobs["zeff"]
                    .sel(element=imp)
                    .std(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }

        result["PROFILE_STAT"] = {
            "SAMPLE_IDX": np.arange(self.optimiser_context.optimiser_settings.iterations *
                                   (1-self.optimiser_context.optimiser_settings.burn_frac)),
            "RHO_POLOIDAL": self.plasma_context.plasma.rho,
            "NE": self.blobs["electron_density"],
            "NI": self.blobs["ion_density"].sel(element=self.plasma_context.plasma_settings.main_ion),
            "TE": self.blobs["electron_temperature"],
            "TI": self.blobs["ion_temperature"].sel(element=self.plasma_context.plasma_settings.main_ion),
            "NFAST": self.blobs["fast_density"],
            "NNEUTR": self.blobs["neutral_density"],
        }
        result["PROFILE_STAT"] = {
            **result["PROFILE_STAT"],
            **{
                f"NIZ{num_imp + 1}": self.blobs["impurity_density"].sel(element=imp)
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["PROFILE_STAT"] = {
            **result["PROFILE_STAT"],
            **{
                f"TIZ{num_imp + 1}": self.blobs["ion_temperature"].sel(element=imp)
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }

        result["OPTIMISATION"] = {
            "ACCEPT_FRAC": self.opt_samples["accept_frac"],
            "PRIOR_SAMPLE": self.opt_samples["prior_sample"],
            "POST_SAMPLE": self.opt_samples["post_sample"],
            "AUTO_CORR": self.opt_samples["auto_corr"],
            # "GELMAN_RUBIN": gelman_rubin(self.sampler.get_chain(flat=False))
        }

        result["GLOBAL"] = {
            "TI0": self.blobs["ion_temperature"]
                .sel(element=self.plasma_context.plasma_settings.main_ion)
                .sel(rho_poloidal=0, method="nearest")
                .median(dim="index"),
            "TE0": self.blobs["electron_temperature"]
                .sel(rho_poloidal=0, method="nearest")
                .median(dim="index"),
            "NE0": self.blobs["electron_density"]
                .sel(rho_poloidal=0, method="nearest")
                .median(dim="index"),
            "NI0": self.blobs["ion_density"]
                .sel(element=self.plasma_context.plasma_settings.main_ion)
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
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["GLOBAL"] = {
            **result["GLOBAL"],
            **{
                f"TI0Z{num_imp + 1}_ERR": self.blobs["ion_temperature"]
                    .sel(element=imp)
                    .sel(rho_poloidal=0, method="nearest")
                    .std(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["GLOBAL"] = {
            **result["GLOBAL"],
            **{
                f"NI0Z{num_imp + 1}": self.blobs["impurity_density"]
                    .sel(element=imp)
                    .sel(rho_poloidal=0, method="nearest")
                    .median(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        result["GLOBAL"] = {
            **result["GLOBAL"],
            **{
                f"NI0Z{num_imp + 1}_ERR": self.blobs["impurity_density"]
                    .sel(element=imp)
                    .sel(rho_poloidal=0, method="nearest")
                    .std(dim="index")
                for num_imp, imp in enumerate(self.plasma_context.plasma_settings.impurities)
            },
        }
        return result

    def save_pickle(self, result, filepath):
        Path(filepath).mkdir(parents=True, exist_ok=True)
        with open(filepath + "results.pkl", "wb") as handle:
            pickle.dump(result, handle)


    @abstractmethod
    def __call__(self, filepath="./results/test/", **kwargs):
        result = self.run_sampler()
        self.save_pickle(result, filepath=filepath)
        self.result = result

        return self.result

