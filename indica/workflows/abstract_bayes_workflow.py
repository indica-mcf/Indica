from abc import ABC
from abc import abstractmethod
import getpass
from pathlib import Path
import pickle

import git
from datetime import datetime
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

        result["ELEMENT"] = self.plasma_context.plasma.elements
        result["TIME"] = self.plasma_context.plasma.t
        git_id = git.Repo(search_parent_directories=True).head.object.hexsha

        result["INPUT"] = {
            "GIT_ID": f"{git_id}",
            "USER": f"{getpass.getuser()}",
            "SETTINGS": "CONFIG GOES HERE",
            "DATATIME": datetime.utcnow().__str__()

        }
        # TODO fix workflow
        result["INPUT"]["WORKFLOW"] = {
            diag_name.upper(): {
                "PULSE": self.data_context.pulse,
                "USAGE": "".join(
                    [quantity[1] for quantity in quant_list if quantity[0] == diag_name]
                ),
                "RUN": "PLACEHOLDER",
            }
            for diag_name in self.blackbox_settings.diagnostics
        }

        result["DIAG_DATA"] = {
            diag_name.upper(): {
                quantity[1].upper(): self.data_context.opt_data[
                    f"{quantity[0]}.{quantity[1]}"
                ]
                for quantity in quant_list
                if quantity[0] == diag_name
            }
            for diag_name in self.blackbox_settings.diagnostics
        }

        return result

    def _build_result_dict(
        self,
    ):
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
        result["MODEL_DATA"]["SAMPLE_IDX"] = np.arange(0,
                self.opt_samples["post_sample"].shape[1]

        )

        result["PHANTOMS"] = {
            "FLAG": self.data_context.phantoms,
            "NE": self.plasma_context.phantom_profiles["electron_density"],
            "TE": self.plasma_context.phantom_profiles["electron_temperature"],
            "TI": self.plasma_context.phantom_profiles["ion_temperature"],
            "NI": self.plasma_context.phantom_profiles["ion_density"],
            "NNEUTR": self.plasma_context.phantom_profiles["neutral_density"],
            "NFAST": self.plasma_context.phantom_profiles["fast_density"],
            "ZEFF": self.plasma_context.phantom_profiles["zeff"].sum(dim="element"),
            "MEANZ": self.plasma_context.phantom_profiles["meanz"],
            "PTH": self.plasma_context.phantom_profiles["pressure_th"],
            "PFAST": self.plasma_context.phantom_profiles["pressure_fast"],
            "P": self.plasma_context.phantom_profiles["pressure_tot"],

        }


        result["PROFILES"] = {
            "PSI_NORM": {
                "RHOP": self.plasma_context.plasma.rho,
                "RHOT": self.plasma_context.plasma.equilibrium.rhotor.interp(
                    t=self.plasma_context.plasma.t
                ),
                "VOLUME": self.plasma_context.plasma.volume,

                "NE": self.blobs["electron_density"].median(dim="index"),
                "NI": self.blobs["ion_density"].median(dim="index"),
                "TE": self.blobs["electron_temperature"].median(dim="index"),
                "TI": self.blobs["ion_temperature"].median(dim="index"),
                "NFAST": self.blobs["fast_density"].median(dim="index"),
                "NNEUTR": self.blobs["neutral_density"].median(dim="index"),
                "P": self.blobs["pressure_tot"].median(dim="index"),
                "PTH": self.blobs["pressure_th"].median(dim="index"),
                "PFAST": self.blobs["pressure_fast"].median(dim="index"),
                "ZEFF": self.blobs["zeff"].sum("element").median(dim="index"),
                "MEANZ": self.blobs["meanz"].median(dim="index"),

                "NE_ERR": self.blobs["electron_density"].std(dim="index"),
                "NI_ERR": self.blobs["ion_density"].std(dim="index"),
                "TE_ERR": self.blobs["electron_temperature"].std(dim="index"),
                "TI_ERR": self.blobs["ion_temperature"].std(dim="index"),
                "NFAST_ERR": self.blobs["fast_density"].std(dim="index"),
                "NNEUTR_ERR": self.blobs["neutral_density"].std(dim="index"),
                "P_ERR": self.blobs["pressure_tot"].std(dim="index"),
                "PTH_ERR": self.blobs["pressure_th"].std(dim="index"),
                "PFAST_ERR": self.blobs["pressure_fast"].std(dim="index"),
                "ZEFF_ERR": self.blobs["zeff"].sum("element").std(dim="index"),
                "MEANZ_ERR": self.blobs["meanz"].std(dim="index"),

            },

            "R_MIDPLANE": {
                "RPOS": self.midplane_blobs["electron_temperature"].R,
                "ZPOS": self.midplane_blobs["electron_temperature"].z,

                "NE": self.midplane_blobs["electron_density"].median(dim="index"),
                "NI": self.midplane_blobs["ion_density"].median(dim="index"),
                "TE": self.midplane_blobs["electron_temperature"].median(dim="index"),
                "TI": self.midplane_blobs["ion_temperature"].median(dim="index"),
                "NFAST": self.midplane_blobs["fast_density"].median(dim="index"),
                "NNEUTR": self.midplane_blobs["neutral_density"].median(dim="index"),
                "P": self.midplane_blobs["pressure_tot"].median(dim="index"),
                "PTH": self.midplane_blobs["pressure_th"].median(dim="index"),
                "PFAST": self.midplane_blobs["pressure_fast"].median(dim="index"),
                "ZEFF": self.midplane_blobs["zeff"].sum("element").median(dim="index"),
                "MEANZ": self.midplane_blobs["meanz"].median(dim="index"),

                "NE_ERR": self.midplane_blobs["electron_density"].std(dim="index"),
                "NI_ERR": self.midplane_blobs["ion_density"].std(dim="index"),
                "TE_ERR": self.midplane_blobs["electron_temperature"].std(dim="index"),
                "TI_ERR": self.midplane_blobs["ion_temperature"].std(dim="index"),
                "NFAST_ERR": self.midplane_blobs["fast_density"].std(dim="index"),
                "NNEUTR_ERR": self.midplane_blobs["neutral_density"].std(dim="index"),
                "P_ERR": self.midplane_blobs["pressure_tot"].std(dim="index"),
                "PTH_ERR": self.midplane_blobs["pressure_th"].std(dim="index"),
                "PFAST_ERR": self.midplane_blobs["pressure_fast"].std(dim="index"),
                "ZEFF_ERR": self.midplane_blobs["zeff"].sum("element").std(dim="index"),
                "MEANZ_ERR": self.midplane_blobs["meanz"].median(dim="index"),
            },
        }

        result["PROFILE_STAT"] = {
            "SAMPLE_IDX": np.arange(0,
                self.opt_samples["post_sample"].shape[1]
            ),
            "RHOP": self.plasma_context.plasma.rho,
            "NE": self.blobs["electron_density"],
            "NI": self.blobs["ion_density"],
            "TE": self.blobs["electron_temperature"],
            "TI": self.blobs["ion_temperature"],
            "NFAST": self.blobs["fast_density"],
            "NNEUTR": self.blobs["neutral_density"],
            "P": self.blobs["pressure_tot"],
            "PTH": self.blobs["pressure_th"],
            "PFAST": self.blobs["pressure_fast"],
            "ZEFF": self.blobs["zeff"].sum("element"),
            "MEANZ": self.blobs["meanz"],
        }

        result["OPTIMISATION"] = {
            "ACCEPT_FRAC": self.opt_samples["accept_frac"],
            "PRIOR_SAMPLE": self.opt_samples["prior_sample"],
            "POST_SAMPLE": self.opt_samples["post_sample"],
            "AUTO_CORR": self.opt_samples["auto_corr"],
            "PARAM_NAMES": self.optimiser_context.optimiser_settings.param_names
            # "GELMAN_RUBIN": gelman_rubin(self.sampler.get_chain(flat=False))
        }

        result["GLOBAL"] = {
            "VOLUME": self.plasma_context.plasma.volume.max(dim="rho_poloidal"),
            "TI0": self.blobs["ion_temperature"]
            .sel(rho_poloidal=0, method="nearest")
            .median(dim="index"),
            "TE0": self.blobs["electron_temperature"]
            .sel(rho_poloidal=0, method="nearest")
            .median(dim="index"),
            "NE0": self.blobs["electron_density"]
            .sel(rho_poloidal=0, method="nearest")
            .median(dim="index"),
            "NI0": self.blobs["ion_density"]  # TODO: where to concat the impurity_density onto this
            .sel(rho_poloidal=0, method="nearest")
            .median(dim="index"),
            "WP": self.blobs["wp"].median(dim="index"),
            "WTH": self.blobs["wth"].median(dim="index"),
            "ZEFF_AVG": self.midplane_blobs["zeff"].sum(dim="element").median(dim="index").mean(dim="R"),
            "NNEUTR0": self.blobs["neutral_density"].sel(rho_poloidal=0, method="nearest").median(dim="index"),
            "NNEUTRB": self.blobs["neutral_density"].sel(rho_poloidal=1, method="nearest").median(dim="index"),

            "TI0_ERR": self.blobs["ion_temperature"]
                .sel(rho_poloidal=0, method="nearest")
                .std(dim="index"),
            "TE0_ERR": self.blobs["electron_temperature"]
                .sel(rho_poloidal=0, method="nearest")
                .std(dim="index"),
            "NE0_ERR": self.blobs["electron_density"]
                .sel(rho_poloidal=0, method="nearest")
                .std(dim="index"),
            "NI0_ERR": self.blobs["ion_density"]
                .sel(rho_poloidal=0, method="nearest")
                .std(dim="index"),
            "WTH_ERR": self.blobs["wth"].std(dim="index"),
            "WP_ERR": self.blobs["wp"].std(dim="index"),
            "ZEFF_AVG_ERR": self.midplane_blobs["zeff"].sum(dim="element").std(dim="index").mean(dim="R"),
            "NNEUTR0_ERR": self.blobs["neutral_density"].sel(rho_poloidal=0, method="nearest").std(dim="index"),
            "NNEUTRB_ERR": self.blobs["neutral_density"].sel(rho_poloidal=1, method="nearest").std(dim="index"),
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
