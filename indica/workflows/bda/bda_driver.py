from datetime import datetime
import getpass
import logging

import git
import numpy as np
import xarray as xr

from indica.plasma import PlasmaProfiler
from indica.readers.modelreader import ModelReader
from indica.workflows.bda.bayesblackbox import BayesBlackBox
from indica.workflows.bda.optimisers import OptimiserContext
from indica.workflows.bda.priors import PriorManager


class BDADriver:
    def __init__(
        self,
        quant_to_optimise: list,
        opt_data: dict,
        plasma_profiler: PlasmaProfiler,
        prior_manager: PriorManager,
        modelreader: ModelReader,
        optimiser_context: OptimiserContext,
    ):
        self.quant_to_optimise = quant_to_optimise
        self.opt_data = opt_data
        self.plasma_profiler = plasma_profiler
        self.prior_manager = prior_manager
        self.modelreader = modelreader
        self.optimiser_context = optimiser_context

        self.blackbox = BayesBlackBox(
            opt_data=self.opt_data,
            quant_to_optimise=quant_to_optimise,
            ln_prior=self.prior_manager.ln_prior,
            plasma_profiler=self.plasma_profiler,
            build_bckc=self.modelreader.__call__,
        )

        self.optimiser_context.init_optimiser(self.blackbox.ln_posterior)

    def _build_inputs_dict(self):
        """

        Returns
        -------

        dictionary of inputs in MDS+ structure

        """

        result = {}
        quant_list = [item.split(".") for item in self.quant_to_optimise]

        result["ELEMENT"] = self.plasma_profiler.plasma.elements
        result["TIME"] = self.plasma_profiler.plasma.t.values
        # git_id = git.Repo(search_parent_directories=True).head.object.hexsha

        result["INPUT"] = {
            # "GIT_ID": f"{git_id}",
            "USER": f"{getpass.getuser()}",
            "SETTINGS": self.config,
            "DATETIME": datetime.utcnow().__str__(),
        }

        result["DIAG_DATA"] = {
            diag_name.upper(): {
                quantity[1].upper(): self.opt_data[f"{quantity[0]}.{quantity[1]}"]
                for quantity in quant_list
                if quantity[0] == diag_name
            }
            for diag_name in self.modelreader.models.keys()
        }

        return result

    def _build_result_dict(
        self,
    ):
        result = {}
        quant_list = [item.split(".") for item in self.quant_to_optimise]

        result["MODEL_DATA"] = {
            diag_name.upper(): {
                quantity[1].upper(): self.blobs[f"{quantity[0]}.{quantity[1]}"]
                for quantity in quant_list
                if quantity[0] == diag_name
            }
            for diag_name in self.modelreader.models.keys()
        }
        result["MODEL_DATA"]["SAMPLE_IDX"] = np.arange(
            0, self.opt_samples["post_sample"].shape[1]
        )

        result["PHANTOMS"] = {
            "FLAG": self.plasma_profiler.phantom,
            "PSI_NORM": {
                "RHOP": self.plasma_profiler.plasma.rhop,
                "NE": self.plasma_profiler.phantom_profiles["PSI_NORM"][
                    "electron_density"
                ],
                "TE": self.plasma_profiler.phantom_profiles["PSI_NORM"][
                    "electron_temperature"
                ],
                "TI": self.plasma_profiler.phantom_profiles["PSI_NORM"][
                    "ion_temperature"
                ],
                "NI": self.plasma_profiler.phantom_profiles["PSI_NORM"]["ion_density"],
                "NNEUTR": self.plasma_profiler.phantom_profiles["PSI_NORM"][
                    "neutral_density"
                ],
                "NFAST": self.plasma_profiler.phantom_profiles["PSI_NORM"][
                    "fast_ion_density"
                ],
                "ZEFF": self.plasma_profiler.phantom_profiles["PSI_NORM"]["zeff"].sum(
                    dim="element"
                ),
                "MEANZ": self.plasma_profiler.phantom_profiles["PSI_NORM"]["meanz"],
                "PTH": self.plasma_profiler.phantom_profiles["PSI_NORM"][
                    "thermal_pressure"
                ],
                "PFAST": self.plasma_profiler.phantom_profiles["PSI_NORM"][
                    "fast_ion_pressure"
                ],
                "P": self.plasma_profiler.phantom_profiles["PSI_NORM"]["pressure"],
                "VTOR": self.plasma_profiler.phantom_profiles["PSI_NORM"][
                    "toroidal_rotation"
                ],
            },
            "R_MIDPLANE": {
                "RPOS": self.plasma_profiler.plasma.R_midplane,
                "ZPOS": self.plasma_profiler.plasma.z_midplane,
                "NE": self.plasma_profiler.phantom_profiles["R_MIDPLANE"][
                    "electron_density"
                ],
                "TE": self.plasma_profiler.phantom_profiles["R_MIDPLANE"][
                    "electron_temperature"
                ],
                "TI": self.plasma_profiler.phantom_profiles["R_MIDPLANE"][
                    "ion_temperature"
                ],
                "NI": self.plasma_profiler.phantom_profiles["R_MIDPLANE"][
                    "ion_density"
                ],
                "NNEUTR": self.plasma_profiler.phantom_profiles["R_MIDPLANE"][
                    "neutral_density"
                ],
                "NFAST": self.plasma_profiler.phantom_profiles["R_MIDPLANE"][
                    "fast_ion_density"
                ],
                "ZEFF": self.plasma_profiler.phantom_profiles["R_MIDPLANE"]["zeff"].sum(
                    dim="element"
                ),
                "MEANZ": self.plasma_profiler.phantom_profiles["R_MIDPLANE"]["meanz"],
                "PTH": self.plasma_profiler.phantom_profiles["R_MIDPLANE"][
                    "thermal_pressure"
                ],
                "PFAST": self.plasma_profiler.phantom_profiles["R_MIDPLANE"][
                    "fast_ion_pressure"
                ],
                "P": self.plasma_profiler.phantom_profiles["R_MIDPLANE"]["pressure"],
                "VTOR": self.plasma_profiler.phantom_profiles["R_MIDPLANE"][
                    "toroidal_rotation"
                ],
            },
        }

        result["PROFILES"] = {
            "PSI_NORM": {
                "RHOP": self.plasma_profiler.plasma.rhop,
                "RHOT": self.plasma_profiler.plasma.equilibrium.rhot.interp(
                    t=self.plasma_profiler.plasma.t
                ),
                "VOLUME": self.plasma_profiler.plasma.volume,
                "NE": self.blobs["electron_density"].median(dim="sample_idx"),
                "NI": self.blobs["ion_density"].median(dim="sample_idx"),
                "TE": self.blobs["electron_temperature"].median(dim="sample_idx"),
                "TI": self.blobs["ion_temperature"].median(dim="sample_idx"),
                "NFAST": self.blobs["fast_ion_density"].median(dim="sample_idx"),
                "NNEUTR": self.blobs["neutral_density"].median(dim="sample_idx"),
                "P": self.blobs["pressure"].median(dim="sample_idx"),
                "PTH": self.blobs["thermal_pressure"].median(dim="sample_idx"),
                "PFAST": self.blobs["fast_ion_pressure"].median(dim="sample_idx"),
                "ZEFF": self.blobs["zeff"].sum("element").median(dim="sample_idx"),
                "MEANZ": self.blobs["meanz"].median(dim="sample_idx"),
                "NE_ERR": self.blobs["electron_density"].std(dim="sample_idx"),
                "NI_ERR": self.blobs["ion_density"].std(dim="sample_idx"),
                "TE_ERR": self.blobs["electron_temperature"].std(dim="sample_idx"),
                "TI_ERR": self.blobs["ion_temperature"].std(dim="sample_idx"),
                "NFAST_ERR": self.blobs["fast_ion_density"].std(dim="sample_idx"),
                "NNEUTR_ERR": self.blobs["neutral_density"].std(dim="sample_idx"),
                "P_ERR": self.blobs["pressure"].std(dim="sample_idx"),
                "PTH_ERR": self.blobs["thermal_pressure"].std(dim="sample_idx"),
                "PFAST_ERR": self.blobs["fast_ion_pressure"].std(dim="sample_idx"),
                "ZEFF_ERR": self.blobs["zeff"].sum("element").std(dim="sample_idx"),
                "MEANZ_ERR": self.blobs["meanz"].std(dim="sample_idx"),
                "VTOR": self.blobs["toroidal_rotation"].median(dim="sample_idx"),
                "VTOR_ERR": self.blobs["toroidal_rotation"].std(dim="sample_idx"),
            },
            "R_MIDPLANE": {
                "RPOS": self.plasma_profiler.plasma.R_midplane,
                "ZPOS": self.plasma_profiler.plasma.z_midplane,
                "NE": self.midplane_blobs["electron_density"].median(dim="sample_idx"),
                "NI": self.midplane_blobs["ion_density"].median(dim="sample_idx"),
                "TE": self.midplane_blobs["electron_temperature"].median(
                    dim="sample_idx"
                ),
                "TI": self.midplane_blobs["ion_temperature"].median(dim="sample_idx"),
                "NFAST": self.midplane_blobs["fast_ion_density"].median(
                    dim="sample_idx"
                ),
                "NNEUTR": self.midplane_blobs["neutral_density"].median(
                    dim="sample_idx"
                ),
                "P": self.midplane_blobs["pressure"].median(dim="sample_idx"),
                "PTH": self.midplane_blobs["thermal_pressure"].median(dim="sample_idx"),
                "PFAST": self.midplane_blobs["fast_ion_pressure"].median(
                    dim="sample_idx"
                ),
                "ZEFF": self.midplane_blobs["zeff"]
                .sum("element")
                .median(dim="sample_idx"),
                "MEANZ": self.midplane_blobs["meanz"].median(dim="sample_idx"),
                "NE_ERR": self.midplane_blobs["electron_density"].std(dim="sample_idx"),
                "NI_ERR": self.midplane_blobs["ion_density"].std(dim="sample_idx"),
                "TE_ERR": self.midplane_blobs["electron_temperature"].std(
                    dim="sample_idx"
                ),
                "TI_ERR": self.midplane_blobs["ion_temperature"].std(dim="sample_idx"),
                "NFAST_ERR": self.midplane_blobs["fast_ion_density"].std(
                    dim="sample_idx"
                ),
                "NNEUTR_ERR": self.midplane_blobs["neutral_density"].std(
                    dim="sample_idx"
                ),
                "P_ERR": self.midplane_blobs["pressure"].std(dim="sample_idx"),
                "PTH_ERR": self.midplane_blobs["thermal_pressure"].std(
                    dim="sample_idx"
                ),
                "PFAST_ERR": self.midplane_blobs["fast_ion_pressure"].std(
                    dim="sample_idx"
                ),
                "ZEFF_ERR": self.midplane_blobs["zeff"]
                .sum("element")
                .std(dim="sample_idx"),
                "MEANZ_ERR": self.midplane_blobs["meanz"].std(dim="sample_idx"),
                "VTOR": self.midplane_blobs["toroidal_rotation"].median(
                    dim="sample_idx"
                ),
                "VTOR_ERR": self.midplane_blobs["toroidal_rotation"].std(
                    dim="sample_idx"
                ),
            },
        }

        result["PROFILE_STAT"] = {
            "SAMPLE_IDX": np.arange(0, self.opt_samples["post_sample"].shape[1]),
            "PSI_NORM": {
                "RHOP": self.plasma_profiler.plasma.rhop,
                "NE": self.blobs["electron_density"],
                "NI": self.blobs["ion_density"],
                "TE": self.blobs["electron_temperature"],
                "TI": self.blobs["ion_temperature"],
                "NFAST": self.blobs["fast_ion_density"],
                "NNEUTR": self.blobs["neutral_density"],
                "P": self.blobs["pressure"],
                "PTH": self.blobs["thermal_pressure"],
                "PFAST": self.blobs["fast_ion_pressure"],
                "ZEFF": self.blobs["zeff"].sum("element"),
                "MEANZ": self.blobs["meanz"],
                "VTOR": self.blobs["toroidal_rotation"],
            },
            "R_MIDPLANE": {
                "RPOS": self.plasma_profiler.plasma.R_midplane,
                "NE": self.midplane_blobs["electron_density"],
                "NI": self.midplane_blobs["ion_density"],
                "TE": self.midplane_blobs["electron_temperature"],
                "TI": self.midplane_blobs["ion_temperature"],
                "NFAST": self.midplane_blobs["fast_ion_density"],
                "NNEUTR": self.midplane_blobs["neutral_density"],
                "P": self.midplane_blobs["pressure"],
                "PTH": self.midplane_blobs["thermal_pressure"],
                "PFAST": self.midplane_blobs["fast_ion_pressure"],
                "ZEFF": self.midplane_blobs["zeff"].sum("element"),
                "MEANZ": self.midplane_blobs["meanz"],
                "VTOR": self.midplane_blobs["toroidal_rotation"],
            },
        }

        result["OPTIMISATION"] = {
            "PRIOR_SAMPLE": self.opt_samples["prior_sample"],
            "POST_SAMPLE": self.opt_samples["post_sample"],
            "PARAM_NAMES": self.optimiser_context.optimiser_settings.param_names,
            "CONVERGENCE": self.convergence,
        }

        result["GLOBAL"] = {
            "VOLUME": self.plasma_profiler.plasma.volume.max(dim="rhop"),
            "TI0": self.blobs["ion_temperature"]
            .sel(rhop=0, method="nearest")
            .median(dim="sample_idx"),
            "TE0": self.blobs["electron_temperature"]
            .sel(rhop=0, method="nearest")
            .median(dim="sample_idx"),
            "NE0": self.blobs["electron_density"]
            .sel(rhop=0, method="nearest")
            .median(dim="sample_idx"),
            "NI0": self.blobs[
                "ion_density"
            ]  # TODO: where to concat the impurity_density onto this
            .sel(rhop=0, method="nearest")
            .median(dim="sample_idx"),
            "WP": self.blobs["wp"].median(dim="sample_idx"),
            "WTH": self.blobs["wth"].median(dim="sample_idx"),
            "ZEFF_AVG": self.midplane_blobs["zeff"]
            .sum(dim="element")
            .median(dim="sample_idx")
            .mean(dim="R"),
            "NNEUTR0": self.blobs["neutral_density"]
            .sel(rhop=0, method="nearest")
            .median(dim="sample_idx"),
            "NNEUTRB": self.blobs["neutral_density"]
            .sel(rhop=1, method="nearest")
            .median(dim="sample_idx"),
            "TI0_ERR": self.blobs["ion_temperature"]
            .sel(rhop=0, method="nearest")
            .std(dim="sample_idx"),
            "TE0_ERR": self.blobs["electron_temperature"]
            .sel(rhop=0, method="nearest")
            .std(dim="sample_idx"),
            "NE0_ERR": self.blobs["electron_density"]
            .sel(rhop=0, method="nearest")
            .std(dim="sample_idx"),
            "NI0_ERR": self.blobs["ion_density"]
            .sel(rhop=0, method="nearest")
            .std(dim="sample_idx"),
            "WTH_ERR": self.blobs["wth"].std(dim="sample_idx"),
            "WP_ERR": self.blobs["wp"].std(dim="sample_idx"),
            "ZEFF_AVG_ERR": self.midplane_blobs["zeff"]
            .sum(dim="element")
            .std(dim="sample_idx")
            .mean(dim="R"),
            "NNEUTR0_ERR": self.blobs["neutral_density"]
            .sel(rhop=0, method="nearest")
            .std(dim="sample_idx"),
            "NNEUTRB_ERR": self.blobs["neutral_density"]
            .sel(rhop=1, method="nearest")
            .std(dim="sample_idx"),
            "VTOR0": self.blobs["toroidal_rotation"]
            .sel(rhop=0, method="nearest")
            .median(dim="sample_idx"),
            "VTOR0_ERR": self.blobs["toroidal_rotation"]
            .sel(rhop=0, method="nearest")
            .std(dim="sample_idx"),
        }
        return result

    def __call__(
        self,
        config=None,
        time_points: np.ndarray = None,
    ):
        logger = logging.getLogger()
        self.config = config
        self.result = self._build_inputs_dict()
        results = []

        # TODO: problem with selecting time index and floating point precision errors
        if time_points is None:
            time_points = iter(self.plasma_profiler.plasma.t)

        for time in time_points:
            self.plasma_profiler.plasma.time_to_calculate = time
            logger.info(f"Time: {time.values:.2f}")
            self.optimiser_context.sample_start_points()
            self.optimiser_context.run()
            results.append(self.optimiser_context.post_process_results())
            self.optimiser_context.reset_optimiser()

        # unpack results and add time axis
        blobs = {}
        for key in results[0]["blobs"].keys():
            _blob = [result["blobs"][key] for result in results]
            blobs[key] = xr.concat(_blob, self.plasma_profiler.plasma.t)
        self.blobs = blobs
        self.midplane_blobs = self.plasma_profiler.map_plasma_profile_to_midplane(blobs)

        self.convergence = {}
        for key in results[0]["convergence"].keys():
            self.convergence[key] = [result["convergence"][key] for result in results]

        opt_samples = {}
        for key in results[0].keys():
            if key == "blobs":
                continue
            _opt_samples = [result[key] for result in results]
            opt_samples[key] = np.array(_opt_samples)
        self.opt_samples = opt_samples

        result = self._build_result_dict()
        self.result = dict(self.result, **result)
        return self.result
