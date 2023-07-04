import pickle

import emcee
import numpy as np
import xarray as xr
import pandas as pd
import flatdict
from scipy.stats import loguniform
from pathlib import Path

from indica.readers.read_st40 import ReadST40
from indica.bayesmodels import BayesModels, get_uniform
from indica.models.interferometry import Interferometry
from indica.models.helike_spectroscopy import Helike_spectroscopy
from indica.models.charge_exchange import ChargeExchange
from indica.models.equilibrium_reconstruction import EquilibriumReconstruction
from indica.models.plasma import Plasma
from indica.converters.line_of_sight import LineOfSightTransform

from abstract_bayes_workflow import BayesWorkflow, sample_with_autocorr

# global configurations
from indica.workflows.bayes_plots import plot_bayes_result

DEFAULT_PHANTOM_PARAMS = {
    "Ne_prof.y0": 5e19,
    "Ne_prof.wcenter": 0.4,
    "Ne_prof.peaking": 2,
    "Ne_prof.y1": 2e18,
    "Ne_prof.yend": 1e18,
    "Ne_prof.wped": 2,
    "Nimp_prof.y0": 3e16,
    "Nimp_prof.y1": 0.5e16,
    "Nimp_prof.wcenter": 0.4,
    "Nimp_prof.wped": 6,
    "Nimp_prof.peaking": 2,
    "Te_prof.y0": 3000,
    "Te_prof.y1": 50,
    "Te_prof.wcenter": 0.4,
    "Te_prof.wped": 3,
    "Te_prof.peaking": 2,
    "Ti_prof.y0": 6000,
    "Ti_prof.y1": 50,
    "Ti_prof.wcenter": 0.4,
    "Ti_prof.wped": 3,
    "Ti_prof.peaking": 2,
}

DEFAULT_PRIORS = {
    "Ne_prof.y0": get_uniform(2e19, 4e20),
    "Ne_prof.y1": get_uniform(1e18, 1e19),
    "Ne_prof.y0/Ne_prof.y1": lambda x1, x2: np.where((x1 > x2 * 2), 1, 0),
    "Ne_prof.wped": get_uniform(1, 6),
    "Ne_prof.wcenter": get_uniform(0.1, 0.8),
    "Ne_prof.peaking": get_uniform(1, 6),
    "ar_conc": loguniform(0.0001, 0.01),
    "Nimp_prof.y0": get_uniform(1e16, 1e18),
    "Nimp_prof.y1": get_uniform(1e15, 2e16),
    "Ne_prof.y0/Nimp_prof.y0": lambda x1, x2: np.where(
        (x1 > x2 * 100) & (x1 < x2 * 1e5), 1, 0
    ),
    "Nimp_prof.y0/Nimp_prof.y1": lambda x1, x2: np.where((x1 > x2), 1, 0),
    "Nimp_prof.wped": get_uniform(1, 6),
    "Nimp_prof.wcenter": get_uniform(0.1, 0.8),
    "Nimp_prof.peaking": get_uniform(1, 20),
    "Nimp_prof.peaking/Ne_prof.peaking": lambda x1, x2: np.where(
        (x1 > x2), 1, 0
    ),  # impurity always more peaked
    "Te_prof.y0": get_uniform(1000, 5000),
    "Te_prof.wped": get_uniform(1, 6),
    "Te_prof.wcenter": get_uniform(0.1, 0.6),
    "Te_prof.peaking": get_uniform(1, 6),
    "Ti_prof.y0/Te_prof.y0": lambda x1, x2: np.where(x1 > x2, 1, 0),  # hot ion mode
    "Ti_prof.y0": get_uniform(3000, 10000),
    "Ti_prof.wped": get_uniform(1, 6),
    "Ti_prof.wcenter": get_uniform(0.1, 0.6),
    "Ti_prof.peaking": get_uniform(1, 20),
}

OPTIMISED_PARAMS = [
    # "Ne_prof.y1",
    "Ne_prof.y0",
    # "Ne_prof.peaking",
    # "Ne_prof.wcenter",
    # "Ne_prof.wped",
    # "ar_conc",
    # "Nimp_prof.y1",
    "Nimp_prof.y0",
    # "Nimp_prof.wcenter",
    # "Nimp_prof.wped",
    # "Nimp_prof.peaking",
    "Te_prof.y0",
    # "Te_prof.wped",
    # "Te_prof.wcenter",
    # "Te_prof.peaking",
    "Ti_prof.y0",
    # "Ti_prof.wped",
    # "Ti_prof.wcenter",
    # "Ti_prof.peaking",
]

OPTIMISED_QUANTITY = ["cxff_pi.ti", "efit.wp", "smmh1.ne"]


class TestWorkflow(BayesWorkflow):
    def __init__(
            self,
            pulse=None,
            param_names=None,
            opt_quantity=None,

            nwalkers=50,
            tstart=0.02,
            tend=0.10,
            dt=0.01,
            tsample=0.06,
            iterations=100,
            burn_frac=0,

            phantoms=False,
            diagnostics=None,
    ):
        self.pulse = pulse
        self.param_names = param_names
        self.opt_quantity = opt_quantity

        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.tsample = tsample
        self.nwalkers = nwalkers
        self.iterations = iterations
        self.burn_frac = burn_frac


        self.phantoms = phantoms
        self.diagnostics = diagnostics

        self.setup_plasma()
        self.save_phantom_profiles()
        self.read_data(diagnostics)
        self.setup_opt_data(phantoms=self.phantoms)
        self.setup_models(diagnostics)
        self.setup_optimiser()

    def setup_plasma(self):
        self.plasma = Plasma(
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            main_ion="h",
            impurities=("ar", "c"),
            impurity_concentration=(
                0.001,
                0.02,
            ),
            full_run=False,
            n_rad=20,
        )
        self.plasma.time_to_calculate = self.plasma.t[
            np.abs(self.tsample - self.plasma.t).argmin()
        ]
        self.plasma.update_profiles(DEFAULT_PHANTOM_PARAMS)
        self.plasma.build_atomic_data(calc_power_loss=False)


    def read_data(self, diagnostics: list):
        self.reader = ReadST40(
            self.pulse, tstart=self.tstart, tend=self.tend, dt=self.dt
        )
        self.reader(diagnostics)
        self.plasma.set_equilibrium(self.reader.equilibrium)
        self.data = self.reader.binned_data

    def setup_opt_data(self, phantoms=False):
        if phantoms:
            self._phantom_data()
        else:
            self._exp_data()

    def setup_models(self, diagnostics: list):
        self.models = {}
        for diag in self.diagnostics:
            if diag == "smmh1":
                # los_transform = self.data["smmh1"]["ne"].transform
                machine_dims = self.plasma.machine_dimensions
                origin = np.array([[-0.38063365, 0.91893092, 0.01]])
                # end = np.array([[0,  0, 0.01]])
                direction = np.array([[0.38173721, -0.92387953, -0.02689453]])
                los_transform = LineOfSightTransform(
                    origin[:, 0],
                    origin[:, 1],
                    origin[:, 2],
                    direction[:, 0],
                    direction[:, 1],
                    direction[:, 2],
                    name="",
                    machine_dimensions=machine_dims,
                    passes=2,
                )
                los_transform.set_equilibrium(self.plasma.equilibrium)
                model = Interferometry(name=diag)
                model.set_los_transform(los_transform)
                model.plasma = self.plasma

            elif diag == "xrcs":
                los_transform = self.data["xrcs"]["te_kw"].transform
                model = Helike_spectroscopy(
                    name="xrcs",
                    window_masks=[slice(0.394, 0.396)],
                    window_vector=self.data[diag][
                                      "spectra"
                                  ].wavelength.values
                                  * 0.1,
                )
                model.set_los_transform(los_transform)
                model.plasma = self.plasma

            elif diag == "efit":
                model = EquilibriumReconstruction(name="efit")
                model.plasma = self.plasma

            elif diag == "cxff_pi":
                transform = self.data[diag]["ti"].transform
                transform.set_equilibrium(self.plasma.equilibrium)
                model = ChargeExchange(name=diag, element="ar")
                model.set_transect_transform(transform)
                model.plasma = self.plasma
            else:
                raise ValueError(f"{diag} not found in setup_models")

            self.models[diag] = model

    def setup_optimiser(self, ):

        self.bayesopt = BayesModels(
            plasma=self.plasma,
            data=self.opt_data,
            diagnostic_models=[*self.models.values()],
            quant_to_optimise=OPTIMISED_QUANTITY,
            priors=DEFAULT_PRIORS,
        )

        ndim = len(OPTIMISED_PARAMS)

        self.move = [(emcee.moves.StretchMove(), 0.9), (emcee.moves.DEMove(), 0.1)]
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers,
            ndim,
            log_prob_fn=self.bayesopt.ln_posterior,
            parameter_names=OPTIMISED_PARAMS,
            moves=self.move,
        )

        self.start_points = self.bayesopt.sample_from_priors(
            OPTIMISED_PARAMS, size=self.nwalkers
        )

    def _phantom_data(self, noise=False, noise_factor=0.1):
        self.opt_data = {}
        if "smmh1" in self.diagnostics:
            self.opt_data["smmh1.ne"] = (
                self.models["smmh1"]()
                    .pop("ne")
                    .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
        if "xrcs" in self.diagnostics:
            self.opt_data["xrcs.spectra"] = (
                self.models["xrcs"]()
                    .pop("spectra")
                    .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
            self.opt_data["xrcs.background"] = None
        if "cxff_pi" in self.diagnostics:
            cxrs_data = (
                self.models["cxff_pi"]()
                    .pop("ti")
                    .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
            self.opt_data["cxff_pi.ti"] = cxrs_data.where(cxrs_data.channel == 2)
        if "efit" in self.diagnostics:
            self.opt_data["efit.wp"] = (
                self.models["efit"]()
                    .pop("wp")
                    .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )

        if noise:
            self.opt_data["smmh1.ne"] = self.opt_data["smmh1.ne"] + self.opt_data[
                "smmh1.ne"
            ].max().values * np.random.normal(0, noise_factor, None)
            self.opt_data["xrcs.spectra"] = self.opt_data[
                                                "xrcs.spectra"
                                            ] + np.random.normal(
                0,
                np.sqrt(self.opt_data["xrcs.spectra"].values[0,]),
                self.opt_data["xrcs.spectra"].shape[1],
            )
            self.opt_data["cxff_pi.ti"] = self.opt_data[
                                              "cxff_pi.ti"
                                          ] + self.opt_data["cxff_pi.ti"].max().values * np.random.normal(
                0, noise_factor, self.opt_data["cxff_pi.ti"].shape[1]
            )
            self.opt_data["efit.wp"] = self.opt_data["efit.wp"] + self.opt_data[
                "efit.wp"
            ].max().values * np.random.normal(0, noise_factor, None)

        self.phantom_profiles = {}
        for key in ["electron_density", "impurity_density", "electron_temperature", "ion_temperature",
                    "ion_density", "fast_density", "neutral_density"]:
            self.phantom_profiles[key] = getattr(self.plasma, key).sel(
                t=self.plasma.time_to_calculate
            ).copy()

    def _exp_data(self):
        self.opt_data = flatdict.FlatDict(self.data, ".")
        if "xrcs" in self.diagnostics:
            self.opt_data["xrcs.spectra"]["wavelength"] = (
                    self.opt_data["xrcs.spectra"].wavelength * 0.1
            )
            background = self.opt_data["xrcs.spectra"].where(
                (self.opt_data["xrcs.spectra"].wavelength < 0.392)
                & (self.opt_data["xrcs.spectra"].wavelength > 0.388),
                drop=True,
            )
            self.opt_data["xrcs.background"] = background.mean(dim="wavelength")
            self.opt_data["xrcs.spectra"]["error"] = np.sqrt(
                self.opt_data["xrcs.spectra"] + background.std(dim="wavelength") ** 2
            )

        if "cxff_pi" in self.diagnostics:
            self.opt_data["cxff_pi"]["ti"] = self.opt_data["cxff_pi"]["ti"].where(
                self.opt_data["cxff_pi"]["ti"].channel == 2, drop=True
            )

    def __call__(self, filepath = "./results/test/", **kwargs):
        self.run_sampler()
        self.save_pickle(filepath=filepath)
        plot_bayes_result(self.result, filepath)
        return self.result



if __name__ == "__main__":
    run = TestWorkflow(
        pulse=10009,
        dt=0.005,
        tsample=0.060,
        diagnostics=["efit", "smmh1", "cxff_pi"],

        iterations=20,
        nwalkers=10,
        burn_frac=0.10,
        param_names = OPTIMISED_PARAMS,
        opt_quantity=OPTIMISED_QUANTITY
    )

    test = run(filepath="./results/test/")

    flattest = flatdict.FlatDict(test, delimiter=".")
    for key in flattest.keys():
        print(key)
