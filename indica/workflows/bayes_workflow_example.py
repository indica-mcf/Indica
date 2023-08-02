import emcee
import numpy as np
import flatdict
import copy
from scipy.stats import loguniform

from indica.readers.read_st40 import ReadST40
from indica.bayesmodels import BayesModels, get_uniform
from indica.workflows.bayes_plots import plot_bayes_result
from indica.models.interferometry import Interferometry
from indica.models.helike_spectroscopy import Helike_spectroscopy
from indica.models.charge_exchange import ChargeExchange
from indica.models.equilibrium_reconstruction import EquilibriumReconstruction
from indica.models.plasma import Plasma
from indica.converters.line_of_sight import LineOfSightTransform

from indica.workflows.abstract_bayes_workflow import AbstractBayesWorkflow
from indica.writers.bda_tree import create_nodes, write_nodes, check_analysis_run

# global configurations
DEFAULT_PROFILE_PARAMS = {
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
    "Nimp_prof.y0": loguniform(1e16, 1e18),
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
    "Te_prof.wcenter",
    "Te_prof.peaking",
    "Ti_prof.y0",
    "Ti_prof.wped",
    "Ti_prof.wcenter",
    "Ti_prof.peaking",
]

OPTIMISED_QUANTITY = [
    "xrcs.spectra",
    "cxff_pi.ti",
    "efit.wp",
    "smmh1.ne",
]


class BayesWorkflowExample(AbstractBayesWorkflow):
    def __init__(
            self,
            pulse: int = None,
            pulse_to_write: bool = None,
            run: str = "RUN01",
            diagnostics: list = None,
            param_names: list = None,
            opt_quantity: list = None,
            priors: dict = None,
            profile_params: dict = None,
            phantoms: bool = False,
            model_kwargs: dict = {},

            nwalkers: int = 50,
            tstart: float = 0.02,
            tend: float = 0.10,
            dt: float = 0.01,
            tsample: float = 0.06,
            iterations: int = 100,
            burn_frac: float = 0,

            mds_write: bool = False,
            plot: bool = True,
            sample_high_density: bool = False,
    ):
        self.pulse = pulse
        self.pulse_to_write = pulse_to_write
        self.run = run
        self.diagnostics = diagnostics
        self.param_names = param_names
        self.opt_quantity = opt_quantity
        self.priors = priors
        self.profile_params = profile_params
        self.model_kwargs = model_kwargs
        self.phantoms = phantoms

        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.tsample = tsample
        self.nwalkers = nwalkers
        self.iterations = iterations
        self.burn_frac = burn_frac

        self.mds_write = mds_write
        self.plot = plot
        self.sample_high_density = sample_high_density

        for attribute in ["pulse", "param_names", "opt_quantity", "priors", "diagnostics", "profile_params"]:
            if getattr(self, attribute) is None:
                raise ValueError(f"{attribute} needs to be defined")

        self.read_data(self.diagnostics)
        self.setup_plasma()
        self.save_phantom_profiles()
        self.setup_models(self.diagnostics)
        self.setup_opt_data(phantoms=self.phantoms)
        self.setup_optimiser(self.model_kwargs)

    def setup_plasma(self):
        self.plasma = Plasma(
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            main_ion="h",
            impurities=("ar", "c"),
            impurity_concentration=(
                0.001,
                0.04,
            ),
            full_run=False,
            n_rad=20,
        )
        self.plasma.time_to_calculate = self.plasma.t[
            np.abs(self.tsample - self.plasma.t).argmin()
        ]
        self.plasma.set_equilibrium(self.reader.equilibrium)
        self.plasma.update_profiles(self.profile_params)
        self.plasma.build_atomic_data(calc_power_loss=False)

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

            elif diag == "xrcs":
                los_transform = self.data["xrcs"]["te_kw"].transform
                model = Helike_spectroscopy(
                    name="xrcs",
                    window_masks=[slice(0.394, 0.396)],
                    window_vector=self.data[diag][
                                      "spectra"
                                  ].wavelength.values * 0.1,
                )
                model.set_los_transform(los_transform)

            elif diag == "efit":
                model = EquilibriumReconstruction(name="efit")

            elif diag == "cxff_pi":
                transform = self.data[diag]["ti"].transform
                transform.set_equilibrium(self.plasma.equilibrium)
                model = ChargeExchange(name=diag, element="ar")
                model.set_transect_transform(transform)
            else:
                raise ValueError(f"{diag} not found in setup_models")
            model.plasma = self.plasma
            self.models[diag] = model

    def setup_opt_data(self, phantoms=False):
        if phantoms:
            self.opt_data = self._phantom_data()
        else:
            self.opt_data = self._exp_data()

    def _phantom_data(self, noise=False, noise_factor=0.1):
        opt_data = {}
        if "smmh1" in self.diagnostics:
            opt_data["smmh1.ne"] = (
                self.models["smmh1"]()
                    .pop("ne")
                    .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
        if "xrcs" in self.diagnostics:
            opt_data["xrcs.spectra"] = (
                self.models["xrcs"]()
                    .pop("spectra")
                    .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
            opt_data["xrcs.spectra"]["error"] = np.sqrt(
                opt_data["xrcs.spectra"]
            )
        if "cxff_pi" in self.diagnostics:
            cxrs_data = (
                self.models["cxff_pi"]()
                    .pop("ti")
                    .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
            opt_data["cxff_pi.ti"] = cxrs_data.where(cxrs_data.channel == 2, drop=True)
        if "efit" in self.diagnostics:
            opt_data["efit.wp"] = (
                self.models["efit"]()
                    .pop("wp")
                    .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )

        if noise:
            opt_data["smmh1.ne"] = opt_data["smmh1.ne"] + self.opt_data[
                "smmh1.ne"
            ].max().values * np.random.normal(0, noise_factor, None)
            opt_data["xrcs.spectra"] = opt_data[
                                           "xrcs.spectra"
                                       ] + np.random.normal(
                0,
                np.sqrt(opt_data["xrcs.spectra"].values[0,]),
                opt_data["xrcs.spectra"].shape[1],
            )
            opt_data["cxff_pi.ti"] = opt_data[
                                         "cxff_pi.ti"
                                     ] + opt_data["cxff_pi.ti"].max().values * np.random.normal(
                0, noise_factor, opt_data["cxff_pi.ti"].shape[1]
            )
            opt_data["efit.wp"] = opt_data["efit.wp"] + opt_data[
                "efit.wp"
            ].max().values * np.random.normal(0, noise_factor, None)
        return opt_data

    def _exp_data(self):
        opt_data = flatdict.FlatDict(self.data, ".")
        if "xrcs" in self.diagnostics:
            opt_data["xrcs.spectra"]["wavelength"] = (
                    opt_data["xrcs.spectra"].wavelength * 0.1
            )
            background = opt_data["xrcs.spectra"].where(
                (opt_data["xrcs.spectra"].wavelength < 0.392)
                & (opt_data["xrcs.spectra"].wavelength > 0.388),
                drop=True,
            )
            self.model_kwargs["xrcs_background"] = background.mean(dim="wavelength").sel(t=self.plasma.time_to_calculate)
            opt_data["xrcs.spectra"]["error"] = np.sqrt(
                opt_data["xrcs.spectra"] + background.std(dim="wavelength") ** 2
            )

        if "cxff_pi" in self.diagnostics:
            opt_data["cxff_pi"]["ti"] = opt_data["cxff_pi"]["ti"].where(
                opt_data["cxff_pi"]["ti"].channel == 2, drop=True
            )
        return opt_data

    def setup_optimiser(self, model_kwargs):

        self.bayesmodel = BayesModels(
            plasma=self.plasma,
            data=self.opt_data,
            diagnostic_models=[*self.models.values()],
            quant_to_optimise=self.opt_quantity,
            priors=self.priors,
        )

        ndim = len(self.param_names)
        self.move = [(emcee.moves.StretchMove(), 0.9), (emcee.moves.DEMove(), 0.1)]
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers,
            ndim,
            log_prob_fn=self.bayesmodel.ln_posterior,
            parameter_names=self.param_names,
            moves=self.move,
            kwargs=model_kwargs,
        )
        self.start_points = self._sample_start_points(sample_high_density=self.sample_high_density)

    def _sample_start_points(self, sample_high_density: bool=True):
        if sample_high_density:
            self.start_points = self.bayesopt.sample_from_high_density_region(self.param_names, self.sampler,
                                                                              self.nwalkers)
        else:
            self.start_points = self.bayesopt.sample_from_priors(
                self.param_names, size=self.nwalkers
            )

    def __call__(self, filepath="./results/test/", **kwargs):

        if self.mds_write:
            # check_analysis_run(self.pulse, self.run)
            self.node_structure = create_nodes(pulse_to_write=self.pulse_to_write,
                                               diagnostic_quantities=self.opt_quantity,
                                               mode="NEW")

        self.run_sampler()
        self.save_pickle(filepath=filepath)

        if self.plot:  # currently requires result with DataArrays
            plot_bayes_result(self.result, filepath)

        self.result = self.dict_of_dataarray_to_numpy(self.result)
        if self.mds_write:
            write_nodes(self.pulse_to_write, self.node_structure, self.result)

        return self.result


if __name__ == "__main__":
    run = BayesWorkflowExample(
        pulse=10009,
        pulse_to_write=23000101,
        run="RUN01",
        diagnostics=["xrcs", "efit", "smmh1", "cxff_pi"],
        opt_quantity=OPTIMISED_QUANTITY,
        param_names=OPTIMISED_PARAMS,
        profile_params=DEFAULT_PROFILE_PARAMS,
        priors=DEFAULT_PRIORS,
        model_kwargs={"xrcs_moment_analysis":False, },
        phantoms=True,

        iterations=200,
        nwalkers=50,
        burn_frac=0.10,
        dt=0.005,
        tsample=0.060,

        mds_write=True,
        plot=True,
        sample_high_density=False,
    )
    results = run(filepath="./results/test/", )
