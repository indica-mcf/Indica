from typing import Any, List, Tuple
from typing import Dict

import emcee
import flatdict
import numpy as np
from scipy.stats import loguniform
import xarray as xr

from indica.bayesmodels import BayesModels
from indica.bayesmodels import get_uniform
from indica.models.charge_exchange import ChargeExchange
from indica.models.charge_exchange import pi_transform_example
from indica.models.equilibrium_reconstruction import EquilibriumReconstruction
from indica.models.helike_spectroscopy import helike_transform_example
from indica.models.helike_spectroscopy import HelikeSpectrometer
from indica.models.interferometry import Interferometry
from indica.models.interferometry import smmh1_transform_example
from indica.models.thomson_scattering import ThomsonScattering
from indica.models.thomson_scattering import ts_transform_example
from indica.models.plasma import Plasma
from indica.workflows.abstract_bayes_workflow import AbstractBayesWorkflow
from indica.workflows.bayes_plots import plot_bayes_result
from indica.writers.bda_tree import create_nodes
from indica.writers.bda_tree import write_nodes
from indica.readers.read_st40 import ReadST40
from indica.equilibrium import Equilibrium



# global configurations
DEFAULT_PROFILE_PARAMS = {
    "Ne_prof.y0": 5e19,
    "Ne_prof.y1": 2e18,
    "Ne_prof.yend": 1e18,
    "Ne_prof.wped": 3,
    "Ne_prof.wcenter": 0.3,
    "Ne_prof.peaking": 1.2,

    "Nimp_prof.y0": 1e17,
    "Nimp_prof.y1": 1e15,
    "Nimp_prof.yend": 1e15,
    "Nimp_prof.wcenter": 0.3,
    "Nimp_prof.wped": 3,
    "Nimp_prof.peaking": 2,

    "Te_prof.y0": 3000,
    "Te_prof.y1": 50,
    "Te_prof.yend": 10,
    "Te_prof.wcenter": 0.2,
    "Te_prof.wped": 3,
    "Te_prof.peaking": 1.5,

    "Ti_prof.y0": 6000,
    "Ti_prof.y1": 50,
    "Ti_prof.yend": 10,
    "Ti_prof.wcenter": 0.2,
    "Ti_prof.wped": 3,
    "Ti_prof.peaking": 1.5,
}

DEFAULT_PRIORS = {
    "Ne_prof.y0": get_uniform(2e19, 4e20),
    "Ne_prof.y1": get_uniform(1e18, 2e19),
    "Ne_prof.y0/Ne_prof.y1": lambda x1, x2: np.where((x1 > x2 * 2), 1, 0),
    "Ne_prof.wped": get_uniform(2, 6),
    "Ne_prof.wcenter": get_uniform(0.2, 0.4),
    "Ne_prof.peaking": get_uniform(1, 4),
    "Nimp_prof.y0": loguniform(1e15, 1e18),
    "Nimp_prof.y1": loguniform(1e14, 1e16),
    "Ne_prof.y0/Nimp_prof.y0": lambda x1, x2: np.where(
        (x1 > x2 * 100) & (x1 < x2 * 1e5), 1, 0
    ),
    "Nimp_prof.y0/Nimp_prof.y1": lambda x1, x2: np.where((x1 > x2), 1, 0),
    "Nimp_prof.wped": get_uniform(2, 6),
    "Nimp_prof.wcenter": get_uniform(0.2, 0.4),
    "Nimp_prof.peaking": get_uniform(1, 6),
    "Nimp_prof.peaking/Ne_prof.peaking": lambda x1, x2: np.where(
        (x1 > x2), 1, 0
    ),  # impurity always more peaked

    "Te_prof.y0": get_uniform(1000, 5000),
    "Te_prof.wped": get_uniform(2, 6),
    "Te_prof.wcenter": get_uniform(0.2, 0.4),
    "Te_prof.peaking": get_uniform(1, 4),
    # "Ti_prof.y0/Te_prof.y0": lambda x1, x2: np.where(x1 > x2, 1, 0),  # hot ion mode
    "Ti_prof.y0": get_uniform(1000, 10000),
    "Ti_prof.wped": get_uniform(2, 6),
    "Ti_prof.wcenter": get_uniform(0.2, 0.4),
    "Ti_prof.peaking": get_uniform(1, 6),
    "xrcs.pixel_offset": get_uniform(-4.01, -4.0),
}

OPTIMISED_PARAMS = [
    "Ne_prof.y1",
    "Ne_prof.y0",
    "Ne_prof.peaking",
    # "Ne_prof.wcenter",
    "Ne_prof.wped",
    # "Nimp_prof.y1",
    "Nimp_prof.y0",
    # "Nimp_prof.wcenter",
    # "Nimp_prof.wped",
    "Nimp_prof.peaking",
    "Te_prof.y0",
    # "Te_prof.wped",
    "Te_prof.wcenter",
    "Te_prof.peaking",
    "Ti_prof.y0",
    # "Ti_prof.wped",
    "Ti_prof.wcenter",
    "Ti_prof.peaking",
]
OPTIMISED_QUANTITY = [
    "xrcs.spectra",
    "cxff_pi.ti",
    "efit.wp",
    # "smmh1.ne",
    "ts.te",
    "ts.ne",
]


class BayesWorkflowExample(AbstractBayesWorkflow):
    def __init__(
        self,
        diagnostics: list,
        param_names: list,
        opt_quantity: list,
        priors: dict,
        profile_params: dict,
        pulse: int = None,
        tstart=0.02,
        tend=0.10,
        dt=0.005,

        phantoms: bool = False,
        fast_particles = False,
        astra_run=None,
        astra_pulse_range=13000000,
        astra_equilibrium=False,
        efit_revision = 0,
        set_ts_profiles = False,
        set_all_profiles=False,
        astra_wp = False,
    ):
        self.pulse = pulse
        self.diagnostics = diagnostics
        self.param_names = param_names
        self.opt_quantity = opt_quantity
        self.priors = priors
        self.profile_params = profile_params
        self.tstart = tstart
        self.tend = tend
        self.dt = dt

        self.phantoms = phantoms
        self.fast_particles = fast_particles
        self.astra_run= astra_run
        self.astra_pulse_range = astra_pulse_range
        self.astra_equilibrium = astra_equilibrium
        self.efit_revision = efit_revision
        self.set_ts_profiles = set_ts_profiles
        self.set_all_profiles = set_all_profiles
        self.astra_wp = astra_wp

        self.model_kwargs = {}

        for attribute in [
            "param_names",
            "opt_quantity",
            "priors",
            "diagnostics",
            "profile_params",
        ]:
            if getattr(self, attribute) is None:
                raise ValueError(f"{attribute} needs to be defined")

        if self.pulse is None and self.phantoms is False:
            raise ValueError(
                "Set phantoms to True when running phantom plasma i.e. pulse=None"
            )

        # TODO: Add some abstraction here
        if pulse is None:
            print("Running in test mode")
            example_transforms = {
                "xrcs": helike_transform_example(1),
                "smmh1": smmh1_transform_example(1),
                "cxff_pi": pi_transform_example(5),
                "ts":ts_transform_example(11),
            }
            self.read_test_data(
                example_transforms, tstart=self.tstart, tend=self.tend, dt=self.dt
            )
        else:
            self.read_data(
                self.diagnostics, tstart=self.tstart, tend=self.tend, dt=self.dt
            )
        if self.efit_revision != 0:
            efit_equilibrium = self.reader.reader_equil.get("", "efit", self.efit_revision)
            self.equilibrium = Equilibrium(efit_equilibrium)

        self.setup_models(self.diagnostics)

    def setup_plasma(
        self,
        tstart=None,
        tend=None,
        dt=None,
        tsample=None,
        main_ion="h",
        impurities=("ar", "c"),
        impurity_concentration=(0.001, 0.04),
        n_rad=20,
        **kwargs,
    ):
        if not all([tstart, tend, dt]):
            tstart = self.tstart
            tend = self.tend
            dt = self.dt

        # TODO: move to plasma.py
        self.plasma = Plasma(
            tstart=tstart,
            tend=tend,
            dt=dt,
            main_ion=main_ion,
            impurities=impurities,
            impurity_concentration=impurity_concentration,
            full_run=False,
            n_rad=n_rad,
        )

        if tsample == None:
            self.tsample = self.plasma.t
        else:
            self.tsample = self.plasma.t[
                np.abs(tsample - self.plasma.t).argmin()
            ]

        self.plasma.time_to_calculate = self.tsample
        self.plasma.set_equilibrium(self.equilibrium)
        self.plasma.update_profiles(self.profile_params)
        if self.fast_particles:
            self._init_fast_particles(run=self.astra_run)
            self.plasma.update_profiles({})

        self.plasma.build_atomic_data(calc_power_loss=False)
        self.save_phantom_profiles()

    def setup_models(self, diagnostics: list):
        self.models: Dict[str, Any] = {}
        model: Any = None
        for diag in diagnostics:
            if diag == "smmh1":
                los_transform = self.transforms[diag]
                # machine_dims = ((0.15, 0.95), (-0.7, 0.7))
                # origin = np.array([[-0.38063365, 0.91893092, 0.01]])
                # # end = np.array([[0,  0, 0.01]])
                # direction = np.array([[0.38173721, -0.92387953, -0.02689453]])
                # los_transform = LineOfSightTransform(
                #     origin[:, 0],
                #     origin[:, 1],
                #     origin[:, 2],
                #     direction[:, 0],
                #     direction[:, 1],
                #     direction[:, 2],
                #     name="",
                #     machine_dimensions=machine_dims,
                #     passes=2,
                # )
                los_transform.set_equilibrium(self.equilibrium, force=True)
                model = Interferometry(name=diag)
                model.set_los_transform(los_transform)

            elif diag == "xrcs":
                los_transform = self.transforms[diag]
                los_transform.set_equilibrium(self.equilibrium, force=True)
                window = None
                if hasattr(self, "data"):
                    if diag in self.data.keys():
                        window = self.data[diag]["spectra"].wavelength.values

                model = HelikeSpectrometer(
                    name="xrcs",
                    window_masks=[slice(0.394, 0.396)],
                    window=window,
                )
                model.set_los_transform(los_transform)

            elif diag == "efit":
                model = EquilibriumReconstruction(name="efit")

            elif diag == "cxff_pi":
                transform = self.transforms[diag]
                transform.set_equilibrium(self.equilibrium, force=True)
                model = ChargeExchange(name=diag, element="ar")
                model.set_transect_transform(transform)

            elif diag == "cxff_tws_c":
                transform = self.transforms[diag]
                transform.set_equilibrium(self.equilibrium, force=True)
                model = ChargeExchange(name=diag, element="c")
                model.set_transect_transform(transform)

            elif diag == "ts":
                transform = self.transforms[diag]
                transform.set_equilibrium(self.equilibrium, force=True)
                model = ThomsonScattering(name=diag, )
                model.set_transect_transform(transform)
            else:
                raise ValueError(f"{diag} not found in setup_models")
            self.models[diag] = model

    def _init_fast_particles(self, run="RUN602", ):

        st40_code = ReadST40(self.astra_pulse_range + self.pulse, self.tstart-self.dt, self.tend+self.dt, dt=self.dt, tree="astra")
        astra_data = st40_code.get_raw_data("", "astra", run)

        if self.astra_equilibrium:
            self.equilibrium = Equilibrium(astra_data)
            self.plasma.equilibrium = self.equilibrium

        st40_code.bin_data_in_time(["astra"], self.tstart, self.tend, self.dt)
        code_data = st40_code.binned_data["astra"]
        Nf = (
            code_data["nf"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
            * 1.0e19
        )
        self.plasma.fast_density.values = Nf.values
        Nn = (
            code_data["nn"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
            * 1.0e19
        )
        self.plasma.neutral_density.values = Nn.values
        Pblon = code_data["pblon"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
        self.plasma.pressure_fast_parallel.values = Pblon.values
        Pbper = code_data["pbper"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
        self.plasma.pressure_fast_perpendicular.values = Pbper.values
        self.astra_data = code_data

        if self.set_ts_profiles:
            overwritten_params = [param for param in self.param_names if any(xs in param for xs in ["Te", "Ne"])]
            if any(overwritten_params):
                raise ValueError(f"Te/Ne set by TS but then the following params overwritten: {overwritten_params}")
            Te = code_data["te"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e3
            self.plasma.Te_prof = lambda: Te.values
            Ne = code_data["ne"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e19
            self.plasma.Ne_prof = lambda:  Ne.values

        if self.set_all_profiles:
            overwritten_params = [param for param in self.param_names if any(xs in param for xs in ["Te", "Ti", "Ne", "Nimp"])]
            if any(overwritten_params):
                raise ValueError(f"Te/Ne set by TS but then the following params overwritten: {overwritten_params}")
            Te = code_data["te"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e3
            self.plasma.Te_prof = lambda: Te.values
            Ne = code_data["ne"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e19
            self.plasma.Ne_prof = lambda: Ne.values
            Ti = code_data["ti"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e3
            self.plasma.Ti_prof = lambda: Ti.values
            Nimp = code_data["niz1"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e19
            self.plasma.Nimp_prof = lambda: Nimp.values

    def setup_opt_data(self, phantoms=False, **kwargs):
        if not hasattr(self, "plasma"):
            raise ValueError("Missing plasma object required for setup_opt_data")
        for model in self.models.values():  # Maybe refactor here...
            model.plasma = self.plasma

        if phantoms:
            self.opt_data = self._phantom_data(**kwargs)
        else:
            self.opt_data = self._exp_data(**kwargs)

    def _phantom_data(self, noise=False, noise_factor=0.1, **kwargs):
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
            opt_data["xrcs.spectra"]["error"] = np.sqrt(opt_data["xrcs.spectra"])
        if "cxff_pi" in self.diagnostics:
            cxrs_data = (
                self.models["cxff_pi"]()
                .pop("ti")
                .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
            opt_data["cxff_pi.ti"] = cxrs_data.where(cxrs_data.channel == 2, drop=True)
            
        if "ts" in self.diagnostics:
            _ts_data = self.models["ts"]()
            ts_data = {key: _ts_data[key].expand_dims(dim={"t": [self.plasma.time_to_calculate]}) for key in ["te", "ne"]}
            opt_data["ts.te"] = ts_data["te"]
            opt_data["ts.ne"] = ts_data["ne"]
            opt_data["ts.te"]["error"] = opt_data["ts.te"] / opt_data["ts.te"] * (
                        0.10 * opt_data["ts.te"].max(dim="channel"))
            opt_data["ts.ne"]["error"] = opt_data["ts.ne"] / opt_data["ts.ne"] * (
                        0.10 * opt_data["ts.ne"].max(dim="channel"))
            
        if "efit" in self.diagnostics:
            opt_data["efit.wp"] = (
                self.models["efit"]()
                .pop("wp")
                .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )

        # TODO: add chers

        if noise:
            #TODO: add TS
            opt_data["smmh1.ne"] = opt_data["smmh1.ne"] + opt_data[
                "smmh1.ne"
            ].max().values * np.random.normal(0, noise_factor, None)
            opt_data["xrcs.spectra"] = opt_data["xrcs.spectra"] + np.random.normal(
                0,
                np.sqrt(
                    opt_data["xrcs.spectra"].values[
                        0,
                    ]
                ),
                opt_data["xrcs.spectra"].shape[1],
            )
            opt_data["cxff_pi.ti"] = opt_data["cxff_pi.ti"] + opt_data[
                "cxff_pi.ti"
            ].max().values * np.random.normal(
                0, noise_factor, opt_data["cxff_pi.ti"].shape[1]
            )
            opt_data["efit.wp"] = opt_data["efit.wp"] + opt_data[
                "efit.wp"
            ].max().values * np.random.normal(0, noise_factor, None)

        return opt_data

    def _exp_data(self, **kwargs):
        opt_data = flatdict.FlatDict(self.data, ".")
        if "xrcs" in self.diagnostics:
            opt_data["xrcs.spectra"]["wavelength"] = (
                opt_data["xrcs.spectra"].wavelength
            )
            background = opt_data["xrcs.spectra"].where(
                (opt_data["xrcs.spectra"].wavelength < 0.392)
                & (opt_data["xrcs.spectra"].wavelength > 0.388),
                drop=True,
            )
            self.model_kwargs["xrcs.background"] = background.mean(
                dim="wavelength"
            ).sel(t=self.plasma.time_to_calculate)
            opt_data["xrcs.spectra"]["error"] = np.sqrt(
                opt_data["xrcs.spectra"] + background.std(dim="wavelength") ** 2
            )

        if "cxff_pi" in self.diagnostics:
            opt_data["cxff_pi"]["ti"] = opt_data["cxff_pi"]["ti"].where(
                opt_data["cxff_pi"]["ti"] != 0,
            )

            opt_data["cxff_pi"]["ti"] = opt_data["cxff_pi"]["ti"].where(
                opt_data["cxff_pi"]["ti"].channel > 3,
            )
            opt_data["cxff_pi"]["ti"] = opt_data["cxff_pi"]["ti"].where(
                opt_data["cxff_pi"]["ti"].channel < 6,
            )

        if "cxff_tws_c" in self.diagnostics:
            opt_data["cxff_tws_c"]["ti"] = opt_data["cxff_tws_c"]["ti"].where(
                opt_data["cxff_tws_c"]["ti"] != 0,
            )
            opt_data["cxff_tws_c"]["ti"] = opt_data["cxff_tws_c"]["ti"].where(
                opt_data["cxff_tws_c"]["ti"].channel < 2,
            )

        if "ts" in self.diagnostics:
            # TODO: fix error, for now flat error
            opt_data["ts.te"] = opt_data["ts.te"].where(opt_data["ts.te"].channel >19, drop=True)
            opt_data["ts.ne"] = opt_data["ts.ne"].where(opt_data["ts.ne"].channel > 19, drop=True)

            opt_data["ts.te"]["error"] = opt_data["ts.te"].max(dim="channel") * 0.05
            opt_data["ts.ne"]["error"] = opt_data["ts.ne"].max(dim="channel") * 0.05

        if self.astra_wp:
            opt_data["efit.wp"] = self.astra_data["wth"]

        return opt_data

    def setup_optimiser(
        self,
        model_kwargs,
        nwalkers=50,
        burn_frac=0.10,
        sample_method="random",
        **kwargs,
    ):
        self.model_kwargs = model_kwargs
        self.nwalkers = nwalkers
        self.burn_frac = burn_frac
        self.sample_method = sample_method

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
            nwalkers,
            ndim,
            log_prob_fn=self.bayesmodel.ln_posterior,
            parameter_names=self.param_names,
            moves=self.move,
            kwargs=model_kwargs,
        )



    def _sample_start_points(self, sample_method: str = "random", nsamples=100, **kwargs):
        if sample_method == "high_density":
            start_points = self.bayesmodel.sample_from_high_density_region(
                self.param_names, self.sampler, self.nwalkers, nsamples=nsamples
            )

        elif sample_method == "ga":
            self.start_points = self._sample_start_points(sample_method="random", **kwargs)
            samples_in_weird_format = self.ga_opt(**kwargs)
            sample_start_points = np.array([idx[1] for idx in samples_in_weird_format])

            start_points = np.random.normal(
                np.mean(sample_start_points, axis=0),
                np.std(sample_start_points, axis=0),
                size=(self.nwalkers, sample_start_points.shape[1]),
            )


        elif sample_method == "random":
            start_points = self.bayesmodel.sample_from_priors(
                self.param_names, size=self.nwalkers
            )

        else:
            print(f"Sample method: {sample_method} not recognised, Defaulting to random sampling")
            start_points = self.bayesmodel.sample_from_priors(
                self.param_names, size=self.nwalkers
            )
        return start_points


    def ga_opt(self, num_gens=30, popsize=50, sols_to_return=5, mutation_probability=None, **kwargs) -> list(tuple((float, []))):
        """Runs the GA optimization, and returns a number of the best solutions. Uses
        a population convergence stopping criteria: fitness does not improve in 3 successive generations, we stop.

        Args:
            num_gens (int, optional): Maximum number of generations to run. Defaults to 30.
            popsize (int, optional): Population size. Defaults to 50.
            sols_to_return (int, optional): How many of the best solutions the function shall return. Defaults to 5.

        Returns:
            list(tuple(float, np.Array(float))): list of tuples, where first element is fitness, second np.array of the parameters.
        """

        import pygad
        import time

        # Packaged evaluation function
        def idiot_proof(ga_instance, x, sol_idx):
            res, _ = self.bayesmodel.ln_posterior(dict(zip(self.param_names, x)))
            # print(-res)
            return float(res)

        print(f"Running GA for a maximum of {num_gens} generations of {popsize} individuals each.")

        # Initialize the GA instance
        ga_instance = pygad.GA(num_generations=num_gens,
                               num_parents_mating=20,
                               sol_per_pop=popsize,
                               num_genes=len(self.start_points[0]),
                               fitness_func=idiot_proof,
                               initial_population=self.start_points,
                               save_best_solutions=True,
                               stop_criteria="saturate_5",
                               mutation_probability=mutation_probability)

        st = time.time()
        # Execute
        ga_instance.run()

        print(
            f"Time ran: {time.time() - st:.2f} seconds. Ran total of {ga_instance.generations_completed} generations.")

        # Saves the fitness evolution plot
        # figure = ga_instance.plot_fitness()
        # figure.savefig(f'GA_plot.png', dpi=300)

        # Organizing all the non-inf individuals from the last generation
        feasible_indices = [i for i in range(len(ga_instance.last_generation_fitness)) if
                            ga_instance.last_generation_fitness[i] != -np.inf]
        feasible_individuals = [ga_instance.population[i] for i in feasible_indices]
        feasible_fitnesses = [ga_instance.last_generation_fitness[i] for i in feasible_indices]

        # for i, item in enumerate(feasible_individuals_with_keywords):
        #    item["fitness"]=feasible_fitnesses[i
        # feasible_individuals_with_keywords=sorted(feasible_individuals_with_keywords,key= lambda d:d['fitness'],reverse=True)
        # feasible_individuals_and_fitnesses=[tuple(feasible_fitnesses[i],feasible_individuals[i]) for i in len(feasible_individuals)]

        # Combining the last individuals to a collection and sorting
        feasible_individuals_and_fitnesses = []
        for i in range(len(feasible_fitnesses)):
            feasible_individuals_and_fitnesses.append(tuple((feasible_fitnesses[i], feasible_individuals[i])))
        feasible_individuals_and_fitnesses = sorted(feasible_individuals_and_fitnesses, key=lambda x: x[0],
                                                    reverse=True)

        return feasible_individuals_and_fitnesses[:sols_to_return]

    def __call__(
        self,
        filepath="./results/test/",
        run=None,
        mds_write=False,
        pulse_to_write=None,
        plot=False,
        iterations=100,
        burn_frac=0.10,
        **kwargs,
    ):

        self.iterations = iterations
        self.burn_frac = burn_frac

        if mds_write:
            # check_analysis_run(self.pulse, self.run)
            self.node_structure = create_nodes(
                pulse_to_write=pulse_to_write,
                run=run,
                diagnostic_quantities=self.opt_quantity,
                mode="EDIT",
            )

        self.result = self._build_inputs_dict()
        results = []

        if not self.tsample.shape:
            self.tsample = np.array([self.tsample])

        self.plasma.time_to_calculate = self.tsample[0]
        self.start_points = self._sample_start_points(
            sample_method=self.sample_method, **kwargs
        )

        for t in self.tsample:
            self.plasma.time_to_calculate = t
            print(f"Time: {t}")
            self.run_sampler(iterations=iterations, burn_frac=burn_frac)
            _result = self._build_result_dict()
            results.append(_result)

            self.start_points = self.sampler.get_chain()[-1,:,:]
            self.sampler.reset()

            _result = dict(_result, ** self.result)

            self.save_pickle(_result, filepath=filepath, )

            if plot:  # currently requires result with DataArrays
                plot_bayes_result(_result, filepath)

        self.result = dict(self.result, ** results[-1])
        profiles = {}
        globals = {}
        for key, prof in results[0]["PROFILES"].items():
            if key == "RHO_POLOIDAL":
                profiles[key] = results[0]["PROFILES"]["RHO_POLOIDAL"]
            elif key == "RHO_TOR":
                profiles[key] = results[0]["PROFILES"]["RHO_TOR"]
            else:
                _profs = [result["PROFILES"][key] for result in results]
                profiles[key] = xr.concat(_profs, self.tsample)


        for key, prof in results[0]["GLOBAL"].items():
            _glob = [result["GLOBAL"][key] for result in results]
            globals[key] = xr.concat(_glob, self.tsample)

        result = {"PROFILES":profiles, "GLOBAL":globals}

        self.result = dict(self.result, **result,)

        self.result = self.dict_of_dataarray_to_numpy(self.result)
        if mds_write:
            write_nodes(pulse_to_write, self.node_structure, self.result)

        return self.result


if __name__ == "__main__":
    run = BayesWorkflowExample(
        pulse=None,
        diagnostics=[
                    "xrcs",
                    "efit",
                    "smmh1",
                    "cxff_pi",
                    "ts",
                    ],
        param_names=OPTIMISED_PARAMS,
        opt_quantity=OPTIMISED_QUANTITY,
        priors=DEFAULT_PRIORS,
        profile_params=DEFAULT_PROFILE_PARAMS,
        phantoms=True,
        fast_particles=False,
        tstart=0.02,
        tend=0.10,
        dt=0.005,
    )

    run.setup_plasma(
        tsample=0.05,
    )
    run.setup_opt_data(phantoms=run.phantoms)
    run.setup_optimiser(nwalkers=50, sample_method="high_density", model_kwargs=run.model_kwargs, nsamples=100)
    # run.setup_optimiser(nwalkers=50, sample_method="ga", model_kwargs=run.model_kwargs, num_gens=50, popsize=100, sols_to_return=3,  mutation_probability=None)
    results = run(
        filepath=f"./results/test/",
        pulse_to_write=25000000,
        run="RUN01",
        mds_write=True,
        plot=True,
        burn_frac=0.2,
        iterations=500,
    )
