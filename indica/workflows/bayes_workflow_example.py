from typing import Any
from typing import Dict

import emcee
import flatdict
import numpy as np
from scipy.stats import loguniform

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


# global configurations
DEFAULT_PROFILE_PARAMS = {
    "Ne_prof.y0": 5e19,
    "Ne_prof.y1": 2e18,
    "Ne_prof.yend": 1e18,
    "Ne_prof.wped": 3,
    "Ne_prof.wcenter": 0.3,
    "Ne_prof.peaking": 1.2,

    "Nimp_prof.y0": 1e17,
    "Nimp_prof.y1": 5e15,
    "Nimp_prof.yend": 1e15,
    "Nimp_prof.wcenter": 0.3,
    "Nimp_prof.wped": 6,
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
    "Ne_prof.y1": get_uniform(1e18, 1e19),
    "Ne_prof.y0/Ne_prof.y1": lambda x1, x2: np.where((x1 > x2 * 2), 1, 0),
    "Ne_prof.wped": get_uniform(2, 6),
    "Ne_prof.wcenter": get_uniform(0.2, 0.4),
    "Ne_prof.peaking": get_uniform(1, 4),

    "Nimp_prof.y0": loguniform(1e16, 1e18),
    "Nimp_prof.y1": get_uniform(1e15, 1e16),
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
    "xrcs.pixel_offset": get_uniform(-4.01, -3.99),
}

OPTIMISED_PARAMS = [
    "Ne_prof.y1",
    "Ne_prof.y0",
    "Ne_prof.peaking",
    "Ne_prof.wcenter",
    "Ne_prof.wped",
    # "Nimp_prof.y1",
    "Nimp_prof.y0",
    # "Nimp_prof.wcenter",
    # "Nimp_prof.wped",
    "Nimp_prof.peaking",
    "Te_prof.y0",
    "Te_prof.wped",
    "Te_prof.wcenter",
    "Te_prof.peaking",
    "Ti_prof.y0",
    "Ti_prof.wped",
    "Ti_prof.wcenter",
    "Ti_prof.peaking",
]
OPTIMISED_QUANTITY = [
    "xrcs.spectra",
    # "cxff_pi.ti",
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
        phantoms: bool = False,
        fast_particles = False,
        tstart=0.02,
        tend=0.10,
        dt=0.005,
    ):
        self.pulse = pulse
        self.diagnostics = diagnostics
        self.param_names = param_names
        self.opt_quantity = opt_quantity
        self.priors = priors
        self.profile_params = profile_params
        self.phantoms = phantoms
        self.fast_particles = fast_particles
        self.tstart = tstart
        self.tend = tend
        self.dt = dt
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
        self.setup_models(self.diagnostics)

    def setup_plasma(
        self,
        tstart=None,
        tend=None,
        dt=None,
        tsample=0.050,
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
        self.tsample = tsample
        self.plasma.time_to_calculate = self.plasma.t[
            np.abs(tsample - self.plasma.t).argmin()
        ]
        self.plasma.set_equilibrium(self.equilibrium)
        self.plasma.update_profiles(self.profile_params)
        if self.fast_particles:
            self._init_fast_particles()

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
                los_transform.set_equilibrium(self.equilibrium)
                model = Interferometry(name=diag)
                model.set_los_transform(los_transform)

            elif diag == "xrcs":
                los_transform = self.transforms[diag]
                los_transform.set_equilibrium(self.equilibrium)
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
                transform.set_equilibrium(self.equilibrium)
                model = ChargeExchange(name=diag, element="ar")
                model.set_transect_transform(transform)

            elif diag == "ts":
                transform = self.transforms[diag]
                transform.set_equilibrium(self.equilibrium)
                model = ThomsonScattering(name=diag, )
                model.set_transect_transform(transform)

            else:
                raise ValueError(f"{diag} not found in setup_models")
            self.models[diag] = model

    def _init_fast_particles(self):
        st40_code = ReadST40(13000000 + self.pulse, self.tstart, self.tend, dt=self.dt, tree="astra")
        st40_code.get_raw_data("", "astra", "RUN602")
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
            # opt_data["cxff_pi"]["ti"] = opt_data["cxff_pi"]["ti"].where(
            #     opt_data["cxff_pi"]["ti"].channel == 0,
            # )
        if "ts" in self.diagnostics:
            # TODO: fix error, for now flat error
            opt_data["ts.te"] = opt_data["ts.te"].where(opt_data["ts.te"].channel >19, drop=True)
            opt_data["ts.ne"] = opt_data["ts.ne"].where(opt_data["ts.ne"].channel >19, drop=True)

            opt_data["ts.te"]["error"] = opt_data["ts.te"] * 0.10 + 10
            opt_data["ts.ne"]["error"] = opt_data["ts.ne"] * 0.10 + 10

        return opt_data

    def setup_optimiser(
        self,
        model_kwargs,
        nwalkers=50,
        burn_frac=0.10,
        sample_high_density=False,
        **kwargs,
    ):
        self.model_kwargs = model_kwargs
        self.nwalkers = nwalkers
        self.burn_frac = burn_frac
        self.sample_high_density = sample_high_density

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
        self.start_points = self._sample_start_points(
            sample_high_density=self.sample_high_density, **kwargs
        )

    def _sample_start_points(self, sample_high_density: bool = True, nsamples=100, **kwargs):
        if sample_high_density:
            start_points = self.bayesmodel.sample_from_high_density_region(
                self.param_names, self.sampler, self.nwalkers, nsamples=nsamples
            )
        else:
            start_points = self.bayesmodel.sample_from_priors(
                self.param_names, size=self.nwalkers
            )
        return start_points

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
        if mds_write:
            # check_analysis_run(self.pulse, self.run)
            self.node_structure = create_nodes(
                pulse_to_write=pulse_to_write,
                run=run,
                diagnostic_quantities=self.opt_quantity,
                mode="EDIT",
            )

        self.run_sampler(iterations=iterations, burn_frac=burn_frac)
        self.save_pickle(filepath=filepath)

        if plot:  # currently requires result with DataArrays
            plot_bayes_result(self.result, filepath)

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
        fast_particles=True,
        tstart=0.02,
        tend=0.10,
        dt=0.005,
    )

    run.setup_plasma(
        tsample=0.05,
    )
    run.setup_opt_data(phantoms=run.phantoms)
    run.setup_optimiser(nwalkers=50, sample_high_density=True, model_kwargs=run.model_kwargs)
    results = run(
        filepath=f"./results/test/",
        pulse_to_write=25000000,
        run="RUN01",
        mds_write=True,
        plot=True,
        burn_frac=0.10,
        iterations=100,
    )

