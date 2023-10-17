from typing import Any, List, Tuple, Callable
from typing import Dict
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial

import emcee
import flatdict
import numpy as np
from scipy.stats import loguniform, describe
import xarray as xr
import pandas as pd

from indica.bayesblackbox import BayesBlackBox, ln_prior
from indica.bayesblackbox import get_uniform
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
from indica.equilibrium import fake_equilibrium
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
    "Te_prof.wped": get_uniform(1, 6),
    "Te_prof.wcenter": get_uniform(0.2, 0.4),
    "Te_prof.peaking": get_uniform(1, 4),
    # "Ti_prof.y0/Te_prof.y0": lambda x1, x2: np.where(x1 > x2, 1, 0),  # hot ion mode
    "Ti_prof.y0": get_uniform(1000, 10000),
    "Ti_prof.wped": get_uniform(1, 6),
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
    "cxff_pi.ti",
    "efit.wp",
    # "smmh1.ne",
    "ts.te",
    "ts.ne",
]

DEFAULT_DIAG_NAMES = [
    "xrcs",
    "efit",
    "smmh1",
    "cxff_pi",
    "ts",
]

FAST_DIAG_NAMES = [
    # "xrcs",
    "efit",
    "smmh1",
    "cxff_pi",
    "ts",
]

FAST_OPT_QUANTITY = [
    # "xrcs.spectra",
    "cxff_pi.ti",
    "efit.wp",
    "smmh1.ne",
    "ts.te",
    "ts.ne",
]

FAST_OPT_PARAMS = [
    # "Ne_prof.y1",
    "Ne_prof.y0",
    # "Ne_prof.peaking",
    # "Ne_prof.wcenter",
    # "Ne_prof.wped",
    # "Nimp_prof.y1",
    # "Nimp_prof.y0",
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


def sample_with_moments(sampler, start_points, iterations, n_params, auto_sample=10, stopping_factor=10, debug=True):
    # TODO: Compare old_chain to new_chain: if moments are different then keep going / convergence diagnostics here

    autocorr = np.ones(shape=(iterations, n_params)) * np.nan
    old_mean = np.inf
    success_flag = False  # requires succeeding check twice in a row
    for sample in sampler.sample(
            start_points,
            iterations=iterations,
            progress=True,
            skip_initial_state_check=False,
    ):
        if sampler.iteration % auto_sample:
            continue
        new_tau = sampler.get_autocorr_time(tol=0)
        autocorr[sampler.iteration - 1] = new_tau

        dist_stats = describe(sampler.get_chain(flat=True))

        new_mean = dist_stats.mean

        dmean = np.abs(new_mean - old_mean)
        nsamples = sampler.flatchain.shape[0]

        dmean_normed = dmean / old_mean * nsamples

        if debug:
            print("")
            print(f"dmean_normed: {dmean_normed.max()}")
        if dmean_normed.max() < stopping_factor:
            if success_flag:
                break
            else:
                success_flag = True
        else:
            success_flag = False
        old_mean = new_mean

    autocorr = autocorr[
               : sampler.iteration,
               ]
    return autocorr

def sample_with_autocorr(sampler, start_points, iterations, n_params, auto_sample=5, ):
    autocorr = np.ones(shape=(iterations, n_params)) * np.nan
    old_tau = np.inf
    for sample in sampler.sample(
            start_points,
            iterations=iterations,
            progress=True,
            skip_initial_state_check=False,
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


def gelman_rubin(chain):
    ssq = np.var(chain, axis=1, ddof=1)
    w = np.mean(ssq, axis=0)
    theta_b = np.mean(chain, axis=1)
    theta_bb = np.mean(theta_b, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((theta_bb - theta_b) ** 2, axis=0)
    var_theta = (n - 1) / n * w + 1 / n * B
    R = np.sqrt(var_theta / w)
    return R


def dict_of_dataarray_to_numpy(dict_of_dataarray):
    """
    Mutates input dictionary to change xr.DataArray objects to np.array

    """
    for key, value in dict_of_dataarray.items():
        if isinstance(value, dict):
            dict_of_dataarray_to_numpy(value)
        elif isinstance(value, xr.DataArray):
            dict_of_dataarray[key] = dict_of_dataarray[key].values
    return dict_of_dataarray


@dataclass
class BayesSettings:
    diagnostics: list = field(default_factory=lambda: DEFAULT_DIAG_NAMES)
    param_names: list = field(default_factory=lambda: OPTIMISED_PARAMS)
    opt_quantity: list = field(default_factory=lambda: OPTIMISED_QUANTITY)
    priors: dict = field(default_factory=lambda: DEFAULT_PRIORS)

    """
    TODO: default methods / getter + setters
          print warning if using default values
    """

    def __post_init__(self):
        missing_quantities = [quant for quant in self.opt_quantity if quant.split(".")[0] not in self.diagnostics]
        if missing_quantities:
            raise ValueError(f"{missing_quantities} missing the relevant diagnostic")


@dataclass
class PlasmaSettings:
    main_ion: str = "h"
    impurities: Tuple[str] = ("ar", "c")
    impurity_concentration: Tuple[float] = (0.001, 0.04)
    n_rad: int = 20


@dataclass
class PlasmaContext:
    plasma_settings: PlasmaSettings
    profile_params: dict = field(default_factory=lambda: DEFAULT_PROFILE_PARAMS)

    """
    set profiles / profiler
    """

    def init_plasma(
            self,
            equilibrium: Equilibrium,
            tstart=None,
            tend=None,
            dt=None,
    ):

        self.plasma = Plasma(
            tstart=tstart,
            tend=tend,
            dt=dt,
            main_ion=self.plasma_settings.main_ion,
            impurities=self.plasma_settings.impurities,
            impurity_concentration=self.plasma_settings.impurity_concentration,
            full_run=False,
            n_rad=self.plasma_settings.n_rad,
        )

        self.plasma.set_equilibrium(equilibrium)
        self.update_profiles(self.profile_params)
        self.plasma.build_atomic_data(calc_power_loss=False)

    def update_profiles(self, params: dict):
        if not hasattr(self, "plasma"):
            raise ValueError("plasma not initialised")

        self.plasma.update_profiles(params)

    def time_iterator(self):  # TODO: Why is this being called anytime plasma attributes are accessed?
        print("resetting time iterator")
        return iter(self.plasma.t)

    def return_plasma_attrs(self):
        PLASMA_ATTRIBUTES = [
            "electron_temperature",
            "electron_density",
            "ion_temperature",
            "ion_density",
            "impurity_density",
            "fast_density",
            "neutral_density",
            "zeff",
            "wp",
            "wth",
            "ptot",
            "pth",
        ]
        plasma_attributes = {}
        for plasma_key in PLASMA_ATTRIBUTES:
            if hasattr(self.plasma, plasma_key):
                plasma_attributes[plasma_key] = getattr(self.plasma, plasma_key).sel(
                    t=self.plasma.time_to_calculate
                )
            else:
                raise ValueError(f"plasma does not have attribute {plasma_key}")
        return plasma_attributes

    def save_phantom_profiles(self, kinetic_profiles=None, phantoms=None):
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
        if phantoms:
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

    #
    # def _init_fast_particles(self, run="RUN602", ):
    #
    #     st40_code = ReadST40(self.astra_pulse_range + self.pulse, self.tstart-self.dt, self.tend+self.dt, dt=self.dt, tree="astra")
    #     astra_data = st40_code.get_raw_data("", "astra", run)
    #
    #     if self.astra_equilibrium:
    #         self.equilibrium = Equilibrium(astra_data)
    #         self.plasma.equilibrium = self.equilibrium
    #
    #     st40_code.bin_data_in_time(["astra"], self.tstart, self.tend, self.dt)
    #     code_data = st40_code.binned_data["astra"]
    #     Nf = (
    #         code_data["nf"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
    #         * 1.0e19
    #     )
    #     self.plasma.fast_density.values = Nf.values
    #     Nn = (
    #         code_data["nn"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
    #         * 1.0e19
    #     )
    #     self.plasma.neutral_density.values = Nn.values
    #     Pblon = code_data["pblon"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
    #     self.plasma.pressure_fast_parallel.values = Pblon.values
    #     Pbper = code_data["pbper"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
    #     self.plasma.pressure_fast_perpendicular.values = Pbper.values
    #     self.astra_data = code_data
    #
    #     if self.set_ts_profiles:
    #         overwritten_params = [param for param in self.param_names if any(xs in param for xs in ["Te", "Ne"])]
    #         if any(overwritten_params):
    #             raise ValueError(f"Te/Ne set by TS but then the following params overwritten: {overwritten_params}")
    #         Te = code_data["te"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e3
    #         self.plasma.Te_prof = lambda: Te.values
    #         Ne = code_data["ne"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e19
    #         self.plasma.Ne_prof = lambda:  Ne.values
    #
    #     if self.set_all_profiles:
    #         overwritten_params = [param for param in self.param_names if any(xs in param for xs in ["Te", "Ti", "Ne", "Nimp"])]
    #         if any(overwritten_params):
    #             raise ValueError(f"Te/Ne set by TS but then the following params overwritten: {overwritten_params}")
    #         Te = code_data["te"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e3
    #         self.plasma.Te_prof = lambda: Te.values
    #         Ne = code_data["ne"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e19
    #         self.plasma.Ne_prof = lambda: Ne.values
    #         Ti = code_data["ti"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e3
    #         self.plasma.Ti_prof = lambda: Ti.values
    #         Nimp = code_data["niz1"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.time_to_calculate) * 1e19
    #         self.plasma.Nimp_prof = lambda: Nimp.values


@dataclass
class ModelContext:
    diagnostics: list
    plasma_context: PlasmaContext
    equilibrium: Equilibrium
    transforms: dict
    model_kwargs: Dict[str, Dict[str, Any]]

    """
    Setup models so that they have transforms / plasma / equilibrium etc.. everything needed to produce bckc from models

    TODO: remove repeating code / likely make general methods
    """

    def update_model_kwargs(self, data: dict):
        if self.model_kwargs is None:
            self.model_kwargs = {}
        for diag in diagnostics:
            if diag not in self.model_kwargs.keys():
                self.model_kwargs[diag] = {}

        if "xrcs" in data.keys():
            self.model_kwargs["xrcs"]["window"] = data["xrcs"]["spectra"].wavelength.values

            # TODO: handling model calls dependent on exp data
            background = data["xrcs"]["spectra"].where(
                (data["xrcs"]["spectra"].wavelength < 0.392)
                & (data["xrcs"]["spectra"].wavelength > 0.388),
                drop=True,
            )
            self.model_kwargs["xrcs"]["background"] = background.mean(dim="wavelength")

    def init_models(self, ):
        if not hasattr(self, "plasma_context"):
            raise ValueError("needs plasma_context to setup_models")
        if not hasattr(self.plasma_context, "plasma"):
            raise ValueError("plasma_context needs plasma to setup_models")

        self.models: Dict[str, Any] = {}
        for diag in self.diagnostics:
            if diag == "smmh1":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = Interferometry(name=diag, **self.model_kwargs[diag])
                self.models[diag].set_los_transform(self.transforms[diag])

            elif diag == "efit":
                self.models[diag] = EquilibriumReconstruction(name=diag, **self.model_kwargs[diag])

            elif diag == "cxff_pi":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = ChargeExchange(name=diag, **self.model_kwargs[diag])
                self.models[diag].set_transect_transform(self.transforms[diag])

            elif diag == "cxff_tws_c":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = ChargeExchange(name=diag, **self.model_kwargs[diag])
                self.models[diag].set_transect_transform(self.transforms[diag])

            elif diag == "ts":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = ThomsonScattering(name=diag, **self.model_kwargs[diag])
                self.models[diag].set_transect_transform(self.transforms[diag])

            elif diag == "xrcs":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = HelikeSpectrometer(name="xrcs", **self.model_kwargs[diag])
                self.models[diag].set_los_transform(self.transforms[diag])
            else:
                raise ValueError(f"{diag} not implemented in ModelHandler.setup_models")

        for model_name, model in self.models.items():
            model.plasma = self.plasma_context.plasma

        return self.models

    def _build_bckc(self, params: dict, **kwargs):
        """
        Parameters
        ----------
        params - dictionary which is updated by optimiser
        kwargs - passed to model call i.e. settings

        Returns
        -------
        bckc of results
        """

        # Float128 since rounding of small numbers causes problems
        # when initial results are bad fits
        # model_data = self.bckc[key].astype("float128")

        self.bckc: dict = {}
        for model_name, model in self.models.items():
            # removes "model.name." from params and kwargs then passes them to model
            # e.g. xrcs.background -> background
            _nuisance_params = {
                param_name.replace(model.name + ".", ""): param_value
                for param_name, param_value in params.items()
                if model.name in param_name
            }
            _model_settings = {
                kwarg_name.replace(model.name + ".", ""): kwarg_value
                for kwarg_name, kwarg_value in kwargs.items()
                if model.name in kwarg_name
            }

            _model_kwargs = {
                **_nuisance_params,
                **_model_settings,
            }  # combine dictionaries
            _bckc = model(**_model_kwargs)
            _model_bckc = {
                f"{model.name}.{value_name}": value
                for value_name, value in _bckc.items()
            }  # prepend model name to bckc
            self.bckc = dict(self.bckc, **_model_bckc)
        return self.bckc


@dataclass
class ReaderSettings:
    revisions: dict = field(default_factory=lambda: {})
    filters: dict = field(default_factory=lambda: {})


@dataclass
class DataContext(ABC):
    reader_settings: ReaderSettings
    pulse: int
    tstart: float
    tend: float
    dt: float
    transforms = None
    equilibrium = None
    phantoms = False

    @abstractmethod
    def read_data(self, ):
        self.equilbrium = None
        self.transforms = None
        self.raw_data = None
        self.binned_data = None

    def check_if_data_present(self, data_strategy: Callable):
        if not self.binned_data:
            print("Data not given: using data strategy")
            self.binned_data = data_strategy()

    @abstractmethod
    def process_data(self, model_context: ModelContext):
        self.model_context = model_context  # TODO: handle this dependency (phantom data) some other way
        self.check_if_data_present(lambda: {})
        self.opt_data = flatdict.FlatDict(self.binned_data)


@dataclass
class ExpData(DataContext):
    pulse: int
    diagnostics: list
    phantoms = False

    """
    Considering: either rewriting this class to take over from ReadST40 or vice versa
    
    """

    def read_data(self, ):
        self.reader = ReadST40(self.pulse, tstart=self.tstart, tend=self.tend, dt=self.dt, )
        self.reader(self.diagnostics, revisions=self.reader_settings.revisions, filters=self.reader_settings.filters)
        missing_keys = set(diagnostics) - set(self.reader.binned_data.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"missing data: {missing_keys}")
        self.equilibrium = self.reader.equilibrium
        self.transforms = self.reader.transforms
        # TODO raw data included
        self.raw_data = self.reader.raw_data
        self.binned_data = self.reader.binned_data

    def process_data(self, model_callables: Callable):
        self.model_callables = model_callables
        self.check_if_data_present(self._fail_strategy)
        self.opt_data = self._process_data()

    def _fail_strategy(self):
        raise ValueError("Data strategy: Fail")

    def _process_data(self):
        opt_data = flatdict.FlatDict(self.binned_data, ".")

        if "xrcs.spectra" in opt_data.keys():
            background = opt_data["xrcs.spectra"].where(
                (opt_data["xrcs.spectra"].wavelength < 0.392)
                & (opt_data["xrcs.spectra"].wavelength > 0.388),
                drop=True,
            )
            opt_data["xrcs.spectra"]["error"] = np.sqrt(
                opt_data["xrcs.spectra"] + background.std(dim="wavelength") ** 2
            )
        if "ts.ne" in opt_data.keys():
            opt_data["ts.ne"]["error"] = opt_data["ts.ne"].max(dim="channel") * 0.05

        if "ts.te" in opt_data.keys():
            opt_data["ts.ne"]["error"] = opt_data["ts.ne"].max(dim="channel") * 0.05

        return opt_data


@dataclass
class MockData(DataContext):
    pulse = None
    diagnostic_transforms: dict = field(default_factory=lambda: {})
    model_call_params: dict = field(default_factory=lambda: {})
    phantoms = True

    def read_data(self):
        # Used with phantom data for purposes of tests
        print("Reading mock equilibrium / transforms")
        self.equilibrium = fake_equilibrium(
            tstart,
            tend,
            dt,
        )
        self.transforms = self.diagnostic_transforms
        self.binned_data: dict = {}
        self.raw_data: dict = {}

    def process_data(self, model_callables: Callable):
        self.model_callables = model_callables
        self.check_if_data_present(self._gen_data_strategy)

    def _gen_data_strategy(self):
        print("Data strategy: generating from model bckc")
        self.binned_data = self._process_data()

    def _process_data(self):
        binned_data = self.model_callables()
        self.opt_data = flatdict.FlatDict(binned_data, ".")
        return binned_data


@dataclass
class PhantomData(DataContext):
    pulse: int
    diagnostics: list
    model_call_kwargs: dict = field(default_factory=lambda: {})
    phantoms = True

    def read_data(self, ):
        self.reader = ReadST40(self.pulse, tstart=self.tstart, tend=self.tend, dt=self.dt, )
        self.reader(self.diagnostics, revisions=self.reader_settings.revisions, filters=self.reader_settings.filters)
        missing_keys = set(diagnostics) - set(self.reader.binned_data.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"missing data: {missing_keys}")
        self.equilibrium = self.reader.equilibrium
        self.transforms = self.reader.transforms
        self.raw_data = {}
        self.binned_data = {}

    def process_data(self, model_callables: Callable):
        self.model_callables = model_callables
        self.check_if_data_present(self._gen_data_strategy)

    def _gen_data_strategy(self):
        print("Data strategy: generating from model bckc")
        self.binned_data = self._process_data()

    def _process_data(self):
        binned_data = self.model_callables()
        self.opt_data = flatdict.FlatDict(binned_data, ".")
        return binned_data


@dataclass
class OptimiserEmceeSettings:
    param_names: list
    priors: dict
    iterations: int = 1000
    nwalkers: int = 50
    burn_frac: float = 0.20
    sample_method: str = "random"
    starting_samples: int = 100
    stopping_criterion: str = "mode"
    stopping_criterion_factor: float = 10


@dataclass
class OptimiserContext(ABC):
    optimiser_settings: OptimiserEmceeSettings

    @abstractmethod
    def init_optimiser(self, *args, **kwargs):
        self.optimiser = None

    @abstractmethod
    def sample_start_points(self, *args, **kwargs):
        self.start_points = None

    @abstractmethod
    def format_results(self):
        self.results = {}

    @abstractmethod
    def run(self):
        results = None
        return results


@dataclass
class EmceeOptimiser(OptimiserContext):

    def init_optimiser(self, blackbox_func: Callable, model_kwargs: dict = field(default={})
                       ):
        ndim = len(self.optimiser_settings.param_names)
        self.move = [(emcee.moves.StretchMove(), 0.9), (emcee.moves.DEMove(), 0.1)]
        self.optimiser = emcee.EnsembleSampler(
            self.optimiser_settings.nwalkers,
            ndim,
            log_prob_fn=blackbox_func,
            parameter_names=self.optimiser_settings.param_names,
            moves=self.move,
            kwargs=model_kwargs,
        )

    def sample_start_points(self, ):

        if self.optimiser_settings.sample_method == "high_density":

            self.start_points = self.sample_from_high_density_region(
                param_names=self.optimiser_settings.param_names,
                priors=self.optimiser_settings.priors,
                optimiser=self.optimiser,
                nwalkers=self.optimiser_settings.nwalkers,
                nsamples=self.optimiser_settings.starting_samples
            )

        elif self.optimiser_settings.sample_method == "random":
            self.start_points = self.sample_from_priors(
                param_names=self.optimiser_settings.param_names,
                priors=self.optimiser_settings.priors,
                size=self.optimiser_settings.nwalkers
            )
        else:
            raise ValueError(f"Sample method: {self.optimiser_settings.sample_method} "
                             f"not recognised, Defaulting to random sampling")

    def sample_from_priors(self, param_names: list, priors: dict, size=10):
        #  Use priors to generate samples
        for name in param_names:
            if name in priors.keys():
                if hasattr(priors[name], "rvs"):
                    continue
                else:
                    raise TypeError(f"prior object {name} missing rvs method")
            else:
                raise ValueError(f"Missing prior for {name}")

        #  Throw out samples that don't meet conditional priors and redraw
        samples = np.empty((param_names.__len__(), 0))
        while samples.size < param_names.__len__() * size:
            # Some mangling of dictionaries so _ln_prior works
            # Increase size * n if too slow / looping too much
            new_sample = {
                name: priors[name].rvs(size=size * 2) for name in param_names
            }
            _ln_prior = ln_prior(priors, new_sample)
            # Convert from dictionary of arrays -> array,
            # then filtering out where ln_prior is -infinity
            accepted_samples = np.array(list(new_sample.values()))[
                               :, _ln_prior != -np.inf
                               ]
            samples = np.append(samples, accepted_samples, axis=1)
        samples = samples[:, 0:size]
        return samples.transpose()

    def sample_from_high_density_region(
            self, param_names: list, priors: dict, optimiser, nwalkers: int, nsamples=100
    ):

        # TODO: remove repeated code
        start_points = self.sample_from_priors(param_names, priors, size=nsamples)

        ln_prob, _ = optimiser.compute_log_prob(start_points)
        num_best_points = 3
        index_best_start = np.argsort(ln_prob)[-num_best_points:]
        best_start_points = start_points[index_best_start, :]
        best_points_std = np.std(best_start_points, axis=0)

        # Passing samples through ln_prior and redrawing if they fail
        samples = np.empty((param_names.__len__(), 0))
        while samples.size < param_names.__len__() * nwalkers:
            sample = np.random.normal(
                np.mean(best_start_points, axis=0),
                best_points_std,
                size=(nwalkers * 5, len(param_names)),
            )
            start = {name: sample[:, idx] for idx, name in enumerate(param_names)}
            _ln_prior = ln_prior(priors, start, )
            # Convert from dictionary of arrays -> array,
            # then filtering out where ln_prior is -infinity
            accepted_samples = np.array(list(start.values()))[:, _ln_prior != -np.inf]
            samples = np.append(samples, accepted_samples, axis=1)
        start_points = samples[:, 0:nwalkers].transpose()
        return start_points

    def run(self, ):

        if self.optimiser_settings.stopping_criterion == "mode":
            self.autocorr = sample_with_moments(
                self.optimiser,
                self.start_points,
                self.optimiser_settings.iterations,
                self.optimiser_settings.param_names.__len__(),
                auto_sample=10,
                stopping_factor=self.optimiser_settings.stopping_criterion_factor
            )
        else:
            raise ValueError(f"Stopping criterion: {self.optimiser_settings.stopping_criterion} not recognised")

        optimiser_results = self.format_results()
        return optimiser_results

    def format_results(self):
        results = {}
        _blobs = self.optimiser.get_blobs(discard=int(self.optimiser.iteration * self.optimiser_settings.burn_frac),
                                          flat=True)
        blobs = [blob for blob in _blobs if blob]  # remove empty blobs

        blob_names = blobs[0].keys()
        samples = np.arange(0, blobs.__len__())

        results["blobs"] = {
            blob_name: xr.concat(
                [data[blob_name] for data in blobs],
                dim=pd.Index(samples, name="index"),
            )
            for blob_name in blob_names
        }
        results["accept_frac"] = self.optimiser.acceptance_fraction.sum()
        results["prior_sample"] = self.sample_from_priors(
            self.optimiser_settings.param_names, self.optimiser_settings.priors, size=int(1e4)
        )

        post_sample = self.optimiser.get_chain(
            discard=int(self.optimiser.iteration * self.optimiser_settings.burn_frac), flat=True)
        # pad index dim with maximum number of iterations
        max_iter = self.optimiser_settings.iterations * self.optimiser_settings.nwalkers
        npad = ((0, int(max_iter*(1-self.optimiser_settings.burn_frac) - post_sample.shape[0])), (0, 0))
        results["post_sample"] = np.pad(post_sample, npad, constant_values=np.nan)

        results["auto_corr"] = self.autocorr
        return results


class BayesWorkflow(AbstractBayesWorkflow):
    def __init__(
            self,
            bayes_settings: BayesSettings,
            data_context: DataContext,
            plasma_context: PlasmaContext,
            model_context: ModelContext,
            optimiser_context: EmceeOptimiser,

            tstart: float = 0.02,
            tend: float = 0.10,
            dt: float = 0.01,

    ):
        self.bayes_settings = bayes_settings
        self.data_context = data_context
        self.plasma_context = plasma_context
        self.model_context = model_context
        self.optimiser_context = optimiser_context

        self.tstart = tstart
        self.tend = tend
        self.dt = dt

        self.plasma_context.init_plasma(equilibrium=self.data_context.equilibrium, tstart=self.tstart, tend=self.tend,
                                        dt=self.dt, )
        self.plasma_context.save_phantom_profiles(phantoms=self.data_context.phantoms)

        self.model_context.update_model_kwargs(self.data_context.binned_data)
        self.model_context.init_models()

        self.model_call_kwargs = {"xrcs.pixel_offset": 4.0}
        self.data_context.process_data(partial(self.model_context._build_bckc, self.model_call_kwargs))

        self.blackbox = BayesBlackBox(data=self.data_context.opt_data,
                                      plasma_context=self.plasma_context,
                                      model_context=self.model_context,
                                      quant_to_optimise=self.bayes_settings.opt_quantity,
                                      priors=self.bayes_settings.priors)

        self.optimiser_context.init_optimiser(self.blackbox.ln_posterior, model_kwargs=self.model_call_kwargs)

    def __call__(
            self,
            filepath="./results/test/",
            run="RUN01",
            mds_write=False,
            pulse_to_write=None,
            plot=False,
            **kwargs,
    ):

        self.result = self._build_inputs_dict()
        results = []

        for time in self.plasma_context.time_iterator():
            self.plasma_context.plasma.time_to_calculate = time
            print(f"Time: {time.values:.2f}")
            self.optimiser_context.sample_start_points()
            self.optimiser_context.run()
            results.append(self.optimiser_context.format_results())
            self.optimiser_context.optimiser.reset()

        # unpack results and add time axis
        blobs = {}
        for key in results[0]["blobs"].keys():
            _blob = [result["blobs"][key] for result in results]
            blobs[key] = xr.concat(_blob, self.plasma_context.plasma.t)
        self.blobs = blobs

        opt_samples = {}
        for key in results[0].keys():
            if key == "blobs":
                continue
            _opt_samples = [result[key] for result in results]
            opt_samples[key] = np.array(_opt_samples)
        self.opt_samples = opt_samples

        result = self._build_result_dict()
        self.result = dict(self.result, **result)

        self.save_pickle(self.result, filepath=filepath, )

        if plot:  # currently requires result with DataArrays
            plot_bayes_result(self.result, filepath)

        self.result = dict_of_dataarray_to_numpy(self.result)

        if mds_write:
            # check_analysis_run(self.pulse, self.run)
            self.node_structure = create_nodes(
                pulse_to_write=pulse_to_write,
                run=run,
                diagnostic_quantities=self.bayes_settings.opt_quantity,
                mode="NEW",
            )
            write_nodes(pulse_to_write, self.node_structure, self.result)

        return


if __name__ == "__main__":
    pulse = 11336
    tstart = 0.07
    tend = 0.08
    dt = 0.01

    diagnostics = [
        # "xrcs",
        # "efit",
        # "smmh1",
        # "cxff_pi",
        "ts",
    ]
    # diagnostic_quantities
    opt_quant = [
                # "xrcs.spectra",
                #  "efit.wp",
                "ts.te"
                ]
    opt_params = [
                  "Te_prof.y0",
                  "Te_prof.peaking",
                  "Te_prof.wped",
                    "Te_prof.wcenter",

                  # "Ti_prof.y0",
                  # "Ne_prof.y0",
                  ]

    # BlackBoxSettings
    bayes_settings = BayesSettings(diagnostics=diagnostics, param_names=opt_params,
                                   opt_quantity=opt_quant, priors=DEFAULT_PRIORS, )

    data_settings = ReaderSettings(filters={},
                                   revisions={})  # Add general methods for filtering data co-ords to ReadST40

    # mock_transforms = {"xrcs": helike_transform_example(1),
    #  "smmh1": smmh1_transform_example(1),
    #  "cxff_pi": pi_transform_example(5),
    #  "ts": ts_transform_example(11), }
    # data_context = MockData(pulse=None, diagnostic_transforms=mock_transforms, )

    data_context = ExpData(pulse=pulse, diagnostics=diagnostics,
                           tstart=tstart, tend=tend, dt=dt, reader_settings=data_settings, )
    data_context.read_data()

    plasma_settings = PlasmaSettings(main_ion="h", impurities=("ar", "c"), impurity_concentration=(0.001, 0.04),
                                     n_rad=20)
    plasma_context = PlasmaContext(plasma_settings=plasma_settings, profile_params=DEFAULT_PROFILE_PARAMS)

    model_init_kwargs = {
        "cxff_pi": {"element": "ar"},
        "cxff_tws_c": {"element": "c"},
        "xrcs": {
            "window_masks": [slice(0.394, 0.396)],
        },
    }

    model_context = ModelContext(diagnostics=diagnostics,
                                 plasma_context=plasma_context,
                                 equilibrium=data_context.equilibrium,
                                 transforms=data_context.transforms,
                                 model_kwargs=model_init_kwargs,
                                 )

    optimiser_settings = OptimiserEmceeSettings(param_names=bayes_settings.param_names, nwalkers=20, iterations=500,
                                                sample_method="high_density", starting_samples=100, burn_frac=0.20,
                                                stopping_criterion="mode", stopping_criterion_factor=10,
                                                priors=bayes_settings.priors)
    optimiser_context = EmceeOptimiser(optimiser_settings=optimiser_settings)

    workflow = BayesWorkflow(tstart=tstart, tend=tend, dt=dt,
                            bayes_settings=bayes_settings, data_context=data_context,
                            optimiser_context=optimiser_context,
                            plasma_context=plasma_context, model_context=model_context)

    workflow(pulse_to_write=25000000, run="RUN01", mds_write=True, plot=True, filepath="./results/test_moments/")
