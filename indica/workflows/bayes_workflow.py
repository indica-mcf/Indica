from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import emcee
import flatdict
import numpy as np
import pandas as pd
from scipy.stats import describe
from scipy.stats import loguniform
from sklearn.gaussian_process import kernels
import xarray as xr

from indica.bayesblackbox import BayesBlackBox
from indica.bayesblackbox import get_uniform
from indica.bayesblackbox import ln_prior
from indica.equilibrium import Equilibrium
from indica.equilibrium import fake_equilibrium
from indica.models.charge_exchange import ChargeExchange
from indica.models.charge_exchange import pi_transform_example
from indica.models.equilibrium_reconstruction import EquilibriumReconstruction
from indica.models.helike_spectroscopy import helike_transform_example
from indica.models.helike_spectroscopy import HelikeSpectrometer
from indica.models.interferometry import Interferometry
from indica.models.interferometry import smmh1_transform_example
from indica.models.plasma import Plasma
from indica.models.thomson_scattering import ThomsonScattering
from indica.models.thomson_scattering import ts_transform_example
from indica.operators.gpr_fit import gpr_fit_ts
from indica.operators.gpr_fit import post_process_ts
from indica.readers.read_st40 import ReadST40
from indica.workflows.abstract_bayes_workflow import AbstractBayesWorkflow
from indica.workflows.bayes_plots import plot_bayes_result
from indica.writers.bda_tree import create_nodes
from indica.writers.bda_tree import does_tree_exist
import standard_utility as util


# global configurations
DEFAULT_PROFILE_PARAMS = {
    "Ne_prof.y0": 5e19,
    "Ne_prof.y1": 2e18,
    "Ne_prof.yend": 1e18,
    "Ne_prof.wped": 3,
    "Ne_prof.wcenter": 0.3,
    "Ne_prof.peaking": 1.2,
    # "Niz1_prof.y0": 1e17,
    # "Niz1_prof.y1": 1e15,
    # "Niz1_prof.yend": 1e15,
    # "Niz1_prof.wcenter": 0.3,
    # "Niz1_prof.wped": 3,
    # "Niz1_prof.peaking": 2,
    "Nh_prof.y0": 1e14,
    "Nh_prof.y1": 5e15,
    "Nh_prof.yend": 5e15,
    "Nh_prof.wcenter": 0.01,
    "Nh_prof.wped": 18,
    "Nh_prof.peaking": 1,
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
    "Ne_prof.wped": loguniform(2, 20),
    "Ne_prof.wcenter": get_uniform(0.2, 0.4),
    "Ne_prof.peaking": get_uniform(1, 4),
    "Niz1_prof.y0": loguniform(2e15, 5e17),
    "Niz1_prof.y1": loguniform(1e14, 1e16),
    "Ne_prof.y0/Niz1_prof.y0": lambda x1, x2: np.where(
        (x1 > x2 * 100) & (x1 < x2 * 1e5), 1, 0
    ),
    "Niz1_prof.y0/Niz1_prof.y1": lambda x1, x2: np.where((x1 > x2), 1, 0),
    "Niz1_prof.wped": get_uniform(2, 6),
    "Niz1_prof.wcenter": get_uniform(0.2, 0.4),
    "Niz1_prof.peaking": get_uniform(1, 6),
    "Niz1_prof.peaking/Ne_prof.peaking": lambda x1, x2: np.where(
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
}

OPTIMISED_PARAMS = [
    "Ne_prof.y1",
    "Ne_prof.y0",
    "Ne_prof.peaking",
    # "Ne_prof.wcenter",
    "Ne_prof.wped",
    # "Niz1_prof.y1",
    "Niz1_prof.y0",
    # "Niz1_prof.wcenter",
    # "Niz1_prof.wped",
    "Niz1_prof.peaking",
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
    # "Niz1_prof.y1",
    # "Niz1_prof.y0",
    # "Niz1_prof.wcenter",
    # "Niz1_prof.wped",
    # "Niz1_prof.peaking",
    "Te_prof.y0",
    # "Te_prof.wped",
    # "Te_prof.wcenter",
    # "Te_prof.peaking",
    "Ti_prof.y0",
    # "Ti_prof.wped",
    # "Ti_prof.wcenter",
    # "Ti_prof.peaking",
]


def sample_with_moments(
    sampler,
    start_points,
    iterations,
    n_params,
    auto_sample=10,
    stopping_factor=0.01,
    debug=False,
):
    # TODO: Compare old_chain to new_chain
    #  if moments are different then keep going / convergence diagnostics here

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
        rel_dmean = dmean / old_mean

        if debug:
            print("")
            print(f"rel_dmean: {rel_dmean.max()}")
        if rel_dmean.max() < stopping_factor:
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


def sample_with_autocorr(
    sampler,
    start_points,
    iterations,
    n_params,
    auto_sample=5,
):
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


def sample_from_priors(param_names: list, priors: dict, size=10):
    """
    TODO: may be able to remove param_names from here and at some point earlier
        remove priors that aren't used
        then loop over remaining priors while handling conditional priors somehow...
        The order of samples may need to be checked / reordered at some point then
    """

    #  Throw out samples that don't meet conditional priors and redraw
    samples = np.empty((param_names.__len__(), 0))
    while samples.size < param_names.__len__() * size:
        # Some mangling of dictionaries so _ln_prior works
        # Increase size * n if too slow / looping too much
        new_sample = {name: priors[name].rvs(size=size * 2) for name in param_names}
        _ln_prior = ln_prior(priors, new_sample)
        # Convert from dictionary of arrays -> array,
        # then filtering out where ln_prior is -infinity
        accepted_samples = np.array(list(new_sample.values()))[:, _ln_prior != -np.inf]
        samples = np.append(samples, accepted_samples, axis=1)
    samples = samples[:, 0:size]
    return samples.transpose()


@dataclass
class PlasmaSettings:
    main_ion: str = "h"
    impurities: Tuple[str, ...] = ("ar", "c")
    impurity_concentration: Tuple[float, ...] = (0.001, 0.04)
    n_rad: int = 20


@dataclass
class PlasmaContext:
    plasma_settings: PlasmaSettings
    profile_params: dict = field(default_factory=lambda: DEFAULT_PROFILE_PARAMS)

    plasma_attribute_names: list = field(default_factory=lambda:[
        "electron_temperature",
        "electron_density",
        "ion_temperature",
        "ion_density",
        "impurity_density",
        "fast_density",
        "pressure_fast",
        "neutral_density",
        "zeff",
        "meanz",
        "wp",
        "wth",
        "pressure_tot",
        "pressure_th",])
    plasma_profile_names: list = field(default_factory=lambda: [
        "electron_temperature",
        "electron_density",
        "ion_temperature",
        "ion_density",
        "impurity_density",
        "fast_density",
        "pressure_fast",
        "neutral_density",
        "zeff",
        "meanz",
        "pressure_tot",
        "pressure_th", ])

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

    def time_iterator(self):
        print("resetting time iterator")
        return iter(self.plasma.t)

    def return_plasma_attrs(self):
        plasma_attributes = {}
        for plasma_key in self.plasma_attribute_names:

            if hasattr(self.plasma, plasma_key):
                plasma_attributes[plasma_key] = getattr(self.plasma, plasma_key).sel(
                    t=self.plasma.time_to_calculate
                )
            else:
                raise ValueError(f"plasma does not have attribute {plasma_key}")
        return plasma_attributes

    def save_phantom_profiles(self, kinetic_profiles=None, phantoms=None):
        if kinetic_profiles is None:
            kinetic_profiles = self.plasma_attribute_names
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

    def fit_ts_profile(self, pulse, tstart, tend, dt, split="LFS", quant="ne", R_shift=0.0):

        # Temp hack for setting TS data

        reader = ReadST40(pulse=pulse, tstart=tstart, tend= tend, dt=dt)
        reader(["ts", "efit"], R_shift=R_shift)

        kernel = 1.0 * kernels.RationalQuadratic(
            alpha_bounds=(0.5, 1.0), length_scale_bounds=(0.3, 0.7)
        ) + kernels.WhiteKernel(noise_level_bounds=(0.01, 10))

        processed_data = post_process_ts(
            deepcopy(reader.binned_data),
            reader.equilibrium,
            quant,
            pulse,

            split=split,
        )
        fit, _ = gpr_fit_ts(
            data=processed_data,
            xdim="rho",
            virtual_obs=True,
            virtual_symmetry=True,
            kernel=kernel,
            save_fig=True,
        )
        fit = xr.where(fit < 0, 1e-10, fit)
        return fit, processed_data

    def set_ts_profiles(self, data_context, split="LFS", R_shift=0.0):

        ne_fit, ne_data = self.fit_ts_profile(data_context.pulse, data_context.tstart, data_context.tend, data_context.dt, quant="ne", split=split, R_shift=R_shift)
        te_fit, te_data = self.fit_ts_profile(data_context.pulse, data_context.tstart, data_context.tend, data_context.dt, quant="te", split=split, R_shift=R_shift)
        ne_fit *= 1e19
        te_fit *= 1e3


        self.plasma.electron_density.loc[dict()] = ne_fit.interp(rho=self.plasma.rho)
        self.plasma.electron_temperature.loc[dict()] = te_fit.interp(
            rho=self.plasma.rho
        )

    def map_profiles_to_midplane(self, blobs):
        nchan = len(self.plasma.R_midplane)
        chan = np.arange(nchan)
        R = xr.DataArray(self.plasma.R_midplane.values, coords=[("channel", chan)])
        z = xr.DataArray(self.plasma.z_midplane.values, coords=[("channel", chan)])

        rho = self.plasma.equilibrium.rho.interp(t=self.plasma.t, R=R, z=z).drop_vars(
            ["R", "z"]
        )
        midplane_profiles = {}
        for profile in self.plasma_profile_names:
            midplane_profiles[profile] = (
                blobs[profile].interp(rho_poloidal=rho).drop_vars("rho_poloidal")
            )
            midplane_profiles[profile]["R"] = R
            midplane_profiles[profile]["z"] = z
            midplane_profiles[profile] = midplane_profiles[profile].swap_dims(
                {"channel": "R"}
            )

        return midplane_profiles


@dataclass
class ModelSettings:
    init_kwargs: dict = field(
        default_factory=lambda: {
            "cxff_pi": {"element": "ar"},
            "cxff_tws_c": {"element": "c"},
            "xrcs": {
                "window_masks": [slice(0.394, 0.396)],
            },
        }
    )
    call_kwargs: dict = field(default_factory=lambda: {"xrcs": {"pixel_offset": 0.0}})



@dataclass
class ModelContext:
    model_settings: ModelSettings
    diagnostics: list
    plasma_context: PlasmaContext
    equilibrium: Union[Equilibrium, None]
    transforms: Union[dict, None]

    """
    Setup models so that they have transforms / plasma / equilibrium etc..
    everything needed to produce bckc from models

    TODO: remove repeating code / likely make general methods
    """

    def __post_init__(self):
        # Create empty dictionaries for diagnostics where no init or call kwargs defined
        for diag in self.diagnostics:
            if diag not in self.model_settings.init_kwargs.keys():
                self.model_settings.init_kwargs[diag] = {}
            if diag not in self.model_settings.call_kwargs.keys():
                self.model_settings.call_kwargs[diag] = {}

    def update_model_kwargs(self, data: dict):
        if "xrcs" in data.keys():
            self.model_settings.init_kwargs["xrcs"]["window"] = data["xrcs"][
                "spectra"
            ].wavelength.values

            # TODO: handling model calls dependent on exp data
            background = data["xrcs"]["spectra"].where(
                (data["xrcs"]["spectra"].wavelength < 0.392)
                & (data["xrcs"]["spectra"].wavelength > 0.388),
                drop=True,
            )
            self.model_settings.init_kwargs["xrcs"]["background"] = background.mean(
                dim="wavelength"
            )

    def init_models(
        self,
    ):
        if not hasattr(self, "plasma_context"):
            raise ValueError("needs plasma_context to setup_models")
        if not hasattr(self.plasma_context, "plasma"):
            raise ValueError("plasma_context needs plasma to setup_models")

        self.models: Dict[str, Any] = {}
        for diag in self.diagnostics:
            if diag == "smmh1":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = Interferometry(
                    name=diag, **self.model_settings.init_kwargs[diag]
                )
                self.models[diag].set_los_transform(self.transforms[diag])

            elif diag == "efit":
                self.models[diag] = EquilibriumReconstruction(
                    name=diag, **self.model_settings.init_kwargs[diag]
                )

            elif diag == "cxff_pi":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = ChargeExchange(
                    name=diag, **self.model_settings.init_kwargs[diag]
                )
                self.models[diag].set_transect_transform(self.transforms[diag])

            elif diag == "cxff_tws_c":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = ChargeExchange(
                    name=diag, **self.model_settings.init_kwargs[diag]
                )
                self.models[diag].set_transect_transform(self.transforms[diag])

            elif diag == "ts":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = ThomsonScattering(
                    name=diag, **self.model_settings.init_kwargs[diag]
                )
                self.models[diag].set_transect_transform(self.transforms[diag])

            elif diag == "xrcs":
                self.transforms[diag].set_equilibrium(self.equilibrium, force=True)
                self.models[diag] = HelikeSpectrometer(
                    name="xrcs", **self.model_settings.init_kwargs[diag]
                )
                self.models[diag].set_los_transform(self.transforms[diag])
            else:
                raise ValueError(f"{diag} not implemented in ModelHandler.setup_models")

        for model_name, model in self.models.items():
            model.plasma = self.plasma_context.plasma

        return self.models

    def _build_bckc(
        self,
        params: dict = None,
    ):
        """
        Parameters
        ----------
        params - dictionary which is updated by optimiser

        Returns
        -------
        nested bckc of results
        """

        if params is None:
            params = {}
        self.bckc: dict = {}
        for model_name, model in self.models.items():
            # removes "model.name." from params and kwargs then passes them to model
            # e.g. xrcs.background -> background
            _call_params = {
                param_name.replace(model.name + ".", ""): param_value
                for param_name, param_value in params.items()
                if model.name in param_name
            }
            # call_kwargs defined in model_settings
            _call_kwargs = {
                kwarg_name: kwarg_value
                for kwarg_name, kwarg_value in self.model_settings.call_kwargs[
                    model_name
                ].items()
            }
            _model_kwargs = {
                **_call_kwargs,
                **_call_params,
            }  # combine dictionaries

            _bckc = model(**_model_kwargs)
            _model_bckc = {
                model.name: {value_name: value for value_name, value in _bckc.items()}
            }
            # prepend model name to bckc
            self.bckc = dict(self.bckc, **_model_bckc)
        return self.bckc


@dataclass
class ReaderSettings:
    revisions: dict = field(default_factory=lambda: {})
    filters: dict = field(default_factory=lambda: {})


@dataclass  # type: ignore[misc]
class DataContext(ABC):
    reader_settings: ReaderSettings
    pulse: Optional[int]
    tstart: float
    tend: float
    dt: float
    diagnostics: list
    phantoms = False

    @abstractmethod
    def read_data(
        self,
    ):
        self.equilbrium = None
        self.transforms = None
        self.raw_data = None
        self.binned_data = None

    @abstractmethod
    def data_strategy(self):
        return None

    def _check_if_data_present(self, data_strategy: Callable = lambda: None):
        if not self.binned_data:
            print("Data not given: using data strategy")
            self.binned_data = data_strategy()

    def pre_process_data(self, model_callable: Callable):
        self.model_callable = model_callable
        # TODO: handle this dependency (phantom data) some other way?
        self._check_if_data_present(self.data_strategy)

    @abstractmethod
    def process_data(self, model_callable: Callable):
        self.pre_process_data(model_callable)
        self.opt_data = flatdict.FlatDict(self.binned_data)


@dataclass
class ExpData(DataContext):
    phantoms = False

    """
    Considering: either rewriting this class to take over from ReadST40 or vice versa

    """

    def read_data(
        self,
    ):
        self.reader = ReadST40(
            self.pulse,
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
        )
        self.reader(
            self.diagnostics,
            revisions=self.reader_settings.revisions,
            filters=self.reader_settings.filters,
        )
        missing_keys = set(self.diagnostics) - set(self.reader.binned_data.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"missing data: {missing_keys}")
        self.equilibrium = self.reader.equilibrium
        self.transforms = self.reader.transforms
        # TODO raw data included
        self.raw_data = self.reader.raw_data
        self.binned_data = self.reader.binned_data

    def data_strategy(self):
        raise ValueError("Data strategy: Fail")

    def process_data(self, model_callable: Callable):
        self.pre_process_data(model_callable)
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
        # TODO move the channel filtering to the read_data method in filtering = {}
        if "ts.ne" in opt_data.keys():
            opt_data["ts.ne"]["error"] = opt_data["ts.ne"].max(dim="channel") * 0.05
            # opt_data["ts.ne"] = opt_data["ts.ne"].where(
            #     opt_data["ts.ne"].channel < 21)

        if "ts.te" in opt_data.keys():
            opt_data["ts.ne"]["error"] = opt_data["ts.ne"].max(dim="channel") * 0.05
            # opt_data["ts.te"] = opt_data["ts.te"].where(
            # opt_data["ts.te"].channel < 21)

        if "cxff_tws_c.ti" in opt_data.keys():
            opt_data["cxff_tws_c.ti"] = opt_data["cxff_tws_c.ti"].where(
                opt_data["cxff_tws_c.ti"].channel == 0
            )

        if "cxff_pi.ti" in opt_data.keys():
            opt_data["cxff_pi.ti"] = opt_data["cxff_pi.ti"].where(
                (opt_data["cxff_pi.ti"].channel > 2) & (opt_data["cxff_pi.ti"].channel < 5)
            )

        self.opt_data = opt_data


@dataclass
class PhantomData(DataContext):
    phantoms = True

    def read_data(
        self,
    ):
        self.reader = ReadST40(
            self.pulse,
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
        )
        self.reader(
            self.diagnostics,
            revisions=self.reader_settings.revisions,
            filters=self.reader_settings.filters,
        )
        missing_keys = set(self.diagnostics) - set(self.reader.binned_data.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"missing data: {missing_keys}")
        self.equilibrium = self.reader.equilibrium
        self.transforms = self.reader.transforms
        self.raw_data = {}
        self.binned_data = {}

    def data_strategy(self):
        print("Data strategy: Phantom data")
        return self.model_callable()

    def process_data(self, model_callable: Callable):
        self.pre_process_data(model_callable)
        self.opt_data = flatdict.FlatDict(self.binned_data, ".")


@dataclass
class MockData(PhantomData):
    diagnostic_transforms: dict = field(
        default_factory=lambda: {
            "xrcs": helike_transform_example(1),
            "smmh1": smmh1_transform_example(1),
            "cxff_pi": pi_transform_example(5),
            "cxff_tws_c": pi_transform_example(3),
            "ts": ts_transform_example(11),
            "efit": lambda: None,
            # placeholder to stop missing_transforms error
        }
    )

    def read_data(self):
        print("Reading mock equilibrium / transforms")
        self.equilibrium = fake_equilibrium(
            self.tstart,
            self.tend,
            self.dt,
        )
        missing_transforms = list(
            set(self.diagnostics).difference(self.diagnostic_transforms.keys())
        )
        if missing_transforms:
            raise ValueError(f"Missing transforms: {missing_transforms}")

        self.transforms = self.diagnostic_transforms
        self.binned_data: dict = {}
        self.raw_data: dict = {}


@dataclass
class OptimiserEmceeSettings:
    param_names: list
    priors: dict
    iterations: int = 1000
    nwalkers: int = 50
    burn_frac: float = 0.20
    sample_method: str = "random"
    starting_samples: int = 100
    stopping_criteria: str = "mode"
    stopping_criteria_factor: float = 0.01
    stopping_criteria_sample: int = 20
    stopping_criteria_debug: bool = False


@dataclass  # type: ignore[misc]
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
    def init_optimiser(self, blackbox_func: Callable, *args, **kwargs):  # type: ignore
        ndim = len(self.optimiser_settings.param_names)
        self.move = [
            (emcee.moves.StretchMove(), 0.0),
            (emcee.moves.DEMove(), 1.0),
            (emcee.moves.DESnookerMove(), 0.0),
        ]
        self.optimiser = emcee.EnsembleSampler(
            self.optimiser_settings.nwalkers,
            ndim,
            log_prob_fn=blackbox_func,
            parameter_names=self.optimiser_settings.param_names,
            moves=self.move,
        )

    def sample_start_points(
        self,
    ):

        if self.optimiser_settings.sample_method == "high_density":

            self.start_points = self.sample_from_high_density_region(
                param_names=self.optimiser_settings.param_names,
                priors=self.optimiser_settings.priors,
                optimiser=self.optimiser,
                nwalkers=self.optimiser_settings.nwalkers,
                nsamples=self.optimiser_settings.starting_samples,
            )

        elif self.optimiser_settings.sample_method == "random":
            self.start_points = sample_from_priors(
                param_names=self.optimiser_settings.param_names,
                priors=self.optimiser_settings.priors,
                size=self.optimiser_settings.nwalkers,
            )
        else:
            raise ValueError(
                f"Sample method: {self.optimiser_settings.sample_method} "
                f"not recognised, Defaulting to random sampling"
            )

    def sample_from_high_density_region(
        self, param_names: list, priors: dict, optimiser, nwalkers: int, nsamples=100
    ):

        # TODO: remove repeated code
        start_points = sample_from_priors(param_names, priors, size=nsamples)

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
            _ln_prior = ln_prior(
                priors,
                start,
            )
            # Convert from dictionary of arrays -> array,
            # then filtering out where ln_prior is -infinity
            accepted_samples = np.array(list(start.values()))[:, _ln_prior != -np.inf]
            samples = np.append(samples, accepted_samples, axis=1)
        start_points = samples[:, 0:nwalkers].transpose()
        return start_points

    def run(
        self,
    ):

        if self.optimiser_settings.stopping_criteria == "mode":
            self.autocorr = sample_with_moments(
                self.optimiser,
                self.start_points,
                self.optimiser_settings.iterations,
                self.optimiser_settings.param_names.__len__(),
                auto_sample=self.optimiser_settings.stopping_criteria_sample,
                stopping_factor=self.optimiser_settings.stopping_criteria_factor,
                debug=self.optimiser_settings.stopping_criteria_debug,
            )
        else:
            raise ValueError(
                f"Stopping criteria: "
                f"{self.optimiser_settings.stopping_criteria} invalid"
            )

        optimiser_results = self.format_results()
        return optimiser_results

    def format_results(self):
        results = {}
        _blobs = self.optimiser.get_blobs(
            discard=int(self.optimiser.iteration * self.optimiser_settings.burn_frac),
            flat=True,
        )
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
        results["prior_sample"] = sample_from_priors(
            self.optimiser_settings.param_names,
            self.optimiser_settings.priors,
            size=int(1e4),
        )

        post_sample = self.optimiser.get_chain(
            discard=int(self.optimiser.iteration * self.optimiser_settings.burn_frac),
            flat=True,
        )
        # pad index dim with maximum number of iterations
        max_iter = self.optimiser_settings.iterations * self.optimiser_settings.nwalkers
        npad = (
            (
                0,
                int(
                    max_iter * (1 - self.optimiser_settings.burn_frac)
                    - post_sample.shape[0]
                ),
            ),
            (0, 0),
        )
        results["post_sample"] = np.pad(post_sample, npad, constant_values=np.nan)

        results["auto_corr"] = self.autocorr
        return results


@dataclass
class BayesBBSettings:
    diagnostics: list = field(default_factory=lambda: DEFAULT_DIAG_NAMES)
    param_names: list = field(default_factory=lambda: OPTIMISED_PARAMS)
    opt_quantity: list = field(default_factory=lambda: OPTIMISED_QUANTITY)
    priors: dict = field(default_factory=lambda: DEFAULT_PRIORS)
    percent_error: float = 0.10

    """
    TODO: default methods / getter + setters
          print warning if using default values
    """

    def __post_init__(self):
        missing_quantities = [
            quant
            for quant in self.opt_quantity
            if quant.split(".")[0] not in self.diagnostics
        ]
        if missing_quantities:
            raise ValueError(f"{missing_quantities} missing the relevant diagnostic")

        # check all priors are defined
        for name in self.param_names:
            if name in self.priors.keys():
                if hasattr(self.priors[name], "rvs"):
                    continue
                else:
                    raise TypeError(f"prior object {name} missing rvs method")
            else:
                raise ValueError(f"Missing prior for {name}")


class BayesWorkflow(AbstractBayesWorkflow):
    def __init__(
        self,
        blackbox_settings: BayesBBSettings,
        data_context: DataContext,
        plasma_context: PlasmaContext,
        model_context: ModelContext,
        optimiser_context: EmceeOptimiser,
        tstart: float = 0.02,
        tend: float = 0.10,
        dt: float = 0.01,
    ):
        self.blackbox_settings = blackbox_settings
        self.data_context = data_context
        self.plasma_context = plasma_context
        self.model_context = model_context
        self.optimiser_context = optimiser_context

        self.tstart = tstart
        self.tend = tend
        self.dt = dt

        self.blackbox = BayesBlackBox(
            data=self.data_context.opt_data,
            plasma_context=self.plasma_context,
            model_context=self.model_context,
            quant_to_optimise=self.blackbox_settings.opt_quantity,
            priors=self.blackbox_settings.priors,
            percent_error=self.blackbox_settings.percent_error,
        )

        self.optimiser_context.init_optimiser(
            self.blackbox.ln_posterior,
        )

    def __call__(
        self,
        filepath="./results/test/",
        run="RUN01",
        run_info="Default run",
        mds_write=False,
        best=True,
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
        self.midplane_blobs = self.plasma_context.map_profiles_to_midplane(blobs)

        opt_samples = {}
        for key in results[0].keys():
            if key == "blobs":
                continue
            _opt_samples = [result[key] for result in results]
            opt_samples[key] = np.array(_opt_samples)
        self.opt_samples = opt_samples

        result = self._build_result_dict()
        self.result = dict(self.result, **result)

        if mds_write or plot:
            self.save_pickle(
                self.result,
                filepath=filepath,
            )

        self.result = dict_of_dataarray_to_numpy(self.result)

        if mds_write:
            print("Writing to MDS+")
            tree_exists = does_tree_exist(pulse_to_write)
            if tree_exists:
                mode = "EDIT"
            else:
                mode = "NEW"

            self.node_structure = create_nodes(
                pulse_to_write=pulse_to_write,
                best=best,
                run=run,
                diagnostic_quantities=self.blackbox_settings.opt_quantity,
                mode=mode,
            )

            util.standard_fn_MDSplus.make_ST40_subtree("BDA", pulse_to_write)

            util.StandardNodeWriting(
                pulse_number=pulse_to_write,  # pulse number for which data should be written
                dict_node_info=self.node_structure,  # node information file
                nodes_to_write=[],  # selective nodes to be written
                data_to_write=self.result,
                debug=False,
            )
        if plot:
            plot_bayes_result(filepath=filepath)
        return


if __name__ == "__main__":
    pulse = None
    tstart = 0.05
    tend = 0.06
    dt = 0.01

    diagnostics = [
        # "xrcs",
        # "efit",
        # "smmh1",
        "cxff_pi",
        "cxff_tws_c",
        # "ts",
    ]
    # diagnostic_quantities
    opt_quant = [
        # "xrcs.spectra",
        # "efit.wp",
        # "smmh1.ne",
        "cxff_pi.ti",
        "cxff_tws_c.ti",
        # "ts.te",
        # "ts.ne",
    ]
    opt_params = [
        # "Ne_prof.y0",
        # "Ne_prof.peaking",
        # "Te_prof.y0",
        # "Te_prof.peaking",
        # "Te_prof.wped",
        # "Te_prof.wcenter",
        "Ti_prof.y0",
        # "Ti_prof.peaking",
        "Ti_prof.wped",
        "Ti_prof.wcenter",
        # "Niz1_prof.y0",
        # "Niz1_prof.peaking",
        # "Niz1_prof.wcenter",
        # "Niz1_prof.wped",
    ]

    # BlackBoxSettings
    bayes_settings = BayesBBSettings(
        diagnostics=diagnostics,
        param_names=opt_params,
        opt_quantity=opt_quant,
        priors=DEFAULT_PRIORS,
    )

    data_settings = ReaderSettings(
        filters={}, revisions={}
    )  # Add general methods for filtering data co-ords to ReadST40
    data_context = MockData(
        pulse=pulse,
        diagnostics=diagnostics,
        tstart=tstart,
        tend=tend,
        dt=dt,
        reader_settings=data_settings,
    )
    # data_context = PhantomData(pulse=pulse, diagnostics=diagnostics,
    #                tstart=tstart, tend=tend, dt=dt, reader_settings=data_settings, )
    # data_context = ExpData(pulse=pulse, diagnostics=diagnostics,
    #                tstart=tstart, tend=tend, dt=dt, reader_settings=data_settings, )
    data_context.read_data()

    plasma_settings = PlasmaSettings(
        main_ion="h",
        impurities=("ar", "c"),
        impurity_concentration=(0.001, 0.04),
        n_rad=20,
    )
    plasma_context = PlasmaContext(
        plasma_settings=plasma_settings, profile_params=DEFAULT_PROFILE_PARAMS
    )

    model_settings = ModelSettings(call_kwargs={"xrcs": {"pixel_offset": 0.0}})

    model_context = ModelContext(
        diagnostics=diagnostics,
        plasma_context=plasma_context,
        equilibrium=data_context.equilibrium,
        transforms=data_context.transforms,
        model_settings=model_settings,
    )

    # Initialise context objects
    plasma_context.init_plasma(
        equilibrium=data_context.equilibrium,
        tstart=tstart,
        tend=tend,
        dt=dt,
    )

    # plasma_context.set_ts_profiles(data_context, split="LFS")

    plasma_context.save_phantom_profiles(phantoms=data_context.phantoms)

    model_context.update_model_kwargs(data_context.binned_data)
    model_context.init_models()

    data_context.process_data(
        model_context._build_bckc,
    )

    optimiser_settings = OptimiserEmceeSettings(
        param_names=bayes_settings.param_names,
        nwalkers=20,
        iterations=100,
        sample_method="high_density",
        starting_samples=50,
        burn_frac=0.20,
        stopping_criteria="mode",
        stopping_criteria_factor=0.005,
        stopping_criteria_debug=True,
        priors=bayes_settings.priors,
    )
    optimiser_context = EmceeOptimiser(optimiser_settings=optimiser_settings)

    workflow = BayesWorkflow(
        tstart=tstart,
        tend=tend,
        dt=dt,
        blackbox_settings=bayes_settings,
        data_context=data_context,
        optimiser_context=optimiser_context,
        plasma_context=plasma_context,
        model_context=model_context,
    )

    workflow(
        pulse_to_write=43000000,
        run="RUN01",
        mds_write=True,
        plot=True,
        filepath="./results/test/",
    )
