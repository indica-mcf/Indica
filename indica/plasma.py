from copy import deepcopy
from functools import lru_cache
import hashlib
import pickle
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import xarray as xr

from indica import Equilibrium
from indica.available_quantities import PLASMA_QUANTITIES
from indica.configs import MACHINE_CONFS
from indica.converters.time import get_tlabels_dt
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import default_atomic_data
import indica.physics as ph
from indica.profilers.profiler_base import ProfilerBase
from indica.utilities import build_dataarrays
from indica.utilities import format_dataarray
from indica.utilities import get_element_info


class Plasma:
    def __init__(
        self,
        tstart: float = 0.01,
        tend: float = 0.14,
        dt: float = 0.01,
        machine: str = "st40",
        impurities: Tuple[str, ...] = ("c", "ar"),
        impurity_concentration: Tuple[float, ...] = (0.02, 0.001),  # should be deleted!
        main_ion: str = "h",
        full_run: bool = False,
        n_rad: int = 41,
        n_R: int = 100,
        n_z: int = 100,
        verbose: bool = False,
    ):
        """
        Class for plasma objects.
        - Completely independent of experimental data.
        - Assign an equilibrium object for remapping
        - Independent parameters can be set, dependent ones are properties
        TODO: concentration should not be inputted in initialization!

        tstart
            Start time (s)
        tend
            End time (s)
        dt
            Delta t of time window
        machine
            Machine string identifier
        impurities
            Impurity elements present
        main_ion
            Main ion
        full_run
            If True: compute ionisation balance at every iteration
            If False: calculate default and interpolate
        """
        self.machine_conf = MACHINE_CONFS[machine]()
        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.full_run = full_run
        self.verbose = verbose
        elements: Tuple[str, ...] = (main_ion,)
        for elem in impurities:
            elements += (elem,)
        self.elements = elements
        self.main_ion = main_ion
        self.impurities = impurities
        self.impurity_concentration = impurity_concentration
        self.rho_type = "rhop"
        self.public_attributes = PLASMA_QUANTITIES["public_attrs"]
        self.private_attributes = PLASMA_QUANTITIES["private_attrs"]

        self.initialize_variables(n_rad, n_R, n_z)
        self.build_atomic_data()

    def set_equilibrium(self, equilibrium: Equilibrium):
        """Assign equilibrium object and associated private variables"""
        self.equilibrium = equilibrium

    def set_adf11(self, adf11: dict):
        self.adf11 = adf11

    def initialize_variables(self, n_rad: int = 41, n_R: int = 100, n_z: int = 100):
        """Initialize all class attributes"""
        # Define coordinates
        attrs_data: dict = {}
        R0, R1 = self.machine_conf.MACHINE_DIMS[0]
        z0, z1 = self.machine_conf.MACHINE_DIMS[1]
        attrs_data["R"] = np.linspace(R0, R1, n_R)
        attrs_data["z"] = np.linspace(z0, z1, n_z)
        attrs_data["index"] = np.arange(n_R)
        attrs_data["R_midplane"] = np.linspace(
            attrs_data["R"].min(), attrs_data["R"].max(), n_R
        )
        attrs_data["z_midplane"] = np.full_like(attrs_data["R_midplane"], 0.0)
        attrs_data["rhop"] = np.linspace(0, 1.0, n_rad)
        attrs_data["t"] = get_tlabels_dt(self.tstart, self.tend, self.dt)
        attrs_data["time_to_calculate"] = deepcopy(attrs_data["t"])
        attrs_data["element"] = list(self.elements)
        attrs_data["impurity"] = list(self.impurities)
        attrs_data["element_z"] = []
        attrs_data["element_a"] = []
        attrs_data["element_name"] = []
        attrs_data["element_symbol"] = []
        attrs_data["ion_charge"] = []
        for elem in self.elements:
            _z, _a, _name, _symbol = get_element_info(elem)
            attrs_data["element_z"].append(_z)
            attrs_data["element_a"].append(_a)
            attrs_data["element_name"].append(_name)
            attrs_data["element_symbol"].append(_symbol)

        # Create dataarrays that will be assigned to class attributes
        # excluding ionisation-stage-dependent variables
        _special = ["fz", "lz_tot"]
        all_quantities = deepcopy(self.public_attributes)
        all_quantities.update(self.private_attributes)
        for to_pop in _special:
            all_quantities.pop(to_pop)
        for quantity in all_quantities:
            if quantity in attrs_data:
                continue

            _, dims = all_quantities[quantity]
            _shape = []
            for dim in dims:
                _shape.append(np.size(attrs_data[dim]))
            attrs_data[quantity] = np.zeros(shape=tuple(_shape))
        attrs_dataarrays = build_dataarrays(
            attrs_data, all_quantities, include_error=False
        )

        # Fix coordinate inconsistencies
        attrs_dataarrays["impurity_density"] = attrs_dataarrays[
            "impurity_density"
        ].rename({"impurity": "element"})

        # Manually add ionisation-stage-dependent private attributes
        for quantity in _special:
            attrs_dataarrays[quantity]: dict = {}
        for i, elem in enumerate(attrs_data["element"]):
            nz = attrs_data["element_z"][i]
            ion_charge = np.arange(nz + 1)
            _coords = {
                "t": attrs_data["t"],
                "rhop": attrs_data["rhop"],
                "ion_charge": ion_charge,
            }
            _data = np.full(
                (
                    np.size(_coords["t"]),
                    np.size(_coords["rhop"]),
                    np.size(_coords["ion_charge"]),
                ),
                0.0,
            )
            for quantity in _special:
                datatype, _ = self.private_attributes[quantity]
                attrs_dataarrays[quantity][elem] = format_dataarray(
                    _data, datatype, _coords, make_copy=True
                )

        # Assign as attributes
        for attr in self.public_attributes:
            setattr(self, attr, attrs_dataarrays[attr])
        for attr in self.private_attributes:
            setattr(self, f"_{attr}", attrs_dataarrays[attr])

        # Parameter dependencies relating dependant to independent quantities
        self.Fz = CachedCalculation(
            self.calc_fz,
            [
                self.electron_density,
                self.electron_temperature,
                self.neutral_density,
                self.tau,
            ],
        )

        self.Ion_density = CachedCalculation(
            self.calc_ion_density,
            [
                self.electron_density,
                self.electron_temperature,
                self.impurity_density,
                self.fast_ion_density,
            ],
        )

        self.Zeff = CachedCalculation(
            self.calc_zeff,
            [
                self.electron_density,
                self.ion_density,
                self.meanz,
            ],
        )

        self.Pth = CachedCalculation(
            self.calc_pth,
            [
                self.electron_density,
                self.ion_density,
                self.electron_temperature,
                self.ion_temperature,
            ],
        )

        self.Ptot = CachedCalculation(
            self.calc_ptot,
            [
                self.electron_density,
                self.ion_density,
                self.electron_temperature,
                self.ion_temperature,
                self.fast_ion_pressure,
            ],
        )

        self.Lz_tot = CachedCalculation(
            self.calc_lz_tot,
            [
                self.electron_density,
                self.electron_temperature,
                self.fz,
                self.neutral_density,
            ],
        )

        self.Total_radiation = CachedCalculation(
            self.calc_total_radiation,
            [
                self.electron_density,
                self.ion_density,
                self.lz_tot,
            ],
        )

    @property
    def time_to_calculate(self):
        return self._time_to_calculate

    @time_to_calculate.setter
    def time_to_calculate(self, value):
        if np.size(value) == 1:
            self._time_to_calculate = float(value)
        else:
            self._time_to_calculate = np.array(value)

    @property
    def electron_pressure(self):
        self._electron_pressure.values = ph.calc_pressure(
            self.electron_density, self.electron_temperature
        )
        return self._electron_pressure

    @property
    def thermal_pressure(self):
        self._thermal_pressure.values = (
            ph.calc_pressure(self.ion_density, self.ion_temperature).sum("element")
            + self.electron_pressure
        )
        return self._thermal_pressure

    @property
    def pressure(self):
        self._pressure.values = self.thermal_pressure + self.fast_ion_pressure
        return self._pressure

    @property
    def fast_ion_pressure(self):
        # TODO: check whether degrees of freedom are correctly included...
        self._fast_ion_pressure.values = (
            self.fast_ion_pressure_parallel / 3
            + self.fast_ion_pressure_perpendicular * 2 / 3
        )
        return self._fast_ion_pressure

    @property
    def pth(self):
        return self.Pth()

    def calc_pth(self):
        for t in np.array(self.time_to_calculate, ndmin=1):
            self._pth.loc[dict(t=t)] = np.trapz(
                self.thermal_pressure.sel(t=t), self.volume.sel(t=t)
            )
        return self._pth

    @property
    def ptot(self):
        return self.Ptot()

    def calc_ptot(self):
        for t in np.array(self.time_to_calculate, ndmin=1):
            self._ptot.loc[dict(t=t)] = np.trapz(
                self.pressure.sel(t=t), self.volume.sel(t=t)
            )
        return self._ptot

    @property
    def wth(self):
        self._wth.values = 3 / 2 * self.pth
        return self._wth

    @property
    def wp(self):
        self._wp.values = 3 / 2 * self.ptot
        return self._wp

    @property
    def fz(self):
        return self.Fz()

    def calc_fz(self):
        for elem in self.elements:
            for t in np.array(self.time_to_calculate, ndmin=1):
                Te = self.electron_temperature.sel(t=t)
                Ne = self.electron_density.sel(t=t)
                tau = None
                if np.any(self.tau != 0):
                    tau = self.tau.sel(t=t)
                Nh = None
                if np.any(self.neutral_density != 0):
                    Nh = self.neutral_density.sel(t=t)
                if any(np.logical_not((Te > 0) * (Ne > 0))):
                    continue
                fz_tmp = self.fract_abu[elem](
                    Te, Ne=Ne, Nh=Nh, tau=tau, full_run=self.full_run
                )
                self._fz[elem].loc[dict(t=t)] = fz_tmp.transpose()
        return self._fz

    @property
    def zeff(self):
        return self.Zeff()

    def calc_zeff(self):
        self._zeff.values = self.ion_density * self.meanz**2 / self.electron_density
        return self._zeff

    @property
    def ion_density(self):
        return self.Ion_density()
        # return self.calc_ion_density()

    def calc_ion_density(self):
        for elem in self.impurities:
            self._ion_density.loc[dict(element=elem)] = self.impurity_density.sel(
                element=elem
            )

        self._ion_density.loc[dict(element=self.main_ion)] = (
            self.electron_density
            - self.fast_ion_density * self.meanz.sel(element=self.main_ion)
            - (self.impurity_density * self.meanz).sum("element")
        )
        return self._ion_density

    @property
    def lz_tot(self):
        return self.Lz_tot()

    def calc_lz_tot(self):
        fz = self.fz
        for elem in self.elements:
            for t in np.array(self.time_to_calculate, ndmin=1):
                Ne = self.electron_density.sel(t=t)
                Te = self.electron_temperature.sel(t=t)
                if any(np.logical_not((Te > 0) * (Ne > 0))):
                    continue
                Fz = fz[elem].sel(t=t).transpose()
                Nh = None
                if np.any(self.neutral_density.sel(t=t) != 0):
                    Nh = self.neutral_density.sel(t=t)
                self._lz_tot[elem].loc[dict(t=t)] = self.power_loss_tot[elem](
                    Te, Fz, Ne=Ne, Nh=Nh, full_run=self.full_run
                ).transpose()
        return self._lz_tot

    @property
    def total_radiation(self):
        return self.Total_radiation()

    def calc_total_radiation(self):
        lz_tot = self.lz_tot
        ion_density = self.ion_density
        for elem in self.elements:
            total_radiation = (
                lz_tot[elem].sum("ion_charge")
                * self.electron_density
                * ion_density.sel(element=elem)
            )
            self._total_radiation.loc[dict(element=elem)] = xr.where(
                total_radiation >= 0,
                total_radiation,
                0.0,
            )
        return self._total_radiation

    @property
    def meanz(self):
        fz = self.fz
        for elem in self.elements:
            self._meanz.loc[dict(element=elem)] = (fz[elem] * fz[elem].ion_charge).sum(
                "ion_charge"
            )
        return self._meanz

    @property
    def prad_tot(self):
        total_radiation = self.total_radiation
        for elem in self.elements:
            for t in np.array(self.time_to_calculate, ndmin=1):
                self._prad_tot.loc[dict(element=elem, t=t)] = np.trapz(
                    total_radiation.sel(element=elem, t=t), self.volume.sel(t=t)
                )
        return self._prad_tot

    @property
    def volume(self):
        return self.equilibrium.volume.interp(rhop=self.rhop, t=self.t)

    @property
    def area(self):
        return self.equilibrium.area.interp(rhop=self.rhop, t=self.t)

    @property
    def rmjo(self):
        return self.equilibrium.rmjo.interp(rhop=self.rhop, t=self.t)

    @property
    def rmji(self):
        return self.equilibrium.rmji.interp(rhop=self.rhop, t=self.t)

    @property
    def rmag(self):
        return self.equilibrium.rmag.interp(t=self.t)

    @property
    def zmag(self):
        return self.equilibrium.zmag.interp(t=self.t)

    @property
    def rmin(self):
        return (self.rmjo - self.rmji) / 2.0

    def set_impurity_concentration(
        self,
        element: str,
        concentration: float,
        t: Optional[LabeledArray] = None,
        flat_zeff: bool = False,
    ):
        """
        Sets impurity density for a specific element = concentration * electron_density

        Parameters
        ----------
        element
            string impurity identifier
        concentration
            value of the desired concentration
        t
            time for which concentration is to be set
        flat_zeff
            if True, modifies impurity density to get a ~ flat Zeff contribution
        """
        if t is None:
            t = self.time_to_calculate

        if element in self.elements:
            el_dens = self.electron_density.sel(t=t)
            _imp_dens = el_dens * concentration
            if flat_zeff and np.count_nonzero(_imp_dens) != 0:
                meanz = self.meanz.sel(element=element).sel(t=t)
                _zeff = _imp_dens * meanz**2 / el_dens
                zeff_core = _zeff.where(self.rhop < 0.5).mean("rhop")
                imp_dens = el_dens * zeff_core / meanz**2
            else:
                imp_dens = _imp_dens

            self.impurity_density.loc[dict(element=element, t=t)] = imp_dens.values

    def build_atomic_data(self):
        """
        Assigns default atomic fractional abundance and radiated power operators
        """
        fract_abu, power_loss_tot = default_atomic_data(self.elements)
        self.fract_abu = fract_abu
        self.power_loss_tot = power_loss_tot

    # TODO: if ion asymmetry parameters are not == 0, calculate 2D (R, z) maps
    def map_to_2d(self):
        """
        Calculate total radiated power on a 2D poloidal plane
        including effects from poloidal asymmetries
        """
        print("\n Not implemented yet")

    def write_to_pickle(self, pulse: int = None):
        with open(f"data_{pulse}.pkl", "wb") as f:
            pickle.dump(
                self,
                f,
            )


# Generalized dependency caching
class TrackDependecies:
    def __init__(
        self,
        operator: Callable,
        dependencies: list,
    ):
        """
        Call operator only if dependencies variables have changed.
        Currently using np array, can change if needed

        Parameters
        ----------
        operator
            Function to be called
        dependencies
            Tuple of variables to be tracked
        """
        self.operator = operator
        self.dependencies = dependencies

    def numpyhash(
        self,
        nparray: np.array,
    ):
        a = nparray.view(np.uint8)
        return hashlib.sha1(a).hexdigest()

    def __hash__(self):
        """
        Caching of dependencies

        xr.DataArray, np.ndarray and dictionaries of xr.DataArrays currently permitted

        TODO: upgrade so other objects being tracked, e.g. Equilibrium
        """
        _dependencies = []
        for dependency in self.dependencies:
            if type(dependency) == dict:
                for data in dependency.values():
                    _dependencies.append(data.data)
            elif type(dependency) == xr.DataArray:
                _dependencies.append(dependency.data)
            elif type(dependency) == np.ndarray:
                _dependencies.append(dependency)
            else:
                print(type(dependency))
                raise NotImplementedError(
                    "Hashing implemented for xr.DataArray, np.ndarray"
                )

        hashable = tuple((self.numpyhash(dependency),) for dependency in _dependencies)
        return hash(hashable)


class CachedCalculation(TrackDependecies):
    def __init__(self, operator: Callable, dependencies: list, verbose: bool = False):
        self.verbose = verbose
        super(CachedCalculation, self).__init__(operator, dependencies)

    @lru_cache()
    def __call__(self):
        if self.verbose:
            print("Calculating")
        return deepcopy(self.operator())


class PlasmaProfiler:
    def __init__(
        self,
        plasma: Plasma,
        profilers: dict[ProfilerBase],
        plasma_attribute_names=None,
        map_vtor: bool = False,
    ):
        """
        Interface Profiler objects with Plasma object to generate plasma profiles
        and update them.

        Parameters
        ----------
        plasma
            Plasma object
        profilers
            dictionary of Profiler objects to generate profiles
        """

        if plasma_attribute_names is None:
            plasma_attribute_names = [
                "electron_temperature",
                "electron_density",
                "ion_temperature",
                "ion_density",
                "impurity_density",
                "fast_ion_density",
                "fast_ion_pressure",
                "neutral_density",
                "zeff",
                "meanz",
                "wp",
                "wth",
                "pressure",
                "thermal_pressure",
                "toroidal_rotation",
            ]
        self.plasma = plasma
        self.profilers = profilers
        self.plasma_attribute_names = plasma_attribute_names
        self.map_vtor = map_vtor
        self.phantom = None
        self.phantom_profiles = None

    def update_profilers(self, profilers: dict):
        for profile_name, profiler in profilers.items():
            self.profilers[profile_name] = profiler

    def set_profiles(self, profiles: dict[xr.DataArray], t: float = None):
        if t is None:
            t = self.plasma.time_to_calculate

        for profile_name, profile in profiles.items():
            _prof_identifiers = profile_name.split(
                ":"
            )  # impurities have ':' to identify elements
            if profile_name.__contains__(":"):
                if _prof_identifiers[1] in self.plasma.elements:
                    getattr(self.plasma, _prof_identifiers[0]).loc[
                        dict(t=t, element=_prof_identifiers[-1])
                    ] = profile
                else:
                    print(
                        f"profile {profile_name} can't be set because "
                        f"{_prof_identifiers[1]} not in plasma.elements"
                    )
            else:
                getattr(self.plasma, profile_name).loc[dict(t=t)] = profile

    def save_phantoms(self, phantom=False):
        #  if phantoms return profiles otherwise return empty arrays
        self.phantom = phantom
        phantom_profiles = {"PSI_NORM": self.plasma_attributes()}
        if not phantom:
            for key, value in phantom_profiles["PSI_NORM"].items():
                phantom_profiles["PSI_NORM"][key] = value * 0
        phantom_profiles["R_MIDPLANE"] = self.map_plasma_profile_to_midplane(
            phantom_profiles["PSI_NORM"]
        )
        self.phantom_profiles = phantom_profiles
        return phantom_profiles

    def map_plasma_profile_to_midplane(self, profiles: dict):
        """
        Map profiles from flux space to real space on z=0
        """
        midplane_profiles: dict = {}

        R = self.plasma.R_midplane
        z = self.plasma.z_midplane
        _rhop, _, _ = self.plasma.equilibrium.flux_coords(R, z, self.plasma.t)
        rhop = _rhop.swap_dims({"dim_0": "R"})

        for key, value in profiles.items():
            if "rhop" not in value.dims:
                continue
            if not hasattr(self.plasma, key):
                continue
            midplane_profiles[key] = value.interp(rhop=rhop)
        return midplane_profiles

    def plasma_attributes(self):
        plasma_attributes = {}
        for attribute in self.plasma_attribute_names:
            plasma_attributes[attribute] = getattr(self.plasma, attribute).sel(
                t=self.plasma.time_to_calculate
            )
        return plasma_attributes

    def map_toroidal_rotation_to_ion_temperature(
        self,
    ):

        self.plasma.toroidal_rotation = (
            self.plasma.ion_temperature
            / self.plasma.ion_temperature.max("rhop")
            * self.plasma.toroidal_rotation.max("rhop")
        )

    def __call__(self, parameters: dict = None, t=None):
        """
        Set parameters of given profilers and assign to plasma profiles

        parameters
            Flat dictionary of {"profile_name.param_name":value}
            Special case for impurity density:
                {"profile_name:element.param_name":value}
        t
            time points to apply changes
        """
        if parameters is None:
            parameters = {}

        _profiles_to_update: list = []

        # set params for all profilers
        for parameter_name, parameter in parameters.items():
            profile_name, profile_param_name = parameter_name.split(".")
            if profile_name not in self.profilers.keys():
                continue

            if not hasattr(self.profilers[profile_name], profile_param_name):
                raise ValueError(
                    f"No parameter {profile_param_name} available for {profile_name}"
                )
            self.profilers[profile_name].set_parameters(
                **{profile_param_name: parameter}
            )
            _profiles_to_update.append(profile_name)

        # Update only desired profiles or if no parameters given update all
        if _profiles_to_update:
            profiles_to_update = list(set(_profiles_to_update))
        else:
            print("no profile params given so updating all")
            profiles_to_update = list(self.profilers.keys())

        updated_profiles = {
            profile_to_update: self.profilers[profile_to_update]()
            for profile_to_update in profiles_to_update
        }
        self.set_profiles(updated_profiles, t)

        if "ion_temperature" in _profiles_to_update and self.map_vtor:
            self.map_toroidal_rotation_to_ion_temperature()
