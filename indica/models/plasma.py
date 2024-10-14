from copy import deepcopy
from functools import lru_cache
import hashlib
import pickle
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import xarray as xr
from xarray import DataArray

from indica.available_quantities import PLASMA_QUANTITIES
from indica.converters.time import convert_in_time_dt
from indica.converters.time import get_tlabels_dt
from indica.equilibrium import Equilibrium
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import default_atomic_data
import indica.physics as ph
from indica.profilers.profiler_base import ProfilerBase
from indica.utilities import build_dataarrays
from indica.utilities import format_dataarray
from indica.utilities import get_element_info
from indica.utilities import print_like


class Plasma:
    def __init__(
        self,
        tstart: float = 0.01,
        tend: float = 0.14,
        dt: float = 0.01,
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
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
        - completely independent of experimental data
        - can be assigned an equilibrium object for remapping (currently 1D)
        - independent parameters can be set, dependent ones are properties
        TODO: concentration should not be inputted in initialization!

        Parameters
        ----------
        tstart
            Start time (s)
        tend
            End time (s)
        dt
            Delta t of time window
        machine_dimensions
            (R, z) limits of tokamak (m)
        impurities
            Impurity elements present
        main_ion
            Main ion
        pulse
            Pulse number, if this is associated with an experiment
        full_run
            If True: compute ionisation balance at every iteration
            If False: calculate default and interpolate
        """

        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.full_run = full_run
        self.verbose = verbose
        self.main_ion = main_ion
        self.impurities = impurities
        self.impurity_concentration = impurity_concentration
        elements: Tuple[str, ...] = (self.main_ion,)
        for elem in impurities:
            elements += (elem,)
        self.elements = elements
        self.machine_dimensions = machine_dimensions
        self.rho_type = "rhop"
        if self.rho_type != "rhop":
            print_like("Only rhop in input for the time being...")
            raise AssertionError

        self.public_attributes = PLASMA_QUANTITIES["public_attrs"]
        self.private_attributes = PLASMA_QUANTITIES["private_attrs"]
        self.all_attributes = self.public_attributes.update(self.private_attributes)

        self.initialize_variables(n_rad, n_R, n_z)
        self.build_atomic_data()

        self.equilibrium: Equilibrium

    def set_equilibrium(self, equilibrium: Equilibrium):
        """
        Assign equilibrium object
        """
        self.equilibrium = equilibrium
        self._volume.values = self.convert_in_time(
            self.equilibrium.volume.interp(rhop=self.rho)
        )
        self._area.values = self.convert_in_time(
            self.equilibrium.area.interp(rhop=self.rho)
        )
        self._rmjo.values = self.convert_in_time(
            self.equilibrium.rmjo.interp(rhop=self.rho)
        )
        self._rmji.values = self.convert_in_time(
            self.equilibrium.rmji.interp(rhop=self.rho)
        )
        self._rmag.values = self.convert_in_time(self.equilibrium.rmag)
        self._zmag.values = self.convert_in_time(self.equilibrium.zmag)
        self._rmin.values = (self._rmjo - self._rmji) / 2.0

    def set_adf11(self, adf11: dict):
        self.adf11 = adf11

    def initialize_variables(self, n_rad: int = 41, n_R: int = 100, n_z: int = 100):
        """
        Initialize all class attributes
        """

        # Coordinates
        attrs_data: dict = {}
        R0, R1 = self.machine_dimensions[0]
        z0, z1 = self.machine_dimensions[1]
        attrs_data["R"] = np.linspace(R0, R1, n_R)
        attrs_data["z"] = np.linspace(z0, z1, n_z)
        attrs_data["index"] = np.arange(n_R)
        attrs_data["R_midplane"] = np.linspace(self.R.min(), self.R.max(), n_R)
        attrs_data["z_midplane"] = np.full_like(attrs_data["R_midplane"], 0.0)
        attrs_data["rho"] = np.linspace(0, 1.0, n_rad)
        attrs_data["t"] = get_tlabels_dt(self.tstart, self.tend, self.dt)
        attrs_data["time_to_calculate"] = deepcopy(attrs_data["t"])
        element_z, element_a, element_name, element_symbol, ion_charge = (
            [],
            [],
            [],
            [],
            [],
        )
        for elem in self.elements:
            _z, _a, _name, _symbol = get_element_info(elem)
            element_z.append(_z)
            element_a.append(_a)
            element_name.append(_name)
            element_symbol.append(_symbol)
        attrs_data["element"] = list(
            self.element
        )  # to be harmonised with self.elements....
        attrs_data["element_z"] = element_z
        attrs_data["element_a"] = element_a
        attrs_data["element_name"] = element_name
        attrs_data["element_symbol"] = element_symbol

        # Initialize plasma attributes, public and private (see PLASMA_QUANTITIES)
        # and create dataarrays excluding ionisation-stage-dependent variables
        _special = ["fz", "lz_tot", "lz_sxr"]
        all_quantities = deepcopy(self.all_attributes)
        for to_pop in _special:
            all_quantities.pop(to_pop)
        for quantity in all_quantities:
            _, dims = all_quantities[quantity]
            _shape = ()
            _coords: dict = {}
            for dim in dims:
                _coords[dim] = attrs_data[dim]
                _shape += len(attrs_data[dim])
            attrs_data[quantity] = np.zeros(shape=_shape)
        attrs_dataarrays = build_dataarrays(
            attrs_data, self.quantities, include_error=False
        )

        # Manually add ionisation-stage-dependent private attributes
        for quantity in _special:
            attrs_dataarrays[quantity] = {}

        for i, elem in enumerate(attrs_data["element"]):
            nz = element_z[i]
            ion_charge = np.arange(nz + 1)
            _coords = {
                "t": attrs_data["t"],
                "rhop": attrs_data["rho"],
                "ion_charge": ion_charge,
            }
            _data = np.full(
                (len(_coords["t"]), len(_coords["rho"]), _coords["ion_charge"]), 0.0
            )
            for quantity in _special:
                datatype, _ = self.all_attributes[quantity]
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
                self.fast_density,
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
                self.pressure_fast,
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

        self.Lz_sxr = CachedCalculation(
            self.calc_lz_sxr,
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

        self.Sxr_radiation = CachedCalculation(
            self.calc_sxr_radiation,
            [
                self.electron_density,
                self.ion_density,
                self.lz_sxr,
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
    def pressure_el(self):
        self._pressure_el.values = ph.calc_pressure(
            self.electron_density, self.electron_temperature
        )
        return self._pressure_el

    @property
    def pressure_th(self):
        self._pressure_th.values = (
            ph.calc_pressure(self.ion_density, self.ion_temperature).sum("element")
            + self.pressure_el
        )
        return self._pressure_th

    @property
    def pressure_tot(self):
        self._pressure_tot.values = self.pressure_th + self.pressure_fast
        return self._pressure_tot

    @property
    def pressure_fast(self):
        # TODO: check whether degrees of freedom are correctly included...
        self._pressure_fast.values = (
            self.pressure_fast_parallel / 3 + self.pressure_fast_perpendicular * 2 / 3
        )
        return self._pressure_fast

    @property
    def pth(self):
        return self.Pth()

    def calc_pth(self):
        for t in np.array(self.time_to_calculate, ndmin=1):
            self._pth.loc[dict(t=t)] = np.trapz(
                self.pressure_th.sel(t=t), self.volume.sel(t=t)
            )
        return self._pth

    @property
    def ptot(self):
        return self.Ptot()

    def calc_ptot(self):
        for t in np.array(self.time_to_calculate, ndmin=1):
            self._ptot.loc[dict(t=t)] = np.trapz(
                self.pressure_tot.sel(t=t), self.volume.sel(t=t)
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
            - self.fast_density * self.meanz.sel(element=self.main_ion)
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
    def lz_sxr(self):
        return self.Lz_sxr()

    def calc_lz_sxr(self):
        fz = self.fz
        for elem in self.elements:
            if elem not in self.power_loss_sxr.keys():
                continue
            for t in np.array(self.time_to_calculate, ndmin=1):
                Ne = self.electron_density.sel(t=t)
                Te = self.electron_temperature.sel(t=t)
                if any(np.logical_not((Te > 0) * (Ne > 0))):
                    continue
                Fz = fz[elem].sel(t=t).transpose()
                Nh = None
                if np.any(self.neutral_density.sel(t=t) != 0):
                    Nh = self.neutral_density.sel(t=t)
                self._lz_sxr[elem].loc[dict(t=t)] = self.power_loss_sxr[elem](
                    Te, Fz, Ne=Ne, Nh=Nh, full_run=self.full_run
                ).transpose()
        return self._lz_sxr

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
    def sxr_radiation(self):
        return self.Sxr_radiation()

    def calc_sxr_radiation(self):
        if not hasattr(self, "power_loss_sxr"):
            return None

        lz_sxr = self.lz_sxr
        ion_density = self.ion_density
        for elem in self.elements:
            sxr_radiation = (
                lz_sxr[elem].sum("ion_charge")
                * self.electron_density
                * ion_density.sel(element=elem)
            )
            self._sxr_radiation.loc[dict(element=elem)] = xr.where(
                sxr_radiation >= 0,
                sxr_radiation,
                0.0,
            )
        return self._sxr_radiation

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
    def prad_sxr(self):
        if not hasattr(self, "power_loss_sxr"):
            return None

        sxr_radiation = self.sxr_radiation
        for elem in self.elements:
            for t in np.array(self.time_to_calculate, ndmin=1):
                self._prad_sxr.loc[dict(element=elem, t=t)] = np.trapz(
                    sxr_radiation.sel(element=elem, t=t), self.volume.sel(t=t)
                )
        return self._prad_sxr

    @property
    def volume(self):
        return self._volume

    @property
    def area(self):
        return self._area

    @property
    def rmjo(self):
        return self._rmjo

    @property
    def rmji(self):
        return self._rmji

    @property
    def rmag(self):
        return self._rmag

    @property
    def zmag(self):
        return self._zmag

    @property
    def rmin(self):
        return self._rmin

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
                zeff_core = _zeff.where(self.rho < 0.5).mean("rhop")
                imp_dens = el_dens * zeff_core / meanz**2
            else:
                imp_dens = _imp_dens

            self.impurity_density.loc[dict(element=element, t=t)] = imp_dens.values

    def convert_in_time(self, value: DataArray, method="linear"):
        binned = convert_in_time_dt(
            self.tstart, self.tend, self.dt, value, method=method
        )

        return binned

    def build_atomic_data(self):
        """
        Assigns default atomic fractional abundance and radiated power operators
        TODO: SXR radiation shouldn't be here? can it be set only in diagnostic model?
        """
        fract_abu, power_loss_tot, power_loss_sxr = default_atomic_data(self.elements)
        self.fract_abu = fract_abu
        self.power_loss_tot = power_loss_tot
        self.power_loss_sxr = power_loss_sxr

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
                "fast_density",
                "pressure_fast",
                "neutral_density",
                "zeff",
                "meanz",
                "wp",
                "wth",
                "pressure_tot",
                "pressure_th",
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
        phantom_profiles = self.plasma_attributes()
        if not phantom:
            for key, value in phantom_profiles.items():
                phantom_profiles[key] = value * 0
        self.phantom_profiles = phantom_profiles
        return phantom_profiles

    def map_plasma_profile_to_midplane(self, profiles: dict):
        """
        Map profiles from flux space to real space on z=0
        """
        midplane_profiles: dict = {}

        R = self.plasma.R_midplane
        z = self.plasma.z_midplane
        _rho, _, _ = self.plasma.equilibrium.flux_coords(R, z, self.plasma.t)
        rho = _rho.swap_dims({"index": "R"}).drop_vars("index")

        for key, value in profiles.items():
            if "rhop" not in value.dims:
                continue
            if not hasattr(self.plasma, key):
                continue
            midplane_profiles[key] = value.interp(rhop=rho)
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
