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
from indica.configs import MACHINE_CONFS
from indica.converters.time import get_tlabels_dt
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import default_atomic_data
import indica.physics as ph
from indica.profilers.profiler_base import ProfilerBase
from indica.utilities import format_coord
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
        self.equilibrium: Equilibrium
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

        # Machine attributes
        R0, R1 = self.machine_conf.MACHINE_DIMS[0]
        z0, z1 = self.machine_conf.MACHINE_DIMS[1]
        self.R = format_coord(np.linspace(R0, R1, n_R), "R")
        self.z = format_coord(np.linspace(z0, z1, n_z), "z")

        index = np.arange(n_R)
        R_midplane = np.linspace(self.R.min(), self.R.max(), n_R)
        z_midplane = np.full_like(R_midplane, 0.0)
        coords_midplane = {"index": index}
        self.R_midplane = format_dataarray(R_midplane, "R_midplane", coords_midplane)
        self.z_midplane = format_dataarray(z_midplane, "z_midplane", coords_midplane)

        # Time and radial grid
        self.rhop = format_coord(np.linspace(0, 1.0, n_rad), self.rho_type)
        self.t = format_coord(get_tlabels_dt(self.tstart, self.tend, self.dt), "t")
        self.time_to_calculate = deepcopy(self.t)

        # Elements (ions and specifics of impurities)
        element_z, element_a, element_name, element_symbol = [], [], [], []
        for elem in self.elements:
            _z, _a, _name, _symbol = get_element_info(elem)
            element_z.append(_z)
            element_a.append(_a)
            element_name.append(_name)
            element_symbol.append(_symbol)

        coords_elem = {"element": list(self.elements)}
        self.element_z = format_dataarray(element_z, "atomic_number", coords_elem)
        self.element_a = format_dataarray(element_a, "atomic_weight", coords_elem)
        self.element_name = format_dataarray(element_name, "element_name", coords_elem)
        self.element_symbol = format_dataarray(
            element_symbol, "element_symbol", coords_elem
        )

        # Assign data to variables
        nt = len(self.t)
        nr = len(self.rhop)
        nel = len(self.elements)
        nimp = len(self.impurities)
        data1d_time = np.zeros(nt)
        data2d = np.zeros((nt, nr))
        data2d_elem = np.zeros((nel, nt))
        data3d = np.zeros((nel, nt, nr))
        data3d_imp = np.zeros((nimp, nt, nr))

        coords1d_time = {"t": self.t}
        coords2d = {"t": self.t, self.rho_type: self.rhop}
        coords2d_elem = {"element": list(self.elements), "t": self.t}
        coords3d = {
            "element": list(self.elements),
            "t": self.t,
            self.rho_type: self.rhop,
        }
        coords3d_imp = {
            "element": list(self.impurities),
            "t": self.t,
            self.rho_type: self.rhop,
        }

        # Independent plasma quantities
        self.electron_temperature = format_dataarray(
            data2d, "electron_temperature", coords2d, make_copy=True
        )
        self.electron_density = format_dataarray(
            data2d, "electron_density", coords2d, make_copy=True
        )
        self.neutral_density = format_dataarray(
            data2d, "neutral_density", coords2d, make_copy=True
        )
        self.tau = format_dataarray(data2d, "residence_time", coords2d, make_copy=True)
        self.ion_temperature = format_dataarray(
            data2d, "ion_temperature", coords2d, make_copy=True
        )
        self.toroidal_rotation = format_dataarray(
            data2d, "toroidal_rotation", coords2d, make_copy=True
        )
        self.impurity_density = format_dataarray(
            data3d_imp, "impurity_density", coords3d_imp, make_copy=True
        )
        self.fast_ion_density = format_dataarray(
            data2d, "fast_ion_density", coords2d, make_copy=True
        )
        self.parallel_fast_ion_pressure = format_dataarray(
            data2d, "parallel_fast_ion_pressure", coords2d, make_copy=True
        )
        self.perpendicular_fast_ion_pressure = format_dataarray(
            data2d, "perpendicular_fast_ion_pressure", coords2d, make_copy=True
        )

        # Private variables for class property variables
        self._fast_ion_pressure = format_dataarray(
            data2d, "fast_ion_pressure", coords2d, make_copy=True
        )
        self._electron_pressure = format_dataarray(
            data2d, "electron_pressure", coords2d, make_copy=True
        )
        self._thermal_pressure = format_dataarray(
            data2d, "thermal_pressure", coords2d, make_copy=True
        )
        self._pressure = format_dataarray(data2d, "pressure", coords2d, make_copy=True)
        self._wth = format_dataarray(
            data1d_time, "thermal_stored_energy", coords1d_time, make_copy=True
        )
        self._wfast = format_dataarray(
            data1d_time, "fast_ion_stored_energy", coords1d_time, make_copy=True
        )
        self._wp = format_dataarray(
            data1d_time, "stored_energy", coords1d_time, make_copy=True
        )
        self._zeff = format_dataarray(
            data3d, "effective_charge", coords3d, make_copy=True
        )
        self._ion_density = format_dataarray(
            data3d, "ion_density", coords3d, make_copy=True
        )
        self._ion_concentration = format_dataarray(
            data3d, "concentration", coords3d, make_copy=True
        )
        self._meanz = format_dataarray(data3d, "mean_charge", coords3d, make_copy=True)
        self._total_radiation = format_dataarray(
            data3d, "total_radiation", coords3d, make_copy=True
        )
        self._prad_tot = format_dataarray(
            data2d_elem, "total_radiated_power", coords2d_elem, make_copy=True
        )

        _fz = {}
        _lz_tot = {}
        for elem in self.elements:
            nz = self.element_z.sel(element=elem).values + 1
            ion_charge = format_coord(np.arange(nz), "ion_charge")
            coords3d_fract = {
                "t": self.t,
                "rhop": self.rhop,
                "ion_charge": ion_charge,
            }
            data3d_fz = np.full((len(self.t), len(self.rhop), nz), 0.0)
            _fz[elem] = format_dataarray(
                data3d_fz, "fractional_abundance", coords3d_fract, make_copy=True
            )
            _lz_tot[elem] = format_dataarray(
                data3d_fz,
                "total_radiation_loss_parameter",
                coords3d_fract,
                make_copy=True,
            )
        self._fz = _fz
        self._lz_tot = _lz_tot

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

        self.Meanz = CachedCalculation(
            self.calc_meanz,
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

        self.Ion_concentration = CachedCalculation(
            self.calc_ion_concentration,
            [
                self.electron_density,
                self.ion_density,
            ],
        )

        self.Zeff = CachedCalculation(
            self.calc_zeff,
            [
                self.electron_density,
                self.electron_temperature,
                self.impurity_density,
                self.fast_ion_density,
                self.neutral_density,
                self.tau,
            ],
        )

        self.Lz_tot = CachedCalculation(
            self.calc_lz_tot,
            [
                self.electron_density,
                self.electron_temperature,
                self.tau,
                self.neutral_density,
            ],
        )

        self.Total_radiation = CachedCalculation(
            self.calc_total_radiation,
            [
                self.electron_density,
                self.electron_temperature,
                self.impurity_density,
                self.fast_ion_density,
                self.tau,
                self.neutral_density,
            ],
        )

        self.Electron_pressure = CachedCalculation(
            self.calc_electron_pressure,
            [
                self.electron_density,
                self.electron_temperature,
            ],
        )

        self.Thermal_pressure = CachedCalculation(
            self.calc_thermal_pressure,
            [
                self.ion_temperature,
                self.electron_density,
                self.electron_temperature,
                self.impurity_density,
                self.fast_ion_density,
            ],
        )

        self.Wth = CachedCalculation(
            self.calc_wth,
            [
                self.ion_temperature,
                self.electron_density,
                self.electron_temperature,
                self.impurity_density,
                self.fast_ion_density,
                # Should include volume but requires equilibrium
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
        return self.Electron_pressure()

    def calc_electron_pressure(self):
        self._electron_pressure.values = ph.calc_pressure(
            self.electron_density, self.electron_temperature
        )
        return self._electron_pressure

    def calc_thermal_pressure(self):
        self._thermal_pressure.values = (
            ph.calc_pressure(self.ion_density, self.ion_temperature).sum("element")
            + self.electron_pressure
        )
        return self._thermal_pressure

    @property
    def thermal_pressure(self):
        return self.Thermal_pressure()

    @property
    def pressure(self):
        self._pressure.values = self.thermal_pressure + self.fast_ion_pressure
        return self._pressure

    @property
    def fast_ion_pressure(self):
        # TODO: check whether degrees of freedom are correctly included...
        self._fast_ion_pressure.values = (
            self.parallel_fast_ion_pressure / 3
            + self.perpendicular_fast_ion_pressure * 2 / 3
        )
        return self._fast_ion_pressure

    def calc_wth(self):
        for t in np.array(self.time_to_calculate, ndmin=1):
            self._wth.loc[dict(t=t)] = (
                3 / 2 * np.trapz(self.thermal_pressure.sel(t=t), self.volume.sel(t=t))
            )
        return self._wth

    @property
    def wth(self):
        return self.Wth()

    @property
    def wfast(self):
        for t in np.array(self.time_to_calculate, ndmin=1):
            self._wfast.loc[dict(t=t)] = (
                3 / 2 * np.trapz(self.fast_ion_pressure.sel(t=t), self.volume.sel(t=t))
            )
        return self._wfast

    @property
    def wp(self):
        return self.wth + self.wfast

    @property
    def fz(self):
        return self.Fz()

    def calc_fz(self):
        for elem in self.elements:
            for t in np.array(self.time_to_calculate, ndmin=1):
                electron_temperature = self.electron_temperature.sel(t=t)
                electron_density = self.electron_density.sel(t=t)
                tau = None
                if np.any(self.tau != 0):
                    tau = self.tau.sel(t=t)
                neutral_density = None
                if np.any(self.neutral_density != 0):
                    neutral_density = self.neutral_density.sel(t=t)
                if any(
                    np.logical_not((electron_temperature > 0) * (electron_density > 0))
                ):
                    continue
                fz_tmp = self.fract_abu[elem](
                    electron_temperature,
                    Ne=electron_density,
                    Nh=neutral_density,
                    tau=tau,
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
    def ion_concentration(self):
        return self.Ion_concentration()

    def calc_ion_concentration(self):
        for elem in self.impurities:
            self._ion_concentration.loc[dict(element=elem)] = (
                self.ion_density.sel(element=elem) / self.electron_density
            )
        return self._ion_concentration

    @property
    def lz_tot(self):
        return self.Lz_tot()

    def calc_lz_tot(self):
        fz = self.fz
        for elem in self.elements:
            for t in np.array(self.time_to_calculate, ndmin=1):
                electron_density = self.electron_density.sel(t=t)
                electron_temperature = self.electron_temperature.sel(t=t)
                if any(
                    np.logical_not((electron_temperature > 0) * (electron_density > 0))
                ):
                    continue
                Fz = fz[elem].sel(t=t).transpose()
                neutral_density = None
                if np.any(self.neutral_density.sel(t=t) != 0):
                    neutral_density = self.neutral_density.sel(t=t)
                self._lz_tot[elem].loc[dict(t=t)] = self.power_loss_tot[elem](
                    electron_temperature,
                    Fz,
                    Ne=electron_density,
                    Nh=neutral_density,
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
        return self.Meanz()

    def calc_meanz(self):
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
            if True, modifies impurity density to get a ~ flat zeff contribution
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

    def set_equilibrium(self, equilibrium: Equilibrium):
        """Assign equilibrium object and associated private variables"""
        self.equilibrium = equilibrium

    def set_adf11(self, adf11: dict):
        self.adf11 = adf11

    def build_atomic_data(self):
        """
        Assigns default atomic fractional abundance and radiated power operators
        """
        fract_abu, power_loss_tot = default_atomic_data(
            self.elements, full_run=self.full_run
        )
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
        Hashing of dependencies for caching

        xr.DataArray, and np.ndarray are the only types currently permitted.
        Dictionaries contained in dependencies were not updating to match the
        plasma attributes and so were removed.

        TODO: upgrade so other objects can be hashed, e.g. Equilibrium
        """
        _dependencies = []
        for dependency in self.dependencies:
            if type(dependency) == dict:
                raise NotImplementedError(
                    "dictionary dependencies are not working correctly"
                )
            elif type(dependency) == xr.DataArray:
                _dependencies.append(dependency.data)
            elif type(dependency) == np.ndarray:
                _dependencies.append(dependency)
            else:
                raise NotImplementedError(
                    f"Hashing only implemented for xr.DataArray and np.ndarray "
                    f"not {type(dependency)}"
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
        profilers: dict[str, ProfilerBase],
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
                "total_radiation",
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
        rhop = _rhop.swap_dims({"index": "R"})

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
