from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import pickle
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.time import convert_in_time_dt
from indica.converters.time import get_tlabels_dt
from indica.equilibrium import Equilibrium
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import default_atomic_data
import indica.physics as ph
from indica.profiles_gauss import Profiles as ProfilesGauss
from indica.utilities import format_coord
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
        self.rho_type = "rho_poloidal"
        if self.rho_type != "rho_poloidal":
            print_like("Only rho_poloidal in input for the time being...")
            raise AssertionError

        self.initialize_variables(n_rad, n_R, n_z)

        self.equilibrium: Equilibrium

    def set_equilibrium(self, equilibrium: Equilibrium):
        """
        Assign equilibrium object
        """
        self.equilibrium = equilibrium
        self._volume.values = self.convert_in_time(
            self.equilibrium.volume.interp(rho_poloidal=self.rho)
        )
        self._area.values = self.convert_in_time(
            self.equilibrium.area.interp(rho_poloidal=self.rho)
        )
        self._rmjo.values = self.convert_in_time(
            self.equilibrium.rmjo.interp(rho_poloidal=self.rho)
        )
        self._rmji.values = self.convert_in_time(
            self.equilibrium.rmji.interp(rho_poloidal=self.rho)
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

        self.optimisation: dict = {}
        self.forward_models: dict = {}
        self.power_loss_sxr: dict = {}
        self.power_loss_tot: dict = {}

        # Machine attributes
        R0, R1 = self.machine_dimensions[0]
        z0, z1 = self.machine_dimensions[1]
        self.R = format_coord(np.linspace(R0, R1, n_R), "R")
        self.z = format_coord(np.linspace(z0, z1, n_z), "z")

        index = np.arange(n_R)
        R_midplane = np.linspace(self.R.min(), self.R.max(), n_R)
        z_midplane = np.full_like(R_midplane, 0.0)
        coords_midplane = [("index", format_coord(index, "index"))]
        self.R_midplane = format_dataarray(R_midplane, "R_midplane", coords_midplane)
        self.z_midplane = format_dataarray(z_midplane, "z_midplane", coords_midplane)

        # Time and radial grid
        self.rho = format_coord(np.linspace(0, 1.0, n_rad), self.rho_type)
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

        coords_elem = [("element", list(self.elements))]
        self.element_z = format_dataarray(element_z, "atomic_number", coords_elem)
        self.element_a = format_dataarray(element_a, "atomic_weight", coords_elem)
        self.element_name = format_dataarray(element_name, "element_name", coords_elem)
        self.element_symbol = format_dataarray(
            element_symbol, "element_symbol", coords_elem
        )

        # Assign data to variables
        nt = len(self.t)
        nr = len(self.rho)
        nel = len(self.elements)
        nimp = len(self.impurities)
        data1d_time = np.zeros(nt)
        data2d = np.zeros((nt, nr))
        data2d_elem = np.zeros((nel, nt))
        data3d = np.zeros((nel, nt, nr))
        data3d_imp = np.zeros((nimp, nt, nr))

        coords1d_time = [("t", self.t)]
        coords2d = [("t", self.t), (self.rho_type, self.rho)]
        coords2d_elem = [("element", list(self.elements)), ("t", self.t)]
        coords3d = [
            ("element", list(self.elements)),
            ("t", self.t),
            (self.rho_type, self.rho),
        ]
        coords3d_imp = [
            ("element", list(self.impurities)),
            ("t", self.t),
            (self.rho_type, self.rho),
        ]

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
            data3d, "toroidal_rotation", coords3d, make_copy=True
        )
        self.impurity_density = format_dataarray(
            data3d_imp, "impurity_density", coords3d_imp, make_copy=True
        )
        self.fast_density = format_dataarray(
            data2d, "fast_ion_density", coords2d, make_copy=True
        )
        self.pressure_fast_parallel = format_dataarray(
            data2d, "parallel_fast_ion_pressure", coords2d, make_copy=True
        )
        self.pressure_fast_perpendicular = format_dataarray(
            data2d, "perpendicular_fast_ion_pressure", coords2d, make_copy=True
        )

        # Private variables for class property variables
        self._rmag: DataArray = format_dataarray(
            data1d_time, "major_radius_magnetic_axis", coords1d_time, make_copy=True
        )
        self._zmag: DataArray = format_dataarray(
            data1d_time, "z_magnetic_axis", coords1d_time, make_copy=True
        )
        self._rmji: DataArray = format_dataarray(
            data2d, "major_radius_hfs", coords2d, make_copy=True
        )
        self._rmjo: DataArray = format_dataarray(
            data2d, "major_radius_lfs", coords2d, make_copy=True
        )
        self._rmin: DataArray = format_dataarray(
            data2d, "minor_radius", coords2d, make_copy=True
        )
        self._volume: DataArray = format_dataarray(
            data2d, "volume", coords2d, make_copy=True
        )
        self._area: DataArray = format_dataarray(
            data2d, "area", coords2d, make_copy=True
        )
        self._pressure_fast = format_dataarray(
            data2d, "total_fast_ion_pressure", coords2d, make_copy=True
        )
        self._pressure_el = format_dataarray(
            data2d, "electron_pressure", coords2d, make_copy=True
        )
        self._pressure_th = format_dataarray(
            data2d, "thermal_pressure", coords2d, make_copy=True
        )
        self._pressure_tot = format_dataarray(
            data2d, "total_pressure", coords2d, make_copy=True
        )
        self._pth = format_dataarray(
            data1d_time, "thermal_pressure_integral", coords1d_time, make_copy=True
        )
        self._ptot = format_dataarray(
            data1d_time, "total_pressure_integral", coords1d_time, make_copy=True
        )
        self._wth = format_dataarray(
            data1d_time, "thermal_stored_energy", coords1d_time, make_copy=True
        )
        self._wp = format_dataarray(
            data1d_time, "total_stored_energy", coords1d_time, make_copy=True
        )
        self._zeff = format_dataarray(
            data3d, "effective_charge", coords3d, make_copy=True
        )
        self._ion_density = format_dataarray(
            data3d, "ion_density", coords3d, make_copy=True
        )
        self._meanz = format_dataarray(data3d, "mean_charge", coords3d, make_copy=True)
        self._total_radiation = format_dataarray(
            data3d, "total_radiated_power_emission", coords3d, make_copy=True
        )
        self._sxr_radiation = format_dataarray(
            data3d, "sxr_radiated_power_emission", coords3d, make_copy=True
        )
        self._prad_tot = format_dataarray(
            data2d_elem, "total_radiated_power", coords2d_elem, make_copy=True
        )
        self._prad_sxr = format_dataarray(
            data2d_elem, "sxr_radiated_power", coords2d_elem, make_copy=True
        )

        _fz = {}
        _lz_tot = {}
        _lz_sxr = {}
        for elem in self.elements:
            nz = self.element_z.sel(element=elem).values + 1
            ion_charge = format_coord(np.arange(nz), "ion_charge")
            coords3d_fract = [
                ("t", self.t),
                ("rho_poloidal", self.rho),
                ("ion_charge", ion_charge),
            ]
            data3d_fz = np.full((len(self.t), len(self.rho), nz), 0.0)
            _fz[elem] = format_dataarray(
                data3d_fz, "fractional_abundance", coords3d_fract, make_copy=True
            )
            _lz_tot[elem] = format_dataarray(
                data3d_fz,
                "total_radiation_loss_parameter",
                coords3d_fract,
                make_copy=True,
            )
            _lz_sxr[elem] = format_dataarray(
                data3d_fz,
                "sxr_radiation_loss_parameter",
                coords3d_fract,
                make_copy=True,
            )
        self._fz = _fz
        self._lz_tot = _lz_tot
        self._lz_sxr = _lz_sxr

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

        main_ion_density = (
            self.electron_density
            - self.fast_density * self.meanz.sel(element=self.main_ion)
            - (self.impurity_density * self.meanz).sum("element")
        )

        self._ion_density.loc[dict(element=self.main_ion)] = main_ion_density
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
        t: LabeledArray = None,
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
                zeff_core = _zeff.where(self.rho < 0.5).mean("rho_poloidal")
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

    def map_to_midplane(self, attrs: List[str]):
        """
        Map profiles from flux space to real space on z=0

        TODO: _HI and _LOW from old implementation should be substituted with
              somethiing  more memorable and sensible, e.g. _ERR?
              check with Michael how this is implemented on his end
        """

        R = self.R_midplane
        z = self.z_midplane

        midplane_profiles: dict = {}
        for attr in attrs:
            if not hasattr(self, attr):
                continue

            midplane_profiles[attr] = []
            _prof = getattr(self, attr)
            _rho = self.equilibrium.rho.interp(t=self.t).interp(R=R, z=z)
            rho = _rho.swap_dims({"index": "R"}).drop_vars("index")

            _prof_midplane = _prof.interp(rho_poloidal=rho)
            midplane_profiles[attr] = xr.where(
                np.isfinite(_prof_midplane), _prof_midplane, 0.0
            )

        return midplane_profiles

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


class PlasmaProfiles:
    def __init__(self, plasma: Plasma, profiler_object: Callable = ProfilesGauss):
        """
        Interface a Profiler class with a Plasma object to generate plasma profiles
        and update them.

        Parameters
        ----------
        plasma
            Plasma object
        profiler
            Object to generate profiles

        TODO: currently testing keys == datatypes e.g. "electron_temperature" instead of
              "Te_prof" so that profiler key = plasma attribute.
        """
        self.plasma = plasma
        impurities = plasma.impurities

        # Ion_temperature and toroidal_rotation currently taken as equal for all ion
        profile_datatypes = [
            "electron_temperature",
            "ion_temperature",
            "electron_density",
            "neutral_density",
            "toroidal_rotation",
        ]
        profilers: dict = {}
        for profile_datatype in profile_datatypes:
            profilers[profile_datatype] = profiler_object(
                datatype=profile_datatype, xspl=plasma.rho
            )
        # Impurities have separate profilers identified by datatype:element, e.g.
        # impurity_density:c or impurity_density:ar
        profile_datatype = "impurity_density"
        for element in impurities:
            profilers[f"{profile_datatype}:{element}"] = profiler_object(
                datatype=profile_datatype, xspl=plasma.rho
            )
        self.profilers = profilers

    def __call__(self, parameters: dict, t: float = None):
        """
        Set parameters of desired profilers and assign to plasma class profiles

        Parameters
        ----------
        parameters
            Flat dictionary of {"datatype.parameter":value} with:
            - datatype in profiler_dict.keys()
            - parameter in profiler.profile_parameters.keys().
            Special case for impurity density:
                {"datatype:element.parameter":value}
        """
        plasma = self.plasma
        profilers = self.profilers
        _profiles_to_update: list = []

        # Set all parameters for all profilers
        for identifier, value in parameters.items():
            profile_datatype, parameter = identifier.split(".")

            if profile_datatype not in profilers.keys():
                raise ValueError(f"No profiler available for {profile_datatype}")
            if not hasattr(profilers[profile_datatype], parameter):
                raise ValueError(
                    f"No parameter {parameter} available for {profile_datatype}"
                )
            setattr(profilers[profile_datatype], parameter, value)
            _profiles_to_update.append(profile_datatype)

        if t is None:
            t = plasma.time_to_calculate

        # Update only desired profiles, distinguish case of impurity_density
        profiles_to_update = np.unique(_profiles_to_update)
        for profile_datatype in profiles_to_update:
            profiler_output = profilers[profile_datatype]()
            if "impurity_density" in profile_datatype:
                datatype, element = profile_datatype.split(":")
                getattr(plasma, datatype).loc[
                    dict(t=t, element=element)
                ] = profiler_output
            else:
                datatype = profile_datatype
                getattr(plasma, datatype).loc[dict(t=t)] = profiler_output


def example_plasma(
    machine: str = "st40",
    pulse: int = None,
    tstart=0.02,
    tend=0.1,
    dt=0.01,
    main_ion="h",
    impurities: Tuple[str, ...] = ("c", "ar", "he"),
    load_from_pkl: bool = True,
    **kwargs,
):

    default_plasma_file = (
        f"{Path(__file__).parent.parent}/data/{machine}_default_plasma_phantom.pkl"
    )

    if load_from_pkl and pulse is not None:
        try:
            print(f"\n Loading phantom plasma class from {default_plasma_file}. \n")
            return pickle.load(open(default_plasma_file, "rb"))
        except FileNotFoundError:
            print(
                f"\n\n No phantom plasma class file {default_plasma_file}. \n"
                f" Building it and saving to file. \n\n"
            )

    # TODO: swap all profiles to new version!

    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=main_ion,
        impurities=impurities,
        **kwargs,
    )
    plasma.build_atomic_data()

    update_profiles = PlasmaProfiles(plasma)

    # Assign profiles to time-points
    nt = len(plasma.t)
    ne_peaking = np.linspace(1, 2, nt)
    te_peaking = np.linspace(1, 2, nt)
    _y0 = update_profiles.profilers["toroidal_rotation"].y0
    vrot0 = np.linspace(
        _y0 * 1.1,
        _y0 * 2.5,
        nt,
    )
    vrot_peaking = np.linspace(1, 2, nt)

    _y0 = update_profiles.profilers["ion_temperature"].y0
    ti0 = np.linspace(_y0 * 1.1, _y0 * 2.5, nt)

    _y0 = update_profiles.profilers[f"impurity_density:{impurities[0]}"].y0
    nimp_y0 = _y0 * 5 * np.linspace(1, 8, nt)
    nimp_peaking = np.linspace(1, 5, nt)
    nimp_wcenter = np.linspace(0.4, 0.1, nt)
    for i, t in enumerate(plasma.t):
        parameters = {
            "electron_temperature.peaking": te_peaking[i],
            "ion_temperature.peaking": te_peaking[i],
            "ion_temperature.y0": ti0[i],
            "toroidal_rotation.peaking": vrot_peaking[i],
            "toroidal_rotation.y0": vrot0[i],
            "electron_density.peaking": ne_peaking[i],
            "impurity_density:ar.peaking": nimp_peaking[i],
            "impurity_density:ar.y0": nimp_y0[i],
            "impurity_density:ar.wcenter": nimp_wcenter[i],
        }
        update_profiles(parameters, t=t)

    if load_from_pkl and pulse is not None:
        print(f"\n Saving phantom plasma class in {default_plasma_file} \n")
        pickle.dump(plasma, open(default_plasma_file, "wb"))

    return plasma


if __name__ == "__main__":
    example_plasma()
