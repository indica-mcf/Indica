from copy import deepcopy
from functools import lru_cache
import hashlib
import pickle
from typing import Callable

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.time import get_tlabels_dt
from indica.equilibrium import Equilibrium
from indica.equilibrium import fake_equilibrium_data
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
import indica.physics as ph
from indica.profiles_gauss import Profiles
from indica.readers import ADASReader
from indica.readers.adas import ADF11
from indica.utilities import format_coord
from indica.utilities import format_dataarray
from indica.utilities import get_element_info
from indica.utilities import print_like

plt.ion()

# TODO: add elongation and triangularity in all equations


class Plasma:
    def __init__(
        self,
        tstart: float = 0.01,
        tend: float = 0.14,
        dt: float = 0.01,
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
        impurities: tuple = ("c", "ar"),
        main_ion: str = "h",
        impurity_concentration: tuple = (0.02, 0.001),
        pulse: int = None,
        full_run: bool = False,
        n_rad: int = 41,
        verbose: bool = False,
    ):
        """
        Class for plasma objects.
        - completely independent of experimental data
        - can be assigned an equilibrium object for remapping (currently 1D)
        - independent parameters can be set, dependent ones are properties

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
        impurity_concentration
            Default concentration of each impurity element
        pulse
            Pulse number, if this is associated with an experiment
        full_run
            If True: compute ionisation balance at every iteration
            If False: calculate default and interpolate
        """

        self.pulse = pulse
        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.full_run = full_run
        self.verbose = verbose
        self.ADASReader = ADASReader()
        self.main_ion = main_ion
        self.impurities = impurities
        self.elements = [self.main_ion]
        for elem in self.impurities:
            self.elements.append(elem)
        self.set_adf11(ADF11)
        self.machine_dimensions = machine_dimensions
        self.n_rad = n_rad
        self.rho_type = "rho_poloidal"
        if self.rho_type != "rho_poloidal":
            print_like("Only rho_poloidal in input for the time being...")
            raise AssertionError

        self.initialize_variables(impurity_concentration=impurity_concentration)

        self.equilibrium: Equilibrium

    def set_equilibrium(self, equilibrium: Equilibrium):
        """
        Assign equilibrium object
        """
        self.equilibrium = equilibrium

    def set_adf11(self, adf11: dict):
        self.adf11 = adf11

    def initialize_variables(self, impurity_concentration: tuple):
        """
        Initialize all class attributes

        Description of variables being initialized
        ------------------------------------------
        time_to_calculate
            subset of time-point(s) to use for computation of the dependent variables
            (to be used e.g. in optimisation workflows)
        """

        self.optimisation: dict = {}
        self.forward_models: dict = {}
        self.power_loss_sxr: dict = {}
        self.power_loss_tot: dict = {}

        # Assign plasma and machine attributes
        self.machine_R = np.linspace(
            self.machine_dimensions[0][0], self.machine_dimensions[0][1], 100
        )
        self.machine_z = np.linspace(
            self.machine_dimensions[1][0], self.machine_dimensions[1][1], 100
        )
        R_midplane = np.linspace(self.machine_R.min(), self.machine_R.max(), 100)
        self.R_midplane = R_midplane
        z_midplane = np.full_like(R_midplane, 0.0)
        self.z_midplane = z_midplane

        _time = get_tlabels_dt(self.tstart, self.tend, self.dt)
        _radius = np.linspace(0, 1.0, self.n_rad)
        _elem = list(self.elements)
        _imp = list(self.impurities)

        self.rho = format_coord(_radius, self.rho_type)
        self.element = format_coord(_elem, "element")
        self.impurity = format_coord(_imp, "element")
        z_elem, a_elem, name_elem, symbol_elem = [], [], [], []
        for elem in self.element.values:
            _z, _a, _name, _symbol = get_element_info(elem)
            z_elem.append(_z)
            a_elem.append(_a)
            name_elem.append(_name)
            symbol_elem.append(_symbol)
        self.element.assign_coords(Z=("element", z_elem))
        self.element.assign_coords(A=("element", a_elem))
        self.element.assign_coords(name=("element", name_elem))
        self.element.assign_coords(symbol=("element", symbol_elem))
        self.t = format_coord(_time, "t")
        self.time_to_calculate = deepcopy(self.t)

        # Assign data to variables
        nt = len(_time)
        nr = len(_radius)
        nel = len(_elem)
        nimp = len(_imp)
        data1d_time = np.zeros(nt)
        data2d = np.zeros((nt, nr))
        data2d_elem = np.zeros((nel, nt))
        data3d = np.zeros((nel, nt, nr))
        data3d_imp = np.zeros((nimp, nt, nr))

        coords1d_time = [("t", self.t)]
        coords1d_impurity = [("element", self.impurity)]
        coords2d = [("t", self.t), (self.rho_type, self.rho)]
        coords2d_elem = [("element", self.element), ("t", self.t)]
        coords3d = [("element", self.element), ("t", self.t), (self.rho_type, self.rho)]
        coords3d_imp = [
            ("element", self.impurity),
            ("t", self.t),
            (self.rho_type, self.rho),
        ]

        # TODO: should be deleted with a method set_concentration implemented
        # to set densities from inputted concentration & element
        _conc = np.array(impurity_concentration)
        self.impurity_concentration = format_dataarray(
            _conc, "element", coords1d_impurity
        )

        # Profilers
        # TODO: Move out of this class
        # TODO: Profiles still to be converted to use of new DATATYPES and UNITS
        self.Te_prof = Profiles(datatype=("temperature", "electron"), xspl=self.rho)
        self.Ti_prof = Profiles(datatype=("temperature", "ion"), xspl=self.rho)
        self.Ne_prof = Profiles(datatype=("density", "electron"), xspl=self.rho)
        self.Nimp_prof = Profiles(datatype=("density", "impurity"), xspl=self.rho)
        self.Niz1_prof = Profiles(datatype=("density", "impurity"), xspl=self.rho)
        self.Niz2_prof = Profiles(datatype=("density", "impurity"), xspl=self.rho)
        self.Nh_prof = Profiles(datatype=("density", "thermal_neutral"), xspl=self.rho)
        self.Vrot_prof = Profiles(datatype=("rotation", "toroidal"), xspl=self.rho)

        # Independent plasma quantities
        self.electron_temperature = format_dataarray(
            data2d, "electron_temperature", coords2d
        )
        self.electron_density = format_dataarray(data2d, "electron_density", coords2d)
        self.neutral_density = format_dataarray(
            data2d, "thermal_neutral_density", coords2d
        )
        self.tau = format_dataarray(data2d, "residence_time", coords2d)
        self.ion_temperature = format_dataarray(data3d, "ion_temperature", coords3d)
        self.toroidal_rotation = format_dataarray(data3d, "toroidal_rotation", coords3d)
        self.impurity_density = format_dataarray(
            data3d_imp, "impurity_density", coords3d_imp
        )
        self.fast_density = format_dataarray(data2d, "fast_ion_density", coords2d)
        self.pressure_fast_parallel = format_dataarray(
            data2d, "parallel_fast_ion_pressure", coords2d
        )
        self.pressure_fast_perpendicular = format_dataarray(
            data2d, "perpendicular_fast_ion_pressure", coords2d
        )

        # Private variables for class property variables
        self._rmag = format_dataarray(
            data1d_time, "major_radius_magnetic_axis", coords1d_time
        )
        self._zmag = format_dataarray(data1d_time, "z_magnetic_axis", coords1d_time)
        self._rmji = format_dataarray(data2d, "major_radius_hfs", coords2d)
        self._rmjo = format_dataarray(data2d, "major_radius_lfs", coords2d)
        self._rmin = format_dataarray(data2d, "minor_radius", coords2d)
        self._volume = format_dataarray(data2d, "volume", coords2d)
        self._area = format_dataarray(data2d, "area", coords2d)
        self._pressure_fast = format_dataarray(
            data2d, "total_fast_ion_pressure", coords2d
        )
        self._pressure_el = format_dataarray(data2d, "electron_pressure", coords2d)
        self._pressure_th = format_dataarray(data2d, "thermal_pressure", coords2d)
        self._pressure_tot = format_dataarray(data2d, "total_pressure", coords2d)
        self._pth = format_dataarray(
            data1d_time, "thermal_pressure_integral", coords1d_time
        )
        self._ptot = format_dataarray(
            data1d_time, "total_pressure_integral", coords1d_time
        )
        self._wth = format_dataarray(
            data1d_time, "thermal_stored_energy", coords1d_time
        )
        self._wp = format_dataarray(data1d_time, "total_stored_energy", coords1d_time)
        self._zeff = format_dataarray(data3d, "effective_charge", coords3d)
        self._ion_density = format_dataarray(data3d, "ion_density", coords3d)
        self._meanz = format_dataarray(data3d, "mean_charge", coords3d)
        self._total_radiation = format_dataarray(
            data3d, "total_radiated_power_emission", coords3d
        )
        self._sxr_radiation = format_dataarray(
            data3d, "sxr_radiated_power_emission", coords3d
        )
        self._prad_tot = format_dataarray(
            data2d_elem, "total_radiated_power", coords2d_elem
        )
        self._prad_sxr = format_dataarray(
            data2d_elem, "sxr_radiated_power", coords2d_elem
        )

        _fz = {}
        _lz_tot = {}
        _lz_sxr = {}
        for elem in self.elements:
            _z, _, _, _ = get_element_info(elem)
            nz = _z + 1
            coords3d_fract = [
                ("t", self.t),
                ("rho_poloidal", self.rho),
                ("ion_charges", np.arange(nz)),
            ]
            data3d_fz = np.full((len(self.t), len(self.rho), nz), 0.0)
            _fz[elem] = format_dataarray(
                data3d_fz, "fractional_abundance", coords3d_fract
            )
            _lz_tot[elem] = format_dataarray(
                data3d_fz, "total_radiation_loss_parameter", coords3d_fract
            )
            _lz_sxr[elem] = format_dataarray(
                data3d_fz, "sxr_radiation_loss_parameter", coords3d_fract
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

    # TODO: should be implemented elsewhere!
    def assign_profiles(
        self, profile: str = "electron_density", t: float = None, element: str = None
    ):
        # TODO: impurities and elements should be both either tuples or lists...
        elements: list = []
        impurities: tuple = ()
        if element is None:
            elements = self.elements
            impurities = self.impurities
        else:
            elements = [element]
            if element in self.impurities:
                impurities = (element,)
        if profile == "electron_density":
            self.electron_density.loc[dict(t=t)] = self.Ne_prof()
        elif profile == "electron_temperature":
            self.electron_temperature.loc[dict(t=t)] = self.Te_prof()
        elif profile == "ion_temperature":
            for elem in elements:
                self.ion_temperature.loc[dict(t=t, element=elem)] = self.Ti_prof()
        elif profile == "toroidal_rotation":
            for elem in elements:
                self.toroidal_rotation.loc[dict(t=t, element=elem)] = self.Vrot_prof()
        elif profile == "impurity_density":
            for imp in impurities:
                self.impurity_density.loc[dict(t=t, element=imp)] = self.Nimp_prof()
        elif profile == "neutral_density":
            self.neutral_density.loc[dict(t=t)] = self.Nh_prof()
        else:
            raise ValueError(
                f"{profile} currently not found in possible Plasma properties"
            )

    # TODO: should be implemented elsewhere!
    def update_profiles(
        self,
        parameters: dict,
    ):
        """
        Update plasma profiles with profile parameters i.e.
        {"Ne_prof.y0":1e19} -> Ne_prof.y0
        TODO: refactor profiles into profiler structure
            and take care of initialisation of impurity profiles
        """
        profile_prefixes: list = [
            "Te_prof",
            "Ti_prof",
            "Ne_prof",
            "Niz1_prof",
            "Niz2_prof",
            "Nh_prof",
            "Vrot_prof",
        ]
        for param, value in parameters.items():
            _prefix = [pref for pref in profile_prefixes if pref in param]
            if _prefix:
                prefix: str = _prefix[0]
                key = param.replace(prefix + ".", "")
                profile = getattr(self, prefix)
                if hasattr(profile, key):
                    setattr(profile, key, value)
                else:
                    raise ValueError(f"parameter: {key} not found in {prefix}")

        # Only update profiles which are given in parameters
        parameter_prefixes = [key.split(".")[0] for key in parameters.keys()]
        profile_names = set(parameter_prefixes) & set(profile_prefixes)

        if "Te_prof" in profile_names:
            self.electron_temperature.loc[
                dict(t=self.time_to_calculate)
            ] = self.Te_prof()
        if "Ti_prof" in profile_names:
            self.ion_temperature.loc[dict(t=self.time_to_calculate)] = self.Ti_prof()
        if "Ne_prof" in profile_names:
            self.electron_density.loc[dict(t=self.time_to_calculate)] = self.Ne_prof()
        if "Nh_prof" in profile_names:
            self.neutral_density.loc[dict(t=self.time_to_calculate)] = self.Nh_prof()
        if "Vrot_prof" in profile_names:
            self.electron_temperature.loc[
                dict(t=self.time_to_calculate)
            ] = self.Ne_prof()
        if "Niz1_prof" in profile_names:
            self.impurity_density.loc[
                dict(t=self.time_to_calculate, element=self.impurities[0])
            ] = self.Niz1_prof()
        else:
            self.impurity_density.loc[
                dict(t=self.time_to_calculate, element=self.impurities[0])
            ] = (self.Ne_prof() * self.impurity_concentration[0])

        if "Niz2_prof" in profile_names:
            self.impurity_density.loc[
                dict(t=self.time_to_calculate, element=self.impurities[1])
            ] = self.Niz2_prof()
        else:
            self.impurity_density.loc[
                dict(t=self.time_to_calculate, element=self.impurities[1])
            ] = (self.Ne_prof() * self.impurity_concentration[1])

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
        self._pressure_th = (
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
        pressure_th = self.pressure_th
        for t in np.array(self.time_to_calculate, ndmin=1):
            self._pth.loc[dict(t=t)] = np.trapz(
                pressure_th.sel(t=t), self.volume.sel(t=t)
            )
        return self._pth

    @property
    def ptot(self):
        return self.Ptot()

    def calc_ptot(self):
        pressure_tot = self.pressure_tot
        for t in np.array(self.time_to_calculate, ndmin=1):
            self._ptot.loc[dict(t=t)] = np.trapz(
                pressure_tot.sel(t=t), self.volume.sel(t=t)
            )
        return self._ptot

    @property
    def wth(self):
        pth = self.pth
        self._wth.values = 3 / 2 * pth.values
        return self._wth

    @property
    def wp(self):
        ptot = self.ptot
        self._wp.values = 3 / 2 * ptot.values
        return self._wp

    @property
    def fz(self):
        return self.Fz()
        # return self.calc_fz()  # self.Fz()

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
                self._fz[elem].loc[dict(t=t)] = fz_tmp.transpose().values
        return self._fz

    @property
    def zeff(self):
        return self.Zeff()
        # return self.calc_zeff()

    def calc_zeff(self):
        self._zeff = self.ion_density * self.meanz**2 / self.electron_density
        return self._zeff

    @property
    def ion_density(self):
        return self.Ion_density()
        # return self.calc_ion_density()

    def calc_ion_density(self):
        meanz = self.meanz
        for elem in self.impurities:
            self._ion_density.loc[dict(element=elem)] = self.impurity_density.sel(
                element=elem
            ).values

        main_ion_density = (
            self.electron_density
            - self.fast_density * meanz.sel(element=self.main_ion)
            - (self.impurity_density * meanz).sum("element")
        )

        self._ion_density.loc[dict(element=self.main_ion)] = main_ion_density
        return self._ion_density

    @property
    def lz_tot(self):
        return self.calc_lz_tot()  # self.Lz_tot()

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
                self._lz_tot[elem].loc[dict(t=t)] = (
                    self.power_loss_tot[elem](
                        Te, Fz, Ne=Ne, Nh=Nh, full_run=self.full_run
                    )
                    .transpose()
                    .values
                )
        return self._lz_tot

    @property
    def lz_sxr(self):
        return self.calc_lz_sxr()  # self.Lz_sxr()

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
                self._lz_sxr[elem].loc[dict(t=t)] = (
                    self.power_loss_sxr[elem](
                        Te, Fz, Ne=Ne, Nh=Nh, full_run=self.full_run
                    )
                    .transpose()
                    .values
                )
        return self._lz_sxr

    @property
    def total_radiation(self):
        return self.calc_total_radiation()  # self.Total_radiation()

    def calc_total_radiation(self):
        lz_tot = self.lz_tot
        ion_density = self.ion_density
        for elem in self.elements:
            total_radiation = (
                lz_tot[elem].sum("ion_charges")
                * self.electron_density
                * ion_density.sel(element=elem)
            )
            self._total_radiation.loc[dict(element=elem)] = xr.where(
                total_radiation >= 0,
                total_radiation,
                0.0,
            ).values
        return self._total_radiation

    @property
    def sxr_radiation(self):
        return self.calc_sxr_radiation()  # self.Sxr_radiation()

    def calc_sxr_radiation(self):
        if not hasattr(self, "power_loss_sxr"):
            return None

        lz_sxr = self.lz_sxr
        ion_density = self.ion_density
        for elem in self.elements:
            sxr_radiation = (
                lz_sxr[elem].sum("ion_charges")
                * self.electron_density
                * ion_density.sel(element=elem)
            )
            self._sxr_radiation.loc[dict(element=elem)] = xr.where(
                sxr_radiation >= 0,
                sxr_radiation,
                0.0,
            ).values
        return self._sxr_radiation

    @property
    def meanz(self):
        fz = self.fz

        for elem in self.elements:
            self._meanz.loc[dict(element=elem)] = (
                (fz[elem] * fz[elem].ion_charges).sum("ion_charges").values
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
        if len(np.where(self._volume > 0)[0]) == 0 and hasattr(self, "equilibrium"):
            self._volume = self.convert_in_time(
                self.equilibrium.volume.interp(rho_poloidal=self.rho)
            )
        return self._volume

    @property
    def area(self):
        if len(np.where(self._area > 0)[0]) == 0 and hasattr(self, "equilibrium"):
            self._area = self.convert_in_time(
                self.equilibrium.area.interp(rho_poloidal=self.rho)
            )
        return self._area

    @property
    def rmjo(self):
        if len(np.where(self._rmjo > 0)[0]) == 0 and hasattr(self, "equilibrium"):
            self._rmjo = self.convert_in_time(
                self.equilibrium.rmjo.interp(rho_poloidal=self.rho)
            )
        return self._rmjo

    @property
    def rmji(self):
        if len(np.where(self._rmji > 0)[0]) == 0 and hasattr(self, "equilibrium"):
            self._rmji = self.convert_in_time(
                self.equilibrium.rmji.interp(rho_poloidal=self.rho)
            )
        return self._rmji

    @property
    def rmag(self):
        if len(np.where(self._rmag > 0)[0]) == 0 and hasattr(self, "equilibrium"):
            self._rmag = self.convert_in_time(self.equilibrium.rmag)
        return self._rmag

    @property
    def zmag(self):
        if len(np.where(self._zmag > 0)[0]) == 0 and hasattr(self, "equilibrium"):
            self._zmag = self.convert_in_time(self.equilibrium.zmag)
        return self._zmag

    @property
    def rmin(self):
        if len(np.where(self._rmin > 0)[0]) == 0:
            self._rmin = (self.rmjo - self.rmji) / 2.0
        return self._zmag

    # TODO: implementation not checked
    # @property
    # def vloop(self):
    #     zeff = self.zeff
    #     j_phi = self.j_phi
    #     self.conductivity = ph.conductivity_neo(
    #         self.electron_density,
    #         self.electron_temperature,
    #         zeff.sum("element"),
    #         self.minor_radius,
    #         self.minor_radius.interp(rho_poloidal=1.0),
    #         self.R_mag,
    #         self.q_prof,
    #         approx="sauter",
    #     )
    #     for t in np.array(self.time_to_calculate, ndmin=1):
    #         resistivity = 1.0 / self.conductivity.sel(t=t)
    #         ir = np.where(np.isfinite(resistivity))
    #         vloop = ph.vloop(
    #             resistivity[ir], j_phi.sel(t=t)[ir], self.area.sel(t=t)[ir]
    #         )
    #         self._vloop.loc[dict(t=t)] = vloop.values
    #     return self._vloop

    # TODO: currently not used --> check that it's not required or implement elsewhere!
    # def calc_impurity_density(self, elements):
    #     """
    #     Calculate impurity density from concentration and electron density
    #     """
    #     for elem in elements:
    #         conc = self.impurity_concentration.sel(element=elem)
    #         Nimp = self.electron_density * conc
    #         self.impurity_density.loc[
    #             dict(
    #                 element=elem,
    #             )
    #         ] = Nimp.values

    # TODO: currently not used --> check that it's not required or implement elsewhere!
    # def impose_flat_zeff(self):
    #     """
    #     Adapt impurity concentration to generate flat Zeff contribution
    #     """

    #     for elem in self.impurities:
    #         if np.count_nonzero(self.ion_density.sel(element=elem)) != 0:
    #             zeff_tmp = (
    #                 self.ion_density.sel(element=elem)
    #                 * self.meanz.sel(element=elem) ** 2
    #                 / self.electron_density
    #             )
    #             rho = zeff_tmp.rho_poloidal
    #             value = zeff_tmp.where(rho < 0.2).mean("rho_poloidal")
    #             zeff_tmp = zeff_tmp / zeff_tmp * value
    #             ion_density_tmp = zeff_tmp / (
    #                 self.meanz.sel(element=elem) ** 2 / self.electron_density
    #             )
    #             self.ion_density.loc[dict(element=elem)] = ion_density_tmp.values

    # def convert_in_time(self, value: DataArray, method="linear"):
    #     binned = convert_in_time_dt(self.tstart, self.tend, self.dt, value).interp(
    #         t=self.t, method=method
    #     )

    #     return binned

    # TODO: should be moved outside of here
    def build_atomic_data(
        self,
        Te: DataArray = None,
        Ne: DataArray = None,
        Nh: DataArray = None,
        tau: DataArray = None,
        default=True,
        calc_power_loss=True,
    ):
        if default:
            xend = 1.02
            rho_end = 1.01
            rho = np.abs(np.linspace(rho_end, 0, 100) ** 1.8 - rho_end - 0.01)
            Te_prof = Profiles(
                datatype=("temperature", "electron"),
                xspl=rho,
                xend=xend,
            )
            Te_prof.y0 = 10.0e3
            Te = Te_prof()
            Ne_prof = Profiles(datatype=("density", "electron"), xspl=rho, xend=xend)
            Ne = Ne_prof()
            Nh_prof = Profiles(
                datatype=("density", "thermal_neutral"),
                xspl=rho,
                xend=xend,
            )
            Nh = Nh_prof()
            tau = None
        else:
            if Te is None or Ne is None:
                raise ValueError("Input Te and Ne if default == False")

        # print_like("Initialize fractional abundance and power loss objects")
        fract_abu, power_loss_tot, power_loss_sxr = {}, {}, {}
        for elem in self.elements:
            scd = self.ADASReader.get_adf11("scd", elem, self.adf11[elem]["scd"])
            acd = self.ADASReader.get_adf11("acd", elem, self.adf11[elem]["acd"])
            ccd = self.ADASReader.get_adf11("ccd", elem, self.adf11[elem]["ccd"])
            fract_abu[elem] = FractionalAbundance(scd, acd, CCD=ccd)
            fract_abu[elem](Ne=Ne, Te=Te, Nh=Nh, tau=tau, full_run=self.full_run)

            plt = self.ADASReader.get_adf11("plt", elem, self.adf11[elem]["plt"])
            prb = self.ADASReader.get_adf11("prb", elem, self.adf11[elem]["prb"])
            prc = self.ADASReader.get_adf11("prc", elem, self.adf11[elem]["prc"])
            power_loss_tot[elem] = PowerLoss(plt, prb, PRC=prc)
            try:
                pls = self.ADASReader.get_adf11("pls", elem, self.adf11[elem]["pls"])
                prs = self.ADASReader.get_adf11("prs", elem, self.adf11[elem]["prs"])
                power_loss_sxr[elem] = PowerLoss(pls, prs)
            except Exception:
                print("No SXR-filtered data available")

            if calc_power_loss:
                F_z_t = fract_abu[elem].F_z_t
                power_loss_tot[elem](Te, F_z_t, Ne=Ne, Nh=Nh, full_run=self.full_run)
                if elem in power_loss_sxr.keys():
                    power_loss_sxr[elem](Te, F_z_t, Ne=Ne, full_run=self.full_run)

        self.adf11 = self.adf11
        self.fract_abu = fract_abu
        self.power_loss_tot = power_loss_tot
        self.power_loss_sxr = power_loss_sxr

    # TODO: check that it's correctly implemented outside before deleting
    # def set_neutral_density(self, y0=1.0e10, y1=1.0e15, decay=12):
    #     self.Nh_prof.y0 = y0
    #     self.Nh_prof.y1 = y1
    #     self.Nh_prof.yend = y1
    #     self.Nh_prof.wped = decay
    #     self.Nh_prof()
    #     for t in np.array(self.time_to_calculate, ndmin=1):
    #         self.neutral_density.loc[dict(t=t)] = self.Nh_prof()

    def map_to_midplane(self):
        # TODO: streamline to avoid re-calculating quantities e.g. ion_density..
        # TODO: this should be moved outside of the class
        keys = [
            "electron_density",
            "ion_density",
            "neutral_density",
            "electron_temperature",
            "ion_temperature",
            "pressure_th",
            "toroidal_rotation",
            "zeff",
            "meanz",
            "volume",
        ]

        nchan = len(self.R_midplane)
        chan = np.arange(nchan)
        R = DataArray(self.R_midplane, coords=[("channel", chan)])
        z = DataArray(self.z_midplane, coords=[("channel", chan)])

        midplane_profiles = {}
        for k in keys:
            k_hi = f"{k}_hi"
            k_lo = f"{k}_lo"

            midplane_profiles[k] = []
            if hasattr(self, k_hi):
                midplane_profiles[k_hi] = []
            if hasattr(self, k_lo):
                midplane_profiles[k_lo] = []

        for k in midplane_profiles.keys():
            prof_rho = getattr(self, k)
            for t in np.array(self.time_to_calculate, ndmin=1):
                rho = (
                    self.equilibrium.rho.sel(t=t, method="nearest")
                    .interp(R=R, z=z)
                    .drop_vars(["R", "z"])
                )
                midplane_profiles[k].append(
                    prof_rho.sel(t=t, method="nearest")
                    .interp(rho_poloidal=rho)
                    .drop_vars("rho_poloidal")
                )
            midplane_profiles[k] = xr.concat(midplane_profiles[k], "t").assign_coords(
                t=self.t
            )
            midplane_profiles[k] = xr.where(
                np.isfinite(midplane_profiles[k]), midplane_profiles[k], 0.0
            )

        self.midplane_profiles = midplane_profiles

    # TODO: there is now a separate class for this!!!
    # def calc_centrifugal_asymmetry(
    #     self, time=None, test_toroidal_rotation=None, plot=False
    # ):
    #     """
    #     Calculate (R, z) maps of the ion densities caused by centrifugal asymmetry
    #     """
    #     if time is None:
    #         time = self.t

    #     # TODO: make this attribute creation a property and standardize?
    #     if not hasattr(self, "ion_density_2d"):
    #         self.rho_2d = self.equilibrium.rho.interp(t=self.t, method="nearest")
    #         tmp = deepcopy(self.rho_2d)
    #         ion_density_2d = []
    #         for elem in self.elements:
    #             ion_density_2d.append(tmp)

    #         self.ion_density_2d = xr.concat(ion_density_2d, "element").assign_coords(
    #             element=self.elements
    #         )
    #         format_dataarray(self.ion_density_2d, "ion_density")
    #         self.centrifugal_asymmetry = deepcopy(self.ion_density)
    #         format_dataarray(self.centrifugal_asymmetry, "centrifugal_asymmetry")
    #         self.asymmetry_multiplier = deepcopy(self.ion_density_2d)
    #         format_dataarray(
    #             self.asymmetry_multiplier, "centrifugal_asymmetry_multiplier"
    #         )

    #     # If toroidal rotation != 0 calculate ion density on 2D poloidal plane
    #     if test_toroidal_rotation is not None:
    #         toroidal_rotation = deepcopy(self.ion_temperature)
    #         format_dataarray(toroidal_rotation, "toroidal_rotation")
    #         toroidal_rotation /= toroidal_rotation.max("rho_poloidal")
    #         toroidal_rotation *= test_toroidal_rotation  # rad/s
    #         self.toroidal_rotation = toroidal_rotation

    #     if not np.any(self.toroidal_rotation != 0):
    #         return

    #     ion_density = self.ion_density
    #     meanz = self.meanz
    #     zeff = self.zeff.sum("element")
    #     rho = self.rho_2d
    #     R_0 = self.maj_r_lfs.interp(rho_poloidal=rho).drop_vars("rho_poloidal")
    #     for elem in self.elements:
    #         main_ion_mass = get_element_info(self.main_ion)[1]
    #         mass = get_element_info(elem)[1]
    #         asymm = ph.centrifugal_asymmetry(
    #             self.ion_temperature.sel(element=elem).drop_vars("element"),
    #             self.electron_temperature,
    #             mass,
    #             meanz.sel(element=elem).drop_vars("element"),
    #             zeff,
    #             main_ion_mass,
    #             toroidal_rotation=self.toroidal_rotation.sel(element=elem).drop_vars(
    #                 "element"
    #             ),
    #         )
    #         self.centrifugal_asymmetry.loc[dict(element=elem)] = asymm
    #         asymmetry_factor = asymm.interp(rho_poloidal=self.rho_2d)
    #         self.asymmetry_multiplier.loc[dict(element=elem)] = np.exp(
    #             asymmetry_factor * (self.rho_2d.R**2 - R_0**2)
    #         )

    #     self.ion_density_2d = (
    #         ion_density.interp(rho_poloidal=self.rho_2d).drop_vars("rho_poloidal")
    #         * self.asymmetry_multiplier
    #     )
    #     format_dataarray(self.ion_density_2d, "ion_density")

    #     if plot:
    #         t = self.t[6]
    #         for elem in self.elements:
    #             plt.figure()
    #             z = self.z_mag.sel(t=t)
    #             rho = self.rho_2d.sel(t=t).sel(z=z, method="nearest")
    #             plt.plot(
    #                 rho,
    #                 self.ion_density_2d.sel(element=elem).sel(
    #                     t=t, z=z, method="nearest"
    #                 ),
    #             )
    #             self.ion_density.sel(element=elem).sel(t=t).plot(linestyle="dashed")
    #             plt.title(elem)

    #         elem = "ar"
    #         plt.figure()
    #         np.log(
    #             self.ion_density_2d.sel(element=elem).sel(t=t, method="nearest")
    #         ).plot()
    #         self.rho_2d.sel(t=t, method="nearest").plot.contour(
    #             levels=10, colors="white"
    #         )
    #         plt.xlabel("R (m)")
    #         plt.ylabel("z (m)")
    #         plt.title(f"log({elem} density")
    #         plt.axis("scaled")
    #         plt.xlim(0, 0.8)
    #         plt.ylim(-0.6, 0.6)

    # TODO: as above, should be moved to separate class
    # def calc_rad_power_2d(self):
    #     """
    #     Calculate total and SXR filtered radiated power on a 2D poloidal plane
    #     including effects from poloidal asymmetries
    #     """
    #     for elem in self.elements:
    #         total_radiation = (
    #             self.lz_tot[elem].sum("ion_charges")
    #             * self.electron_density
    #             * self.ion_density.sel(element=elem)
    #         )
    #         total_radiation = xr.where(
    #             total_radiation >= 0,
    #             total_radiation,
    #             0.0,
    #         )
    #         self.total_radiation.loc[dict(element=elem)] = total_radiation.values

    #         sxr_radiation = (
    #             self.lz_sxr[elem].sum("ion_charges")
    #             * self.electron_density
    #             * self.ion_density.sel(element=elem)
    #         )
    #         sxr_radiation = xr.where(
    #             sxr_radiation >= 0,
    #             sxr_radiation,
    #             0.0,
    #         )
    #         self.sxr_radiation.loc[dict(element=elem)] = sxr_radiation.values

    #         if not hasattr(self, "prad_tot"):
    #             self.prad_tot = deepcopy(self.prad)
    #             self.prad_sxr = deepcopy(self.prad)
    #             format_dataarray(self.prad_sxr, "sxr_radiation_power")

    #         prad_tot = self.prad_tot.sel(element=elem)
    #         prad_sxr = self.prad_sxr.sel(element=elem)
    #         for t in np.array(self.time_to_calculate, ndmin=1):
    #             prad_tot.loc[dict(t=t)] = np.trapz(
    #                 total_radiation.sel(t=t), self.volume.sel(t=t)
    #             )
    #             prad_sxr.loc[dict(t=t)] = np.trapz(
    #                 sxr_radiation.sel(t=t), self.volume.sel(t=t)
    #             )
    #         self.prad_tot.loc[dict(element=elem)] = prad_tot.values
    #         self.prad_sxr.loc[dict(element=elem)] = prad_sxr.values

    def write_to_pickle(self):
        with open(f"data_{self.pulse}.pkl", "wb") as f:
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

        DataArray and dictionaries of DataArrays currently permitted

        TODO: upgrade so other objects being tracked, e.g. Equilibrium
        """
        _dependencies = []
        for dependency in self.dependencies:
            if type(dependency) == dict:
                for data in dependency.values():
                    _dependencies.append(data.data)
            elif type(dependency) == DataArray:
                _dependencies.append(dependency.data)
            else:
                help(dependency)
                print(type(dependency))
                raise NotImplementedError(
                    "Hashing implemented for DataArray and Dict[DataArray] only"
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
            print("Recalculating")
        return deepcopy(self.operator())


def example_run(
    pulse: int = None,
    tstart=0.02,
    tend=0.1,
    dt=0.01,
    main_ion="h",
    impurities=("c", "ar", "he"),
    impurity_concentration=(0.03, 0.001, 0.01),
    verbose: bool = True,
    n_rad: int = 41,
    full_run=False,
    calc_power_loss: bool = False,
    **kwargs,
):
    # TODO: swap all profiles to new version!

    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=main_ion,
        impurities=impurities,
        impurity_concentration=impurity_concentration,
        full_run=full_run,
        verbose=verbose,
        n_rad=n_rad,
        **kwargs,
    )
    plasma.build_atomic_data(default=True, calc_power_loss=calc_power_loss)
    # Assign profiles to time-points
    nt = len(plasma.t)
    ne_peaking = np.linspace(1, 2, nt)
    te_peaking = np.linspace(1, 2, nt)
    vrot_peaking = np.linspace(1, 2, nt)
    vrot0 = np.linspace(plasma.Vrot_prof.y0 * 1.1, plasma.Vrot_prof.y0 * 2.5, nt)
    ti0 = np.linspace(plasma.Ti_prof.y0 * 1.1, plasma.Te_prof.y0 * 2.5, nt)
    nimp_peaking = np.linspace(1, 5, nt)
    nimp_y0 = plasma.Nimp_prof.y0 * 5 * np.linspace(1, 8, nt)
    nimp_wcenter = np.linspace(0.4, 0.1, nt)
    for i, t in enumerate(plasma.t):
        plasma.Te_prof.peaking = te_peaking[i]
        plasma.assign_profiles("electron_temperature", t=t)

        plasma.Ti_prof.peaking = te_peaking[i]
        plasma.Ti_prof.y0 = ti0[i]
        plasma.assign_profiles("ion_temperature", t=t)

        plasma.Vrot_prof.peaking = vrot_peaking[i]
        plasma.Vrot_prof.y0 = vrot0[i]
        plasma.assign_profiles("toroidal_rotation", t=t)

        plasma.Ne_prof.peaking = ne_peaking[i]
        plasma.assign_profiles("electron_density", t=t)

        plasma.Nimp_prof.peaking = nimp_peaking[i]
        plasma.Nimp_prof.y0 = nimp_y0[i]
        plasma.Nimp_prof.wcenter = nimp_wcenter[i]
        plasma.assign_profiles(profile="impurity_density", t=t)

    if pulse is None:
        equilibrium_data = fake_equilibrium_data(
            tstart=tstart, tend=tend, dt=dt / 2, machine_dims=plasma.machine_dimensions
        )
    else:
        from indica.readers import ST40Reader

        reader = ST40Reader(pulse, plasma.tstart - plasma.dt, plasma.tend + plasma.dt)
        equilibrium_data = reader.get("", "efit", 0)

    equilibrium = Equilibrium(equilibrium_data)
    plasma.set_equilibrium(equilibrium)

    return plasma


if __name__ == "__main__":
    example_run()
