from copy import deepcopy
from functools import lru_cache
import hashlib
import pickle
from typing import Callable

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.time import convert_in_time_dt
from indica.converters.time import get_tlabels_dt
from indica.datatypes import ELEMENTS
from indica.equilibrium import Equilibrium
from indica.equilibrium import fake_equilibrium_data
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
import indica.physics as ph
from indica.profiles_gauss import Profiles
from indica.readers import ADASReader
from indica.readers import ST40Reader
from indica.utilities import assign_data
from indica.utilities import assign_datatype
from indica.utilities import print_like

plt.ion()

# TODO: add elongation and triangularity in all equations

ADF11: dict = {
    "h": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
        "pls": "15",
        "prs": "15",
    },
    "he": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
        "pls": "15",
        "prs": "15",
    },
    "c": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
        "pls": "15",
        "prs": "15",
    },
    "ar": {
        "scd": "89",
        "acd": "89",
        "ccd": "89",
        "plt": "89",
        "prb": "89",
        "prc": "89",
        "pls": "15",
        "prs": "15",
    },
    "ne": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
        "pls": "15",
        "prs": "15",
    },
    "mo": {
        "scd": "89",
        "acd": "89",
        "ccd": "89",
        "plt": "89",
        "prb": "89",
        "prc": "89",
    },
    "w": {
        "scd": "89",
        "acd": "89",
        "ccd": "89",
        "plt": "89",
        "prb": "89",
        "prc": "89",
        "pls": "15",
        "prs": "15",
    },
}


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
        self.impurity_concentration = assign_data(
            DataArray(
                np.array(impurity_concentration),
                coords=[("element", list(self.impurities))],
            ),
            ("concentration", "impurity"),
        )
        self.set_adf11(ADF11)
        self.radial_coordinate = np.linspace(0, 1.0, n_rad)
        self.radial_coordinate_type = "rho_poloidal"
        self.machine_dimensions = machine_dimensions

        self.initialize_variables()

        self.equilibrium: Equilibrium

    def set_equilibrium(self, equilibrium: Equilibrium):
        """
        Assign equilibrium object
        """
        self.equilibrium = equilibrium
        self._rmag = assign_data(self.data1d_time, ("major_radius", "magnetic"), "m")
        self._zmag = assign_data(self.data1d_time, ("z", "magnetic"), "m")
        self._rmji = assign_data(self.data2d, ("radius", "major"), "m")
        self._rmjo = assign_data(self.data2d, ("radius", "major"), "m")
        self._rmin = assign_data(
            self.data2d, ("minor_radius", "plasma", "m")
        )  # LFS-HFS averaged value
        self._volume = assign_data(self.data2d, ("volume", "plasma"), "$m^3$")
        self._area = assign_data(self.data2d, ("area", "plasma"), "$m^2$")

    def set_adf11(self, adf11: dict):
        self.adf11 = adf11

    def initialize_variables(self):
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

        time = get_tlabels_dt(self.tstart, self.tend, self.dt)
        self.time_to_calculate = time

        nt = len(time)
        nr = len(self.radial_coordinate)
        nel = len(self.elements)
        nimp = len(self.impurities)

        coords_radius = assign_data(
            self.radial_coordinate,
            ("poloidal", "rho"),
            coords=[(self.radial_coordinate_type, self.radial_coordinate)],
        )
        coords_time = assign_data(
            time,
            ("", "time"),
            "s",
            coords=[("t", time)],
        )
        coords_elem = assign_data(
            list(self.elements),
            ("", "element"),
            "",
            coords=[("element", list(self.elements))],
        )
        coords_imp = assign_data(
            list(self.impurities),
            ("", "element"),
            "",
            coords=[("element", list(self.impurities))],
        )

        self.data1d_time: DataArray = DataArray(np.zeros(nt), coords=[coords_time])
        self.data1d_rho: DataArray = DataArray(np.zeros(nr), coords=[coords_radius])
        self.data2d: DataArray = DataArray(
            np.zeros((nt, nr)), coords=[coords_time, coords_radius]
        )
        self.data2d_elem: DataArray = DataArray(
            np.zeros((nel, nt)), coords=[coords_elem, coords_time]
        )
        self.data3d: DataArray = DataArray(
            np.zeros((nel, nt, nr)), coords=[coords_elem, coords_time, coords_radius]
        )
        self.data3d_imp: DataArray = DataArray(
            np.zeros((nimp, nt, nr)), coords=[coords_imp, coords_time, coords_radius]
        )

        self.t = assign_data(self.data1d_time, ("t", "plasma"), "s")
        self.t.values = time

        rho_type = self.radial_coordinate_type.split("_")
        if rho_type[1] != "poloidal":
            print_like("Only poloidal rho in input for the time being...")
            raise AssertionError
        self.rho = assign_data(self.data1d_rho, (rho_type[0], rho_type[1]))
        self.rho.values = self.radial_coordinate

        self.Te_prof = Profiles(datatype=("temperature", "electron"), xspl=self.rho)
        self.Ti_prof = Profiles(datatype=("temperature", "ion"), xspl=self.rho)
        self.Ne_prof = Profiles(datatype=("density", "electron"), xspl=self.rho)
        self.Nimp_prof = Profiles(datatype=("density", "impurity"), xspl=self.rho)
        self.Nh_prof = Profiles(datatype=("density", "thermal_neutral"), xspl=self.rho)
        self.Vrot_prof = Profiles(datatype=("rotation", "toroidal"), xspl=self.rho)

        self.ipla = assign_data(self.data1d_time, ("current", "plasma"), "A")
        self.ne_0 = assign_data(self.data1d_time, ("density", "electron"), "$m^{-3}$")
        self.te_0 = assign_data(self.data1d_time, ("temperature", "electron"), "eV")
        self.ti_0 = assign_data(self.data1d_time, ("temperature", "ion"), "eV")
        self.j_phi = assign_data(self.data2d, ("current_density", "plasma"), "A $m^2$")
        self.b_pol = assign_data(self.data2d, ("field", "poloidal"), "T")
        self.b_tor_lfs = assign_data(self.data2d, ("field", "LFS_toroidal"), "T")
        self.b_tor_hfs = assign_data(self.data2d, ("field", "HFS_toroidal"), "T")
        self.q_prof = assign_data(self.data2d, ("factor", "safety"), "")
        self.conductivity = assign_data(self.data2d, ("conductivity", "plasma"), "")
        self.l_i = assign_data(self.data1d_time, ("inductance", "internal"), "")

        # Independent plasma quantities
        self.electron_temperature = assign_data(
            self.data2d, ("temperature", "electron"), "eV"
        )
        self.electron_density: DataArray = assign_data(
            self.data2d, ("density", "electron"), "$m^{-3}$"
        )
        self.neutral_density: DataArray = assign_data(
            self.data2d, ("density", "thermal_neutral"), "$m^{-3}$"
        )
        self.tau: DataArray = assign_data(self.data2d, ("time", "residence"), "s")

        self.ion_temperature: DataArray = assign_data(
            self.data3d, ("temperature", "ion"), "eV"
        )
        self.toroidal_rotation: DataArray = assign_data(
            self.data3d, ("toroidal_rotation", "ion"), "rad $s^{-1}$"
        )
        self.impurity_density: DataArray = assign_data(
            self.data3d_imp, ("density", "impurity"), "$m^{-3}$"
        )
        self.fast_temperature: DataArray = assign_data(
            self.data2d, ("temperature", "fast"), "eV"
        )
        self.fast_density: DataArray = assign_data(
            self.data2d, ("density", "fast"), "$m^3$"
        )

        self.pressure_fast_parallel: DataArray = assign_data(
            self.data2d, ("pressure_parallel", "fast"), "Pa $m^{-3}$"
        )
        self.pressure_fast_perpendicular: DataArray = assign_data(
            self.data2d, ("pressure_perpendicular", "fast"), "Pa $m^{-3}$"
        )
        self._pressure_fast: DataArray = assign_data(
            self.data2d, ("pressure", "fast"), "Pa $m^{-3}$"
        )

        # Private variables for class property variables
        self._pressure_el = assign_data(
            self.data2d, ("pressure", "electron"), "Pa $m^{-3}$"
        )
        self._pressure_th = assign_data(
            self.data2d, ("pressure", "thermal"), "Pa $m^{-3}$"
        )
        self._pressure_tot = assign_data(
            self.data2d, ("pressure", "total"), "Pa $m^{-3}$"
        )
        self._pth = assign_data(self.data1d_time, ("pressure", "thermal"), "Pa")
        self._ptot = assign_data(self.data1d_time, ("pressure", "total"), "Pa")
        self._wth = assign_data(
            self.data1d_time, ("stored_energy", "thermal"), "J", long_name="Wth"
        )
        self._wp = assign_data(
            self.data1d_time, ("stored_energy", "total"), "J", long_name="Wp"
        )
        self._zeff = assign_data(
            self.data3d, ("charge", "effective"), "", long_name="Zeff"
        )
        self._ion_density = assign_data(self.data3d, ("density", "ion"), "$m^{-3}$")
        self._meanz = assign_data(self.data3d, ("charge", "mean"), "")
        self._total_radiation = assign_data(
            self.data3d, ("radiation_emission", "total"), "W $m^{-3}$"
        )
        self._sxr_radiation = assign_data(
            self.data3d, ("radiation_emission", "sxr"), "W $m^{-3}$"
        )
        self._prad_tot = assign_data(self.data2d_elem, ("radiation", "total"), "W")
        self._prad_sxr = assign_data(self.data2d_elem, ("radiation", "sxr"), "W")
        _fz = {}
        _lz_tot = {}
        _lz_sxr = {}
        for elem in self.elements:
            z_elem, a_elem, name_elem = ELEMENTS[elem]
            nz = z_elem + 1
            ion_charges = np.arange(nz)
            data3d_fz = DataArray(
                np.full((len(self.t), len(self.rho), nz), 0.0),
                coords=[
                    ("t", self.t),
                    ("rho_poloidal", self.rho),
                    ("ion_charges", ion_charges),
                ],
            )
            _fz[elem] = assign_data(data3d_fz, ("fractional_abundance", "ion"), "")
            _lz_tot[elem] = assign_data(
                data3d_fz, ("radiation_loss_parameter", "total"), "W $m^3$"
            )
            _lz_sxr[elem] = assign_data(
                data3d_fz, ("radiation_loss_parameter", "sxr"), "W $m^3$"
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

    def update_profiles(
        self,
        parameters: dict,
    ):
        """
        Update plasma profiles with profile parameters i.e.
        {"Ne_prof.y0":1e19} -> Ne_prof.y0
        """
        profile_prefixes: list = [
            "Te_prof",
            "Ti_prof",
            "Ne_prof",
            "Nimp_prof",
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

        profile_mapping = {
            "Te_prof": "electron_temperature",
            "Ti_prof": "ion_temperature",
            "Ne_prof": "electron_density",
            "Vrot_prof": "toroidal_rotation",
            "Nimp_prof": "impurity_density",
            "Nh_prof": "neutral_density",
        }

        for key in profile_names:
            self.assign_profiles(profile_mapping[key], t=self.time_to_calculate)

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
        self._pressure_th = ph.calc_pressure(self.ion_density, self.ion_temperature).sum("element") + self.pressure_el
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
        self._zeff = self.ion_density * self.meanz ** 2 / self.electron_density
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

        main_ion_density = self.electron_density - self.fast_density * meanz.sel(element=self.main_ion) \
                           - (self.impurity_density * meanz).sum("element")

        self._ion_density.loc[dict(element=self.main_ion)] =  main_ion_density
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

    @property
    def vloop(self):
        zeff = self.zeff
        j_phi = self.j_phi
        self.conductivity = ph.conductivity_neo(
            self.electron_density,
            self.electron_temperature,
            zeff.sum("element"),
            self.minor_radius,
            self.minor_radius.interp(rho_poloidal=1.0),
            self.R_mag,
            self.q_prof,
            approx="sauter",
        )
        for t in np.array(self.time_to_calculate, ndmin=1):
            resistivity = 1.0 / self.conductivity.sel(t=t)
            ir = np.where(np.isfinite(resistivity))
            vloop = ph.vloop(
                resistivity[ir], j_phi.sel(t=t)[ir], self.area.sel(t=t)[ir]
            )
            self._vloop.loc[dict(t=t)] = vloop.values
        return self._vloop

    def calc_impurity_density(self, elements):
        """
        Calculate impurity density from concentration and electron density
        """
        for elem in elements:
            Nimp = self.electron_density * self.impurity_concentration.sel(element=elem)
            self.impurity_density.loc[
                dict(
                    element=elem,
                )
            ] = Nimp.values

    def impose_flat_zeff(self):
        """
        Adapt impurity concentration to generate flat Zeff contribution
        """

        for elem in self.impurities:
            if np.count_nonzero(self.ion_density.sel(element=elem)) != 0:
                zeff_tmp = (
                    self.ion_density.sel(element=elem)
                    * self.meanz.sel(element=elem) ** 2
                    / self.electron_density
                )
                value = zeff_tmp.where(zeff_tmp.rho_poloidal < 0.2).mean("rho_poloidal")
                zeff_tmp = zeff_tmp / zeff_tmp * value
                ion_density_tmp = zeff_tmp / (
                    self.meanz.sel(element=elem) ** 2 / self.electron_density
                )
                self.ion_density.loc[dict(element=elem)] = ion_density_tmp.values

    def convert_in_time(self, value: DataArray, method="linear"):
        binned = convert_in_time_dt(self.tstart, self.tend, self.dt, value).interp(
            t=self.t, method=method
        )

        return binned

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

    def set_neutral_density(self, y0=1.0e10, y1=1.0e15, decay=12):
        self.Nh_prof.y0 = y0
        self.Nh_prof.y1 = y1
        self.Nh_prof.yend = y1
        self.Nh_prof.wped = decay
        self.Nh_prof()
        for t in np.array(self.time_to_calculate, ndmin=1):
            self.neutral_density.loc[dict(t=t)] = self.Nh_prof()

    def map_to_midplane(self):
        # TODO: streamline to avoid re-calculating quantities e.g. ion_density..
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

    def calc_centrifugal_asymmetry(
        self, time=None, test_toroidal_rotation=None, plot=False
    ):
        """
        Calculate (R, z) maps of the ion densities caused by centrifugal asymmetry
        """
        if time is None:
            time = self.t

        # TODO: make this attribute creation a property and standardize?
        if not hasattr(self, "ion_density_2d"):
            self.rho_2d = self.equilibrium.rho.interp(t=self.t, method="nearest")
            tmp = deepcopy(self.rho_2d)
            ion_density_2d = []
            for elem in self.elements:
                ion_density_2d.append(tmp)

            self.ion_density_2d = xr.concat(ion_density_2d, "element").assign_coords(
                element=self.elements
            )
            assign_datatype(self.ion_density_2d, ("density", "ion"))
            self.centrifugal_asymmetry = deepcopy(self.ion_density)
            assign_datatype(self.centrifugal_asymmetry, ("asymmetry", "centrifugal"))
            self.asymmetry_multiplier = deepcopy(self.ion_density_2d)
            assign_datatype(
                self.asymmetry_multiplier, ("asymmetry_multiplier", "centrifugal")
            )

        # If toroidal rotation != 0 calculate ion density on 2D poloidal plane
        if test_toroidal_rotation is not None:
            toroidal_rotation = deepcopy(self.ion_temperature)
            assign_datatype(toroidal_rotation, ("rotation", "toroidal"), "rad/s")
            toroidal_rotation /= toroidal_rotation.max("rho_poloidal")
            toroidal_rotation *= test_toroidal_rotation  # rad/s
            self.toroidal_rotation = toroidal_rotation

        if not np.any(self.toroidal_rotation != 0):
            return

        ion_density = self.ion_density
        meanz = self.meanz
        zeff = self.zeff.sum("element")
        R_0 = self.maj_r_lfs.interp(rho_poloidal=self.rho_2d).drop_vars("rho_poloidal")
        for elem in self.elements:
            main_ion_mass = ELEMENTS[self.main_ion][1]
            mass = ELEMENTS[elem][1]
            asymm = ph.centrifugal_asymmetry(
                self.ion_temperature.sel(element=elem).drop_vars("element"),
                self.electron_temperature,
                mass,
                meanz.sel(element=elem).drop_vars("element"),
                zeff,
                main_ion_mass,
                toroidal_rotation=self.toroidal_rotation.sel(element=elem).drop_vars(
                    "element"
                ),
            )
            self.centrifugal_asymmetry.loc[dict(element=elem)] = asymm
            asymmetry_factor = asymm.interp(rho_poloidal=self.rho_2d)
            self.asymmetry_multiplier.loc[dict(element=elem)] = np.exp(
                asymmetry_factor * (self.rho_2d.R**2 - R_0**2)
            )

        self.ion_density_2d = (
            ion_density.interp(rho_poloidal=self.rho_2d).drop_vars("rho_poloidal")
            * self.asymmetry_multiplier
        )
        assign_datatype(self.ion_density_2d, ("density", "ion"), "m^-3")

        if plot:
            t = self.t[6]
            for elem in self.elements:
                plt.figure()
                z = self.z_mag.sel(t=t)
                rho = self.rho_2d.sel(t=t).sel(z=z, method="nearest")
                plt.plot(
                    rho,
                    self.ion_density_2d.sel(element=elem).sel(
                        t=t, z=z, method="nearest"
                    ),
                )
                self.ion_density.sel(element=elem).sel(t=t).plot(linestyle="dashed")
                plt.title(elem)

            elem = "ar"
            plt.figure()
            np.log(
                self.ion_density_2d.sel(element=elem).sel(t=t, method="nearest")
            ).plot()
            self.rho_2d.sel(t=t, method="nearest").plot.contour(
                levels=10, colors="white"
            )
            plt.xlabel("R (m)")
            plt.ylabel("z (m)")
            plt.title(f"log({elem} density")
            plt.axis("scaled")
            plt.xlim(0, 0.8)
            plt.ylim(-0.6, 0.6)

    def calc_rad_power_2d(self):
        """
        Calculate total and SXR filtered radiated power on a 2D poloidal plane
        including effects from poloidal asymmetries
        """
        for elem in self.elements:
            total_radiation = (
                self.lz_tot[elem].sum("ion_charges")
                * self.electron_density
                * self.ion_density.sel(element=elem)
            )
            total_radiation = xr.where(
                total_radiation >= 0,
                total_radiation,
                0.0,
            )
            self.total_radiation.loc[dict(element=elem)] = total_radiation.values

            sxr_radiation = (
                self.lz_sxr[elem].sum("ion_charges")
                * self.electron_density
                * self.ion_density.sel(element=elem)
            )
            sxr_radiation = xr.where(
                sxr_radiation >= 0,
                sxr_radiation,
                0.0,
            )
            self.sxr_radiation.loc[dict(element=elem)] = sxr_radiation.values

            if not hasattr(self, "prad_tot"):
                self.prad_tot = deepcopy(self.prad)
                self.prad_sxr = deepcopy(self.prad)
                assign_data(self.prad_sxr, ("radiation", "sxr"))

            prad_tot = self.prad_tot.sel(element=elem)
            prad_sxr = self.prad_sxr.sel(element=elem)
            for t in np.array(self.time_to_calculate, ndmin=1):
                prad_tot.loc[dict(t=t)] = np.trapz(
                    total_radiation.sel(t=t), self.volume.sel(t=t)
                )
                prad_sxr.loc[dict(t=t)] = np.trapz(
                    sxr_radiation.sel(t=t), self.volume.sel(t=t)
                )
            self.prad_tot.loc[dict(element=elem)] = prad_tot.values
            self.prad_sxr.loc[dict(element=elem)] = prad_sxr.values

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
        return self.operator()


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
        reader = ST40Reader(pulse, plasma.tstart - plasma.dt, plasma.tend + plasma.dt)
        equilibrium_data = reader.get("", "efit", 0)

    equilibrium = Equilibrium(equilibrium_data)
    plasma.set_equilibrium(equilibrium)

    return plasma


if __name__ == "__main__":
    example_run()
