from copy import deepcopy
import pickle

import hda.physics as ph
import hda.profiles as profiles
from hda.profiles import Profiles
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import least_squares
import xarray as xr
from xarray import DataArray
from xarray import Dataset
import inspect

from indica.converters import FluxSurfaceCoordinates
from indica.converters.time import bin_in_time_dt
from indica.converters.time import get_tlabels_dt
from indica.datatypes import ELEMENTS
from indica.equilibrium import Equilibrium
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.provenance import get_prov_attribute
from indica.readers import ADASReader
from indica.session import hash_vals

plt.ion()

# TODO: add elongation and triangularity in all equations

ADF11 = {
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
        "plt": "00",
        "prb": "00",
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
}


class Plasma:
    def __init__(
        self,
        tstart=0.01,
        tend=0.14,
        dt=0.01,
        ntheta=5,
        machine_dimensions=((0.15, 0.9), (-0.8, 0.8)),
        elements=("h", "c", "ar"),
    ):
        """

        Parameters
        ----------
        pulse

        """

        self.ADASReader = ADASReader()
        self.elements = elements
        self.impurities = elements[1:]
        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.t = get_tlabels_dt(self.tstart, self.tend, self.dt)
        self.theta = np.linspace(0, 2 * np.pi, ntheta + 1)[:-1]
        self.radial_coordinate = np.linspace(0, 1.0, 41)
        self.radial_coordinate_type = "rho_poloidal"
        self.machine_dimensions = machine_dimensions

        self.forward_models = {}

        self.initialize_variables()

    def initialize_variables(self):
        """
        Initialize all class attributes

        Assign elements, machine dimensions and coordinates used throughout the analysis
            rho
            time
            theta

        TODO: add units to each variable --> e.g. vtor in rad/s not in km/s
        """
        # Assign attributes
        self.machine_R = np.linspace(
            self.machine_dimensions[0][0], self.machine_dimensions[0][1], 100
        )
        self.machine_z = np.linspace(
            self.machine_dimensions[1][0], self.machine_dimensions[1][1], 100
        )

        self.optimisation = {
            "equil": "",
            "el_dens": "",
            "el_temp": "",
            "ion_temp": "",
            "stored_en": "",
        }
        self.pulse = None

        nt = len(self.t)
        nr = len(self.radial_coordinate)
        nel = len(self.elements)
        nimp = len(self.impurities)
        nth = len(self.theta)

        R_midplane = np.linspace(self.machine_R.min(), self.machine_R.max(), 50)
        self.R_midplane = R_midplane
        z_midplane = np.full_like(R_midplane, 0.0)
        self.z_midplane = z_midplane

        coords_radius = (self.radial_coordinate_type, self.radial_coordinate)
        # coords_rmid = ("R", R_midplane)
        coords_theta = ("poloidal_angle", self.theta)
        coords_time = ("t", self.t)
        coords_elem = ("element", list(self.elements))
        coords_imp = ("element", list(self.impurities))

        data0d = DataArray(0.0)
        data1d_theta = DataArray(np.zeros(nth), coords=[coords_theta])
        data1d_time = DataArray(np.zeros(nt), coords=[coords_time])
        data1d_rho = DataArray(np.zeros(nr), coords=[coords_radius])
        data2d = DataArray(np.zeros((nt, nr)), coords=[coords_time, coords_radius])
        data2d_elem = DataArray(np.zeros((nel, nt)), coords=[coords_elem, coords_time])
        data3d = DataArray(
            np.zeros((nel, nt, nr)), coords=[coords_elem, coords_time, coords_radius]
        )
        data3d_imp = DataArray(
            np.zeros((nimp, nt, nr)), coords=[coords_imp, coords_time, coords_radius]
        )

        self.time = assign_data(data1d_time, ("t", "plasma"), "s")
        self.time.values = self.t
        self.freq = 1.0 / self.dt

        rho_type = self.radial_coordinate_type.split("_")
        if rho_type[1] != "poloidal":
            print_like("Only poloidal rho in input for the time being...")
            raise AssertionError
        self.rho = assign_data(data1d_rho, (rho_type[0], rho_type[1]))
        self.rho.values = self.radial_coordinate

        data3d_fz = {}
        for elem in self.elements:
            nz = ELEMENTS[elem][0] + 1
            ion_charges = np.arange(nz)
            data3d_fz[elem] = DataArray(
                np.full((len(self.t), len(self.rho), nz), np.nan),
                coords=[
                    ("t", self.t),
                    ("rho_poloidal", self.rho),
                    ("ion_charges", ion_charges),
                ],
            )

        self.Te_prof = Profiles(datatype=("temperature", "electron"), xspl=self.rho)
        self.Ti_prof = Profiles(datatype=("temperature", "ion"), xspl=self.rho)
        self.Ne_prof = Profiles(datatype=("density", "electron"), xspl=self.rho)
        self.Nimp_prof = Profiles(datatype=("density", "impurity"), xspl=self.rho)
        self.Nimp_prof.y1 = 3.0e19
        self.Nimp_prof.yend = 2.0e19
        self.Nimp_prof.build_profile()
        self.Nh_prof = Profiles(datatype=("neutral_density", "neutrals"), xspl=self.rho)
        self.Vrot_prof = Profiles(datatype=("rotation", "ion"), xspl=self.rho)

        # TODO: add derivation of rho-toroidal
        self.theta = assign_data(data1d_theta, ("angle", "poloidal"), "deg")
        self.ipla = assign_data(data1d_time, ("current", "plasma"), "A")
        self.bt_0 = assign_data(data1d_time, ("field", "toroidal"), "T")
        self.R_bt_0 = assign_data(data0d, ("major_radius", "toroidal_field"), "T")
        self.R_0 = assign_data(data1d_time, ("major_radius", "geometric"), "m")
        self.R_mag = assign_data(data1d_time, ("major_radius", "magnetic"))
        self.z_mag = assign_data(data1d_time, ("z", "magnetic"))
        self.maj_r_lfs = assign_data(data2d, ("radius", "major"))
        self.maj_r_hfs = assign_data(data2d, ("radius", "major"))
        self.ne_0 = assign_data(data1d_time, ("density", "electron"))
        self.te_0 = assign_data(data1d_time, ("temperature", "electron"))
        self.ti_0 = assign_data(data1d_time, ("temperature", "ion"))
        self.el_temp = assign_data(data2d, ("temperature", "electron"))
        self.el_dens = assign_data(data2d, ("density", "electron"))
        self.neutral_dens = assign_data(data2d, ("density", "neutral"))
        self.tau = assign_data(data2d, ("time", "residence"))
        self.min_r = assign_data(data2d, ("minor_radius", "plasma"))
        self.cr0 = assign_data(data1d_time, ("minor_radius", "separatrizx"))
        self.volume = assign_data(data2d, ("volume", "plasma"))
        self.area = assign_data(data2d, ("area", "plasma"))
        self.r_a = assign_data(data1d_time, ("minor_radius", "LFS"))
        self.r_b = assign_data(data1d_time, ("minor_radius", "top"))
        self.r_c = assign_data(data1d_time, ("minor_radius", "HFS"))
        self.r_d = assign_data(data1d_time, ("minor_radius", "bottom"))
        self.kappa = assign_data(data1d_time, ("elongation", "plasma"))
        self.delta = assign_data(data1d_time, ("triangularity", "plasma"))
        self.j_phi = assign_data(data2d, ("current", "density"))
        self.b_pol = assign_data(data2d, ("field", "poloidal"))
        self.b_tor_lfs = assign_data(data2d, ("field", "toroidal"))
        self.b_tor_hfs = assign_data(data2d, ("field", "toroidal"))
        self.q_prof = assign_data(data2d, ("factor", "safety"))
        self.conductivity = assign_data(data2d, ("conductivity", "plasma"))
        self.l_i = assign_data(data1d_time, ("inductance", "internal"))

        self.main_ion = self.elements[0]
        self.impurities = self.elements[1:]
        self.imp_conc = assign_data(
            DataArray(np.zeros(len(self.impurities)), coords=[coords_imp]),
            ("concentration", "ion"),
        )
        self.ion_temp = assign_data(data3d, ("temperature", "ion"))
        self.vtor = assign_data(data3d, ("toroidal_rotation", "ion"))
        self.imp_dens = assign_data(data3d_imp, ("density", "ion"), "m^-3")
        self.fast_temp = assign_data(data2d, ("temperature", "fast"))
        self.fast_dens = assign_data(data2d, ("density", "fast"))

        # Private variables for properties
        self.properties = {
            "ion_dens": ("el_dens", "fast_dens", "meanz", "main_ion", "imp_dens",),
            "zeff": ("el_dens", "ion_dens", "meanz"),
            "meanz": ("fz",),
            "fz": ("el_temp", "el_dens", "tau", "neutral_dens",),
            "lz_tot": ("el_temp, el_dens, neutral_dens", "fz"),
            "lz_sxr": ("el_temp, el_dens, neutral_dens", "fz"),
            "tot_rad": ("lz_tot", "el_dens", "ion_dens"),
            "sxr_rad": ("lz_sxr", "el_dens", "ion_dens"),
            "prad_tot": ("tot_rad"),
            "prad_sxr": ("sxr_rad"),
            "pressure_el": ("el_dens", "el_temp"),
            "pressure_th": ("el_dens", "ion_dens", "el_temp", "ion_temp"),
            "pressure_tot": ("pressure_th", "fast_dens", "fast_temp"),
            "pth": ("pressure_th",),
            "ptot": ("pressure_tot",),
            "wth": ("pth",),
            "wp": ("ptot",),
        }

        self.property_hashes = {}
        for k in self.properties:
            self.property_hashes[k] = None

        self._ion_dens = assign_data(data3d, ("density", "ion"), "m^-3")
        self._zeff = assign_data(data3d, ("charge", "effective"), "")
        self._fz = deepcopy(data3d_fz)
        for elem in self.elements:
            assign_data(self._fz[elem], ("fractional_abundance", "ion"), "")
        self._lz_tot = deepcopy(data3d_fz)
        for elem in self.elements:
            assign_data(self._lz_tot[elem], ("cooling_factor", "total"), "")
        self._lz_sxr = deepcopy(data3d_fz)
        for elem in self.elements:
            assign_data(self._lz_sxr[elem], ("cooling_factor", "sxr"), "")
        self._meanz = assign_data(data3d, ("charge", "mean"), "")
        self._tot_rad = assign_data(data3d, ("radiation_emission", "total"), "W m^-3")
        self._sxr_rad = assign_data(data3d, ("radiation_emission", "sxr"), "W m^-3")
        self._prad_tot = assign_data(data2d_elem, ("radiation", "total"), "W")
        self._prad_sxr = assign_data(data2d_elem, ("radiation", "sxr"), "W")
        self._pressure_el = assign_data(data2d, ("pressure", "electron"), "Pa m^-3")
        self._pressure_th = assign_data(data2d, ("pressure", "thermal"), "Pa m^-3")
        self._pressure_tot = assign_data(data2d, ("pressure", "total"), "Pa m^-3")
        self._pth = assign_data(data1d_time, ("pressure", "thermal"), "Pa")
        self._ptot = assign_data(data1d_time, ("pressure", "total"), "Pa")
        self._wth = assign_data(data1d_time, ("stored_energy", "thermal"), "J")
        self._wp = assign_data(data1d_time, ("stored_energy", "total"), "J")
        self._beta_pol = assign_data(data1d_time, ("beta", "poloidal"), "J")

    def get_property_value(self, dependencies: list):
        """
        Get value of property defining dependencies, private variable name and function to call
        """
        called_from = str(inspect.stack()[1][3])
        var_name = f"_{called_from}"
        func_name = f"calc_{called_from}"
        # print(f"getting {called_from}")
        if self.data_changed(called_from, dependencies):
            # print(f"..{called_from} dependencies have changed..")
            getattr(self, func_name)()
        return getattr(self, var_name)

    def calc_pressure_el(self):
        self._pressure_el.values = ph.calc_pressure(self.el_dens, self.el_temp)
        return self._pressure_el

    def calc_pressure_th(self):
        ion_dens = self.ion_dens
        self._pressure_th.values = self.pressure_el
        for elem in self.elements:
            self._pressure_th.values += ph.calc_pressure(
                ion_dens.sel(element=elem).values,
                self.ion_temp.sel(element=elem).values,
            )
        return self._pressure_th

    def calc_pressure_tot(self):
        self._pressure_tot.values = self.pressure_th + ph.calc_pressure(
            self.fast_dens, self.fast_temp
        )
        return self._pressure_tot

    def calc_pth(self):
        pressure_th = self.pressure_th
        for t in self.t:
            self._pth.loc[dict(t=t)] = np.trapz(
                pressure_th.sel(t=t), self.volume.sel(t=t)
            )
        return self._pth

    def calc_ptot(self):
        pressure_tot = self.pressure_tot
        for t in self.t:
            self._ptot.loc[dict(t=t)] = np.trapz(
                pressure_tot.sel(t=t), self.volume.sel(t=t)
            )
        return self._ptot

    def calc_wth(self):
        pth = self.pth
        self._wth.values = 3 / 2 * pth.values
        return self._wth

    def calc_wp(self):
        ptot = self.ptot
        self._wtot.values = 3 / 2 * ptot.values
        return self._wtot

    def calc_fz(self):
        for elem in self.elements:
            for t in self.t:
                Te = self.el_temp.sel(t=t)
                Ne = self.el_dens.sel(t=t)
                tau = None
                if np.any(self.tau != 0):
                    tau = self.tau.sel(t=t)
                Nh = None
                if np.any(self.neutral_dens != 0):
                    Nh = self.neutral_dens.sel(t=t)
                if any(np.logical_not((Te > 0) * (Ne > 0))):
                    continue
                fz_tmp = self.fract_abu[elem](Ne, Te, Nh=Nh, tau=tau)
                self._fz[elem].loc[dict(t=t)] = fz_tmp.transpose().values
        return self._fz

    def calc_zeff(self):
        ion_dens = self.ion_dens
        meanz = self.meanz
        for elem in self.elements:
            self._zeff.loc[dict(element=elem)] = ph.effective_charge(
                self.el_dens.values,
                ion_dens.sel(element=elem).values,
                meanz.sel(element=elem).values,
            )
        return self._zeff

    def calc_ion_dens(self):
        imp_dens = self.imp_dens
        meanz = self.meanz
        main_ion_dens = self.el_dens - self.fast_dens * meanz.sel(element=self.main_ion)
        for elem in self.impurities:
            self._ion_dens.loc[dict(element=elem)] = imp_dens.sel(element=elem).values
            main_ion_dens -= imp_dens.sel(element=elem) * meanz.sel(element=elem)

        self._ion_dens.loc[dict(element=self.main_ion)] = main_ion_dens.values
        return self._ion_dens

    def calc_meanz(self):
        fz = self.fz
        for elem in self.elements:
            self._meanz.loc[dict(element=elem)] = (
                (fz[elem] * fz[elem].ion_charges).sum("ion_charges").values
            )
        return self._meanz

    def calc_lz_tot(self):
        fz = self.fz
        for elem in self.elements:
            for t in self.t:
                Ne = self.el_dens.sel(t=t)
                Te = self.el_temp.sel(t=t)
                if any(np.logical_not((Te > 0) * (Ne > 0))):
                    continue
                Fz = fz[elem].sel(t=t).transpose()
                Nh = None
                if np.any(self.neutral_dens.sel(t=t) != 0):
                    Nh = self.neutral_dens.sel(t=t)
                self._lz_tot[elem].loc[dict(t=t)] = (
                    self.power_loss_tot[elem](Ne, Te, Fz, Nh=Nh, bounds_check=False)
                    .transpose()
                    .values
                )
        return self._lz_tot

    def calc_lz_sxr(self):
        fz = self.fz
        for elem in self.elements:
            for t in self.t:
                Ne = self.el_dens.sel(t=t)
                Te = self.el_temp.sel(t=t)
                if any(np.logical_not((Te > 0) * (Ne > 0))):
                    continue
                Fz = fz[elem].sel(t=t).transpose()
                Nh = None
                if np.any(self.neutral_dens.sel(t=t) != 0):
                    Nh = self.neutral_dens.sel(t=t)
                self._lz_sxr[elem].loc[dict(t=t)] = (
                    self.power_loss_sxr[elem](Ne, Te, Fz, Nh=Nh, bounds_check=False)
                    .transpose()
                    .values
                )
        return self._lz_tot

    def calc_tot_rad(self):
        lz_tot = self.lz_tot
        ion_dens = self.ion_dens
        for elem in self.elements:
            tot_rad = (
                lz_tot[elem].sum("ion_charges")
                * self.el_dens
                * ion_dens.sel(element=elem)
            )
            self._tot_rad.loc[dict(element=elem)] = xr.where(
                tot_rad >= 0, tot_rad, 0.0,
            ).values
        return self._tot_rad

    def calc_sxr_rad(self):
        lz_sxr = self.lz_sxr
        ion_dens = self.ion_dens
        for elem in self.elements:
            tot_sxr = (
                lz_sxr[elem].sum("ion_charges")
                * self.el_dens
                * ion_dens.sel(element=elem)
            )
            self._sxr_rad.loc[dict(element=elem)] = xr.where(
                sxr_rad >= 0, tot_rad, 0.0,
            ).values
        return self._sxr_rad

    @property
    def pressure_el(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def pressure_th(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def pressure_th(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def pressure_tot(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def pth(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def ptot(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def wth(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def wp(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def fz(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def zeff(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def ion_dens(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def meanz(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def lz_tot(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def lz_sxr(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def tot_rad(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def sxr_rad(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    @property
    def prad_tot(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    def calc_prad_tot(self):
        tot_rad = self.tot_rad
        for elem in self.elements:
            for t in self.t:
                self._prad_tot.loc[dict(element=elem, t=t)] = np.trapz(
                    tot_rad.sel(element=elem, t=t), self.volume.sel(t=t)
                )
        return self._prad_tot

    @property
    def prad_sxr(self):
        property_name = get_func_name()
        return self.get_property_value(self.properties[property_name])

    def calc_prad_sxr(self):
        sxr_rad = self.sxr_rad
        for elem in self.elements:
            for t in self.t:
                self._prad_sxr.loc[dict(element=elem, t=t)] = np.trapz(
                    sxr_rad.sel(element=elem, t=t), self.volume.sel(t=t)
                )
        return self._prad_sxr

    def build_current_density(self):
        """
        Build current density profile (A/m**2) given the total plasma current,
        plasma geometry and a shape parameter
        """

        for t in self.time:
            rho = self.rho.values
            ipla = self.ipla.sel(t=t).values
            r_a = self.r_a.sel(t=t).values
            area = self.area.sel(t=t).values
            prof_shape = self.el_temp.sel(t=t) / self.el_temp.sel(t=t).max()

            j_phi = ph.current_density(ipla, rho, r_a, area, prof_shape)

            self.j_phi.loc[dict(t=t)] = (j_phi * 10).values

    def calc_magnetic_field(self):
        """
        Calculate magnetic field profiles (poloidal & toroidal)
        """

        for t in self.time:
            R_bt_0 = self.R_bt_0.values
            R_mag = self.R_mag.sel(t=t).values
            ipla = self.ipla.sel(t=t).values
            bt_0 = self.bt_0.sel(t=t).values
            # z_mag = self.z_mag.sel(t=t).values
            maj_r_lfs = self.maj_r_lfs.sel(t=t).values
            maj_r_hfs = self.maj_r_hfs.sel(t=t).values
            j_phi = self.j_phi.sel(t=t).values
            r_a = self.r_a.sel(t=t).values
            min_r = self.min_r.sel(t=t).values
            volume = self.volume.sel(t=t).values
            area = self.area.sel(t=t).values

            self.b_tor_lfs.loc[dict(t=t)] = ph.toroidal_field(
                bt_0, R_bt_0, maj_r_lfs
            ).values
            self.b_tor_hfs.loc[dict(t=t)] = ph.toroidal_field(
                bt_0, R_bt_0, maj_r_hfs
            ).values

            b_pol = ph.poloidal_field(j_phi, min_r, area)
            self.b_pol.loc[dict(t=t)] = b_pol
            self.l_i.loc[dict(t=t)] = ph.internal_inductance(
                b_pol, ipla, volume, approx=2, R_mag=R_mag
            ).values

            b_tor = ((self.b_tor_lfs.sel(t=t) + self.b_tor_hfs.sel(t=t)) / 2.0).values

            self.q_prof.loc[dict(t=t)] = ph.safety_factor(
                b_tor, b_pol, min_r, r_a, R_mag
            ).values

    def calc_beta_poloidal(self):
        """
        Calculate Beta poloidal

        ??? Use total or thermal pressure ???
        """

        for t in self.time:
            b_pol = self.b_pol.sel(t=t).values
            pressure = self.pressure_tot.sel(t=t).values
            volume = self.volume.sel(t=t).values

            self.beta_pol.loc[dict(t=t)] = ph.beta_poloidal(
                b_pol, pressure, volume
            ).values

    def calc_vloop(self):
        """
        Given Zeff, Te and Ne: calculate resistivity and Vloop
        """

        zeff = self.zeff.sum("element")
        self.conductivity = ph.conductivity_neo(
            self.el_dens,
            self.el_temp,
            zeff,
            self.min_r,
            self.r_a,
            self.R_mag,
            self.q_prof,
            approx="sauter",
        )
        vloop = []
        for t in self.time:
            resistivity = 1.0 / self.conductivity.sel(t=t)
            ir = np.where(np.isfinite(resistivity))

            j_phi = self.j_phi.sel(t=t)
            area = self.area.sel(t=t)

            vloop.append(ph.vloop(resistivity[ir], j_phi[ir], area[ir]))

        vloop = DataArray(np.array(vloop), coords=[("t", self.t)])

        return vloop

    def build_data(self, data, pulse: int = None, equil="efit", instrument=""):
        """
        Reorganise raw data dictionary on the desired time axis and generate
        geometry information from equilibrium reconstruction

        Parameters
        ----------
        data
            Raw data dictionary
        equil
            Equilibrium code to use for equilibrium object
        instrument
            Build data only for specified instrument

        Returns
        -------

        """
        print_like("Building data class")

        if len(instrument) == 0:
            self.initialize_variables()

            self.pulse = pulse
            revision = get_prov_attribute(data[equil]["rmag"].provenance, "revision")
            self.optimisation["equil"] = f"{equil}:{revision}"

            t_ip = data["efit"]["ipla"].t
            if self.tstart < t_ip.min():
                print_like("Start time changed to stay inside Ip limit")
                self.tstart = t_ip.min().values
            if self.tend > t_ip.max():
                print_like("End time changed to stay inside Ip limit")
                self.tend = t_ip.max().values

            self.equil = equil

            if equil in data.keys():
                print_like("Initialise equilibrium object")
                self.equilibrium = Equilibrium(data[equil])
                self.flux_coords = FluxSurfaceCoordinates("poloidal")
                self.flux_coords.set_equilibrium(self.equilibrium)

        print_like("Assign equilibrium, bin data in time")
        binned_data = {}

        # np.seterr(all="reise")
        for kinstr in data.keys():
            if (len(instrument) > 0) and (kinstr != instrument):
                continue
            instrument_data = {}

            if type(data[kinstr]) != dict:
                value = deepcopy(data[kinstr])
                if np.size(value) > 1:
                    value = bin_in_time_dt(self.tstart, self.tend, self.dt, value)
                binned_data[kinstr] = value
                continue

            for kquant in data[kinstr].keys():
                value = data[kinstr][kquant]
                if "t" in value.coords:
                    value = bin_in_time_dt(self.tstart, self.tend, self.dt, value)

                if "transform" in data[kinstr][kquant].attrs:
                    value.attrs["transform"] = data[kinstr][kquant].transform
                    value.transform.set_equilibrium(self.equilibrium, force=True)
                    if "LinesOfSightTransform" in str(value.attrs["transform"]):
                        geom_attrs = remap_diagnostic(value, self.flux_coords)
                        for kattrs in geom_attrs:
                            value.attrs[kattrs] = geom_attrs[kattrs]

                if "provenance" in data[kinstr][kquant].attrs:
                    value.attrs["provenance"] = data[kinstr][kquant].provenance

                instrument_data[kquant] = value

            binned_data[kinstr] = instrument_data
            if kinstr == instrument:
                break

        if (len(instrument) == 0) and ("efit" in binned_data.keys()):
            self.ipla.values = binned_data["efit"]["ipla"]
            self.cr0.values = (
                binned_data["efit"]["rmjo"] - binned_data["efit"]["rmji"]
            ).sel(rho_poloidal=1) / 2.0

            self.R_mag = binned_data["efit"]["rmag"]
            self.z_mag = binned_data["efit"]["zmag"]

            self.R_bt_0 = binned_data["R_bt_0"]
            self.bt_0 = binned_data["bt_0"]

        # np.seterr(all="warn")
        return binned_data

    def apply_limits(
        self,
        data,
        diagnostic: str,
        quantity=None,
        val_lim=(np.nan, np.nan),
        err_lim=(np.nan, np.nan),
    ):
        """
        Set to Nan all data whose value or relative error aren't within specified limits
        """

        if quantity is None:
            quantity = list(data[diagnostic])
        else:
            quantity = list(quantity)

        for q in quantity:
            error = None
            value = data[diagnostic][q]
            if "error" in value.attrs.keys():
                error = data[diagnostic][q].attrs["error"]

            if np.isfinite(val_lim[0]):
                print(val_lim[0])
                value = xr.where(value >= val_lim[0], value, np.nan)
            if np.isfinite(val_lim[1]):
                print(val_lim[1])
                value = xr.where(value <= val_lim[1], value, np.nan)

            if error is not None:
                if np.isfinite(err_lim[0]):
                    print(err_lim[0])
                    value = xr.where((error / value) >= err_lim[0], value, np.nan)
                if np.isfinite(err_lim[1]):
                    print(err_lim[1])
                    value = xr.where((error / value) <= err_lim[1], value, np.nan)

            data[diagnostic][q].values = value.values

        return data

    def calculate_geometry(self):
        if hasattr(self, "equilibrium"):
            print_like("Calculate geometric quantities")
            for ith, th in enumerate(self.theta):
                min_r_tmp, _ = self.equilibrium.minor_radius(
                    self.equilibrium.rmji.rho_poloidal, th
                )
                if ith == 0:
                    min_r = min_r_tmp
                else:
                    min_r += min_r_tmp

            min_r /= len(self.theta)
            min_r = min_r.interp(rho_poloidal=self.rho.values, method="cubic")
            min_r = bin_in_time_dt(self.tstart, self.tend, self.dt, min_r,).interp(
                t=self.time, method="linear"
            )
            self.min_r = min_r

            # TODO: fix volume and area for all devices to get the value from equilibrium
            # dpsin = self.equilibrium.psin[1] - self.equilibrium.psin[0]
            # volume = (self.equilibrium.vjac * dpsin).cumsum("rho_poloidal")
            volume, area, _ = self.equilibrium.enclosed_volume(self.rho)
            volume = bin_in_time_dt(self.tstart, self.tend, self.dt, volume,).interp(
                t=self.time, method="linear"
            )
            area = bin_in_time_dt(self.tstart, self.tend, self.dt, area,).interp(
                t=self.time, method="linear"
            )
            self.area.values = area.values
            self.volume.values = volume.values

            self.r_a.values = self.min_r.sel(rho_poloidal=1.0)
            self.r_b.values = self.r_a.values
            self.r_c.values = self.r_a.values
            self.r_d.values = self.r_a.values
            self.kappa.values = (self.r_b / self.r_a).values
            self.delta.values = ((self.r_c + self.r_d) / (2 * self.r_a)).values

            self.maj_r_lfs = bin_in_time_dt(
                self.tstart,
                self.tend,
                self.dt,
                self.equilibrium.rmjo.interp(rho_poloidal=self.rho),
            )
            self.maj_r_hfs = bin_in_time_dt(
                self.tstart,
                self.tend,
                self.dt,
                self.equilibrium.rmji.interp(rho_poloidal=self.rho),
            )

    def build_atomic_data(self, adf11: dict = None):
        fz, fract_abu, power_loss_tot, power_loss_sxr = {}, {}, {}, {}
        for elem in self.elements:
            if adf11 is None:
                adf11 = ADF11

            scd = self.ADASReader.get_adf11("scd", elem, adf11[elem]["scd"])
            acd = self.ADASReader.get_adf11("acd", elem, adf11[elem]["acd"])
            ccd = self.ADASReader.get_adf11("ccd", elem, adf11[elem]["ccd"])
            fract_abu[elem] = FractionalAbundance(scd, acd, CCD=ccd)

            plt = self.ADASReader.get_adf11("plt", elem, adf11[elem]["plt"])
            prb = self.ADASReader.get_adf11("prb", elem, adf11[elem]["prb"])
            prc = self.ADASReader.get_adf11("prc", elem, adf11[elem]["prc"])
            power_loss_tot[elem] = PowerLoss(plt, prb, PRC=prc)

            pls = self.ADASReader.get_adf11("pls", elem, adf11[elem]["pls"])
            prs = self.ADASReader.get_adf11("prs", elem, adf11[elem]["prs"])
            power_loss_sxr[elem] = PowerLoss(pls, prs)

            ion_charges = np.arange(len(fract_abu[elem].SCD.ion_charges) + 1)
            fz[elem] = DataArray(
                np.full((len(self.t), len(self.rho), len(ion_charges)), np.nan),
                coords=[
                    ("t", self.t),
                    ("rho_poloidal", self.rho),
                    ("ion_charges", ion_charges),
                ],
            )
        # self.data3d_fz = fz
        self.adf11 = adf11
        self.fract_abu = fract_abu
        self.power_loss_tot = power_loss_tot
        self.power_loss_sxr = power_loss_sxr

    def set_neutral_density(self, y0=1.0e10, y1=1.0e15, decay=12):
        self.Nh_prof.y0 = y0
        self.Nh_prof.y1 = y1
        self.Nh_prof.yend = y1
        self.Nh_prof.wped = decay
        self.Nh_prof.build_profile()
        for t in self.t:
            self.neutral_dens.loc[dict(t=t)] = self.Nh_prof.yspl.values

    def interferometer(self, data, bckc={}, diagnostic=None, quantity=None):
        """
        Calculate expected diagnostic measurement given plasma profile
        """

        if diagnostic is None:
            diagnostic = ["nirh1", "nirh1_bin", "smmh1"]
        if quantity is None:
            quantity = ["ne"]
        diagnostic = list(diagnostic)
        quantity = list(quantity)

        for diag in diagnostic:
            if diag not in data.keys():
                continue

            for quant in quantity:
                if quant not in data[diag].keys():
                    continue

                bckc = initialize_bckc(diag, quant, data, bckc=bckc)

                bckc[diag][quant].values = self.calc_ne_los_int(
                    data[diag][quant]
                ).values

        return bckc

    def map_to_midplane(self):
        keys = [
            "el_dens",
            "ion_dens",
            "neutral_dens",
            "el_temp",
            "ion_temp",
            "pressure_th",
            "vtor",
            "zeff",
            "meanz",
            "volume",
        ]

        nchan = len(self.R_midplane)
        chan = np.arange(nchan)
        R = DataArray(self.R_midplane, coords=[("channel", chan)])
        z = DataArray(self.z_midplane, coords=[("channel", chan)])

        midplane_profs = {}
        for k in keys:
            k_hi = f"{k}_hi"
            k_lo = f"{k}_lo"

            midplane_profs[k] = []
            if hasattr(self, k_hi):
                midplane_profs[k_hi] = []
            if hasattr(self, k_lo):
                midplane_profs[k_lo] = []

        for k in midplane_profs.keys():
            for t in self.t:
                rho = (
                    self.equilibrium.rho.sel(t=t, method="nearest")
                    .interp(R=R, z=z)
                    .drop(["R", "z"])
                )
                midplane_profs[k].append(
                    getattr(self, k)
                    .sel(t=t, method="nearest")
                    .interp(rho_poloidal=rho)
                    .drop("rho_poloidal")
                )
            midplane_profs[k] = xr.concat(midplane_profs[k], "t").assign_coords(
                t=self.t
            )
            midplane_profs[k] = xr.where(
                np.isfinite(midplane_profs[k]), midplane_profs[k], 0.0
            )

        self.midplane_profs = midplane_profs

    def bremsstrahlung(
        self,
        data,
        bckc={},
        diagnostic="lines",
        quantity="brems",
        wavelength=532.0,
        cal=2.5e-5,
    ):
        """
        Estimate back-calculated Bremsstrahlung measurement from plasma quantities

        Parameters
        ----------
        data
            diagnostic data as returned by self.build_data()
        bckc
            back-calculated data
        diagnostic
            name of diagnostic usef for bremsstrahlung measurement
        quantity
            Measurement to be used for the bremsstrahlung
        wavelength
            Wavelength of measurement
        cal
            Calibration factor for measurement
            Default value calculated to match Zeff before Ar puff from
            LINES.BREMS_MP for pulse 9408

        Returns
        -------
        bckc
            dictionary with back calculated value(s)

        """
        zeff = self.zeff
        brems = ph.zeff_bremsstrahlung(
            self.el_temp, self.el_dens, wavelength, zeff=zeff.sum("element")
        )
        if diagnostic in data.keys():
            if quantity in data[diagnostic].keys():
                bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)

                bckc[diagnostic][quantity].values = self.calc_los_int(
                    data[diagnostic][quantity], brems * cal
                ).values

        bckc[diagnostic][quantity].attrs["calibration"] = cal
        return bckc

    def match_xrcs_temperatures(
        self,
        data,
        bckc={},
        diagnostic: str = "xrcs",
        quantity_te="te_n3w",
        quantity_ti="ti_w",
        half_los=True,
        use_ratios=False,
        use_satellites=False,
        time=None,
        calc_error=False,
        use_ref=True,
        wcenter_exp=0.05,
        method="dogbox",
    ):
        """
        Rescale temperature profiles to match the XRCS spectrometer measurements

        Parameters
        ----------
        data
            diagnostic data as returned by self.build_data()
        bckc
            back-calculated data
        diagnostic
            diagnostic name corresponding to xrcs
        quantity_te
            Measurement to be used for the electron temperature optimisation
        quantity_ti
            Measurement to be used for the ion temperature optimisation
        niter
            Number of iterations

        Returns
        -------

        """

        def line_ratios(forward_model, quantity_ratio):
            if quantity_ratio == "int_k/int_w":
                ratio_bckc = forward_model.intensity["k"] / forward_model.intensity["w"]
            elif quantity_ratio == "int_n3/int_w":
                ratio_bckc = (
                    forward_model.intensity["n3"] / forward_model.intensity["w"]
                )
            elif quantity_ratio == "int_n3/int_tot":
                int_tot = (
                    forward_model.intensity["n3"]
                    + forward_model.intensity["n345"]
                    + forward_model.intensity["w"]
                )
                ratio_bckc = forward_model.intensity["n3"] / int_tot

            return ratio_bckc

        def residuals_te_ratio(te0):
            """
            Optimisation for line ratios
            """
            Te_prof.y0 = te0
            Te_prof.build_profile()
            Te = Te_prof.yspl
            _ = forward_model(
                Te,
                Ne,
                Nimp=Nimp,
                Nh=Nh,
                rho_los=rho_los,
                dl=dl,
                use_satellites=use_satellites,
            )
            ratio_bckc = line_ratios(forward_model, quantity_ratio)
            resid = ratio_data - ratio_bckc
            return resid

        def residuals_te(te0):
            Te_prof.y0 = te0
            Te_prof.build_profile()
            Te = Te_prof.yspl
            Ti = Ti_prof.yspl
            _bckc = forward_model(
                Te,
                Ne,
                Nimp=Nimp,
                Nh=Nh,
                Ti=Ti,
                rho_los=rho_los,
                dl=dl,
                use_satellites=use_satellites,
            )
            te_bckc = _bckc[quantity_te]
            resid = te_data - te_bckc
            return resid

        def residuals_ti(ti0):
            Ti_prof.y0 = ti0
            if use_ref:
                Ti_prof.build_profile(y0_ref=Te_prof.y0, wcenter_exp=wcenter_exp)
            else:
                Ti_prof.build_profile()
            Ti = Ti_prof.yspl

            _bckc = forward_model.moment_analysis(
                Ti,
                rho_los=rho_los,
                dl=dl,
                half_los=half_los,
                use_satellites=use_satellites,
            )
            ti_bckc = _bckc[quantity_ti]
            resid = ti_data - ti_bckc
            return resid

        if diagnostic not in data.keys():
            print_like(f"No {diagnostic.upper()} data available")
            return

        print_like(
            f"Re-calculating temperature profiles to match {diagnostic.upper()} values"
        )

        if time is None:
            time = self.t

        if diagnostic not in bckc:
            bckc[diagnostic] = {}

        if quantity_te == "te_kw":
            quantity_ratio = "int_k/int_w"
        elif quantity_te == "te_n3w":
            quantity_ratio = "int_n3/int_tot"
            # quantity_ratio = "int_n3/int_w"
        else:
            print_like(f"{quantity_te} not available for ratio calculation")
            raise ValueError

        bckc = initialize_bckc(diagnostic, quantity_te, data, bckc=bckc)
        bckc = initialize_bckc(diagnostic, quantity_ti, data, bckc=bckc)
        bckc = initialize_bckc(diagnostic, quantity_ratio, data, bckc=bckc)

        # Initialize back calculated values of diagnostic quantities
        forward_model = self.forward_models[diagnostic]
        Te_prof = self.Te_prof
        Ti_prof = self.Ti_prof

        pos = xr.full_like(data[diagnostic][quantity_te].t, np.nan)
        err_in = deepcopy(pos)
        err_out = deepcopy(pos)
        te_pos = Dataset({"value": pos, "err_in": err_in, "err_out": err_out})
        ti_pos = deepcopy(te_pos)
        bounds_te = (100.0, 10.0e3)
        bounds_ti = (100.0, 30.0e3)
        if method == "lm":
            bounds_te = (-np.inf, np.inf)
            bounds_ti = (-np.inf, np.inf)

        # Initialize variables
        emiss_tmp: list = []
        fz_tmp: list = []
        forward_model.radiation_characteristics(self.Te_prof.yspl, self.Ne_prof.yspl)
        for t in self.t:
            emiss_tmp.append(forward_model.emiss["w"] * 0.0)
            fz_tmp.append(forward_model.fz["ar"] * 0.0)
        emiss = xr.concat(emiss_tmp, "t").assign_coords(t=self.t)
        fz = xr.concat(fz_tmp, "t").assign_coords(t=self.t)

        dl = data[diagnostic][quantity_ti].attrs["dl"]

        if calc_error:
            self.el_temp_hi = deepcopy(self.el_temp)
            self.el_temp_lo = deepcopy(self.el_temp)
            self.ion_temp_hi = deepcopy(self.ion_temp)
            self.ion_temp_lo = deepcopy(self.ion_temp)

        for t in time:
            print(t)
            Ne = self.el_dens.sel(t=t)
            Nimp = {"ar": self.ion_dens.sel(element="ar").sel(t=t)}
            Nh = self.neutral_dens.sel(t=t)

            if use_ratios:
                ratio_data = data[diagnostic][quantity_ratio].sel(t=t)
            te_data = deepcopy(data[diagnostic][quantity_te].sel(t=t))
            ti_data = deepcopy(data[diagnostic][quantity_ti].sel(t=t))
            rho_los = data[diagnostic][quantity_ti].attrs["rho"].sel(t=t)

            if t == time[0]:
                te0 = te_data.values
                ti0 = ti_data.values
            else:
                te0 = Te_prof.y0
                ti0 = Ti_prof.y0

            if not ((te0 > 0) * (ti0 > 0)):
                self.el_temp.loc[dict(t=t)] = np.full_like(Te_prof.yspl.values, np.nan)
                for elem in self.elements:
                    self.ion_temp.loc[dict(t=t, element=elem)] = np.full_like(
                        Te_prof.yspl.values, np.nan
                    )
                continue

            if use_ratios:
                if calc_error:
                    ratio_tmp = deepcopy(ratio_data)

                    # Upper limit of the ratio
                    ratio_data = ratio_tmp + ratio_tmp.attrs["error"]
                    least_squares(
                        residuals_te_ratio, te0, bounds=bounds_te, method=method
                    )
                    least_squares(residuals_ti, ti0, bounds=bounds_ti, method=method)

                    Te_lo = deepcopy(Te_prof)
                    Ti_lo = deepcopy(Ti_prof)

                    # Lower limit of the ratio
                    ratio_data = ratio_tmp - ratio_tmp.attrs["error"]
                    least_squares(
                        residuals_te_ratio, te0, bounds=bounds_te, method=method
                    )
                    least_squares(residuals_ti, ti0, bounds=bounds_ti, method=method)
                    Te_hi = deepcopy(Te_prof)
                    Ti_hi = deepcopy(Ti_prof)

                    te0 = np.mean(Te_hi.y0 + Te_lo.y0) / 2.0
                    ti0 = np.mean(Ti_hi.y0 + Ti_lo.y0) / 2.0
                    ratio_data = ratio_tmp

                _ = least_squares(
                    residuals_te_ratio, te0, bounds=bounds_te, method=method
                )
                least_squares(residuals_ti, ti0, bounds=bounds_ti, method=method)
            else:
                if calc_error:
                    print_like(
                        """Error calculation currently available
                        only for ratio optimisation"""
                    )
                    raise ValueError

                least_squares(residuals_te, te0, bounds=bounds_te, method=method)
                least_squares(residuals_ti, ti0, bounds=bounds_ti, method=method)

            Te = Te_prof.yspl
            Ti = Ti_prof.yspl
            _bckc = forward_model(
                Te,
                Ne,
                Nimp=Nimp,
                Nh=Nh,
                Ti=Ti,
                rho_los=rho_los,
                dl=dl,
                use_satellites=use_satellites,
            )
            if use_ratios:
                ratio_bckc = line_ratios(forward_model, quantity_ratio)
            te_bckc = _bckc[quantity_te]
            ti_bckc = _bckc[quantity_ti]

            te_pos.value.loc[dict(t=t)] = te_bckc.rho_poloidal.values[0]
            te_pos.err_in.loc[dict(t=t)] = te_bckc.attrs["rho_poloidal_err"]["in"]
            te_pos.err_out.loc[dict(t=t)] = te_bckc.attrs["rho_poloidal_err"]["out"]

            ti_pos.value.loc[dict(t=t)] = ti_bckc.rho_poloidal.values[0]
            ti_pos.err_in.loc[dict(t=t)] = ti_bckc.attrs["rho_poloidal_err"]["in"]
            ti_pos.err_out.loc[dict(t=t)] = ti_bckc.attrs["rho_poloidal_err"]["out"]

            emiss.loc[dict(t=t)] = forward_model.emiss["w"].values
            fz.loc[dict(t=t)] = forward_model.fz["ar"].values

            bckc[diagnostic][quantity_te].loc[dict(t=t)] = te_bckc.values[0]
            bckc[diagnostic][quantity_ti].loc[dict(t=t)] = ti_bckc.values[0]
            if use_ratios:
                bckc[diagnostic][quantity_ratio].loc[dict(t=t)] = ratio_bckc.values

            self.el_temp.loc[dict(t=t)] = Te_prof.yspl.values
            if calc_error:
                self.el_temp_hi.loc[dict(t=t)] = Te_hi.yspl.values
                self.el_temp_lo.loc[dict(t=t)] = Te_lo.yspl.values
            for elem in self.elements:
                self.ion_temp.loc[dict(t=t, element=elem)] = Ti_prof.yspl.values
                if calc_error:
                    self.ion_temp_hi.loc[dict(t=t, element=elem)] = Ti_hi.yspl.values
                    self.ion_temp_lo.loc[dict(t=t, element=elem)] = Ti_lo.yspl.values

        bckc[diagnostic][quantity_te].attrs["pos"] = te_pos
        bckc[diagnostic][quantity_te].attrs["emiss"] = emiss
        bckc[diagnostic][quantity_te].attrs["fz"] = fz

        bckc[diagnostic][quantity_ti].attrs["pos"] = ti_pos
        bckc[diagnostic][quantity_ti].attrs["emiss"] = emiss
        bckc[diagnostic][quantity_ti].attrs["fz"] = fz

        revision = get_prov_attribute(
            data[diagnostic][quantity_te].provenance, "revision"
        )
        self.optimisation["el_temp"] = f"{diagnostic}.{quantity_te}:{revision}"

        revision = get_prov_attribute(
            data[diagnostic][quantity_ti].provenance, "revision"
        )
        self.optimisation["ion_temp"] = f"{diagnostic}.{quantity_ti}:{revision}"

        return bckc

    def match_xrcs_intensity(
        self,
        data,
        bckc={},
        diagnostic: str = "xrcs",
        quantity: str = "int_w",
        elem="ar",
        cal=1.0e3,
        niter=2,
        time=None,
        scale=True,
    ):
        """
        TODO: separate calculation of line intensity from optimisation
        TODO: tau currently not included in calculation
        Compute Ar density to match the XRCS spectrometer measurements

        Parameters
        ----------
        data
            diagnostic data as returned by self.build_data()
        bckc
            back-calculated data
        diagnostic
            diagnostic name corresponding to xrcs
        quantity_int
            Measurement to be used for determining the impurity concentration
            from line intensity
        cal
            Calibration factor for measurement
            Default value calculated to match Zeff from LINES.BREMS_MP for pulse 9408
        elem
            Element responsible for measured spectra
        niter
            Number of iterations

        Returns
        -------

        """

        if diagnostic not in data.keys():
            print_like(f"No {diagnostic.upper()} data available")
            return

        print_like(
            f"Re-calculating Ar density profiles to match {diagnostic.upper()} values"
        )

        if time is None:
            time = self.t

        if diagnostic not in bckc:
            bckc[diagnostic] = {}
        if quantity not in bckc[diagnostic].keys():
            bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)
        line = quantity.split("_")[1]

        # Initialize back calculated values of diagnostic quantities
        forward_model = self.forward_models[diagnostic]
        dl = data[diagnostic][quantity].attrs["dl"]
        for t in time:
            print(t)

            int_data = data[diagnostic][quantity].sel(t=t)
            Te = self.el_temp.sel(t=t)
            if np.isnan(Te).any():
                continue

            Ne = self.el_dens.sel(t=t)
            # tau = self.tau.sel(t=t)
            Nh = self.neutral_dens.sel(t=t)
            if np.isnan(Te).any():
                continue
            rho_los = data[diagnostic][quantity].attrs["rho"].sel(t=t)

            const = 1.0
            for j in range(niter):
                Nimp = {elem: self.ion_dens.sel(element=elem, t=t) * const}
                _ = forward_model(Te, Ne, Nimp=Nimp, Nh=Nh, rho_los=rho_los, dl=dl,)
                int_bckc = forward_model.intensity[line] * cal
                const = (int_data / int_bckc).values

                if (np.abs(1 - const) < 1.0e-4) or not (scale):
                    break

            self.ion_dens.loc[dict(element=elem, t=t)] = Nimp[elem].values
            bckc[diagnostic][quantity].loc[dict(t=t)] = int_bckc.values

        bckc[diagnostic][quantity].attrs["calibration"] = cal

        revision = get_prov_attribute(data[diagnostic][quantity].provenance, "revision")
        self.optimisation["imp_dens"] = f"{diagnostic}.{quantity}:{revision}"

        return bckc

    def match_interferometer(
        self,
        data,
        bckc={},
        diagnostic: str = "nirh1",
        quantity: str = "ne_bin",
        error=False,
        niter=3,
        time=None,
    ):
        """
        Rescale density profiles to match the interferometer measurements

        Parameters
        ----------
        data
            diagnostic data as returned by self.build_data()
        bckc
            back calculated data dictionary
        interf
            Name of interferometer to be used

        Returns
        -------

        """
        print_like(
            f"Re-calculating density profiles to match {diagnostic}.{quantity} values"
        )

        if time is None:
            time = self.t

        # TODO: make more elegant optimisation

        bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)

        Ne_prof = self.Ne_prof
        for t in time:
            const = 1.0
            for j in range(niter):
                ne0 = Ne_prof.yspl.sel(rho_poloidal=0) * const
                ne0 = xr.where((ne0 <= 0) or (not np.isfinite(ne0)), 5.0e19, ne0)
                Ne_prof.y0 = ne0.values
                Ne_prof.build_profile()
                self.el_dens.loc[dict(t=t)] = Ne_prof.yspl.values
                ne_bckc = self.calc_ne_los_int(data[diagnostic][quantity], t=t)
                const = (data[diagnostic][quantity].sel(t=t) / ne_bckc).values

            bckc[diagnostic][quantity].loc[dict(t=t)] = ne_bckc.values

        revision = get_prov_attribute(data[diagnostic][quantity].provenance, "revision")
        self.optimisation["el_dens"] = f"{diagnostic}.{quantity}:{revision}"
        self.optimisation["stored_en"] = ""

        return bckc

    def recover_density(
        self, data, diagnostic: str = "efit", quantity: str = "wp", niter=3,
    ):
        """
        Match stored energy by adapting electron density

        Parameters
        ----------
        data
        diagnostic
        quantity
        niter

        Returns
        -------

        """
        print("\n Re-calculating density to match plasma energy \n")

        Ne_prof = self.Ne_prof
        const = DataArray([1.0] * len(self.t), coords=[("t", self.t)])
        ne0 = self.el_dens.sel(rho_poloidal=0)
        data_tmp = data[diagnostic][quantity]
        for j in range(niter):
            for t in self.t:
                if np.isfinite(const.sel(t=t)):
                    Ne_prof.y0 = (ne0 * const).sel(t=t).values
                    Ne_prof.build_profile()
                    self.el_dens.loc[dict(t=t)] = Ne_prof.yspl.values
            self.calc_imp_dens()
            self.calc_main_ion_dens()
            # self.calc_zeff()
            self.calc_pressure()
            bckc_tmp = self.wp.sel()
            const = 1 + (data_tmp - bckc_tmp) / bckc_tmp

        revision = get_prov_attribute(data[diagnostic][quantity].provenance, "revision")
        self.optimisation["stored_en"] = f"{diagnostic}.{quantity}:{revision}"

    def calc_ne_los_int(self, data, passes=2, t=None):
        """
        Calculate line of sight integral for a specified number of
        passes through the plasma

        Returns
        -------
        los_int
            Integral along the line of sight

        """
        dl = data.attrs["dl"]
        rho = data.attrs["rho"]
        transform = data.attrs["transform"]

        x2_name = transform.x2_name

        el_dens = self.el_dens.interp(rho_poloidal=rho)
        if t is not None:
            el_dens = el_dens.sel(t=t, method="nearest")
            rho = rho.sel(t=t, method="nearest")
        el_dens = xr.where(rho <= 1, el_dens, 0,)
        el_dens_int = passes * el_dens.sum(x2_name) * dl

        return el_dens_int

    def calc_los_int(self, data, profile, t=None):
        """
        Calculate line of sight integral of quantity saved as attr in class

        Parameters
        ----------
        data
            raw data with LOS information of specified diagnostic
        quantity
            Quantity to be integrated along the LOS
        t
            Time

        Returns
        -------
        Integral along the line of sight

        """
        dl = data.attrs["dl"]
        rho = data.attrs["rho"]
        transform = data.attrs["transform"]

        x2_name = transform.x2_name

        value = profile.interp(rho_poloidal=rho)
        if t is not None:
            value = value.sel(t=t)
            rho = rho.sel(t=t)
        value = xr.where(rho <= 1, value, 0,)
        los_int = value.sum(x2_name) * dl

        return los_int

    def calc_imp_dens(self, time=None):
        """
        Calculate impurity density from concentration
        """

        ion_dens = self._ion_dens

        if time is None:
            time = self.t
        imp_prof = self.Nimp_prof.yspl / self.Nimp_prof.yspl.sel(rho_poloidal=0)
        for elem in self.impurities:
            imp_dens_0 = self.el_dens.sel(rho_poloidal=0) * self.imp_conc.sel(
                element=elem
            )
            for t in time:
                self.imp_dens.loc[dict(element=elem, t=t)] = (
                    imp_prof * imp_dens_0.sel(t=t).values
                )

    def calc_centrifugal_asymmetry(self, time=None, test_vtor=None, plot=False):
        """
        Calculate (R, z) maps of the ion densities caused by centrifugal asymmetry
        """
        if time is None:
            time = self.t

        # TODO: make this attribute creation a property and standardize?
        if not hasattr(self, "ion_dens_2d"):
            self.rho_2d = self.equilibrium.rho.interp(t=self.t, method="nearest")
            tmp = deepcopy(self.rho_2d)
            ion_dens_2d = []
            for elem in self.elements:
                ion_dens_2d.append(tmp)

            self.ion_dens_2d = xr.concat(ion_dens_2d, "element").assign_coords(
                element=self.elements
            )
            assign_data(self.ion_dens_2d, ("density", "ion"))
            self.centrifugal_asymmetry = deepcopy(self.ion_dens)
            assign_data(self.centrifugal_asymmetry, ("asymmetry", "centrifugal"))
            self.asymmetry_multiplier = deepcopy(self.ion_dens_2d)
            assign_data(
                self.asymmetry_multiplier, ("asymmetry_multiplier", "centrifugal")
            )

        # If toroidal rotation != 0 calculate ion density on 2D poloidal plane
        if test_vtor is not None:
            vtor = deepcopy(self.ion_temp)
            assign_data(vtor, ("rotation", "toroidal"), "rad/s")
            vtor /= vtor.max("rho_poloidal")
            vtor *= test_vtor  # rad/s
            self.vtor = vtor

        if not np.any(self.vtor != 0):
            return

        zeff = self.zeff.sum("element")
        R_0 = self.maj_r_lfs.interp(rho_poloidal=self.rho_2d).drop("rho_poloidal")
        for elem in self.elements:
            main_ion_mass = ELEMENTS[self.main_ion][1]
            mass = ELEMENTS[elem][1]
            asymm = ph.centrifugal_asymmetry(
                self.ion_temp.sel(element=elem).drop("element"),
                self.el_temp,
                mass,
                self.meanz.sel(element=elem).drop("element"),
                zeff,
                main_ion_mass,
                toroidal_rotation=self.vtor.sel(element=elem).drop("element"),
            )
            self.centrifugal_asymmetry.loc[dict(element=elem)] = asymm
            asymmetry_factor = asymm.interp(rho_poloidal=self.rho_2d)
            self.asymmetry_multiplier.loc[dict(element=elem)] = np.exp(
                asymmetry_factor * (self.rho_2d.R ** 2 - R_0 ** 2)
            )

        self.ion_dens_2d = (
            self.ion_dens.interp(rho_poloidal=self.rho_2d).drop("rho_poloidal")
            * self.asymmetry_multiplier
        )
        assign_data(self.ion_dens_2d, ("density", "ion"), "m^-3")

        if plot:
            t = self.t[6]
        for elem in self.elements:
            plt.figure()
            z = self.z_mag.sel(t=t)
            rho = self.rho_2d.sel(t=t).sel(z=z, method="nearest")
            plt.plot(
                rho, self.ion_dens_2d.sel(element=elem).sel(t=t, z=z, method="nearest"),
            )
            self.ion_dens.sel(element=elem).sel(t=t).plot(linestyle="dashed")
            plt.title(elem)

        elem = "ar"
        plt.figure()
        np.log(self.ion_dens_2d.sel(element=elem).sel(t=t, method="nearest")).plot()
        self.rho_2d.sel(t=t, method="nearest").plot.contour(levels=10, colors="white")
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
            tot_rad = (
                self.lz_tot[elem].sum("ion_charges")
                * self.el_dens
                * self.ion_dens.sel(element=elem)
            )
            tot_rad = xr.where(tot_rad >= 0, tot_rad, 0.0,)
            self.tot_rad.loc[dict(element=elem)] = tot_rad.values

            sxr_rad = (
                self.lz_sxr[elem].sum("ion_charges")
                * self.el_dens
                * self.ion_dens.sel(element=elem)
            )
            sxr_rad = xr.where(sxr_rad >= 0, sxr_rad, 0.0,)
            self.sxr_rad.loc[dict(element=elem)] = sxr_rad.values

            if not hasattr(self, "prad_tot"):
                self.prad_tot = deepcopy(self.prad)
                self.prad_sxr = deepcopy(self.prad)
                assign_data(self.prad_sxr, ("radiation", "sxr"))

            prad_tot = self.prad_tot.sel(element=elem)
            prad_sxr = self.prad_sxr.sel(element=elem)
            for t in self.t:
                prad_tot.loc[dict(t=t)] = np.trapz(
                    tot_rad.sel(t=t), self.volume.sel(t=t)
                )
                prad_sxr.loc[dict(t=t)] = np.trapz(
                    sxr_rad.sel(t=t), self.volume.sel(t=t)
                )
            self.prad_tot.loc[dict(element=elem)] = prad_tot.values
            self.prad_sxr.loc[dict(element=elem)] = prad_sxr.values

    def impose_flat_zeff(self):
        """
        Adapt impurity concentration to generate flat Zeff contribution
        """

        for elem in self.impurities:
            if np.count_nonzero(self.ion_dens.sel(element=elem)) != 0:
                zeff_tmp = self.zeff.sel(element=elem)
                value = zeff_tmp.where(zeff_tmp.rho_poloidal < 0.2).mean("rho_poloidal")
                zeff_tmp = zeff_tmp / zeff_tmp * value
                ion_dens_tmp = zeff_tmp / (
                    self.meanz.sel(element=elem) ** 2 / self.el_dens
                )
                self.ion_dens.loc[dict(element=elem)] = ion_dens_tmp.values

    def add_transport(self):
        """
        Modify ionization distribution including transport
        """

        x_ped = 0.85
        diffusion = (
            xr.where(
                self.rho < x_ped,
                ph.gaussian(self.rho, 0.2, 0.02, x_ped, 0.3),
                ph.gaussian(self.rho, 0.2, 0.01, x_ped, 0.04),
            )
            * 2
        )

        for elem in self.elements:
            fz = (
                self.atomic_data[elem]["fz"]
                .interp(electron_temperature=self.el_temp, method="cubic")
                .drop_vars(["electron_temperature"])
            )
            fz_transp = deepcopy(fz)
            for t in self.time:
                fz_tmp = fz_transp.sel(t=t, drop=True)
                for i, rho in enumerate(self.rho):
                    gauss = (
                        ph.gaussian(self.rho, diffusion[i], 0, rho, diffusion[i] / 3)
                        * diffusion
                    )
                    gauss /= np.sum(gauss)
                    fz_tmp.loc[dict(rho_poloidal=rho)] = (
                        (fz_tmp * gauss).sum("rho_poloidal").values
                    )
                for ir, rho in enumerate(self.rho):
                    norm = np.nansum(fz_tmp.sel(rho_poloidal=rho), axis=0)
                    fz_tmp.loc[dict(rho_poloidal=rho)] = (fz_tmp / norm).values
                    fz_transp.loc[dict(t=t)] = fz_tmp.values

                plt.figure()
                colors = cm.rainbow(np.linspace(0, 1, len(fz.ion_charges)))
                for i in fz.ion_charges:
                    plt.plot(
                        fz.rho_poloidal,
                        fz.sel(ion_charges=i).sel(t=t),
                        color=colors[i],
                    )
                    plt.plot(
                        fz_transp.rho_poloidal,
                        fz_transp.sel(ion_charges=i).sel(t=t),
                        "--",
                        color=colors[i],
                    )
                plt.title(f"Time = {t}")

    def write_to_pickle(self):

        with open(f"data_{self.pulse}.pkl", "wb") as f:
            pickle.dump(
                self, f,
            )

    def data_changed(self, var_name, dependencies: list):
        var_hash = hash_vals(
            dependencies=[getattr(self, attr) for attr in dependencies]
        )

        if var_name not in self.property_hashes.keys():
            print(f"\n {var_name} property not included in property_hashes...")
            raise ValueError

        if self.property_hashes[var_name] != var_hash:
            self.property_hashes[var_name] = var_hash
            recalculate = True
        else:
            recalculate = False

        return recalculate


def initialize_bckc(diagnostic, quantity, data, bckc={}):
    """
    Initialise back-calculated data with all info as original data, apart
    from provenance and revision attributes

    Parameters
    ----------
    data
        DataArray of original data to be "cloned"

    Returns
    -------

    """
    if diagnostic not in bckc:
        bckc[diagnostic] = {}

    data_tmp = data[diagnostic][quantity]
    bckc_tmp = xr.full_like(data_tmp, np.nan)
    attrs = bckc_tmp.attrs
    if type(bckc_tmp) == DataArray:
        if "error" in attrs.keys():
            attrs["error"] = xr.full_like(attrs["error"], np.nan)
        if "partial_provenance" in attrs.keys():
            attrs.pop("partial_provenance")
            attrs.pop("provenance")
    bckc_tmp.attrs = attrs

    bckc[diagnostic][quantity] = bckc_tmp

    return bckc


def remap_diagnostic(diag_data, flux_transform, npts=100):
    """
    Calculate maping on equilibrium for speccified diagnostic

    Returns
    -------

    """
    new_attrs = {}
    trans = diag_data.attrs["transform"]
    x1 = diag_data.coords[trans.x1_name]
    x2_arr = np.linspace(0, 1, npts)
    x2 = DataArray(x2_arr, dims=trans.x2_name)
    dl = trans.distance(trans.x2_name, DataArray(0), x2[0:2], 0)[1]
    new_attrs["x2"] = x2
    new_attrs["dl"] = dl
    new_attrs["R"], new_attrs["z"] = trans.convert_to_Rz(x1, x2, 0)

    dt_equil = flux_transform.equilibrium.rho.t[1] - flux_transform.equilibrium.rho.t[0]
    dt_data = diag_data.t[1] - diag_data.t[0]
    if dt_data > dt_equil:
        t = diag_data.t
    else:
        t = None
    rho_equil, _ = flux_transform.convert_from_Rz(new_attrs["R"], new_attrs["z"], t=t)
    rho = rho_equil.interp(t=diag_data.t, method="linear")
    rho = xr.where(rho >= 0, rho, 0.0)
    rho.coords[trans.x2_name] = x2
    new_attrs["rho"] = rho

    return new_attrs


def assign_data(data: DataArray, datatype: tuple, unit="", _deepcopy = True):
    if _deepcopy:
        new_data = deepcopy(data)
    else:
        new_data = data

    new_data.name = f"{datatype[1]}_{datatype[0]}"
    new_data.attrs["datatype"] = datatype
    if len(unit) > 0:
        new_data.attrs["unit"] = unit

    return new_data


def print_like(string):
    print(f"\n {string}")


def test_observabels():
    pl = Plasma()

    profs = profiles.profile_scans(rho=pl.rho)
    pl.Ne_prof = profs["Ne"]["peaked"]
    pl.Te_prof = profs["Te"]["peaked"]
    pl.Ti_prof = profs["Ti"]["peaked"]
    pl.Nimp_prof = profs["Nimp"]["peaked"]
    pl.Vrot_prof = profs["Vrot"]["peaked"]

    pl.imp_conc.values = [0.03, 0.001]
    for t in pl.t:
        pl.el_temp.loc[dict(t=t)] = pl.Te_prof.yspl
        pl.el_dens.loc[dict(t=t)] = pl.Ne_prof.yspl
        for elem in pl.elements:
            pl.ion_temp.loc[dict(t=t, element=elem)] = pl.Te_prof.yspl
            if elem in pl.impurities:
                pl.imp_dens.loc[
                    dict(t=t, element=elem)
                ] = pl.Ne_prof.yspl * pl.imp_conc.sel(element=elem)
    pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
    # pl.build_atomic_data()

    return pl


def get_func_name():
    return str(inspect.stack()[1][3])
