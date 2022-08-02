from copy import deepcopy
import pickle

import hda.physics as ph
from hda.profiles import Profiles
from hda.utils import assign_data, print_like
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters import FluxSurfaceCoordinates
from indica.converters.time import bin_in_time_dt
from indica.converters.time import get_tlabels_dt
from indica.datatypes import ELEMENTS
from indica.equilibrium import Equilibrium
from indica.provenance import get_prov_attribute
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader

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
        impurities=("c", "ar"),
        main_ion="h",
        imp_conc=(0.02, 0.001),
    ):
        """

        Parameters
        ----------
        pulse

        """

        self.ADASReader = ADASReader()
        self.main_ion = main_ion
        self.impurities = impurities
        self.elements = [self.main_ion]
        for elem in self.impurities:
            self.elements.append(elem)
        self.imp_conc = assign_data(
            DataArray(np.array(imp_conc), coords=[("element", list(self.impurities))]),
            ("concentration", "impurity"),
        )

        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.t = get_tlabels_dt(self.tstart, self.tend, self.dt)
        self.theta = np.linspace(0, 2 * np.pi, ntheta + 1)[:-1]
        self.radial_coordinate = np.linspace(0, 1.0, 41)
        self.radial_coordinate_type = "rho_poloidal"
        self.machine_dimensions = machine_dimensions

        self.forward_models = {}

        self.build_atomic_data(ADF11)
        self.initialize_variables()

    def set_equilibrium(self, data: dict, diagnostic: str):
        """
        Assign equilibrium and flux surface transform objects, calculate geometry parameters

        Parameters
        ----------
        data
            Equilibrium data dictionary as returned by abstractreader.py
        diagnostic
            Identifier of equilibrium diagnostic/code used

        Returns
        -------

        """
        equil_keys = list(data.keys())
        revision = get_prov_attribute(data[equil_keys[0]].provenance, "revision")
        self.optimisation["equil"] = f"{diagnostic}:{revision}"

        self.equilibrium = Equilibrium(data)
        self.flux_coords = FluxSurfaceCoordinates("poloidal")
        self.flux_coords.set_equilibrium(self.equilibrium)

        self.calculate_geometry()

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

        self.optimisation = {}
        self.pulse = None
        self.freq = 1.0 / self.dt

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

        self.theta = assign_data(data1d_theta, ("angle", "poloidal"), "deg")
        self.ipla = assign_data(data1d_time, ("current", "plasma"), "A")
        # self.bt_0 = assign_data(data1d_time, ("field", "toroidal"), "T")
        # self.R_bt_0 = assign_data(data0d, ("major_radius", "toroidal_field"), "T")
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
        self.min_r = assign_data(
            data2d, ("minor_radius", "plasma")
        )  # LFS-HFS averaged value
        self.volume = assign_data(data2d, ("volume", "plasma"))
        self.area = assign_data(data2d, ("area", "plasma"))
        # self.r_a = assign_data(data1d_time, ("minor_radius", "LFS"))
        # self.r_b = assign_data(data1d_time, ("minor_radius", "top"))
        # self.r_c = assign_data(data1d_time, ("minor_radius", "HFS"))
        # self.r_d = assign_data(data1d_time, ("minor_radius", "bottom"))
        # self.kappa = assign_data(data1d_time, ("elongation", "plasma"))
        # self.delta = assign_data(data1d_time, ("triangularity", "plasma"))
        self.j_phi = assign_data(data2d, ("current", "density"))
        self.b_pol = assign_data(data2d, ("field", "poloidal"))
        self.b_tor_lfs = assign_data(data2d, ("field", "toroidal"))
        self.b_tor_hfs = assign_data(data2d, ("field", "toroidal"))
        self.q_prof = assign_data(data2d, ("factor", "safety"))
        self.conductivity = assign_data(data2d, ("conductivity", "plasma"))
        self.l_i = assign_data(data1d_time, ("inductance", "internal"))

        self.ion_temp = assign_data(data3d, ("temperature", "ion"))
        self.vtor = assign_data(data3d, ("toroidal_rotation", "ion"))
        self.imp_dens = assign_data(data3d_imp, ("density", "impurity"), "m^-3")
        self.fast_temp = assign_data(data2d, ("temperature", "fast"))
        self.fast_dens = assign_data(data2d, ("density", "fast"))

        # Private variables for class property variables
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
        self._vloop = assign_data(data1d_time, ("density", "ion"), "m^-3")
        self._j_phi = assign_data(
            data1d_time, ("toroidal_current", "density"), "A m^-2"
        )
        self._btot = assign_data(data1d_time, ("magnetic_field", "total"), "T")

    @property
    def pressure_el(self):
        self._pressure_el.values = ph.calc_pressure(self.el_dens, self.el_temp)
        return self._pressure_el

    @property
    def pressure_th(self):
        ion_dens = self.ion_dens
        self._pressure_th.values = self.pressure_el
        for elem in self.elements:
            self._pressure_th.values += ph.calc_pressure(
                ion_dens.sel(element=elem).values,
                self.ion_temp.sel(element=elem).values,
            )
        return self._pressure_th

    @property
    def pressure_tot(self):
        self._pressure_tot.values = self.pressure_th + ph.calc_pressure(
            self.fast_dens, self.fast_temp
        )
        return self._pressure_tot

    @property
    def pth(self):
        pressure_th = self.pressure_th
        for t in self.time:
            self._pth.loc[dict(t=t)] = np.trapz(
                pressure_th.sel(t=t), self.volume.sel(t=t)
            )
        return self._pth

    @property
    def ptot(self):
        pressure_tot = self.pressure_tot
        for t in self.time:
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
        for elem in self.elements:
            for t in self.time:
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

    @property
    def zeff(self):
        ion_dens = self.ion_dens
        meanz = self.meanz
        for elem in self.elements:
            self._zeff.loc[dict(element=elem)] = (
                (ion_dens.sel(element=elem) * meanz.sel(element=elem) ** 2)
                / self.el_dens
            ).values
        return self._zeff

    @property
    def ion_dens(self):
        imp_dens = self.imp_dens
        meanz = self.meanz
        main_ion_dens = self.el_dens - self.fast_dens * meanz.sel(element=self.main_ion)
        for elem in self.impurities:
            self._ion_dens.loc[dict(element=elem)] = imp_dens.sel(element=elem).values
            main_ion_dens -= imp_dens.sel(element=elem) * meanz.sel(element=elem)

        self._ion_dens.loc[dict(element=self.main_ion)] = main_ion_dens.values
        return self._ion_dens

    @property
    def meanz(self):
        fz = self.fz
        for elem in self.elements:
            self._meanz.loc[dict(element=elem)] = (
                (fz[elem] * fz[elem].ion_charges).sum("ion_charges").values
            )
        return self._meanz

    @property
    def lz_tot(self):
        fz = self.fz
        for elem in self.elements:
            for t in self.time:
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

    @property
    def lz_sxr(self):
        fz = self.fz
        for elem in self.elements:
            for t in self.time:
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

    @property
    def tot_rad(self):
        lz_tot = self.lz_tot
        ion_dens = self.ion_dens
        for elem in self.elements:
            tot_rad = (
                lz_tot[elem].sum("ion_charges")
                * self.el_dens
                * ion_dens.sel(element=elem)
            )
            self._tot_rad.loc[dict(element=elem)] = xr.where(
                tot_rad >= 0,
                tot_rad,
                0.0,
            ).values
        return self._tot_rad

    @property
    def sxr_rad(self):
        lz_sxr = self.lz_sxr
        ion_dens = self.ion_dens
        for elem in self.elements:
            sxr_rad = (
                lz_sxr[elem].sum("ion_charges")
                * self.el_dens
                * ion_dens.sel(element=elem)
            )
            self._sxr_rad.loc[dict(element=elem)] = xr.where(
                sxr_rad >= 0,
                sxr_rad,
                0.0,
            ).values
        return self._sxr_rad

    @property
    def prad_tot(self):
        tot_rad = self.tot_rad
        for elem in self.elements:
            for t in self.time:
                self._prad_tot.loc[dict(element=elem, t=t)] = np.trapz(
                    tot_rad.sel(element=elem, t=t), self.volume.sel(t=t)
                )
        return self._prad_tot

    @property
    def prad_sxr(self):
        sxr_rad = self.sxr_rad
        for elem in self.elements:
            for t in self.time:
                self._prad_sxr.loc[dict(element=elem, t=t)] = np.trapz(
                    sxr_rad.sel(element=elem, t=t), self.volume.sel(t=t)
                )
        return self._prad_sxr

    @property
    def vloop(self):
        zeff = self.zeff
        j_phi = self.j_phi
        self.conductivity = ph.conductivity_neo(
            self.el_dens,
            self.el_temp,
            zeff.sum("element"),
            self.min_r,
            self.min_r.interp(rho_poloidal=1.0),
            self.R_mag,
            self.q_prof,
            approx="sauter",
        )
        for t in self.t:
            resistivity = 1.0 / self.conductivity.sel(t=t)
            ir = np.where(np.isfinite(resistivity))
            vloop = ph.vloop(
                resistivity[ir], j_phi.sel(t=t)[ir], self.area.sel(t=t)[ir]
            )
            self._vloop.loc[dict(t=t)] = vloop.values
        return self._vloop

    def calc_imp_dens(self, time=None):
        """
        Calculate impurity density from concentration
        """
        if time is None:
            time = self.t
        profile = self.Nimp_prof.yspl / self.Nimp_prof.yspl.sel(rho_poloidal=0)
        for elem in self.impurities:
            dens_0 = self.el_dens.sel(rho_poloidal=0) * self.imp_conc.sel(element=elem)
            for t in time:
                Nimp = profile * dens_0.sel(t=t)
                self.imp_dens.loc[dict(element=elem, t=t)] = Nimp.values

    def impose_flat_zeff(self):
        """
        Adapt impurity concentration to generate flat Zeff contribution
        """

        for elem in self.impurities:
            if np.count_nonzero(self.ion_dens.sel(element=elem)) != 0:
                zeff_tmp = (
                    self.ion_dens.sel(element=elem)
                    * self.meanz.sel(element=elem) ** 2
                    / self.el_dens
                )
                value = zeff_tmp.where(zeff_tmp.rho_poloidal < 0.2).mean("rho_poloidal")
                zeff_tmp = zeff_tmp / zeff_tmp * value
                ion_dens_tmp = zeff_tmp / (
                    self.meanz.sel(element=elem) ** 2 / self.el_dens
                )
                self.ion_dens.loc[dict(element=elem)] = ion_dens_tmp.values

        self.calc_zeff()

    def calculate_geometry(self):
        if hasattr(self, "equilibrium"):
            bin_in_time = self.bin_in_time
            rho = self.rho
            equilibrium = self.equilibrium
            print_like("Calculate geometric quantities")

            self.volume.values = bin_in_time(
                equilibrium.volume.interp(rho_poloidal=rho)
            )
            self.area.values = bin_in_time(equilibrium.area.interp(rho_poloidal=rho))
            self.maj_r_lfs.values = bin_in_time(
                equilibrium.rmjo.interp(rho_poloidal=rho)
            )
            self.maj_r_hfs.values = bin_in_time(
                equilibrium.rmji.interp(rho_poloidal=rho)
            )
            self.R_mag.values = bin_in_time(equilibrium.rmag.interp(rho_poloidal=rho))
            self.z_mag.values = bin_in_time(equilibrium.zmag.interp(rho_poloidal=rho))
            self.min_r.values = (self.maj_r_lfs - self.maj_r_hfs) / 2.0
        else:
            print_like(
                "Plasma class doesn't have equilibrium: skipping geometry assignments..."
            )

    def bin_in_time(self, value: DataArray, method="linear"):
        binned = bin_in_time_dt(
            self.tstart,
            self.tend,
            self.dtdt,
            value,
        ).interp(t=self.time, method=method)

        return binned

    def build_atomic_data(self, adf11: dict = None):
        print_like("Initialize fractional abundance objects")
        fract_abu, power_loss_tot, power_loss_sxr = {}, {}, {}
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

    def map_to_midplane(self):
        # TODO: streamline this to avoid continuously re-calculating quantities e.g. ion_dens..
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
            tot_rad = xr.where(
                tot_rad >= 0,
                tot_rad,
                0.0,
            )
            self.tot_rad.loc[dict(element=elem)] = tot_rad.values

            sxr_rad = (
                self.lz_sxr[elem].sum("ion_charges")
                * self.el_dens
                * self.ion_dens.sel(element=elem)
            )
            sxr_rad = xr.where(
                sxr_rad >= 0,
                sxr_rad,
                0.0,
            )
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
                self,
                f,
            )


def average_runs(plasma_dict: dict):
    runs = list(plasma_dict)
    pl_avrg = deepcopy(plasma_dict[runs[0]])
    el_dens, imp_dens, neutral_dens, el_temp, ion_temp = (
        [],
        [],
        [],
        [],
        [],
    )
    runs = []
    for run_name, pl in plasma_dict.items():
        runs.append(run_name)
        el_dens.append(pl.el_dens)
        imp_dens.append(pl.imp_dens)
        neutral_dens.append(pl.neutral_dens)
        el_temp.append(pl.el_temp)
        ion_temp.append(pl.ion_temp)

    el_dens = xr.concat(el_dens, "run_name").assign_coords({"run_name": runs})
    stdev = el_dens.std("run_name")
    pl_avrg.el_dens = el_dens.mean("run_name")
    pl_avrg.el_dens_hi = pl_avrg.el_dens + stdev
    pl_avrg.el_dens_lo = pl_avrg.el_dens - stdev

    ion_dens = xr.concat(imp_dens, "run_name").assign_coords({"run_name": runs})
    stdev = ion_dens.std("run_name")
    pl_avrg.imp_dens = ion_dens.mean("run_name")
    pl_avrg.imp_dens_hi = pl_avrg.imp_dens + stdev
    pl_avrg.imp_dens_lo = pl_avrg.imp_dens - stdev

    neutral_dens = xr.concat(neutral_dens, "run_name").assign_coords({"run_name": runs})
    stdev = neutral_dens.std("run_name")
    pl_avrg.neutral_dens = neutral_dens.mean("run_name")
    pl_avrg.neutral_dens_hi = pl_avrg.neutral_dens + stdev
    pl_avrg.neutral_dens_lo = pl_avrg.neutral_dens - stdev

    el_temp = xr.concat(el_temp, "run_name").assign_coords({"run_name": runs})
    stdev = el_temp.std("run_name")
    pl_avrg.el_temp = el_temp.mean("run_name")
    pl_avrg.el_temp_hi = pl_avrg.el_temp + stdev
    pl_avrg.el_temp_lo = pl_avrg.el_temp - stdev

    ion_temp = xr.concat(ion_temp, "run_name").assign_coords({"run_name": runs})
    stdev = ion_temp.std("run_name")
    pl_avrg.ion_temp = ion_temp.mean("run_name")
    pl_avrg.ion_temp_hi = pl_avrg.ion_temp + stdev
    pl_avrg.ion_temp_lo = pl_avrg.ion_temp - stdev

    return pl


def apply_limits(
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
