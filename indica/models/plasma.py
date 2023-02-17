from copy import deepcopy
import pickle

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.time import convert_in_time_dt
from indica.converters.time import get_tlabels_dt
from indica.datatypes import ELEMENTS
from indica.equilibrium import Equilibrium
from indica.models.equilibrium import fake_equilibrium_data
from indica.numpy_typing import LabeledArray
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
        self.full_run = full_run
        self.ADASReader = ADASReader()
        self.main_ion = main_ion
        self.impurities = impurities
        self.elements = [self.main_ion]
        for element in self.impurities:
            self.elements.append(element)
        self.impurity_concentration = assign_data(
            DataArray(
                np.array(impurity_concentration),
                coords=[("element", list(self.impurities))],
            ),
            ("concentration", "impurity"),
        )
        self.set_adf11(ADF11)
        self.radial_coordinate = np.linspace(0, 1.0, 41)
        self.radial_coordinate_type = "rho_poloidal"
        self.machine_dimensions = machine_dimensions

        self.build_atomic_data(default=True)

        self.initialize_variables(tstart, tend, dt)

        self.equilibrium: Equilibrium

    def set_equilibrium(self, equilibrium: Equilibrium):
        """
        Assign equilibrium object
        """
        self.equilibrium = equilibrium
        self.calculate_geometry()

    def set_adf11(self, adf11: dict):
        self.adf11 = adf11

    def initialize_variables(self, tstart: float, tend: float, dt: float):
        """
        Initialize all class attributes

        Parameters
        ----------
        tstart
            start time
        tend
            end time
        dt
            time-step

        Description of variables being initialized
        ------------------------------------------
        time_to_calculate
            subset of time-point(s) to use for computation of the dependent variables
            (to be used e.g. in optimisation workflows)
        """
        self.tstart = tstart
        self.tend = tend
        self.dt = dt

        # Dictionary keeping track of deta use for optimisations
        self.optimisation: dict = {}
        self.forward_models: dict = {}

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

        self.t = assign_data(data1d_time, ("t", "plasma"), "s")
        self.t.values = time

        rho_type = self.radial_coordinate_type.split("_")
        if rho_type[1] != "poloidal":
            print_like("Only poloidal rho in input for the time being...")
            raise AssertionError
        self.rho = assign_data(data1d_rho, (rho_type[0], rho_type[1]))
        self.rho.values = self.radial_coordinate

        self.Te_prof = Profiles(datatype=("temperature", "electron"), xspl=self.rho)
        self.Ti_prof = Profiles(datatype=("temperature", "ion"), xspl=self.rho)
        self.Ne_prof = Profiles(datatype=("density", "electron"), xspl=self.rho)
        self.Nimp_prof = Profiles(datatype=("density", "impurity"), xspl=self.rho)
        self.Nh_prof = Profiles(datatype=("density", "thermal_neutrals"), xspl=self.rho)
        self.Vrot_prof = Profiles(datatype=("rotation", "toroidal"), xspl=self.rho)

        self.ipla = assign_data(data1d_time, ("current", "plasma"), "A")
        self.R_0 = assign_data(data1d_time, ("major_radius", "geometric"), "m")
        self.R_mag = assign_data(data1d_time, ("major_radius", "magnetic"), "m")
        self.z_mag = assign_data(data1d_time, ("z", "magnetic"), "m")
        self.maj_r_lfs = assign_data(data2d, ("radius", "major"), "m")
        self.maj_r_hfs = assign_data(data2d, ("radius", "major"), "m")
        self.ne_0 = assign_data(data1d_time, ("density", "electron"), "$m^{-3}$")
        self.te_0 = assign_data(data1d_time, ("temperature", "electron"), "eV")
        self.ti_0 = assign_data(data1d_time, ("temperature", "ion"), "eV")
        self.electron_temperature = assign_data(
            data2d, ("temperature", "electron"), "eV"
        )
        self.electron_density = assign_data(data2d, ("density", "electron"), "$m^{-3}$")
        self.neutral_density = assign_data(data2d, ("density", "neutral"), "eV")
        self.tau = assign_data(data2d, ("time", "residence"), "s")
        self.minor_radius = assign_data(
            data2d, ("minor_radius", "plasma", "m")
        )  # LFS-HFS averaged value
        self.volume = assign_data(data2d, ("volume", "plasma"), "$m^3$")
        self.area = assign_data(data2d, ("area", "plasma"), "$m^2$")
        self.j_phi = assign_data(data2d, ("current", "density"), "A $m^2$")
        self.b_pol = assign_data(data2d, ("field", "poloidal"), "T")
        self.b_tor_lfs = assign_data(data2d, ("field", "toroidal"), "T")
        self.b_tor_hfs = assign_data(data2d, ("field", "toroidal"), "T")
        self.q_prof = assign_data(data2d, ("factor", "safety"), "")
        self.conductivity = assign_data(data2d, ("conductivity", "plasma"), "")
        self.l_i = assign_data(data1d_time, ("inductance", "internal"), "")

        self.ion_temperature = assign_data(data3d, ("temperature", "ion"), "eV")
        self.toroidal_rotation = assign_data(
            data3d, ("toroidal_rotation", "ion"), "rad $s^{-1}$"
        )
        self.impurity_density = assign_data(
            data3d_imp, ("density", "impurity"), "$m^{-3}$"
        )
        self.fast_temperature = assign_data(data2d, ("temperature", "fast"), "eV")
        self.fast_density = assign_data(data2d, ("density", "fast"), "$m^3$")

        # Private variables for class property variables
        self._pressure_th = assign_data(data2d, ("pressure", "thermal"), "Pa $m^{-3}$")
        self._pressure_tot = assign_data(data2d, ("pressure", "total"), "Pa $m^{-3}$")
        self._pth = assign_data(data1d_time, ("pressure_integral", "thermal"), "Pa")
        self._ptot = assign_data(data1d_time, ("pressure_integral", "total"), "Pa")
        self._wth = assign_data(data1d_time, ("stored_energy", "thermal"), "J")
        self._wp = assign_data(data1d_time, ("stored_energy", "total"), "J")
        self._prad_tot = RadiationIntegral(
            assign_data(data2d_elem, ("radiation", "total"), "W")
        )
        self._prad_sxr = RadiationIntegral(
            assign_data(data2d_elem, ("radiation", "sxr"), "W")
        )
        self._ion_density = IonDensity(
            assign_data(data3d, ("density", "ion"), "$m^{-3}$")
        )
        self._zeff = Zeff(assign_data(data3d, ("charge", "effective"), ""))
        self._meanz = Meanz(assign_data(data3d, ("charge", "mean"), ""))
        self._pressure_el = PressureProfile(
            assign_data(data2d, ("pressure", "electron"), "Pa $m^{-3}$")
        )
        self._pressure_ion = PressureProfile(
            assign_data(data3d, ("pressure", "ion"), "Pa $m^{-3}$")
        )
        self._pressure_fast = PressureProfile(
            assign_data(data2d, ("pressure", "fast"), "Pa $m^{-3}$")
        )
        self._pel = PressureIntegral(
            assign_data(data1d_time, ("pressure_integral", "electron"), "Pa")
        )
        self._pion = PressureIntegral(
            assign_data(data1d_time, ("pressure_integral", "ion"), "Pa")
        )
        self._pfast = PressureIntegral(
            assign_data(data1d_time, ("pressure_integral", "fast"), "Pa")
        )
        self._total_radiation = RadiationProfile(
            assign_data(data3d, ("radiation_emission", "total"), "W $m^{-3}$")
        )
        self._sxr_radiation = RadiationProfile(
            assign_data(data3d, ("radiation_emission", "sxr"), "W $m^{-3}$")
        )

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
        self._fz = Fz(_fz)
        self._lz_tot = Lz(_lz_tot)
        self._lz_sxr = Lz(_lz_sxr)

    @property
    def time_to_calculate(self):
        return self._time_to_calculate

    @time_to_calculate.setter
    def time_to_calculate(self, value: LabeledArray):
        if np.size(value) == 1:
            self._time_to_calculate = float(value)
        else:
            self._time_to_calculate = np.array(value)

    @property
    def pressure_el(self):
        return self._pressure_el(self.electron_density, self.electron_temperature)

    @property
    def pressure_ion(self):
        return self._pressure_ion(self.ion_density, self.ion_temperature)

    @property
    def pressure_fast(self):
        return self._pressure_fast(self.fast_density, self.fast_temperature)

    @property
    def pressure_th(self):
        self._pressure_th = self.pressure_el + self.pressure_ion.sum("element")
        return self._pressure_th

    @property
    def pressure_tot(self):
        return self._pressure_th + self._pressure_fast

    @property
    def pel(self):
        return self._pel(self.pressure_el, self.volume)

    @property
    def pion(self):
        return self._pion(self.pressure_ion, self.volume)

    @property
    def pfast(self):
        return self._pfast(self.pressure_fast, self.volume)

    @property
    def pth(self):
        self._pth = self.pel + self.pion.sum("element")
        return self._pth

    @property
    def ptot(self):
        self._ptot = self.pth + self.pfast
        return self._ptot

    @property
    def wth(self):
        self._wth.values = 3 / 2 * self.pth.values
        return self._wth

    @property
    def wp(self):
        self._wp.values = 3 / 2 * self.ptot.values
        return self._wp

    @property
    def fz(self):
        return self._fz(
            self.electron_temperature,
            self.electron_density,
            self.neutral_density,
            self.tau,
            self.fract_abu,
            self.time_to_calculate,
            self.full_run,
        )

    @property
    def lz_tot(self):
        return self._lz_tot(
            self.electron_temperature,
            self.electron_density,
            self.neutral_density,
            self.fz,
            self.power_loss_tot,
            self.time_to_calculate,
            self.full_run,
        )

    @property
    def lz_sxr(self):
        return self._lz_sxr(
            self.electron_temperature,
            self.electron_density,
            self.neutral_density,
            self.fz,
            self.power_loss_sxr,
            self.time_to_calculate,
            self.full_run,
        )

    @property
    def total_radiation(self):
        return self._total_radiation(
            self.lz_tot, self.electron_density, self.ion_density
        )

    @property
    def sxr_radiation(self):
        return self._sxr_radiation(self.lz_sxr, self.electron_density, self.ion_density)

    @property
    def meanz(self):
        return self._meanz(self.fz)

    @property
    def zeff(self):
        return self._zeff(self.electron_density, self.ion_density, self.meanz)

    @property
    def ion_density(self):
        return self._ion_density(
            self.electron_density,
            self.impurity_density,
            self.fast_density,
            self.meanz,
            self.main_ion,
        )

    @property
    def prad_tot(self):
        return self._prad_tot(self.total_radiation, self.volume)

    @property
    def prad_sxr(self):
        return self._prad_tot(self.sxr_radiation, self.volume)

    def calc_impurity_density(self, t=None):
        """
        Calculate impurity density from concentration
        """
        if t is None:
            t = self.t
        if type(t) is not LabeledArray:
            t = [t]

        profile_shape = self.Nimp_prof.yspl / self.Nimp_prof.yspl.sel(rho_poloidal=0)
        for elem in self.impurities:
            conc = self.impurity_concentration.sel(element=elem)
            for _t in t:
                dens_0 = self.electron_density.sel(rho_poloidal=0, t=t) * conc
                Nimp = profile_shape * dens_0.sel(t=_t)
                self.impurity_density.loc[dict(element=elem, t=_t)] = Nimp.values

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

    def calculate_geometry(self):
        if hasattr(self, "equilibrium"):
            rho = self.rho
            equilibrium = self.equilibrium
            # print_like("Calculate geometric quantities")

            self.volume.values = self.convert_in_time(
                equilibrium.volume.interp(rho_poloidal=rho)
            )
            self.area.values = self.convert_in_time(
                equilibrium.area.interp(rho_poloidal=rho)
            )
            self.maj_r_lfs.values = self.convert_in_time(
                equilibrium.rmjo.interp(rho_poloidal=rho)
            )
            self.maj_r_hfs.values = self.convert_in_time(
                equilibrium.rmji.interp(rho_poloidal=rho)
            )
            self.R_mag.values = self.convert_in_time(equilibrium.rmag)
            self.z_mag.values = self.convert_in_time(equilibrium.zmag)
            self.minor_radius.values = (self.maj_r_lfs - self.maj_r_hfs) / 2.0
        else:
            print_like(
                "Plasma class doesn't have equilibrium: "
                "skipping geometry assignments..."
            )

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
        default=False,
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
                datatype=("density", "thermal_neutrals"),
                xspl=rho,
                xend=xend,
            )
            Nh = Nh_prof()
            tau = None

        # print_like("Initialize fractional abundance and power loss objects")
        fract_abu, power_loss_tot, power_loss_sxr = {}, {}, {}
        for elem in self.elements:
            scd = self.ADASReader.get_adf11("scd", elem, self.adf11[elem]["scd"])
            acd = self.ADASReader.get_adf11("acd", elem, self.adf11[elem]["acd"])
            ccd = self.ADASReader.get_adf11("ccd", elem, self.adf11[elem]["ccd"])
            fract_abu[elem] = FractionalAbundance(scd, acd, CCD=ccd)
            if Te is not None and Ne is not None:
                fract_abu[elem](Ne=Ne, Te=Te, Nh=Nh, tau=tau, full_run=self.full_run)

            plt = self.ADASReader.get_adf11("plt", elem, self.adf11[elem]["plt"])
            prb = self.ADASReader.get_adf11("prb", elem, self.adf11[elem]["prb"])
            prc = self.ADASReader.get_adf11("prc", elem, self.adf11[elem]["prc"])
            power_loss_tot[elem] = PowerLoss(plt, prb, PRC=prc)
            if Te is not None and Ne is not None:
                F_z_t = fract_abu[elem].F_z_t
                power_loss_tot[elem](Te, F_z_t, Ne=Ne, Nh=Nh, full_run=self.full_run)

            if "pls" in self.adf11[elem].keys() and "prs" in self.adf11[elem].keys():
                pls = self.ADASReader.get_adf11("pls", elem, self.adf11[elem]["pls"])
                prs = self.ADASReader.get_adf11("prs", elem, self.adf11[elem]["prs"])
                power_loss_sxr[elem] = PowerLoss(pls, prs)
                if Te is not None and Ne is not None:
                    F_z_t = fract_abu[elem].F_z_t
                    power_loss_sxr[elem](Te, F_z_t, Ne=Ne, full_run=self.full_run)

        self.adf11 = self.adf11
        self.fract_abu = fract_abu
        self.power_loss_tot = power_loss_tot
        if "pls" in self.adf11[elem].keys() and "prs" in self.adf11[elem].keys():
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

    def assign_profiles(
        self, profile: str = "electron_density", t: float = None, element: str = "ar"
    ):
        if profile == "electron_density":
            self.electron_density.loc[dict(t=t)] = self.Ne_prof()
        elif profile == "electron_temperature":
            self.electron_temperature.loc[dict(t=t)] = self.Te_prof()
        elif profile == "ion_temperature":
            self.ion_temperature.loc[dict(t=t)] = self.Ti_prof()
        elif profile == "toroidal_rotation":
            self.toroidal_rotation.loc[dict(t=t)] = self.Vrot_prof()
        elif profile == "impurity_density":
            self.impurity_density.loc[dict(t=t, element=element)] = self.Nimp_prof()
        elif profile == "neutral_density":
            self.neutral_density.loc[dict(t=t, element=element)] = self.Nh_prof()
        else:
            raise ValueError(
                f"{profile} currently not found in possible Plasma properties"
            )

    def update_profiles(
        self,
        parameters: dict,
        profile_prefixs: list = [
            "Te_prof",
            "Ti_prof",
            "Ne_prof",
            "Nimp_prof",
            "Vrot_prof",
        ],
    ):
        """
        Update plasma profiles with profile parameters i.e. {"Ne_prof.y0":1e19} -> Ne_prof.y0
        """
        for param, value in parameters.items():
            _prefix = [pref for pref in profile_prefixs if pref in param]
            if _prefix:
                prefix: str = _prefix[0]
                key = param.replace(prefix + ".", "")
                profile = getattr(self, prefix)
                if hasattr(profile, key):
                    setattr(profile, key, value)
                else:
                    raise ValueError(f"parameter: {key} not found in {prefix}")

        for key in [
            "electron_density",
            "electron_temperature",
            "ion_temperature",
            "toroidal_rotation",
            "impurity_density",
        ]:
            self.assign_profiles(key, t=self.time_to_calculate)


class PressureProfile:
    """Pressure profile"""

    def __init__(self, _data: DataArray):
        self._data = _data
        self.temperature: DataArray = DataArray(None)
        self.density: DataArray = DataArray(None)

    def __call__(self, density: DataArray, temperature: DataArray):
        if not self.density.equals(density) or not self.temperature.equals(temperature):
            self.density = deepcopy(density)
            self.temperature = deepcopy(temperature)
            self._data.values = ph.calc_pressure(self.density, self.temperature)
        return self._data


class PressureIntegral:
    """Volume integral of the pressure"""

    def __init__(self, _data: DataArray):
        self._data = _data
        self.pressure: DataArray = DataArray(None)
        self.volume: DataArray = DataArray(None)

    def __call__(self, pressure: DataArray, volume: DataArray):
        if not self.pressure.equals(pressure) or not self.volume.equals(volume):
            self.pressure = deepcopy(pressure)
            self.volume = deepcopy(volume)
            self._data = xr.full_like(self._data, 0.0)
            if "element" in pressure.dims:
                for elem in pressure.element:
                    self._data.values += np.trapz(
                        self.pressure.sel(element=elem), self.volume
                    )
            else:
                self._data.values = np.trapz(self.pressure, self.volume)
        return self._data


class Meanz:
    """Mean charge profile (element dependent)"""

    def __init__(self, _data: DataArray):
        self._data = _data
        self.elements = self._data.element.values
        self.fz: dict = None

    def __call__(self, fz: dict):
        recalculate = False
        if self.fz is not None:
            for elem in self.elements:
                if not self.fz[elem].equals(fz[elem]):
                    recalculate = True
                    break
        else:
            recalculate = True

        if recalculate:
            self.fz = deepcopy(fz)
            for elem in self.elements:
                self._data.loc[dict(element=elem)] = (
                    (self.fz[elem] * self.fz[elem].ion_charges)
                    .sum("ion_charges")
                    .values
                )

        return self._data


class Zeff:
    """Effective charge profile (element dependent)"""

    def __init__(self, _data: DataArray):
        self._data = _data
        self.ne: DataArray = DataArray(None)
        self.ni: DataArray = DataArray(None)
        self.meanz: DataArray = DataArray(None)

    def __call__(self, ne: DataArray, ni: DataArray, meanz: DataArray):
        if (
            not self.ne.equals(ne)
            or not self.ni.equals(ni)
            or not self.meanz.equals(meanz)
        ):
            self.ne = deepcopy(ne)
            self.ni = deepcopy(ni)
            self.meanz = deepcopy(meanz)
            self._data.values = ((ni * meanz**2) / ne).values
        return self._data


class IonDensity:
    """Main ion themal density from dilution (including fast ion density)"""

    def __init__(self, _data: DataArray):
        self._data = _data
        self.ne: DataArray = DataArray(None)
        self.nimp: DataArray = DataArray(None)
        self.nf: DataArray = DataArray(None)
        self.meanz: DataArray = DataArray(None)

    def __call__(
        self,
        ne: DataArray,
        nimp: DataArray,
        nf: DataArray,
        meanz: DataArray,
        main_ion: str,
    ):
        if (
            not self.ne.equals(ne)
            or not self.nimp.equals(nimp)
            or not self.nf.equals(nf)
            or not self.meanz.equals(meanz)
        ):
            self.ne = deepcopy(ne)
            self.nimp = deepcopy(nimp)
            self.nf = deepcopy(nf)
            self.meanz = deepcopy(meanz)
            impurity_dilution = (self.nimp * self.meanz).sum("element")
            fast_ion_dilution = self.nf * self.meanz.sel(element=main_ion)
            self._data.loc[dict(element=main_ion)] = (
                self.ne - impurity_dilution - fast_ion_dilution
            ).values
            for element in self.nimp.element:
                self._data.loc[dict(element=element)] = self.nimp.sel(
                    element=element
                ).values

        return self._data


class Fz:
    """Fractional abundance (element dependent)"""

    def __init__(self, _data: dict):
        self._data = _data
        self.elements = list(self._data)
        self.te: DataArray = DataArray(None)
        self.ne: DataArray = DataArray(None)
        self.nh: DataArray = DataArray(None)
        self.tau: DataArray = DataArray(None)
        self.fract_abu: dict = None

    def __call__(
        self,
        te: DataArray,
        ne: DataArray,
        nh: DataArray,
        tau: DataArray,
        fract_abu: dict,
        t: LabeledArray,
        full_run: bool = False,
    ):

        if (
            not self.te.equals(te)
            or not self.ne.equals(ne)
            or not self.nh.equals(nh)
            or not self.tau.equals(tau)
            or not self.fract_abu != fract_abu
        ):
            self.te = deepcopy(te)
            self.ne = deepcopy(ne)
            self.nh = deepcopy(nh)
            self.tau = deepcopy(tau)
            self.fract_abu = deepcopy(fract_abu)
            for t in np.array(t, ndmin=1):
                for elem in self.elements:
                    _te = te.sel(t=t)
                    _ne = ne.sel(t=t)
                    _tau = None
                    if np.any(tau != 0):
                        _tau = tau.sel(t=t)
                    _nh = None
                    if np.any(nh != 0):
                        _nh = nh.sel(t=t)
                    fz_tmp = self.fract_abu[elem](
                        _te,
                        Ne=_ne,
                        Nh=_nh,
                        tau=_tau,
                        full_run=full_run,
                    )
                    self._data[elem].loc[dict(t=t)] = fz_tmp.transpose().values
        return self._data


class Lz:
    """Radiation loss parameter (element dependent)"""

    def __init__(self, _data: dict):
        self._data = _data
        self.te: DataArray = DataArray(None)
        self.ne: DataArray = DataArray(None)
        self.nh: DataArray = DataArray(None)
        self.fz: dict = None
        self.power_loss: dict = None
        self.elements = list(self._data)

    def __call__(
        self,
        te: DataArray,
        ne: DataArray,
        nh: DataArray,
        fz: dict,
        power_loss: dict,
        t: LabeledArray,
        full_run: bool = False,
    ):
        recalculate = False
        if self.fz is not None:
            for elem in self.elements:
                if not self.fz[elem].equals(fz[elem]):
                    recalculate = True
                    break
            if not recalculate:
                if (
                    not self.te.equals(te)
                    or not self.ne.equals(ne)
                    or not self.nh.equals(nh)
                    or not self.power_loss != power_loss
                ):
                    recalculate = True
        else:
            recalculate = True

        if recalculate:
            self.te = deepcopy(te)
            self.ne = deepcopy(ne)
            self.nh = deepcopy(nh)
            self.fz = deepcopy(fz)
            self.power_loss = deepcopy(power_loss)
            for t in np.array(t, ndmin=1):
                for elem in self.elements:
                    _te = te.sel(t=t)
                    _ne = ne.sel(t=t)
                    _nh = None
                    if np.any(nh != 0):
                        _nh = nh.sel(t=t)
                    lz_tmp = self.power_loss[elem](
                        _te,
                        self.fz,
                        Ne=_ne,
                        Nh=_nh,
                        bounds_check=False,
                        full_run=full_run,
                    )
                    self._data[elem].loc[dict(t=t)] = lz_tmp.transpose().values
        return self._data


class RadiationProfile:
    """Radiation profile (element dependent)"""

    def __init__(self, _data: DataArray):
        self._data = _data
        self.ne: DataArray = None
        self.ni: DataArray = None
        self.lz: dict = None
        self.elements = self._data.element.values

    def __call__(self, lz: dict, ne: DataArray, ni: DataArray):
        recalculate = False
        if self.lz is not None:
            for elem in self.elements:
                if not self.lz[elem].equals(lz[elem]):
                    recalculate = True
                    break
        else:
            recalculate = True

        if recalculate or not self.ne.equals(ne) or not self.ni.equals(ni):
            self.lz = deepcopy(lz)
            self.ne = deepcopy(ne)
            self.ni = deepcopy(ni)
            for elem in self.elements:
                total_radiation = (
                    self.lz[elem].sum("ion_charges")
                    * self.ne
                    * self.ni.sel(element=elem)
                )
                self._data.loc[dict(element=elem)] = xr.where(
                    total_radiation >= 0,
                    total_radiation,
                    0.0,
                ).values

        return self._data


class RadiationIntegral:
    """Volume integral of the radiation"""

    def __init__(self, _data: DataArray):
        self._data = _data
        self.radiation_profile: DataArray = DataArray(None)
        self.volume: DataArray = DataArray(None)

    def __call__(self, radiation_profile: DataArray, volume: DataArray):
        if not self.radiation_profile.equals(
            radiation_profile
        ) or not self.volume.equals(volume):
            self.radiation_profile = deepcopy(radiation_profile)
            self.volume = deepcopy(volume)
            self._data = xr.full_like(self._data, 0.0)
            if "element" in radiation_profile.dims:
                for elem in radiation_profile.element:
                    self._data.values += np.trapz(
                        self.radiation_profile.sel(element=elem), self.volume
                    )
            else:
                self._data.values = np.trapz(self.radiation_profile, self.volume)
        return self._data


def example_run(
    pulse: int = None,
    tstart=0.02,
    tend=0.1,
    dt=0.01,
    main_ion="h",
    impurities=("c", "ar", "he"),
    impurity_concentration=(0.03, 0.001, 0.01),
):
    # TODO: swap all profiles to new version!
    full_run = False

    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=main_ion,
        impurities=impurities,
        impurity_concentration=impurity_concentration,
        full_run=full_run,
    )
    # plasma.build_atomic_data(default=True)
    # Assign profiles to time-points
    nt = len(plasma.t)
    ne_peaking = np.linspace(1, 2, nt)
    te_peaking = np.linspace(1, 2, nt)
    vrot_peaking = np.linspace(1, 2, nt)
    vrot0 = np.linspace(plasma.Vrot_prof.y0 * 1.1, plasma.Vrot_prof.y0 * 2.5, nt)
    ti0 = np.linspace(plasma.Ti_prof.y0 * 1.1, plasma.Te_prof.y0 * 2.5, nt)
    nimp_peaking = np.linspace(1, 5, nt)
    nimp_y0 = plasma.Nimp_prof.y0 * np.linspace(1, 8, nt)
    nimp_wcenter = np.linspace(0.4, 0.1, nt)
    for i, t in enumerate(plasma.t):
        plasma.Te_prof.peaking = te_peaking[i]
        plasma.assign_profiles(profile="electron_temperature", t=t)

        plasma.Ti_prof.peaking = te_peaking[i]
        plasma.Ti_prof.y0 = ti0[i]
        plasma.assign_profiles(profile="ion_temperature", t=t)

        plasma.Vrot_prof.peaking = vrot_peaking[i]
        plasma.Vrot_prof.y0 = vrot0[i]
        plasma.assign_profiles(profile="toroidal_rotation", t=t)

        plasma.Ne_prof.peaking = ne_peaking[i]
        plasma.assign_profiles(profile="electron_density", t=t)

        plasma.Nimp_prof.peaking = nimp_peaking[i]
        plasma.Nimp_prof.y0 = nimp_y0[i]
        plasma.Nimp_prof.wcenter = nimp_wcenter[i]
        for elem in plasma.impurities:
            plasma.assign_profiles(profile="impurity_density", t=t, element=elem)

        for elem in plasma.elements:
            plasma.assign_profiles(profile="toroidal_rotation", t=t, element=elem)

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
