from copy import deepcopy
import pickle
from hda.utils import get_funcion_name

import hda.physics as ph
from hda.profiles import Profiles
from hda.utils import assign_data, assign_datatype, print_like

from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters import FluxSurfaceCoordinates
from indica.converters.time import convert_in_time_dt
from indica.converters.time import get_tlabels_dt
from indica.datatypes import ELEMENTS
from indica.equilibrium import Equilibrium
from indica.provenance import get_prov_attribute
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader
from indica.numpy_typing import LabeledArray

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
        machine_dimensions=((0.15, 0.75), (-0.7, 0.7)),
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
        for elem in self.impurities:
            self.elements.append(elem)
        self.impurity_concentration = assign_data(
            DataArray(
                np.array(impurity_concentration),
                coords=[("element", list(self.impurities))],
            ),
            ("concentration", "impurity"),
        )
        self.ADF11 = ADF11
        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.t = get_tlabels_dt(self.tstart, self.tend, self.dt)
        self.radial_coordinate = np.linspace(0, 1.0, 41)
        self.radial_coordinate_type = "rho_poloidal"
        self.machine_dimensions = machine_dimensions

        self.forward_models = {}

        self.initialize_variables()

    def set_equilibrium(self, equilibrium: Equilibrium):
        """
        Assign equilibrium object
        """
        self.equilibrium = equilibrium

    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        """
        Assign flux surface transform object for remapping
        """
        self.flux_transform = flux_transform

        if hasattr(self, "equilibrium"):
            if not hasattr(self.flux_transform, "equilibrium"):
                self.flux_transform.set_equilibrium(self.equilibrium)
            if self.flux_transform.equilibrium != self.equilibrium:
                raise ValueError(
                    "Plasma class equilibrium and flux_transform are not the same object...s"
                )
        else:
            if hasattr(flux_transform, "equilibrium"):
                self.equilibrium = flux_transform.equilibrium

    def initialize_variables(self):
        """
        Initialize all class attributes
        """

        # Dictionary keeping track of deta use for optimisations
        self.optimisation = {}

        # Assign plasma and machine attributes
        self.machine_R = np.linspace(
            self.machine_dimensions[0][0], self.machine_dimensions[0][1], 100
        )
        self.machine_z = np.linspace(
            self.machine_dimensions[1][0], self.machine_dimensions[1][1], 100
        )

        nt = len(self.t)
        nr = len(self.radial_coordinate)
        nel = len(self.elements)
        nimp = len(self.impurities)

        R_midplane = np.linspace(self.machine_R.min(), self.machine_R.max(), 100)
        self.R_midplane = R_midplane
        z_midplane = np.full_like(R_midplane, 0.0)
        self.z_midplane = z_midplane

        coords_radius = (self.radial_coordinate_type, self.radial_coordinate)
        coords_time = ("t", self.t)
        coords_elem = ("element", list(self.elements))
        coords_imp = ("element", list(self.impurities))

        self.data0d = DataArray(0.0)
        self.data1d_time = DataArray(np.zeros(nt), coords=[coords_time])
        self.data1d_rho = DataArray(np.zeros(nr), coords=[coords_radius])
        self.data2d = DataArray(np.zeros((nt, nr)), coords=[coords_time, coords_radius])
        self.data2d_elem = DataArray(
            np.zeros((nel, nt)), coords=[coords_elem, coords_time]
        )
        self.data3d = DataArray(
            np.zeros((nel, nt, nr)), coords=[coords_elem, coords_time, coords_radius]
        )
        self.data3d_imp = DataArray(
            np.zeros((nimp, nt, nr)), coords=[coords_imp, coords_time, coords_radius]
        )

        self.time = assign_data(self.data1d_time, ("t", "plasma"), "s")
        self.time.values = self.t

        rho_type = self.radial_coordinate_type.split("_")
        if rho_type[1] != "poloidal":
            print_like("Only poloidal rho in input for the time being...")
            raise AssertionError
        self.rho = assign_data(self.data1d_rho, (rho_type[0], rho_type[1]))
        self.rho.values = self.radial_coordinate

        self.data3d_fz = {}
        for elem in self.elements:
            nz = ELEMENTS[elem][0] + 1
            ion_charges = np.arange(nz)
            self.data3d_fz[elem] = DataArray(
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
        self.Nh_prof = Profiles(datatype=("density", "thermal_neutrals"), xspl=self.rho)
        self.Vrot_prof = Profiles(datatype=("rotation", "toroidal"), xspl=self.rho)

        self.ipla = assign_data(self.data1d_time, ("current", "plasma"), "A")
        # self.bt_0 = assign_data(self.data1d_time, ("field", "toroidal"), "T")
        # self.R_bt_0 = assign_data(self.data0d, ("major_radius", "toroidal_field"), "T")
        self.R_0 = assign_data(self.data1d_time, ("major_radius", "geometric"), "m")
        self.R_mag = assign_data(self.data1d_time, ("major_radius", "magnetic"))
        self.z_mag = assign_data(self.data1d_time, ("z", "magnetic"))
        self.maj_r_lfs = assign_data(self.data2d, ("radius", "major"))
        self.maj_r_hfs = assign_data(self.data2d, ("radius", "major"))
        self.ne_0 = assign_data(self.data1d_time, ("density", "electron"))
        self.te_0 = assign_data(self.data1d_time, ("temperature", "electron"))
        self.ti_0 = assign_data(self.data1d_time, ("temperature", "ion"))
        self.electron_temperature = assign_data(self.data2d, ("temperature", "electron"))
        self.electron_density = assign_data(self.data2d, ("density", "electron"))
        self.neutral_density = assign_data(self.data2d, ("density", "neutral"))
        self.tau = assign_data(self.data2d, ("time", "residence"))
        self.minor_radius = assign_data(
            self.data2d, ("minor_radius", "plasma")
        )  # LFS-HFS averaged value
        self.volume = assign_data(self.data2d, ("volume", "plasma"))
        self.area = assign_data(self.data2d, ("area", "plasma"))
        self.j_phi = assign_data(self.data2d, ("current", "density"))
        self.b_pol = assign_data(self.data2d, ("field", "poloidal"))
        self.b_tor_lfs = assign_data(self.data2d, ("field", "toroidal"))
        self.b_tor_hfs = assign_data(self.data2d, ("field", "toroidal"))
        self.q_prof = assign_data(self.data2d, ("factor", "safety"))
        self.conductivity = assign_data(self.data2d, ("conductivity", "plasma"))
        self.l_i = assign_data(self.data1d_time, ("inductance", "internal"))

        self.ion_temperature = assign_data(self.data3d, ("temperature", "ion"))
        self.toroidal_rotation = assign_data(self.data3d, ("toroidal_rotation", "ion"))
        self.impurity_density = assign_data(self.data3d_imp, ("density", "impurity"), "m^-3")
        self.fast_temperature = assign_data(self.data2d, ("temperature", "fast"))
        self.fast_density = assign_data(self.data2d, ("density", "fast"))

        # Private variables for class property variables
        # TODO: transpose in dependencies, i.e. what depends on e.g. electron_density?
        # self.dependencies = {"electron_density":("ion_densityity", "zeff", "fz", )}
        self.properties = {
            "ion_densityity": ("electron_density", "fast_density", "meanz", "main_ion", "impurity_density",),
            "zeff": ("electron_density", "ion_densityity", "meanz"),
            "meanz": ("fz",),
            "fz": ("electron_temperature", "electron_density", "tau", "neutral_density",),
            "lz_tot": ("electron_temperature, electron_density, neutral_density", "fz"),
            "lz_sxr": ("electron_temperature, electron_density, neutral_density", "fz"),
            "total_radiation": ("lz_tot", "electron_density", "ion_densityity"),
            "sxr_radiation": ("lz_sxr", "electron_density", "ion_densityity"),
            "prad_tot": ("total_radiation"),
            "prad_sxr": ("sxr_radiation"),
            "pressure_el": ("electron_density", "electron_temperature"),
            "pressure_th": ("electron_density", "ion_densityity", "electron_temperature", "ion_temperature"),
            "pressure_tot": ("pressure_th", "fast_density", "fast_temperature"),
            "pth": ("pressure_th",),
            "ptot": ("pressure_tot",),
            "wth": ("pth",),
            "wp": ("ptot",),
        }
        for attr in self.properties.keys():
            setattr(self, f"_{attr}", None)
        #
        # self._beta_pol = assign_data(self.data1d_time, ("beta", "poloidal"), "J")
        # self._vloop = assign_data(self.data1d_time, ("density", "ion"), "m^-3")
        # self._j_phi = assign_data(
        #     self.data1d_time, ("toroidal_current", "density"), "A m^-2"
        # )
        # self._btot = assign_data(self.data1d_time, ("magnetic_field", "total"), "T")

    def check_property(self, property_name: str):
        return None
        value = getattr(self, f"_{property_name}")
        if value is not None:
            return value

    @property
    def pressure_el(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._pressure_el = assign_data(
            self.data2d, ("pressure", "electron"), "Pa m^-3"
        )
        self._pressure_el.values = ph.calc_pressure(self.electron_density, self.electron_temperature)
        return self._pressure_el

    @property
    def pressure_th(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._pressure_th = assign_data(self.data2d, ("pressure", "thermal"), "Pa m^-3")
        self._ion_density = assign_data(self.data3d, ("density", "ion"), "m^-3")
        ion_density = self.ion_density
        self._pressure_th.values = self.pressure_el
        for elem in self.elements:
            self._pressure_th.values += ph.calc_pressure(
                ion_density.sel(element=elem).values,
                self.ion_temperature.sel(element=elem).values,
            )
        return self._pressure_th

    @property
    def pressure_tot(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._pressure_tot = assign_data(self.data2d, ("pressure", "total"), "Pa m^-3")
        self._pressure_tot.values = self.pressure_th + ph.calc_pressure(
            self.fast_density, self.fast_temperature
        )
        return self._pressure_tot

    @property
    def pth(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._pth = assign_data(self.data1d_time, ("pressure", "thermal"), "Pa")
        pressure_th = self.pressure_th
        for t in self.time:
            self._pth.loc[dict(t=t)] = np.trapz(
                pressure_th.sel(t=t), self.volume.sel(t=t)
            )
        return self._pth

    @property
    def ptot(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._ptot = assign_data(self.data1d_time, ("pressure", "total"), "Pa")
        pressure_tot = self.pressure_tot
        for t in self.time:
            self._ptot.loc[dict(t=t)] = np.trapz(
                pressure_tot.sel(t=t), self.volume.sel(t=t)
            )
        return self._ptot

    @property
    def wth(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._wth = assign_data(self.data1d_time, ("stored_energy", "thermal"), "J")
        pth = self.pth
        self._wth.values = 3 / 2 * pth.values
        return self._wth

    @property
    def wp(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._wp = assign_data(self.data1d_time, ("stored_energy", "total"), "J")
        ptot = self.ptot
        self._wp.values = 3 / 2 * ptot.values
        return self._wp

    @property
    def fz(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._fz = deepcopy(self.data3d_fz)
        for elem in self.elements:
            self._fz[elem].attrs["datatype"] = ("fractional_abundance", "ion")
            self._fz[elem].attrs["unit"] = ""

        for elem in self.elements:
            for t in self.time:
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
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._zeff = assign_data(self.data3d, ("charge", "effective"), "")
        ion_density = self.ion_density
        meanz = self.meanz
        for elem in self.elements:
            self._zeff.loc[dict(element=elem)] = (
                (ion_density.sel(element=elem) * meanz.sel(element=elem) ** 2)
                / self.electron_density
            ).values
        return self._zeff

    @property
    def ion_density(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._ion_density = assign_data(self.data3d, ("density", "ion"), "m^-3")
        impurity_density = self.impurity_density
        meanz = self.meanz
        main_ion_density = self.electron_density - self.fast_density * meanz.sel(element=self.main_ion)
        for elem in self.impurities:
            self._ion_density.loc[dict(element=elem)] = impurity_density.sel(element=elem).values
            main_ion_density -= impurity_density.sel(element=elem) * meanz.sel(element=elem)

        self._ion_density.loc[dict(element=self.main_ion)] = main_ion_density.values
        return self._ion_density

    @property
    def meanz(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._meanz = assign_data(self.data3d, ("charge", "mean"), "")
        fz = self.fz
        for elem in self.elements:
            self._meanz.loc[dict(element=elem)] = (
                (fz[elem] * fz[elem].ion_charges).sum("ion_charges").values
            )
        return self._meanz

    @property
    def lz_tot(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._lz_tot = deepcopy(self.data3d_fz)
        for elem in self.elements:
            self._lz_tot[elem].attrs["datatype"] = ("radiation_loss_parameter", "total")
            self._lz_tot[elem].attrs["unit"] = "W m^3"

        fz = self.fz
        for elem in self.elements:
            for t in self.time:
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
                        Te, Fz, Ne=Ne, Nh=Nh, bounds_check=False, full_run=self.full_run
                    )
                    .transpose()
                    .values
                )
        return self._lz_tot

    @property
    def lz_sxr(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._lz_sxr = deepcopy(self.data3d_fz)

        if not hasattr(self, "power_loss_sxr"):
            return self._lz_sxr

        for elem in self.elements:
            self._lz_sxr[elem].attrs["datatype"] = ("radiation_loss_parameter", "sxr")
            self._lz_sxr[elem].attrs["unit"] = "W m^3"

        fz = self.fz
        for elem in self.elements:
            for t in self.time:
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
                        Te, Fz, Ne=Ne, Nh=Nh, bounds_check=False, full_run=self.full_run
                    )
                    .transpose()
                    .values
                )
        return self._lz_sxr

    @property
    def total_radiation(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._total_radiation = assign_data(
            self.data3d, ("radiation_emission", "total"), "W m^-3"
        )
        lz_tot = self.lz_tot
        ion_density = self.ion_density
        for elem in self.elements:
            total_radiation = (
                lz_tot[elem].sum("ion_charges")
                * self.electron_density
                * ion_density.sel(element=elem)
            )
            self._total_radiation.loc[dict(element=elem)] = xr.where(
                total_radiation >= 0, total_radiation, 0.0,
            ).values
        return self._total_radiation

    @property
    def sxr_radiation(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._sxr_radiation = assign_data(
            self.data3d, ("radiation_emission", "sxr"), "W m^-3"
        )
        if not hasattr(self, "power_loss_sxr"):
            return self._sxr_radiation

        lz_sxr = self.lz_sxr
        ion_density = self.ion_density
        for elem in self.elements:
            sxr_radiation = (
                lz_sxr[elem].sum("ion_charges")
                * self.electron_density
                * ion_density.sel(element=elem)
            )
            self._sxr_radiation.loc[dict(element=elem)] = xr.where(
                sxr_radiation >= 0, sxr_radiation, 0.0,
            ).values
        return self._sxr_radiation

    @property
    def prad_tot(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._prad_tot = assign_data(self.data2d_elem, ("radiation", "total"), "W")
        total_radiation = self.total_radiation
        for elem in self.elements:
            for t in self.time:
                self._prad_tot.loc[dict(element=elem, t=t)] = np.trapz(
                    total_radiation.sel(element=elem, t=t), self.volume.sel(t=t)
                )
        return self._prad_tot

    @property
    def prad_sxr(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

        self._prad_sxr = assign_data(self.data2d_elem, ("radiation", "sxr"), "W")
        if not hasattr(self, "power_loss_sxr"):
            return self._prad_sxr

        sxr_radiation = self.sxr_radiation
        for elem in self.elements:
            for t in self.time:
                self._prad_sxr.loc[dict(element=elem, t=t)] = np.trapz(
                    sxr_radiation.sel(element=elem, t=t), self.volume.sel(t=t)
                )
        return self._prad_sxr

    @property
    def vloop(self):
        value = self.check_property(get_funcion_name())
        if value is not None:
            return value

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
        for t in self.t:
            resistivity = 1.0 / self.conductivity.sel(t=t)
            ir = np.where(np.isfinite(resistivity))
            vloop = ph.vloop(
                resistivity[ir], j_phi.sel(t=t)[ir], self.area.sel(t=t)[ir]
            )
            self._vloop.loc[dict(t=t)] = vloop.values
        return self._vloop

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
            print_like("Calculate geometric quantities")

            self.volume.values = self.convert_in_time(
                equilibrium.volume.interp(rho_poloidal=rho)
            )
            self.area.values = self.convert_in_time(equilibrium.area.interp(rho_poloidal=rho))
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
                "Plasma class doesn't have equilibrium: skipping geometry assignments..."
            )

    def convert_in_time(self, value: DataArray, method="linear"):
        binned = convert_in_time_dt(self.tstart, self.tend, self.dt, value).interp(
            t=self.time, method=method
        )

        return binned

    def build_atomic_data(
        self,
        adf11: dict = None,
        Te: DataArray = None,
        Ne: DataArray = None,
        Nh: DataArray = None,
        tau: DataArray = None,
        default=False,
    ):
        if default:
            xend = 1.02
            rho_end = 1.01
            rho = np.abs(np.linspace(rho_end, 0, 100) ** 1.8 - rho_end-0.01)
            Te = Profiles(datatype=("temperature", "electron"), xspl=rho, xend=xend)
            Te.y0 = 10.0e3
            Te.build_profile()
            Te = Te.yspl
            Ne = Profiles(datatype=("density", "electron"), xspl=rho, xend=xend).yspl
            Nh = Profiles(datatype=("density", "thermal_neutrals"), xspl=rho, xend=xend).yspl
            tau = None

        print_like("Initialize fractional abundance and power loss objects")
        fract_abu, power_loss_tot, power_loss_sxr = {}, {}, {}
        for elem in self.elements:
            if adf11 is None:
                adf11 = self.ADF11

            scd = self.ADASReader.get_adf11("scd", elem, adf11[elem]["scd"])
            acd = self.ADASReader.get_adf11("acd", elem, adf11[elem]["acd"])
            ccd = self.ADASReader.get_adf11("ccd", elem, adf11[elem]["ccd"])
            fract_abu[elem] = FractionalAbundance(scd, acd, CCD=ccd)
            if Te is not None and Ne is not None:
                fract_abu[elem](Ne=Ne, Te=Te, Nh=Nh, tau=tau, full_run=self.full_run)

            plt = self.ADASReader.get_adf11("plt", elem, adf11[elem]["plt"])
            prb = self.ADASReader.get_adf11("prb", elem, adf11[elem]["prb"])
            prc = self.ADASReader.get_adf11("prc", elem, adf11[elem]["prc"])
            power_loss_tot[elem] = PowerLoss(plt, prb, PRC=prc)
            if Te is not None and Ne is not None:
                F_z_t = fract_abu[elem].F_z_t
                power_loss_tot[elem](Te, F_z_t, Ne=Ne, Nh=Nh, full_run=self.full_run)

            if "pls" in adf11[elem].keys() and "prs" in adf11[elem].keys():
                pls = self.ADASReader.get_adf11("pls", elem, adf11[elem]["pls"])
                prs = self.ADASReader.get_adf11("prs", elem, adf11[elem]["prs"])
                power_loss_sxr[elem] = PowerLoss(pls, prs)
                if Te is not None and Ne is not None:
                    F_z_t = fract_abu[elem].F_z_t
                    power_loss_sxr[elem](Te, F_z_t, Ne=Ne, full_run=self.full_run)

        self.adf11 = adf11
        self.fract_abu = fract_abu
        self.power_loss_tot = power_loss_tot
        if "pls" in adf11[elem].keys() and "prs" in adf11[elem].keys():
            self.power_loss_sxr = power_loss_sxr

    def set_neutral_density(self, y0=1.0e10, y1=1.0e15, decay=12):
        self.Nh_prof.y0 = y0
        self.Nh_prof.y1 = y1
        self.Nh_prof.yend = y1
        self.Nh_prof.wped = decay
        self.Nh_prof.build_profile()
        for t in self.t:
            self.neutral_density.loc[dict(t=t)] = self.Nh_prof.yspl.values

    def map_to_midplane(self):
        # TODO: streamline this to avoid continuously re-calculating quantities e.g. ion_density..
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
            for t in self.t:
                rho = (
                    self.equilibrium.rho.sel(t=t, method="nearest")
                    .interp(R=R, z=z)
                    .drop(["R", "z"])
                )
                midplane_profiles[k].append(
                    prof_rho.sel(t=t, method="nearest")
                    .interp(rho_poloidal=rho)
                    .drop("rho_poloidal")
                )
            midplane_profiles[k] = xr.concat(midplane_profiles[k], "t").assign_coords(
                t=self.t
            )
            midplane_profiles[k] = xr.where(
                np.isfinite(midplane_profiles[k]), midplane_profiles[k], 0.0
            )

        self.midplane_profiles = midplane_profiles

    def calc_centrifugal_asymmetry(self, time=None, test_toroidal_rotation=None, plot=False):
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
        R_0 = self.maj_r_lfs.interp(rho_poloidal=self.rho_2d).drop("rho_poloidal")
        for elem in self.elements:
            main_ion_mass = ELEMENTS[self.main_ion][1]
            mass = ELEMENTS[elem][1]
            asymm = ph.centrifugal_asymmetry(
                self.ion_temperature.sel(element=elem).drop("element"),
                self.electron_temperature,
                mass,
                meanz.sel(element=elem).drop("element"),
                zeff,
                main_ion_mass,
                toroidal_rotation=self.toroidal_rotation.sel(element=elem).drop("element"),
            )
            self.centrifugal_asymmetry.loc[dict(element=elem)] = asymm
            asymmetry_factor = asymm.interp(rho_poloidal=self.rho_2d)
            self.asymmetry_multiplier.loc[dict(element=elem)] = np.exp(
                asymmetry_factor * (self.rho_2d.R ** 2 - R_0 ** 2)
            )

        self.ion_density_2d = (
            ion_density.interp(rho_poloidal=self.rho_2d).drop("rho_poloidal")
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
                    self.ion_density_2d.sel(element=elem).sel(t=t, z=z, method="nearest"),
                )
                self.ion_density.sel(element=elem).sel(t=t).plot(linestyle="dashed")
                plt.title(elem)

            elem = "ar"
            plt.figure()
            np.log(self.ion_density_2d.sel(element=elem).sel(t=t, method="nearest")).plot()
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
            total_radiation = xr.where(total_radiation >= 0, total_radiation, 0.0,)
            self.total_radiation.loc[dict(element=elem)] = total_radiation.values

            sxr_radiation = (
                self.lz_sxr[elem].sum("ion_charges")
                * self.electron_density
                * self.ion_density.sel(element=elem)
            )
            sxr_radiation = xr.where(sxr_radiation >= 0, sxr_radiation, 0.0,)
            self.sxr_radiation.loc[dict(element=elem)] = sxr_radiation.values

            if not hasattr(self, "prad_tot"):
                self.prad_tot = deepcopy(self.prad)
                self.prad_sxr = deepcopy(self.prad)
                assign_data(self.prad_sxr, ("radiation", "sxr"))

            prad_tot = self.prad_tot.sel(element=elem)
            prad_sxr = self.prad_sxr.sel(element=elem)
            for t in self.t:
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
                self, f,
            )
