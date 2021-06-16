from copy import deepcopy

import pickle

from scipy import constants
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import math
import hda.fac_profiles as fac
from hda.forward_models import Spectrometer
import hda.physics as ph
from hda.atomdat import fractional_abundance
from hda.atomdat import get_atomdat
from hda.atomdat import radiated_power
from hda.hdaadas import ADASReader

from indica.equilibrium import Equilibrium
from indica.readers import ST40Reader
from indica.converters import FluxSurfaceCoordinates
from indica.converters.time import bin_in_time

import xarray as xr
from xarray import DataArray

plt.ion()

# TODO: add elongation and triangularity in all equations


class HDAdata:
    def __init__(
        self,
        pulse=8256,
        tstart=0.01,
        tend=0.1,
        dt=0.01,
        ntheta=5,
        machine_dimensions=((0.15, 0.9), (-0.8, 0.8)),
        R_bt_0=0.4,
        elements=("h", "c", "ar"),
        ion_conc=(1, 0.02, 0.001),
    ):
        """

        Parameters
        ----------
        pulse

        """
        self.ADASReader = ADASReader()

        t = np.arange(tstart, tend, dt)
        theta = np.linspace(0, 2 * np.pi, ntheta + 1)[:-1]

        if pulse is not None:
            print(f"Reading data for pulse {pulse}")
            self.raw_data = {}

            self.pulse = pulse
            self.tstart = tstart
            self.tend = tend

            self.reader = ST40Reader(
                pulse, tstart - 2 * dt, tend + 2 * dt, max_freq=1.0e4
            )

            if pulse == 8303 or pulse == 8322 or pulse == 8323 or pulse == 8324:
                revision = 2
            else:
                revision = 0
            efit = self.reader.get("", "efit", revision)

            if revision ==0:
                whichRun, _ = self.reader._get_signal("", "efit", ":best_run", revision)
                whichRun = whichRun[-2:]
                if whichRun[0] == "0":
                    whichRun = whichRun[-1]
                revision = int(whichRun)
            else:
                whichRun = self.reader.get_revision_name(revision)[1:].upper()

            self.equilibrium = Equilibrium(efit)
            self.flux_coords = FluxSurfaceCoordinates("poloidal")
            self.flux_coords.set_equilibrium(self.equilibrium)

            efit["revision"] = revision
            efit["run"] = whichRun
            self.raw_data["efit"] = efit

            rho_ped = 0.85
            rho_core = np.linspace(0, rho_ped, 30)[:-1]
            rho_edge = np.linspace(rho_ped, 1.0, 15)  # [:-1]
            # rho_sep = np.linspace(1.0, 1.05, 5)
            rho = np.concatenate([rho_core, rho_edge])  # , rho_sep])

            inputs = {
                "t": t,
                "elements": elements,
                "ion_conc": ion_conc,
                "rho": rho,
                "rho_type": "rho_poloidal",
                "theta": theta,
                "machine_dimensions": machine_dimensions,
                "R_bt_0": R_bt_0,
            }
            self.initialize_variables(inputs)

            xrcs = self.reader.get("sxr", "xrcs", 0)
            for k in xrcs.keys():
                xrcs[k].attrs["transform"].set_equilibrium(self.equilibrium)
            self.raw_data["xrcs"] = xrcs
            self.ti_xrcs = bin_in_time(tstart, tend, self.freq, xrcs["ti_w"]).interp(
                t=self.time
            )
            self.te_xrcs = bin_in_time(tstart, tend, self.freq, xrcs["te_kw"]).interp(
                t=self.time
            )

            nirh1 = self.reader.get("", "nirh1", 0)
            for k in nirh1.keys():
                nirh1[k].attrs["transform"].set_equilibrium(self.equilibrium)
            self.raw_data["nirh1"] = nirh1

            smmh1 = self.reader.get("", "smmh1", 0)
            for k in smmh1.keys():
                smmh1[k].attrs["transform"].set_equilibrium(self.equilibrium)
            self.raw_data["smmh1"] = smmh1

            self.ne_l = bin_in_time(tstart, tend, self.freq, nirh1["ne"]).interp(
                t=self.time
            )
            self.ne_l.attrs = nirh1["ne"].attrs

            # self.ne_l = bin_in_time(tstart, tend, self.freq, smmh1["ne"]).interp(
            #     t=self.time
            # )
            # self.ne_l.attrs = smmh1["ne"].attrs

            # TODO temporary MAG reader --> : insert in reader class !!!
            vloop, vloop_path = self.reader._get_signal("", "mag", ".floop.l026:v", 0)
            vloop, vloop_path = self.reader._get_signal("", "mag", ".floop.l016:v", 0)
            vloop_dims, _ = self.reader._get_signal_dims(vloop_path, len(vloop.shape))
            vloop = DataArray(vloop, dims=("t",), coords={"t": vloop_dims[0]},)
            vloop = vloop.sel(t=slice(self.reader._tstart, self.reader._tend))
            meta = {
                "datatype": ("voltage", "loop"),
                "error": xr.zeros_like(vloop),
            }
            vloop.attrs = meta
            self.raw_data["vloop"] = vloop
            self.vloop = bin_in_time(tstart, tend, self.freq, vloop).interp(t=self.time)

            # TODO temporary BT reader --> to be calculated using equilibrium class
            tf_i, tf_i_path = self.reader._get_signal("", "psu", ".tf:i", -1)
            tf_i_dims, _ = self.reader._get_signal_dims(tf_i_path, len(tf_i.shape))
            bt_0 = tf_i * 24.0 * constants.mu_0 / (2 * np.pi * 0.4)
            bt_0 = DataArray(bt_0, dims=("t",), coords={"t": tf_i_dims[0]},)
            bt_0 = bt_0.sel(t=slice(self.reader._tstart, self.reader._tend))
            meta = {
                "datatype": ("field", "toroidal"),
                "error": xr.zeros_like(bt_0),
            }
            bt_0.attrs = meta
            self.raw_data["bt_0"] = bt_0
            self.bt_0 = bin_in_time(tstart, tend, self.freq, bt_0).interp(t=self.time)

            self.ipla = bin_in_time(tstart, tend, self.freq, efit["ipla"]).interp(
                t=self.time
            )
            self.R_mag = bin_in_time(
                tstart, tend, self.freq, self.equilibrium.rmag
            ).interp(t=self.time)
            self.R_0 = bin_in_time(tstart, tend, self.freq, efit["rmag"]).interp(
                t=self.time
            )
            self.wmhd = bin_in_time(tstart, tend, self.freq, efit["wp"]).interp(
                t=self.time
            )

            # if pkl:
            #     self.write_to_pickle()

        else:
            print("\n Setting default synthetic plasma data no longer enabled\n")

    def build_data(
        self, ne_shape=0.9, te_shape=0.6, regime="l_mode",
    ):
        """
        Create plasma data give information in inputs dictionary
        """

        print("\n Building data class \n")

        self.ne_shape = ne_shape
        self.te_shape = te_shape
        self.regime = regime

        if self.regime == "l_mode":
            self.profs.l_mode(ne_shape=self.ne_shape, te_shape=self.te_shape)
        else:
            self.profs.h_mode(ne_shape=self.ne_shape, te_shape=self.te_shape)

        print(" Calculating LOS info of all diagnostics")
        rho_xrcs = []
        trans_xrcs = self.te_xrcs.attrs["transform"]
        x1 = self.te_xrcs.coords[trans_xrcs.x1_name]
        x2_arr = np.linspace(0, 1, 300)
        x2 = DataArray(x2_arr, dims=trans_xrcs.x2_name)
        dl = trans_xrcs.distance(trans_xrcs.x2_name, DataArray(0), x2[0:2], 0)[1]
        self.te_xrcs.attrs["x2"] = x2
        self.ti_xrcs.attrs["x2"] = x2
        self.te_xrcs.attrs["dl"] = dl
        self.ti_xrcs.attrs["dl"] = dl
        self.te_xrcs.attrs["R"], self.te_xrcs.attrs["z"] = trans_xrcs.convert_to_Rz(
            x1, x2, 0
        )
        self.ti_xrcs.attrs["R"], self.ti_xrcs.attrs["z"] = trans_xrcs.convert_to_Rz(
            x1, x2, 0
        )
        for t in self.te_xrcs.t:
            rho_tmp, _ = self.flux_coords.convert_from_Rz(
                self.te_xrcs.attrs["R"], self.te_xrcs.attrs["z"], t
            )
            rho_xrcs.append(rho_tmp)
        rho_xrcs = xr.concat(rho_xrcs, "t")
        rho_xrcs = xr.where(rho_xrcs >= 0, rho_xrcs, 0.0)
        self.te_xrcs.attrs["rho"] = rho_xrcs
        self.ti_xrcs.attrs["rho"] = rho_xrcs

        rho_ne_l = []
        trans_ne_l = self.ne_l.attrs["transform"]
        x1 = self.ne_l.coords[trans_ne_l.x1_name]
        x2 = DataArray(x2_arr, dims=trans_ne_l.x2_name)
        dl = trans_ne_l.distance(trans_ne_l.x2_name, DataArray(0), x2[0:2], 0)[1]
        self.ne_l.attrs["dl"] = dl
        self.ne_l.attrs["R"], self.ne_l.attrs["z"] = trans_ne_l.convert_to_Rz(x1, x2, 0)
        for t in self.ne_l.t:
            rho_tmp, _ = self.flux_coords.convert_from_Rz(
                self.ne_l.attrs["R"], self.ne_l.attrs["z"], t
            )
            rho_ne_l.append(rho_tmp)
        rho_ne_l = xr.concat(rho_ne_l, "t")
        rho_ne_l = xr.where(rho_ne_l >= 0, rho_ne_l, 0.0)
        self.ne_l.attrs["rho"] = rho_ne_l

        # Minor radius (flux-surface averaged)
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
        min_r = bin_in_time(self.tstart, self.tend, self.freq, min_r,).interp(
            t=self.time
        )
        self.min_r = min_r

        # approximate area and volume for quasi-circular geometry
        # with minor radius = flux-surface averaged value
        self.area.values = (math.pi * self.min_r ** 2).values
        self.volume.values = (self.area * 2 * math.pi * self.R_0).values

        self.r_a.values = self.min_r.sel(rho_poloidal=1.0)
        self.r_b.values = self.r_a.values
        self.r_c.values = self.r_a.values
        self.r_d.values = self.r_a.values
        self.kappa.values = (self.r_b / self.r_a).values
        self.delta.values = ((self.r_c + self.r_d) / (2 * self.r_a)).values

        maj_r_lfs = bin_in_time(
            self.tstart,
            self.tend,
            self.freq,
            self.equilibrium.rmjo.interp(rho_poloidal=self.rho),
        )
        self.maj_r_lfs.values = maj_r_lfs.interp(t=self.time)
        maj_r_hfs = bin_in_time(
            self.tstart,
            self.tend,
            self.freq,
            self.equilibrium.rmji.interp(rho_poloidal=self.rho),
        )
        self.maj_r_hfs.values = maj_r_hfs.interp(t=self.time)

        dens = self.profs.ne.interp(rho_poloidal=self.rho)
        temp = self.profs.te.interp(rho_poloidal=self.rho)
        vrot = self.profs.vrot.interp(rho_poloidal=self.rho)
        for t in self.time:
            if hasattr(self, "te_xrcs"):
                te_0 = self.te_xrcs.sel(t=t).values
                ti_0 = self.ti_xrcs.sel(t=t).values
            else:
                te_0 = self.te_0.sel(t=t).values
                ti_0 = self.ti_0.sel(t=t).values

            self.el_dens.loc[dict(t=t)] = dens.values
            self.el_temp.loc[dict(t=t)] = (temp / temp.max()).values * te_0
            for i, elem in enumerate(self.elements):
                self.ion_temp.loc[dict(t=t, element=elem)] = (
                    temp / temp.max()
                ).values * ti_0
                self.vtor.loc[dict(t=t, element=elem)] = vrot.values

        # Rescale density to match LOS-integrated value (radial view across midplane,
        # crossing the plasma twice, to the inner column and back)
        el_dens_int = self.calc_ne_los_int()
        self.el_dens *= self.ne_l / (el_dens_int)

        for elem in self.elements:
            self.ion_dens.loc[dict(element=elem)] = self.el_dens * self.ion_conc.sel(
                element=elem
            )

        # Dilution, Zeff, radiated powers
        self.atomic_data = {}
        for elem in self.elements:
            # Read atomic data
            _, atomdat = get_atomdat(self.ADASReader, elem, charge="")

            # Interpolate on electron density and drop coordinate
            for k in atomdat.keys():
                atomdat[k] = (
                    atomdat[k]
                    .interp(electron_density=5.0e19, method="nearest")
                    .drop_vars(["electron_density"])
                )

            # Calculate fractional abundance, meanz and cooling factor
            # Add SXR when atomic data becomes available
            atomdat["fz"] = fractional_abundance(
                atomdat["scd"], atomdat["acd"], element=elem
            )
            atomdat["meanz"] = (atomdat["fz"] * atomdat["fz"].ion_charges).sum(
                "ion_charges"
            )
            atomdat["lz_tot"] = radiated_power(
                atomdat["plt"], atomdat["prb"], atomdat["fz"], element=elem
            )

            self.atomic_data[elem] = atomdat

        self.build_current_density()
        self.calc_magnetic_field()
        self.calc_meanz()
        self.calc_main_ion_dens(fast_dens=False)
        self.impose_flat_zeff()
        self.calc_zeff()
        self.calc_rad_power()
        # self.calc_pressure()
        # self.calc_beta_poloidal()

    def propagate_parameters(self):
        # Current density, poloidal field, li
        # self.build_current_density()
        self.calc_magnetic_field()

        # Mean charge
        self.calc_meanz()

        # Main ion density
        self.calc_main_ion_dens(fast_dens=False)

        # Adapt impurity concentration to flatten Zeff contribution from each element
        self.impose_flat_zeff()

        # Calculate Zeff with final impurity density profiles
        self.calc_zeff()

        # Add SXR radiation when data becomes available
        self.calc_rad_power()

        # Pressure profile
        self.calc_pressure()

        # Beta poloidal
        self.calc_beta_poloidal()

        # Resistivity and Vloop
        self.calc_vloop()

    def calc_ne_los_int(self):
        """
        Calculate line of sight integral assuming only one pass across the plasma

        Returns
        -------

        """
        x2_name = self.ne_l.attrs["transform"].x2_name

        el_dens = xr.where(
            self.ne_l.attrs["rho"] <= 1,
            self.el_dens.interp(rho_poloidal=self.ne_l.attrs["rho"]),
            0,
        )
        el_dens_int = el_dens.sum(x2_name) * self.ne_l.attrs["dl"]

        return el_dens_int

    def calc_main_ion_dens(self, fast_dens=True):
        """
        Calculate main ion density from quasi-neutrality given electron and impurity densities

        Parameters
        ----------
        fast_dens
            Include fast ion density in calculation
        """

        ion_dens_meanz = self.ion_dens * self.meanz
        main_ion_dens = deepcopy(self.el_dens)
        for elem in self.impurities:
            main_ion_dens -= ion_dens_meanz.loc[dict(element=elem)]

        if fast_dens is True:
            main_ion_dens -= self.fast_dens

        self.ion_dens.loc[dict(element=self.main_ion)] = main_ion_dens

    def calc_meanz(self):
        """
        Calculate mean charge
        """
        for elem in self.elements:
            meanz_tmp = (
                self.atomic_data[elem]["meanz"]
                .interp(electron_temperature=self.el_temp, method="cubic",)
                .drop_vars(["electron_temperature"])
            )
            self.meanz.loc[dict(element=elem)] = meanz_tmp

    def calc_pressure(self):
        """
        Calculate pressure profiles (thermal and total), MHD and diamagnetic energies
        """
        p_el = ph.calc_pressure(self.el_dens.values, self.el_temp.values)

        p_ion = ph.calc_pressure(
            self.ion_dens.sel(element=self.main_ion).values,
            self.ion_temp.sel(element=self.main_ion).values,
        )
        for elem in self.impurities:
            p_ion += ph.calc_pressure(
                self.ion_dens.sel(element=elem).values,
                self.ion_temp.sel(element=elem).values,
            )
        p_fast = ph.calc_pressure(self.fast_dens.values, self.fast_temp.values)

        self.pressure_th.values = p_el + p_ion
        self.pressure_tot.values = p_el + p_ion + p_fast

        for t in self.time:
            self.pth.loc[dict(t=t)] = np.trapz(
                self.pressure_th.sel(t=t), self.volume.sel(t=t)
            )
            self.ptot.loc[dict(t=t)] = np.trapz(
                self.pressure_tot.sel(t=t), self.volume.sel(t=t)
            )

        self.wmhd.values = 3 / 2 * self.ptot
        self.wdia.values = 3 / 2 * self.pth

    def calc_zeff(self):
        """
        Calculate Zeff including all ion species
        """
        for elem in self.elements:
            self.zeff.loc[dict(element=elem)] = (
                self.ion_dens.sel(element=elem) * self.meanz.sel(element=elem) ** 2
            ) / self.el_dens

    def calc_vloop(self):
        """
        Given Zeff, Te and Ne: calculate resistivity and Vloop
        """

        self.conductivity = ph.conductivity_neo(
            self.el_dens,
            self.el_temp,
            self.zeff.sum("element"),
            self.min_r,
            self.r_a,
            self.R_mag,
            self.q_prof,
            approx="sauter",
        )
        for t in self.time:
            resistivity = 1.0 / self.conductivity.sel(t=t)
            ir = np.where(np.isfinite(resistivity))

            j_phi = self.j_phi.sel(t=t)
            area = self.area.sel(t=t)

            vloop = ph.vloop(resistivity[ir], j_phi[ir], area[ir])

            self.vloop.loc[dict(t=t)] = vloop

    def calc_rad_power(self):
        """
        Calculate total and SXR filtered radiated power
        """
        for elem in self.elements:
            tot_rad_tmp = (
                self.atomic_data[elem]["lz_tot"]
                .sum("ion_charges")
                .interp(electron_temperature=self.el_temp, method="cubic")
                * self.el_dens
                * self.ion_dens.sel(element=elem)
            )
            self.tot_rad.loc[dict(element=elem)] = tot_rad_tmp

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
                self.ion_dens.loc[dict(element=elem)] = ion_dens_tmp

    def build_current_density(self, centr=0.0, sigm=0.3):
        """
        Build current density profile (A/m**2) given the total plasma current,
        plasma geometry and a shape parameter
        """

        for t in self.time:
            rho = self.rho.values
            ipla = self.ipla.sel(t=t).values
            r_a = self.r_a.sel(t=t).values
            area = self.area.sel(t=t).values

            j_phi = ph.current_density(ipla, rho, r_a, area, centr=centr, sigm=sigm)

            self.j_phi.loc[dict(t=t)] = j_phi

    def calc_magnetic_field(self):
        """
        Calculate magnetic field profiles (poloidal & toroidal)
        """

        for t in self.time:
            R_bt_0 = self.R_bt_0.values
            R_mag = self.R_mag.sel(t=t).values
            ipla = self.ipla.sel(t=t).values
            bt_0 = self.bt_0.sel(t=t).values
            maj_r_lfs = self.maj_r_lfs.sel(t=t).values
            maj_r_hfs = self.maj_r_hfs.sel(t=t).values
            j_phi = self.j_phi.sel(t=t).values
            r_a = self.r_a.sel(t=t).values
            min_r = self.min_r.sel(t=t).values
            volume = self.volume.sel(t=t).values
            area = self.area.sel(t=t).values

            self.b_tor_lfs.loc[dict(t=t)] = ph.toroidal_field(bt_0, R_bt_0, maj_r_lfs)
            self.b_tor_hfs.loc[dict(t=t)] = ph.toroidal_field(bt_0, R_bt_0, maj_r_hfs)

            b_pol = ph.poloidal_field(j_phi, min_r, area)
            self.b_pol.loc[dict(t=t)] = b_pol
            self.l_i.loc[dict(t=t)] = ph.internal_inductance(
                b_pol, ipla, volume, approx=2, R_mag=R_mag
            )

            b_tor = ((self.b_tor_lfs.sel(t=t) + self.b_tor_hfs.sel(t=t)) / 2.0).values

            self.q_prof.loc[dict(t=t)] = ph.safety_factor(
                b_tor, b_pol, min_r, r_a, R_mag
            )

    def calc_beta_poloidal(self):
        """
        Calculate Beta poloidal

        ??? Use total or thermal pressure ???
        """

        for t in self.time:
            rho = self.rho.values
            ir = np.where(rho <= 1)
            b_pol = self.b_pol.sel(t=t).values
            pressure = self.pressure_tot.sel(t=t).values
            volume = self.volume.sel(t=t).values

            self.beta_pol.loc[dict(t=t)] = ph.beta_poloidal(b_pol, pressure, volume)

    def propagate_ion_dens(self, fast_dens=False):
        """
        After having modified anything in the ion_density data, propagate the result to all
        other variables depending on it
        """
        self.calc_main_ion_dens(fast_dens=fast_dens)
        self.impose_flat_zeff()
        self.calc_pressure()
        self.calc_zeff()
        self.calc_vloop()
        self.calc_beta_poloidal()
        self.calc_rad_power()

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
                    fz_tmp.loc[dict(rho_poloidal=rho)] = (fz_tmp * gauss).sum(
                        "rho_poloidal"
                    )
                for ir, rho in enumerate(self.rho):
                    norm = np.nansum(fz_tmp.sel(rho_poloidal=rho), axis=0)
                    fz_tmp.loc[dict(rho_poloidal=rho)] = fz_tmp / norm
                    fz_transp.loc[dict(t=t)] = fz_tmp

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

    def simulate_spectrometers(self):
        self.spectrometers = {}
        if "princeton" in self.raw_data.keys():
            geometry = deepcopy(self.ti_princeton.attrs)
            del geometry["datatype"]
            del geometry["error"]

            self.spectrometers["passive_c5"] = Spectrometer(
                self.ADASReader,
                "c",
                "5",
                transition="n=8-n=7",
                wavelength=5292.7,
                geometry=geometry,
            )
            for te in (
                self.spectrometers["passive_c5"].atomdat["pec"].electron_temperature
            ):
                if te < 150 or te > 3000:
                    self.spectrometers["passive_c5"].atomdat["pec"].loc[
                        {"electron_temperature": te}
                    ] = 0

        if "xrcs" in self.raw_data.keys():
            geometry = deepcopy(self.te_xrcs.attrs)
            del geometry["datatype"]
            del geometry["error"]

            self.spectrometers["he_like"] = Spectrometer(
                self.ADASReader,
                "ar",
                "16",
                transition="(1)1(1.0)-(1)0(0.0)",
                wavelength=4.0,
                geometry=geometry,
            )

        for k in self.spectrometers.keys():
            print(
                self.spectrometers[k].element,
                self.spectrometers[k].charge,
                self.spectrometers[k].wavelength,
            )
            self.spectrometers[k].simulate_measurements(
                self.el_dens, self.el_temp, self.ion_temp.sel(element=self.main_ion),
            )

    def initialize_variables(self, inputs):
        """
        Initialize all class attributes

        Assign elements, machine dimensions and coordinates used throughout the analysis
            rho
            time
            theta
        """

        # Assign attributes
        self.elements = inputs["elements"]
        self.machine_dimensions = inputs["machine_dimensions"]
        self.machine_R = np.linspace(
            self.machine_dimensions[0][0], self.machine_dimensions[0][1], 100
        )
        self.machine_z = np.linspace(
            self.machine_dimensions[1][0], self.machine_dimensions[1][1], 100
        )

        nt = len(inputs["t"])
        nr = len(inputs["rho"])
        nel = len(inputs["elements"])
        nth = len(inputs["theta"])

        coords_radius = (inputs["rho_type"], inputs["rho"])
        coords_theta = ("poloidal_angle", inputs["theta"])
        coords_time = ("t", inputs["t"])
        coords_elem = ("element", list(inputs["elements"]))

        data0d = DataArray(0.0)
        data1d_theta = DataArray(np.zeros(nth), coords=[coords_theta])
        data1d_time = DataArray(np.zeros(nt), coords=[coords_time])
        data1d_rho = DataArray(np.zeros(nr), coords=[coords_radius])
        data2d = DataArray(np.zeros((nt, nr)), coords=[coords_time, coords_radius])
        data3d = DataArray(
            np.zeros((nel, nt, nr)), coords=[coords_elem, coords_time, coords_radius]
        )

        self.time = deepcopy(data1d_time)
        self.time.values = inputs["t"]
        assign_datatype(self.time, ("t", "plasma"))
        self.freq = 1.0 / (self.time[1] - self.time[0]).values

        self.rho = deepcopy(data1d_rho)
        self.rho.values = inputs["rho"]
        assign_datatype(self.rho, ("rho", "poloidal"))
        self.profs = fac.Plasma_profs(self.rho.values)

        self.theta = deepcopy(data1d_theta)
        self.theta.values = inputs["theta"]
        assign_datatype(self.theta, ("angle", "poloidal"))

        self.ipla = deepcopy(data1d_time)
        assign_datatype(self.ipla, ("current", "plasma"))

        self.bt_0 = deepcopy(data1d_time)
        assign_datatype(self.bt_0, ("field", "toroidal"))

        self.R_bt_0 = deepcopy(data0d)
        self.R_bt_0.values = inputs["R_bt_0"]
        assign_datatype(self.R_bt_0, ("major_radius", "toroidal_field"))

        # Geometric major radius
        self.R_0 = deepcopy(data1d_time)
        assign_datatype(self.R_0, ("major_radius", "geometric"))

        self.R_mag = deepcopy(data1d_time)
        assign_datatype(self.R_mag, ("major_radius", "magnetic"))

        # Major radius array at midplane
        self.maj_r_lfs = deepcopy(data2d)
        assign_datatype(self.maj_r_lfs, ("radius", "major"))
        self.maj_r_hfs = deepcopy(data2d)
        assign_datatype(self.maj_r_hfs, ("radius", "major"))

        # Main plasma profiles
        self.ne_0 = deepcopy(data1d_time)
        assign_datatype(self.ne_0, ("density", "electron"))

        self.te_0 = deepcopy(data1d_time)
        assign_datatype(self.te_0, ("temperature", "electron"))

        self.ti_0 = deepcopy(data1d_time)
        assign_datatype(self.ti_0, ("temperature", "ion"))

        self.el_temp = deepcopy(data2d)
        assign_datatype(self.el_temp, ("temperature", "electron"))
        self.el_dens = deepcopy(data2d)
        assign_datatype(self.el_dens, ("density", "electron"))

        # Other geometrical quantities
        self.min_r = deepcopy(data2d)
        assign_datatype(self.min_r, ("radius", "minor"))
        self.volume = deepcopy(data2d)
        assign_datatype(self.volume, ("volume", "plasma"))
        self.area = deepcopy(data2d)
        assign_datatype(self.area, ("area", "plasma"))

        self.r_a = deepcopy(data1d_time)
        assign_datatype(self.r_a, ("radius", "minor"))
        self.r_b = deepcopy(data1d_time)
        assign_datatype(self.r_b, ("radius", "minor"))
        self.r_c = deepcopy(data1d_time)
        assign_datatype(self.r_c, ("radius", "minor"))
        self.r_d = deepcopy(data1d_time)
        assign_datatype(self.r_d, ("radius", "minor"))

        self.kappa = deepcopy(data1d_time)
        assign_datatype(self.kappa, ("elongation", "plasma"))
        self.delta = deepcopy(data1d_time)
        assign_datatype(self.delta, ("triangularity", "plasma"))

        # Fast particle density and temperature
        self.fast_temp = deepcopy(data2d)
        assign_datatype(self.fast_temp, ("temperature", "fast"))

        self.fast_dens = deepcopy(data2d)
        assign_datatype(self.fast_dens, ("density", "fast"))

        # Current density, poloidal field, li, resistivity, vloop, q-profile
        self.j_phi = deepcopy(data2d)
        assign_datatype(self.j_phi, ("current", "density"))
        self.b_pol = deepcopy(data2d)
        assign_datatype(self.b_pol, ("field", "poloidal"))
        self.b_tor_lfs = deepcopy(data2d)
        assign_datatype(self.b_tor_lfs, ("field", "toroidal"))
        self.b_tor_hfs = deepcopy(data2d)
        assign_datatype(self.b_tor_hfs, ("field", "toroidal"))
        self.q_prof = deepcopy(data2d)
        assign_datatype(self.q_prof, ("factor", "safety"))
        self.conductivity = deepcopy(data2d)
        assign_datatype(self.conductivity, ("conductivity", "plasma"))
        self.l_i = deepcopy(data1d_time)
        assign_datatype(self.l_i, ("inductance", "internal"))
        self.vloop = deepcopy(data1d_time)
        assign_datatype(self.vloop, ("voltage", "loop"))

        # Ion densities
        self.main_ion = self.elements[0]
        self.impurities = self.elements[1:]

        self.ion_conc = DataArray(np.zeros(len(self.elements)), coords=[coords_elem])
        assign_datatype(self.ion_conc, ("concentration", "ion"))
        self.ion_conc.values = np.array(inputs["ion_conc"])
        self.ion_dens = deepcopy(data3d)
        assign_datatype(self.ion_dens, ("density", "ion"))
        self.ion_temp = deepcopy(data3d)
        assign_datatype(self.ion_temp, ("temperature", "ion"))
        self.vtor = deepcopy(data3d)
        assign_datatype(self.vtor, ("temperature", "ion"))
        self.meanz = deepcopy(data3d)
        assign_datatype(self.meanz, ("charge", "mean"))

        self.zeff = deepcopy(data3d)
        assign_datatype(self.zeff, ("charge", "effective"))

        self.tot_rad = deepcopy(data3d)
        assign_datatype(self.tot_rad, ("emission", "total"))
        self.sxr_rad = deepcopy(data3d)
        assign_datatype(self.sxr_rad, ("emission", "sxr"))

        self.pressure_th = deepcopy(data2d)
        assign_datatype(self.pressure_th, ("pressure", "thermal"))
        self.pressure_tot = deepcopy(data2d)
        assign_datatype(self.pressure_tot, ("pressure", "total"))

        self.pth = deepcopy(data1d_time)
        assign_datatype(self.pth, ("pressure", "thermal"))
        self.ptot = deepcopy(data1d_time)
        assign_datatype(self.ptot, ("pressure", "total"))

        self.wmhd = deepcopy(data1d_time)
        assign_datatype(self.wmhd, ("energy", "total"))
        self.wdia = deepcopy(data1d_time)
        assign_datatype(self.wmhd, ("energy", "diamagnetic"))
        self.beta_pol = deepcopy(data1d_time)
        assign_datatype(self.wmhd, ("beta", "poloidal"))

    def write_to_pickle(self):

        with open(f"data_{self.pulse}.pkl", "wb") as f:
            pickle.dump(
                self, f,
            )


def assign_datatype(data_array: DataArray, datatype: tuple):
    data_array.name = f"{datatype[1]}_{datatype[0]}"
    data_array.attrs["datatype"] = datatype
