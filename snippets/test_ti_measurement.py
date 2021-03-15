from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
from snippets.atomdat import fractional_abundance
from snippets.atomdat import radiated_power
import snippets.fac_profiles as fac
import snippets.forward_models as forward_models
import xarray

from indica.numpy_typing import ArrayLike

plt.ion()


class simulate_spectra:
    def __init__(self, te_0=np.arange(0.75, 4, 0.75) * 1.0e3, tau=1):
        self.te_0 = te_0
        self.tau = tau
        self.profs = fac.main_plasma_profs()

        # Build fake, time-evolving temperature and density profiles
        el_temp = []
        el_dens = []
        volume = []
        self.time = np.linspace(0, 0.1, len(te_0))
        for i, t in enumerate(self.time):
            self.profs.l_mode(te_0=te_0[i])
            el_temp.append(self.profs.te)
            el_dens.append(self.profs.ne)
            volume.append(self.profs.te.coords["rho_poloidal"] ** 2 * 3.0)
        self.el_temp = xarray.concat(el_temp, dim="time")
        self.el_dens = xarray.concat(el_dens, dim="time")
        self.volume = xarray.concat(volume, dim="time")
        self.el_temp = self.el_temp.assign_coords(time=self.time)
        self.el_dens = self.el_dens.assign_coords(time=self.time)
        self.volume = self.volume.assign_coords(time=self.time)

        # Invent ion temperature and assume Zeff = 1 for ion density
        self.ion_temp = deepcopy(self.el_temp) / 2.0
        self.ion_dens = deepcopy(self.el_dens)

        pressure = []
        ptot = []
        for t in self.time:
            p_el, ptot_el = calc_pressure(
                self.el_dens.sel(time=t),
                self.el_temp.sel(time=t),
                volume=self.volume.sel(time=t),
            )
            p_ion, ptot_ion = calc_pressure(
                self.ion_dens.sel(time=t),
                self.ion_temp.sel(time=t),
                volume=self.volume.sel(time=t),
            )
            pressure.append(p_el + p_ion)
            ptot.append(ptot_el + ptot_ion)
        self.pressure = xarray.concat(pressure, dim="time")
        self.ptot = xarray.concat(ptot, dim="time")

        # Calculate what the spectrometers are going to measure scanning parameters
        self.passive_c5 = forward_models.spectrometer_passive_c5(el_dens=5.0e19)
        self.he_like = forward_models.spectrometer_he_like(el_dens=5.0e19)

        self.he_like.exp = self.scan_parameters(self.he_like, tau=tau)
        self.passive_c5.exp = self.scan_parameters(self.passive_c5, tau=tau)

        # Try to simulate profiles from experimental measurements
        self.sim = self.sim_values()

    def scan_parameters(self, spectrometer_model, tau=1.0):
        # Forward model of experimental measurements given input profiles
        results = {
            "fz": [],
            "pec": [],
            "emiss": [],
            "tot_rad_pow": [],
            "pos": [],
            "pos_err": {"in": [], "out": []},
            "el_temp": [],
            "el_temp_err": {"in": [], "out": []},
            "el_dens": [],
            "el_dens_err": {"in": [], "out": []},
            "ion_temp": [],
            "ion_temp_err": {"in": [], "out": []},
        }
        rho = self.el_dens.coords["rho_poloidal"]

        print(
            spectrometer_model.element,
            spectrometer_model.charge,
            spectrometer_model.wavelength,
        )
        for i, t in enumerate(self.time):
            el_dens = self.el_dens.sel(time=t)
            el_temp = self.el_temp.sel(time=t)
            ion_temp = self.ion_temp.sel(time=t)
            print(f"Te(0) = {el_temp.max().values}")

            fz, emiss, tot_rad_pow = self.radiation_characteristics(
                spectrometer_model, el_temp, el_dens, tau=tau
            )

            # Measurement position average ad "error"
            pos_avrg, pos_err_in, pos_err_out, ind_in, ind_out = calc_moments(
                emiss, rho, sim=True
            )
            te_avrg, te_err_in, te_err_out, _, _ = calc_moments(
                emiss, el_temp, ind_in=ind_in, ind_out=ind_out, sim=True
            )
            ti_avrg, ti_err_in, ti_err_out, _, _ = calc_moments(
                emiss, ion_temp, ind_in=ind_in, ind_out=ind_out, sim=True
            )

            ne_avrg, ne_err_in, ne_err_out, _, _ = calc_moments(
                emiss, el_dens, ind_in=ind_in, ind_out=ind_out, sim=True
            )

            results["emiss"].append(emiss)
            results["fz"].append(fz)
            results["tot_rad_pow"].append(tot_rad_pow)
            results["pos"].append(pos_avrg)
            results["pos_err"]["in"].append(pos_err_in)
            results["pos_err"]["out"].append(pos_err_out)
            results["el_temp"].append(te_avrg)
            results["el_temp_err"]["in"].append(te_err_in)
            results["el_temp_err"]["out"].append(te_err_out)
            results["ion_temp"].append(ti_avrg)
            results["ion_temp_err"]["in"].append(ti_err_in)
            results["ion_temp_err"]["out"].append(ti_err_out)
            results["el_dens"].append(ne_avrg)
            results["el_dens_err"]["in"].append(ne_err_in)
            results["el_dens_err"]["out"].append(ne_err_out)

        return results

    def sim_values(self, tau=1):
        """From the measured Te and Ti values, search for Te that best
        matches total pressure.

        Ti profile shape estimated from passive spectrometer measurements
        and spectral line emission(Te)

        Ne and plasma volume are taken as given, the former from
        interferometer <Ne> measurements and assuming a fixed profile shape,
        the latter from equilibrium reconstruction

        Treat each time-point independently...
        """

        results = {
            "pos": [],
            "pos_err": {"in": [], "out": []},
            "el_temp": [],
            "el_temp_err": {"in": [], "out": []},
            "ion_temp": [],
            "ion_temp_err": {"in": [], "out": []},
            "pressure": [],
            "ptot": [],
        }

        rho = self.el_dens.coords["rho_poloidal"]
        x_ped = 0.85
        for i, t in enumerate(self.time):
            print(f"time = {t}")

            # Known values from experiment
            volume = self.volume.sel(time=t)
            el_dens = self.el_dens.sel(time=t)
            ion_dens = self.ion_dens.sel(time=t)
            ptot = self.ptot.sel(time=t)

            # Start assuming Te(He-like) = Te(0)
            te_0 = self.he_like.exp["el_temp"][i]
            ti_he = self.he_like.exp["ion_temp"][i]
            ti_c = self.passive_c5.exp["ion_temp"][i]

            const = 1.0
            nrounds = 2
            for j in range(nrounds):
                self.profs.l_mode(te_0=te_0 * const)
                el_temp = self.profs.te

                # He-like emission characteristics with specified Te profile
                fz, emiss, tot_rad_pow = self.radiation_characteristics(
                    self.he_like, el_temp, el_dens, tau=tau
                )
                (
                    pos_avrg_he,
                    pos_err_in_he,
                    pos_err_out_he,
                    ind_in,
                    ind_out,
                ) = calc_moments(emiss, rho, sim=True)
                te_avrg_he, te_err_in_he, te_err_out_he, _, _ = calc_moments(
                    emiss, el_temp, ind_in=ind_in, ind_out=ind_out, sim=True
                )
                # Estimate of central ion temperature from He-like measurement
                ti_0 = (el_temp * ti_he / te_avrg_he).sel(rho_poloidal=0).values

                # Passive C5+ emission characteristics with specified Te profile
                fz_c, emiss_c, tot_rad_pow_c = self.radiation_characteristics(
                    self.passive_c5, el_temp, el_dens, tau=tau
                )
                pos_avrg_c, pos_err_in_c, pos_err_out_c, ind_in, ind_out = calc_moments(
                    emiss_c, rho, sim=True
                )
                te_avrg_c, te_err_in_c, te_err_out_c, _, _ = calc_moments(
                    emiss_c, el_temp, ind_in=ind_in, ind_out=ind_out, sim=True
                )
                # Estimate of pedestal ion temperature from passive C5+ measurement
                ti_ped = (el_temp * ti_c / te_avrg_c).sel(rho_poloidal=x_ped).values

                # Build ion temperature profile estimate using both measurements
                ion_temp = self.profs.build_temperature(
                    y_0=ti_0, y_ped=ti_ped, x_ped=x_ped, datatype=("temperature", "ion")
                )

                # Calculate estimate of total pressure
                p_el, ptot_el = calc_pressure(el_dens, el_temp, volume=volume)
                p_ion, ptot_ion = calc_pressure(ion_dens, ion_temp, volume=volume)
                pressure_sim = p_el + p_ion
                ptot_sim = ptot_ion + ptot_el

                # Compare with experimental value and rescale profiles
                dpth_tot = ptot - ptot_sim  # missing pressure in estimated value
                # const = (1 + dpth_tot / ptot_sim).values
                const = (1 + dpth_tot / ptot_el).values

            results["pos"].append([pos_avrg_he, pos_avrg_c])
            results["pos_err"]["in"].append([pos_err_in_he, pos_err_in_c])
            results["pos_err"]["out"].append([pos_err_out_he, pos_err_out_c])
            results["el_temp"].append(el_temp)
            results["ion_temp"].append(ion_temp)
            results["pressure"].append(pressure_sim)
            results["ptot"].append(ptot_sim)
        results["ptot"] = xarray.concat(results["ptot"], dim="time")

        return results

    def radiation_characteristics(self, spectrometer_model, el_temp, el_dens, tau=1.0):
        atomdat = deepcopy(spectrometer_model.atomdat)
        for k in atomdat.keys():
            # try:
            atomdat[k] = atomdat[k].interp(
                electron_temperature=el_temp, method="quadratic"
            )
            # except ValueError:
            #     atomdat[k] = atomdat[k].interp(
            #         electron_temperature=el_temp, method="quadratic"
            #     )

        fz = fractional_abundance(
            atomdat["scd"],
            atomdat["acd"],
            ne_tau=tau,
            element=spectrometer_model.element,
        )

        tot_rad_pow_fz = radiated_power(
            atomdat["plt"], atomdat["prb"], fz, element=spectrometer_model.element
        )
        tot_rad_pow = tot_rad_pow_fz.sum(axis=0)

        emiss = (
            atomdat["pec"]
            * fz.sel(ion_charges=spectrometer_model.charge)
            * el_dens ** 2
        )
        emiss.name = (
            f"{spectrometer_model.element}{spectrometer_model.charge}+ "
            f"{spectrometer_model.wavelength} A emission region"
        )
        emiss[emiss < 0] = 0.0

        return fz, emiss, tot_rad_pow

    def plot_res(self, name="", savefig=False):
        nt = self.time.size
        colors = cm.rainbow(np.linspace(0, 1, nt))

        # Electron temperature
        plt.figure()
        labels = get_labels(set_none=True)
        for i in range(nt):
            if i == (nt - 1):
                labels = get_labels()
            self.el_temp[i].plot(color=colors[i], label=labels["el_temp"])

            plt.scatter(
                self.he_like.exp["pos"][i],
                self.he_like.exp["el_temp"][i],
                color=colors[i],
                label=labels["he_like_te"],
            )
            plt.hlines(
                self.he_like.exp["el_temp"][i],
                self.he_like.exp["pos"][i] - self.he_like.exp["pos_err"]["in"][i],
                self.he_like.exp["pos"][i] + self.he_like.exp["pos_err"]["out"][i],
                alpha=0.5,
                color=colors[i],
            )

            if hasattr(self, "sim"):
                self.sim["el_temp"][i].plot(
                    color=colors[i], label=labels["el_temp_sim"], linestyle="--"
                )

        plt.title("Electron Temperature")
        plt.ylabel("(eV)")
        plt.xlabel(r"$\rho_{pol}$")
        plt.legend()
        ylim = plt.ylim()
        plt.ylim(ylim)
        if savefig:
            save_figure(fig_name=name + "_el_temperature")

        # Ion temperature
        plt.figure()
        labels = get_labels(set_none=True)
        for i in range(nt):
            if i == (nt - 1):
                labels = get_labels()
            self.ion_temp[i].plot(
                color=colors[i],
                label=labels["ion_temp"],
            )

            plt.scatter(
                self.he_like.exp["pos"][i],
                self.he_like.exp["ion_temp"][i],
                color=colors[i],
                label=labels["he_like_ti"],
            )
            plt.hlines(
                self.he_like.exp["ion_temp"][i],
                self.he_like.exp["pos"][i] - self.he_like.exp["pos_err"]["in"][i],
                self.he_like.exp["pos"][i] + self.he_like.exp["pos_err"]["out"][i],
                alpha=0.5,
                color=colors[i],
            )

            plt.scatter(
                self.passive_c5.exp["pos"][i],
                self.passive_c5.exp["ion_temp"][i],
                color=colors[i],
                marker="x",
                label=labels["passive_c5_ti"],
            )
            plt.hlines(
                self.passive_c5.exp["ion_temp"][i],
                self.passive_c5.exp["pos"][i] - self.passive_c5.exp["pos_err"]["in"][i],
                self.passive_c5.exp["pos"][i]
                + self.passive_c5.exp["pos_err"]["out"][i],
                alpha=0.5,
                color=colors[i],
            )

            if hasattr(self, "sim"):
                self.sim["ion_temp"][i].plot(
                    color=colors[i], label=labels["ion_temp_sim"], linestyle="--"
                )

        plt.title("Ion Temperature")
        plt.ylabel("(eV)")
        plt.xlabel(r"$\rho_{pol}$")
        plt.legend()
        plt.ylim(ylim)
        if savefig:
            save_figure(fig_name=name + "_ion_temperature")

        # Electron density
        plt.figure()
        for i in range(nt):
            self.el_dens[i].plot(color=colors[i])

        plt.title("Electron Density")
        plt.ylabel("($m^{-3}$)")
        plt.xlabel(r"$\rho_{pol}$")
        if savefig:
            save_figure(fig_name=name + "_density")

        plt.figure()
        labels = get_labels(set_none=True)
        for i in range(nt):
            if i == (nt - 1):
                labels = get_labels()
            self.pressure[i].plot(color=colors[i], label=labels["pressure"])
            if hasattr(self, "sim"):
                self.sim["pressure"][i].plot(
                    color=colors[i], label=labels["pressure_sim"], linestyle="--"
                )
        plt.title("Thermal pressure (ion + electrons)")
        plt.ylabel("($Pa$ $m^{-3}$)")
        plt.xlabel(r"$\rho_{pol}$")
        plt.legend()
        if savefig:
            save_figure(fig_name=name + "_thermal_pressure")

        plt.figure()
        ptot = self.ptot.assign_coords(
            electron_temperature=("time", self.el_temp.sel(rho_poloidal=0))
        )
        ptot = ptot.swap_dims({"time": "electron_temperature"})
        ptot_sim = self.sim["ptot"].assign_coords(
            electron_temperature=("time", self.el_temp.sel(rho_poloidal=0))
        )
        ptot_sim = ptot_sim.swap_dims({"time": "electron_temperature"})

        ptot.plot(color="k", linestyle="dashed", label=labels["ptot"])
        if hasattr(self, "sim"):
            ptot_sim.plot(label=labels["ptot_sim"])
        plt.title("Total thermal pressure (ion + electrons)")
        plt.ylabel("($Pa$)")
        plt.xlabel("Te(0) (eV)")
        plt.legend()
        if savefig:
            save_figure(fig_name=name + "_ptot_pressure")

        for results in [self.he_like, self.passive_c5]:
            titles = {
                "fz": f"{results.element}{results.charge}+ fractional abundance",
                "pec": f"{results.element}{results.charge}+ "
                f"{results.wavelength}A PEC",
                "emiss": f"{results.element}{results.charge}+ "
                f"{results.wavelength}A emission shell",
                "tot_rad_pow": f"{results.element} total radiated power",
            }

            plt.figure()
            for tmp in results.exp["fz"]:
                plt.plot(
                    tmp.coords["rho_poloidal"], tmp.transpose(), alpha=0.2, color="k"
                )
            for i, tmp in enumerate(results.exp["fz"]):
                tmp.sel(ion_charges=results.charge).plot(
                    linestyle="--", color=colors[i]
                )
            plt.title(titles["fz"])
            plt.ylabel("")
            plt.xlabel(r"$\rho_{pol}$")
            if savefig:
                save_figure(fig_name=name + f"_{results.element}_fract_abu")

            plt.figure()
            for i, tmp in enumerate(results.exp["emiss"]):
                (tmp / tmp.max()).plot(color=colors[i])
            plt.title(titles["emiss"])
            plt.ylabel("")
            plt.xlabel(r"$\rho_{pol}$")
            if savefig:
                save_figure(
                    fig_name=name
                    + f"_{results.element}{results.charge}_emission_ragion"
                )

            plt.figure()
            for i, tmp in enumerate(results.exp["tot_rad_pow"]):
                (tmp * self.el_dens[i, :] ** 2).plot(color=colors[i])
            plt.title(titles["tot_rad_pow"])
            plt.ylabel("(W)")
            plt.xlabel(r"$\rho_{pol}$")
            if savefig:
                save_figure(fig_name=name + f"_{results.element}_total_radiated_power")

        plt.figure()
        plt.plot(self.te_0, np.array(self.he_like.exp["pos"]), "k", alpha=0.5)
        plt.fill_between(
            self.te_0,
            np.array(self.he_like.exp["pos"])
            - np.array(self.he_like.exp["pos_err"]["in"]),
            np.array(self.he_like.exp["pos"])
            + np.array(self.he_like.exp["pos_err"]["out"]),
            label="He-like",
            alpha=0.5,
        )
        plt.plot(self.te_0, np.array(self.passive_c5.exp["pos"]), "k", alpha=0.5)
        plt.fill_between(
            self.te_0,
            np.array(self.passive_c5.exp["pos"])
            - np.array(self.passive_c5.exp["pos_err"]["in"]),
            np.array(self.passive_c5.exp["pos"])
            + np.array(self.passive_c5.exp["pos_err"]["out"]),
            label="Passive C5+",
            alpha=0.5,
        )
        plt.ylim(0, 1)
        plt.ylabel(r"$\rho_{pol}$")
        plt.xlabel("$T_e(0)$ (eV)")
        plt.legend()
        if savefig:
            save_figure(fig_name=name + "_emission_locations")

        plt.figure()
        plt.plot(self.te_0, self.te_0, "k--", label="$T_e(0)$")
        plt.plot(self.te_0, self.he_like.exp["el_temp"])
        plt.fill_between(
            self.te_0,
            np.array(self.he_like.exp["el_temp"])
            - np.array(self.he_like.exp["el_temp_err"]["in"]),
            np.array(self.he_like.exp["el_temp"])
            + np.array(self.he_like.exp["el_temp_err"]["out"]),
            alpha=0.5,
            label="$T_e(He-like)$",
        )
        plt.ylabel(r"$T_e$ (eV)")
        plt.xlabel(r"$T_e$ (eV)")
        plt.legend()
        if savefig:
            save_figure(fig_name=name + "_electron_temperature_center")


def save_figure(fig_name="", orientation="landscape", ext=".jpg"):
    plt.savefig(
        "figures/" + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def get_labels(set_none=False, lkey=None):

    labels = {
        "el_temp": "$T_e$",
        "el_temp_sim": "$T_e (simulated)$",
        "ion_temp": "$T_i$",
        "ion_temp_sim": "$T_i (simulated)$",
        "he_like_te": "$T_e$ (He-like)",
        "he_like_ti": "$T_i$ (He-like)",
        "he_like_ne": "$n_e$ (He-like)",
        "passive_c5_te": "$T_e$ (Passive C5+)",
        "passive_c5_ti": "$T_i$ (Passive C5+)",
        "passive_c5_ne": "$n_e$ (Passive C5+)",
        "pressure": "Real value",
        "pressure_sim": "Simulated",
        "ptot": "Real value",
        "ptot_sim": "Simulated",
    }

    if lkey is not None:
        if lkey in labels.keys():
            labels = labels[lkey]

    if set_none:
        for k in labels.keys():
            labels[k] = None

    return labels


def calc_moments(y: ArrayLike, x: ArrayLike, ind_in=None, ind_out=None, sim=False):
    x_avrg = np.nansum(y * x) / np.nansum(y)

    if (ind_in is None) and (ind_out is None):
        ind_in = x <= x_avrg
        ind_out = x >= x_avrg
        if sim is True:
            ind_in = ind_in + ind_out
            ind_out = ind_in

    x_err_in = np.sqrt(
        np.nansum(y[ind_in] * (x[ind_in] - x_avrg) ** 2) / np.nansum(y[ind_in])
    )

    x_err_out = np.sqrt(
        np.nansum(y[ind_out] * (x[ind_out] - x_avrg) ** 2) / np.nansum(y[ind_out])
    )

    return x_avrg, x_err_in, x_err_out, ind_in, ind_out


def calc_pressure(dens, temp, volume=None):
    ptot = None
    pressure = (dens * temp) * 11604 * (1.38 * 10 ** -23)  # (eV-->K) * k_B

    # Invent a plasma volume vs rho for volume integration of the pressures
    if volume is not None:
        pressure = pressure.assign_coords(volume=("rho_poloidal", volume))
        ptot = pressure.integrate("volume")

    return pressure, ptot
