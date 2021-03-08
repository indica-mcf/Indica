import matplotlib.pylab as plt
import numpy as np
import snippets.fac_profiles as fac
import snippets.forward_models as forward_models
import xarray
from snippets.atomdat import fractional_abundance
from snippets.atomdat import radiated_power
from indica.numpy_typing import ArrayLike
import matplotlib.cm as cm
from copy import deepcopy

plt.ion()


class simulate_spectra:
    def __init__(self, te_0=np.arange(0.75, 4, 0.75) * 1.0e3, tau=1):
        self.te_0 = te_0
        self.tau = tau
        self.profs = fac.main_plasma_profs()

        # Build fake, time-evolving temperature and density profiles
        el_temp = []
        el_dens = []
        time = np.linspace(0, 0.1, len(te_0))
        for i, t in enumerate(time):
            self.profs.l_mode(te_0=te_0[i])
            el_temp.append(self.profs.te)
            el_dens.append(self.profs.ne)
        self.el_temp = xarray.concat(el_temp, dim="time")
        self.el_dens = xarray.concat(el_dens, dim="time")
        self.el_temp = self.el_temp.assign_coords(time=time)
        self.el_dens = self.el_dens.assign_coords(time=time)

        self.passive_c5 = forward_models.spectrometer_passive_c5()
        self.he_like = forward_models.spectrometer_he_like()
        self.he_like.ti = te_0 / 3
        self.passive_c5.ti = self.he_like.ti / 4

        self.he_like.sim = self.scan_parameters(self.he_like, tau=tau)
        self.passive_c5.sim = self.scan_parameters(self.passive_c5, tau=tau)

        # Build ion temperature profiles based on the two measurements
        # taking Ti(0) from He-like spectrometer estimated temperature, Te(ped) from passive C5+
        x_ped = 0.85
        ion_temp = []
        for i in range(len(time)):
            ti_0 = self.he_like.sim["ion_temp"][i][0].values
            ti_ped = (
                self.passive_c5.sim["ion_temp"][i]
                .sel(rho_poloidal=x_ped, method="nearest")
                .values
            )
            ion_temp.append(
                self.profs.build_temperature(
                    y_0=ti_0, y_ped=ti_ped, x_ped=x_ped, datatype=("temperature", "ion")
                )
            )
        self.ion_temp = ion_temp

        # Calculate total pressure assuming Zeff = 1
        # Implement modules with transformations eV <--> K, nm <--> eV, as well as list of physical constants
        self.ion_dens = deepcopy(self.el_dens)
        self.pressure = (
            (self.el_dens * self.el_temp + self.ion_dens * self.ion_temp)
            * 11604
            * (1.38 * 10 ** -23)
        )  # (eV-->K) * k_B

    def plot_sim(self, name="", save_fig=False):
        nt = self.el_temp.coords["time"].size
        colors = cm.rainbow(np.linspace(0, 1, nt))

        plt.figure()
        for i in range(nt):
            self.el_temp[i].plot(color=colors[i])
        plt.title("Electron Temperature")
        plt.ylabel("(eV)")
        plt.xlabel(r"$\rho_{pol}$")
        if save_fig:
            save_figure(fig_name=name + "_electron_temperature")

        plt.figure()
        labels = {
            "ion_temp": "$T_i$",
            "he_like": "$T_i(He-like)$",
            "passive_c5": "$T_i(Passive C5+)$",
        }
        for i in range(nt):
            if i > 0:
                for k in labels.keys():
                    labels[k] = None
            self.ion_temp[i].plot(
                color=colors[i], linestyle="--", label=labels["ion_temp"]
            )

            plt.scatter(
                self.he_like.sim["pos"][i],
                self.he_like.ti[i],
                color=colors[i],
                label=labels["he_like"],
            )
            plt.hlines(
                self.he_like.ti[i],
                self.he_like.sim["pos"][i] - self.he_like.sim["pos_err"]["in"][i],
                self.he_like.sim["pos"][i] + self.he_like.sim["pos_err"]["out"][i],
                alpha=0.5,
                color=colors[i],
            )

            plt.scatter(
                self.passive_c5.sim["pos"][i],
                self.passive_c5.ti[i],
                color=colors[i],
                marker="x",
                label=labels["passive_c5"],
            )
            plt.hlines(
                self.passive_c5.ti[i],
                self.passive_c5.sim["pos"][i] - self.passive_c5.sim["pos_err"]["in"][i],
                self.passive_c5.sim["pos"][i]
                + self.passive_c5.sim["pos_err"]["out"][i],
                alpha=0.5,
                color=colors[i],
            )
        plt.title("Ion Temperature")
        plt.ylabel("(eV)")
        plt.xlabel(r"$\rho_{pol}$")
        plt.legend()
        if save_fig:
            save_figure(fig_name=name + "_ion_temperature")

        plt.figure()
        labels = {
            "el_temp": "$T_e$",
            "ion_temp": "$T_i$",
            "he_like": "$T_i(He-like)$",
            "passive_c5": "$T_i(Passive C5+)$",
        }
        for i in range(nt):
            if i > 0:
                for k in labels.keys():
                    labels[k] = None
            self.el_temp[i].plot(color=colors[i], label=labels["el_temp"])
            self.ion_temp[i].plot(
                color=colors[i], linestyle="--", label=labels["ion_temp"]
            )

            plt.scatter(
                self.he_like.sim["pos"][i],
                self.he_like.ti[i],
                color=colors[i],
                label=labels["he_like"],
            )
            plt.hlines(
                self.he_like.ti[i],
                self.he_like.sim["pos"][i] - self.he_like.sim["pos_err"]["in"][i],
                self.he_like.sim["pos"][i] + self.he_like.sim["pos_err"]["out"][i],
                alpha=0.5,
                color=colors[i],
            )

            plt.scatter(
                self.passive_c5.sim["pos"][i],
                self.passive_c5.ti[i],
                color=colors[i],
                marker="x",
                label=labels["passive_c5"],
            )
            plt.hlines(
                self.passive_c5.ti[i],
                self.passive_c5.sim["pos"][i] - self.passive_c5.sim["pos_err"]["in"][i],
                self.passive_c5.sim["pos"][i]
                + self.passive_c5.sim["pos_err"]["out"][i],
                alpha=0.5,
                color=colors[i],
            )
        plt.title("Temperature")
        plt.ylabel("(eV)")
        plt.xlabel(r"$\rho_{pol}$")
        plt.legend()
        if save_fig:
            save_figure(fig_name=name + "_temperature")

        plt.figure()
        for i in range(nt):
            self.el_dens[i].plot(color=colors[i])
        plt.title("Electron Density")
        plt.ylabel("($m^{-3}$)")
        plt.xlabel(r"$\rho_{pol}$")
        if save_fig:
            save_figure(fig_name=name + "_density")

        plt.figure()
        for i in range(nt):
            self.pressure[i].plot(color=colors[i])
        plt.title("Thermal pressure (ion + electrons)")
        plt.ylabel("($Pa$)")
        plt.xlabel(r"$\rho_{pol}$")
        if save_fig:
            save_figure(fig_name=name + "_thermal_pressure")

        for results in [self.he_like, self.passive_c5]:
            titles = {
                "fz": f"{results.element}{results.charge}+ fractional abundance",
                "pec": f"{results.element}{results.charge}+ {results.wavelength}A PEC",
                "emiss": f"{results.element}{results.charge}+ {results.wavelength}A emission shell",
                "tot_rad_pow": f"{results.element} total radiated power",
            }

            plt.figure()
            for tmp in results.sim["fz"]:
                plt.plot(
                    tmp.coords["rho_poloidal"], tmp.transpose(), alpha=0.2, color="k"
                )
            for i, tmp in enumerate(results.sim["fz"]):
                tmp.sel(ion_charges=results.charge).plot(
                    linestyle="--", color=colors[i]
                )
            plt.title(titles["fz"])
            plt.ylabel("")
            plt.xlabel(r"$\rho_{pol}$")
            if save_fig:
                save_figure(fig_name=name + f"_{results.element}_fract_abu")

            plt.figure()
            for i, tmp in enumerate(results.sim["emiss"]):
                (tmp / tmp.max()).plot(color=colors[i])
            plt.title(titles["emiss"])
            plt.ylabel("")
            plt.xlabel(r"$\rho_{pol}$")
            if save_fig:
                save_figure(
                    fig_name=name
                    + f"_{results.element}{results.charge}_emission_ragion"
                )

            plt.figure()
            for i, tmp in enumerate(results.sim["tot_rad_pow"]):
                (tmp * self.el_dens[i, :] ** 2).plot(color=colors[i])
            plt.title(titles["tot_rad_pow"])
            plt.ylabel("(W)")
            plt.xlabel(r"$\rho_{pol}$")
            if save_fig:
                save_figure(fig_name=name + f"_{results.element}_total_radiated_power")

        plt.figure()
        plt.plot(self.te_0, np.array(self.he_like.sim["pos"]), "k", alpha=0.5)
        plt.fill_between(
            self.te_0,
            np.array(self.he_like.sim["pos"])
            - np.array(self.he_like.sim["pos_err"]["in"]),
            np.array(self.he_like.sim["pos"])
            + np.array(self.he_like.sim["pos_err"]["out"]),
            label="He-like",
            alpha=0.5,
        )
        plt.plot(self.te_0, np.array(self.passive_c5.sim["pos"]), "k", alpha=0.5)
        plt.fill_between(
            self.te_0,
            np.array(self.passive_c5.sim["pos"])
            - np.array(self.passive_c5.sim["pos_err"]["in"]),
            np.array(self.passive_c5.sim["pos"])
            + np.array(self.passive_c5.sim["pos_err"]["out"]),
            label="Passive C5+",
            alpha=0.5,
        )
        plt.ylim(0, 1)
        plt.ylabel(r"$\rho_{pol}$")
        plt.xlabel("$T_e(0)$ (eV)")
        plt.legend()
        if save_fig:
            save_figure(fig_name=name + f"_emission_locations")

        plt.figure()
        plt.plot(self.te_0, self.te_0, "k--", label="$T_e(0)$")
        plt.plot(self.te_0, self.he_like.sim["el_temp"])
        plt.fill_between(
            self.te_0,
            np.array(self.he_like.sim["el_temp"])
            - np.array(self.he_like.sim["el_temp_err"]["in"]),
            np.array(self.he_like.sim["el_temp"])
            + np.array(self.he_like.sim["el_temp_err"]["out"]),
            alpha=0.5,
            label="$T_e(He-like)$",
        )
        plt.ylabel(r"$T_e$ (eV)")
        plt.xlabel(r"$T_e$ (eV)")
        plt.legend()
        if save_fig:
            save_figure(fig_name=name + f"_electron_temperature_center")

    def scan_parameters(self, spectrometer_model, tau=1.0):
        sim_results = {
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

        scd = spectrometer_model.atomdat["scd"]
        acd = spectrometer_model.atomdat["acd"]
        pec = spectrometer_model.atomdat["pec"]

        # Use average density as reference for interpolation of atomic data
        dens_tmp = self.el_dens.mean()

        # Interpolate on electron density and drop dimension
        pec = pec.interp(electron_density=dens_tmp, method="nearest")

        scd = scd.interp(log10_electron_density=np.log10(dens_tmp), method="nearest")
        acd = acd.interp(log10_electron_density=np.log10(dens_tmp), method="nearest")
        drop = ["log10_electron_density"]
        acd = acd.drop_vars(drop)
        scd = scd.drop_vars(drop)

        fz = fractional_abundance(
            scd, acd, ne_tau=tau, element=spectrometer_model.element
        )

        line_rad = spectrometer_model.atomdat["plt"].interp(
            log10_electron_density=np.log10(dens_tmp), method="nearest"
        )
        recomb_rad = spectrometer_model.atomdat["prb"].interp(
            log10_electron_density=np.log10(dens_tmp), method="nearest"
        )
        line_rad = line_rad.drop_vars(drop)
        recomb_rad = recomb_rad.drop_vars(drop)

        tot_rad_pow_fz = radiated_power(
            line_rad, recomb_rad, fz, element=spectrometer_model.element
        )
        tot_rad_pow = tot_rad_pow_fz.sum(axis=0)

        print(
            spectrometer_model.element,
            spectrometer_model.charge,
            spectrometer_model.wavelength,
        )
        for i, t in enumerate(self.el_temp.coords["time"]):
            dens_tmp = self.el_dens
            temp_tmp = self.el_temp.sel(time=t)
            print(f"Te(0) = {temp_tmp.max().values}")

            pec_tmp = pec.interp(electron_temperature=temp_tmp, method="quadratic")
            fz_tmp = fz.interp(
                log10_electron_temperature=np.log10(temp_tmp), method="quadratic"
            )
            tot_rad_pow_tmp = tot_rad_pow.interp(
                log10_electron_temperature=np.log10(temp_tmp), method="cubic"
            )

            # Profile of the emission region
            emiss_tmp = (
                pec_tmp
                * fz_tmp.sel(ion_charges=spectrometer_model.charge)
                * self.el_dens[i] ** 2
            )
            emiss_tmp.name = (
                f"{spectrometer_model.element}{spectrometer_model.charge}+ "
                f"{spectrometer_model.wavelength} A emission region"
            )

            emiss_tmp[emiss_tmp < 0] = 0.0

            # Measurement position average ad "error"
            pos_avrg, pos_err_in, pos_err_out, ind_in, ind_out = calc_moments(
                emiss_tmp, rho, sim=True
            )
            te_avrg, te_err_in, te_err_out, _, _ = calc_moments(
                emiss_tmp, temp_tmp, ind_in=ind_in, ind_out=ind_out, sim=True
            )
            ti_prof = temp_tmp * spectrometer_model.ti[i] / te_avrg
            ti_err_in = np.abs(
                ti_prof - temp_tmp * spectrometer_model.ti[i] / (te_avrg - te_err_in)
            )
            ti_err_out = np.abs(
                ti_prof - temp_tmp * spectrometer_model.ti[i] / (te_avrg + te_err_in)
            )

            # ne_avrg, ne_err_in, ne_err_out, _, _ = calc_moments(emiss_tmp, dens_tmp,
            #                                                     ind_in=ind_in, ind_out=ind_out)

            sim_results["pec"].append(pec_tmp)
            sim_results["emiss"].append(emiss_tmp)
            sim_results["fz"].append(fz_tmp)
            sim_results["tot_rad_pow"].append(tot_rad_pow_tmp)
            sim_results["pos"].append(pos_avrg)
            sim_results["pos_err"]["in"].append(pos_err_in)
            sim_results["pos_err"]["out"].append(pos_err_out)
            sim_results["el_temp"].append(te_avrg)
            sim_results["el_temp_err"]["in"].append(te_err_in)
            sim_results["el_temp_err"]["out"].append(te_err_out)
            sim_results["ion_temp"].append(ti_prof)
            sim_results["ion_temp_err"]["in"].append(ti_err_in)
            sim_results["ion_temp_err"]["out"].append(ti_err_out)
            # sim_results["el_dens"].append(ne_avrg)
            # sim_results["el_dens_err"]["in"].append(ne_err_in)
            # sim_results["el_dens_err"]["out"].append(ne_err_out)

        return sim_results


def save_figure(fig_name="", orientation="landscape", ext=".jpg"):
    plt.savefig(
        "figures/" + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def calc_moments(y: ArrayLike, x: ArrayLike, ind_in=None, ind_out=None, sim=False):
    x_avrg = np.nansum(y * x) / np.nansum(y)

    if (ind_in is None) and (ind_out is None):
        ind_in = x <= x_avrg
        ind_out = x >= x_avrg
        if sim == True:
            ind_in = ind_in + ind_out
            ind_out = ind_in

    x_err_in = np.sqrt(
        np.nansum(y[ind_in] * (x[ind_in] - x_avrg) ** 2) / np.nansum(y[ind_in])
    )

    x_err_out = np.sqrt(
        np.nansum(y[ind_out] * (x[ind_out] - x_avrg) ** 2) / np.nansum(y[ind_out])
    )

    return x_avrg, x_err_in, x_err_out, ind_in, ind_out
