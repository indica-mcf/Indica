import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm

plt.ion()


def HDAplot(data1, data2=None, label1="", label2="bckc", correl="t", plot_spectr=False, name="", savefig=False):
    """

    Parameters
    ----------
    hda
        class hda from hdaworkflow.py
    name
        initial string of plots filename
    savefig
        set to True if figures to be saved to file

    """
    data1.ne_0.values = data1.el_dens.sel(rho_poloidal=0)
    has_data2 = False
    if data2 is not None:
        data2.ne_0.values = data2.el_dens.sel(rho_poloidal=0)
        has_data2 = True

    time = data1.time
    nt = len(time)
    colors = cm.rainbow(np.linspace(0, 1, nt))

    # Electron temperature
    plt.figure()
    ldata = None
    ldata_he = None
    ldata2 = None
    ldata2_he = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("el_temp")
            ldata_he = ldata + get_labels("he_like")
            ldata2 = ldata + get_labels(label2)
            ldata2_he = ldata_he + get_labels(label2)
        data1.el_temp.sel(t=t).plot(color=colors[i], label=ldata)

        plt.scatter(
            data1.spectrometers["he_like"].pos[i],
            data1.spectrometers["he_like"].el_temp[i],
            color=colors[i],
            label=ldata_he,
        )
        plt.hlines(
            data1.spectrometers["he_like"].el_temp[i],
            data1.spectrometers["he_like"].pos[i]
            - data1.spectrometers["he_like"].pos.attrs["err_in"][i],
            data1.spectrometers["he_like"].pos[i]
            + data1.spectrometers["he_like"].pos.attrs["err_out"][i],
            alpha=0.5,
            color=colors[i],
        )

        if has_data2:
            data2.el_temp.sel(t=t).plot(color=colors[i], label=ldata2, linestyle="--")

            plt.scatter(
                data2.spectrometers["he_like"].pos[i],
                data2.spectrometers["he_like"].el_temp[i],
                color=colors[i],
                label=ldata2_he,
                facecolors="none",
            )
            plt.hlines(
                data2.spectrometers["he_like"].el_temp[i],
                data2.spectrometers["he_like"].pos[i]
                - data2.spectrometers["he_like"].pos.attrs["err_in"][i],
                data2.spectrometers["he_like"].pos[i]
                + data2.spectrometers["he_like"].pos.attrs["err_out"][i],
                alpha=0.5,
                color=colors[i],
            )

    plt.title("Electron Temperature")
    plt.ylabel("(eV)")
    plt.xlabel(r"$\rho_{pol}$")
    plt.legend()
    ylim = (0, np.max([data1.el_temp.max(), data1.ion_temp.max()]))
    if has_data2:
        ylim = (0, np.max([ylim[1], data2.el_temp.max(), data2.ion_temp.max()]))
    plt.ylim(ylim)
    if savefig:
        save_figure(fig_name=name + "_electron_temperature")

    # Ion temperature
    plt.figure()
    ldata = None
    ldata2 = None
    ldata_he = None
    ldata_c5 = None
    ldata2_he = None
    ldata2_c5 = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("ion_temp")
            ldata_he = ldata + get_labels("he_like")
            ldata_c5 = ldata + get_labels("passive_c5")
            ldata2 = ldata + get_labels(label2)
            ldata2_he = ldata_he + get_labels(label2)
            ldata2_c5 = ldata_c5 + get_labels(label2)
        data1.ion_temp.sel(t=t).sel(element=data1.main_ion).plot(
            color=colors[i], label=ldata,
        )

        if "he_like" in data1.spectrometers.keys():
            plt.scatter(
                data1.spectrometers["he_like"].pos[i],
                data1.spectrometers["he_like"].ion_temp[i],
                color=colors[i],
                label=ldata_he,
            )
            plt.hlines(
                data1.spectrometers["he_like"].ion_temp[i],
                data1.spectrometers["he_like"].pos[i]
                - data1.spectrometers["he_like"].pos.attrs["err_in"][i],
                data1.spectrometers["he_like"].pos[i]
                + data1.spectrometers["he_like"].pos.attrs["err_out"][i],
                alpha=0.5,
                color=colors[i],
            )

        if "passive_c5" in data1.spectrometers.keys():
            plt.scatter(
                data1.spectrometers["passive_c5"].pos[i],
                data1.spectrometers["passive_c5"].ion_temp[i],
                label=ldata_c5,
                marker="s",
            )
            plt.hlines(
                data1.spectrometers["passive_c5"].ion_temp[i],
                data1.spectrometers["passive_c5"].pos[i]
                - data1.spectrometers["passive_c5"].pos.attrs["err_in"][i],
                data1.spectrometers["passive_c5"].pos[i]
                + data1.spectrometers["passive_c5"].pos.attrs["err_out"][i],
                alpha=0.5,
                color=colors[i],
            )

        if has_data2:
            data2.ion_temp.sel(t=t).sel(element=data1.main_ion).plot(
                color=colors[i], label=ldata2, linestyle="--"
            )

            plt.scatter(
                data2.spectrometers["he_like"].pos[i],
                data2.spectrometers["he_like"].ion_temp[i],
                color=colors[i],
                facecolors="none",
                label=ldata2_he,
            )
            plt.hlines(
                data2.spectrometers["he_like"].ion_temp[i],
                data2.spectrometers["he_like"].pos[i]
                - data2.spectrometers["he_like"].pos.attrs["err_in"][i],
                data2.spectrometers["he_like"].pos[i]
                + data2.spectrometers["he_like"].pos.attrs["err_out"][i],
                alpha=0.5,
                color=colors[i],
            )

            if "passive_c5" in data2.spectrometers.keys():
                plt.scatter(
                    data2.spectrometers["passive_c5"].pos[i],
                    data2.spectrometers["passive_c5"].ion_temp[i],
                    color=colors[i],
                    marker="s",
                    facecolors="none",
                    label=ldata2_c5,
                )
                plt.hlines(
                    data2.spectrometers["passive_c5"].ion_temp[i],
                    data2.spectrometers["passive_c5"].pos[i]
                    - data2.spectrometers["passive_c5"].pos.attrs["err_in"][i],
                    data2.spectrometers["passive_c5"].pos[i]
                    + data2.spectrometers["passive_c5"].pos.attrs["err_out"][i],
                    alpha=0.5,
                    color=colors[i],
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
    ldata = None
    ldata2 = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("el_dens")
            ldata2 = ldata + get_labels(label2)
        data1.el_dens.sel(t=t).plot(color=colors[i], label=ldata)

        if has_data2:
            data2.el_dens.sel(t=t).plot(color=colors[i], linestyle="--", label=ldata2)

    plt.title("Electron Density")
    plt.ylabel("($m^{-3}$)")
    plt.xlabel(r"$\rho_{pol}$")
    if savefig:
        save_figure(fig_name=name + "_electron_density")

    # Main ion density
    plt.figure()
    ldata = None
    ldata2 = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("ion_dens")
            ldata2 = ldata + get_labels(label2)
        data1.ion_dens.sel(t=t).sel(element=data1.main_ion).plot(
            color=colors[i], label=ldata
        )

        if has_data2:
            data2.ion_dens.sel(t=t).sel(element=data1.main_ion).plot(
                color=colors[i], linestyle="--", label=ldata2
            )

    plt.title("Main ion Density")
    plt.ylabel("($m^{-3}$)")
    plt.xlabel(r"$\rho_{pol}$")
    plt.legend()
    if savefig:
        save_figure(fig_name=name + "_ion_density")

    # Effective charge
    # --------------------------------------------
    plt.figure()
    ldata = None
    ldata2 = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("zeff")
            ldata2 = ldata + get_labels(label2)
        data1.zeff.sel(t=t).sum("element").plot(color=colors[i], label=ldata)
        if has_data2:
            data2.zeff.sel(t=t).sum("element").plot(
                color=colors[i], label=ldata2, linestyle="--"
            )
        plt.ylim(0, )
    plt.title("Effective charge")
    plt.ylabel("$Z_{eff}$")
    plt.xlabel(r"$\rho_{pol}$")
    plt.legend()
    if savefig:
        save_figure(fig_name=name + "_effective_charge")

    # Total radiated power
    # --------------------------------------------
    plt.figure()
    ldata = None
    ldata2 = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("tot_rad")
            ldata2 = ldata + get_labels(label2)
        (data1.tot_rad.sel(t=t).sum("element") / 1.0e3).plot(
            color=colors[i], label=ldata
        )
        if has_data2:
            (data2.tot_rad.sel(t=t).sum("element") / 1.0e3).plot(
                color=colors[i], label=ldata2, linestyle="--"
            )
    plt.title("Total radiated power")
    plt.ylabel("(kW)")
    plt.xlabel(r"$\rho_{pol}$")
    plt.legend()
    if savefig:
        save_figure(fig_name=name + "_total_radiated_power")

    # Total pressure profile
    # --------------------------------------------
    plt.figure()
    ldata = None
    ldata2 = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("pressure")
            ldata2 = ldata + get_labels(label2)
        (data1.pressure_th / 1.0e3).sel(t=t).plot(color=colors[i], label=ldata)
        if has_data2:
            (data2.pressure_th / 1.0e3).sel(t=t).plot(
                color=colors[i], label=ldata2, linestyle="--"
            )
    plt.title("Thermal pressure (ion + electrons)")
    plt.ylabel("($kPa$ $m^{-3}$)")
    plt.xlabel(r"$\rho_{pol}$")
    plt.legend()
    if savefig:
        save_figure(fig_name=name + "_thermal_pressure")

    # Current density profile
    # --------------------------------------------
    plt.figure()
    ldata = None
    ldata2 = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("j_phi")
            ldata2 = ldata + get_labels(label2)
        (data1.j_phi / 1.0e3).sel(t=t).plot(color=colors[i], label=ldata)
        if has_data2:
            (data2.j_phi / 1.0e3).sel(t=t).plot(
                color=colors[i], label=ldata2, linestyle="--"
            )
    plt.title("Current density")
    plt.ylabel("($kA$ $m^{-2}$)")
    plt.xlabel(r"$\rho_{pol}$")
    plt.legend()
    if savefig:
        save_figure(fig_name=name + "_thermal_pressure")

    # Plasma energy
    # --------------------------------------------
    plt.figure()
    wmhd = data1.wmhd
    x_val = getattr(data1, "time")
    if correl != "t":
        x_val = getattr(data1, correl)
    if correl != "t":
        wmhd.assign_coords(correl=("t", x_val))
        wmhd = wmhd.swap_dims({"t": "correl"})

    ldata = get_labels("wmhd")
    ldata2 = ldata + get_labels(label2)
    (wmhd / 1.0e3).plot(label=ldata)
    if has_data2:
        wmhd_data2 = data2.wmhd
        if correl != "t":
            wmhd_data2 = wmhd_data2.assign_coords(correl=("t", x_val))
            wmhd_data2 = wmhd_data2.swap_dims({"t": "correl"})
        (wmhd_data2 / 1.0e3).plot(color="k", linestyle="dashed", label=ldata2)

    if hasattr(data1, "raw_data") and correl == "t":
        (data1.raw_data["efit"]["wp"] / 1.0e3).plot()

    for i in range(len(x_val)):
        plt.scatter(
            x_val[i].values, wmhd[i].values / 1.0e3, marker="o", color=colors[i]
        )
        if has_data2:
            plt.scatter(
                x_val[i].values,
                wmhd_data2[i].values / 1.0e3,
                facecolors="none",
                color=colors[i],
            )

    plt.title("Plasma thermal energy")
    plt.ylabel("($kJ$)")
    plt.xlabel(correl_label(correl))
    plt.legend()
    if savefig:
        save_figure(fig_name=name + "_thermal_energy")

    # Vloop
    # --------------------------------------------
    plt.figure()
    vloop = data1.vloop
    if correl != "t":
        vloop = vloop.assign_coords(correl=("t", x_val))
        vloop = vloop.swap_dims({"t": "correl"})

    ldata = get_labels("vloop")
    ldata2 = ldata + get_labels(label2)
    vloop.plot(label=ldata)
    if has_data2:
        vloop_data2 = data2.vloop
        if correl != "t":
            vloop_data2 = vloop_data2.assign_coords(correl=("t", x_val))
            vloop_data2 = vloop_data2.swap_dims({"t": "correl"})
        vloop_data2.plot(color="k", linestyle="dashed", label=ldata2)

    if hasattr(data1, "raw_data") and correl == "t":
        (data1.raw_data["vloop"]).plot()

    for i in range(len(x_val)):
        plt.scatter(x_val[i].values, vloop[i].values, marker="o", color=colors[i])
        if has_data2:
            plt.scatter(
                x_val[i].values,
                vloop_data2[i].values,
                facecolors="none",
                color=colors[i],
            )
    plt.title("$V_{loop}$")
    plt.ylabel("($V$)")
    plt.xlabel(correl_label(correl))
    plt.legend()
    if savefig:
        save_figure(fig_name=name)

    # LOS-integrated electron density
    # --------------------------------------------
    plt.figure()
    interferom = ["nirh1", "smmh1"]
    marker = ["o", "s"]
    ylim = [0, 0]
    for isys, system in enumerate(interferom):
        if not hasattr(data1, system):
            continue
        interf_data1 = getattr(data1, system)
        ylim[1] = np.max([ylim[1], interf_data1.max()])
        if correl != "t":
            interf_data1.assign_coords(correl=("t", x_val))
            interf_data1 = interf_data1.swap_dims({"t": "correl"})

        ldata = get_labels(system)
        ldata2 = ldata + get_labels(label2)
        interf_data1.plot(label=ldata)
        if has_data2:
            interf_data2 = getattr(data2, system)
            ylim[1] = np.max([ylim[1], interf_data2.max()])
            if correl != "t":
                interf_data2.assign_coords(correl=("t", x_val))
                interf_data2 = interf_data2.swap_dims({"t": "correl"})
            interf_data2.plot(color="k", linestyle="dashed", label=ldata2)

        if hasattr(data1, "raw_data") and correl == "t":
            (data1.raw_data[system]["ne"]).plot()
            ylim[1] = np.max([ylim[1], data1.raw_data[system]["ne"].max()])

        for i in range(len(x_val)):
            plt.scatter(
                x_val[i].values, interf_data1[i].values, color=colors[i], marker=marker[isys]
            )
            if has_data2:
                plt.scatter(
                    x_val[i].values,
                    interf_data2[i].values,
                    color=colors[i],
                    marker=marker[isys],
                    facecolors="none",
                )
    ylim[1] *= 1.05
    plt.title("$N_{e}$ LOS-integrated")
    plt.ylabel("($m^{-2}$)")
    plt.xlabel(correl_label(correl))
    plt.legend()
    plt.ylim(ylim)
    if savefig:
        save_figure(fig_name=name + "_interf")

    # Spectrometer emission characteristics and
    # element ionization balance
    # --------------------------------------------
    if plot_spectr is not True:
        return

    for k, results in data1.spectrometers.items():
        titles = {
            "fz": f"{results.element}{results.charge}+ fractional abundance",
            "pec": f"{results.element}{results.charge}+ " f"{results.wavelength}A PEC",
            "emiss": f"{results.element}{results.charge}+ "
            f"{results.wavelength}A emission shell",
        }

        plt.figure()
        for tmp in results.fz:
            plt.plot(tmp.coords["rho_poloidal"], tmp.transpose(), alpha=0.2, color="k")
        for i, tmp in enumerate(results.fz):
            tmp.sel(ion_charges=results.charge).plot(linestyle="--", color=colors[i])

        plt.title(titles["fz"])
        plt.ylabel("")
        plt.xlabel(r"$\rho_{pol}$")
        if savefig:
            save_figure(fig_name=name + f"_{results.element}_fract_abu")

        plt.figure()
        for i, tmp in enumerate(results.emiss):
            (tmp / tmp.max()).plot(color=colors[i])
        plt.title(titles["emiss"])
        plt.ylabel("")
        plt.xlabel(r"$\rho_{pol}$")
        if savefig:
            save_figure(
                fig_name=name + f"_{results.element}{results.charge}_emission_ragion"
            )

    # Spectrometer integration region
    # --------------------------------------------
    plt.figure()
    if "he_like" in data1.spectrometers.keys():
        plt.plot(
            data1.te_0, np.array(data1.spectrometers["he_like"].pos), "k", alpha=0.5,
        )
        plt.fill_between(
            data1.te_0,
            np.array(data1.spectrometers["he_like"].pos)
            - np.array(data1.spectrometers["he_like"].pos.attrs["err_in"]),
            np.array(data1.spectrometers["he_like"].pos)
            + np.array(data1.spectrometers["he_like"].pos.attrs["err_out"]),
            label="He-like",
            alpha=0.5,
        )
    if "passive_c5" in data1.spectrometers.keys():
        plt.plot(
            data1.te_0, np.array(data1.spectrometers["passive_c5"].pos), "k", alpha=0.5,
        )
        plt.fill_between(
            data1.te_0,
            np.array(data1.spectrometers["passive_c5"].pos)
            - np.array(data1.spectrometers["passive_c5"].pos.attrs["err_in"]),
            np.array(data1.spectrometers["passive_c5"].pos)
            + np.array(data1.spectrometers["passive_c5"].pos.attrs["err_out"]),
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
    plt.plot(data1.te_0, data1.te_0, label="$T_e(0)$")
    if "he_like" in data1.spectrometers.keys():
        plt.plot(
            data1.te_0,
            data1.spectrometers["he_like"].el_temp,
            color="k",
            linestyle="dashed",
        )
        plt.fill_between(
            data1.te_0,
            np.array(data1.spectrometers["he_like"].el_temp)
            - np.array(data1.spectrometers["he_like"].el_temp.attrs["err_in"]),
            np.array(data1.spectrometers["he_like"].el_temp)
            + np.array(data1.spectrometers["he_like"].el_temp.attrs["err_out"]),
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
        "/home/marco.sertoli/python/figures/Indica/" + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def get_labels(lkey=None):

    labels = {
        "el_dens": "$n_e$",
        "ion_dens": "$n_i$",
        "el_temp": "$T_e$",
        "ion_temp": "$T_i$",
        "pressure": "$P_{tot}$",
        "j_phi": "$j_{\phi}$",
        "wdia": "3/2$ \int P_{th} dV$",
        "wmhd": "3/2$ \int P_{tot} dV$",
        "zeff": "$Z_{eff}$",
        "interf": "$N_e$ LOS-int",
        "nirh1": "$N_e$ LOS-int (NIRH1)",
        "smmh1": "$N_e$ LOS-int (SMMH1)",
        "tot_rad": "$P_{rad}$",
        "vloop": "$V_{loop}$",
        "bckc": " (back-calculated)",
        "he_like": " (He-like)",
        "passive_c5": " (Passive C5+)",
        "real_value": "Real value",
        "bckc_value": "Back-calculated",
    }

    if lkey is not None:
        if lkey in labels.keys():
            labels = labels[lkey]

    return labels


def correl_label(key):
    correl_label = {
        "te_0": "Te(0)",
        "nirh1": "$N_e$ LOS-int NIRH1",
        "smmh1": "$N_e$ LOS-int SMMH1",
        "t": "Time (s)",
    }
    if key in correl_label:
        label = correl_label[key]
    else:
        label = ""

    return label
