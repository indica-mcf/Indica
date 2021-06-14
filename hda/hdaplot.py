import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm

plt.ion()

def HDAplot(data, bckc=None, correl="t", plot_spectr=False, name="", savefig=False):
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
    data.ne_0.values = data.el_dens.sel(rho_poloidal=0)
    has_bckc = False
    if bckc is not None:
        bckc.ne_0.values = bckc.el_dens.sel(rho_poloidal=0)
        has_bckc = True

    time = data.time
    nt = len(time)
    colors = cm.rainbow(np.linspace(0, 1, nt))

    # Electron temperature
    plt.figure()
    ldata = None
    ldata_he = None
    lbckc = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("el_temp")
            ldata_he = ldata + get_labels("he_like")
            lbckc = ldata + get_labels("bckc")
        data.el_temp.sel(t=t).plot(color=colors[i], label=ldata)

        plt.scatter(
            data.spectrometers["he_like"].pos[i],
            data.spectrometers["he_like"].el_temp[i],
            color=colors[i],
            label=ldata_he,
        )
        plt.hlines(
            data.spectrometers["he_like"].el_temp[i],
            data.spectrometers["he_like"].pos[i]
            - data.spectrometers["he_like"].pos.attrs["err_in"][i],
            data.spectrometers["he_like"].pos[i]
            + data.spectrometers["he_like"].pos.attrs["err_out"][i],
            alpha=0.5,
            color=colors[i],
        )

        if has_bckc:
            bckc.el_temp.sel(t=t).plot(color=colors[i], label=lbckc, linestyle="--")

    plt.title("Electron Temperature")
    plt.ylabel("(eV)")
    plt.xlabel(r"$\rho_{pol}$")
    plt.legend()
    ylim = (0, np.max([data.el_temp.max(), data.ion_temp.max()]))
    if has_bckc:
        ylim = (0, np.max([ylim[1], bckc.el_temp.max(), bckc.ion_temp.max()]))
    plt.ylim(ylim)
    if savefig:
        save_figure(fig_name=name + "_electron_temperature")

    # Ion temperature
    plt.figure()
    ldata = None
    lbckc = None
    ldata_he = None
    ldata_c5 = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("ion_temp")
            ldata_he = ldata + get_labels("he_like")
            ldata_c5 = ldata + get_labels("passive_c5")
            lbckc = ldata + get_labels("bckc")
        data.ion_temp.sel(t=t).sel(element=data.main_ion).plot(
            color=colors[i], label=ldata,
        )

        if "he_like" in data.spectrometers.keys():
            plt.scatter(
                data.spectrometers["he_like"].pos[i],
                data.spectrometers["he_like"].ion_temp[i],
                color=colors[i],
                label=ldata_he,
            )
            plt.hlines(
                data.spectrometers["he_like"].ion_temp[i],
                data.spectrometers["he_like"].pos[i]
                - data.spectrometers["he_like"].pos.attrs["err_in"][i],
                data.spectrometers["he_like"].pos[i]
                + data.spectrometers["he_like"].pos.attrs["err_out"][i],
                alpha=0.5,
                color=colors[i],
            )

        if "passive_c5" in data.spectrometers.keys():
            plt.scatter(
                data.spectrometers["passive_c5"].pos[i],
                data.spectrometers["passive_c5"].ion_temp[i],
                color=colors[i],
                marker="x",
                label=ldata_c5,
            )
            plt.hlines(
                data.spectrometers["passive_c5"].ion_temp[i],
                data.spectrometers["passive_c5"].pos[i]
                - data.spectrometers["passive_c5"].pos.attrs["err_in"][i],
                data.spectrometers["passive_c5"].pos[i]
                + data.spectrometers["passive_c5"].pos.attrs["err_out"][i],
                alpha=0.5,
                color=colors[i],
            )

        if has_bckc:
            bckc.ion_temp.sel(t=t).sel(element=data.main_ion).plot(
                color=colors[i], label=lbckc, linestyle="--"
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
    lbckc = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("el_dens")
            lbckc = ldata + get_labels("bckc")
        data.el_dens.sel(t=t).plot(color=colors[i], label=ldata)

        if has_bckc:
            bckc.el_dens.sel(t=t).plot(color=colors[i], linestyle="--", label=lbckc)

    plt.title("Electron Density")
    plt.ylabel("($m^{-3}$)")
    plt.xlabel(r"$\rho_{pol}$")
    if savefig:
        save_figure(fig_name=name + "_electron_density")

    # Main ion density
    plt.figure()
    ldata = None
    lbckc = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("ion_dens")
            lbckc = ldata + get_labels("bckc")
        data.ion_dens.sel(t=t).sel(element=data.main_ion).plot(
            color=colors[i], label=ldata
        )

        if has_bckc:
            bckc.ion_dens.sel(t=t).sel(element=data.main_ion).plot(
                color=colors[i], linestyle="--", label=lbckc
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
    lbckc = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("zeff")
            lbckc = ldata + get_labels("bckc")
        data.zeff.sel(t=t).sum("element").plot(color=colors[i], label=ldata)
        if has_bckc:
            bckc.zeff.sel(t=t).sum("element").plot(
                color=colors[i], label=lbckc, linestyle="--"
            )
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
    lbckc = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("tot_rad")
            lbckc = ldata + get_labels("bckc")
        (data.tot_rad.sel(t=t).sum("element") / 1.0e3).plot(
            color=colors[i], label=ldata
        )
        if has_bckc:
            (bckc.tot_rad.sel(t=t).sum("element") / 1.0e3).plot(
                color=colors[i], label=lbckc, linestyle="--"
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
    lbckc = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("pressure")
            lbckc = ldata + get_labels("bckc")
        (data.pressure_th / 1.0e3).sel(t=t).plot(color=colors[i], label=ldata)
        if has_bckc:
            (bckc.pressure_th / 1.0e3).sel(t=t).plot(
                color=colors[i], label=lbckc, linestyle="--"
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
    lbckc = None
    for i, t in enumerate(time):
        if i == (nt - 1):
            ldata = get_labels("j_phi")
            lbckc = ldata + get_labels("bckc")
        (data.j_phi / 1.0e3).sel(t=t).plot(color=colors[i], label=ldata)
        if has_bckc:
            (bckc.j_phi / 1.0e3).sel(t=t).plot(
                color=colors[i], label=lbckc, linestyle="--"
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
    wmhd = data.wmhd
    x_val = getattr(data, "time")
    if correl != "t":
        x_val = getattr(data, correl)
    if correl != "t":
        wmhd.assign_coords(correl=("t", x_val))
        wmhd = wmhd.swap_dims({"t": "correl"})

    ldata = get_labels("wmhd")
    lbckc = ldata + get_labels("bckc")
    (wmhd / 1.0e3).plot(label=ldata)
    if has_bckc:
        wmhd_bckc = bckc.wmhd
        if correl != "t":
            wmhd_bckc = wmhd_bckc.assign_coords(correl=("t", x_val))
            wmhd_bckc = wmhd_bckc.swap_dims({"t": "correl"})
        (wmhd_bckc / 1.0e3).plot(color="k", linestyle="dashed", label=lbckc)

    if hasattr(data, "raw_data") and correl == "t":
        (data.raw_data["efit"]["wp"] / 1.0e3).plot()

    for i in range(len(x_val)):
        plt.scatter(
            x_val[i].values, wmhd[i].values / 1.0e3, marker="o", color=colors[i]
        )
        if has_bckc:
            plt.scatter(
                x_val[i].values,
                wmhd_bckc[i].values / 1.0e3,
                marker="x",
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
    vloop = data.vloop
    if correl != "t":
        vloop = vloop.assign_coords(correl=("t", x_val))
        vloop = vloop.swap_dims({"t": "correl"})

    ldata = get_labels("vloop")
    lbckc = ldata + get_labels("bckc")
    vloop.plot(label=ldata)
    if has_bckc:
        vloop_bckc = bckc.vloop
        if correl != "t":
            vloop_bckc = vloop_bckc.assign_coords(correl=("t", x_val))
            vloop_bckc = vloop_bckc.swap_dims({"t": "correl"})
        vloop_bckc.plot(color="k", linestyle="dashed", label=lbckc)

    if hasattr(data, "raw_data") and correl == "t":
        (data.raw_data["vloop"]).plot()

    for i in range(len(x_val)):
        plt.scatter(x_val[i].values, vloop[i].values, marker="o", color=colors[i])
        if has_bckc:
            plt.scatter(
                x_val[i].values, vloop_bckc[i].values, marker="x", color=colors[i]
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
    ne_l = data.ne_l
    if correl != "t":
        ne_l.assign_coords(correl=("t", x_val))
        ne_l = ne_l.swap_dims({"t": "correl"})

    ldata = get_labels("ne_l")
    lbckc = ldata + get_labels("bckc")
    ne_l.plot(label=ldata)
    if has_bckc:
        ne_l_bckc = bckc.ne_l
        if correl != "t":
            ne_l_bckc.assign_coords(correl=("t", x_val))
            ne_l_bckc = ne_l_bckc.swap_dims({"t": "correl"})
        ne_l_bckc.plot(color="k", linestyle="dashed", label=lbckc)

    if hasattr(data, "raw_data") and correl == "t":

        if ne_l.name.split("_")[0] == "smmh1":
            (data.raw_data["smmh1"]["ne"]).plot()
        if ne_l.name.split("_")[0] == "nirh1":
            (data.raw_data["nirh1"]["ne"]).plot()

    for i in range(len(x_val)):
        plt.scatter(x_val[i].values, ne_l[i].values, marker="o", color=colors[i])
        if has_bckc:
            plt.scatter(
                x_val[i].values, ne_l_bckc[i].values, marker="x", color=colors[i]
            )
    plt.title("$N_{e}$ LOS-integrated")
    plt.ylabel("($m^{-2}$)")
    plt.xlabel(correl_label(correl))
    plt.legend()
    if savefig:
        save_figure(fig_name=name + "_ne_l")

    # Spectrometer emission characteristics and
    # element ionization balance
    # --------------------------------------------
    if plot_spectr is not True:
        return

    for k, results in data.spectrometers.items():
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
    if "he_like" in data.spectrometers.keys():
        plt.plot(
            data.te_0, np.array(data.spectrometers["he_like"].pos), "k", alpha=0.5,
        )
        plt.fill_between(
            data.te_0,
            np.array(data.spectrometers["he_like"].pos)
            - np.array(data.spectrometers["he_like"].pos.attrs["err_in"]),
            np.array(data.spectrometers["he_like"].pos)
            + np.array(data.spectrometers["he_like"].pos.attrs["err_out"]),
            label="He-like",
            alpha=0.5,
        )
    if "passive_c5" in data.spectrometers.keys():
        plt.plot(
            data.te_0, np.array(data.spectrometers["passive_c5"].pos), "k", alpha=0.5,
        )
        plt.fill_between(
            data.te_0,
            np.array(data.spectrometers["passive_c5"].pos)
            - np.array(data.spectrometers["passive_c5"].pos.attrs["err_in"]),
            np.array(data.spectrometers["passive_c5"].pos)
            + np.array(data.spectrometers["passive_c5"].pos.attrs["err_out"]),
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
    plt.plot(data.te_0, data.te_0, label="$T_e(0)$")
    if "he_like" in data.spectrometers.keys():
        plt.plot(
            data.te_0,
            data.spectrometers["he_like"].el_temp,
            color="k",
            linestyle="dashed",
        )
        plt.fill_between(
            data.te_0,
            np.array(data.spectrometers["he_like"].el_temp)
            - np.array(data.spectrometers["he_like"].el_temp.attrs["err_in"]),
            np.array(data.spectrometers["he_like"].el_temp)
            + np.array(data.spectrometers["he_like"].el_temp.attrs["err_out"]),
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
        "ne_l": "$N_e LOS-int$",
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
        "te_0": "Te(0) (eV)",
        "ne_l": "Ne LOS-int. midplane ($m^{-2}$)",
        "t": "Time (s)",
    }
    if key in correl_label:
        label = correl_label[key]
    else:
        label = ""

    return label
