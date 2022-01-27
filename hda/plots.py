import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm, rcParams
import xarray as xr

plt.ion()

rcParams.update({"font.size": 12})


def compare_data_bckc(
    data, bckc, raw_data={}, pulse=None, xlim=None, savefig=False, name="", title="", ploterr=True,
):
    colors = ("black", "blue", "purple", "orange", "red")
    _title = ""
    if pulse is not None:
        _title = f"{pulse}"
    if len(title) > 1:
        _title += f" {title}"

    figname = get_figname(pulse=pulse, name=name)
    if "xrcs" in bckc.keys():
        # Temperatures
        plt.figure()
        ylim0, ylim1 = [], []
        for i, quant in enumerate(bckc["xrcs"].keys()):
            if ("ti" not in quant) and ("te" not in quant):
                continue
            marker = "o"
            if "ti" in quant:
                marker = "x"

            if "xrcs" in raw_data.keys():
                raw_data["xrcs"][quant].plot(
                    color=colors[i], linestyle="dashed", alpha=0.5,
                )
            plt.fill_between(
                data["xrcs"][quant].t,
                data["xrcs"][quant].values + data["xrcs"][quant].attrs["error"],
                data["xrcs"][quant].values - data["xrcs"][quant].attrs["error"],
                color=colors[i],
                alpha=0.5,
            )
            data["xrcs"][quant].plot(
                marker=marker,
                color=colors[i],
                linestyle="dashed",
                label=f"{quant.upper()} XRCS",
            )
            ylim0.append(np.nanmin(data["xrcs"][quant]))
            ylim1.append(np.nanmax(data["xrcs"][quant]) * 1.3)
        for i, quant in enumerate(bckc["xrcs"].keys()):
            if ("ti" not in quant) and ("te" not in quant):
                continue
            bckc["xrcs"][quant].plot(color=colors[i], label="Back-calc", linewidth=3)

        plt.xlim(xlim)
        plt.ylim(0, np.max(ylim1))
        plt.title(f"{_title} Electron and ion temperature")
        plt.xlabel("Time (s)")
        plt.ylabel("(eV)")
        plt.legend()
        if savefig:
            save_figure(fig_name=f"{figname}data_XRCS_electron_temperature")

        # Intensity
        plt.figure()
        i = -1
        ylim0, ylim1 = [], []
        for quant in bckc["xrcs"].keys():
            if "int" not in quant:
                continue
            i += 1
            marker = "o"

            plt.figure()
            if "xrcs" in raw_data.keys():
                raw_data["xrcs"][quant].plot(
                    color=colors[i], linestyle="dashed", alpha=0.5,
                )
            plt.fill_between(
                data["xrcs"][quant].t,
                data["xrcs"][quant].values + data["xrcs"][quant].attrs["error"],
                data["xrcs"][quant].values - data["xrcs"][quant].attrs["error"],
                color=colors[i],
                alpha=0.5,
            )
            data["xrcs"][quant].plot(
                marker=marker,
                color=colors[i],
                linestyle="dashed",
                label=f"{quant.upper()} XRCS",
            )
            ylim0.append(np.nanmin(data["xrcs"][quant]))
            ylim1.append(np.nanmax(data["xrcs"][quant]) * 1.3)

            bckc["xrcs"][quant].plot(color=colors[i], label="Back-calc", linewidth=3)

            plt.xlim(xlim)
            plt.ylim(0, np.max(ylim1))
            plt.title(f"{_title} Spectral line intensities ({quant})")
            plt.xlabel("Time (s)")
            plt.ylabel("(a.u.)")
            plt.legend()
            if savefig:
                quant_str = quant[:]
                if "/" in quant:
                    quant_str = quant_str.replace("/", "_ov_")
                save_figure(fig_name=f"{figname}data_XRCS_line_intensities_{quant_str}")

    if "nirh1" in bckc.keys() or "nirh1_bin" in bckc.keys() or "smmh1" in bckc.keys():
        plt.figure()
        ylim0, ylim1 = [], []
        for i, diag in enumerate(("nirh1", "nirh1_bin", "smmh1")):
            if diag in bckc.keys():
                if diag in raw_data.keys():
                    raw_data[diag]["ne"].plot(
                        color=colors[i], linestyle="dashed",
                    )
                plt.fill_between(
                    data[diag]["ne"].t,
                    data[diag]["ne"].values + data[diag]["ne"].attrs["error"],
                    data[diag]["ne"].values - data[diag]["ne"].attrs["error"],
                    color=colors[i],
                    alpha=0.5,
                )
                data[diag]["ne"].plot(
                    marker="o",
                    color=colors[i],
                    linestyle="dashed",
                    label=f"Ne {diag.upper()}",
                )
                ylim0.append(np.nanmin(data[diag]["ne"]))
                ylim1.append(np.nanmax(data[diag]["ne"]) * 1.3)

        for i, diag in enumerate(("nirh1", "nirh1_bin", "smmh1")):
            if diag in bckc.keys():
                bckc[diag]["ne"].plot(color=colors[i], label="Back-calc", linewidth=3)

        plt.xlim(xlim)
        plt.ylim(0, np.max(ylim1))
        plt.title(f"{_title} Electron density")
        plt.xlabel("Time (s)")
        plt.ylabel("(m$^{-2}$)")
        plt.legend()
        if savefig:
            save_figure(fig_name=f"{figname}data_electron_density")

    diag = "efit"
    if diag in data.keys():
        plt.figure()
        ylim0, ylim1 = [], []
        i = 0
        if diag in raw_data.keys():
            raw_data[diag]["wp"].plot(
                color=colors[i], linestyle="dashed", alpha=0.5,
            )
        data[diag]["wp"].plot(
            marker="o", color=colors[i], linestyle="dashed", label=f"Wp {diag.upper()}",
        )
        ylim0.append(np.nanmin(data[diag]["wp"]))
        ylim1.append(np.nanmax(data[diag]["wp"]) * 1.3)

        if diag in bckc.keys():
            bckc[diag]["wp"].plot(color=colors[i], label="Back-calc", linewidth=3)
        plt.xlim(xlim)
        plt.title(f"{_title} Stored Energy")
        plt.xlabel("Time (s)")
        plt.ylabel("(J)")
        plt.ylim(0, np.max(ylim1))
        plt.legend()
        if savefig:
            save_figure(fig_name=f"{figname}data_EFIT_stored_energy")

    diag = "lines"
    quant = "brems"
    if diag in data.keys():
        plt.figure()
        i = 0
        ylim0, ylim1 = [], []
        if diag in raw_data.keys():
            raw_data[diag][quant].plot(
                color=colors[i], linestyle="dashed", alpha=0.5,
            )
        data[diag][quant].plot(
            marker="o",
            color=colors[i],
            linestyle="dashed",
            label=f"Brems {diag.upper()}",
        )
        ylim0.append(np.nanmin(data[diag][quant]) * 0.7)
        ylim1.append(np.nanmax(data[diag][quant]) * 1.3)

        if diag in bckc.keys():
            bckc[diag][quant].plot(color=colors[i], label="Back-calc", linewidth=3)
        plt.xlim(xlim)
        plt.ylim(np.min(ylim0), np.max(ylim1))
        plt.title(f"{_title} Bremsstrahlung Intensity")
        plt.xlabel("Time (s)")
        plt.ylabel("(a.u.)")
        plt.legend()
        if savefig:
            save_figure(fig_name=f"{figname}data_LINES_bremsstrahlung")


def time_evol(plasma, data, bckc={}, savefig=False, name="", title="", ploterr=True):
    figname = get_figname(pulse=plasma.pulse, name=name)
    _title = f"{plasma.pulse}"
    if len(title) > 1:
        _title += f" {title}"

    elem_str = {}
    for j, elem in enumerate(plasma.elements):
        _str = elem.upper()
        if len(elem) > 1:
            _str = elem[0].upper() + elem[1]
        elem_str[elem] = _str

    colors = ("black", "blue", "orange", "red")
    linestyles = ("dotted", (0, (5, 1)), (0, (5, 5)), (0, (5, 10)))

    plt.figure()
    ylim = (
        0,
        np.max(
            [
                plasma.el_temp.max() * 1.05,
                plasma.ion_temp.sel(element=plasma.main_ion).max() * 1.05,
            ]
        ),
    )
    if hasattr(plasma, "el_temp_hi") and ploterr:
        plt.fill_between(
            plasma.time,
            plasma.el_temp_hi.sel(rho_poloidal=0),
            plasma.el_temp_lo.sel(rho_poloidal=0),
            color="blue",
            alpha=0.5,
        )
        plt.fill_between(
            plasma.time,
            plasma.ion_temp_hi.sel(element=plasma.main_ion, rho_poloidal=0),
            plasma.ion_temp_lo.sel(element=plasma.main_ion, rho_poloidal=0),
            color="red",
            alpha=0.5,
        )
        ylim = (
            0,
            np.max(
                [
                    plasma.el_temp_hi.max() * 1.05,
                    plasma.ion_temp_hi.sel(element=plasma.main_ion).max() * 1.05,
                    ylim[1],
                ]
            ),
        )
    plasma.el_temp.sel(rho_poloidal=0).plot(label="Te(0)", color="blue", alpha=0.8)
    plasma.ion_temp.sel(element=plasma.main_ion, rho_poloidal=0).plot(
        color="red", label="Ti(0)", alpha=0.8
    )
    plt.title(f"{_title} Central temperatures")
    plt.xlabel("Time (s)")
    plt.ylabel("(eV)")
    plt.ylim(ylim)
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_central_temperatures")

    plt.figure()
    ylim = (0, plasma.el_dens.max() * 1.05)
    if hasattr(plasma, "el_dens_hi") and ploterr:
        plt.fill_between(
            plasma.time,
            plasma.el_dens_hi.sel(rho_poloidal=0),
            plasma.el_dens_lo.sel(rho_poloidal=0),
            color="blue",
            alpha=0.5,
        )
        plt.fill_between(
            plasma.time,
            plasma.ion_dens_hi.sel(element=plasma.main_ion, rho_poloidal=0),
            plasma.ion_dens_lo.sel(element=plasma.main_ion, rho_poloidal=0),
            color="red",
            alpha=0.5,
        )
        ylim = (
            0,
            np.max(
                [
                    plasma.el_dens_hi.max() * 1.05,
                    plasma.ion_dens_hi.sel(element=plasma.main_ion).max() * 1.05,
                    ylim[1],
                ]
            ),
        )
    plasma.el_dens.sel(rho_poloidal=0).plot(label="Ne(0)", color="blue", alpha=0.8)
    plasma.ion_dens.sel(element=plasma.main_ion, rho_poloidal=0).plot(
        color="red", label="Ni(0)", alpha=0.8
    )
    plt.title(f"{_title} Central densities")
    plt.xlabel("Time (s)")
    plt.ylabel("(m$^{-3}$)")
    plt.ylim(ylim)
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_central_densities")

    plt.figure()
    prad_los_int = xr.zeros_like(plasma.prad)
    # TODO: using XRCS LOS. Comparison with experimental values when diagnostics available
    for j, elem in enumerate(plasma.elements):
        los_int = plasma.calc_los_int(
            data["xrcs"]["ti_w"], plasma.tot_rad.sel(element=elem)
        )
        prad_los_int.loc[dict(element=elem)] = los_int
    prad_los_int.sum("element").plot(label="Total", color="black")
    prad_los_int.sel(element=plasma.main_ion).plot(
        color="black", label=elem_str[plasma.main_ion], linestyle="dotted"
    )
    for j, elem in enumerate(plasma.elements):
        prad_los_int.sel(element=elem).plot(
            color=colors[j], label=elem_str[elem], linestyle="dashed"
        )
    plt.title(f"{_title} Total radiated power")
    plt.xlabel("Time (s)")
    plt.ylabel("")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_total_rad_power")

    plt.figure()
    # TODO: using XRCS LOS, add SXR diode experimental value and LOS info
    prad_los_int_sxr = xr.zeros_like(plasma.prad)
    Te_lim = 800
    for elem in plasma.elements:
        rad_tmp = xr.where(
            plasma.el_temp > Te_lim, plasma.tot_rad.sel(element=elem), 0.0
        )
        los_int = plasma.calc_los_int(data["xrcs"]["ti_w"], rad_tmp)
        prad_los_int_sxr.loc[dict(element=elem)] = los_int
    prad_los_int_sxr.sum("element").plot(label="Total", color="black")
    for j, elem in enumerate(plasma.elements):
        prad_los_int_sxr.sel(element=elem).plot(
            color=colors[j], label=elem_str[elem], linestyle=linestyles[j]
        )
    plt.title(f"{_title} SXR radiated power (Te>{Te_lim}) eV)")
    plt.xlabel("Time (s)")
    plt.ylabel("")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_sxr_rad_power")

    plt.figure()
    zeff = plasma.zeff.mean("rho_poloidal")
    zeff_main_ion = zeff.sel(element=plasma.main_ion)
    zeff.sum("element").plot(label="Total", color="black")
    zeff_main_ion.plot(color=colors[0], label=elem_str[elem], linestyle=linestyles[0])
    for j, elem in enumerate(plasma.elements):
        if elem != plasma.main_ion:
            (zeff.sel(element=elem) + zeff_main_ion).plot(
                color=colors[j], label=elem_str[elem], linestyle=linestyles[j]
            )
    plt.title(f"{_title} Average effective charge")
    plt.xlabel("Time (s)")
    plt.ylabel("")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_effective_charge")

    plt.figure()
    c_ion = (plasma.ion_dens / plasma.el_dens).sel(rho_poloidal=0)
    c_ion.sel(element=elem).plot(
        color=colors[0], label=elem_str[elem], linestyle=linestyles[0]
    )
    for j, elem in enumerate(plasma.elements):
        if elem != plasma.main_ion:
            c_ion.sel(element=elem).plot(
                color=colors[j], label=elem_str[elem], linestyle=linestyles[j]
            )
    plt.title(f"{_title} Central ion concentration")
    plt.xlabel("Time (s)")
    plt.ylabel("")
    plt.yscale("log")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_central_ion_conc")

    plt.figure()
    plasma.q_prof.sel(rho_poloidal=0).plot(color="black")
    plt.title(f"{_title} Safety factor on axis")
    plt.xlabel("Time (s)")
    plt.ylabel("")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_axis_safety_factor")

    plt.figure()
    plasma.vloop.plot(color="black")
    plt.title(f"{_title} Loop voltage")
    plt.xlabel("Time (s)")
    plt.ylabel("")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_loop_voltage")


def profiles(plasma, bckc={}, savefig=False, name="", alpha=0.8, title="", ploterr=True):
    figname = get_figname(pulse=plasma.pulse, name=name)
    _title = f"{plasma.pulse}"
    if len(title)>1:
        _title += f" {title}"

    elem_str = {}
    for j, elem in enumerate(plasma.elements):
        _str = elem.upper()
        if len(elem) > 1:
            _str = elem[0].upper() + elem[1]
        elem_str[elem] = _str

    time = plasma.t
    cmap = cm.rainbow
    varr = np.linspace(0, 1, len(time))
    colors = cmap(varr)

    # Impurity linestyles
    linestyle_imp = ((0, (5, 1)), (0, (5, 5)), (0, (5, 10)))
    linestyle_ion = "dotted"

    # Electron and ion density
    plt.figure()
    plasma.el_dens.sel(t=time[0]).plot(color=colors[0], label="el.", alpha=alpha)
    plasma.ion_dens.sel(element=plasma.main_ion).sel(t=time[0]).plot(
        color=colors[0], linestyle=linestyle_ion, label=plasma.main_ion, alpha=alpha
    )
    for i, t in enumerate(time):
        plasma.el_dens.sel(t=t).plot(color=colors[i], alpha=alpha)
        plasma.ion_dens.sel(element=plasma.main_ion).sel(t=t).plot(
            color=colors[i], linestyle=linestyle_ion, alpha=alpha,
        )
    plt.title(f"{_title} Electron and Ion densities")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(m$^{-3}$)")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_electron_ion_density")

    # Neutral density
    plt.figure()
    for i, t in enumerate(time):
        plasma.neutral_dens.sel(t=t).plot(color=colors[i], alpha=alpha)
    plt.title(f"{_title} Neutral density")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(m$^{-3}$)")
    plt.yscale("log")
    if savefig:
        save_figure(fig_name=f"{figname}profiles_neutral_density")

    # Electron temperature
    plt.figure()
    ylim = (0, np.max([plasma.el_temp.max(), plasma.ion_temp.max()]) * 1.05)
    data = None
    if len(plasma.optimisation["el_temp"]) > 0 and len(bckc) > 0:
        diagn, quant = plasma.optimisation["el_temp"].split(".")
        quant, _ = quant.split(":")
        data = bckc[diagn][quant]
        pos = bckc[diagn][quant].pos.value
        pos_in = bckc[diagn][quant].pos.value - bckc[diagn][quant].pos.err_in
        pos_out = bckc[diagn][quant].pos.value + bckc[diagn][quant].pos.err_out
    for i, t in enumerate(time):
        plasma.el_temp.sel(t=t).plot(color=colors[i], alpha=alpha)
        if data is not None:
            plt.plot(pos[i], data.values[i], color=colors[i], marker="o", alpha=alpha)
            plt.hlines(data[i], pos_in[i], pos_out[i], color=colors[i], alpha=alpha)
    plt.ylim(ylim)
    plt.title(f"{_title} Electron temperature")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(eV)")
    if savefig:
        save_figure(fig_name=f"{figname}profiles_electron_temperature")

    # Ion temperature
    plt.figure()
    data = None
    if len(plasma.optimisation["ion_temp"]) > 0 and len(bckc) > 0:
        diagn, quant = plasma.optimisation["ion_temp"].split(".")
        quant, _ = quant.split(":")
        data = bckc[diagn][quant]
        pos = bckc[diagn][quant].pos.value
        pos_in = bckc[diagn][quant].pos.value - bckc[diagn][quant].pos.err_in
        pos_out = bckc[diagn][quant].pos.value + bckc[diagn][quant].pos.err_out
    for i, t in enumerate(time):
        plasma.ion_temp.sel(element="h").sel(t=t).plot(color=colors[i], alpha=alpha)
        if data is not None:
            plt.plot(pos[i], data[i], color=colors[i], marker="o", alpha=alpha)
            plt.hlines(data[i], pos_in[i], pos_out[i], color=colors[i], alpha=alpha)
    plt.ylim(ylim)
    plt.title(f"{_title} ion temperature")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(eV)")
    if savefig:
        save_figure(fig_name=f"{figname}profiles_ion_temperature")
    if data is not None:
        plt.figure()
        for i, t in enumerate(time):
            data.attrs["emiss"].sel(t=t).plot(color=colors[i], alpha=alpha)
        plt.title(f"{_title} {diagn.upper()} {quant.upper()} emission")
        plt.xlabel("Rho-poloidal")
        plt.ylabel("(a.u.)")
        if savefig:
            save_figure(
                fig_name=f"{figname}profiles_{diagn.upper()}_{quant.upper()}_emission"
            )

        if len(plasma.optimisation["ion_temp"]) > 0 and len(bckc) > 0:
            plt.figure()
            ylim = (0, 1.05)
            for i, t in enumerate(time):
                for q in data.attrs["fz"].sel(t=t).ion_charges:
                    data.attrs["fz"].sel(t=t, ion_charges=q).plot(
                        color=colors[i], alpha=alpha
                    )
            plt.ylim(ylim)
            plt.title(
                f"{_title} {diagn.upper()} {quant.upper()} ionization balance"
            )
            plt.xlabel("Rho-poloidal")
            plt.ylabel("(eV)")
            if savefig:
                save_figure(fig_name=f"{figname}fractional_abundance")

    # Total radiated power
    plt.figure()
    plasma.tot_rad.sum("element").sel(t=time[0]).plot(color=colors[0], label="Total")
    plasma.tot_rad.sel(element=plasma.main_ion).sel(t=time[0]).plot(
        color=colors[0],
        linestyle=linestyle_ion,
        alpha=alpha,
        label=elem_str[plasma.main_ion],
    )
    for j, elem in enumerate(plasma.impurities):
        plasma.tot_rad.sel(element=elem).sel(t=time[0]).plot(
            color=colors[0],
            linestyle=linestyle_imp[j],
            label=elem_str[elem],
            alpha=alpha,
        )
    for i, t in enumerate(time):
        plasma.tot_rad.sum("element").sel(t=t).plot(color=colors[i])
        plasma.tot_rad.sel(element=plasma.main_ion).sel(t=t).plot(
            color=colors[i], linestyle=linestyle_ion, alpha=alpha
        )
        for j, elem in enumerate(plasma.impurities):
            plasma.tot_rad.sel(element=elem).sel(t=t).plot(
                color=colors[i], linestyle=linestyle_imp[j], alpha=alpha
            )
    plt.title(f"{_title} Total radiated power")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(W m$^{-3}$)")
    # plt.yscale("log")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_total_radiated_power")

    # Impurity density
    plt.figure()
    for j, elem in enumerate(plasma.impurities):
        plasma.ion_dens.sel(element=elem).sel(t=time[0]).plot(
            color=colors[0],
            linestyle=linestyle_imp[j],
            label=elem_str[elem],
            alpha=alpha,
        )
    for i, t in enumerate(time):
        for j, elem in enumerate(plasma.impurities):
            plasma.ion_dens.sel(element=elem).sel(t=time[i]).plot(
                color=colors[i], linestyle=linestyle_imp[j], alpha=alpha,
            )
    plt.title(f"{_title} Impurity density")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(W)")
    plt.yscale("log")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_impurity_density")

    # Impurity concentration
    plt.figure()
    for j, elem in enumerate(plasma.impurities):
        (plasma.ion_dens.sel(element=elem) / plasma.el_dens).sel(t=time[0]).plot(
            color=colors[0],
            linestyle=linestyle_imp[j],
            label=elem_str[elem],
            alpha=alpha,
        )
    for i, t in enumerate(time):
        for j, elem in enumerate(plasma.impurities):
            (plasma.ion_dens.sel(element=elem) / plasma.el_dens).sel(t=time[i]).plot(
                color=colors[i], linestyle=linestyle_imp[j], alpha=alpha,
            )
    plt.title(f"{_title} Impurity concentration")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(W)")
    plt.yscale("log")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_impurity_concentration")

    # Effective charge
    plt.figure()
    plasma.zeff.sum("element").sel(t=time[0]).plot(color=colors[0], label="Total")
    plasma.zeff.sel(element=plasma.main_ion).sel(t=time[0]).plot(
        color=colors[0], linestyle=linestyle_ion, label=elem_str[plasma.main_ion]
    )
    for j, elem in enumerate(plasma.impurities):
        (plasma.zeff.sel(element=elem) + plasma.zeff.sel(element=plasma.main_ion)).sel(
            t=time[0]
        ).plot(
            color=colors[0],
            linestyle=linestyle_imp[j],
            label=elem_str[elem],
            alpha=alpha,
        )
    for i, t in enumerate(time):
        plasma.zeff.sum("element").sel(t=t).plot(color=colors[i])
        plasma.zeff.sel(element=plasma.main_ion).sel(t=t).plot(
            color=colors[i], linestyle=linestyle_ion
        )
        for j, elem in enumerate(plasma.impurities):
            (
                plasma.zeff.sel(element=elem) + plasma.zeff.sel(element=plasma.main_ion)
            ).sel(t=t).plot(color=colors[i], linestyle=linestyle_imp[j], alpha=alpha)
    plt.title(f"{_title} Effective charge")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_effective_charge")

    # Equilibrium reconstruction
    plt.figure()
    R = plasma.equilibrium.rho.R
    z = plasma.equilibrium.rho.z
    vmin = np.linspace(1, 0, len(plasma.time))
    for i, t in enumerate(plasma.time):
        rho = plasma.equilibrium.rho.sel(t=t, method="nearest")
        plt.contour(
            R,
            z,
            rho,
            levels=[1.0],
            alpha=alpha,
            cmap=cmap,
            vmin=vmin[i],
            vmax=vmin[i] + 1,
        )
        plt.plot(
            plasma.equilibrium.rmag.sel(t=t, method="nearest"),
            plasma.equilibrium.zmag.sel(t=t, method="nearest"),
            color=colors[i],
            marker="o",
            alpha=0.5,
        )

    plt.title(f"{_title} Plasma equilibrium")
    plt.xlabel("R (m)")
    plt.ylabel("z (m)")
    plt.axis("scaled")
    plt.xlim(0, 0.8)
    plt.ylim(-0.6, 0.6)
    if savefig:
        save_figure(fig_name=f"{figname}2D_equilibrium")


# def geometry(data):
#     plt.figure()
#     diag = data.keys()
#     quant = data[diag[0]].keys()
#
#     machine_dimensions =data[diag][quant].transform.


def save_figure(fig_name="", orientation="landscape", ext=".jpg"):
    plt.savefig(
        "/home/marco.sertoli/figures/Indica/" + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def get_figname(pulse=None, name=""):

    figname = ""
    if pulse is not None:
        figname = f"{str(int(pulse))}_"

    if len(name) > 0:
        figname += f"{name}_"

    return figname
