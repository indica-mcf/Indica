import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm, rcParams
import xarray as xr
from scipy import constants

plt.ion()

rcParams.update({"font.size": 12})


def compare_data_bckc(
    data,
    bckc,
    raw_data={},
    pulse=None,
    xlim=None,
    savefig=False,
    name="",
    title="",
    ploterr=True,
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
        icol = -1
        for quant in bckc["xrcs"].keys():
            if ("ti" not in quant) and ("te" not in quant):
                continue
            marker = "o"
            if "ti" in quant:
                marker = "x"

            if "xrcs" in raw_data.keys():
                raw_data["xrcs"][quant].plot(
                    color=colors[icol], linestyle="dashed", alpha=0.5,
                )
            plt.fill_between(
                data["xrcs"][quant].t,
                data["xrcs"][quant].values + data["xrcs"][quant].attrs["error"],
                data["xrcs"][quant].values - data["xrcs"][quant].attrs["error"],
                color=colors[icol],
                alpha=0.5,
            )
            data["xrcs"][quant].plot(
                marker=marker,
                color=colors[icol],
                linestyle="dashed",
                label=f"{quant.upper()} XRCS",
            )
            ylim0.append(np.nanmin(data["xrcs"][quant]))
            ylim1.append(np.nanmax(data["xrcs"][quant]) * 1.3)
        for i, quant in enumerate(bckc["xrcs"].keys()):
            if ("ti" not in quant) and ("te" not in quant):
                continue
            bckc["xrcs"][quant].plot(color=colors[icol], label="Back-calc", linewidth=3)

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
        icol = -1
        for quant in bckc["xrcs"].keys():
            if "int" not in quant or "/" in quant:
                continue
            icol += 1
            marker = "o"

            plt.figure()
            if "xrcs" in raw_data.keys():
                raw_data["xrcs"][quant].plot(
                    color=colors[icol], linestyle="dashed", alpha=0.5,
                )
            plt.fill_between(
                data["xrcs"][quant].t,
                data["xrcs"][quant].values + data["xrcs"][quant].attrs["error"],
                data["xrcs"][quant].values - data["xrcs"][quant].attrs["error"],
                color=colors[icol],
                alpha=0.5,
            )
            data["xrcs"][quant].plot(
                marker=marker,
                color=colors[icol],
                linestyle="dashed",
                label=f"{quant.upper()} XRCS",
            )

            bckc["xrcs"][quant].plot(color=colors[icol], label="Back-calc", linewidth=3)

            plt.xlim(xlim)
            plt.ylim(0, )
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
    quant = "wp"
    if diag in data.keys():
        plt.figure()
        ylim0, ylim1 = [], []
        i = 0
        if diag in raw_data.keys():
            (raw_data[diag][quant] / 1.0e3).plot(
                color=colors[i], linestyle="dashed", alpha=0.5,
            )
        if "error" in data[diag][quant].attrs:
            plt.fill_between(
                data[diag][quant].t,
                (data[diag][quant].values + data[diag][quant].attrs["error"]) / 1.0e3,
                (data[diag][quant].values - data[diag][quant].attrs["error"]) / 1.0e3,
                color=colors[i],
                alpha=0.5,
            )

        (data[diag][quant] / 1.0e3).plot(
            marker="o", color=colors[i], linestyle="dashed", label=f"Wp {diag.upper()}",
        )
        ylim0.append(np.nanmin(data[diag]["wp"] / 1.0e3))
        ylim1.append(np.nanmax(data[diag]["wp"] / 1.0e3) * 1.3)

        if diag in bckc.keys():
            (bckc[diag]["wp"] / 1.0e3).plot(
                color=colors[i], label="Back-calc", linewidth=3
            )
        plt.xlim(xlim)
        plt.title(f"{_title} Stored Energy")
        plt.xlabel("Time (s)")
        plt.ylabel("(kJ)")
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
        if "error" in data[diag][quant].attrs:
            plt.fill_between(
                data[diag][quant].t,
                (data[diag][quant].values + data[diag][quant].attrs["error"]),
                (data[diag][quant].values - data[diag][quant].attrs["error"]),
                color=colors[i],
                alpha=0.5,
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
            ylim0.append(np.nanmin(bckc[diag][quant]) * 0.7)
            ylim1.append(np.nanmax(bckc[diag][quant]) * 1.3)
        plt.xlim(xlim)
        plt.ylim(np.min(ylim0), np.max(ylim1))
        plt.title(f"{_title} Bremsstrahlung Intensity")
        plt.xlabel("Time (s)")
        plt.ylabel("(a.u.)")
        plt.legend()
        if savefig:
            save_figure(fig_name=f"{figname}data_LINES_bremsstrahlung")

    diag = "mag"
    quant = "vloop"
    if diag in data.keys():
        plt.figure()
        i = 0
        ylim0, ylim1 = [], []
        if diag in raw_data.keys():
            raw_data[diag][quant].plot(
                color=colors[i], linestyle="dashed", alpha=0.5,
            )

        if "error" in data[diag][quant].attrs:
            plt.fill_between(
                data[diag][quant].t,
                (data[diag][quant].values + data[diag][quant].attrs["error"]),
                (data[diag][quant].values - data[diag][quant].attrs["error"]),
                color=colors[i],
                alpha=0.5,
            )

        data[diag][quant].plot(
            marker="o",
            color=colors[i],
            linestyle="dashed",
            label=f"Vloop {diag.upper()}",
        )
        ylim0.append(np.nanmin(data[diag][quant]) * 0.7)
        ylim1.append(np.nanmax(data[diag][quant]) * 1.3)

        if diag in bckc.keys():
            bckc[diag][quant].plot(color=colors[i], label="Back-calc", linewidth=3)
        plt.xlim(xlim)
        plt.ylim(np.min(ylim0), np.max(ylim1))
        plt.title(f"{_title} Loop voltage")
        plt.xlabel("Time (s)")
        plt.ylabel("(V)")
        plt.legend()
        if savefig:
            save_figure(fig_name=f"{figname}data_MAG_Vloop")


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
                plasma.electron_temperature.max() * 1.05,
                plasma.ion_temperature.sel(element=plasma.main_ion).max() * 1.05,
            ]
        ),
    )

    ion_density = plasma.ion_density
    zeff = plasma.zeff
    total_radiation = plasma.total_radiation
    # sxr_radiation = plasma.sxr_radiation
    prad_tot = plasma.prad_tot
    prad_sxr = plasma.prad_sxr
    if hasattr(plasma, "electron_temperature_hi") and ploterr:
        plt.fill_between(
            plasma.time,
            plasma.electron_temperature_hi.sel(rho_poloidal=0),
            plasma.electron_temperature_lo.sel(rho_poloidal=0),
            color="blue",
            alpha=0.5,
        )
        plt.fill_between(
            plasma.time,
            plasma.ion_temperature_hi.sel(element=plasma.main_ion, rho_poloidal=0),
            plasma.ion_temperature_lo.sel(element=plasma.main_ion, rho_poloidal=0),
            color="red",
            alpha=0.5,
        )
        ylim = (
            0,
            np.max(
                [
                    plasma.electron_temperature_hi.max() * 1.05,
                    plasma.ion_temperature_hi.sel(element=plasma.main_ion).max() * 1.05,
                    ylim[1],
                ]
            ),
        )
    plasma.electron_temperature.sel(rho_poloidal=0).plot(label="Te(0)", color="blue", alpha=0.8)
    plasma.ion_temperature.sel(element=plasma.main_ion, rho_poloidal=0).plot(
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
    ylim = (0, plasma.electron_density.max() * 1.05)
    if hasattr(plasma, "electron_density_hi") and ploterr:
        plt.fill_between(
            plasma.time,
            plasma.electron_density_hi.sel(rho_poloidal=0),
            plasma.electron_density_lo.sel(rho_poloidal=0),
            color="blue",
            alpha=0.5,
        )
        plt.fill_between(
            plasma.time,
            plasma.ion_density_hi.sel(element=plasma.main_ion, rho_poloidal=0),
            plasma.ion_density_lo.sel(element=plasma.main_ion, rho_poloidal=0),
            color="red",
            alpha=0.5,
        )
        ylim = (
            0,
            np.max(
                [
                    plasma.electron_density_hi.max() * 1.05,
                    plasma.ion_density_hi.max() * 1.05,
                    ylim[1],
                ]
            ),
        )
    plasma.electron_density.sel(rho_poloidal=0).plot(label="Ne(0)", color="blue", alpha=0.8)
    ion_density.sel(element=plasma.main_ion, rho_poloidal=0).plot(
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
    prad_tot.sum("element").plot(label="Total", color="black")
    prad_tot.sel(element=plasma.main_ion).plot(
        color="black", label=elem_str[plasma.main_ion], linestyle="dotted"
    )
    for j, elem in enumerate(plasma.elements):
        prad_tot.sel(element=elem).plot(
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
    prad_sxr.sum("element").plot(label="Total", color="black")
    for j, elem in enumerate(plasma.elements):
        prad_sxr.sel(element=elem).plot(
            color=colors[j], label=elem_str[elem], linestyle=linestyles[j]
        )
    plt.title(f"{_title} SXR radiated power")
    plt.xlabel("Time (s)")
    plt.ylabel("")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_sxr_radiation_power")

    plt.figure()
    zeff_mean = zeff.mean("rho_poloidal")
    zeff_main_ion = zeff_mean.sel(element=plasma.main_ion)
    zeff_mean.sum("element").plot(label="Total", color="black")
    zeff_main_ion.plot(color=colors[0], label=elem_str[elem], linestyle=linestyles[0])
    for j, elem in enumerate(plasma.elements):
        if elem != plasma.main_ion:
            (zeff_mean.sel(element=elem) + zeff_main_ion).plot(
                color=colors[j], label=elem_str[elem], linestyle=linestyles[j]
            )
    plt.title(f"{_title} Average effective charge")
    plt.xlabel("Time (s)")
    plt.ylabel("")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}time_evol_effective_charge")

    plt.figure()
    c_ion = (ion_density / plasma.electron_density).sel(rho_poloidal=0)
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

    # plt.figure()
    # plasma.q_prof.sel(rho_poloidal=0).plot(color="black")
    # plt.title(f"{_title} Safety factor on axis")
    # plt.xlabel("Time (s)")
    # plt.ylabel("")
    # plt.legend()
    # if savefig:
    #     save_figure(fig_name=f"{figname}time_evol_axis_safety_factor")
    #
    # plt.figure()
    # plasma.vloop.plot(color="black")
    # plt.title(f"{_title} Loop voltage")
    # plt.xlabel("Time (s)")
    # plt.ylabel("")
    # plt.legend()
    # if savefig:
    #     save_figure(fig_name=f"{figname}time_evol_loop_voltage")


def data_time_evol(
    plasma, raw_data, data=None, savefig=False, name="", title="", ploterr=True
):
    figname = get_figname(pulse=plasma.pulse, name=name)
    _title = f"{plasma.pulse}"
    if len(title) > 1:
        _title += f" {title}"

    colors = ("black", "blue", "orange", "red")
    linestyles = ("dotted", (0, (5, 1)), (0, (5, 5)), (0, (5, 10)))

    fontsize = 9
    legendsize = 8
    markersize = 5
    plt.rc("font", size=fontsize)
    fig, axs = plt.subplots(4)

    ion_density = plasma.ion_density
    zeff = plasma.zeff
    total_radiation = plasma.total_radiation
    # sxr_radiation = plasma.sxr_radiation

    iax = 0
    const = 1.0e-6
    tmp = raw_data["efit"]["ipla"]
    axs[iax].plot(tmp.t, tmp.values * const, label="I$_P$(EFIT)")
    axs[iax].set_ylim(bottom=0, top=np.ceil(tmp.max().values * const))
    axs[iax].set_ylabel("(MA)")
    axs[iax].legend(fontsize=legendsize)
    axs[iax].set_title(_title)

    iax = 1
    const = 1.0e-3
    tmp = raw_data["efit"]["wp"]
    axs[iax].plot(tmp.t, tmp.values * const, label="W$_P$(EFIT)")
    axs[iax].set_ylim(bottom=0, top=50)
    axs[iax].set_ylabel("(kJ)")
    axs[iax].legend(fontsize=legendsize)

    iax = 2
    const = 1.0e-19
    tmp = raw_data["smmh1"]["ne"]
    axs[iax].plot(tmp.t, tmp.values * const, label="n$_e$(SMM)")
    axs[iax].set_ylim(bottom=0, top=np.ceil(tmp.max().values * const))
    axs[iax].set_ylabel("(10$^{19}$ m$^{-3}$)")
    axs[iax].legend(fontsize=legendsize)

    iax = 3
    const = 1.0e-3
    tmp = raw_data["xrcs"]["ti_w"]
    axs[iax].plot(
        tmp.t, tmp.values * const, label="Ti(XRCS)", marker="o", markersize=markersize
    )
    tmp = raw_data["xrcs"]["te_kw"]
    axs[iax].plot(
        tmp.t, tmp.values * const, label="Te(XRCS)", marker="o", markersize=markersize
    )

    chan = 2
    tmp_diag = data["princeton"]
    markers = ["v", "x"]
    for i, k in enumerate(tmp_diag.keys()):
        tmp = tmp_diag[k].sel(princeton_ti_coords=chan)
        err = tmp_diag[k].error.sel(princeton_ti_coords=chan)
        axs[iax].errorbar(
            tmp.t,
            tmp.values * const,
            err.values * const,
            label="Ti(CXRS)",
            marker=markers[i],
            markersize=markersize,
        )
    ev2k = constants.physical_constants["electron volt-kelvin relationship"][0]
    xlim = plt.xlim()
    plt.hlines(
        100.0e6 / ev2k / 1.0e3, xlim[0], xlim[1], color="red", linestyle="dotted",
    )
    axs[iax].set_ylabel("(keV)")
    axs[iax].legend(fontsize=legendsize)
    axs[iax].set_ylim(bottom=0, top=10)

    plt.xlabel("Time (s)")
    for ax in axs:
        ax.label_outer()

    if savefig:
        save_figure(fig_name=f"{figname}time_evol_raw_data")


def profiles(
    plasma,
    data=None,
    bckc=None,
    savefig=False,
    name="",
    alpha=0.8,
    title="",
    ploterr=False,
    tplot=None,
):
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

    if tplot is not None:
        if type(tplot) != list:
            tplot = [tplot]
    else:
        tplot = plasma.t

    if len(tplot) == 1:
        _title += f" t={tplot[0]:.3f}"
    cmap = cm.rainbow
    if len(tplot) > 1:
        varr = np.linspace(0, 1, len(tplot))
        colors = cmap(varr)
    else:
        colors = ["b"]

    # Impurity linestyles
    linestyle_imp = ((0, (5, 1)), (0, (5, 5)), (0, (5, 10)))
    linestyle_ion = "dotted"
    linestyle_fast = "dashed"

    ion_density = plasma.ion_density
    zeff = plasma.zeff
    total_radiation = plasma.total_radiation
    # sxr_radiation = plasma.sxr_radiation

    # Electron and ion density
    plt.figure()
    plasma.electron_density.sel(t=tplot[0], method="nearest").plot(
        color=colors[0], label="el.", alpha=alpha
    )
    ion_density.sel(element=plasma.main_ion).sel(t=tplot[0], method="nearest").plot(
        color=colors[0], linestyle=linestyle_ion, label=plasma.main_ion, alpha=alpha
    )
    plasma.fast_density.sel(t=tplot[0], method="nearest").plot(
        color=colors[0], linestyle=linestyle_fast, label="Fast ion", alpha=alpha,
    )

    if len(plasma.optimisation["electron_temperature"]) > 0 and bckc is not None:
        diagn = plasma.optimisation["electron_density"]["diagnostic"]
        quant = plasma.optimisation["electron_density"]["quantities"][0]
        value = bckc[diagn][quant]
        error = xr.zeros_like(value)
        if "error" in value.attrs.keys():
            error = value.error
    for i, t in enumerate(tplot):
        if hasattr(plasma, "electron_density_hi") and ploterr:
            plt.fill_between(
                plasma.electron_density.rho_poloidal,
                plasma.electron_density_hi.sel(t=t, method="nearest"),
                plasma.electron_density_lo.sel(t=t, method="nearest"),
                color=colors[i],
                alpha=0.5,
            )
            plt.fill_between(
                ion_density.rho_poloidal,
                plasma.ion_density_hi.sel(element=plasma.main_ion).sel(
                    t=t, method="nearest"
                ),
                plasma.ion_density_lo.sel(element=plasma.main_ion).sel(
                    t=t, method="nearest"
                ),
                color=colors[i],
                alpha=0.5,
            )
            plt.fill_between(
                plasma.fast_density.rho_poloidal,
                plasma.fast_density_hi.sel(t=t, method="nearest"),
                plasma.fast_density_lo.sel(t=t, method="nearest"),
                color=colors[i],
                alpha=0.5,
            )
        plasma.electron_density.sel(t=t, method="nearest").plot(color=colors[i], alpha=alpha)
        ion_density.sel(element=plasma.main_ion).sel(t=t, method="nearest").plot(
            color=colors[i], linestyle=linestyle_ion, alpha=alpha,
        )
        plasma.fast_density.sel(t=t, method="nearest").plot(
            color=colors[i], linestyle=linestyle_fast, alpha=alpha,
        )

    plt.title(f"{_title} Electron and Ion densities")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(m$^{-3}$)")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_electron_ion_density")

    # Neutral density
    plt.figure()
    for i, t in enumerate(tplot):
        if hasattr(plasma, "neutral_density_hi") and ploterr:
            plt.fill_between(
                plasma.neutral_density.rho_poloidal,
                plasma.neutral_density_hi.sel(t=t, method="nearest"),
                plasma.neutral_density_lo.sel(t=t, method="nearest"),
                color=colors[i],
                alpha=0.5,
            )
        plasma.neutral_density.sel(t=t, method="nearest").plot(
            color=colors[i], alpha=alpha
        )
    plt.title(f"{_title} Neutral density")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(m$^{-3}$)")
    plt.yscale("log")
    if savefig:
        save_figure(fig_name=f"{figname}profiles_neutral_density")

    # Electron temperature
    plt.figure()
    ylim = (0, np.max([plasma.electron_temperature.max(), plasma.ion_temperature.max() * 1.05]))
    value = None
    if len(plasma.optimisation["electron_temperature"]) > 0 and bckc is not None:
        diagn = plasma.optimisation["electron_temperature"]["diagnostic"]
        quant = plasma.optimisation["electron_temperature"]["quantities"][0]
        value = bckc[diagn][quant]
        error = xr.zeros_like(value)
        if "error" in value.attrs.keys():
            error = value.error
        pos = bckc[diagn][quant].pos["value"]
        pos_in = bckc[diagn][quant].pos["value"] - bckc[diagn][quant].pos["err_in"]
        pos_out = bckc[diagn][quant].pos["value"] + bckc[diagn][quant].pos["err_out"]
    for i, t in enumerate(tplot):
        if hasattr(plasma, "electron_temperature_hi") and ploterr:
            plt.fill_between(
                plasma.electron_temperature.rho_poloidal,
                plasma.electron_temperature_hi.sel(t=t, method="nearest"),
                plasma.electron_temperature_lo.sel(t=t, method="nearest"),
                color=colors[i],
                alpha=0.5,
            )
            ylim = (0, plasma.electron_temperature_hi.max() * 1.05)
        plasma.electron_temperature.sel(t=t, method="nearest").plot(color=colors[i], alpha=alpha)
        if value is not None:
            plt.errorbar(
                pos.sel(t=t, method="nearest"),
                value.sel(t=t, method="nearest"),
                yerr=error.sel(t=t, method="nearest"),
                color=colors[i],
                marker="o",
                alpha=alpha,
            )
            plt.hlines(
                value.sel(t=t, method="nearest"),
                pos_in.sel(t=t, method="nearest"),
                pos_out.sel(t=t, method="nearest"),
                color=colors[i],
                alpha=alpha,
            )
    plt.ylim(ylim)
    plt.title(f"{_title} Electron temperature")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(eV)")
    if savefig:
        save_figure(fig_name=f"{figname}profiles_electron_temperature")

    # Ion temperature
    plt.figure()
    value = None
    if len(plasma.optimisation["ion_temperature"]) > 0 and bckc is not None:
        diagn = plasma.optimisation["ion_temperature"]["diagnostic"]
        quant = plasma.optimisation["ion_temperature"]["quantities"][0]
        value = bckc[diagn][quant]
        error = xr.zeros_like(value)
        if "error" in value.attrs.keys():
            error = value.error
        pos = bckc[diagn][quant].pos["value"]
        pos_in = bckc[diagn][quant].pos["value"] - bckc[diagn][quant].pos["err_in"]
        pos_out = bckc[diagn][quant].pos["value"] + bckc[diagn][quant].pos["err_out"]
    for i, t in enumerate(tplot):
        if hasattr(plasma, "ion_temperature_hi") and ploterr:
            plt.fill_between(
                plasma.ion_temperature.rho_poloidal,
                plasma.ion_temperature_hi.sel(element="h").sel(t=t, method="nearest"),
                plasma.ion_temperature_lo.sel(element="h").sel(t=t, method="nearest"),
                color=colors[i],
                alpha=0.5,
            )
            ylim = (0, plasma.ion_temperature_hi.max() * 1.05)
        plasma.ion_temperature.sel(element="h").sel(t=t, method="nearest").plot(
            color=colors[i], alpha=alpha
        )
        if value is not None:
            plt.errorbar(
                pos.sel(t=t, method="nearest"),
                value.sel(t=t, method="nearest"),
                yerr=error.sel(t=t, method="nearest"),
                color=colors[i],
                marker="o",
                alpha=alpha,
            )
            plt.hlines(
                value.sel(t=t, method="nearest"),
                pos_in.sel(t=t, method="nearest"),
                pos_out.sel(t=t, method="nearest"),
                color=colors[i],
                alpha=alpha,
            )
            if "princeton" in data.keys():
                markers = ["v", "x"]
                for k, key in enumerate(data["princeton"].keys()):
                    # if key == "cxsfit_full":
                    #     continue
                    if np.min(np.abs(t - data["princeton"][key].t)) > plasma.dt / 2.0:
                        continue
                    chans = [2]
                    for j, ch in enumerate(data["princeton"][key].princeton_ti_coords):
                        if ch not in chans:
                            continue
                        _value = data["princeton"][key].sel(
                            princeton_ti_coords=ch, t=t, method="nearest"
                        )
                        _error = xr.zeros_like(value)
                        if "error" in value.attrs.keys():
                            _error = _value.error.sel(
                                princeton_ti_coords=ch, t=t, method="nearest"
                            )

                        _pos = _value.transform[j].rho_nbi.sel(t=t, method="nearest")
                        _pos_in = _value.transform[j].rho_in.sel(t=t, method="nearest")
                        _pos_out = _value.transform[j].rho_out.sel(
                            t=t, method="nearest"
                        )
                        plt.errorbar(
                            _pos,
                            _value,
                            yerr=_error,
                            color=colors[i],
                            marker=markers[k],
                            alpha=alpha,
                        )
                        plt.hlines(
                            _value, _pos_in, _pos_out, color=colors[i], alpha=alpha,
                        )
                        if len(chans) == 1:
                            plt.errorbar(
                                _pos - 0.08,
                                _value,
                                yerr=_error,
                                marker=markers[k],
                                alpha=0.5,
                                fmt="black",
                                mfc="black",
                            )
                            plt.hlines(
                                _value,
                                np.abs(_pos_in - 0.08),
                                np.abs(_pos_out - 0.08),
                                color="black",
                                alpha=0.5,
                                linestyle="dashed",
                            )
                            from scipy import constants

                            ev2k = constants.physical_constants[
                                "electron volt-kelvin relationship"
                            ][0]
                            plt.hlines(
                                100.0e6 / ev2k, 0, 1, color="red", linestyle="dotted", alpha=0.5,
                            )

    plt.ylim(ylim)
    plt.title(f"{_title} ion temperature")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(eV)")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_ion_temperature")
    if value is not None:
        plt.figure()
        for i, t in enumerate(tplot):
            value.attrs["emiss"].sel(t=t, method="nearest").plot(
                color=colors[i], alpha=alpha
            )
        plt.title(f"{_title} {diagn.upper()} {quant.upper()} emission")
        plt.xlabel("Rho-poloidal")
        plt.ylabel("(a.u.)")
        if savefig:
            save_figure(
                fig_name=f"{figname}profiles_{diagn.upper()}_{quant.upper()}_emission"
            )

        if len(plasma.optimisation["ion_temperature"]) > 0 and data is not None:
            fz = plasma.fz
            plt.figure()
            ylim = (0, 1.05)
            for i, t in enumerate(tplot):
                for q in fz["ar"].sel(t=t, method="nearest").ion_charges:
                    fz["ar"].sel(t=t, ion_charges=q, method="nearest").plot(
                        color=colors[i], alpha=alpha
                    )
            plt.ylim(ylim)
            plt.title(f"{_title} {diagn.upper()} {quant.upper()} ionization balance")
            plt.xlabel("Rho-poloidal")
            plt.ylabel("(eV)")
            if savefig:
                save_figure(fig_name=f"{figname}fractional_abundance")

    # Total radiated power
    const = 1.0e-3
    plt.figure()
    (total_radiation * const).sum("element").sel(t=tplot[0], method="nearest").plot(
        color=colors[0], label="Total"
    )
    (total_radiation * const).sel(element=plasma.main_ion).sel(
        t=tplot[0], method="nearest"
    ).plot(
        color=colors[0],
        linestyle=linestyle_ion,
        alpha=alpha,
        label=elem_str[plasma.main_ion],
    )
    for j, elem in enumerate(plasma.impurities):
        (total_radiation * const).sel(element=elem).sel(
            t=tplot[0], method="nearest"
        ).plot(
            color=colors[0],
            linestyle=linestyle_imp[j],
            label=elem_str[elem],
            alpha=alpha,
        )
    for i, t in enumerate(tplot):
        (total_radiation * const).sum("element").sel(t=t, method="nearest").plot(
            color=colors[i]
        )
        (total_radiation * const).sel(element=plasma.main_ion).sel(
            t=t, method="nearest"
        ).plot(color=colors[i], linestyle=linestyle_ion, alpha=alpha)
        for j, elem in enumerate(plasma.impurities):
            (total_radiation * const).sel(element=elem).sel(t=t, method="nearest").plot(
                color=colors[i], linestyle=linestyle_imp[j], alpha=alpha
            )
    plt.title(f"{_title} Total radiated power")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(kW m$^{-3}$)")
    # plt.yscale("log")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_total_radiated_power")

    # SXR radiated power
    const = 1.0e-3
    plot_sxr = False
    if hasattr(plasma, "_sxr_radiation") and plot_sxr:
        plt.figure()
        (sxr_radiation.sum("element").sel(t=tplot[0], method="nearest") * const).plot(
            color=colors[0], label="Total"
        )
        (
            sxr_radiation.sel(element=plasma.main_ion).sel(
                t=tplot[0], method="nearest"
            )
            * const
        ).plot(
            color=colors[0],
            linestyle=linestyle_ion,
            alpha=alpha,
            label=elem_str[plasma.main_ion],
        )
        for j, elem in enumerate(plasma.impurities):
            (
                sxr_radiation.sel(element=elem).sel(t=tplot[0], method="nearest")
                * const
            ).plot(
                color=colors[0],
                linestyle=linestyle_imp[j],
                label=elem_str[elem],
                alpha=alpha,
            )
        for i, t in enumerate(tplot):
            (sxr_radiation.sum("element").sel(t=t, method="nearest") * const).plot(
                color=colors[i]
            )
            (
                sxr_radiation.sel(element=plasma.main_ion).sel(t=t, method="nearest")
                * const
            ).plot(color=colors[i], linestyle=linestyle_ion, alpha=alpha)
            for j, elem in enumerate(plasma.impurities):
                (
                    sxr_radiation.sel(element=elem).sel(t=t, method="nearest") * const
                ).plot(color=colors[i], linestyle=linestyle_imp[j], alpha=alpha)
        plt.title(f"{_title} SXR radiated power")
        plt.xlabel("Rho-poloidal")
        plt.ylabel("(kW m$^{-3}$)")
        # plt.yscale("log")
        plt.legend()
        if savefig:
            save_figure(fig_name=f"{figname}profiles_total_radiated_power")

    # Impurity density
    plt.figure()
    for j, elem in enumerate(plasma.impurities):
        ion_density.sel(element=elem).sel(t=tplot[0], method="nearest").plot(
            color=colors[0],
            linestyle=linestyle_imp[j],
            label=elem_str[elem],
            alpha=alpha,
        )
    for i, t in enumerate(tplot):
        for j, elem in enumerate(plasma.impurities):
            ion_density.sel(element=elem).sel(t=tplot[i], method="nearest").plot(
                color=colors[i], linestyle=linestyle_imp[j], alpha=alpha,
            )
    plt.title(f"{_title} Impurity density")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(m$^{-3}$)")
    plt.yscale("log")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_impurity_density")

    # Impurity concentration
    plt.figure()
    for j, elem in enumerate(plasma.impurities):
        (ion_density.sel(element=elem) / plasma.electron_density).sel(
            t=tplot[0], method="nearest"
        ).plot(
            color=colors[0],
            linestyle=linestyle_imp[j],
            label=elem_str[elem],
            alpha=alpha,
        )
    for i, t in enumerate(tplot):
        for j, elem in enumerate(plasma.impurities):
            (ion_density.sel(element=elem) / plasma.electron_density).sel(
                t=t, method="nearest"
            ).plot(
                color=colors[i], linestyle=linestyle_imp[j], alpha=alpha,
            )
    plt.title(f"{_title} Impurity concentration")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("")
    plt.yscale("log")
    plt.legend()
    if savefig:
        save_figure(fig_name=f"{figname}profiles_impurity_concentration")

    # Effective charge
    plt.figure()
    zeff.sum("element").sel(t=tplot[0], method="nearest").plot(
        color=colors[0], label="Total"
    )
    zeff.sel(element=plasma.main_ion).sel(t=tplot[0], method="nearest").plot(
        color=colors[0], linestyle=linestyle_ion, label=elem_str[plasma.main_ion]
    )
    for j, elem in enumerate(plasma.impurities):
        (zeff.sel(element=elem) + zeff.sel(element=plasma.main_ion)).sel(
            t=tplot[0], method="nearest"
        ).plot(
            color=colors[0],
            linestyle=linestyle_imp[j],
            label=elem_str[elem],
            alpha=alpha,
        )
    for i, t in enumerate(tplot):
        zeff.sum("element").sel(t=t, method="nearest").plot(color=colors[i])
        zeff.sel(element=plasma.main_ion).sel(t=t, method="nearest").plot(
            color=colors[i], linestyle=linestyle_ion
        )
        for j, elem in enumerate(plasma.impurities):
            (
                zeff.sel(element=elem) + zeff.sel(element=plasma.main_ion)
            ).sel(t=t, method="nearest").plot(
                color=colors[i], linestyle=linestyle_imp[j], alpha=alpha
            )
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
    # if tplot is not None:
    #     vmin = [0]*len(plasma.time)
    for i, t in enumerate(tplot):
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
