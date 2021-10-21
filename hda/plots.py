import matplotlib.pylab as plt
import numpy as np

plt.ion()


def compare_data_bckc(data, bckc={}, raw_data={}, pulse=None, xlim=None):
    colors = ("black", "blue", "purple", "orange", "red")

    if "xrcs" in data.keys():
        plt.figure()
        for i, quant in enumerate(data["xrcs"].keys()):
            marker = "o"
            if "ti" in quant:
                marker = "x"
            if "xrcs" in raw_data.keys():
                raw_data["xrcs"][quant].plot(
                    color=colors[i],
                    linestyle="dashed",
                    alpha=0.5,
                )
            plt.fill_between(
                data["xrcs"][quant].t,
                data["xrcs"][quant].values + data["xrcs"][quant].attrs["error"],
                data["xrcs"][quant].values - data["xrcs"][quant].attrs["error"],
                color=colors[i], alpha=0.5,
            )
            data["xrcs"][quant].plot(
                marker=marker,
                color=colors[i],
                linestyle="dashed",
                label=f"{quant.upper()} XRCS",
            )
            if "xrcs" in bckc.keys():
                if quant in bckc["xrcs"]:
                    bckc["xrcs"][quant].plot(color=colors[i], label="Back-calc", linewidth=3)
        plt.xlim(xlim)
        plt.ylim(0,)
        plt.title(f"{pulse} Electron and ion temperature")
        plt.xlabel("Time (s)")
        plt.ylabel("(eV)")
        plt.legend()

    if "nirh1" in data.keys() or "smmh1" in data.keys():
        plt.figure()
        for i, diag in enumerate(("nirh1", "nirh1_bin", "smmh1")):
            if diag in data.keys():
                if diag in raw_data.keys():
                    raw_data[diag]["ne"].plot(
                        color=colors[i],
                        linestyle="dashed",
                    )
                plt.fill_between(
                    data[diag]["ne"].t,
                    data[diag]["ne"].values + data[diag]["ne"].attrs["error"],
                    data[diag]["ne"].values - data[diag]["ne"].attrs["error"],
                    color=colors[i],alpha=0.5,
                )
                data[diag]["ne"].plot(
                    marker="o",
                    color=colors[i],
                    linestyle="dashed",
                    label=f"Ne {diag.upper()}",
                )

            if diag in bckc.keys():
                bckc[diag]["ne"].plot(color=colors[i], label="Back-calc", linewidth=3)
        plt.xlim(xlim)
        plt.ylim(0,)
        plt.title(f"{pulse} Electron density")
        plt.xlabel("Time (s)")
        plt.ylabel("(m$^{-3}$)")
        plt.legend()

    diag = "efit"
    if diag in data.keys():
        plt.figure()
        i = 0
        if diag in raw_data.keys():
            raw_data[diag]["wp"].plot(
                color=colors[i],
                linestyle="dashed",
                alpha=0.5,
            )
        data[diag]["wp"].plot(
            marker="o",
            color=colors[i],
            linestyle="dashed",
            label=f"Wp {diag.upper()}",
        )

        if diag in bckc.keys():
            bckc[diag]["wp"].plot(color=colors[i], label="Back-calc", linewidth=3)
        plt.xlim(xlim)
        plt.title(f"{pulse} Stored Energy")
        plt.xlabel("Time (s)")
        plt.ylabel("(J)")
        plt.legend()


def profiles(plasma):
    plt.figure()
    for t in plasma.t:
        plasma.el_dens.sel(t=t).plot()
    plt.title(f"{plasma.pulse} Electron density")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(m$^{-3}$)")

    ylim = (0, np.max([plasma.el_temp.max(), plasma.ion_temp.max()]) * 1.05)
    plt.figure()
    for t in plasma.t:
        plasma.el_temp.sel(t=t).plot()
    plt.ylim(ylim)
    plt.title(f"{plasma.pulse} Electron temperature")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(eV)")

    plt.figure()
    for t in plasma.t:
        plasma.ion_temp.sel(element="h").sel(t=t).plot()
    plt.ylim(ylim)
    plt.title(f"{plasma.pulse} ion density")
    plt.xlabel("Rho-poloidal")
    plt.ylabel("(eV)")


# def geometry(data):
#     plt.figure()
#     diag = data.keys()
#     quant = data[diag[0]].keys()
#
#     machine_dimensions =data[diag][quant].transform.
