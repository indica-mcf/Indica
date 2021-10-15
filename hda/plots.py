import matplotlib.pylab as plt
from hda.plasma import Plasma

plt.ion()


def compare_data_bckc(data, bckc={}, pulse=None):
    colors = ("black", "blue", "cyan", "orange", "red")

    if "xrcs" in data.keys():
        plt.figure()
        for i, quant in enumerate(data["xrcs"].keys()):
            marker = "o"
            if "ti" in quant:
                marker = "x"
            data["xrcs"][quant].plot(
                marker=marker,
                color=colors[i],
                linestyle="dashed",
                label=f"{quant.upper()} XRCS",
                alpha=0.5,
            )
            if "xrcs" in bckc.keys():
                if quant in bckc["xrcs"]:
                    bckc["xrcs"][quant].plot(color=colors[i])
        plt.title(f"{pulse} Electron and ion temperature")
        plt.xlabel("Time (s)")
        plt.ylabel("(eV)")
        plt.legend()

    if "nirh1" in data.keys() or "smmh1" in data.keys():
        plt.figure()
        for i, diag in enumerate(("nirh1", "smmh1")):
            if diag in data.keys():
                data[diag]["ne"].plot(
                    marker="o",
                    color=colors[i],
                    linestyle="dashed",
                    label=f"Ne {diag.upper()}",
                    alpha=0.5,
                )
            if diag in bckc.keys():
                bckc[diag]["ne"].plot(color=colors[i])
        plt.title(f"{pulse} Electron density")
        plt.xlabel("Time (s)")
        plt.ylabel("(m$^{-3}$)")
        plt.legend()
