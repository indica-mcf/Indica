import matplotlib.pylab as plt
from hda.plasma import Plasma

plt.ion()


def compare_data_bckc(plasma: Plasma):
    colors = ("black", "blue", "cyan", "orange", "red")

    plt.figure()
    for i, quant in enumerate(plasma.data["xrcs"].keys()):
        marker = "o"
        if "ti" in quant:
            marker = "x"
        plasma.data["xrcs"][quant].plot(
            marker=marker,
            color=colors[i],
            linestyle="dashed",
            label=f"{quant.upper()} XRCS",
            alpha=0.5,
        )
        if quant in plasma.bckc["xrcs"]:
            plasma.bckc["xrcs"][quant].plot(color=colors[i])
    plt.title(f"{plasma.pulse} Electron and ion temperature")
    plt.xlabel("Time (s)")
    plt.ylabel("(eV)")
    plt.legend()

    plt.figure()
    for i, diag in enumerate(("nirh1", "smmh1")):
        if diag in plasma.data.keys():
            plasma.data[diag]["ne"].plot(
                marker="o",
                color=colors[i],
                linestyle="dashed",
                label=f"Ne {diag.upper()}",
                alpha=0.5,
            )
        if diag in plasma.bckc.keys():
            plasma.bckc[diag]["ne"].plot(color=colors[i])
    plt.title(f"{plasma.pulse} Electron density")
    plt.xlabel("Time (s)")
    plt.ylabel("(m$^{-3}$)")
    plt.legend()
