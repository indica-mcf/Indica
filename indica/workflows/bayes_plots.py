from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
import pickle

from indica.utilities import set_plot_rcparams, set_axis_sci


def plot_profile(
    profile,
    blobkey: str,
    figheader="./results/test/",
    phantom_profile=None,
    logscale=False,
    sharefig=False,
    filename="",
    filetype=".png",
    linestyle="--",
    color="blue",
):
    set_plot_rcparams("profiles")

    if blobkey == "electron_temperature":
        legkey = "Te"
    elif blobkey == "ion_temperature":
        legkey = "Ti"
    elif blobkey == "ion_density":
        legkey = "Ni"
    elif blobkey == "electron_density":
        legkey = "Ne"
    elif blobkey == "impurity_density":
        legkey = "Nimp"
    elif blobkey == "fast_density":
        legkey = "Nfast"
    elif blobkey == "neutral_density":
        legkey = "Nneut"
    plt.fill_between(
        profile.rho_poloidal,
        profile.quantile(0.16, dim="index"),
        profile.quantile(0.84, dim="index"),
        label=f"{legkey}, 68% Confidence",
        zorder=3,
        color=color,
        alpha=0.9,
    )
    if legkey != "Nfast":
        plt.fill_between(
            profile.rho_poloidal,
            profile.quantile(0.025, dim="index"),
            profile.quantile(0.975, dim="index"),
            label=f"{legkey}, 95% Confidence",
            zorder=2,
            color="grey",
            alpha=0.4,
        )
        plt.fill_between(
            profile.rho_poloidal,
            profile.quantile(0.00, dim="index"),
            profile.quantile(1.00, dim="index"),
            label=f"{legkey}, Max-Min",
            zorder=1,
            color="lightgrey",
            alpha=0.2,
        )

    if phantom_profile is not None:
        if "element" in phantom_profile[blobkey].dims:
            phantom = phantom_profile[blobkey].sel(element="ar")
        else:
            phantom = phantom_profile[blobkey]
        phantom.plot(
            label=f"{legkey}, phantom profile",
            linestyle=linestyle,
            color="black",
            zorder=4,
        )

    plt.legend()
    if sharefig:
        return

    set_axis_sci()
    if logscale:
        plt.yscale("log")

    plt.xlabel("Rho poloidal")
    plt.ylabel(f"{profile.datatype[0].capitalize()} [{profile.units}]")
    if filename:
        plt.savefig(figheader + f"{filename}{filetype}")
    else:
        plt.savefig(figheader + f"{blobkey}{filetype}")
    plt.close("all")


def _plot_1d(
    blobs: dict,
    blobkey: str,
    diag_data: dict,
    filename: str,
    figheader="./results/test/",
    ylabel="a.u.",
    xlabel="[]",
    xlim=None,
    figsize=(6.4, 4.8),
    **kwargs,
):
    if blobkey not in blobs.keys():
        raise ValueError(f"{blobkey} not in blobs")

    plt.figure(figsize=figsize)
    blob_data = blobs[blobkey]
    dims = tuple(name for name in blob_data.dims if name != "index")
    plt.fill_between(
        blob_data.__getattr__(dims[0]),
        blob_data.quantile(0.16, dim="index"),
        blob_data.quantile(0.84, dim="index"),
        label=f"{blobkey}, 68% Confidence",
        zorder=3,
        color="blue",
    )
    plt.fill_between(
        blob_data.__getattr__(dims[0]),
        blob_data.quantile(0.025, dim="index"),
        blob_data.quantile(0.975, dim="index"),
        label=f"{blobkey}, 95% Confidence",
        zorder=2,
        color="grey",
    )
    plt.fill_between(
        blob_data.__getattr__(dims[0]),
        blob_data.quantile(0.00, dim="index"),
        blob_data.quantile(1.00, dim="index"),
        label=f"{blobkey}, Max-Min",
        zorder=1,
        color="lightgrey",
    )
    if "channel" in dims:
        plt.plot(
            diag_data[blobkey].__getattr__(dims[0]),
            diag_data[blobkey].sel(t=blob_data.t).values,
            "k^",
            label=f"{blobkey} data",
            zorder=4,
        )
    else:
        plt.plot(
            diag_data[blobkey].__getattr__(dims[0]),
            diag_data[blobkey].sel(t=blob_data.t).values,
            "k-",
            label=f"{blobkey} data",
            zorder=4,
        )
    set_axis_sci()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    plt.legend()
    plt.savefig(figheader + filename)
    plt.close()


def violinplot(
    data: dict,
    key: str,
    diag_data: dict,
    filename: str,
    ylabel="[a.u.]",
    figheader="./results/test/",
    **kwargs,
):
    set_plot_rcparams("multi")
    fig, axs = plt.subplots(
        1,
        1,
    )
    _data = data[key].values
    _data = _data[
        ((_data > np.quantile(_data, 0.16)) & (_data < np.quantile(_data, 0.84)))
    ]
    violin = axs.violinplot(
        _data,
        showextrema=False,
        # quantiles=[0.025, 0.975, 0.16, 0.84],
        # showmedians=True,
    )
    violin["bodies"][0].set_edgecolor("black")
    axs.set_xlabel(key)
    top = axs.get_ylim()[1]
    bot = axs.get_ylim()[0]
    axs.set_ylim( top=top * 1.1, bottom=bot * 0.9)
    axs.set_ylabel(f"{ylabel}")
    y = diag_data[key].sel(t=data[key].t).values
    axs.errorbar(
        1, y=y, yerr=y * 0.10, fmt="D", ecolor="black", capsize=10, color="black"
    )
    set_axis_sci()
    plt.setp([axs.get_xticklabels()], visible=False)
    plt.savefig(figheader + filename)
    plt.close()


def histograms(data, diag_data, filename):
    nfig = len(data)
    fig, axs = plt.subplots(1, nfig, figsize=(16, 6))
    for idx, key in enumerate(data.keys()):
        n, bins, patches = axs[idx].hist(data[key], 50, density=True)
        q1 = (np.percentile(data[key], 16), np.percentile(data[key], 84))
        q2 = (np.percentile(data[key], 2.5), np.percentile(data[key], 97.5))
        idx_high = np.argwhere((bins > q1[0]) & (bins < q1[1])).flatten()
        idx_low = np.argwhere((bins > q2[0]) & (bins < q2[1])).flatten()
        for patch in patches:
            patch.set_facecolor("lightgrey")
        for i in idx_low:
            patches[i].set_facecolor("grey")
        for i in idx_high:
            patches[i].set_facecolor("red")

        axs[idx].set_xlabel(f"{key} ({data[key].datatype[0]})")

        axs[idx].axvline(
            x=diag_data[key].sel(t=data[key].t).values,
            color="black",
            linestyle="-.",
            label=f"{key} data",
        )
    axs[0].set_ylabel("pdf ()")

    plt.savefig(filename)
    plt.close()


def plot_bayes_result(
    results,
    figheader="./results/test/",
    filetype=".png",
    **kwargs,
):

    diag_data = results["diag_data"]
    blobs = results["blobs"]
    samples = results["samples"]
    prior_samples = results["prior_samples"]
    param_names = results["param_names"]
    phantom_profiles = results["phantom_profiles"]
    autocorr = results["autocorr"]

    plt.figure()
    plt.plot(
        np.arange(0, autocorr.__len__())[np.isfinite(autocorr)],
        autocorr[np.isfinite(autocorr)],
        label="average tau",
    )
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("auto-correlation time (iterations)")
    plt.savefig(figheader + "average_tau" + filetype)
    plt.close()

    if "cxff_pi.ti" in blobs.keys():
        blobs["cxff_pi.ti0"] = blobs["cxff_pi.ti"].sel(
            channel=diag_data["cxff_pi.ti"].channel
        )
        diag_data["cxff_pi.ti0"] = diag_data["cxff_pi.ti"].sel(
            channel=diag_data["cxff_pi.ti"].channel
        )

    key = "cxff_pi.ti0"
    if key in blobs.keys():
        violinplot(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}" + filetype,
            figheader=figheader,
            ylabel="Temperature [eV]",
        )

    key = "efit.wp"
    if key in blobs.keys():
        violinplot(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}" + filetype,
            figheader=figheader,
            ylabel="Energy [J]",
        )
    key = "smmh1.ne"
    if key in blobs.keys():
        violinplot(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}" + filetype,
            figheader=figheader,
            ylabel=r"Line Integrated Density [$m^{-2}$]",
        )
    key = "xrcs.te_kw"
    if key in blobs.keys():
        violinplot(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}" + filetype,
            figheader=figheader,
            ylabel="Temperature [eV]",
        )
    key = "xrcs.ti_w"
    if key in blobs.keys():
        violinplot(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}" + filetype,
            figheader=figheader,
            ylabel="Temperature [eV]",
        )

    set_plot_rcparams("multi")
    key = "xrcs.spectra"
    if key in blobs.keys():
        _plot_1d(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}" + filetype,
            figheader=figheader,
            ylabel="Intensity [a.u.]",
            xlabel="Wavelength [nm]",
            xlim=(0.394, 0.401),
            figsize=(6, 4),
        )
    key = "cxff_pi.ti"
    if key in blobs.keys():
        _plot_1d(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}" + filetype,
            figheader=figheader,
            ylabel="Temperature [eV]",
            xlabel="Channel",
        )

    key = "electron_temperature"
    plot_profile(
        blobs[key],
        key,
        figheader=figheader,
        filetype=filetype,
        phantom_profile=phantom_profiles,
        sharefig=True,
        color="blue",
        linestyle="dashdot",
    )
    key = "ion_temperature"
    plot_profile(
        blobs[key].sel(element="ar"),
        key,
        figheader=figheader,
        filename="temperature",
        filetype=filetype,
        phantom_profile=phantom_profiles,
        color="red",
        linestyle="dotted",
    )

    key = "electron_density"
    plot_profile(
        blobs[key],
        key,
        figheader=figheader,
        filetype=filetype,
        phantom_profile=phantom_profiles,
        color="blue",
        sharefig=True
    )
    key = "ion_density"
    plot_profile(
        blobs[key].sel(element="h"),
        key,
        figheader=figheader,
        filetype=filetype,
        phantom_profile=phantom_profiles,
        sharefig=True,
        color="red",
    )
    key = "fast_density"
    plot_profile(
        blobs[key],
        key,
        figheader=figheader,
        filename="density",
        filetype=filetype,
        phantom_profile=phantom_profiles,
        color="green",
    )

    key = "impurity_density"
    for elem in blobs[key].element.values:
        plot_profile(
            blobs[key].sel(element=elem),
            key,
            figheader=figheader,
            filename=f"{elem} density",
            filetype=filetype,
            phantom_profile=phantom_profiles,
            color="red",
        )

    key = "neutral_density"
    plot_profile(
        blobs[key],
        key,
        filename="",
        figheader=figheader,
        filetype=filetype,
        phantom_profile=phantom_profiles,
        logscale=True,
    )

    corner.corner(samples, labels=param_names)
    plt.savefig(figheader + "posterior" + filetype)

    corner.corner(prior_samples, labels=param_names)
    plt.savefig(figheader + "prior" + filetype)
    plt.close("all")


if __name__ == "__main__":
    filehead = "./results/10009_60ms_long/"
    with open(filehead + "results.pkl", "rb") as handle:
        results = pickle.load(handle)
    plot_bayes_result(results, filehead, filetype=".png")
