import os
from pathlib import Path
import pickle

import corner
import flatdict
import matplotlib.pyplot as plt
import numpy as np
from skopt.plots import plot_evaluations
import xarray as xr

from indica.utilities import set_axis_sci
from indica.utilities import set_plot_rcparams


def plot_profile(
    profile: xr.DataArray,
    phantom_profile: xr.DataArray,
    label: str,
    figheader="./results/test/",
    logscale=False,
    sharefig=False,
    filename="",
    filetype=".png",
    linestyle="--",
    color="blue",
):
    set_plot_rcparams("profiles")
    plt.fill_between(
        profile.rho_poloidal,
        profile.quantile(0.16, dim="sample_idx"),
        profile.quantile(0.84, dim="sample_idx"),
        label=f"{label}, 68% Confidence",
        zorder=3,
        color=color,
        alpha=0.9,
    )
    if label != "NFAST":
        plt.fill_between(
            profile.rho_poloidal,
            profile.quantile(0.025, dim="sample_idx"),
            profile.quantile(0.975, dim="sample_idx"),
            label=f"{label}, 95% Confidence",
            zorder=2,
            color="grey",
            alpha=0.4,
        )
        plt.fill_between(
            profile.rho_poloidal,
            profile.quantile(0.005, dim="sample_idx"),
            profile.quantile(0.995, dim="sample_idx"),
            label=f"{label}, Max-Min",
            zorder=1,
            color="lightgrey",
            alpha=0.2,
        )

    if phantom_profile.any():
        phantom_profile.plot(
            label=f"{label}, phantom profile",
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
    # plt.ylabel(f"{profile.datatype[0].capitalize()} [{profile.units}]")
    if filename:
        plt.savefig(figheader + f"{filename}{filetype}")
    else:
        plt.savefig(figheader + f"{label}{filetype}")
    plt.close("all")


def _plot_1d(
    data: xr.DataArray,
    diag_data: xr.DataArray,
    filename: str,
    figheader="./results/test/",
    label="",
    ylabel="a.u.",
    xlabel="[]",
    xlim=None,
    figsize=(6.4, 4.8),
    hide_legend=False,
    capsize=3,
    markersize=4,
    elinewidth=2,
    **kwargs,
):
    set_plot_rcparams("multi")
    plt.figure(figsize=figsize)
    dims = tuple(name for name in data.dims if name != "sample_idx")
    plt.fill_between(
        data.__getattr__(dims[0]),
        data.quantile(0.16, dim="sample_idx"),
        data.quantile(0.84, dim="sample_idx"),
        label=f"{label}, 68% Confidence",
        zorder=3,
        color="blue",
    )
    plt.fill_between(
        data.__getattr__(dims[0]),
        data.quantile(0.025, dim="sample_idx"),
        data.quantile(0.975, dim="sample_idx"),
        label=f"{label}, 95% Confidence",
        zorder=2,
        color="grey",
    )
    plt.fill_between(
        data.__getattr__(dims[0]),
        data.quantile(0.005, dim="sample_idx"),
        data.quantile(0.995, dim="sample_idx"),
        label=f"{label}, Max-Min",
        zorder=1,
        color="lightgrey",
    )
    plt.errorbar(
        diag_data.__getattr__(dims[0]),
        diag_data,
        diag_data.error,
        fmt="k*",
        label=f"{label} data",
        zorder=4,
        capsize=capsize,
        markersize=markersize,
        elinewidth=elinewidth,
    )

    plt.gca().set_ylim(bottom=0)
    set_axis_sci()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    if not hide_legend:
        plt.legend()
    plt.savefig(figheader + filename)
    plt.close()


def violinplot(
    data,
    diag_data,
    filename: str,
    xlabel="",
    ylabel="[a.u.]",
    figheader="./results/test/",
    **kwargs,
):
    set_plot_rcparams("multi")
    fig, axs = plt.subplots(
        1,
        1,
    )
    _data = data[((data > np.quantile(data, 0.16)) & (data < np.quantile(data, 0.84)))]

    violin = axs.violinplot(
        _data,
        showextrema=False,
        # quantiles=[0.025, 0.975, 0.16, 0.84],
        # showmedians=True,
    )
    axs.errorbar(
        1,
        y=diag_data,
        yerr=diag_data.error,
        fmt="D",
        ecolor="black",
        capsize=10,
        color="black",
    )
    violin["bodies"][0].set_edgecolor("black")
    axs.set_xlabel(xlabel)
    top = axs.get_ylim()[1]
    # bot = axs.get_ylim()[0]
    axs.set_ylim(top=top * 1.1, bottom=0)
    axs.set_ylabel(f"{ylabel}")

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

        # axs[idx].set_xlabel(f"{key} ({data[key].datatype[0]})")

        axs[idx].axvline(
            x=diag_data[key].sel(t=data[key].t).values,
            color="black",
            linestyle="-.",
            label=f"{key} data",
        )
    axs[0].set_ylabel("pdf ()")

    plt.savefig(filename)
    plt.close()


def plot_autocorr(autocorr, param_names, figheader, filetype=".png"):
    plt.figure()

    x_data = (
        np.ones(shape=(autocorr.shape))
        * np.arange(0, autocorr[:, 0].__len__())[:, None]
    )
    plt.plot(np.where(np.isfinite(autocorr), x_data, np.nan), autocorr, "x")
    plt.legend(param_names)
    plt.xlabel("iterations")
    plt.ylabel("auto-correlation time (iterations)")
    plt.savefig(figheader + "average_tau" + filetype)
    plt.close()


# flake8: noqa: C901
def plot_bayes_result(
    results=None,
    filepath="./results/test/",
    filetype=".png",
    **kwargs,
):
    # delete all but pickle in directory and remove empty directories
    for root, dirs, files in os.walk(filepath):
        for dir in dirs:
            if not os.listdir(root + dir):
                print(f"Deleting {os.path.join(root, dir)}")
                os.rmdir(os.path.join(root, dir))
        for f in files:
            if f.endswith(".pkl"):
                continue
            else:
                print(f"Deleting {os.path.join(root, f)}")
                os.remove(os.path.join(root, f))

    if results is None:
        with open(filepath + "results.pkl", "rb") as handle:
            results = pickle.load(handle)

    # Create time directories
    time = results["TIME"]
    element = results["ELEMENT"]
    for t in time:
        Path(filepath + f"/t:{t:.2f}").mkdir(parents=True, exist_ok=True)

    diag_data = flatdict.FlatDict(results["DIAG_DATA"], ".")
    model_data = flatdict.FlatDict(results["MODEL_DATA"], ".")
    profiles = flatdict.FlatDict(results["PROFILE_STAT"], ".")
    post_sample = results["OPTIMISATION"]["POST_SAMPLE"]
    prior_sample = results["OPTIMISATION"]["PRIOR_SAMPLE"]
    gp_regression = results["OPTIMISATION"].get("GP_REGRESSION", {})
    auto_corr = results["OPTIMISATION"]["AUTO_CORR"]
    param_names = results["OPTIMISATION"]["PARAM_NAMES"]
    phantom_profiles = flatdict.FlatDict(results["PHANTOMS"], ".")

    # select time sample_idx for plotting
    for t_idx, t in enumerate(time):
        figheader = filepath + f"t:{t:.2f}/"

        if any(gp_regression):
            plot_evaluations(gp_regression[t_idx], dimensions=param_names)
            plt.savefig(figheader + "gp_evaluations")
            plt.close()
            # plot_objective(gp_regression[t_idx], )
            # plt.savefig(figheader + "gp_objective")
            # plt.close()

        plot_autocorr(
            auto_corr[
                t_idx,
            ],
            param_names,
            figheader,
            filetype=filetype,
        )
        # set_plot_rcparams("multi")
        key = "EFIT.WP"
        if key in model_data.keys():
            violinplot(
                model_data[key].sel(t=t),
                diag_data[key].sel(t=t),
                f"{key.replace('.', '_')}" + filetype,
                xlabel=key,
                figheader=figheader,
                ylabel="Energy [J]",
            )
        key = "SMMH1.NE"
        if key in model_data.keys():
            violinplot(
                model_data[key].sel(t=t),
                diag_data[key].sel(t=t),
                f"{key.replace('.', '_')}" + filetype,
                figheader=figheader,
                xlabel=key,
                ylabel=r"Line Integrated Density [$m^{-2}$]",
            )
        key = "XRCS.TE_KW"
        if key in model_data.keys():
            violinplot(
                model_data[key].sel(t=t),
                diag_data[key].sel(t=t),
                f"{key.replace('.', '_')}" + filetype,
                figheader=figheader,
                xlabel=key,
                ylabel="Temperature [eV]",
            )
        key = "XRCS.TI_W"
        if key in model_data.keys():
            violinplot(
                model_data[key].sel(t=t),
                diag_data[key].sel(t=t),
                f"{key.replace('.', '_')}" + filetype,
                figheader=figheader,
                xlabel=key,
                ylabel="Temperature [eV]",
            )

        key = "XRCS.RAW_SPECTRA"
        if key in model_data.keys():
            _plot_1d(
                model_data[key].sel(t=t),
                diag_data[key].sel(t=t),
                f"{key.replace('.', '_')}" + filetype,
                label=key,
                figheader=figheader,
                ylabel="Intensity [count/s]",
                xlabel="Wavelength [nm]",
                xlim=(0.394, 0.401),
                figsize=(15, 6),
                elinewidth=0,
                capsize=0,
            )
        key = "CXFF_PI.TI"
        if key in model_data.keys():
            _plot_1d(
                model_data[key].sel(t=t),
                diag_data[key].sel(t=t),
                f"{key.replace('.', '_')}" + filetype,
                label=key,
                figheader=figheader,
                ylabel="Temperature [eV]",
                xlabel="Channel",
                # hide_legend=True,
            )
        key = "CXFF_TWS_C.TI"
        if key in model_data.keys():
            _plot_1d(
                model_data[key].sel(t=t),
                diag_data[key].sel(t=t),
                f"{key.replace('.', '_')}" + filetype,
                label=key,
                figheader=figheader,
                ylabel="Temperature [eV]",
                xlabel="Channel",
                # hide_legend=True,
            )

        key = "TS.TE"
        if key in model_data.keys():
            _plot_1d(
                model_data[key].sel(t=t),
                diag_data[key].sel(t=t),
                f"{key.replace('.', '_')}" + filetype,
                label=key,
                figheader=figheader,
                ylabel="Temperature [eV]",
                xlabel="Channel",
            )
        key = "TS.NE"
        if key in model_data.keys():
            _plot_1d(
                model_data[key].sel(t=t),
                diag_data[key].sel(t=t),
                f"{key.replace('.', '_')}" + filetype,
                label=key,
                figheader=figheader,
                ylabel="Density [m^-3]",
                xlabel="Channel",
            )

        key = "TE"
        plot_profile(
            profiles[key].sel(t=t),
            phantom_profiles[key].sel(t=t),
            key,
            figheader=figheader,
            filetype=filetype,
            sharefig=True,
            color="blue",
            linestyle="dashdot",
        )
        key = "TI"
        plot_profile(
            profiles[key].sel(
                t=t,
            ),
            phantom_profiles[key].sel(
                t=t,
            ),
            key,
            figheader=figheader,
            filename="temperature",
            filetype=filetype,
            color="red",
            linestyle="dotted",
        )

        key = "NE"
        plot_profile(
            profiles[key].sel(t=t),
            phantom_profiles[key].sel(t=t),
            key,
            figheader=figheader,
            filetype=filetype,
            color="blue",
            sharefig=True,
        )
        key = "NI"
        plot_profile(
            profiles[key].sel(t=t, element=element[0]),
            phantom_profiles[key].sel(t=t, element=element[0]),
            key,
            figheader=figheader,
            filetype=filetype,
            sharefig=True,
            color="red",
        )
        key = "NFAST"
        plot_profile(
            profiles[key].sel(t=t),
            phantom_profiles[key].sel(t=t),
            key,
            figheader=figheader,
            filename="densities",
            filetype=filetype,
            color="green",
        )

        key = "NI"
        for elem in element[1:]:
            plot_profile(
                profiles[key].sel(t=t, element=elem),
                phantom_profiles[key].sel(t=t, element=elem),
                key,
                figheader=figheader,
                filename=f"{elem} density",
                filetype=filetype,
                color="red",
            )

        key = "NNEUTR"
        plot_profile(
            profiles[key].sel(t=t),
            phantom_profiles[key].sel(t=t),
            key,
            filename="neutral density",
            figheader=figheader,
            filetype=filetype,
            logscale=True,
        )

        post_sample_filtered = post_sample[t_idx,][
            ~np.isnan(
                post_sample[
                    t_idx,
                ]
            ).any(axis=1)
        ]
        corner.corner(post_sample_filtered, labels=param_names)
        plt.savefig(figheader + "posterior" + filetype)

        corner.corner(
            prior_sample[
                t_idx,
            ],
            labels=param_names,
        )
        plt.savefig(figheader + "prior" + filetype)
        plt.close("all")


if __name__ == "__main__":
    filehead = "./results/example/"
    plot_bayes_result(filepath=filehead, filetype=".png")
