from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np


def plot_profile(
    profile,
    blobkey: str,
    figheader="./results/test/",
    phantom_profile=None,
    sharefig=False,
    filename="",
    linestyle="--",
    color="blue",
):
    if not plt.get_fignums():  # If no figure is open
        plt.figure(figsize=(8, 6))

    plt.fill_between(
        profile.rho_poloidal,
        profile.quantile(0.16, dim="index"),
        profile.quantile(0.84, dim="index"),
        label=f"{blobkey}, 68% Confidence",
        zorder=3,
        color=color,
        alpha=0.8,
    )
    plt.fill_between(
        profile.rho_poloidal,
        profile.quantile(0.025, dim="index"),
        profile.quantile(0.975, dim="index"),
        label=f"{blobkey}, 95% Confidence",
        zorder=2,
        color="grey",
        alpha=0.6,
    )
    plt.fill_between(
        profile.rho_poloidal,
        profile.quantile(0.00, dim="index"),
        profile.quantile(1.00, dim="index"),
        label=f"{blobkey}, Max-Min",
        zorder=1,
        color="lightgrey",
        alpha=0.6,
    )

    if phantom_profile is not None:
        if "element" in phantom_profile[blobkey].dims:
            phantom = phantom_profile[blobkey].sel(element="ar")
        else:
            phantom = phantom_profile[blobkey]
        phantom.plot(
            label=f"{blobkey}, phantom profile",
            linestyle=linestyle,
            color="black",
            zorder=4,
        )

    plt.legend()
    if sharefig:
        return

    if filename:
        plt.savefig(figheader + f"{filename}.png")
    else:
        plt.savefig(figheader + f"{blobkey}.png")
    plt.close("all")


def _plot_0d(
    blobs: dict,
    blobkey: str,
    diag_data: dict,
    filename: str,
    figheader="./results/test/",
    xlabel="samples ()",
    ylabel="a.u.",
    figsize=(6.4, 4.8),
    **kwargs,
):
    if blobkey not in blobs.keys():
        raise ValueError(f"{blobkey} not in blobs")
    plt.figure(figsize=figsize)
    blob_data = blobs[blobkey]
    plt.plot(blob_data, label=f"{blobkey} model")
    plt.axhline(
        y=diag_data[blobkey].sel(t=blob_data.t).values,
        color="black",
        linestyle="-",
        label=f"{blobkey} data",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(figheader + filename)
    plt.close()


def _plot_1d(
    blobs: dict,
    blobkey: str,
    diag_data: dict,
    filename: str,
    figheader="./results/test/",
    ylabel="a.u.",
    xlim = None,
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
    plt.ylabel(ylabel)
    plt.xlabel(dims[0])
    plt.xlim(xlim)
    plt.legend()
    plt.savefig(figheader + filename)
    plt.close()


def plot_bayes_result(
    figheader="./results/test/",
    blobs=None,
    diag_data=None,
    samples=None,
    prior_samples=None,
    param_names=None,
    phantom_profiles=None,
    autocorr=None,
    **kwargs,
):
    Path(figheader).mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(
        np.arange(0, autocorr.__len__())[np.isfinite(autocorr)],
        autocorr[np.isfinite(autocorr)],
        label="average tau",
    )
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("auto-correlation time (iterations)")
    plt.savefig(figheader + "average_tau.png")
    plt.close()

    key = "efit.wp"
    if key in blobs.keys():
        _plot_0d(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}.png",
            figheader=figheader,
            ylabel="Wp (J)",
        )
    key = "smmh1.ne"
    if key in blobs.keys():
        _plot_0d(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}.png",
            figheader=figheader,
            ylabel="ne_int (m^-2)",
        )
    key = "xrcs.te_kw"
    if key in blobs.keys():
        _plot_0d(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}.png",
            figheader=figheader,
            ylabel="temperature (eV)",
        )
    key = "xrcs.ti_w"
    if key in blobs.keys():
        _plot_0d(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}.png",
            figheader=figheader,
            ylabel="temperature (eV)",
        )
    key = "xrcs.spectra"
    if key in blobs.keys():
        _plot_1d(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}.png",
            figheader=figheader,
            ylabel="intensity (A.U.)",
            xlim = (0.394, 0.401),
            figsize=(12, 10),
        )
    key = "cxff_pi.ti"
    if key in blobs.keys():
        _plot_1d(
            blobs,
            key,
            diag_data,
            f"{key.replace('.', '_')}.png",
            figheader=figheader,
            ylabel="temperature (eV)",
        )

    key = "electron_temperature"
    plot_profile(
        blobs[key],
        key,
        figheader=figheader,
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
        phantom_profile=phantom_profiles,
        color="red",
        linestyle="dotted",
    )
    key = "electron_density"
    plot_profile(
        blobs[key], key, figheader=figheader, phantom_profile=phantom_profiles
    )
    key = "impurity_density"
    for elem in blobs[key].element.values:
        plot_profile(
            blobs[key].sel(element=elem),
            key,
            figheader=figheader,
            filename=f"{elem} density",
            phantom_profile=phantom_profiles,
            color="red",
        )
    key = "ion_density"
    plot_profile(
        blobs[key].sel(element="h"),
        key,
        figheader=figheader,
        filename=f"h density",
        phantom_profile=phantom_profiles,
        color="red",
    )
    key = "fast_density"
    plot_profile(
        blobs[key],
        key,
        figheader=figheader,
        phantom_profile=phantom_profiles,
        color="red",
    )
    key = "neutral_density"
    plot_profile(
        blobs[key],
        key,
        figheader=figheader,
        phantom_profile=phantom_profiles,
    )

    corner.corner(samples, labels=param_names)
    plt.savefig(figheader + "posterior.png")

    corner.corner(prior_samples, labels=param_names)
    plt.savefig(figheader + "prior.png")
    plt.close("all")


def sample_with_autocorr(sampler, start_points, iterations=10, auto_sample=5):
    autocorr = np.ones((iterations,)) * np.nan
    old_tau = np.inf
    for sample in sampler.sample(
        start_points,
        iterations=iterations,
        progress=True,
    ):
        if sampler.iteration % auto_sample:
            continue
        new_tau = sampler.get_autocorr_time(tol=0)
        autocorr[sampler.iteration - 1] = np.mean(new_tau)
        converged = np.all(new_tau * 50 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - new_tau) / new_tau < 0.01)
        if converged:
            break
        old_tau = new_tau
    autocorr = autocorr[: sampler.iteration]
    return autocorr
