from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np


def plot_bayes_result(
    figheader="./results/test/",
    blobs=None,
    diag_data=None,
    samples=None,
    prior_samples=None,
    param_names=None,
    phantom_profiles=None,
    plasma=None,
    autocorr=None,
    **kwargs
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
    plt.ylabel("tau")
    plt.savefig(figheader + "average_tau.png")

    if "xrcs.spectra" in blobs.keys():
        plt.figure()
        temp_data = blobs["xrcs.spectra"]
        plt.fill_between(
            temp_data.wavelength,
            temp_data.quantile(0.05, dim="index"),
            temp_data.quantile(0.95, dim="index"),
            label="XRCS spectrum, 90% Confidence",
            zorder=3,
            color="blue",
        )
        plt.fill_between(
            temp_data.wavelength,
            temp_data.quantile(0.01, dim="index"),
            temp_data.quantile(0.99, dim="index"),
            label="XRCS spectrum, 98% Confidence",
            zorder=2,
            color="grey",
        )
        plt.plot(
            diag_data["xrcs.spectra"].wavelength,
            diag_data["xrcs.spectra"].sel(t=plasma.time_to_calculate).values,
            linestyle="-",
            color="black",
            label="xrcs.spectra data",
            zorder=4,
        )
        plt.legend()
        plt.savefig(figheader + "xrcs_spectra.png")

    if "smmh1.ne" in blobs.keys():
        plt.figure()
        temp_data = blobs["smmh1.ne"]
        plt.xlabel("samples ()")
        plt.ylabel("ne_int (m^-2)")
        plt.plot(temp_data, label="smmh1.ne_int model")
        plt.axhline(
            y=diag_data["smmh1.ne"].sel(t=plasma.time_to_calculate).values,
            color="red",
            linestyle="-",
            label="smmh1.ne_int data",
        )
        plt.legend()
        plt.savefig(figheader + "smmh1_ne.png")

    if "xrcs.te_kw" in blobs.keys():
        plt.figure()
        temp_data = blobs["xrcs.te_kw"][:, 0, 0]
        plt.ylabel("temperature (eV)")
        plt.plot(temp_data, label="xrcs.te_kw model", color="blue")
        plt.axhline(
            y=diag_data["xrcs.te_kw"][
                0,
            ]
            .sel(t=plasma.time_to_calculate)
            .values,
            color="blue",
            linestyle="-",
            label="xrcs.te_kw data",
        )
        plt.legend()
        plt.savefig(figheader + "xrcs_te_kw.png")

    if "xrcs.ti_w" in blobs.keys():
        plt.figure()
        temp_data = blobs["xrcs.ti_w"][:, 0, 0]
        plt.plot(temp_data, label="xrcs.ti_w model", color="red")
        plt.axhline(
            y=diag_data["xrcs.ti_w"][
                0,
            ]
            .sel(t=plasma.time_to_calculate)
            .values,
            color="red",
            linestyle="-",
            label="xrcs.ti_w data",
        )
        plt.legend()
        plt.savefig(figheader + "xrcs_ti_w.png")

    plt.figure()
    prof = blobs["electron_density"]
    plt.fill_between(
        prof.rho_poloidal,
        prof.quantile(0.05, dim="index"),
        prof.quantile(0.95, dim="index"),
        label="Ne, 90% Confidence",
        zorder=3,
        color="blue",
    )
    plt.fill_between(
        prof.rho_poloidal,
        prof.quantile(0.01, dim="index"),
        prof.quantile(0.99, dim="index"),
        label="Ne, 98% Confidence",
        zorder=2,
        color="grey",
    )
    if phantom_profiles:
        phantom_profiles["electron_density"].plot(
            label="phantom_profile", linestyle="--", color="black", zorder=4
        )
    plt.legend()
    plt.savefig(figheader + "electron_density.png")

    plt.figure()
    prof = blobs["electron_temperature"]
    plt.fill_between(
        prof.rho_poloidal,
        prof.quantile(0.05, dim="index"),
        prof.quantile(0.95, dim="index"),
        label="Te, 90% Confidence",
        zorder=3,
        alpha=0.7,
        color="blue",
    )
    plt.fill_between(
        prof.rho_poloidal,
        prof.quantile(0.01, dim="index"),
        prof.quantile(0.99, dim="index"),
        label="Te, 98% Confidence",
        color="grey",
        zorder=2,
        alpha=0.7,
    )
    if phantom_profiles:
        phantom_profiles["electron_temperature"].plot(
            label="Te, phantom_profile", linestyle="--", color="black", zorder=4
        )

    prof = blobs["ion_temperature"].sel(element="ar")
    plt.fill_between(
        prof.rho_poloidal,
        prof.quantile(0.05, dim="index"),
        prof.quantile(0.95, dim="index"),
        label="Ti, 90% Confidence",
        zorder=3,
        alpha=0.7,
        color="red",
    )
    plt.fill_between(
        prof.rho_poloidal,
        prof.quantile(0.01, dim="index"),
        prof.quantile(
            0.99,
            dim="index",
        ),
        label="Ti, 98% Confidence",
        color="grey",
        zorder=2,
        alpha=0.7,
    )
    if phantom_profiles:
        phantom_profiles["ion_temperature"].plot(
            label="Ti, phantom_profile", linestyle="-.", color="black", zorder=4
        )
    plt.legend()
    plt.ylabel("temperature (eV)")
    plt.savefig(figheader + "temperature.png")

    plt.figure()
    prof = blobs["impurity_density"].sel(element="ar")
    plt.fill_between(
        prof.rho_poloidal,
        prof.quantile(0.05, dim="index"),
        prof.quantile(0.95, dim="index"),
        label="Nimp, 90% Confidence",
        zorder=3,
        color="red",
    )
    plt.fill_between(
        prof.rho_poloidal,
        prof.quantile(0.01, dim="index"),
        prof.quantile(
            0.99,
            dim="index",
        ),
        label="Nimp, 98% Confidence",
        color="grey",
    )
    if phantom_profiles:
        phantom_profiles["impurity_density"].plot(
            label="phantom_profile", linestyle="--", color="black", zorder=4
        )
    plt.legend()
    plt.savefig(figheader + "impurity_density.png")

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
