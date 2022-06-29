import numpy as np
from hda.diagnostics.crystal_spectrometer import CrystalSpectrometer
from indica.readers import ST40Reader
from MDSplus import Connection
import xarray as xr
import pandas as pd
from hda.examples.remap_diagnostics import remap_xrcs
from hda.profiles import Profiles

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hda.snippets.hda_profiles import HDA_spectra, func_chi
from hda.snippets.dispersion_characterisation import char_disp

from itertools import compress

import time as time
import json as json
import pickle
from copy import deepcopy


def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)


class BayesData:
    """
    For use in Bayesian optimisation algorithms
    Contains parameters to create plasma profiles/co-ordinates, profile builder, transition models, boundaries, priors
    History contains parameter search as well as model outputs/profiles
    """

    def __init__(self,
                 profiler=Profiles,
                 coords=None,
                 plasma={},
                 nuisance={},
                 sigmas={},
                 bounds={},
                 ):
        self.profiler = profiler
        self.plasma = plasma
        self.sigmas = sigmas
        self.bounds = bounds
        self.nuisance = nuisance

        self.coords = coords
        self.priors = {}  # unused
        self.profiles = {}
        for key, item in self.plasma.items():
            if key != "nuisance":
                self.profiles[key] = self.build_profile(item)

        # history object for interpreting after optimisation...
        self.history = {"model": [],  # model outputs
                        "likelihood": [],  # likelihood of parameters
                        "accepted_bool": [],  # if parameters were accepted
                        "profiles": [],  # profiles built from parameters
                        }

        # json.loads(json.dumps()) faster than deepcopy
        self.accepted = []  # accepted parameters
        self.rejected = []  # rejected parameters

    def build_profile(self, params):
        profile = self.profiler(xspl=self.coords)
        profile.y0 = params["y0"]
        profile.wcenter = params["wc"]
        profile.peaking = params["peak"]
        if "y1" in params:
            profile.y1 = params["y1"]
        profile.build_profile()
        return profile.yspl

    def update_profiles(self, plasma):
        profiles = dict()
        for key in plasma.keys():  # paths to changed params
            if key != "nuisance":
                profiles[key] = self.build_profile(plasma[key])
        profiles["NAr"] = profiles["Ne"]  # temp hack
        return profiles

    def transition_model(self, plasma, sigmas, bounds):
        plasma = json.loads(json.dumps(plasma))  # makes copy so original is unaffected
        for prof_key, sigma_dict in sigmas.items():
            for param_key, trans_func in sigma_dict.items():
                p = trans_func(plasma[prof_key][param_key])
                while (p < bounds[prof_key][param_key].min()) or (p > bounds[prof_key][param_key].max()):
                    p = trans_func(plasma[prof_key][param_key])
                plasma[prof_key][param_key] = p
        return plasma


def acceptance(likelihood_current, likelihood_new):
    if likelihood_new > likelihood_current:
        return True
    else:
        accept = np.random.uniform(0, 1)
    # Since we do a log likelihood, we need to exponentiate in order to compare to the random number
    return accept < (np.exp(likelihood_new - likelihood_current))


def metropolis_hastings(BayesClass, Model, iterations=10, verbose=True):
    """
    TODO: change parameter boundary to prior distribution
    Parameters
    ----------
    BayesClass
        Special class for handling interface of diagnostic models with optimisation functions
    Model
        models that produce likelihoods for given plasma parameters and diagnostic data
    iterations
        amount of times the model is called
    verbose
        if True prints additional information
    Returns
    -------
    """

    start = time.time()
    p_current = BayesClass.plasma
    prof_current = BayesClass.profiles
    likelihood_current, model_current = Model.likelihood(prof_current, p_current)

    BayesClass.history["model"].append(model_current)
    BayesClass.history["likelihood"].append(likelihood_current)
    BayesClass.history["profiles"].append(prof_current)

    for i in range(iterations):

        p_new = BayesClass.transition_model(p_current, BayesClass.sigmas, BayesClass.bounds)
        prof_new = BayesClass.update_profiles(p_new)
        likelihood_new, model_new = Model.likelihood(prof_new, p_new)

        BayesClass.history["model"].append(model_new)
        BayesClass.history["likelihood"].append(likelihood_new)
        BayesClass.history["profiles"].append(prof_new)

        if acceptance(likelihood_current, likelihood_new):
            p_current = p_new
            likelihood_current = likelihood_new
            BayesClass.accepted.append(json.loads(json.dumps(p_new)))
            BayesClass.history["accepted_bool"].append(True)

        else:
            BayesClass.rejected.append(json.loads(json.dumps(p_new)))
            BayesClass.history["accepted_bool"].append(False)

    end = time.time()
    if verbose:
        print(
            f"Metropolis Hastings Monte Carlo took {end - start} seconds / {(end - start) / iterations} seconds per iteration")
        print(
            f"Accepted percentage is {100 * np.sum(BayesClass.history['accepted_bool']) / iterations}% - optimal is ~33%")
    return


class CrystalModel:
    """
    Wrapper for the crystal spectrometer class and experimental data

    """

    def __init__(self,
                 settings={"strahl": False, "plot": False},
                 profiles={},
                 params={},
                 pulse=10014,
                 tsample=0.06,
                 tstart=0.02,
                 tend=0.10,
                 ):
        self.settings = settings
        self.profiles = profiles
        self.params = params
        self.spec, self.spectrum = self.get_data(pulse, tsample, tstart, tend)

        # default profiles
        xrcs = remap_xrcs(pulse, tstart, tend, plot=False)
        rho = xrcs["te_kw"].rho
        xspl = rho.sel(t=tsample, method="nearest")
        self.xspl = xspl.where(xspl < 1, drop=True)[::4]
        self.spline = self.spectrum.interp(wavelength=self.spectrum.wavelength[::2], method="cubic")

        with open("./../snippets/STRAHL_FA_2.pkl", "rb") as handle:  # strahl result
            fz = pickle.load(handle)

        fz = fz["6"]  # This doesn't go past 3.5 keV
        fz["el_temp"] = fz.el_temp.reduce(np.mean, dim="ion_charges")
        self.fz = fz.swap_dims(dict(rho_poloidal="el_temp")).drop("rho_poloidal")

        self.wavelength_ranges = {
            "bg_range": slice(0.394, 0.388),
            "w_range": slice(0.3952, 0.3945),
            "n3_range": slice(0.3958, 0.3954),
            "wn3_range": slice(0.3958, 0.3942)}

        self.bg_data = self.spectrum.sel(wavelength=self.wavelength_ranges["bg_range"])
        self.bg_mean = self.bg_data.mean(dim="wavelength")
        self.bg_std = self.bg_data.std()

    def get_data(self, pulse, tsample, tstart, tend):
        reader = ST40Reader(pulse, tstart, tend)
        spectra, dims = reader._get_data("sxr", "xrcs", ":intensity", 0)

        # Fix binning
        crude_spectra = spectra[0::2, ] + spectra[1::2, ]
        crude_time = (dims[0][0::2] + dims[0][1::2]) / 2
        adjusted_wavelength = char_disp(np.arange(1, 1031))

        result = xr.DataArray(data=crude_spectra, coords=(crude_time, adjusted_wavelength), dims=["t", "wavelength"])
        spectrum = result.sel(t=tsample, method="nearest")
        spectrum = spectrum.assign_coords(wavelength=(spectrum.wavelength * 0.1))
        spec = CrystalSpectrometer(window=spectrum.wavelength.values)

        return spec, spectrum

    def likelihood(self, profiles, params):

        """
        Parameters
        ----------
        profiles
            dictionary of kinetic profiles
        settings
            dictionary of options for running model

        Returns
        -------
        likelihood
            log likelihood of model
        model output
            dictionary of model outputs e.g. intensity/emissivity/FA
        """
        if self.settings["strahl"]:
            interp_fz = self.fz.interp(el_temp=profiles["Te"], kwargs={"fill_value": "extrapolate"})
        else:
            interp_fz = self.spec.fract_abu["ar"](profiles["Ne"], profiles["Te"], profiles["Nh"], tau=None)

        scaling = params["nuisance"]["int_scaling"]

        # Interpolate to the xrcs_los coordinate
        interp_fz = interp_fz.interp(rho_poloidal=self.xspl)
        interp_prof = {}
        for key in profiles:
            interp_prof[key] = profiles[key].interp(rho_poloidal=self.xspl)

        self.spec.intensity = self.spec.make_intensity(self.spec.database, el_temp=interp_prof["Te"],
                                                       el_dens=interp_prof["Ne"],
                                                       fract_abu=interp_fz, Ar_dens=interp_prof["NAr"],
                                                       H_dens=interp_prof["Nh"], int_cal=1e-28)
        self.spec.spectra = self.spec.make_spectra(self.spec.intensity, interp_prof["Ti"], background=0)

        emis = self.spec.spectra["total"].sum(["wavelength"])
        integral = self.spec.spectra["total"].sum(["xrcs_los_position"])
        integral_norm = integral / integral.max()
        intensity = (self.spline.max() * scaling - self.bg_mean) * integral_norm / integral_norm.max() + self.bg_mean

        data_slice = self.spectrum.sel(wavelength=self.wavelength_ranges["wn3_range"])
        model_slice = intensity.sel(wavelength=self.wavelength_ranges["wn3_range"])
        weights = np.sqrt(self.spectrum) + self.bg_std

        likelihood = np.sum(np.log(gaussian(model_slice, data_slice, weights))).values[()]

        if self.settings["plot"]:
            emis.plot()

            plt.figure()
            model_slice.plot()
            data_slice.plot()
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity (arb.)")
            plt.title(f"log likelihood:{likelihood}")
            plt.show(block=True)

        return likelihood, {"intensity": intensity, "fz": interp_fz, "emis": emis}


if __name__ == "__main__":

    # Initialise
    pulse = 10014
    tsample = 0.06

    # pulse = 9780
    # tsample = 0.06

    tstart = 0.02
    tend = 0.09
    #
    # conn = Connection('192.168.1.7:8000')
    # conn.openTree("ST40", pulsehda)
    # place_to_read = f"HDA.{run}:TIME"
    # t_hda = conn.get(place_to_read).data()
    # dt = t_hda[1] - t_hda[0]

    mh = BayesData(coords=np.linspace(0, 1, 15),
                   plasma={
                       "Ti": {"y0": 10e3,
                              "wc": 0.4,
                              "peak": 5,
                              },
                       "Te": {"y0": 3e3,
                              "wc": 0.35,
                              "peak": 2,
                              },
                       "Ne": {"y0": 1e19,
                              "y1": 1e18,
                              "wc": 0.35,
                              "peak": 1.5,
                              },
                       "NAr": {"y0": 1e19,
                               "y1": 1e18,
                               "wc": 0.35,
                               "peak": 1.5,
                               },
                       "Nh": {"y0": 1e10,
                              "wc": 0.35,
                              "peak": 1,
                              },
                       "nuisance": {"int_scaling": 1,

                                    },

                   },
                   sigmas={"Ti": {"y0": lambda x: np.random.normal(x, 1e2),
                                  "peak": lambda x: np.random.normal(x, 0.2),
                                  "wc": lambda x: np.random.normal(x, 0.01),
                                  },
                           "Te": {"y0": lambda x: np.random.normal(x, 1e2),
                                  "peak": lambda x: np.random.normal(x, 0.1),
                                  "wc": lambda x: np.random.normal(x, 0.025),
                                  },
                           "Ne": {
                               # "y0": lambda x: np.random.normal(x, 1e18),
                               "peak": lambda x: np.random.normal(x, 0.1),
                               "wc": lambda x: np.random.normal(x, 0.025),
                                },
                           "nuisance": {"int_scaling": lambda x: np.random.normal(x, 0.005),

                                        },
                           },
                   bounds={
                       "Ti": {"y0": np.array([9.5e3, 10.1e3]),
                              "peak": np.array([4, 20]),
                              "wc": np.array([0.35, 0.45]),
                              },
                       "Te": {"y0": np.array([1e3, 5e3]),
                              "peak": np.array([1, 4]),
                              "wc": np.array([0.2, 0.5]),
                              },
                       "Ne": {"y0": np.array([1e18, 1e20]),
                              "peak": np.array([1, 5]),
                              "wc": np.array([0.2, 0.5]),
                              },
                       "nuisance": {"int_scaling": np.array([0.90, 1.05]),

                                    },
                   }, )

    model = CrystalModel(pulse=pulse,
                         tsample=tsample,
                         tstart=tstart,
                         tend=tend,
                         settings={"plot": False,
                                   "strahl": False})

    metropolis_hastings(mh, model, iterations=100000)

    df = pd.json_normalize(mh.accepted)
    df_hist = df.loc[:, ["Ti.y0", "Ti.peak", "Ti.wc", "Te.y0", "Te.peak", "Te.wc",
                         "Ne.wc", "Ne.peak", "nuisance.int_scaling"]]
    df_hist.hist(bins=20)

    # ------------- plotting --------------

    #  plot model output
    profile_array = {}
    profile_stats = {}
    accepted_models = list(compress(mh.history["model"], mh.history["accepted_bool"]))
    model_list = [synth["intensity"] for synth in accepted_models]
    idx = np.where(mh.history["accepted_bool"])[0]
    model_array = xr.concat(model_list, pd.Index(idx, name="index"))
    profile_stats["model"] = {}
    profile_stats["model"]["mean"] = model_array.mean(dim="index")
    profile_stats["model"]["upper"] = model_array.quantile(0.975, dim="index")
    profile_stats["model"]["lower"] = model_array.quantile(0.025, dim="index")
    profile_stats["model"]["max"] = model_array.quantile(1.00, dim="index")
    profile_stats["model"]["min"] = model_array.quantile(0.00, dim="index")

    per_err = 0.05
    plt.figure(figsize=(6, 6))
    spectrum_err = (model.spectrum * per_err + model.bg_std)
    plt.errorbar(model.spectrum.wavelength, model.spectrum, spectrum_err, color="k", marker="*",
                 label=f"exp data", zorder=100, alpha=0.6)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.fill_between(profile_stats["model"]["mean"].wavelength, profile_stats["model"]["max"],
                     profile_stats["model"]["min"], zorder=1, color="lightgrey")
    plt.fill_between(profile_stats["model"]["mean"].wavelength, profile_stats["model"]["upper"],
                     profile_stats["model"]["lower"], zorder=2, color="red")

    best = Line2D([0], [0], color="red", label="Best runs")
    discarded = Line2D([0], [0], color="lightgrey", label="Discarded runs")
    handles.extend([best, discarded])
    plt.legend(handles=handles, loc="upper left")
    # plt.grid(True)
    plt.xlim([0.394, 0.401])
    plt.title(f"Crystal Spectrum {pulse}, time = {tsample * 1000}ms")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (AU)")

    # plot profile mean values + 95% confidence interval
    accepted_profiles = list(compress(mh.history["profiles"], mh.history["accepted_bool"]))

    for key in mh.history["profiles"][0]:
        profile_list = [param[key] for param in accepted_profiles]
        idx = np.where(mh.history["accepted_bool"])[0]
        profile_array[key] = xr.concat(profile_list, pd.Index(idx, name="index"))

        profile_stats[key] = {}
        profile_stats[key]["mean"] = profile_array[key].mean(dim="index")
        profile_stats[key]["upper"] = profile_array[key].quantile(0.975, dim="index")
        profile_stats[key]["lower"] = profile_array[key].quantile(0.025, dim="index")
        profile_stats[key]["max"] = profile_array[key].quantile(1.00, dim="index")
        profile_stats[key]["min"] = profile_array[key].quantile(0.00, dim="index")

    accepted_model = list(compress(mh.history["model"], mh.history["accepted_bool"]))

    profile_stats["emis"] = {}
    profile_list = [param["emis"] for param in accepted_model]
    profile_array["emis"] = xr.concat(profile_list, pd.Index(idx, name="index"))
    profile_stats["emis"]["mean"] = profile_array["emis"].mean(dim="index")
    profile_stats["emis"]["upper"] = profile_array["emis"].quantile(0.975, dim="index")
    profile_stats["emis"]["lower"] = profile_array["emis"].quantile(0.025, dim="index")
    profile_stats["emis"]["max"] = profile_array["emis"].quantile(1.00, dim="index")
    profile_stats["emis"]["min"] = profile_array["emis"].quantile(0.00, dim="index")

    profile_stats["fz"] = {}
    profile_list = [param["fz"] for param in accepted_model]
    profile_array["fz"] = xr.concat(profile_list, pd.Index(idx, name="index"))
    profile_stats["fz"]["mean"] = profile_array["fz"].mean(dim="index")
    profile_stats["fz"]["upper"] = profile_array["fz"].quantile(0.975, dim="index")
    profile_stats["fz"]["lower"] = profile_array["fz"].quantile(0.025, dim="index")
    profile_stats["fz"]["max"] = profile_array["fz"].quantile(1.00, dim="index")
    profile_stats["fz"]["min"] = profile_array["fz"].quantile(0.00, dim="index")

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 8))

    key = "Te"
    axs[0, 0].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["max"],
                           profile_stats[key]["min"], zorder=1, color="lightgrey")
    axs[0, 0].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["upper"],
                           profile_stats[key]["lower"], zorder=2, color="red")

    key = "Ti"
    axs[0, 1].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["max"],
                           profile_stats[key]["min"], zorder=1, color="lightgrey")
    axs[0, 1].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["upper"],
                           profile_stats[key]["lower"], zorder=2, color="red")

    key = "Ne"
    axs[1, 0].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["max"],
                           profile_stats[key]["min"], zorder=1, color="lightgrey")
    axs[1, 0].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["upper"],
                           profile_stats[key]["lower"], zorder=2, color="red")

    key = "NAr"
    axs[1, 1].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["max"],
                           profile_stats[key]["min"], zorder=1, color="lightgrey")
    axs[1, 1].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["upper"],
                           profile_stats[key]["lower"], zorder=2, color="red")

    key = "fz"
    axs[2, 0].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["max"][16,],
                           profile_stats[key]["min"][16,], zorder=1, color="lightgrey")
    axs[2, 0].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["upper"][16,],
                           profile_stats[key]["lower"][16,], zorder=2, color="red")

    key = "emis"
    axs[2, 1].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["max"],
                           profile_stats[key]["min"], zorder=1, color="lightgrey")
    axs[2, 1].fill_between(profile_stats[key]["mean"].rho_poloidal, profile_stats[key]["upper"],
                           profile_stats[key]["lower"], zorder=2, color="red")

    axs[0, 0].set(xlabel="rho_poloidal", ylabel="Electron Temperature (eV)")
    axs[0, 1].set(xlabel="rho_poloidal", ylabel="Ion Temperature (eV)")
    axs[1, 0].set(xlabel="rho_poloidal", ylabel="Electron Density (m^-3)")
    axs[1, 1].set(xlabel="rho_poloidal", ylabel="Argon Density (m^-3)")
    axs[2, 0].set(xlabel="rho_poloidal", ylabel="Fractional Abundance of Ar16+")
    axs[2, 1].set(xlabel="rho_poloidal", ylabel="Emissivity (W/m^3)")

    best = Line2D([0], [0], color="red", label="Best runs")
    discarded = Line2D([0], [0], color="lightgrey", label="Discarded runs")
    for axi in axs:
        for ax in axi:
            ax.legend(handles=[best, discarded], loc="upper left")

    # Save data

    profile_stats["data"] = model.spectrum
    profile_stats["data_err"] = spectrum_err
    profile_stats["raw_profiles"] = mh.history["profiles"]
    # profile_stats["raw_params"] = mh.history["profiles"]
    profile_stats["best"] = mh.history["accepted_bool"]

    with open("profile_stats_nuis.pkl", "wb") as handle:
        pickle.dump(profile_stats, handle)

    plt.show(block=True)
    print()
