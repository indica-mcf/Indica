import numpy as np
from hda.diagnostics.crystal_spectrometer import CrystalSpectrometer
from indica.readers import ST40Reader
from MDSplus import Connection
import xarray as xr
import pandas as pd
from hda.examples.remap_diagnostics import remap_xrcs
from hda.profiles import Profiles

import matplotlib.pyplot as plt
from hda.snippets.hda_profiles import HDA_spectra, func_chi

import time as time
import json as json
from copy import deepcopy

# Initialise
pulse = 9780
pulsehda = 25009780
run = "RUN69"
t_sample = 0.080

tstart = 0.07
tend = 0.09

conn = Connection('192.168.1.7:8000')
conn.openTree("ST40", pulsehda)
place_to_read = f"HDA.{run}:TIME"
t_hda = conn.get(place_to_read).data()
dt = t_hda[1] - t_hda[0]

reader = ST40Reader(pulse, tstart, tend)
spectra, dims = reader._get_data("sxr", "xrcs", ":intensity", 0)

# Fix binning
crude_spectra = spectra[0::2, ] + spectra[1::2, ]
crude_time = (dims[0][0::2] + dims[0][1::2]) / 2
result = xr.DataArray(data=crude_spectra, coords=(crude_time, dims[1]), dims=["t", "wavelength"])
spectrum = result.sel(t=t_sample, method="nearest")
spectrum = spectrum.assign_coords(wavelength=(spectrum.wavelength * 0.1))

spec = Crystal_Spectrometer(window=spectrum.wavelength.values)
# default profiles
xrcs = remap_xrcs(pulse, tstart, tend, plot=False)
rho = xrcs["te_kw"].rho
xspl = rho.sel(t=t_sample, method="nearest")
xspl = xspl.where(xspl < 1, drop=True)[::4]

# background
bg_range = slice(0.394, 0.388)
w_range = slice(0.3952, 0.3945)
n3_range = slice(0.3958, 0.3954)
wn3_range = slice(0.3958, 0.3942)

bg_data = spectrum.sel(wavelength=bg_range)
background = bg_data.mean(dim="wavelength")
bg_std = bg_data.std()

spline = spectrum.interp(wavelength=spectrum.wavelength[::2], method="cubic")


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
                 coords=xspl):
        self.profiler = profiler
        self.plasma = {"Ti": {"y0": 8e3,
                              "wc": 0.35,
                              "peak": 1,
                              },
                       "Te": {"y0": 3e3,
                              "wc": 0.35,
                              "peak": 2,
                              },
                       "Ne": {"y0": 1e19,
                              "wc": 0.35,
                              "peak": 4,
                              },
                       "NAr": {"y0": 1e15,
                               "wc": 0.35,
                               "peak": 4,
                               },
                       "Nh": {"y0": 1e10,
                              "wc": 0.35,
                              "peak": 0.5,
                              }, }

        self.coords = coords
        # Random walk with defined step length
        self.sigmas = {"Ti": {"y0": lambda x: np.random.normal(x, 1e2),
                              "peak": lambda x: np.random.normal(x, 0.05),
                              # "wc": lambda x: np.random.normal(x, 0.05),
                              },
                       # "Te": {"y0": lambda x: np.random.normal(x, 0.5e2),
                       #        "peak": lambda x: np.random.normal(x, 0.05),
                              # "wc": lambda x: np.random.normal(x, 0.05),
                              # },
                       }

        self.bounds = {"Ti": {"y0": np.array([6e3, 12e3]),
                              "peak": np.array([1, 4]),
                              "wc": np.array([0.2, 0.4]),
                              },
                       "Te": {"y0": np.array([1e3, 5e3]),
                              "peak": np.array([1, 4]),
                              "wc": np.array([0.2, 0.4]),
                              },
                       }

        self.priors = {}
        self.profiles = {}
        for key, item in self.plasma.items():
            self.profiles[key] = self.build_profile(item)

        # json.loads(json.dumps()) faster than deepcopy
        self.accepted = [json.loads(json.dumps(self.plasma))]
        # {"plasma": json.loads(json.dumps(self.plasma)), "profiles": json.loads(json.dumps(self.profiles))}]
        self.rejected = []

    def build_profile(self, params):
        profile = self.profiler(xspl=self.coords)
        profile.y0 = params["y0"]
        profile.wcenter = params["wc"]
        profile.peaking = params["peak"]
        profile.build_profile()
        return profile.yspl

    def update_profiles(self, plasma, sigmas):
        profiles = dict()
        # TODO Use sigmas.keys() but combine changed and non changed profiles at the end
        for key in plasma.keys():  # paths to changed params
            profiles[key] = self.build_profile(plasma[key])
        return profiles

    def transition_model(self, plasma, sigmas, bounds):
        plasma = json.loads(json.dumps(plasma))  # make copy so original is unaffected
        for plasma_key, plasma_dict in sigmas.items():
            for param_key, param_item in plasma_dict.items():
                p = param_item(plasma[plasma_key][param_key])
                while (p < bounds[plasma_key][param_key].min()) or (p > bounds[plasma_key][param_key].max()):
                    p = param_item(plasma[plasma_key][param_key])
                plasma[plasma_key][param_key] = p
        return plasma


def acceptance(likelihood_current, likelihood_new):
    if likelihood_new > likelihood_current:
        return True
    else:
        accept = np.random.uniform(0, 1)
    # Since we do a log likelihood, we need to exponentiate in order to compare to the random number
    return accept < (np.exp(likelihood_new - likelihood_current))


def metropolis_hastings(BayesClass, diag_model, iterations=10, verbose=True):
    """
    TODO: Make this into a method of BayesClass / Include priors


    transition_model(x): a function that draws a sample from a symmetric distribution and returns it
    param_init: a starting sample
    iterations: number of accepted to generate time step in model/diagnostic data
    """

    start = time.time()
    accepted = []
    rejected = []
    p_current = BayesClass.plasma
    prof_current = BayesClass.profiles
    likelihood_current = diag_model(prof_current)

    for i in range(iterations):

        p_new = BayesClass.transition_model(p_current, BayesClass.sigmas, BayesClass.bounds)
        prof_new = BayesClass.update_profiles(p_new, BayesClass.sigmas)
        likelihood_new = diag_model(prof_new)

        if acceptance(likelihood_current, likelihood_new):
            p_current = p_new
            # prof_current = prof_new
            likelihood_current = likelihood_new
            accepted.append(prof_new)

            BayesClass.accepted.append(json.loads(json.dumps(p_new)))
            # , "profiles": json.loads(json.dumps(prof_new))})
        else:
            rejected.append(prof_new)
            BayesClass.rejected.append(json.loads(json.dumps(p_new)))
            # , "profiles": json.loads(json.dumps(prof_new))})

    end = time.time()
    if verbose:
        print(
            f"Metropolis Hastings Monte Carlo took {end - start} seconds / {(end - start) / iterations} seconds per iteration")
        print(f"Accepted percentage is {100 * len(accepted) / (len(rejected) + len(accepted))}% - optimal is ~33%")
    return BayesClass


def crystal_model(profiles):
    profiles["fz"] = spec.fract_abu["ar"](profiles["Ne"], profiles["Te"], profiles["Nh"], tau=None)
    spec.intensity = spec.make_intensity(spec.database_offset, el_temp=profiles["Te"], el_dens=profiles["Ne"],
                                         fract_abu=profiles["fz"], Ar_dens=profiles["NAr"],
                                         H_dens=profiles["Nh"], int_cal=1e-28)
    spec.spectra = spec.make_spectra(spec.intensity, profiles["Ti"], background=0)

    profiles["emis"] = spec.spectra["total"].sum(["wavelength"])
    profiles["integral"] = spec.spectra["total"].sum(["rho_poloidal"])
    profiles["integral_norm"] = profiles["integral"] / profiles["integral"].max()
    profiles["intensity"] = (spline.max() - background) * profiles["integral_norm"] / profiles[
        "integral_norm"].max() + background

    data_slice = spectrum.sel(wavelength=wn3_range)
    model_slice = profiles["intensity"].sel(wavelength=w_range)
    weights = np.sqrt(spectrum) + bg_std

    likelihood = np.sum(np.log(gaussian(model_slice, data_slice, weights)))

    if False:
        profiles["emis"].plot()
        plt.show(block=True)

    if False:
        model_slice.plot()
        data_slice.plot()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (arb.)")
        plt.title(f"log likelihood:{likelihood.values}")
        plt.show(block=True)

    # dummy model
    # y = profiles["Ti"].max().values
    # likelihood = gaussian(y, 9e3, 1e2)

    return likelihood


if __name__ == "__main__":
    mh = BayesData()
    mh = metropolis_hastings(mh, crystal_model, iterations=1000)

    df = pd.json_normalize(mh.accepted)
    df_hist = df.loc[:, ["Ti.y0", "Ti.peak", "Te.y0", "Te.peak"]]

    # df2 = pd.json_normalize(mh.rejected)
    # df2_hist = df2.loc[:, ["Ti.y0", "Ti.peak"]]

    df_hist.hist(bins=10)

    # df2_hist.hist(bins=20)

    plt.show(block=True)
    print()
