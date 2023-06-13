import emcee
import numpy as np
import xarray as xr
import pandas as pd
import flatdict
from scipy.stats import loguniform
from pathlib import Path
import pickle

from indica.readers.read_st40 import ReadST40
from indica.bayesmodels import BayesModels, get_uniform
from indica.workflows.bayes_plots import plot_bayes_result
from indica.models.interferometry import Interferometry
from indica.models.helike_spectroscopy import Helike_spectroscopy
from indica.models.charge_exchange import ChargeExchange
from indica.models.equilibrium_reconstruction import EquilibriumReconstruction
from indica.models.plasma import Plasma
from indica.converters.line_of_sight import LineOfSightTransform

from indica.writers.bda_tree import create_nodes, write_nodes

# global configurations
DEFAULT_PHANTOM_PARAMS = {
    "Ne_prof.y0": 5e19,
    "Ne_prof.wcenter": 0.4,
    "Ne_prof.peaking": 2,
    "Ne_prof.y1": 2e18,
    "Ne_prof.yend": 1e18,
    "Ne_prof.wped": 2,
    "Nimp_prof.y0": 3e16,
    "Nimp_prof.y1": 0.5e16,
    "Nimp_prof.wcenter": 0.4,
    "Nimp_prof.wped": 6,
    "Nimp_prof.peaking": 2,
    "Te_prof.y0": 3000,
    "Te_prof.y1": 50,
    "Te_prof.wcenter": 0.4,
    "Te_prof.wped": 3,
    "Te_prof.peaking": 2,
    "Ti_prof.y0": 6000,
    "Ti_prof.y1": 50,
    "Ti_prof.wcenter": 0.4,
    "Ti_prof.wped": 3,
    "Ti_prof.peaking": 2,
}

DEFAULT_PRIORS = {
    "Ne_prof.y0": get_uniform(2e19, 4e20),
    "Ne_prof.y1": get_uniform(1e18, 1e19),
    "Ne_prof.y0/Ne_prof.y1": lambda x1, x2: np.where((x1 > x2 * 2), 1, 0),
    "Ne_prof.wped": get_uniform(1, 6),
    "Ne_prof.wcenter": get_uniform(0.1, 0.8),
    "Ne_prof.peaking": get_uniform(1, 6),
    "ar_conc": loguniform(0.0001, 0.01),
    "Nimp_prof.y0": get_uniform(1e16, 1e18),
    "Nimp_prof.y1": get_uniform(1e15, 2e16),
    "Ne_prof.y0/Nimp_prof.y0": lambda x1, x2: np.where(
        (x1 > x2 * 100) & (x1 < x2 * 1e5), 1, 0
    ),
    "Nimp_prof.y0/Nimp_prof.y1": lambda x1, x2: np.where((x1 > x2), 1, 0),
    "Nimp_prof.wped": get_uniform(1, 6),
    "Nimp_prof.wcenter": get_uniform(0.1, 0.8),
    "Nimp_prof.peaking": get_uniform(1, 20),
    "Nimp_prof.peaking/Ne_prof.peaking": lambda x1, x2: np.where(
        (x1 > x2), 1, 0
    ),  # impurity always more peaked
    "Te_prof.y0": get_uniform(1000, 5000),
    "Te_prof.wped": get_uniform(1, 6),
    "Te_prof.wcenter": get_uniform(0.1, 0.6),
    "Te_prof.peaking": get_uniform(1, 6),
    "Ti_prof.y0/Te_prof.y0": lambda x1, x2: np.where(x1 > x2, 1, 0),  # hot ion mode
    "Ti_prof.y0": get_uniform(3000, 10000),
    "Ti_prof.wped": get_uniform(1, 6),
    "Ti_prof.wcenter": get_uniform(0.1, 0.6),
    "Ti_prof.peaking": get_uniform(1, 20),
}

OPTIMISED_PARAMS = [
    # "Ne_prof.y1",
    "Ne_prof.y0",
    "Ne_prof.peaking",
    "Ne_prof.wcenter",
    "Ne_prof.wped",
    # "ar_conc",
    # "Nimp_prof.y1",
    "Nimp_prof.y0",
    # "Nimp_prof.wcenter",
    # "Nimp_prof.wped",
    # "Nimp_prof.peaking",
    "Te_prof.y0",
    # "Te_prof.wped",
    # "Te_prof.wcenter",
    # "Te_prof.peaking",
    "Ti_prof.y0",
    # "Ti_prof.wped",
    # "Ti_prof.wcenter",
    # "Ti_prof.peaking",
]

OPTIMISED_QUANTITY = ["xrcs.spectra", "cxff_pi.ti", "efit.wp", "smmh1.ne"]


class BayesWorkflow:
    def __init__(
        self,
        pulse=None,
        pulse_to_write = 23000101,
        result_path="./results/example/",
        nwalkers=50,
        tstart=0.02,
        tend=0.10,
        dt=0.01,
        tsample=0.06,
        iterations=100,
        burn_in_fraction=0,

        diagnostics=None,
        center_mass_sampling=True,
        Ti_ref=True,
        profiles=None,
        phantom=True,
        mds_write=False,
        save_plots=True,
    ):
        self.pulse = pulse
        self.pulse_to_write = pulse_to_write
        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.tsample = tsample
        self.result_path = result_path
        self.nwalkers = nwalkers
        self.iterations = iterations
        self.burn_in_fraction = burn_in_fraction
        self.center_mass_sampling = center_mass_sampling
        self.Ti_ref = Ti_ref

        self.diagnostics = diagnostics
        self.phantom = phantom
        self.profiles = profiles
        self.mds_write = mds_write
        self.save_plots = save_plots

        self.plasma = Plasma(
            tstart=tstart,
            tend=tend,
            dt=dt,
            main_ion="h",
            impurities=("ar", "c"),
            impurity_concentration=(
                0.001,
                0.04,
            ),
            full_run=False,
            n_rad=20,
        )
        self.plasma.Ti_ref = self.Ti_ref
        self.tsample = tsample
        self.plasma.time_to_calculate = self.plasma.t[
            np.abs(tsample - self.plasma.t).argmin()
        ]
        self.plasma.update_profiles(DEFAULT_PHANTOM_PARAMS)
        self.plasma.build_atomic_data(calc_power_loss=False)

        self.init_fast_particles()
        self.read_st40(diagnostics)
        self.init_models()

        if self.phantom:
            self.phantom_data()
            self.save_profiles()
        else:
            self.exp_data()

        self.bayes_run = BayesModels(
            plasma=self.plasma,
            data=self.flat_data,
            diagnostic_models=[*self.models.values()],
            quant_to_optimise=OPTIMISED_QUANTITY,
            priors=DEFAULT_PRIORS,
        )

        ndim = len(OPTIMISED_PARAMS)
        self.move = [(emcee.moves.StretchMove(), 0.9), (emcee.moves.DEMove(), 0.1)]
        self.sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn=self.bayes_run.ln_posterior,
            parameter_names=OPTIMISED_PARAMS,
            moves=self.move,
            kwargs={
                "moment_analysis": False,
                "calc_spectra": True,
                "minimum_lines": False,
                "background": self.flat_data["xrcs.background"],
            },
        )

        if not self.center_mass_sampling:
            self.start_points = self.bayes_run.sample_from_priors(
                OPTIMISED_PARAMS, size=self.nwalkers
            )
        else:
            # gaussian mass around most probable starting points TODO: move this function to it's own method
            nsamples = 100
            start_points = self.bayes_run.sample_from_priors(
                OPTIMISED_PARAMS, size=nsamples
            )
            ln_prob, _ = self.sampler.compute_log_prob(start_points)
            num_best_points = int(nsamples * 0.05)
            index_best_start = np.argsort(ln_prob)[-num_best_points:]
            best_start_points = start_points[index_best_start, :]
            best_points_std = np.std(best_start_points, axis=0)

            # Passing samples through ln_prior and redrawing if they fail
            samples = np.empty((OPTIMISED_PARAMS.__len__(), 0))
            while samples.size < OPTIMISED_PARAMS.__len__() * self.nwalkers:
                sample = np.random.normal(
                    np.mean(best_start_points, axis=0),
                    best_points_std * 2,
                    size=(self.nwalkers * 5, len(OPTIMISED_PARAMS)),
                )
                start = {
                    name: sample[:, idx] for idx, name in enumerate(OPTIMISED_PARAMS)
                }
                ln_prior = self.bayes_run._ln_prior(start)
                # Convert from dictionary of arrays -> array,
                # then filtering out where ln_prior is -infinity
                accepted_samples = np.array(list(start.values()))[
                    :, ln_prior != -np.inf
                ]
                samples = np.append(samples, accepted_samples, axis=1)
            self.start_points = samples[:, 0 : self.nwalkers].transpose()

    def init_fast_particles(self):
        st40_code = ReadST40(13110009, self.tstart, self.tend, dt=self.dt, tree="astra")
        st40_code.get_raw_data("", "astra", "RUN573")
        st40_code.bin_data_in_time(["astra"], self.tstart, self.tend, self.dt)
        code_data = st40_code.binned_data["astra"]
        Nf = (
            code_data["nf"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
            * 1.0e19
        )
        self.plasma.fast_density.values = Nf.values
        Nn = (
            code_data["nn"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
            * 1.0e19
        )
        self.plasma.neutral_density.values = Nn.values
        Pblon = code_data["pblon"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
        self.plasma.pressure_fast_parallel.values = Pblon.values
        Pbper = code_data["pbper"].interp(rho_poloidal=self.plasma.rho, t=self.plasma.t)
        self.plasma.pressure_fast_perpendicular.values = Pbper.values

    def read_st40(self, diagnostics=None):
        self.ST40_data = ReadST40(
            self.pulse, tstart=self.tstart, tend=self.tend, dt=self.dt
        )
        self.ST40_data(diagnostics)
        self.plasma.set_equilibrium(self.ST40_data.equilibrium)

    def save_profiles(self):
        phantom_profiles = {
            "electron_density": self.plasma.electron_density.sel(
                t=self.plasma.time_to_calculate
            ).copy(),
            "impurity_density": self.plasma.impurity_density.sel(
                t=self.plasma.time_to_calculate
            ).copy(),
            "electron_temperature": self.plasma.electron_temperature.sel(
                t=self.plasma.time_to_calculate
            ).copy(),
            "ion_temperature": self.plasma.ion_temperature.sel(
                t=self.plasma.time_to_calculate
            ).copy(),
            "fast_density": self.plasma.fast_density.sel(
                t=self.plasma.time_to_calculate
            ).copy(),
            "neutral_density": self.plasma.neutral_density.sel(
                t=self.plasma.time_to_calculate
            ).copy(),
        }
        self.profiles = phantom_profiles

    def init_models(self):
        self.models = {}
        for diag in self.diagnostics:
            if diag == "smmh1":
                # los_transform = self.ST40_data.binned_data["smmh1"]["ne"].transform
                machine_dims = self.plasma.machine_dimensions
                origin = np.array([[-0.38063365, 0.91893092, 0.01]])
                # end = np.array([[0,  0, 0.01]])
                direction = np.array([[0.38173721, -0.92387953, -0.02689453]])
                los_transform = LineOfSightTransform(
                    origin[:, 0],
                    origin[:, 1],
                    origin[:, 2],
                    direction[:, 0],
                    direction[:, 1],
                    direction[:, 2],
                    name="",
                    machine_dimensions=machine_dims,
                    passes=2,
                )
                los_transform.set_equilibrium(self.plasma.equilibrium)
                model = Interferometry(name="smmh1")
                model.set_los_transform(los_transform)
                model.plasma = self.plasma
                self.models["smmh1"] = model

            if "xrcs" in self.diagnostics:
                los_transform = self.ST40_data.binned_data["xrcs"]["te_kw"].transform
                model = Helike_spectroscopy(
                    name="xrcs",
                    window_masks=[slice(0.394, 0.396)],
                    window_vector=self.ST40_data.binned_data["xrcs"][
                        "spectra"
                    ].wavelength.values
                    * 0.1,
                )
                model.set_los_transform(los_transform)
                model.plasma = self.plasma
                self.models["xrcs"] = model

            if "efit" in self.diagnostics:
                model = EquilibriumReconstruction(name="efit")
                model.plasma = self.plasma
                self.models["efit"] = model

            if "cxff_pi" in self.diagnostics:
                transform = self.ST40_data.binned_data["cxff_pi"]["ti"].transform
                transform.set_equilibrium(self.ST40_data.equilibrium)
                model = ChargeExchange(name="cxff_pi", element="ar")
                model.set_transect_transform(transform)
                model.plasma = self.plasma
                self.models["cxff_pi"] = model

    def phantom_data(self, noise=False, noise_factor=0.1):
        self.flat_data = {}
        if "smmh1" in self.diagnostics:
            self.flat_data["smmh1.ne"] = (
                self.models["smmh1"]()
                .pop("ne")
                .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
        if "xrcs" in self.diagnostics:
            self.flat_data["xrcs.spectra"] = (
                self.models["xrcs"]()
                .pop("spectra")
                .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
            self.flat_data["xrcs.background"] = None
        if "cxff_pi" in self.diagnostics:
            cxrs_data = (
                self.models["cxff_pi"]()
                .pop("ti")
                .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )
            self.flat_data["cxff_pi.ti"] = cxrs_data.where(cxrs_data.channel == 2)
        if "efit" in self.diagnostics:
            self.flat_data["efit.wp"] = (
                self.models["efit"]()
                .pop("wp")
                .expand_dims(dim={"t": [self.plasma.time_to_calculate]})
            )

        if noise:
            self.flat_data["smmh1.ne"] = self.flat_data["smmh1.ne"] + self.flat_data[
                "smmh1.ne"
            ].max().values * np.random.normal(0, noise_factor, None)
            self.flat_data["xrcs.spectra"] = self.flat_data[
                "xrcs.spectra"
            ] + np.random.normal(
                0,
                np.sqrt(self.flat_data["xrcs.spectra"].values[0,]),
                self.flat_data["xrcs.spectra"].shape[1],
            )
            self.flat_data["cxff_pi.ti"] = self.flat_data[
                "cxff_pi.ti"
            ] + self.flat_data["cxff_pi.ti"].max().values * np.random.normal(
                0, noise_factor, self.flat_data["cxff_pi.ti"].shape[1]
            )
            self.flat_data["efit.wp"] = self.flat_data["efit.wp"] + self.flat_data[
                "efit.wp"
            ].max().values * np.random.normal(0, noise_factor, None)

    def exp_data(self):
        self.flat_data = flatdict.FlatDict(self.ST40_data.binned_data, ".")
        if "xrcs" in self.diagnostics:
            self.flat_data["xrcs.spectra"]["wavelength"] = (
                self.flat_data["xrcs.spectra"].wavelength * 0.1
            )
            background = self.flat_data["xrcs.spectra"].where(
                (self.flat_data["xrcs.spectra"].wavelength < 0.392)
                & (self.flat_data["xrcs.spectra"].wavelength > 0.388),
                drop=True,
            )
            self.flat_data["xrcs.background"] = background.mean(dim="wavelength")
            self.flat_data["xrcs.spectra"]["error"] = np.sqrt(
                self.flat_data["xrcs.spectra"] + background.std(dim="wavelength") ** 2
            )

        if "cxff_pi" in self.diagnostics:
            self.flat_data["cxff_pi"]["ti"] = self.flat_data["cxff_pi"]["ti"].where(
                self.flat_data["cxff_pi"]["ti"].channel == 2, drop=True
            )

    def __call__(self, *args, **kwargs):
        autocorr = sample_with_autocorr(
            self.sampler, self.start_points, iterations=self.iterations, auto_sample=5
        )
        blobs = self.sampler.get_blobs(
            discard=int(self.iterations * self.burn_in_fraction), flat=True
        )
        blob_names = self.sampler.get_blobs().flatten()[0].keys()
        blob_dict = {
            blob_name: xr.concat(
                [data[blob_name] for data in blobs],
                dim=pd.Index(np.arange(0, blobs.__len__()), name="index"),
            )
            for blob_name in blob_names
        }
        samples = self.sampler.get_chain(flat=True)

        prior_samples = self.bayes_run.sample_from_priors(
            OPTIMISED_PARAMS, size=int(1e4)
        )
        self.results = {
            "blobs": blob_dict,
            "diag_data": self.flat_data,
            "samples": samples,
            "prior_samples": prior_samples,
            "param_names": OPTIMISED_PARAMS,
            "phantom_profiles": self.profiles,
            "autocorr": autocorr,
            "acceptance_fraction": self.sampler.acceptance_fraction.sum(),
        }

        self.nested_results = {
            "DIAG_DATA":{"efit":{"wp":1}},
            # self.ST40_data.binned_data["efit"]["wp"].values[4,]
            # "MODEL_DATA": {},
            "METADATA": {
                            "DIAG_RUNS": "EMPTY",
                            "EQUIL": self.ST40_data.equilibrium.__str__(),
                            "IMP1": self.plasma.impurities[0],
                            "MAIN_ION": self.plasma.main_ion,
            },
            "OPTIMISATION": {
                            "AUTO_CORR":autocorr,
                            "BURN_FRAC":self.burn_in_fraction,
                            "ITER": self.iterations,
                            "NWALKERS": self.nwalkers,
                            "PARAM_NAMES": OPTIMISED_PARAMS,
                            "POST_SAMPLE": samples,
                            "PRIOR_SAMPLE": prior_samples
                            },
            # "PHANTOMS": self.profiles,
            # "PROFILES": {},
            # "TIME": self.plasma.t,
            # "TIME_OPT": self.plasma.time_to_calculate

        }

        # self.acceptance_fraction = self.sampler.acceptance_fraction.sum()
        print(self.results["acceptance_fraction"])

        Path(self.result_path).mkdir(parents=True, exist_ok=True)
        with open(self.result_path + "results.pkl", "wb") as handle:
            pickle.dump(self.results, handle)

        if self.save_plots:
            plot_bayes_result(self.results, figheader=self.result_path, filetype=".png")


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




if __name__ == "__main__":

    mds_write=False
    pulse_to_write = 23000101
    if mds_write:
        node_structure = create_nodes(pulse_to_write=pulse_to_write, diagnostic_quantities=OPTIMISED_QUANTITY)

    run = BayesWorkflow(
        pulse=10009,
        result_path="./results/test/",
        iterations=20,
        nwalkers=50,
        burn_in_fraction=0.10,
        dt=0.005,
        tsample=0.060,
        diagnostics=["xrcs", "efit", "smmh1", "cxff_pi"],
        phantom=False,
        center_mass_sampling=True,
        Ti_ref=True,
        mds_write=mds_write,
        save_plots=True
    )
    run()

    if mds_write:
        write_nodes(pulse_to_write, node_structure, run.nested_results)
