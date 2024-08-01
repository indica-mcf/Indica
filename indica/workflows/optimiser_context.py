from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from dime_sampler import DIMEMove
import emcee
import numpy as np
import pandas as pd
from scipy.stats import describe
import xarray as xr

from indica.workflows.priors import ln_prior
from indica.workflows.priors import PriorManager
from indica.workflows.priors import sample_from_priors


def sample_with_moments(
    sampler,
    start_points,
    iterations,
    n_params,
    auto_sample=10,
    stopping_factor=0.01,
    tune=False,
    debug=False,
):
    # TODO: Compare old_chain to new_chain
    #  if moments are different then keep going / convergence diagnostics here

    autocorr = np.ones(shape=(iterations, n_params)) * np.nan
    old_mean = np.inf
    success_flag = False  # requires succeeding check twice in a row
    for sample in sampler.sample(
        start_points,
        iterations=iterations,
        progress=True,
        skip_initial_state_check=False,
        tune=tune,
    ):
        if sampler.iteration % auto_sample:
            continue
        new_tau = sampler.get_autocorr_time(tol=0)
        autocorr[sampler.iteration - 1] = new_tau

        dist_stats = describe(sampler.get_chain(flat=True))

        new_mean = dist_stats.mean

        dmean = np.abs(new_mean - old_mean)
        rel_dmean = dmean / old_mean

        if debug:
            print("")
            print(f"rel_dmean: {rel_dmean.max()}")
        if rel_dmean.max() < stopping_factor:
            if success_flag:
                break
            else:
                success_flag = True
        else:
            success_flag = False
        old_mean = new_mean

    autocorr = autocorr[
        : sampler.iteration,
    ]
    return autocorr


def sample_with_autocorr(
    sampler,
    start_points,
    iterations,
    n_params,
    auto_sample=5,
):
    autocorr = np.ones(shape=(iterations, n_params)) * np.nan
    old_tau = np.inf
    for sample in sampler.sample(
        start_points,
        iterations=iterations,
        progress=True,
        skip_initial_state_check=False,
    ):
        if sampler.iteration % auto_sample:
            continue
        new_tau = sampler.get_autocorr_time(tol=0)
        autocorr[
            sampler.iteration - 1,
        ] = new_tau
        converged = np.all(new_tau * 50 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - new_tau) / new_tau < 0.01)
        if converged:
            break
        old_tau = new_tau
    autocorr = autocorr[
        : sampler.iteration,
    ]
    return autocorr


def gelman_rubin(chain):
    ssq = np.var(chain, axis=1, ddof=1)
    w = np.mean(ssq, axis=0)
    theta_b = np.mean(chain, axis=1)
    theta_bb = np.mean(theta_b, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((theta_bb - theta_b) ** 2, axis=0)
    var_theta = (n - 1) / n * w + 1 / n * B
    R = np.sqrt(var_theta / w)
    return R


@dataclass
class OptimiserEmceeSettings:
    param_names: list
    iterations: int = 1000
    nwalkers: int = 50
    burn_frac: float = 0.20
    sample_method: str = "random"
    starting_samples: int = 100
    stopping_criteria: str = "mode"
    stopping_criteria_factor: float = 0.01
    stopping_criteria_sample: int = 10
    stopping_criteria_debug: bool = False


class OptimiserContext(ABC):
    @abstractmethod
    def init_optimiser(self, *args, **kwargs):
        self.optimiser = None

    @abstractmethod
    def sample_start_points(self, *args, **kwargs):
        self.start_points = None

    @abstractmethod
    def format_results(self):
        self.results = {}

    @abstractmethod
    def run(self):
        results = None
        return results


class EmceeOptimiser(OptimiserContext):
    def __init__(
        self,
        optimiser_settings: OptimiserEmceeSettings,
        prior_manager: PriorManager,
        model_kwargs=None,
    ):
        self.optimiser_settings = optimiser_settings
        self.prior_manager = prior_manager
        self.model_kwargs = model_kwargs
        self.ndim = len(self.optimiser_settings.param_names)
        self.move = [
            (DIMEMove(aimh_prob=0.2), 1.0)  # differential independence mixture ensemble
            # (emcee.moves.StretchMove(), 1.0),
            # (emcee.moves.DEMove(), 0.1),
            # (emcee.moves.DEMove(), 0.9 * 0.9),
            # (emcee.moves.DESnookerMove(), 0.1),
            # (emcee.moves.DESnookerMove(gammas=1.0), 0.9 * 0.1)
        ]

    def init_optimiser(
        self,
        blackbox_func: Callable,
    ):  # type: ignore

        self.optimiser = emcee.EnsembleSampler(
            self.optimiser_settings.nwalkers,
            self.ndim,
            log_prob_fn=blackbox_func,
            kwargs=self.model_kwargs,
            parameter_names=self.optimiser_settings.param_names,
            moves=self.move,
        )

    def sample_start_points(
        self,
    ):

        if self.optimiser_settings.sample_method == "high_density":
            self.start_points = sample_from_high_density_region(
                param_names=self.optimiser_settings.param_names,
                priors=self.prior_manager.priors,
                optimiser=self.optimiser,
                nwalkers=self.optimiser_settings.nwalkers,
                nsamples=self.optimiser_settings.starting_samples,
            )
        elif self.optimiser_settings.sample_method == "random":
            self.start_points = sample_from_priors(
                param_names=self.optimiser_settings.param_names,
                priors=self.prior_manager.priors,
                size=self.optimiser_settings.nwalkers,
            )
        else:
            raise ValueError(
                f"Sample method: {self.optimiser_settings.sample_method} "
                f"not recognised, Defaulting to random sampling"
            )

    def run(
        self,
    ):

        if self.optimiser_settings.stopping_criteria == "mode":
            self.autocorr = sample_with_moments(
                self.optimiser,
                self.start_points,
                self.optimiser_settings.iterations,
                self.optimiser_settings.param_names.__len__(),
                auto_sample=self.optimiser_settings.stopping_criteria_sample,
                stopping_factor=self.optimiser_settings.stopping_criteria_factor,
                debug=self.optimiser_settings.stopping_criteria_debug,
            )
        else:
            raise ValueError(
                f"Stopping criteria: "
                f"{self.optimiser_settings.stopping_criteria} invalid"
            )

        optimiser_results = self.format_results()
        return optimiser_results

    def format_results(self):
        results = {}
        _blobs = self.optimiser.get_blobs(
            discard=int(self.optimiser.iteration * self.optimiser_settings.burn_frac),
            flat=True,
        )
        blobs = [blob for blob in _blobs if blob]  # remove empty blobs

        blob_names = blobs[0].keys()
        samples = np.arange(0, blobs.__len__())

        results["blobs"] = {
            blob_name: xr.concat(
                [data[blob_name] for data in blobs],
                dim=pd.Index(samples, name="sample_idx"),
            )
            for blob_name in blob_names
        }
        results["accept_frac"] = self.optimiser.acceptance_fraction.sum()
        results["prior_sample"] = sample_from_priors(
            self.optimiser_settings.param_names,
            self.prior_manager.priors,
            size=int(1e3),
        )

        post_sample = self.optimiser.get_chain(
            discard=int(self.optimiser.iteration * self.optimiser_settings.burn_frac),
            flat=True,
        )
        # pad index dim with maximum number of iterations
        max_iter = self.optimiser_settings.iterations * self.optimiser_settings.nwalkers
        npad = (
            (
                0,
                int(
                    max_iter * (1 - self.optimiser_settings.burn_frac)
                    - post_sample.shape[0]
                ),
            ),
            (0, 0),
        )
        results["post_sample"] = np.pad(post_sample, npad, constant_values=np.nan)

        results["auto_corr"] = self.autocorr
        return results


def sample_from_high_density_region(
    param_names: list, priors: dict, optimiser, nwalkers: int, nsamples=100
):
    num_best_points = 2
    index_best_start = []

    while index_best_start.__len__() < num_best_points:
        start_points = sample_from_priors(param_names, priors, size=nsamples)
        ln_prob, _ = optimiser.compute_log_prob(start_points)
        good_indices = np.argsort(ln_prob)[-num_best_points:]
        index_best_start.extend(good_indices)

    index_best_start[:num_best_points] = index_best_start[:num_best_points]
    best_start_points = start_points[index_best_start, :]
    # Passing samples through ln_prior and redrawing if they fail
    samples = np.empty((param_names.__len__(), 0))
    while samples.size < param_names.__len__() * nwalkers:
        sample = np.random.uniform(
            np.min(best_start_points, axis=0),
            np.max(best_start_points, axis=0),
            size=(nwalkers * 2, len(param_names)),
        )
        start = {name: sample[:, idx] for idx, name in enumerate(param_names)}
        _ln_prior = ln_prior(
            priors,
            start,
        )
        # Convert from dictionary of arrays -> array,
        # then filtering out where ln_prior is -infinity
        accepted_samples = np.array(list(start.values()))[:, _ln_prior != -np.inf]
        samples = np.append(samples, accepted_samples, axis=1)
    start_points = samples[:, 0:nwalkers].transpose()
    return start_points
