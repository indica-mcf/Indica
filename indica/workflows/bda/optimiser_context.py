from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from functools import partial
import logging
from operator import itemgetter

from dime_sampler import DIMEMove
import emcee
import numpy as np
import pandas as pd
from scipy.stats import describe
from scipy.stats import gaussian_kde
import skopt
import xarray as xr

from indica.workflows.bda.priors import ln_prior
from indica.workflows.bda.priors import PriorManager
from indica.workflows.bda.priors import sample_best_half
from indica.workflows.bda.priors import sample_from_priors
import mpmath as mp

exp = np.frompyfunc(mp.exp, 1, 1)
to_float = np.frompyfunc(float, 1, 1)


def exp_wrapper(x: np.ndarray):
    x_star = x * mp.mpf(1)
    y_star = exp(x_star)
    # y = to_float(y_star)
    return np.hstack(y_star[:])


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


def bo_wrapper(params: list, dims: list = None, blackbox: callable = None, **kwargs):
    if dims is None:
        raise ValueError("BO wrapper is missing dims")
    if blackbox is None:
        raise ValueError("BO wrapper is missing blackbox")
    params = {dim: params[idx] for idx, dim in enumerate(dims)}
    ln_post, blobs = blackbox({**params}, **kwargs)
    return -ln_post  # probability -> cost function


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


class OptimiserContext(ABC):
    def __init__(self):
        self.start_points = None
        self.optimiser = None

    @abstractmethod
    def init_optimiser(self, *args, **kwargs):
        self.optimiser = None

    @abstractmethod
    def sample_start_points(self, *args, **kwargs):
        self.start_points = None

    @abstractmethod
    def run(self):
        results = None
        return results

    @abstractmethod
    def post_process_results(self):
        results = {}
        return results

    @abstractmethod
    def reset_optimiser(self):
        self.optimiser = None
        self.start_points = None


@dataclass
class EmceeSettings:
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
    move: list = field(default_factory=lambda: [(DIMEMove(aimh_prob=0.2), 1.0)])


class EmceeOptimiser(OptimiserContext):
    def __init__(
        self,
        optimiser_settings: EmceeSettings,
        prior_manager: PriorManager,
        model_kwargs=None,
    ):
        super().__init__()
        self.autocorr = None
        self.optimiser_settings = optimiser_settings
        self.prior_manager = prior_manager
        self.model_kwargs = model_kwargs
        self.ndim = len(self.optimiser_settings.param_names)

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
            moves=self.optimiser_settings.move,
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
        return

    def post_process_results(self):
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

    def reset_optimiser(self):
        self.optimiser.reset()


@dataclass
class BOSettings:
    param_names: list
    n_calls: int = 30
    acq_func: str = "EI"
    xi: float = 0.01
    kappa: float = 1.96
    n_initial_points: int = 10
    noise: float = 1e-10
    initial_point_generator: str = "lhs"
    use_previous_best: bool = True
    boundary_samples: int = int(1e3)
    model_samples: int = 50
    posterior_samples: int = int(1e5)


class BOOptimiser(OptimiserContext):
    def __init__(
        self,
        optimiser_settings: BOSettings,
        prior_manager: PriorManager,
        model_kwargs: dict = None,
    ):

        super().__init__()
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.wrapped_blackbox_func = None
        self.blackbox_func = None
        self.result = None
        self.optimiser_settings = optimiser_settings
        self.prior_manager = prior_manager
        self.ndim = len(self.optimiser_settings.param_names)

        param_samples = sample_from_priors(
            self.optimiser_settings.param_names,
            self.prior_manager.priors,
            size=self.optimiser_settings.boundary_samples,
        )
        self.bounds = [
            (param_samples[:, idx].min(), param_samples[:, idx].max())
            for idx, _ in enumerate(self.optimiser_settings.param_names)
        ]

    def init_optimiser(
        self,
        blackbox_func: Callable,
    ):  # type: ignore
        self.optimiser = skopt.gp_minimize
        self.blackbox_func = partial(blackbox_func, **self.model_kwargs)
        self.wrapped_blackbox_func = partial(
            bo_wrapper,
            dims=self.optimiser_settings.param_names,
            blackbox=blackbox_func,
            **self.model_kwargs,
        )

    def reset_optimiser(self):
        self.optimiser = skopt.gp_minimize
        self.start_points = None

    def sample_start_points(self, *args, **kwargs):

        if self.optimiser_settings.use_previous_best and self.result is not None:
            best_indices = np.argsort(self.result.func_vals)[
                : self.optimiser_settings.n_initial_points
            ]
            self.start_points = list(
                itemgetter(*best_indices)(
                    self.result.x_iters,
                )
            )
        else:
            start_points = sample_best_half(
                self.optimiser_settings.param_names,
                self.prior_manager.priors,
                self.wrapped_blackbox_func,
                self.optimiser_settings.n_initial_points,
            )
            self.start_points = start_points.tolist()

    def run(
        self,
    ):
        self.result = self.optimiser(
            self.wrapped_blackbox_func,
            self.bounds,
            acq_func=self.optimiser_settings.acq_func,
            x0=self.start_points,
            xi=self.optimiser_settings.xi,
            kappa=self.optimiser_settings.kappa,
            n_calls=self.optimiser_settings.n_calls,
            n_initial_points=0,
            noise=self.optimiser_settings.noise,
            acq_optimizer="lbfgs",
            initial_point_generator=self.optimiser_settings.initial_point_generator,
        )

    def post_process_results(
        self,
    ):
        results = {"gp_regression": self.result}
        model = self.result.models[-1]

        log = logging.getLogger()
        log.info("kde posterior estimating")

        real_samples = np.array(
            self.result.space.rvs(n_samples=self.optimiser_settings.posterior_samples)
        )
        normed_samples = self.result.space.transform(real_samples)

        obj_func = model.predict(normed_samples)
        posterior = exp_wrapper(
            -obj_func
        )  # log of objective function -> posterior probability
        posterior /= posterior.max()
        posterior = to_float(posterior)
        posterior[posterior == 0] = 1e-100

        posterior_fit = gaussian_kde(real_samples.T, weights=posterior)

        params = posterior_fit.resample(size=self.optimiser_settings.model_samples)

        blobs = []
        for model_sample_idx in range(self.optimiser_settings.model_samples):
            _params = {
                param_name: params[name_idx, model_sample_idx]
                for name_idx, param_name in enumerate(
                    self.optimiser_settings.param_names
                )
            }

            post, _blobs = self.blackbox_func(_params)
            if _blobs:
                blobs.append(_blobs)

        blob_names = blobs[0].keys()
        samples = np.arange(0, blobs.__len__())

        results["blobs"] = {
            blob_name: xr.concat(
                [data[blob_name] for data in blobs],
                dim=pd.Index(samples, name="sample_idx"),
            )
            for blob_name in blob_names
        }
        results["accept_frac"] = None
        results["prior_sample"] = sample_from_priors(
            self.optimiser_settings.param_names,
            self.prior_manager.priors,
            size=int(1e3),
        )

        results["post_sample"] = posterior_fit.resample(size=int(2e3)).T
        results["auto_corr"] = np.zeros(
            shape=(10, len(self.optimiser_settings.param_names))
        )
        return results
