from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Callable

import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import loguniform
from scipy.stats import uniform


def greater_than(x1, x2):
    return np.where(x1 > x2, 1, 0)


def get_uniform(lower, upper):
    # Less confusing parametrisation of scipy.stats uniform
    return uniform(loc=lower, scale=upper - lower)


class PriorType(Enum):
    """
    PriorType differentiates between the below prior types.
    BASIC: 1D PDF
    COND: generic relationship between 2+ parameters of form: func(*parameters)->float
    COMPOUND: ND PDF
    """

    BASIC = 1
    COND = 2
    COMPOUND = 3


class Prior(ABC):
    def __init__(
        self,
        prior_func: Callable = None,
        labels: tuple = None,
        type: PriorType = None,
    ):
        self.prior_func = prior_func
        self.labels = (
            labels  # to identify mapping between prior names and ndim prior funcs
        )
        self.type = type

    @abstractmethod
    def pdf(self, value):
        return None

    @abstractmethod
    def rvs(self, size):
        return None


class PriorBasic(Prior):
    def __init__(self, prior_func: Callable = None, labels: tuple = None):
        super().__init__(prior_func=prior_func, labels=labels, type=PriorType.BASIC)

    def pdf(self, value):
        return self.prior_func.pdf(value)

    def rvs(self, size):
        return self.prior_func.rvs(size)


class PriorCond(Prior):
    """
    Generic relationship between 2 or more parameters
    """

    def __init__(
        self,
        prior_func: Callable = None,
        labels: tuple = None,
    ):
        super().__init__(prior_func=prior_func, labels=labels, type=PriorType.COND)

    def pdf(self, *values):
        return self.prior_func(*values)

    def rvs(self, size):
        return None


class PriorCompound(Prior):
    """
    ND Probability Distribution Function
    """

    def __init__(
        self,
        prior_func: gaussian_kde = None,
        labels: tuple = None,
    ):
        super().__init__(labels=labels, type=PriorType.COMPOUND)
        self.prior_func = prior_func

    def pdf(self, *values):
        _pdf = self.prior_func(np.array(values))
        if _pdf.size == 1:  # Should return scalar when evaluating one point
            _pdf = _pdf.item()
        return _pdf

    def rvs(self, size):
        return self.prior_func.resample(size).T


class PriorManager:
    def __init__(
        self,
        basic_prior_info: dict = None,
        cond_prior_info: dict = None,
    ):

        self.compound_prior_funcs = {}  # initialised later

        self.basic_prior_info = basic_prior_info
        self.cond_prior_info = cond_prior_info

        self.get_uniform = get_uniform
        self.loguniform = loguniform
        self.greater_than = greater_than

        # Initialise prior objects
        self.priors: dict = {}
        for name, prior in self.basic_prior_info.items():
            prior = getattr(self, prior[0])(*prior[1:])
            self.priors[name] = PriorBasic(prior, labels=tuple([name]))
        for name, prior in self.cond_prior_info.items():
            prior = getattr(self, prior)
            self.priors[name] = PriorCond(prior, labels=tuple(name.split("/")))

    def update_priors(self, new_priors: dict):
        #  update priors but remove all priors that match the profile names first

        prior_prefixes_to_remove = list(
            set([key.split(".")[0] for key in new_priors.keys()])
        )
        priors_to_remove = [
            prior_name
            for prior_name in self.priors.keys()
            if any(
                prior_name
                for prefix in prior_prefixes_to_remove
                if prefix in prior_name
            )
        ]
        print(f"Discarding priors: {priors_to_remove}")
        print(f"Updating with {new_priors.keys()}")
        for prior_name in priors_to_remove:
            self.priors.pop(prior_name)
        self.priors.update(new_priors)

    def get_prior_names_for_profiles(self, profile_names: list) -> list:
        #  Get all priors that correspond to the profile_names given
        prior_names = [
            prior_name
            for prior_name, prior in self.priors.items()
            for profile_name in profile_names
            if profile_name in str(prior.labels)
        ]

        return prior_names

    def get_param_names_for_profiles(self, profile_names: list) -> list:
        #  Get all parameters that correspond to the profile_names given
        param_names = [
            list(prior.labels)
            for prior_name, prior in self.priors.items()
            for profile_name in profile_names
            if profile_name in str(prior.labels)
        ]
        unpacked_names = [
            param_name
            for param_name_list in param_names
            for param_name in param_name_list
        ]
        unique_param_names = list(set(unpacked_names))
        return unique_param_names

    def ln_prior(self, parameters: dict):
        # refactor ln_prior to define interface between cond and regular priors
        return ln_prior(priors=self.priors, parameters=parameters)


def ln_prior(priors: dict, parameters: dict):
    ln_prior = 0

    for prior_name, prior in priors.items():
        param_names_in_prior = [x for x in prior.labels if x in parameters.keys()]
        if param_names_in_prior.__len__() == 0:
            # if prior assigned but no parameter then skip
            continue
        # if not all relevant params given then ignore prior
        if param_names_in_prior.__len__() != prior.labels.__len__():
            continue
        param_values = np.array([parameters[x] for x in param_names_in_prior])
        ln_prior += np.log(prior.pdf(*param_values))
    return ln_prior


def sample_from_priors(param_names: list, priors: dict, size=10) -> np.ndarray:
    samples = np.empty((param_names.__len__(), 0))
    while samples.size < param_names.__len__() * size:
        # Increase 'size * n' if too slow / looping too much
        _new_sample = {}
        for prior_name, prior in priors.items():
            if any([label not in param_names for label in prior.labels]):
                continue
            if prior.type is PriorType.BASIC:
                _new_sample[prior_name] = prior.rvs(size=size * 2)
            elif prior.type is PriorType.COMPOUND:
                tuple_sample = prior.rvs(size=size * 2)
                for idx, value in enumerate(prior.labels):
                    _new_sample[value] = tuple_sample[:, idx]
            elif prior.type is PriorType.COND:
                continue
            else:
                raise TypeError(f"{prior} is not a version of {PriorType}")

        #  Due to looping over priors the samples need to be reordered
        new_sample = {param_name: _new_sample[param_name] for param_name in param_names}
        #  Throw out samples that don't meet conditional priors and redraw
        _ln_prior = ln_prior(priors, new_sample)
        # Convert from dictionary of arrays -> array,
        # then filtering out where ln_prior is -infinity
        accepted_samples = np.array(list(new_sample.values()))[:, _ln_prior != -np.inf]
        samples = np.append(samples, accepted_samples, axis=1)
    samples = samples[:, 0:size]
    return samples.transpose()


def sample_best_half(
    param_names: list, priors: dict, wrappedblackbox: callable, size=10
) -> np.ndarray:
    start_points = sample_from_priors(param_names, priors, size=2 * size)
    ln_post = []
    for idx in range(start_points.shape[0]):
        ln_post.append(wrappedblackbox(start_points[idx, :]))
    index_best_half = np.argsort(ln_post)[:size]
    best_points = start_points[index_best_half, :]
    return best_points


def sample_from_high_density_region(
    param_names: list, priors: dict, optimiser, nwalkers: int, nsamples=100
) -> np.ndarray:
    # TODO: remove repeated code
    start_points = sample_from_priors(param_names, priors, size=nsamples)

    ln_prob, _ = optimiser.compute_log_prob(start_points)
    num_best_points = 3
    index_best_start = np.argsort(ln_prob)[-num_best_points:]
    best_start_points = start_points[index_best_start, :]
    best_points_std = np.std(best_start_points, axis=0)

    # Passing samples through ln_prior and redrawing if they fail
    samples = np.empty((param_names.__len__(), 0))
    while samples.size < param_names.__len__() * nwalkers:
        sample = np.random.normal(
            np.mean(best_start_points, axis=0),
            best_points_std,
            size=(nwalkers * 5, len(param_names)),
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
