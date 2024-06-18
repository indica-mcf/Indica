from abc import abstractmethod, ABC
from enum import Enum
from typing import Callable

from scipy.stats import loguniform, uniform, gaussian_kde
import numpy as np


def get_uniform(lower, upper):
    # Less confusing parametrisation of scipy.stats uniform
    return uniform(loc=lower, scale=upper - lower)

# Move this to a config
DEFAULT_PRIORS = {
    "ion_temperature.y0": get_uniform(1000, 10000),
    "ion_temperature.y1": get_uniform(50, 50.1),
    "ion_temperature.wped": get_uniform(1, 10),
    "ion_temperature.wcenter": get_uniform(0.2, 0.4),
    "ion_temperature.peaking": get_uniform(1, 10),

    "electron_density.y0": get_uniform(2e19, 2e20),
    "electron_density.y1": get_uniform(1e18, 1e19),
    "electron_density.wped": loguniform(2, 20),
    "electron_density.wcenter": get_uniform(0.2, 0.4),
    "electron_density.peaking": get_uniform(1, 4),

    "electron_temperature.y0": get_uniform(1000, 5000),
    "electron_temperature.y1": get_uniform(50, 50.1),
    "electron_temperature.wped": get_uniform(1, 10),
    "electron_temperature.wcenter": get_uniform(0.2, 0.4),
    "electron_temperature.peaking": get_uniform(1, 4),

    "impurity_density:ar.y0": loguniform(1.01e16, 1e18),
    "impurity_density:ar.y1": loguniform(1e16, 1.01e16),
    "impurity_density:ar.wped": get_uniform(2, 6),
    "impurity_density:ar.wcenter": get_uniform(0.2, 0.4),
    "impurity_density:ar.peaking": get_uniform(1, 6),

    "neutral_density.y0": loguniform(1e13, 1e15),
    "neutral_density.y1": loguniform(1e13, 1e16),
    "neutral_density.wped": get_uniform(16, 17),
    "neutral_density.wcenter": get_uniform(0.2, 0.4),
    "neutral_density.peaking": get_uniform(1, 6),
}

DEFAULT_COND_PRIORS = {
    "electron_temperature.y0/electron_temperature.y1": lambda x1, x2: np.where(
        (x1 > x2 * 2), 1, 0
    ),
    "electron_density.y0/electron_density.y1": lambda x1, x2: np.where(
        (x1 > x2 * 2), 1, 0
    ),
    # "electron_density.y0/impurity_density:ar.y0": lambda x1, x2: np.where(
    #     (x1 > x2 * 100) & (x1 < x2 * 1e5), 1, 0
    # ),
    "impurity_density:ar.y0/impurity_density:ar.y1": lambda x1, x2: np.where(
        (x1 > x2), 1, 0
    ),
    # "impurity_density:ar.peaking/electron_density.peaking": lambda x1, x2: np.where(
    #     (x1 > x2), 1, 0
    # ),  # impurity always more peaked
    "ion_temperature.y0/ion_temperature.y1": lambda x1, x2: np.where(
        (x1 > x2 * 2), 1, 0
    ),
    # "ion_temperature.y0/electron_temperature.y0": lambda x1, x2: np.where(
    #     x1 > x2, 1, 0
    # ),  # hot ion mode

}


class PriorType(Enum):
    BASIC = 1
    COND = 2
    COMPOUND = 3


class Prior(ABC):
    def __init__(self,
                 prior_func: Callable = None,
                 labels: tuple = None,
                 type: PriorType = None,
                 ):
        self.prior_func = prior_func
        self.labels = labels  # to identify mapping between prior names and ndim prior funcs
        self.type = type

    @abstractmethod
    def pdf(self, value):
        return None

    @abstractmethod
    def rvs(self, size):
        return None


class PriorBasic(Prior):
    def __init__(self,
                 prior_func: Callable = None,
                 labels: tuple = None
                 ):
        super().__init__(prior_func=prior_func, labels=labels, type=PriorType.BASIC)

    def pdf(self, value):
        return self.prior_func.pdf(value)

    def rvs(self, size):
        return self.prior_func.rvs(size)


class PriorCond(Prior):
    def __init__(self,
                 prior_func: Callable = None,
                 labels: tuple = None,
                 ):
        super().__init__(prior_func=prior_func, labels=labels, type=PriorType.COND)

    def pdf(self, *values):
        return self.prior_func(*values)

    def rvs(self, size):
        return None


class PriorCompound(Prior):
    def __init__(self,
                 prior_func: gaussian_kde = None,
                 labels: tuple = None,
                 ):
        super().__init__(labels=labels, type=PriorType.COMPOUND)
        self.prior_func = prior_func

    def pdf(self, *values):
        return self.prior_func(np.array(values))

    def rvs(self, size):
        return self.prior_func.resample(size).T


class PriorManager:
    def __init__(self,
                 prior_funcs: dict = None,
                 cond_prior_funcs: dict = None,
                 ):

        if prior_funcs is None:
            prior_funcs = DEFAULT_PRIORS
        if cond_prior_funcs is None:
            cond_prior_funcs = DEFAULT_COND_PRIORS
        self.compound_prior_funcs = {}

        self.prior_funcs = prior_funcs
        self.cond_prior_funcs = cond_prior_funcs

        #  Create these somewhere else
        self.priors: dict = {}
        for name, prior in self.prior_funcs.items():
            self.priors[name] = PriorBasic(prior, labels=tuple([name]))
        for name, prior in self.cond_prior_funcs.items():
            self.priors[name] = PriorCond(prior, labels=tuple(name.split("/")))

    def update_priors(self, new_priors: dict):
        #  Remove old priors matching new_priors prefixes
        prior_prefixes_to_remove = list(set([key.split(".")[0] for key in new_priors.keys()]))
        priors_to_remove = [prior_name for prior_name in self.priors.keys() if
                            any(prior_name for prefix in prior_prefixes_to_remove if prefix in prior_name)]
        print(f"Discarding priors: {priors_to_remove}")
        print(f"Updating with {new_priors.keys()}")
        for prior_name in priors_to_remove:
            self.priors.pop(prior_name)
        self.priors.update(new_priors)

    def get_prior_names_for_profiles(self, profile_names: list) -> list:
        #  All priors that correspond to the profile_names given
        prior_names = [prior_name for prior_name, prior in self.priors.items() for profile_name in profile_names if
                       profile_name in str(prior.labels)]

        return prior_names

    def get_param_names_for_profiles(self, profile_names: list) -> list:
        #  All param names that correspond to the profile_names given
        param_names = [list(prior.labels) for prior_name, prior in self.priors.items() for profile_name in profile_names if
                       profile_name in str(prior.labels)]
        unpacked_names = [param_name for param_name_list in param_names for param_name in param_name_list]
        unique_param_names = list(set(unpacked_names))
        return unique_param_names

    def ln_prior(self, parameters: dict):
        # refactor ln_prior to be generalisable / define interface between cond and regular priors
        return ln_prior(priors=self.priors, parameters=parameters)


def ln_prior(priors: dict, parameters: dict):
    ln_prior = 0

    for prior_name, prior in priors.items():
        param_names_in_prior = [x for x in prior.labels if x in parameters.keys()]
        if param_names_in_prior.__len__() == 0:
            # if prior assigned but no parameter then skip
            continue
        # if not all relevant params given then ignore prior (important for conditional priors)
        if param_names_in_prior.__len__() != prior.labels.__len__():
            continue
        param_values = np.array([parameters[x] for x in param_names_in_prior])
        ln_prior += np.log(prior.pdf(*param_values))
    return ln_prior


def sample_from_priors(param_names: list, priors: dict, size=10):
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

        #  Due to looping over priors the samples need to be reordered to match param_names ordering
        new_sample = {param_name: _new_sample[param_name] for param_name in param_names}
        #  Throw out samples that don't meet conditional priors and redraw
        _ln_prior = ln_prior(priors, new_sample)
        # Convert from dictionary of arrays -> array,
        # then filtering out where ln_prior is -infinity
        accepted_samples = np.array(list(new_sample.values()))[:, _ln_prior != -np.inf]
        samples = np.append(samples, accepted_samples, axis=1)
    samples = samples[:, 0:size]
    return samples.transpose()


def sample_from_high_density_region(
        param_names: list, priors: dict, optimiser, nwalkers: int, nsamples=100
):
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


if __name__ == "__main__":
    pm = PriorManager()
    post = pm.ln_prior({"electron_density.y0": 1e20, "electron_density.y1": 1e19})
    samples = sample_from_priors(["electron_density.y0", "electron_density.y1"], pm.priors)

    print(post)
    print(samples)
