from scipy.stats import loguniform, uniform
from indica.bayesblackbox import ln_prior
import numpy as np

def get_uniform(lower, upper):
    # Less confusing parameterisation of scipy.stats uniform
    return uniform(loc=lower, scale=upper - lower)


DEFAULT_PRIORS = {
    "electron_density.y0": get_uniform(2e19, 4e20),
    "electron_density.y1": get_uniform(1e18, 2e19),
    "electron_density.y0/electron_density.y1": lambda x1, x2: np.where(
        (x1 > x2 * 2), 1, 0
    ),
    "electron_density.wped": loguniform(2, 20),
    "electron_density.wcenter": get_uniform(0.2, 0.4),
    "electron_density.peaking": get_uniform(1, 4),
    "impurity_density:ar.y0": loguniform(2e15, 1e18),
    "impurity_density:ar.y1": loguniform(1e14, 1e16),
    "electron_density.y0/impurity_density:ar.y0": lambda x1, x2: np.where(
        (x1 > x2 * 100) & (x1 < x2 * 1e5), 1, 0
    ),
    "impurity_density:ar.y0/impurity_density:ar.y1": lambda x1, x2: np.where(
        (x1 > x2), 1, 0
    ),
    "impurity_density:ar.wped": get_uniform(2, 6),
    "impurity_density:ar.wcenter": get_uniform(0.2, 0.4),
    "impurity_density:ar.peaking": get_uniform(1, 6),
    "impurity_density:ar.peaking/electron_density.peaking": lambda x1, x2: np.where(
        (x1 > x2), 1, 0
    ),  # impurity always more peaked
    "electron_temperature.y0": get_uniform(1000, 5000),
    "electron_temperature.wped": get_uniform(1, 6),
    "electron_temperature.wcenter": get_uniform(0.2, 0.4),
    "electron_temperature.peaking": get_uniform(1, 4),
    # "ion_temperature.y0/electron_temperature.y0": lambda x1, x2: np.where(
    #     x1 > x2, 1, 0
    # ),  # hot ion mode
    "ion_temperature.y0": get_uniform(1000, 10000),
    "ion_temperature.wped": get_uniform(1, 6),
    "ion_temperature.wcenter": get_uniform(0.2, 0.4),
    "ion_temperature.peaking": get_uniform(1, 6),
    # TODO: add thermal neutral density
}


def sample_from_priors(param_names: list, priors: dict, size=10):

    #  Throw out samples that don't meet conditional priors and redraw
    samples = np.empty((param_names.__len__(), 0))
    while samples.size < param_names.__len__() * size:
        # Some mangling of dictionaries so _ln_prior works
        # Increase size * n if too slow / looping too much
        new_sample = {name: priors[name].rvs(size=size * 2) for name in param_names}
        _ln_prior = ln_prior(priors, new_sample)
        # Convert from dictionary of arrays -> array,
        # then filtering out where ln_prior is -infinity
        accepted_samples = np.array(list(new_sample.values()))[:, _ln_prior != -np.inf]
        samples = np.append(samples, accepted_samples, axis=1)
    samples = samples[:, 0:size]
    return samples.transpose()

