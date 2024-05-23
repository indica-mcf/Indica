from scipy.stats import loguniform, uniform
import numpy as np

def get_uniform(lower, upper):
    # Less confusing parametrisation of scipy.stats uniform
    return uniform(loc=lower, scale=upper - lower)


DEFAULT_PRIORS = {
    "ion_temperature.y0": get_uniform(1000, 10000),
    "ion_temperature.y1": get_uniform(0, 500),
    "ion_temperature.wped": get_uniform(1, 6),
    "ion_temperature.wcenter": get_uniform(0.2, 0.4),
    "ion_temperature.peaking": get_uniform(1, 6),

    "electron_density.y0": get_uniform(2e19, 4e20),
    "electron_density.y1": get_uniform(1e18, 2e19),
    "electron_density.wped": loguniform(2, 20),
    "electron_density.wcenter": get_uniform(0.2, 0.4),
    "electron_density.peaking": get_uniform(1, 4),

    "electron_temperature.y0": get_uniform(1000, 5000),
    "electron_temperature.y1": get_uniform(0, 500),
    "electron_temperature.wped": get_uniform(1, 6),
    "electron_temperature.wcenter": get_uniform(0.2, 0.4),
    "electron_temperature.peaking": get_uniform(1, 4),

    "impurity_density:ar.y0": loguniform(2e15, 1e18),
    "impurity_density:ar.y1": loguniform(1e14, 1e16),
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
    "electron_density.y0/electron_density.y1": lambda x1, x2: np.where(
        (x1 > x2 * 2), 1, 0
    ),
    "electron_density.y0/impurity_density:ar.y0": lambda x1, x2: np.where(
        (x1 > x2 * 100) & (x1 < x2 * 1e5), 1, 0
    ),
    "impurity_density:ar.y0/impurity_density:ar.y1": lambda x1, x2: np.where(
        (x1 > x2), 1, 0
    ),
    "impurity_density:ar.peaking/electron_density.peaking": lambda x1, x2: np.where(
        (x1 > x2), 1, 0
    ),  # impurity always more peaked
    # "ion_temperature.y0/electron_temperature.y0": lambda x1, x2: np.where(
    #     x1 > x2, 1, 0
    # ),  # hot ion mode

}


class PriorManager:
    def __init__(self,
                 prior_funcs: dict = None,
                 cond_prior_funcs: dict = None,
                 ):

        if prior_funcs is None:
            prior_funcs = DEFAULT_PRIORS
        if cond_prior_funcs is None:
            cond_prior_funcs = DEFAULT_COND_PRIORS

        self.cond_funcs = cond_prior_funcs
        self.prior_funcs = prior_funcs

    def update_priors(self):
        return

    def ln_prior(self, parameters: dict):
        # refactor ln_prior to be generalisable / define interface between cond and regular priors
        # Probably like: (tuple(param_names), lambda)
        return ln_prior(priors= {**self.prior_funcs, **self.cond_funcs}, parameters=parameters)



def ln_prior(priors: dict, parameters: dict):
    ln_prior = 0
    for prior_name, prior_func in priors.items():
        param_names_in_prior = [x for x in parameters.keys() if x in prior_name]
        if param_names_in_prior.__len__() == 0:
            # if prior assigned but no parameter then skip
            continue
        param_values = [parameters[x] for x in param_names_in_prior]
        if hasattr(prior_func, "pdf"):
            # for scipy.stats objects use pdf / for lambda functions just call
            ln_prior += np.log(prior_func.pdf(*param_values))
        else:
            # if lambda prior with 2+ args is defined when only 1 of
            # its parameters is given ignore it
            if prior_func.__code__.co_argcount != param_values.__len__():
                continue
            else:
                # Sorting to make sure args are given in the same order
                # as the prior_name string
                name_index = [
                    prior_name.find(param_name_in_prior)
                    for param_name_in_prior in param_names_in_prior
                ]
                sorted_name_index, sorted_param_values = (
                    list(x) for x in zip(*sorted(zip(name_index, param_values)))
                )
                ln_prior += np.log(prior_func(*sorted_param_values))
    return ln_prior


def sample_from_priors(param_names: list, priors: dict, size=10):

    #  Throw out samples that don't meet conditional priors and redraw
    samples = np.empty((param_names.__len__(), 0))
    while samples.size < param_names.__len__() * size:
        # Some mangling of dictionaries so _ln_prior works
        # Increase 'size * n' if too slow / looping too much
        new_sample = {name: priors[name].rvs(size=size * 2) for name in param_names}
        _ln_prior = ln_prior(priors, new_sample)
        # Convert from dictionary of arrays -> array,
        # then filtering out where ln_prior is -infinity
        accepted_samples = np.array(list(new_sample.values()))[:, _ln_prior != -np.inf]
        samples = np.append(samples, accepted_samples, axis=1)
    samples = samples[:, 0:size]
    return samples.transpose()

