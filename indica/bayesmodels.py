from copy import deepcopy
import warnings

import flatdict
import numpy as np
from scipy.stats import uniform

np.seterr(all="ignore")

warnings.simplefilter("ignore", category=FutureWarning)


def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)


def get_uniform(lower, upper):
    # Less confusing parameterisation of scipy.stats uniform
    return uniform(loc=lower, scale=upper - lower)


class BayesModels:
    """
    Object that is used with Plasma object to create ln_posterior

    Parameters
    ----------
    plasma
        Plasma object needed for the diagnostic model calls
    data
        processed diagnostic data of format [diagnostic].[quantity]
    quant_to_optimise
        quantity from data which will be optimised with bckc from diagnostic_models
    priors
        prior functions to apply to parameters e.g. scipy.stats.rv_continuous objects
    diagnostic_models
        model objects to be called by ln_posterior
    """

    def __init__(
        self,
        plasma=None,
        data: dict = {},
        quant_to_optimise: list = [],
        priors: dict = {},
        diagnostic_models: list = [],
    ):
        self.plasma = plasma
        self.data = data
        self.quant_to_optimise = quant_to_optimise
        self.diagnostic_models = diagnostic_models
        self.priors = priors

        for diag_model in self.diagnostic_models:
            diag_model.plasma = self.plasma

        missing_data = list(set(quant_to_optimise).difference(data.keys()))
        if missing_data:  # list of keys in quant_to_optimise but not data
            raise ValueError(f"{missing_data} not found in data given")

    def _build_bckc(self, params: dict, **kwargs):
        # TODO: consider how to handle if models have overlapping kwargs
        # Params is a dictionary which is updated by optimiser,
        # kwargs is constant i.e. settings for models
        self.bckc: dict = {}
        for model in self.diagnostic_models:
            self.bckc = dict(
                self.bckc, **{model.name: {**model(**{**params, **kwargs})}}
            )
        self.bckc = flatdict.FlatDict(self.bckc, delimiter=".")
        return

    def _ln_likelihood(self):
        ln_likelihood = 0
        for key in self.quant_to_optimise:
            # TODO: What to use as error?  Assume percentage error if none given...
            # Float128 is used since rounding of small numbers causes
            # problems when initial results are bad fits
            model_data = self.bckc[key].values.astype("float64")
            exp_data = (
                self.data[key]
                .sel(t=self.plasma.time_to_calculate)
                .values.astype("float64")
            )
            _ln_likelihood = np.log(gaussian(model_data, exp_data, exp_data * 0.10)) #check it with 5%
            ln_likelihood += np.nanmean(_ln_likelihood)
        return ln_likelihood

    def _ln_prior(self, parameters: dict):
        ln_prior = 0
        for prior_name, prior_func in self.priors.items():
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

    def sample_from_priors(self, param_names, size=10):
        #  Use priors to generate samples
        for name in param_names:
            if name in self.priors.keys():
                if hasattr(self.priors[name], "rvs"):
                    continue
                else:
                    raise TypeError(f"prior object {name} missing rvs method")
            else:
                raise ValueError(f"Missing prior for {name}")

        #  Throw out samples that don't meet conditional priors and redraw
        samples = np.empty((param_names.__len__(), 0))
        while samples.size < param_names.__len__() * size:
            # Some mangling of dictionaries so _ln_prior works
            # Increase size * n if too slow / looping too much
            new_sample = {
                name: self.priors[name].rvs(size=size * 2) for name in param_names
            }
            ln_prior = self._ln_prior(new_sample)
            # Convert from dictionary of arrays -> array,
            # then filtering out where ln_prior is -infinity
            accepted_samples = np.array(list(new_sample.values()))[
                :, ln_prior != -np.inf
            ]
            samples = np.append(samples, accepted_samples, axis=1)
        samples = samples[:, 0:size]
        return samples.transpose()

    def ln_posterior(self, parameters: dict, **kwargs):
        """
        Posterior probability given to optimisers

        Parameters
        ----------
        parameters
            inputs to optimise
        kwargs
            kwargs for models

        Returns
        -------
        ln_posterior
            log of posterior probability
        blob
            model outputs from bckc and kinetic profiles
        """

        ln_prior = self._ln_prior(parameters)
        if ln_prior == -np.inf:  # Don't call model if outside priors
            return -np.inf, {}

        self.plasma.update_profiles(parameters)
        self._build_bckc(parameters, **kwargs)  # model calls
        ln_likelihood = self._ln_likelihood()  # compare results to data
        ln_posterior = ln_likelihood + ln_prior

        kin_profs = {
            "electron_density": self.plasma.electron_density.sel(
                t=self.plasma.time_to_calculate
            ),
            "electron_temperature": self.plasma.electron_temperature.sel(
                t=self.plasma.time_to_calculate
            ),
            "ion_temperature": self.plasma.ion_temperature.sel(
                t=self.plasma.time_to_calculate
            ),
            "impurity_density": self.plasma.impurity_density.sel(
                t=self.plasma.time_to_calculate
            ),
            "zeff": self.plasma.zeff.sum("element").sel(
                t=self.plasma.time_to_calculate
            ),
            # TODO: add Nh
        }
        blob = deepcopy({**self.bckc, **kin_profs})
        return ln_posterior, blob
