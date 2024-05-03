from copy import deepcopy
import warnings
from typing import Callable

from flatdict import FlatDict
import numpy as np

np.seterr(all="ignore")
warnings.simplefilter("ignore", category=FutureWarning)


def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)


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


class BayesBlackBox:
    """
    Bayesian black box model that creates _ln_posterior function
    from plasma and diagnostic model objects

    Parameters
    ----------
    data
        processed diagnostic data of format [diagnostic].[quantity]
    plasma_context
        plasma context has methods for using plasma object
    model_handler
        model_handler object to be called by ln_posterior
    quant_to_optimise
        quantity from data which will be optimised with bckc from diagnostic_models
    priors
        prior functions to apply to parameters e.g. scipy.stats.rv_continuous objects

    """

    def __init__(
        self,
        data: dict,
        quant_to_optimise: list,
        priors: dict,
        build_bckc: Callable,
        plasma_profiler=None,
    ):
        self.data = data
        self.quant_to_optimise = quant_to_optimise
        self.priors = priors

        self.plasma_profiler = plasma_profiler
        self.build_bckc = build_bckc

        missing_data = list(set(quant_to_optimise).difference(data.keys()))
        if missing_data:  # gives list of keys in quant_to_optimise but not data
            raise ValueError(f"{missing_data} not found in data given")

    def ln_likelihood(self):
        ln_likelihood = 0
        time_coord = self.plasma_profiler.plasma.time_to_calculate

        for key in self.quant_to_optimise:
            model_data = self.bckc[key]
            exp_data = self.data[key].sel(t=time_coord)
            exp_error = self.data[key].error.sel(t=time_coord)

            _ln_likelihood = np.log(gaussian(model_data, exp_data, exp_error))
            # treat channel as key dim which isn't averaged like other dims
            if "channel" in _ln_likelihood.dims:
                _ln_likelihood = _ln_likelihood.sum(dim="channel", skipna=True)
            ln_likelihood += _ln_likelihood.mean(skipna=True).values
        return ln_likelihood

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

        _ln_prior = ln_prior(self.priors, parameters)
        if _ln_prior == -np.inf:  # Don't call models if outside priors
            return -np.inf, {}

        self.plasma_profiler(parameters)
        plasma_attributes = self.plasma_profiler.return_plasma_attributes()

        self.bckc = FlatDict(
            self.build_bckc(parameters, **kwargs), "."
        )  # model calls

        _ln_likelihood = self.ln_likelihood()  # compare results to data
        ln_posterior = _ln_likelihood + _ln_prior

        blob = deepcopy({**self.bckc, **plasma_attributes})
        return ln_posterior, blob
