from copy import deepcopy
import warnings
from typing import Callable

from flatdict import FlatDict
import numpy as np

from indica.workflows.plasma_profiler import PlasmaProfiler
from indica.workflows.priors import PriorManager

np.seterr(all="ignore")
warnings.simplefilter("ignore", category=FutureWarning)


def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)


class BayesBlackBox:
    """
    Bayesian black box model that creates _ln_posterior function
    from plasma and diagnostic model objects

    Parameters
    ----------
    data
        processed diagnostic data of format [diagnostic].[quantity]
    quant_to_optimise
        quantities from data which will be optimised with bckc from diagnostic_models
    plasma_profiler
        plasma interface has methods for setting profiles in plasma
    build_bckc
        function to return model data called by ln_posterior
    prior_manager
        prior class which calculates ln_prior from given parameters

    """

    def __init__(
        self,
        data: dict,
        quant_to_optimise: list,
        prior_manager: PriorManager,
        build_bckc: Callable,
        plasma_profiler = PlasmaProfiler,
    ):
        self.data = data
        self.quant_to_optimise = quant_to_optimise
        self.prior_manager = prior_manager
        self.plasma_profiler = plasma_profiler
        self.build_bckc = build_bckc

        missing_data = list(set(quant_to_optimise).difference(data.keys()))
        if missing_data:
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

        _ln_prior = self.prior_manager.ln_prior(parameters)
        if _ln_prior == -np.inf:  # Don't call models if outside priors
            return -np.inf, {}

        self.plasma_profiler(parameters)
        plasma_attributes = self.plasma_profiler.plasma_attributes()

        self.bckc = FlatDict(
            self.build_bckc(parameters, **kwargs), "."
        )  # model calls

        _ln_likelihood = self.ln_likelihood()  # compare results to data
        ln_posterior = _ln_likelihood + _ln_prior

        blob = deepcopy({**self.bckc, **plasma_attributes})
        return ln_posterior, blob
