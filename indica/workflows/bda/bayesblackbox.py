from copy import deepcopy
from typing import Callable
import warnings

from flatdict import FlatDict
import numpy as np
import mpmath as mp

from indica.workflows.bda.plasma_profiler import PlasmaProfiler
from indica.workflows.bda.priors import PriorManager

np.seterr(all="ignore")
warnings.simplefilter("ignore", category=FutureWarning)

# using mpmath to handle small numbers as floats insufficient
# wrapping mp functions to process numpy arrays
gaussian = np.frompyfunc(mp.npdf, 3, 1)
log = np.frompyfunc(mp.log, 1, 1)
to_float = np.frompyfunc(float, 1, 1)


def mp_log_gauss_wrapper(x, mean, sigma):
    # dropping nans to speed up calculation
    nan_idx = np.isnan(x) | np.isnan(mean) | np.isnan(sigma)
    x_star = x[~nan_idx] * mp.mpf(1)  # change dtype to mp float
    mean_star = mean[~nan_idx] * mp.mpf(1)
    sigma_star = sigma[~nan_idx] * mp.mpf(1)
    y = gaussian(x_star, mean_star, sigma_star)
    ln_y = to_float(log(y))  # mpf -> float
    return np.hstack(ln_y[:]).astype(float)  # convert from datatype object to float


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
        opt_data: dict,
        quant_to_optimise: list,
        prior_manager: PriorManager,
        build_bckc: Callable,
        plasma_profiler: PlasmaProfiler,
    ):
        self.opt_data = opt_data
        self.quant_to_optimise = quant_to_optimise
        self.prior_manager = prior_manager
        self.build_bckc = build_bckc
        self.plasma_profiler = plasma_profiler

        missing_data = list(set(quant_to_optimise).difference(opt_data.keys()))
        if missing_data:
            raise ValueError(f"{missing_data} not found in data given")

    def ln_likelihood(self):
        ln_likelihood = 0
        time_coord = self.plasma_profiler.plasma.time_to_calculate

        for key in self.quant_to_optimise:
            model_data = self.bckc[key].values
            exp_data = self.opt_data[key].sel(t=time_coord).values
            exp_error = self.opt_data[key].sel(t=time_coord).error.values

            _ln_likelihood = mp_log_gauss_wrapper(model_data, exp_data, exp_error)
            ln_likelihood += np.nansum(_ln_likelihood)

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
            self.build_bckc(flat_kwargs=parameters, **kwargs), "."
        )  # model calls

        _ln_likelihood = self.ln_likelihood()  # compare results to data
        ln_posterior = _ln_likelihood + _ln_prior

        blob = deepcopy({**self.bckc, **plasma_attributes})
        return ln_posterior, blob
