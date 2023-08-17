from copy import deepcopy
import warnings

import numpy as np
from scipy.stats import uniform

np.seterr(all="ignore")
warnings.simplefilter("ignore", category=FutureWarning)


PROFILES = [
    "electron_temperature",
    "electron_density",
    "ion_temperature",
    "ion_density",
    "impurity_density",
    "fast_density",
    "neutral_density",
    "zeff",
]


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
        percent_error: float = 0.10,
    ):
        self.plasma = plasma
        self.data = data
        self.quant_to_optimise = quant_to_optimise
        self.diagnostic_models = diagnostic_models
        self.priors = priors
        self.percent_error = percent_error

        for diag_model in self.diagnostic_models:
            diag_model.plasma = self.plasma

        missing_data = list(set(quant_to_optimise).difference(data.keys()))
        if missing_data:  # gives list of keys in quant_to_optimise but not data
            raise ValueError(f"{missing_data} not found in data given")

    def _build_bckc(self, params, **kwargs):
        """
        Parameters
        ----------
        params - dictionary which is updated by optimiser
        kwargs - passed to model i.e. settings

        Returns
        -------
        bckc of results
        """
        self.bckc: dict = {}
        for model in self.diagnostic_models:
            # removes "model.name." from params and kwargs then passes them to model
            # e.g. xrcs.background -> background
            _nuisance_params = {
                param_name.replace(model.name + ".", ""): param_value
                for param_name, param_value in params.items()
                if model.name in param_name
            }
            _model_settings = {
                kwarg_name.replace(model.name + ".", ""): kwarg_value
                for kwarg_name, kwarg_value in kwargs.items()
                if model.name in kwarg_name
            }

            _model_kwargs = {
                **_nuisance_params,
                **_model_settings,
            }  # combine dictionaries
            _bckc = model(**_model_kwargs)
            _model_bckc = {
                f"{model.name}.{value_name}": value
                for value_name, value in _bckc.items()
            }  # prepend model name to bckc
            self.bckc = dict(self.bckc, **_model_bckc)
        return

    def _ln_likelihood(self):
        ln_likelihood = 0
        for key in self.quant_to_optimise:
            # Float128 since rounding of small numbers causes problems
            # when initial results are bad fits
            model_data = self.bckc[key].astype("float128")
            exp_data = (
                self.data[key].sel(t=self.plasma.time_to_calculate).astype("float128")
            )
            exp_error = (
                exp_data * self.percent_error
            )  # Assume percentage error if none given.
            if hasattr(self.data[key], "error"):
                if (
                    self.data[key].error != 0
                ).any():  # TODO: Some models have an error of 0 given
                    exp_error = self.data[key].error.sel(
                        t=self.plasma.time_to_calculate
                    )

            _ln_likelihood = np.log(gaussian(model_data, exp_data, exp_error))
            # treat channel as key dim which isn't averaged like other dims
            if "channel" in _ln_likelihood.dims:
                _ln_likelihood = _ln_likelihood.sum(dim="channel", skipna=True)
            ln_likelihood += _ln_likelihood.mean(skipna=True).values
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

    def sample_from_high_density_region(
        self, param_names: list, sampler, nwalkers: int, nsamples=100
    ):
        start_points = self.sample_from_priors(param_names, size=nsamples)

        ln_prob, _ = sampler.compute_log_prob(start_points)
        num_best_points = int(nsamples * 0.05)
        index_best_start = np.argsort(ln_prob)[-num_best_points:]
        best_start_points = start_points[index_best_start, :]
        best_points_std = np.std(best_start_points, axis=0)

        # Passing samples through ln_prior and redrawing if they fail
        samples = np.empty((param_names.__len__(), 0))
        while samples.size < param_names.__len__() * nwalkers:
            sample = np.random.normal(
                np.mean(best_start_points, axis=0),
                best_points_std * 2,
                size=(nwalkers * 5, len(param_names)),
            )
            start = {name: sample[:, idx] for idx, name in enumerate(param_names)}
            ln_prior = self._ln_prior(start)
            # Convert from dictionary of arrays -> array,
            # then filtering out where ln_prior is -infinity
            accepted_samples = np.array(list(start.values()))[:, ln_prior != -np.inf]
            samples = np.append(samples, accepted_samples, axis=1)
        start_points = samples[:, 0:nwalkers].transpose()
        return start_points

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
        if ln_prior == -np.inf:  # Don't call models if outside priors
            return -np.inf, {}

        self.plasma.update_profiles(parameters)
        self._build_bckc(parameters, **kwargs)  # model calls
        ln_likelihood = self._ln_likelihood()  # compare results to data
        ln_posterior = ln_likelihood + ln_prior

        plasma_profiles = {}
        for profile_key in PROFILES:
            if hasattr(self.plasma, profile_key):
                plasma_profiles[profile_key] = getattr(self.plasma, profile_key).sel(
                    t=self.plasma.time_to_calculate
                )
            else:
                raise ValueError(f"plasma does not have attribute {profile_key}")

        blob = deepcopy({**self.bckc, **plasma_profiles})
        return ln_posterior, blob
