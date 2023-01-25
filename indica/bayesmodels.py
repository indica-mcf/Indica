from copy import deepcopy
import flatdict
import numpy as np
np.seterr(divide="ignore")
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from scipy.stats import uniform

def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)


def get_uniform(lower, upper):
    # Less confusing parameterisation of scipy.stats uniform
    return uniform(loc=lower, scale=upper-lower)


class BayesModels:
    """
    Class which operates with Plasma class to create ln_posterior method

    Parameters
    ----------
    plasma
        Plasma object needed for the optimisation
    data
        processed diagnostic data of format [diagnostic]_[quantity]
    quant_to_optimise
        quantity from data which will be optimised with bckc from diagnostic_models
    priors
        prior functions to apply to parameters for ln_posterior
    diagnostic_models
        model objects to be called inside of ln_posterior
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

        missing_data = list(
            set(quant_to_optimise).difference(data.keys())
        )  # list of keys in quant_to_optimise but not data
        if missing_data:
            raise ValueError(f"{missing_data} not found in data given")

    def _build_bckc(self, params: dict, **kwargs):
        # TODO: consider how to handle if models have overlapping kwargs
        self.bckc = {}
        for model in self.diagnostic_models:
            self.bckc = dict(self.bckc, **{model.name: {**model(**{**params, **kwargs})}})
        self.bckc = flatdict.FlatDict(self.bckc, delimiter=".")
        return

    def _ln_likelihood(self):
        ln_likelihood = 0
        for key in self.quant_to_optimise:
            # TODO: What to use as error?  Assume percentage error if none given...
            ln_likelihood += np.log(
                gaussian(
                    self.bckc[key].values,
                    self.data[key].sel(t=self.plasma.time_to_calculate).values,
                    self.data[key].sel(t=self.plasma.time_to_calculate).values * 0.10,
                )
            )
        return ln_likelihood

    def _ln_prior(self, parameters: dict):
        # TODO: check for conditional priors before giving warning /
        #  Beware if conditional prior is given when only 1 of its parameters is searched it will give too few args
        # params_without_priors = [x for x in parameters.keys() if x not in self.priors.keys()]
        # if params_without_priors.__len__() > 0:
        #     print(f"paramaters {params_without_priors} have no priors assigned")

        ln_prior = 0
        for prior_name, prior_func in self.priors.items():
            param_values = [parameters[x] for x in parameters.keys() if x in prior_name]
            if param_values.__len__() == 0:  # if prior assigned but no parameter then skip
                continue
            elif param_values.__len__() >= 1:
                if hasattr(prior_func, "pdf"):  # for scipy.stats objects use pdf / for lambda functions just call
                    ln_prior += np.log(prior_func.pdf(*param_values))
                else:
                    # if lambda prior with 2+ args is defined when only 1 of its parameters is given ignore it
                    if prior_func.__code__.co_argcount != param_values.__len__():
                        continue
                    else:
                        ln_prior += np.log(prior_func(*param_values))
            else:
                raise ValueError(f"Unexpected value for {param_values}")
        return ln_prior

    def sample_from_priors(self, param_names, size=10):
        #  Use priors to generate samples
        #  Through out samples that don't meet conditional priors and redraw
        for name in param_names:
            if name in self.priors.keys():
                if hasattr(self.priors[name], "rvs"):
                    continue
                else:
                    raise TypeError(f"prior object {name} missing rvs method")
            else:
                raise ValueError(f"Missing prior for {name}")

        samples = np.empty((param_names.__len__(), 0))
        while samples.size < param_names.__len__() * size:
            # Some mangling of dictionaries so _ln_prior works
            new_sample = {name: self.priors[name].rvs(size=size) for name in param_names}
            ln_prior = self._ln_prior(new_sample)
            # Convert back from dictionary of arrays to array where ln_prior is finite
            accepted_samples = np.array(list(new_sample.values()))[:, ln_prior!=-np.inf]
            samples = np.append(samples, accepted_samples, axis=1)
        samples = samples[:,0:size]
        return samples.transpose()


    def ln_posterior(self, parameters: dict, **kwargs):
        """
        Posterior probability given to optimisers

        Parameters
        ----------
        parameters
            inputs to optimise
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

        # Add better way of handling time array
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
            # TODO: add Ni/Nh/Nimp when fz property works 1 timepoint
        }
        blob = deepcopy({**self.bckc, **kin_profs})
        return ln_posterior, blob


