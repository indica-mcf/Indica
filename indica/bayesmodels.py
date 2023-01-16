from copy import deepcopy
import flatdict
import numpy as np


np.seterr(divide="ignore")


def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mean) / sigma) ** 2)


def uniform(x, lower, upper):
    if (x > lower) & (x < upper):
        return 1
    else:
        return 0


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

    def _outside_bounds(self, parameters):
        for param_name, param_value in parameters.items():
            if param_name in self.priors:  # if no prior is defined then ignore
                prior = np.log(self.priors[param_name](param_value))
                if prior == -np.inf:
                    return True
        return False

    def _build_bckc(self, params={}, **kwargs):
        # TODO: consider how to handle if models have overlapping kwargs
        self.bckc = {}
        for model in self.diagnostic_models:
            self.bckc = dict(self.bckc, **{model.name: {**model(params=params, **kwargs)}})
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
        ln_prior = 0
        for param_name, param_value in parameters.items():
            if param_name in self.priors:  # if no prior is defined then ignore
                ln_prior += np.log(self.priors[param_name](param_value))
            else:
                print(f"No prior assigned for {param_name}")
        return ln_prior

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
        outside_bounds = self._outside_bounds(parameters)
        if outside_bounds:
            return -np.inf, {}

        self.plasma.update_profiles(parameters)
        self._build_bckc(parameters, **kwargs)  # model calls
        ln_likelihood = self._ln_likelihood()  # compare results to data
        ln_prior = self._ln_prior(parameters)
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


