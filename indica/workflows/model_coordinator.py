from typing import Dict
import xarray as xr

from indica.equilibrium import Equilibrium
from indica.models import Plasma
from indica.converters.abstractconverter import CoordinateTransform
from indica.models.abstractdiagnostic import DiagnosticModel


class ModelCoordinator:
    def __init__(self,
                 model_settings: dict,
                 models: dict[DiagnosticModel], ):

        self.model_settings = model_settings
        self.models = {}

        for model_name, model in models.items:
            if model_name in model_settings.keys():
                self.models[model_name] = model(name = model_name, **self.model_settings[model_name])
            else:
                self.models[model_name] = model(name = model_name)

    """
    Setup models so that they have transforms / plasma / equilibrium etc..
    everything needed to produce bckc from models
    
    TODO: check when model doesn't need transform such as EquilibriumReconstruction
    """

    def set_plasma(self, plasma: Plasma):
        for model_name, model in self.models.items():
            model.set_plasma(plasma)

    def set_equilibrium(self, equilibrium: Equilibrium):
        for model_name, model in self.models.items():
            model.transform.set_equilibrium(equilibrium)

    def set_transforms(self, transforms: dict[CoordinateTransform]):
        for model_name, model in self.models.items():
            model.set_transform = transforms[model_name]

    def get(
        self,
        instrument: str,
        **kwargs,
    ) -> Dict[str, xr.DataArray]:
        """
        Method set to replicate the get() method of the readers
        """
        if instrument in self.models:
            return self.models[instrument](**kwargs)
        else:
            print(f"{instrument} not is self.models")
            return {}

    def __call__(
            self,
            call_params: dict = None,
            call_kwargs: dict = None,  # model setting or call kwarg?
    ):
        """
        Parameters
        ----------
        call_params - nested dictionary of format dict(model_name: dict(setting_name: setting) )

        Returns
        -------
        nested bckc of results
        """

        if call_params is None:
            call_params = {}
        if call_kwargs is None:
            call_kwargs = {}

        self.bckc: dict = {}

        for model_name, model in self.models.items():
            # removes "model.name." from params and kwargs then passes them to model
            # e.g. xrcs.background -> background
            _call_params = {
                param_name.replace(model.name + ".", ""): param_value
                for param_name, param_value in call_params.items()
                if model.name in param_name
            }
            # call_kwargs defined in model_settings
            _call_kwargs = {
                kwarg_name: kwarg_value
                for kwarg_name, kwarg_value in call_kwargs[
                    model_name
                ].items()
            }
            _model_kwargs = {
                **_call_kwargs,
                **_call_params,
            }  # combine dictionaries

            _bckc = self.get(model_name, **_model_kwargs)

            _model_bckc = {
                model.name: {value_name: value for value_name, value in _bckc.items()}
            }
            # prepend model name to bckc
            self.bckc = dict(self.bckc, **_model_bckc)
        return self.bckc
