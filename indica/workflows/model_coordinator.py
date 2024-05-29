from typing import Dict
import xarray as xr

from indica.equilibrium import Equilibrium
from indica.models import Plasma
from indica.converters.abstractconverter import CoordinateTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic


class ModelCoordinator:
    def __init__(self,
                 models: dict[AbstractDiagnostic],
                 model_settings: dict,
                 ):

        self.model_settings = model_settings
        self.models = {}

        for model_name, model in models.items:
            if model_name in model_settings.keys():
                self.models[model_name] = model(name = model_name, **self.model_settings[model_name])
            else:
                self.models[model_name] = model(name = model_name)

    def set_plasma(self, plasma: Plasma):
        for model_name, model in self.models.items():
            model.set_plasma(plasma)

    def set_transforms(self, transforms: dict[CoordinateTransform]):
        for model_name, model in self.models.items():
            if hasattr(self, "set_transform"):
                model.set_transform = transforms[model_name]

    def set_equilibrium(self, equilibrium: Equilibrium):
        for model_name, model in self.models.items():
            if hasattr(self, "transform"):
                model.transform.set_equilibrium(equilibrium)


    def get(
        self,
        instrument: str,
        *args,
        **kwargs,
    ) -> Dict[str, xr.DataArray]:
        """
        Method set to replicate the get() method of the readers
        """
        if instrument in self.models:
            return self.models[instrument](*args, **kwargs)
        else:
            print(f"{instrument} not is self.models")
            return {}

    def __call__(
            self,
            instruments = None,
            call_params: dict = None,
            call_kwargs: dict = None,
            **kwargs
    ):
        """
        Parameters
        ----------
        call_params - dictionary of model __call__ args with format dict(model_name: dict(setting_name: setting))
        call_kwargs - dictionary of model __call__ kwargs with format dict(model_name: dict(setting_name: setting))

        Returns
        -------
        dictionary of results
        """

        if call_params is None:
            call_params = {}
        if call_kwargs is None:
            call_kwargs = {}
        if instruments is None:
            instruments = self.models.keys()

        self.bckc: dict = {}

        for instrument in instruments:
            # removes "model.name." from params and kwargs then passes them to model
            # e.g. xrcs.background -> background
            _call_params = {
                param_name.replace(instrument + ".", ""): param_value
                for param_name, param_value in call_params.items()
                if instrument in param_name
            }
            # call_kwargs defined in model_settings
            _call_kwargs = {
                kwarg_name: kwarg_value
                for kwarg_name, kwarg_value in call_kwargs[
                    instrument
                ].items()
            }
            _model_kwargs = {
                **_call_kwargs,
                **_call_params,
            }  # combine dictionaries (TODO: optimiser nuisance params more explicitly defined vs kwargs / args)

            _bckc = self.get(instrument, **_model_kwargs)

            _model_bckc = {
                instrument: {value_name: value for value_name, value in _bckc.items()}
            }
            # prepend model name to bckc
            self.bckc = dict(self.bckc, **_model_bckc)
        return self.bckc
