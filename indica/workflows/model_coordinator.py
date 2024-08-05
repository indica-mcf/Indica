from typing import Dict

import xarray as xr

from indica.converters.abstractconverter import CoordinateTransform
from indica.equilibrium import Equilibrium
from indica.models import Plasma
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.readers.read_st40 import apply_filter
from indica.readers.read_st40 import coord_condition
from indica.readers.read_st40 import limit_condition


class ModelCoordinator:
    def __init__(
        self,
        models: dict[str, AbstractDiagnostic],
        model_settings: dict = None,
        verbose=True,
    ):

        self.transforms = None
        self.plasma = None
        self.equilibrium = None
        if model_settings is None:
            model_settings = {}
        self.model_settings = model_settings
        self.model_names = list(models.keys())
        self.verbose = verbose

        self.models: dict[str, AbstractDiagnostic] = {}
        for model_name, model in models.items():
            self.models[model_name] = model(
                name=model_name, **self.model_settings.get(model_name, {})
            )

    def set_plasma(self, plasma: Plasma):
        self.plasma = plasma
        for model_name, model in self.models.items():
            model.set_plasma(plasma)

    def set_transforms(self, transforms: dict[CoordinateTransform]):
        self.transforms = transforms
        for model_name, model in self.models.items():
            if hasattr(model, "set_transform"):
                model.set_transform(transforms[model_name])

    def set_equilibrium(self, equilibrium: Equilibrium):
        self.equilibrium = equilibrium
        for model_name, model in self.models.items():
            if hasattr(model, "transform"):
                model.transform.set_equilibrium(equilibrium, force=True)

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
        instruments=None,
        filter_limits: dict = None,
        filter_coords: dict = None,
        nested_kwargs: dict = None,
        *args,
        **flat_kwargs,
    ):
        """
        Parameters
        ----------
        nested_kwargs - model __call__ args as dict(model_name:
                                                    dict(setting_name: setting))
        flat_kwargs - model __call__ kwargs as dict(model_name.setting_name: setting)

        Returns
        -------
        dictionary of results
        """

        if (
            nested_kwargs is None
        ):  # These can be nuisance parameters passed by optimiser
            nested_kwargs = {}
        if flat_kwargs is None:
            flat_kwargs = {}
        if instruments is None:
            instruments = self.models.keys()
        if filter_coords is None:
            filter_coords = {}
        if filter_limits is None:
            filter_limits = {}

        model_kwargs = {}

        for instrument in instruments:
            # reformat e.g. xrcs.background -> background
            _params = {
                param_name.replace(instrument + ".", ""): param_value
                for param_name, param_value in flat_kwargs.items()
                if instrument in param_name
            }
            # TODO: mish mash of nested and flat dicts is confusing here
            if instrument in nested_kwargs.keys():
                model_kwargs[instrument] = {**_params, **nested_kwargs[instrument]}
            else:
                model_kwargs[instrument] = {**_params}

        self.data: dict = {}
        for instrument in instruments:
            self.data[instrument] = self.get(
                instrument,
                *args,
                **model_kwargs[instrument],
            )

        self.filtered_data = apply_filter(
            self.data,
            filters=filter_limits,
            filter_func=limit_condition,
            filter_func_name="limits",
            verbose=self.verbose,
        )
        self.filtered_data = apply_filter(
            self.filtered_data,
            filters=filter_coords,
            filter_func=coord_condition,
            filter_func_name="co-ordinate",
            verbose=self.verbose,
        )
        self.binned_data = self.filtered_data

        return self.binned_data
