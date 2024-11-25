from typing import Dict

import xarray as xr

from indica import Equilibrium
from indica import Plasma
from indica.converters.abstractconverter import CoordinateTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.readers.read_st40 import apply_filter
from indica.readers.read_st40 import coord_condition
from indica.readers.read_st40 import limit_condition


class ModelCoordinator:
    def __init__(
        self,
        models: dict[str, AbstractDiagnostic],
        model_settings: dict = None,
        verbose=False,
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
            if model_name in transforms.keys():
                model.set_transform(transforms[model_name])
            else:
                print(f"not adding transform to {model_name}")

    def set_equilibrium(self, equilibrium: Equilibrium):
        self.equilibrium = equilibrium
        for model_name, model in self.models.items():
            if hasattr(model, "transform"):
                model.transform.set_equilibrium(equilibrium, force=True)

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
            print(f"{instrument} not in self.models")
            return {}

    def __call__(
        self,
        instruments: list = None,
        filter_limits: dict = None,
        filter_coords: dict = None,
        flat_kwargs: dict = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        instruments - names of instruments to call
        filter_limits - filtering data if outside of filter_limits
        filter_coords - filtering data if coordinates outside of filter_coords
        flat_kwargs - flat kwargs to call models with

        Returns
        -------
        nested dictionary of results
        """

        if flat_kwargs is None:
            flat_kwargs = {}
        if instruments is None:
            instruments = self.model_names
        if filter_coords is None:
            filter_coords = {}
        if filter_limits is None:
            filter_limits = {}

        call_kwargs = {}
        for instrument in instruments:
            # format the model settings before calling them
            # reformat structure to remove prefix e.g. xrcs.background -> background
            _params = {
                param_name.replace(instrument + ".", ""): param_value
                for param_name, param_value in flat_kwargs.items()
                if instrument in param_name
            }
            call_kwargs[instrument] = {**_params, **kwargs.get(instrument, {})}
        self.call_kwargs = call_kwargs

        self.data: dict = {}
        for instrument in instruments:
            self.data[instrument] = self.get(
                instrument,
                **call_kwargs[instrument],
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
