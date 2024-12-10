
from xarray import DataArray

from indica import Plasma, Equilibrium
from indica.configs import ST40Conf
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.readers.readerprocessor import apply_filter, value_condition, coordinate_condition


class ModelReader:
    """Reads output of diagnostic forward models"""

    def __init__(
        self,
        models: dict[str, AbstractDiagnostic],
        model_kwargs: dict,
        conf = ST40Conf,
    ):
        """Reader for synthetic diagnostic measurements making use of:

        Parameters
        ----------
        models

        model_kwargs

        conf

        """
        self.models = models
        self.model_kwargs = model_kwargs
        self.conf = conf()

        self.transforms: dict = {}
        self.plasma = None

        for model_name, model in models.items():
            self.models[model_name] = model(
                name=model_name, **self.model_kwargs.get(model_name, {})
            )

    def set_geometry_transforms(self, transforms: dict, equilibrium: Equilibrium):
        """
        Set instrument geometry and equilibrium
        """

        for instr in self.models.keys():
            if instr not in transforms.keys():
                raise ValueError(f"{instr} not available in given transforms")

            self.transforms[instr] = transforms[instr]
            self.transforms[instr].set_equilibrium(equilibrium, force=True)
            self.models[instr].set_transform(transforms[instr])

    def set_plasma(self, plasma: Plasma):
        """
        Set Plasma class to all models and transforms
        """
        for model_name, model in self.models.items():
            model.set_plasma(plasma)
        self.plasma = plasma

    def update_model_settings(self, **update_kwargs):
        """
        Reinitialise models with model kwargs
        """
        for instrument in self.models:
            self.model_kwargs[instrument] = self.model_kwargs[instrument].update(update_kwargs[instrument])
            self.models = self.models[instrument](**self.model_kwargs[instrument])


    def get(
        self,
        instrument: str,
        **kwargs,
    ) -> dict[str, DataArray]:
        """
        Method set to replicate the get() method of the readers
        """
        if instrument in self.models.keys():
            return self.models[instrument](**kwargs)
        else:
            return {}

    def __call__(self, instruments: list = None, filter_limits=None, filter_coords=None, verbose=False, **call_kwargs):
        if instruments is None:
            instruments = self.models.keys()

        if filter_limits is None:
            filter_limits = self.conf.filter_value
        if filter_coords is None:
            filter_coords = self.conf.filter_coordinates

        bckc: dict = {}
        for instrument in instruments:
            bckc[instrument] = self.get(instrument, **call_kwargs[instrument])

        filtered_bckc = apply_filter(
            bckc,
            filters=filter_limits,
            filter_func=value_condition,
            filter_func_name="limits",
            verbose=verbose,
        )
        filtered_bckc = apply_filter(
            filtered_bckc,
            filters=filter_coords,
            filter_func=coordinate_condition,
            filter_func_name="co-ordinate",
            verbose=verbose,
        )

        return filtered_bckc
