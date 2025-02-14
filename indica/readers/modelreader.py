from typing import Dict

from xarray import DataArray

from indica import Equilibrium
from indica import Plasma
from indica.models.abstract_diagnostic import AbstractDiagnostic


class ModelReader:
    """Reads output of diagnostic forward models"""

    def __init__(
        self,
        models: Dict[str, AbstractDiagnostic],
        model_kwargs: dict = {},
    ):
        """Reader for synthetic diagnostic measurements making use of:

        Parameters
        ----------

        models
        model_kwargs

        """
        self.models = models
        self.model_kwargs = model_kwargs

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

    def update_model_settings(self, update_kwargs={}):
        """
        Reinitialise models with model kwargs
        """
        for instrument in self.models:
            self.model_kwargs[instrument] = self.model_kwargs[instrument].update(
                update_kwargs.get(instrument, {})
            )
            self.models[instrument] = self.models[instrument](
                name=instrument, **self.model_kwargs.get(instrument, {})
            )

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

    def __call__(
        self,
        instruments: list = None,
        **call_kwargs,
    ):
        if instruments is None:
            instruments = self.models.keys()

        bckc: dict = {}
        for instrument in instruments:
            bckc[instrument] = self.get(instrument, **call_kwargs.get(instrument, {}))

        return bckc
