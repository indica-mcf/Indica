from typing import Any
from typing import Dict
from typing import List

from xarray import DataArray

from indica import Plasma
from indica.configs import MACHINE_CONFS
from indica.models import MODELS_METHODS


class ModelReader:
    """Reads output of diagnostic forward models"""

    def __init__(
        self,
        machine: str,
        instruments: List[str],
        **kwargs: Any,
    ):
        """Reader for synthetic diagnostic measurements making use of:
         - A Plasma class to be used by all models
         - Geometry from a standard set or from the experimental database

        Parameters
        ----------
        machine
            Machine string identifier on which the diagnostics are "installed".
        instruments
            List of instruments to be modelled.
        """
        self.models: dict = {}
        self.transforms: dict = {}
        self.machine = machine
        self.machine_conf = MACHINE_CONFS[machine]()
        self.instruments = instruments
        self.kwargs = kwargs

        for instrument in instruments:
            method = self.machine_conf.INSTRUMENT_METHODS[instrument]
            model = MODELS_METHODS[method]
            self.models[instrument] = model(instrument)

    def set_geometry_transforms(self, transforms: dict):
        """
        Set instrument geometry from standard set
        """

        for instr in self.instruments:
            if instr not in transforms.keys():
                raise ValueError(f"{instr} not available in default_geometries file")

            self.transforms[instr] = transforms[instr]
            self.models[instr].set_transform(transforms[instr])

    def set_plasma(self, plasma: Plasma):
        """
        Set Plasma class to all models and transforms
        """
        for instr in self.models.keys():
            if instr not in self.models:
                continue

            self.models[instr].set_plasma(plasma)
            self.transforms[instr].set_equilibrium(
                plasma.equilibrium,
                force=True,
            )

        self.plasma = plasma

    def set_model_parameters(self, instrument: str, **kwargs):
        """
        Update independent model parameters
        """

    def get(
        self,
        uid: str,
        instrument: str,
        **kwargs,
    ) -> Dict[str, DataArray]:
        """
        Method set to replicate the get() method of the readers
        uid is not necessary but kept for consistency
        TODO: think whether it's best to make UID a kwarg instead!
        """
        _ = uid
        if instrument in self.models:
            return self.models[instrument](**kwargs)
        else:
            return {}

    def __call__(self, instruments: list = [], **kwargs):
        if len(instruments) == 0:
            instruments = list(self.models)

        bckc: dict = {}
        for instrument in instruments:
            bckc[instrument] = self.get("", instrument)

        return bckc
