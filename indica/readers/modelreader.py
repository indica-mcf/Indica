from typing import Any
from typing import Dict
from typing import List

from xarray import DataArray

from indica.configs.readers.st40conf import ST40Conf
from indica.models import BolometerCamera
from indica.models import BremsstrahlungDiode
from indica.models import ChargeExchangeSpectrometer
from indica.models import HelikeSpectrometer
from indica.models import Interferometer
from indica.models import SXRcamera
from indica.models import ThomsonScattering
from indica.models.plasma import Plasma

# TODO: First stab, but need to check Michael Gemmell implementation

INSTRUMENT_MODELS = {
    "st40": {
        "smmh": Interferometer,
        "cxff_pi": ChargeExchangeSpectrometer,
        "cxff_tws_c": ChargeExchangeSpectrometer,
        "sxr_spd": SXRcamera,
        "sxrc_xy1": BolometerCamera,
        "sxrc_xy2": SXRcamera,
        "pi": BremsstrahlungDiode,
        "ts": ThomsonScattering,
        "tws_c": BremsstrahlungDiode,
        "xrcs": HelikeSpectrometer,
    }
}


class ModelReader:
    """Reads output of diagnostic forward models"""

    def __init__(
        self,
        machine: str,
        instruments: List[str] = [],
        **kwargs: Any,
    ):
        """Reader for synthetic diagnostic measurements making use of:
         - A Plasma class to be used by all models
         - Geometry from a standard set or from the experimental database

        Parameters
        ----------
        diagnostics
            List of diagnostic string identifiers to model.
        machine
            Machine string identifier on which the diagnostics are "installed".

        # TODO: Los and transect transforms must still be distinguished
            but will be solved once transect == special LOS transform case
        """
        if machine == "st40":
            _conf = ST40Conf()
        else:
            raise ValueError(f"Machine {machine} currently not supported")

        self.models: dict = {}
        self.transforms: dict = {}
        self.machine = machine
        self.machine_conf = _conf

        if len(instruments) == 0:
            self._instruments = list(INSTRUMENT_MODELS[machine])
        else:
            self._instruments = [
                instr
                for instr in instruments
                if instr in INSTRUMENT_MODELS[machine].keys()
            ]

        for instr in self._instruments:
            self.models[instr] = INSTRUMENT_MODELS[machine][instr](name=instr)

    def set_geometry_transforms(self, transforms: dict):
        """
        Set instrument geometry from standard set
        """

        for instr in self._instruments:
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
