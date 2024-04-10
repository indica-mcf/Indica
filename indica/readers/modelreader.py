"""Experimental design for reading data from disk/database.
"""

from typing import Any
from typing import Dict
from typing import Set

from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters.transect import TransectCoordinates
from indica.models.bolometer_camera import Bolometer
from indica.models.charge_exchange import ChargeExchange
from indica.models.diode_filters import BremsstrahlungDiode
from indica.models.helike_spectroscopy import HelikeSpectrometer
from indica.models.interferometry import Interferometry
from indica.models.plasma import example_run as phantom_plasma
from indica.models.plasma import Plasma
from indica.models.sxr_camera import SXRcamera
from indica.models.thomson_scattering import ThomsonScattering
from indica.numpy_typing import RevisionLike
from indica.settings.default_settings import default_geometries
from indica.settings.default_settings import MACHINE_DIMS
from typing import List

# TODO: hardcoded for ST40!!! to be moved somewhere else!!!
INSTRUMENT_MODELS = {
    "smmh": Interferometry,
    "cxff_pi": ChargeExchange,
    "cxff_tws_c": ChargeExchange,
    "sxr_spd": SXRcamera,
    "sxrc_xy1": Bolometer,
    "sxrc_xy2": SXRcamera,
    "pi": BremsstrahlungDiode,
    "ts": ThomsonScattering,
    "tws_c": BremsstrahlungDiode,
    "xrcs": HelikeSpectrometer,
}

# TODO: First stab, but need to check Michael Gemmell implementation

class ModelReader:
    """Reads output of diagnostic forward models"""

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        dt: float,
        dl: float = 0.005,
        instruments:List[str]=[],
        machine: str = "st40",
        **kwargs: Any,
    ):
        """Returns set of phantom diagnostics using a pre-defined phantom plasma class

        Parameters
        ----------
        diagnostics
            List of diagnostic identifiers for which a model exists.
        machine
            Machine identifier on which the diagnostic is "installed".

        # TODO: Los and transect transforms must still be distinguished
            but will be solved once transect == special LOS transform case
        """
        self.models: dict = {}
        self.transforms:dict = {}
        self.machine = machine
        self._machine_dims = MACHINE_DIMS[machine]
        self._tstart = tstart
        self._tend = tend
        self._dt = dt

        if len(instruments) == 0:
            self._instruments = INSTRUMENT_MODELS.keys()
        else:
            self._instruments = [instr for instr in instruments if instr in INSTRUMENT_MODELS.keys()]

        for instr in self._instruments:
            self.models[instr] = INSTRUMENT_MODELS[instr](name=instr)

        # if pulse > 0:
        #
        # else:
        #     _instr_geometries = default_geometries(machine)
        #     for instr, geom in _instr_geometries.items():
        #         geom["machine_dimensions"] = self._machine_dims
        #         geom["name"] = instr
        #         if "origin_x" in geom.keys():
        #             geom["dl"] = dl
        #             self.transforms[instr] = LineOfSightTransform(**geom)
        #             self.models[instr].set_los_transform(self.transforms[instr])
        #         else:
        #             self.transforms[instr] = TransectCoordinates(**geom)
        #             self.models[instr].set_transect_transform(self.transforms[instr])

    def set_plasma(self, plasma: Plasma = None):
        if plasma is None:
            plasma = phantom_plasma(
                pulse=None, tstart=self._tstart, tend=self._tend, dt=self._dt
            )

        for instr in self.models.keys():
            if instr not in self.models:
                continue

            self.models[instr].set_plasma(plasma)
            self.transforms[instr].set_equilibrium(
                plasma.equilibrium,
                force=True,
            )

        self.plasma = plasma

    def get(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike = 0,
        quantities: Set[str] = set(),
        **kwargs,
    ) -> Dict[str, DataArray]:

        if instrument in self.models:
            return self.models[instrument]()
        else:
            return {}

    def __call__(self, instruments: list = [], **kwargs):
        if len(instruments) == 0:
            instruments = list(self.models)

        self.set_plasma()

        binned_data: dict = {}
        for instrument in instruments:
            binned_data[instrument] = self.get("", instrument)

        self.binned_data = binned_data
