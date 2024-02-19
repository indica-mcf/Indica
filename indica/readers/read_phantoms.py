"""Experimental design for reading data from disk/database.
"""

from typing import Any
from typing import Dict
from typing import Set
from xarray import DataArray

from indica.numpy_typing import RevisionLike
from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters.transect import TransectCoordinates
from indica.models.plasma import example_run as phantom_plasma
from indica.readers.read_st40 import default_geometries
from indica.models.interferometry import Interferometry
from indica.models.helike_spectroscopy import HelikeSpectrometer
from indica.models.thomson_scattering import ThomsonScattering
from indica.models.charge_exchange import ChargeExchange
from indica.models.sxr_camera import SXRcamera
from indica.models.bolometer_camera import Bolometer
from indica.models.diode_filters import BremsstrahlungDiode
from indica.models.plasma import Plasma

# TODO: hardcoded for ST40!!! to be moved somewhere else!!!
MACHINE_DIMENSIONS = ((0.15, 0.95), (-0.7, 0.7))
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

class PhantomReader:
    """Reader of phantom plasma and diagnostic forward model data
    """

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        dt: float,
        dl: float = 0.005,
        machine: str = "st40",
        **kwargs: Any,
    ):
        """Returns set of phantom diagnostics using a pre-defined phantom plasma class

        Parameters
        ----------
        tstart
            Start of time range for which to get data.
        tend
            End of time range for which to get data.
        dt
            Delta t of time window
        kwargs
            Any other arguments which should be recorded in the PROV entity for
            the reader.

        """
        self.pulse = pulse
        self._machine = machine
        self._machine_dims = MACHINE_DIMENSIONS
        self._tstart = tstart
        self._tend = tend
        self._dt = dt
        self._instr_geometries = default_geometries("st40")

        instr_models: dict = {}
        for instr, geom in self._instr_geometries.items():
            if instr not in INSTRUMENT_MODELS.keys():
                continue

            _model = INSTRUMENT_MODELS[instr](name=instr)
            geom["machine_dimensions"] = self._machine_dims
            if "name" not in geom:
                geom["name"] = instr

            if "origin_x" in geom.keys():
                geom["dl"] = dl
                _model.set_los_transform(LineOfSightTransform(**geom,))
            else:
                _model.set_transect_transform(TransectCoordinates(**geom))

            instr_models[instr] = _model

        self.instr_models = instr_models

    def set_plasma(self, plasma: Plasma = None):
        if plasma is None:
            plasma = phantom_plasma(
                pulse=None, tstart=self._tstart, tend=self._tend, dt=self._dt
            )

        for instr in self.instr_models.keys():
            if instr not in self.instr_models:
                continue

            self.instr_models[instr].set_plasma(plasma)
            if hasattr(self.instr_models[instr], "los_transform"):
                self.instr_models[instr].los_transform.set_equilibrium(
                    plasma.equilibrium, force=True,
                )
            else:
                self.instr_models[instr].transect_transform.set_equilibrium(
                    plasma.equilibrium, force=True,
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

        if instrument in self.instr_models:
            return self.instr_models[instrument]()
        else:
            return {}

    def __call__(self, instruments: list = [], **kwargs):
        if len(instruments) == 0:
            instruments = list(self.instr_models)

        self.set_plasma()

        binned_data: dict = {}
        for instrument in instruments:
            binned_data[instrument] = self.get("", instrument)

        self.binned_data = binned_data
