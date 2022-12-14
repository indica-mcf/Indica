from abc import ABC
from abc import abstractmethod
import numpy as np

from indica.converters import FluxSurfaceCoordinates
from indica.converters.abstractconverter import CoordinateTransform
from indica.models.plasma import Plasma
from indica.numpy_typing import LabeledArray


class DiagnosticModel(ABC):
    name: str = ""
    bckc: dict = {}
    plasma: Plasma = None
    _t: LabeledArray

    def set_transform(self, transform: CoordinateTransform):
        """
        Set diagnostic coordinate transform of diagnostic
        TODO: some diagnostics should have both trivial and los transforms!!!
        """
        self.transform = transform
        self.bckc = {}

    def set_plasma(self, plasma: Plasma):
        """
        Assign Plasma class to use for computation of forward model
        """
        self.plasma = plasma

    @abstractmethod
    def _build_bckc_dictionary(self):
        self.bckc = {}
        return self.bckc

    @abstractmethod
    def __call__(
        self, **kwargs,
    ):
        """
        Calculate and return diagnostic measured values
        """
        return self._build_bckc_dictionary()
