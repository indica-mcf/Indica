from abc import ABC
from abc import abstractmethod

from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters.transect import TransectCoordinates
from indica.models.plasma import Plasma


class DiagnosticModel(ABC):
    name: str = ""
    bckc: dict = {}
    plasma: Plasma = None
    los_transform: LineOfSightTransform
    transect_transform: TransectCoordinates

    def set_los_transform(self, los_transform: LineOfSightTransform):
        """
        Line-of-sight coordinate transform
        """
        self.los_transform = los_transform

    def set_transect_transform(self, transect_transform: TransectCoordinates):
        """
        Transect coordinate transform
        """
        self.transect_transform = transect_transform

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
        self,
        **kwargs,
    ):
        """
        Calculate and return diagnostic measured values
        """
        return self._build_bckc_dictionary()
