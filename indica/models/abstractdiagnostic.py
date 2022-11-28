from indica.converters.abstractconverter import CoordinateTransform
from indica.models.plasma import Plasma
from indica.converters import FluxSurfaceCoordinates
from abc import ABC
from abc import abstractmethod


class DiagnosticModel(ABC):
    name: str = ""
    bckc: dict = {}
    plasma: Plasma = None

    def set_transform(self, transform: CoordinateTransform):
        """
        Set diagnostic coordinate transform of diagnostic
        """
        self.transform = transform
        self.bckc = {}

    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        """
        set flux surface transform for flux mapping of the line of sight
        """
        if hasattr(self, "transform"):
            if hasattr(self.transform, "set_flux_transform"):
                self.transform.set_flux_transform(flux_transform)
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
            self,
            **kwargs,
    ):
        """
        Calculate and return diagnostic measured values
        """
        return self._build_bckc_dictionary()
