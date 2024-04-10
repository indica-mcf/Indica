from abc import ABC
from abc import abstractmethod

from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters.transect import TransectCoordinates
from indica.models.plasma import Plasma


class DiagnosticModel(ABC):
    name: str
    bckc: dict
    plasma: Plasma
    los_transform: LineOfSightTransform
    transect_transform: TransectCoordinates

    plasma = None

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

    def set_parameters(self, **kwargs):
        """
        Set any model kwargs
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def _build_bckc_dictionary(self):
        """
        Calculate back-calculated expected values that the diagnostic will
        be measuring. This can be directly compared to the data read in by
        the Indica Reader methods.
        """
        raise NotImplementedError(
            "{} does not implement a "
            "'__call__' method.".format(self.__class__.__name__)
        )

    # @abstractmethod
    # def __call__(self, *args, **kwargs) -> dict:
    #     """
    #     Call the model and return back-calculated values
    #     """
    #     raise NotImplementedError(
    #         "{} does not implement a "
    #         "'__call__' method.".format(self.__class__.__name__)
    #     )
