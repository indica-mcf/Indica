"""Coordinate systems based on volume enclosed by flux surfaces."""

from typing import Callable
from typing import cast
from typing import Optional

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .flux_surfaces import FluxSurfaceCoordinates
from ..numpy_typing import ArrayLike
from ..numpy_typing import LabeledArray


class EnclosedVolumeCoordinates(CoordinateTransform):
    """Class for polar coordinate systems using volume enclosed by flux
    surfaces as the radial coordinate.

    Parameters
    ----------
    flux_suraces
        A flux surface coordinate transform, the surfaces of which will be used
        to calculate enclosed volumes.

    """

    x2_name = "theta"

    def __init__(
        self,
        flux_surfaces: FluxSurfaceCoordinates,
    ):
        self.flux_transform = flux_surfaces
        self.equilibrium = flux_surfaces.equilibrium
        self.x1_name = flux_surfaces.x1_name + "_enclosed_volume"

    def get_converter(
        self, other: CoordinateTransform, reverse=False
    ) -> Optional[Callable[[LabeledArray, LabeledArray, LabeledArray], Coordinates]]:
        """Checks if there is a shortcut to convert between these coordiantes,
        returning it if so. This can sometimes save the step of
        converting to (R, z) coordinates first.

        Parameters
        ----------
        other
            The other transform whose coordinate system you want to convert to.
        reverse
            If True, try to return a function which converts _from_ ``other``
            to this coordinate system.

        Returns
        -------
        :
            If a shortcut function is available, return it. Otherwise, None.

        Note
        ----
        Implementations should call ``other.get_converter(self, reverse=True``. For
        obvious reasons, however, they should **only do this when
        ``reverse == False``**.

        """
        if reverse:
            if other == self.flux_transform:
                return cast(FluxSurfaceCoordinates, other)._convert_to_vol
            else:
                return None
        if other == self.flux_transform:
            return self._convert_to_rho
        else:
            return other.get_converter(self, True)

    def _convert_to_rho(
        self, volume: ArrayLike, theta: ArrayLike, t: ArrayLike
    ) -> Coordinates:
        """Convert from this coordinate system to a flux surface coordinate
        system.

        Parameters
        ----------
        volume
            The first spatial coordinate in this system.
        theta
            The second spatial coordinate in this system.
        t
            The time coordinate (if there is one, otherwise ``None``)

        Returns
        -------
        rho
            Flux surface coordinate
        theta
            Poloidal angle coordinate

        """
        rho, t = self.equilibrium.invert_enclosed_volume(
            volume, t, self.flux_transform.flux_kind
        )
        return rho, theta

    def convert_to_Rz(self, x1: ArrayLike, x2: ArrayLike, t: ArrayLike) -> Coordinates:
        """Convert from this coordinate to the R-z coordinate system.

        Parameters
        ----------
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.
        t
            The time coordinate (if there is one, otherwise ``None``)

        Returns
        -------
        R
            Major radius coordinate
        z
            Height coordinate

        """
        rho, theta = self._convert_to_rho(x1, x2, t)
        return self.flux_transform.convert_to_Rz(rho, theta, t)

    def convert_from_Rz(self, R: ArrayLike, z: ArrayLike, t: ArrayLike) -> Coordinates:
        """Convert from the master coordinate system to this coordinate.

        Parameters
        ----------
        R
            Major radius coordinate
        z
            Height coordinate
        t
            Time coordinate)

        Returns
        -------
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.

        """
        rho, theta = self.flux_transform.convert_from_Rz(R, z, t)
        return self.flux_transform._convert_to_vol(rho, theta, t)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        return result and self.flux_transform == other.flux_transform
