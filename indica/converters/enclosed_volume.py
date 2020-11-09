"""Coordinate systems based on volume enclosed by flux surfaces."""

from typing import Dict

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .flux_surfaces import FluxSurfaceCoordinates
from ..numpy_typing import ArrayLike


class EnclosedVolumeCoordinates(CoordinateTransform):
    """Class for polar coordinate systems using volume enclosed by flux
    surfaces as the radial coordinate.

    Parameters
    ----------
    flux_suraces
        A flux surface coordinate transform, the surfaces of which will be used
        to calculate enclosed volumes.

    """

    _CONVERSION_METHODS: Dict[str, str] = {"FluxSurfaceCoordinates": "_convert_to_rho"}

    def __init__(
        self, flux_surfaces: FluxSurfaceCoordinates,
    ):
        # TODO: Set up proper defaults
        super().__init__(
            flux_surfaces.default_x1,
            flux_surfaces.default_x2,
            flux_surfaces.default_R,
            flux_surfaces.default_z,
            flux_surfaces.default_t,
        )
        self.flux_transform = flux_surfaces
        self.equilibrium = flux_surfaces.equilibrium

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
        t
            Time coordinate (if one passed as an argument, then is just a
            pointer to that)

        """
        rho, t = self.equilibrium.invert_enclosed_volume(
            volume, t, self.flux_transform.flux_kind
        )
        return rho, theta, t

    def _convert_to_Rz(self, x1: ArrayLike, x2: ArrayLike, t: ArrayLike) -> Coordinates:
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
        t
            Time coordinate (if one passed as an argument, then is just a
            pointer to that)

        """
        rho, theta, t = self._convert_to_rho(x1, x2, t)
        return self.flux_transform._convert_to_Rz(rho, theta, t)

    def _convert_from_Rz(self, R: ArrayLike, z: ArrayLike, t: ArrayLike) -> Coordinates:
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
        t
            The time coordinate (if one pass as an argument then is just a
            pointer to that)

        """
        rho, theta, t = self.flux_transform._convert_from_Rz(R, z, t)
        return self.flux_transform._convert_to_vol(rho, theta, t)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        return result and self.flux_transform == other.flux_transform
