"""Coordinate systems based on volume enclosed by flux surfaces."""

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .flux_surface import FluxSurfaceCoordinates
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

    # _CONVERSION_METHODS: Dict[str, str] = {"FluxSurfaceCoordinates":"_convert_to_rho"}

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
        """Convert from this coordinate system to a flux surface coordinate system.

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
