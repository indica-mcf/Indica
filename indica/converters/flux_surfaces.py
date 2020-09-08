"""Class to handle conversions to and from flux surface coordinates."""

from typing import Dict

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import ArrayLike


class FluxSurfaceCoordinates(CoordinateTransform):
    """Class for polar coordinate systems using flux surfaces
    for the radial coordinate.

    Parameters
    ----------
    kind
        The type of flux surface to use. Must be a valid argument for methods
        on the :py:class:`indica.equilibrium.Equilibrium` class.
    default_rho
        The default grid to use for the flux surface.
    default_theta
        The default grid to use for the angle in the polar coordinates.
    default_R
        The default grid to use for the R-coordinate when converting to this
        coordinate system.
    default_z
        The default grid to use for the z-coordinate when converting to this
        coordinate system.
    default_t
        The default grid to use for time.

    """

    _CONVERSION_METHODS: Dict[str, str] = {
        "EnclosedVolumeCoordinates": "_convert_to_vol"
    }

    def __init__(
        self,
        kind: str,
        default_rho: ArrayLike,
        default_theta: ArrayLike,
        default_R: ArrayLike,
        default_z: ArrayLike,
        default_t: ArrayLike,
    ):
        self.flux_kind = kind
        super().__init__(default_rho, default_theta, default_R, default_z, default_t)

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
        return self.equilibrium.spatial_coords(x1, x2, t, self.flux_kind)

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
        return self.equilibrium.flux_coords(R, z, t, self.flux_kind)

    def _convert_to_vol(
        self, rho: ArrayLike, theta: ArrayLike, t: ArrayLike
    ) -> Coordinates:
        """Convert from this coordinate system to one using volume enclosed by
        the flux surfaces as a coordinate.

        Parameters
        ----------
        rho
            The first spatial coordinate in this system.
        theta
            The second spatial coordinate in this system.
        t
            The time coordinate (if there is one, otherwise ``None``)

        Returns
        -------
        vol
            Volume enclosed by the flux surface rho.
        theta
            Poloidal angle coordinate
        t
            Time coordinate (if one passed as an argument, then is just a
            pointer to that)

        """
        vol, _ = self.equilibrium.enclosed_volume(rho, t, self.flux_kind)
        return vol, theta, t
