"""Coordinate systems based on strength of magnetic field."""

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import ArrayLike


# TODO: Determine correct behaviour from Marco (seem to assume uniform
# on flux surface, but which?)


class MagneticCoordinates(CoordinateTransform):
    """Class for polar coordinate systems using total magnetic field strength
    for the radial coordinate.

    Parameters
    ----------
    z
        The vertical position of the line of sight along which measurements are
        taken.
    default_B
        The default grid to use for the magnetic field strength.
    default_t
        The default grid to use for time.

    """

    def __init__(self, z: float, default_B: ArrayLike, default_t: ArrayLike):
        pass

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
