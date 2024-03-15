"""Trivial class for transforming to and from R-z coordinate systems."""

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import LabeledArray


class TrivialTransform(CoordinateTransform):
    """
    Class to use for transforms to/from an R-z coordinate systems. This
    is trivial, as R-z coordinates are the go-between for conversion
    to all other coordinate systems.

    """

    x1_name = "R"
    x2_name = "z"

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """
        Convert from this coordinate to the R-z coordinate system.

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
        return x1, x2

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """
        Convert from the master coordinate system to this coordinate.

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
        return R, z

    def __eq__(self, other: object) -> bool:
        """
        Check that two transforms are describing the same coordinate system.

        Parameters
        ----------
        other
            CoordinateTransform object to compare equality against.

        Returns
        -------
        bool
            Whether objects are the same.

        """
        if not isinstance(other, self.__class__):
            return False
        return self._abstract_equals(other)
