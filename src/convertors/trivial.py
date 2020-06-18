"""Trivial class for transforming to and from R-z coordinate systems.
"""

from .abstractconvertor import CoordinateTransform, Number, Coordinates


class TrivialTransform(CoordinateTransform):
    """Class to use for transforms to/from an R-z coordinate systems. This
    is trivial, as R-z coordinates are the go-between for conversion
    to all other coordinate systems.

    Parameters
    ----------
    default_x1
        The default grid to use for the first spatial coordinate.
    default_x1
        The default grid to use for the second spatial coordinate.
    default_R
        The default grid to use for the R-coordinate when converting to this
        coordinate system.
    default_z
        The default grid to use for the z-coordinate when converting to this
        coordinate system.
    default_t
        The default grid to use for time.

    """

    def _convert_to_Rz(self, x1: Number, x2: Number, t: Number) -> Coordinates:
        """Convert from this coordinate to the R-z coordinate system.

        If an arguments is not provided then use the default grid for
        that dimension. This grid is implementation-defined, but
        should be the one which will be most commonly used to allow
        for caching.

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
        return x1, x2, t

    def _convert_from_Rz(self, R: Number, z: Number, t: Number) -> Coordinates:
        """Convert from the master coordinate system to this coordinate.

        If an arguments is not provided then return the master
        coordinates on the default grid for that dimension. This grid is
        implementation-defined, but should be the one which will be
        most commonly used to allow for caching.

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
        raise NotImplementedError(
            "{} does not implement a 'from_master' "
            "method.".format(self.__class__.__name__)
        )
        return R, z, t
