"""Provides an abstract interface for coordinate conversion.
"""

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from ..equilibrium import Equilibrium
from ..numpy_typing import ArrayLike

Coordinates = Tuple[ArrayLike, ArrayLike, ArrayLike]


class EquilibriumException(Exception):
    """Exception raised if a converter object's equilibrium object is set
    twice."""


class CoordinateTransform(ABC):
    """Class for converting between different coordinate systems. This is
    an abstract base class; each coordinate system should provide its own
    implementation.

    Subclasses should allow each instance to have a "default grid" on
    which to calculate results. This can be cached for efficient
    retrieval.

    Note that not all coordinate systems will have an actual x2
    dimension (for example, the lines-of-site for soft X-ray
    data). However, 2 coordinates are still needed to map to the
    global coordinate system. Therefore, x2 is treated as a
    "pseudo-coordinate",in these cases, with values between 0 and 1
    specifying the position along the grid-line for x1. 0 is the start
    and 1 is the end (possibly overlapping, if the coordinate system is
    periodic).

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

    _CONVERSION_METHODS: Dict[str, str] = {}

    def __init__(
        self,
        default_x1: ArrayLike,
        default_x2: ArrayLike,
        default_R: ArrayLike,
        default_z: ArrayLike,
        default_t: ArrayLike,
    ):
        self.default_x1 = default_x1
        self.default_x2 = default_x2
        self.default_R = default_R
        self.default_z = default_z
        self.default_t = default_t
        self.default_distance: List[Tuple[ArrayLike, ArrayLike]] = [
            (None, None),
            (None, None),
        ]
        self.default_to_Rz: Coordinates = (None, None, None)
        self.default_from_Rz: Coordinates = (None, None, None)
        self.equilibrium: Equilibrium

    def set_equilibrium(self, equilibrium: Equilibrium, force: bool = False):
        """Initialise the object using a set of equilibrium data.

        If it has already been initialised with the same equilibrium
        data then do nothing. If already initialised with a different
        equilibrium, throw an
        :py:class:`abstractconverter.EquilibriumException` unless
        ``force == True``.

        Parameters
        ----------
        equilibrium
            A set of equilibrium data with which to calculate coordinate
            transforms.
        force : bool
            If true, re-initialise the transform if provided with a new set of
            equilibrium data.

        """
        if not hasattr(self, "equilibrium") or force:
            self.default_to_Rz = (None, None, None)
            self.default_from_Rz = (None, None, None)
            self.default_distance = [
                (None, None),
                (None, None),
            ]
            self.equilibrium = equilibrium
        elif self.equilibrium != equilibrium:
            raise EquilibriumException("Attempt to set equilibrium twice.")

    def convert_to(
        self,
        other: "CoordinateTransform",
        x1: Optional[ArrayLike] = None,
        x2: Optional[ArrayLike] = None,
        t: Optional[ArrayLike] = None,
    ) -> Coordinates:
        """General routine to map coordinates from this system to those used
        in ``other``. Array broadcasting will be performed as necessary.

        If this transform class provides a specialised method for
        doing this (specified in :py:attr:`_CONVERSION_METHODS`) then that is
        used. Otherwise, the coordinates are converted to R-z using
        :py:meth:`_convert_to_Rz` and then converted to the other coordinate
        system using :py:attr:`_convert_from_Rz`.

        Parameters
        ----------
        other
            The coordinate system to convert to.
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.
        t
            The time coordinate (if there is one, otherwise ``None``)

        Returns
        -------
        x1
            The first spatial coordinate in the ``other`` system.
        x2
            The second spatial coordinate in the ``other`` system.
        t
            The time coordinate (if one pass as an argument then is just a
            pointer to that)

        """
        # TODO: cache all results for default arguments
        other_name = other.__class__.__name__
        if other_name in self._CONVERSION_METHODS:
            converter = getattr(self, self._CONVERSION_METHODS[other_name])
            return converter(x1, x2, t)
        else:
            R, z, t = self.convert_to_Rz(x1, x2, t)
            return other.convert_from_Rz(R, z, t)

    def convert_to_Rz(
        self,
        x1: Optional[ArrayLike] = None,
        x2: Optional[ArrayLike] = None,
        t: Optional[ArrayLike] = None,
    ) -> Coordinates:
        """Convert from this coordinate to the R-z coordinate system.

        If an arguments is not provided then use the default grid for
        that dimension. This grid is implementation-defined, but
        should be the one which will be most commonly used to allow
        for efficient caching of the result.

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
        use_cached = True
        if x1 is None:
            x1 = self.default_x1
        else:
            use_cached = False
        if x2 is None:
            x2 = self.default_x2
        else:
            use_cached = False
        if t is None:
            t = self.default_t
        else:
            use_cached = False
        if use_cached:
            if self.default_to_Rz[0] is None:
                self.default_to_Rz = self._convert_to_Rz(x1, x2, t)
            return self.default_to_Rz
        else:
            return self._convert_to_Rz(x1, x2, t)

    @abstractmethod
    def _convert_to_Rz(self, x1: ArrayLike, x2: ArrayLike, t: ArrayLike) -> Coordinates:
        """Implementation of conversion to the R-z coordinate system, without
        caching or default argument values.
        """
        raise NotImplementedError(
            "{} does not implement a 'to_master' "
            "method.".format(self.__class__.__name__)
        )

    def convert_from_Rz(
        self,
        R: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
        t: Optional[ArrayLike] = None,
    ) -> Coordinates:
        """Convert from the master coordinate system to this coordinate.

        If an arguments is not provided then return the master
        coordinates on the default grid for that dimension. This grid is
        implementation-defined, but should be the one which will be
        most commonly used to allow for efficient caching.

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
        use_cached = True
        if R is None:
            R = self.default_R
        else:
            use_cached = False
        if z is None:
            z = self.default_z
        else:
            use_cached = False
        if t is None:
            t = self.default_t
        else:
            use_cached = False
        if use_cached:
            if self.default_from_Rz[0] is None:
                self.default_from_Rz = self._convert_from_Rz(R, z, t)
            return self.default_from_Rz
        else:
            return self._convert_from_Rz(R, z, t)

    @abstractmethod
    def _convert_from_Rz(self, R: ArrayLike, z: ArrayLike, t: ArrayLike) -> Coordinates:
        """Implementation of conversion from the R-z coordinate system, without
        caching or default argument values.
        """
        raise NotImplementedError(
            "{} does not implement a 'from_master' "
            "method.".format(self.__class__.__name__)
        )

    def _abstract_equals(self, other: "CoordinateTransform") -> bool:
        """Checks that default coordinate values and equilibrium objects are
        the same on two transform classes.

        """
        if not isinstance(other, self.__class__):
            return False
        if not hasattr(self, "equilibrium"):
            result = not hasattr(other, "equilibrium")
        elif not hasattr(other, "equilibrium"):
            result = False
        else:
            result = self.equilibrium == other.equilibrium
        result = result and np.all(self.default_R == other.default_R)
        result = result and np.all(self.default_z == other.default_z)
        result = result and np.all(self.default_x1 == other.default_x1)
        result = result and np.all(self.default_x2 == other.default_x2)
        result = result and np.all(self.default_t == other.default_t)
        return result

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check that two transforms are describing the same coordinate system."""
        raise NotImplementedError(
            "{} does not implement an '__eq__' method".format(self.__class__.__name__)
        )

    def distance(
        self,
        direction: int,
        x1: Optional[ArrayLike] = None,
        x2: Optional[ArrayLike] = None,
        t: Optional[ArrayLike] = None,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Give the distance (in physical space) from the origin in the
        specified direction.

        This is useful for when taking spatial integrals and differentials in
        that direction.

        If an arguments is not provided then return the master
        distancees on the default grid for that dimension. This grid is
        implementation-defined, but should be the one which will be
        most commonly used to allow for caching.

        Note that distance is calculated using Euclidean lines between
        points. As such, it will not be accurate for a curved axis.

        Parameters
        ----------
        direction : {1, 2}
            Which direction (x1 or x2) to give the distance along.
        x1 : array_like
            The first spatial coordinate in this system.
        x2 : array_like
            The second spatial coordinate in this system.
        t : array_like or None
            The time coordinate (if there is one, otherwise ``None``)

        Returns
        -------
        distance : ndarray
           Distance from the origin in the specified direction.
        t
            The time coordinate (if one pass as an argument then is just a
            pointer to that)

        """

        def calc_distance(direction, x1, x2, t):
            R, z, t = self._convert_to_Rz(x1, x2, t)
            slc1 = [slice(None)] * R.ndim
            slc1[direction] = slice(0, -1)
            slc1 = tuple(slc1)
            slc2 = [slice(None)] * R.ndim
            slc2[direction] = slice(1, None)
            slc2 = tuple(slc2)
            spacings = np.sqrt((R[slc2] - R[slc1]) ** 2 + (z[slc2] - z[slc1]) ** 2)
            result = np.zeros(np.broadcast(R, z).shape)
            return np.cumsum(spacings, direction, out=result[slc2]), t

        use_cached = True
        if x1 is None:
            x1 = self.default_x1
        else:
            use_cached = False
        if x2 is None:
            x2 = self.default_x2
        else:
            use_cached = False
        if t is None:
            t = self.default_t
        else:
            use_cached = False
        if use_cached:
            if self.default_distance[direction - 1][0] is None:
                self.default_distance[direction - 1] = calc_distance(
                    direction, x1, x2, t
                )
            return self.default_distance[direction - 1]
        else:
            return calc_distance(direction, x1, x2, t)

    def encode(self) -> str:
        """Returns a JSON representation of this object. Should be sufficient
        to recreate it identically from scratch (except for the
        equilibrium)."""
        pass

    @staticmethod
    def decode(json: str) -> "CoordinateTransform":
        """Takes some JSON and decodes it into a CoordinateTransform object.

        """
        pass
