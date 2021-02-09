"""Provides an abstract interface for coordinate conversion.
"""

from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import cast
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from xarray import DataArray
from xarray import zeros_like

from ..abstract_equilibrium import AbstractEquilibrium
from ..numpy_typing import LabeledArray

Coordinates = Tuple[LabeledArray, LabeledArray]
OptionalCoordinates = Tuple[Optional[LabeledArray], Optional[LabeledArray]]


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

    Attributes
    ----------
    x1_name: str
        Name for the first spacial coordinate. May be class- or
        instance-specific.

    x2_name: str
        Name for the second spacial coordinate. May be class- or
        instance-specific.
    """

    _CONVERSION_METHODS: Dict[str, str] = {}
    _INVERSE_CONVERSION_METHODS: Dict[str, str] = {}

    equilibrium: AbstractEquilibrium
    x1_name: str
    x2_name: str

    def set_equilibrium(self, equilibrium: AbstractEquilibrium, force: bool = False):
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
            self.equilibrium = equilibrium
        elif self.equilibrium != equilibrium:
            raise EquilibriumException("Attempt to set equilibrium twice.")

    def get_converter(
        self, other: "CoordinateTransform", reverse=False
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
            return None
        return other.get_converter(self, True)

    def convert_to(
        self,
        other: "CoordinateTransform",
        x1: LabeledArray,
        x2: LabeledArray,
        t: LabeledArray,
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

        """
        if self == other:
            return x1, x2
        converter = self.get_converter(other)
        if converter:
            return converter(x1, x2, t)
        R, z = self.convert_to_Rz(x1, x2, t)
        return other.convert_from_Rz(R, z, t)

    @abstractmethod
    def convert_to_Rz(
        self,
        x1: LabeledArray,
        x2: LabeledArray,
        t: LabeledArray,
    ) -> Coordinates:
        """Convert from this coordinate to the R-z coordinate system. Each
        subclass must implement this method.

        Parameters
        ----------
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.
        t
            The time coordinate

        Returns
        -------
        R
            Major radius coordinate
        z
            Height coordinate

        """
        raise NotImplementedError(
            "{} does not implement a 'convert_to_Rz' "
            "method.".format(self.__class__.__name__)
        )

    def convert_from_Rz(
        self,
        R: LabeledArray,
        z: LabeledArray,
        t: LabeledArray,
    ) -> Coordinates:
        """Convert from the master coordinate system to this coordinate. Each
        subclass must implement this method.

        Parameters
        ----------
        R
            Major radius coordinate
        z
            Height coordinate
        t
            Time coordinate

        Returns
        -------
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.

        """
        raise NotImplementedError(
            "{} does not implement a 'convert_from_Rz' "
            "method.".format(self.__class__.__name__)
        )

    def _abstract_equals(self, other: "CoordinateTransform") -> bool:
        """Checks that default coordinate values and equilibrium objects are
        the same on two transform classes.

        """
        if not hasattr(self, "equilibrium"):
            return not hasattr(other, "equilibrium")
        elif not hasattr(other, "equilibrium"):
            return False
        else:
            return self.equilibrium == other.equilibrium

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check that two transforms are describing the same coordinate system."""
        raise NotImplementedError(
            "{} does not implement an '__eq__' method".format(self.__class__.__name__)
        )

    def distance(
        self,
        direction: str,
        x1: LabeledArray,
        x2: LabeledArray,
        t: LabeledArray,
    ) -> LabeledArray:
        """Give the distance (in physical space) from the origin along the
        specified direction.

        This is useful for when taking spatial integrals and differentials in
        that direction.

        Note that distance is calculated using Euclidean lines between
        points. As such, it will not be accurate for a curved axis.

        Parameters
        ----------
        direction : str
            Which dimension to give the distance along.
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.
        t
            The time coordinate

        Returns
        -------
        :
           Distance from the origin in the specified direction.

        """
        R, z = cast(Tuple[DataArray, DataArray], self.convert_to_Rz(x1, x2, t))
        if isinstance(R, (int, float)) or isinstance(z, (int, float)):
            raise ValueError("Arguments x1 and x2 must be xarray DataArray objects.")
        spacings = np.sqrt(R.diff(direction) ** 2 + z.diff(direction) ** 2)
        result = zeros_like(R.broadcast_like(z))
        result[{direction: slice(1, None)}] = spacings.cumsum(direction)
        return result

    def encode(self) -> str:
        """Returns a JSON representation of this object. Should be sufficient
        to recreate it identically from scratch (except for the
        equilibrium)."""
        return ""

    @staticmethod
    def decode(json: str) -> "CoordinateTransform":
        """Takes some JSON and decodes it into a CoordinateTransform object."""
        pass
