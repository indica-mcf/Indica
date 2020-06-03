from abc import ABC, abstractmethod
from functools import wraps
from numbers import Number as Scalar
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

Number = Union[np.ndarray, Scalar]
OptNumber = Optional[Number]
Coordinates = (Number, Number, Number)


def check_and_cache_defaults(default1: str, default2: str, default3: str):
    """Decorator to handle checking whether to use default coordinates and to 
    handle caching of the default result.

    Parameters
    ----------
    default_1
        Name of the attribute of the
        :py:class:`abstractconvertor.CoordinateTransform` to provide the
        default for the first argument of the decorated method.

    default_2
        Name of the attribute of the
        :py:class:`abstractconvertor.CoordinateTransform` to provide the
        default for the second argument of the decorated method.

    default_3
        Name of the attribute of the
        :py:class:`abstractconvertor.CoordinateTransform` to provide the
        default for the second argument of the decorated method.

    """
    def check_decorator(method):
        cached_val = None

        @wraps(method)
        def new_method(self, x1, x2, t):
            global cached_val
            use_cached = True
            if x1 is None:
                x1 = getattr(self, default1)
            else:
                use_cached = False
            if x2 is None:
                x2 = getattr(self, default2)
            else:
                use_cached = False
            if t is None:
                t = getattr(self, default3)
            else:
                use_cached = False
            if use_cached:
                if cached_val is None:
                    cached_val = method(self, x1, x2, t)
                return cached_val
            else:
                return method(self, x1, x2, t)

        return new_method

    return check_decorator


class CoordinateTransform(ABC):
    """Class for converting between different coordinate systems. This is
    an abstract base class; each coordinate system should provide its own
    implementation.

    Subclasses should allow each instance to have a "default grid" on
    which to calculate results. This can be cached for efficient
    retrieval. There should probably be some sort of default grid for
    the master coordinate system as well, although how to go about
    specifying this without relying on global variables is unclear.

    Note that not all coordinate systems will have an actual x2
    dimension (for example, the lines-of-site for soft X-ray
    data). However, 2 coordinates are still needed to map to the
    global coordinate system. Therefore, x2 is treated as a
    "pseudo-coordinate",in these cases, with values between 0 and 1
    specifying the position along the grid-line for x1. 0 is the start
    and 1 is the end (possibly overlapping, if the coordinate system is
    periodic).

    """

    _CONVERSION_METHODS: Dict[str, str] = {}

    def __init__(self, default_x, default_y, default_R, default_z, default_t):
        self.default_x = default_x
        self.default_y = default_y
        self.default_t = default_t
        self.default_R = default_R
        self.default_z = default_z
        self.default_distance = [None, None]
        self.default_to_Rz = None
        self.default_from_Rz = None

    @abstractmethod
    def set_equilibrium(self, equilibrium, force=False):
        """Initialise the object using a set of equilibrium data.

        If it has already been initialised with the same equilibrium
        data then do nothing. If already initialised with a different
        equilibrium, throw an error unless ``force == True``.

        Parameters
        ----------
        equilibrium
            A set of equilibrium data with which to calculate coordinate
            transforms.
        force : bool
            If true, re-initialise the transform if provided with a new set of
            equilibrium data.

        """
        raise NotImplementedError("{} does not implement a 'init' "
                                  "method.".format(self.__class__.__name__))

    def convert_to(self, other: "CoordinateTransform", x1: OptNumber = None,
                   x2: OptNumber = None, t: OptNumber = None) -> Coordinates:
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
        other_name = other.__class__.__name__
        if other_name in self.CONVERSION_METHODS:
            convertor = getattr(self, self.CONVERSION_METHODS[other_name])
            return convertor(x1, x2, t)
        else:
            R, z, t = self._convert_to_Rz(x1, x2, t)
            return other._convert_from_Rz(R, z, t)

    @abstractmethod
    def _convert_to_Rz(self, x1: OptNumber = None, x2: OptNumber = None,
                       t: OptNumber = None) -> Coordinates:
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
        raise NotImplementedError("{} does not implement a 'to_master' "
                                  "method.".format(self.__class__.__name__))

    @abstractmethod
    def _convert_from_Rz(self, R: OptNumber = None, z: OptNumber = None,
                         t: OptNumber = None) -> Coordinates:
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
        raise NotImplementedError("{} does not implement a 'from_master' "
                                  "method.".format(self.__class__.__name__))

    def distance(self, direction: int, x1: OptNumber = None, x2: OptNumber =
                 None, t: OptNumber = None) -> (Number, Number):
        """Give the distance (in physical space) from the origin in the
        specified direction.

        This is useful for when taking spatial integrals and differentials in
        that direction.

        If an arguments is not provided then return the master
        distancees on the default grid for that dimension. This grid is
        implementation-defined, but should be the one which will be
        most commonly used to allow for caching.

        Parameters
        ----------
        distance : {1, 2}
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
            R, z, t = self._convert_to_rz(x1, x2, t)
            slc = [slice(None)] * R.ndim
            slc[direction - 1] = slice(0, 1)
            R0 = R[slc]
            z0 = z[slc]
            return np.sqrt((R - R0)**2 + (z - z0)**2), t

        use_cached = True
        if x1 is None:
            x1 = self.default_x1
        else:
            use_cached = False
        if x1 is None:
            x1 = self.default_x1
        else:
            use_cached = False
        if t is None:
            t = self.default_t
        else:
            use_cached = False
        if use_cached:
            if self.default_distance[direction - 1] is None:
                self.default_distance[direction - 1] = calc_distance(direction,
                                                                     x1, x2, t)
            return self.default_distance[direction - 1]
        else:
            return calc_distance(direction, x1, x2, t)
