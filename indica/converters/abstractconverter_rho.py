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
import xarray as xr
from xarray import DataArray
from xarray import zeros_like

from indica.utilities import intersection
from ..equilibrium import Equilibrium
from ..numpy_typing import ArrayLike
from ..numpy_typing import Coordinates
from ..numpy_typing import LabeledArray


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

    equilibrium: Equilibrium
    x1_name: str
    x2_name: str
    x1: LabeledArray
    x2: LabeledArray
    rho: LabeledArray
    t: LabeledArray = None

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
            self.equilibrium = equilibrium
        elif self.equilibrium != equilibrium:
            raise EquilibriumException("Attempt to set equilibrium twice.")

    def check_equilibrium(self):
        if not hasattr(self, "equilibrium"):
            raise Exception("Missing equilibrium object")

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

    def convert_to_xy(
        self,
        x1: LabeledArray,
        x2: LabeledArray,
        t: LabeledArray,
    ) -> Coordinates:
        """Convert from this coordinate to the x-y coordinate system. Each
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
            "{} does not implement a 'convert_to_xy' "
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

    def convert_to_rho(self, t: LabeledArray = None) -> Coordinates:
        """Convert from spatial to flux coordinates

        Parameters
        ----------
        t
            The time coordinate

        Returns
        -------
        rho
            Flux coordinate
        theta
            time

        """
        raise NotImplementedError(
            "{} does not implement a 'convert_to_xy' "
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
            result = self.equilibrium == other.equilibrium
            result = result and self.x1_name == other.x1_name
            result = result and self.x2_name == other.x2_name
            return result

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

    def get_machine_boundaries(
        self,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
        npts: int = 1000,
    ) -> Tuple[dict, ArrayLike]:
        angles = np.linspace(0.0, 2 * np.pi, npts)
        x_wall_inner = machine_dimensions[0][0] * np.cos(angles)
        x_wall_outer = machine_dimensions[0][1] * np.cos(angles)
        y_wall_inner = machine_dimensions[0][0] * np.sin(angles)
        y_wall_outer = machine_dimensions[0][1] * np.sin(angles)
        z_wall_lower = machine_dimensions[1][0]
        z_wall_upper = machine_dimensions[1][1]

        boundaries = {
            "x_in": x_wall_inner,
            "x_out": x_wall_outer,
            "y_in": y_wall_inner,
            "y_out": y_wall_outer,
            "z_up": z_wall_upper,
            "z_low": z_wall_lower,
        }

        return boundaries, angles

    def get_equilibrium_boundaries(
        self, tplot: float, npts: int = 1000
    ) -> Tuple[dict, ArrayLike, DataArray]:

        boundaries = {}
        angles = np.linspace(0.0, 2 * np.pi, npts)
        if hasattr(self, "equilibrium"):
            angles = np.linspace(0.0, 2 * np.pi, npts)
            rho_equil = self.equilibrium.rho.sel(t=tplot, method="nearest")
            rho_equil = xr.where(rho_equil < 1.05, rho_equil, np.nan)
            core_ind = np.where(np.isfinite(rho_equil.interp(z=0)))[0]
            R_lfs = rho_equil.R[core_ind[0]].values
            R_hfs = rho_equil.R[core_ind[-1]].values
            x_plasma_inner = R_hfs * np.cos(angles)
            x_plasma_outer = R_lfs * np.cos(angles)
            y_plasma_inner = R_hfs * np.sin(angles)
            y_plasma_outer = R_lfs * np.sin(angles)

            boundaries = {
                "x_in": x_plasma_inner,
                "x_out": x_plasma_outer,
                "y_in": y_plasma_inner,
                "y_out": y_plasma_outer,
            }
        return boundaries, angles, rho_equil


def find_wall_intersections(
    origin: Tuple,
    direction: Tuple,
    machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (1.83, 3.9),
        (-1.75, 2.0),
    ),
):
    """Function for calculating "start" and "end" positions of the line-of-sight
    given the machine dimensions.

    The end coordinate is calculated by finding the intersections with the
    machine dimensions. If the intersection hits the inner column, the first
    intersection point becomes the "end" position of the line-of-sight. If the
    intersection misses the inner column, the last intersection point with the
    outer column becomes the "end" position.

    Parameters
    ----------
    origin
        A Tuple (1x3) giving the X, Y and Z origin positions of the line-of-sight
    direction
        A Tuple (1x3) giving the X, Y and Z direction of the line-of-sight
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in x-z space:
        ``((xmin, xmax), (zmin, zmax)``. Defaults to values for JET.

    Returns
    -------
    start_coordinates
        A Tuple (1x3) giving the X, Y and Z start positions of the line-of-sight
    end_coordinates
        A Tuple (1x3) giving the X, Y and Z end positions of the line-of-sight
    """

    def line(_start, _end):
        return np.linspace(_start, _end, npts)

    def extremes(_start, _end):
        return np.array(
            [_start, _end],
            dtype=float,
        )

    # Define XYZ lines for inner and outer walls
    npts = 1000
    angles = np.linspace(0.0, 2 * np.pi, npts)
    x_wall_inner = machine_dimensions[0][0] * np.cos(angles)
    y_wall_inner = machine_dimensions[0][0] * np.sin(angles)

    # Define XYZ lines for LOS from origin and direction vectors
    length = (
        np.ceil(
            np.max(
                [
                    machine_dimensions[0][1] * 2,
                    machine_dimensions[1][1] - machine_dimensions[1][0],
                ]
            )
        )
        * 5
    )
    x_start = origin[0]
    x_end = origin[0] + length * direction[0]
    y_start = origin[1]
    y_end = origin[1] + length * direction[1]
    z_start = origin[2]
    z_end = origin[2] + length * direction[2]

    # Find intersections in R, z plane
    x_line = line(x_start, x_end)
    y_line = line(y_start, y_end)
    z_line = line(z_start, z_end)
    R_line = np.sqrt(x_line**2 + y_line**2)
    indices = np.where(
        (R_line >= machine_dimensions[0][0])
        * (R_line <= machine_dimensions[0][1])
        * (z_line >= machine_dimensions[1][0])
        * (z_line <= machine_dimensions[1][1])
    )[0]
    if len(indices) > 0:
        x_start = x_line[indices][0]
        x_end = x_line[indices][-1]
        y_start = y_line[indices][0]
        y_end = y_line[indices][-1]
        z_start = z_line[indices][0]
        z_end = z_line[indices][-1]

    # Find intersections with inner wall
    xx, yx, ix, jx = intersection(
        extremes(x_start, x_end), extremes(y_start, y_end), x_wall_inner, y_wall_inner
    )
    if len(xx) > 0:
        x_line = line(x_start, x_end)
        y_line = line(y_start, y_end)
        z_line = line(z_start, z_end)
        index = []
        for i in range(len(xx)):
            index.append(
                np.argmin(np.sqrt((xx[i] - x_line) ** 2 + (yx[i] - y_line) ** 2))
            )
        indices = np.arange(np.min(index))
        x_start = x_line[indices][0]
        x_end = x_line[indices][-1]
        y_start = y_line[indices][0]
        y_end = y_line[indices][-1]
        z_start = z_line[indices][0]
        z_end = z_line[indices][-1]

    return (x_start, y_start, z_start), (x_end, y_end, z_end)
