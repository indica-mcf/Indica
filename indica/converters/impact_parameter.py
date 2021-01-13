"""Coordinate systems based on volume enclosed by flux surfaces."""

from typing import Callable
from typing import cast
from typing import Optional
from typing import Tuple

import numpy as np
from xarray import DataArray
from xarray import where

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .flux_surfaces import FluxSurfaceCoordinates
from .lines_of_sight import LinesOfSightTransform
from ..numpy_typing import LabeledArray
from ..utilities import coord_array


class ImpactParameterCoordinates(CoordinateTransform):
    """Class for coordinate systems based on lines-of-sight, but using the
    impact parameter (smallest flux value along a line-of-sight) as the
    label for the first coordinate.

    Parameters
    ----------
    lines_of_sight
        A line-of-sight coordinate system for which this transform will get
        the impact parameters.
    flux_surfaces
        The flux surface coordinate system from which flux values will be used
        for the impact parameters.
    num_intervals
        The number of points along the line of sight at which to evaulate the
        flux surface value.
    """

    def __init__(
        self,
        lines_of_sight: LinesOfSightTransform,
        flux_surfaces: FluxSurfaceCoordinates,
        num_intervals: int = 100,
    ):
        # TODO: Set up proper defaults
        self.lines_of_sight = lines_of_sight
        self.flux_surfaces = flux_surfaces
        if not hasattr(flux_surfaces, "equilibrium"):
            raise ValueError(
                "Flux surface coordinate system must have an equilibrium set."
            )
        self.equilibrium = flux_surfaces.equilibrium
        if (
            hasattr(lines_of_sight, "equilibrium")
            and lines_of_sight.equilibrium is not flux_surfaces.equilibrium
        ):
            raise ValueError(
                "Two coordinate systems must have the same equilibrium object."
            )
        rmag = self.equilibrium.rmag
        zmag = self.equilibrium.zmag
        self.x1_name = lines_of_sight.x1_name[:-6] + flux_surfaces.x1_name
        self.x2_name = lines_of_sight.x2_name
        R, z = cast(
            Tuple[DataArray, DataArray],
            lines_of_sight.convert_to_Rz(
                coord_array(np.arange(len(lines_of_sight.R_start)), self.x1_name),
                coord_array(np.linspace(0.0, 1.0, num_intervals + 1), self.x2_name),
                0.0,
            ),
        )
        rho, _ = cast(Tuple[DataArray, DataArray], flux_surfaces.convert_from_Rz(R, z))
        rho = where(rho < 0, float("nan"), rho)
        t = rho.coords["t"]
        loc = rho.argmin(self.x2_name)
        theta = np.arctan2(
            z.sel({self.x2_name: 0.0}).mean()
            - np.mean(lines_of_sight._machine_dims[1]),
            R.sel({self.x2_name: 0.0}).mean()
            - np.mean(lines_of_sight._machine_dims[0]),
        )
        if np.pi / 4 <= np.abs(theta) <= 3 * np.pi / 4:
            sign = where(
                R.isel({self.x2_name: loc}) < rmag.interp(t=t, method="nearest"), -1, 1
            )
        else:
            sign = where(
                z.isel({self.x2_name: loc}) < zmag.interp(t=t, method="nearest"), -1, 1
            )
        self.rho_min = sign * rho.isel({self.x2_name: loc})

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
            if other == self.lines_of_sight:
                return self._convert_from_los
            else:
                return None
        if other == self.lines_of_sight:
            return self._convert_to_los
        else:
            return other.get_converter(self, True)

    def _convert_to_los(
        self, min_rho: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """Convert from this coordinate system to a line-of-sight coordinate system.

        Parameters
        ----------
        min_rho
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.
        t
            The time coordinate

        Returns
        -------
        los
            Index of the line of sight which that impact parameter corresponds
            to.
        theta
            Position along the line of sight.

        """
        # Cubic splines aren't guaranteed to be monotonicuse linear.
        # TODO: Find a better spline that I can ensure is monotonic
        return (
            self.rho_min.interp(t=t, method="nearest").indica.invert_interp(
                min_rho, self.x1_name, method="linear"
            ),
            x2,
        )

    def _convert_from_los(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """Converts from line of sight coordinates to this coordinate system.

        Parameters
        ----------
        x1
            The index for the line of sight.
        x2
            Position along the line of sight.
        t
            The time coordinate

        Returns
        -------
        min_rho
            Lowest flux surface value the line of sight touches.
        x2
            Position along the line of sight.

        """
        # Cubic splines aren't guaranteed to be monotonic, so use linear
        # TODO: Find a better spline that I can ensure is monotonic
        return (
            self.rho_min.interp(t=t, method="nearest").indica.interp2d(
                {self.x1_name: x1}, method="linear"
            ),
            x2,
        )

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
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

        """
        index, position = self._convert_to_los(x1, x2, t)
        return self.lines_of_sight.convert_to_Rz(index, position, t)

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """Convert from the master coordinate system to this coordinate.

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
        x1, x2 = self.lines_of_sight.convert_from_Rz(R, z, t)
        return self._convert_from_los(x1, x2, t)

    def distance(
        self,
        direction: str,
        x1: LabeledArray,
        x2: LabeledArray,
        t: LabeledArray,
    ) -> LabeledArray:
        """Implementation of calculation of physical distances between points
        in this coordinate system. This accounts for potential toroidal skew of
        lines.

        """
        return self.lines_of_sight.distance(
            direction, *self._convert_to_los(x1, x2, t), t
        )

    def drho(self) -> float:
        """Calculates the average difference in impact parameters between
        adjacent lines of sight."""
        drhos = np.abs(
            self.rho_min.isel({self.x1_name: slice(1, None)}).data
            - self.rho_min.isel({self.x1_name: slice(None, -1)}).data
        )
        return np.mean(drhos)

    def rhomax(self) -> float:
        """Returns the time-averaged maximum impact parameter on the low flux
        surface.

        """
        return self.rho_min.mean("t").max()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        result = result and self.flux_surfaces == other.flux_surfaces
        return result and self.lines_of_sight == other.lines_of_sight
