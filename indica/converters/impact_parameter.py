"""Coordinate systems based on volume enclosed by flux surfaces."""

from typing import cast
from typing import Dict
from typing import Tuple

import numpy as np
from xarray import DataArray
from xarray import where

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .flux_surfaces import FluxSurfaceCoordinates
from .lines_of_sight import LinesOfSightTransform
from ..numpy_typing import LabeledArray


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
    """

    _CONVERSION_METHODS: Dict[str, str] = {"LinesOfSightTransform": "_convert_to_los"}
    _INVERSE_CONVERSION_METHODS: Dict[str, str] = {
        "LinesOfSightTransform": "_convert_from_los"
    }

    def __init__(
        self,
        lines_of_sight: LinesOfSightTransform,
        flux_surfaces: FluxSurfaceCoordinates,
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
        R, z, _ = cast(
            Tuple[DataArray, DataArray, DataArray], lines_of_sight.convert_to_Rz()
        )
        rho, _, t = cast(
            Tuple[DataArray, DataArray, DataArray], flux_surfaces.convert_from_Rz(R, z)
        )
        rho = where(rho < 0, float("nan"), rho)
        loc = rho.argmin("x2")
        theta = np.arctan2(
            z.sel(x2=0.0).mean() - np.mean(lines_of_sight._machine_dims[1]),
            R.sel(x2=0.0).mean() - np.mean(lines_of_sight._machine_dims[0]),
        )
        if np.pi / 4 <= np.abs(theta) <= 3 * np.pi / 4:
            sign = where(R.isel(x2=loc) < rmag.interp(t=t, method="nearest"), -1, 1)
        else:
            sign = where(z.isel(x2=loc) < zmag.interp(t=t, method="nearest"), -1, 1)
        self.rho_min = sign * rho.isel(x2=loc)
        super().__init__(
            self.rho_min,
            lines_of_sight.default_x2,
            lines_of_sight.default_R,
            lines_of_sight.default_z,
            lines_of_sight.default_t,
        )

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
        t
            Time coordinate (if one passed as an argument, then is just a
            pointer to that)

        """
        return (
            self.rho_min.interp(t=t, method="nearest").indica.invert_interp(
                min_rho, "index", method="linear"
            ),
            x2,
            t,
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
        t
            Time coordinate (if one passed as an argument, then is just a
            pointer to that)

        """
        return (
            self.rho_min.interp(t=t, method="nearest").indica.interp2d(
                index=x1, method="cubic"
            ),
            x2,
            t,
        )

    def _convert_to_Rz(
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
        t
            Time coordinate (if one passed as an argument, then is just a
            pointer to that)

        """
        index, position, t = self._convert_to_los(x1, x2, t)
        return self.lines_of_sight._convert_to_Rz(index, position, t)

    def _convert_from_Rz(
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
        x1, x2, t = self.lines_of_sight.convert_from_Rz(R, z, t)
        return self._convert_from_los(x1, x2, t)

    def _distance(
        self, direction: str, x1: LabeledArray, x2: LabeledArray, t: LabeledArray,
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Implementation of calculation of physical distances between points
        in this coordinate system. This accounts for potential toroidal skew of
        lines.

        """
        return self.lines_of_sight._distance(
            direction, *self._convert_to_los(x1, x2, t)
        )

    def drho(self) -> float:
        """Calculates the average difference in impact parameters between
        adjacent lines of sight."""
        drhos = np.abs(
            self.rho_min.isel(index=slice(1, None)).data
            - self.rho_min.isel(index=slice(None, -1)).data
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
