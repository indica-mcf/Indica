"""Coordinate systems based on volume enclosed by flux surfaces."""

from typing import Dict

from scipy.interpolate import interp1d

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .lines_of_sight import LinesOfSightTransform
from ..numpy_typing import LabeledArray


class ImpactParameterCoordinates(CoordinateTransform):
    """Class for coordinate systems based on lines-of-sight, but using the
    impact parameter (smallest flux value along a line-of-sight) as the
    label for the first coordinate.

    Parameters
    ----------
    kind
        The type of flux surface to use. Must be a valid argument for methods
        on the :py:class:`indica.equilibrium.Equilibrium` class.
    lines_of_sight
        A line-of-sight coordinate system for which this transform will get
        the impact parameters
    """

    _CONVERSION_METHODS: Dict[str, str] = {"LinesOfSightCoordinates": "_convert_to_los"}
    _INVERSE_CONVERSION_METHODS: Dict[str, str] = {
        "LinesOfSightCoordinates": "_convert_from_los"
    }

    def __init__(
        self, kind: str, lines_of_sight: LinesOfSightTransform,
    ):
        # TODO: Set up proper defaults
        super().__init__(
            lines_of_sight.default_x1,
            lines_of_sight.default_x2,
            lines_of_sight.default_R,
            lines_of_sight.default_z,
            lines_of_sight.default_t,
        )
        self.kind = kind
        self.lines_of_sight = lines_of_sight
        self.equilibrium = lines_of_sight.equilibrium
        # TODO: Actually calculate the impact parameter for each individual
        #       line of sight
        rho_min = None
        self.los_to_rhomin = interp1d(
            self.default_x1, rho_min, kind="cubic", bounds_error=True
        )
        self.rhomin_to_los = interp1d(
            self.default_x1, rho_min, kind="cubic", bounds_error=True
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
        return self.rhomin_to_los(min_rho), x2, t

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
        return self.los_to_rhomin(x1), x2, t

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
