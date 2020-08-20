"""Coordinate systems based on strength of magnetic field."""

import numpy as np
from scipy.optimize import root_scalar

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import ArrayLike


class ConvergenceError(Exception):
    """Exception thrown when a solver failes to converge."""


class MagneticCoordinates(CoordinateTransform):
    """Class for transect-like coordinate systems using total magnetic
    field strength for location along the transect. The line of sight
    is assumed to be perfectly horizontal. The second coordinate in
    this system is the vertical offset from the line of sight (the
    argument for which defaults to 0 when converting to Rz
    coordinates).

    Parameters
    ----------
    z
        The vertical position of the line of sight along which measurements are
        taken. Used as default z position.
    default_B
        The default grid to use for the magnetic field strength.
    default_R
        The default major radius to use for conversions.
    default_t
        The default grid to use for time.

    """

    def __init__(
        self, z: float, default_B: ArrayLike, default_R: ArrayLike, default_t: ArrayLike
    ):
        self.z_los = z
        super().__init__(default_B, 0.0, default_R, z, default_t)

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
        # print("Recieved by _convert_to_Rz", x1, x2, t)
        brackets = [self.default_R.min(), self.default_R.max()]

        @np.vectorize
        def find_root(B: float, x2: float, t: float):
            result = root_scalar(
                lambda R: self._convert_from_Rz(R, x2 + self.z_los, t)[0] - B,
                bracket=brackets,
                xtol=1e-6,
                rtol=1e-6,
            )
            if result.converged:
                return result.root
            raise ConvergenceError(
                f"scipy.optimize.root_scalar failed to converge with flag {result.flag}"
            )

        return find_root(x1, x2, t), x2 + self.z_los, t

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
        B, t2 = self.equilibrium.Btot(R, z, t)
        return B, z - self.z_los, t2
