"""Coordinate systems based on strength of magnetic field."""

from typing import Callable
from typing import Tuple

import numpy as np
from scipy.optimize import root_scalar

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import LabeledArray


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
        self,
        z: float,
        default_B: LabeledArray,
        default_R: LabeledArray,
        default_t: LabeledArray,
    ):
        self.z_los = z
        super().__init__(default_B, 0.0, default_R, z, np.expand_dims(default_t, 1))

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
        # print("Recieved by _convert_to_Rz", x1, x2, t)
        if isinstance(self.default_R, (int, float)):
            left = self.default_R
            right = self.default_R
        else:
            left = self.default_R.min()
            right = self.default_R.max()

        @np.vectorize
        def find_root(B: float, x2: float, t: float) -> float:
            def func(R: float) -> float:
                return self._convert_from_Rz(R, x2 + self.z_los, t)[0] - B

            brackets = find_brackets(left, right, func)
            result = root_scalar(func, bracket=brackets, xtol=1e-8, rtol=1e-6,)
            if result.converged:
                return result.root
            raise ConvergenceError(
                f"scipy.optimize.root_scalar failed to converge with flag {result.flag}"
            )

        return find_root(x1, x2, t), x2 + self.z_los, t

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
        B, t2 = self.equilibrium.Btot(R, z, t)
        return B, z - self.z_los, t2

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        return result and np.all(self.z_los == other.z_los)


def find_brackets(
    left: float, right: float, function: Callable[[float], float]
) -> Tuple[float, float]:
    """Find suitable brackets around a root of the function. Relies in standard
    shape of total magnetic field strength.

    """
    fleft = function(left)
    fright = function(right)
    if fleft * fright <= 0.0:
        return float(left), float(right)
    if fleft < 0.0:
        if left > 1e-1:
            left /= 1.5
        elif left > 0.0:
            left = 0.0
        elif left == 0.0:
            left = -0.1
        else:
            left *= 1.5
        return find_brackets(left, right, function)
    assert fright > 0.0
    # Calculate new right bracket
    if left < -1e-1:
        left /= 1.5
    elif left < 0.0:
        left = 0.0
    elif left == 0.0:
        left = 0.1
    else:
        left *= 1.5
    return find_brackets(left, right, function)
