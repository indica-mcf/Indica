"""Coordinate system for data collected on a 1-D along through the Tokamak"""

import numpy as np
from scipy.interpolate import interp1d

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import LabeledArray


class TransectCoordinates(CoordinateTransform):
    """Coordinate system for measurements taken along a 1-D line through
    the Tokamak.

    The first spatial position corresponds to an arbitrary index along
    the length of the transect. The second is the vertical offset from
    the transect. Typically, all data will fall directly on the
    transect, but it is useful to have the second coordinate so that
    the system is general and to allow conversions to other systems.

    The coordinates along the transect are the indices of the
    positions provided when intialisign the object.

    The default grid for coordinate transforms in this system is a 1-D
    array corresponding to the positions along the axis specified in
    the parmeters used to intialise the object.

    The implementation currently makes use of interpolation, so that
    it is completely general for nonuniformly spaced points and even
    for a curved line (although the latter would be only
    approximated). However, this has some computational overhead, so
    it may be changed in future.

    Parameters
    ----------
    R_positions
        1-D array of major radii of locations along the transect.
    z_positions
        1-D array of vertical position of locations along the transect.

    """

    def __init__(self, R_positions: np.ndarray, z_positions: np.ndarray):
        indices = np.arange(len(R_positions))
        self.R_vals = interp1d(
            indices, R_positions, copy=False, fill_value="extrapolate"
        )
        self.z_vals = interp1d(
            indices, z_positions, copy=False, fill_value="extrapolate"
        )
        self.invert = interp1d(
            R_positions, indices, copy=False, fill_value="extrapolate"
        )
        super().__init__(indices, 0, R_positions, z_positions, 0)

    def _convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        R = self.R_vals(x1)
        z = self.z_vals(x1) + x2
        return R, z, t

    def _convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        x1 = self.invert(R)
        x2 = z - self.z_vals(x1)
        return x1, x2, t

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._abstract_equals(other)
