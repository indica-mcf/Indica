"""Coordinate system for data collected on a 1-D along through the Tokamak"""

import numpy as np
from scipy.interpolate import interp1d
from xarray import DataArray
from xarray import Dataset
from xarray import Variable

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

    def __init__(self, R_positions: LabeledArray, z_positions: LabeledArray):
        assert isinstance(R_positions, (DataArray, Dataset, Variable))
        indices = DataArray(np.arange(len(R_positions)))
        self.R_vals = interp1d(
            indices, R_positions, copy=False, fill_value="extrapolate"
        )
        self.z_vals = interp1d(
            indices, z_positions, copy=False, fill_value="extrapolate"
        )
        self.invert = interp1d(
            R_positions, indices, copy=False, fill_value="extrapolate"
        )
        if isinstance(z_positions, DataArray):
            if R_positions.dims != z_positions.dims:
                raise ValueError(
                    "R_positions and z_positions must have the same dimensiosn."
                )

        self.x1_name = R_positions.dims[0]
        self.x2_name = self.x1_name + "_z_offset"

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
        R = self.R_vals(x1)
        z = self.z_vals(x1) + x2
        return R, z

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
            Time coordinate)

        Returns
        -------
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.

        """
        x1 = self.invert(R)
        x2 = z - self.z_vals(x1)
        return x1, x2

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._abstract_equals(other)
