"""Coordinate system representing a collection of lines of sight.
"""

from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.interpolate import interp2d

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import ArrayLike


class LinesOfSightTransform(CoordinateTransform):
    """Coordinate system for data collected along a number of lines-of-sight.

    The first coordinate in this system is an index indicating which
    line-of-site a location is on. The second coordinate ranges from 0
    to 1 (inclusive) and indicates the position of a location along
    the line-of-sight. Note that diagnostic using this coordinate
    system will usually only be indexed in the first coordinate, as
    the measurements were integrated along the line-of-sight.

    If not passed to the constructor, the default grid for converting
    from the R-z system is chosen as follows:

    - The R-grid ranges from ``min(R_start.min(), R_end.min())`` to
      ``max(R_start.max(), R_end.max())`` with ``num_points`` intervals.
    - The z-grid ranges from ``min(z_start.min(), z_end.min())`` to
      ``max(z_start.max(), z_end.max())`` with ``num_points`` intervals.

    Parameters
    ----------
    R_start
        1-D array of major radii of the start for each line-of-sight.
    z_start
        1-D array of vertical positions of the start for each line-of-sight.
    R_end
        1-D array of major radii of the end for each line-of-sight.
    z_end
        1-D array of vertical positions of the end for each line-of-sight.
    num_intervals
        The number of intervals in the default grid for the second coordinate.
        Note that there will be one more points in the grid than this.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``. Defaults to values for JET.
    default_R
        Default R-grid to use when converting from the R-z coordinate system.
    default_z
        Default z-grid to use when converting from the R-z coordinate system.

    """

    def __init__(
        self,
        R_start: np.ndarray,
        z_start: np.ndarray,
        R_end: np.ndarray,
        z_end: np.ndarray,
        num_intervals: int = 100,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
        default_R: Optional[np.ndarray] = None,
        default_z: Optional[np.ndarray] = None,
    ):
        indices = np.expand_dims(np.arange(len(R_start)), axis=1)
        x2 = np.linspace(0.0, 1.0, num_intervals + 1)
        R_default = (
            np.linspace(
                min(R_start.min(), R_end.min()),
                max(R_start.max(), R_end.max()),
                num_intervals + 1,
            )
            if default_R is None
            else default_R
        )
        z_default = (
            np.expand_dims(
                np.linspace(
                    min(z_start.min(), z_end.min()),
                    max(z_start.max(), z_end.max()),
                    num_intervals + 1,
                ),
                axis=1,
            )
            if default_z is None
            else default_z
        )
        self.R_start = R_start
        self.z_start = z_start
        self.R_end = R_end
        self.z_end = z_end
        self.index_inversion: Optional[
            Callable[[ArrayLike, ArrayLike], ArrayLike]
        ] = None
        self.x2_inversion: Optional[Callable[[ArrayLike, ArrayLike], ArrayLike]] = None
        super().__init__(indices, x2, R_default, z_default, 0)

    def _convert_to_Rz(self, x1: ArrayLike, x2: ArrayLike, t: ArrayLike) -> Coordinates:
        c = np.ceil(x1)
        f = np.floor(x1)
        Rs = (self.R_start[c] - self.R_start[f]) * (x1 - f) + self.R_start[f]
        Re = (self.R_end[c] - self.R_end[f]) * (x1 - f) + self.R_start[f]
        zs = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        ze = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_start[f]
        R = Rs + (Re - Rs) * x2
        z = zs + (ze - zs) * x2
        return R, z, t

    def _convert_from_Rz(self, R: ArrayLike, z: ArrayLike, t: ArrayLike) -> Coordinates:
        # TODO: Consider if there is some way to invert this exactly,
        # rather than rely on interpolation (which is necessarily
        # inexact, as well as computationally expensive).
        if not self.index_inversion:
            R_vals, z_vals, _ = self.convert_to_Rz()
            index_vals = self.default_x1 * np.ones_like(self.default_x2)
            x2_vals = np.ones_like(self.default_x1) * self.default_x2
            self.index_inversion = interp2d(R_vals, z_vals, index_vals, copy=False)
            self.x2_inversion = interp2d(R_vals, z_vals, x2_vals, copy=False)
        assert self.x2_inversion is not None
        x1 = self.index_inversion(R, z)
        x2 = self.x2_inversion(R, z)
        return x1, x2, t
