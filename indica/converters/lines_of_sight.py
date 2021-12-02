"""Coordinate system representing a collection of lines of sight.
"""

from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.optimize import root
from xarray import DataArray
from xarray import zeros_like

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import LabeledArray


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
    T_start
        1-D array of toroidal offset for the start of each line-of-sight.
    R_end
        1-D array of major radii of the end for each line-of-sight.
    z_end
        1-D array of vertical positions of the end for each line-of-sight.
    T_end
        1-D array of toroidal offset for the end of each line-of-sight.
    name
        The name to refer to this coordinate system by, typically taken
        from the instrument it describes.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``. Defaults to values for JET.

    """

    def __init__(
        self,
        x_start: np.ndarray,
        z_start: np.ndarray,
        y_start: np.ndarray,
        x_end: np.ndarray,
        z_end: np.ndarray,
        y_end: np.ndarray,
        name: str,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
    ):
        lengths = _get_wall_intersection_distances(
            x_start, z_start, y_start, x_end, z_end, y_end, machine_dimensions
        )
        new_length = max(lengths)
        los_lengths = np.sqrt(
            (x_start - x_end) ** 2 + (z_start - z_end) ** 2 + (y_start - y_end) ** 2
        )
        factor = new_length / los_lengths
        self.x_start = DataArray(x_start)
        self.z_start = DataArray(z_start)
        self.y_start = DataArray(y_start)
        self._original_x_end = DataArray(x_end)
        self._original_z_end = DataArray(z_end)
        self._original_y_end = DataArray(y_end)
        self._machine_dims = machine_dimensions
        self.x_end = DataArray(x_start + factor * (x_end - x_start))
        self.z_end = DataArray(z_start + factor * (z_end - z_start))
        self.y_end = DataArray(y_start + factor * (y_end - y_start))
        self.index_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None
        self.x2_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None
        self.x1_name = name + "_coords"
        self.x2_name = name + "_los_position"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        result = result and np.all(self.x_start == other.x_start)
        result = result and np.all(self.z_start == other.z_start)
        result = result and np.all(self.y_start == other.y_start)
        result = result and np.all(self.x_end == other.x_end)
        result = result and np.all(self.z_end == other.z_end)
        result = result and np.all(self.y_end == other.y_end)
        result = result and self._machine_dims == other._machine_dims
        return result

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        R_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        R_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        T_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        T_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        R_0 = R_s + (R_e - R_s) * x2
        T_0 = T_s + (T_e - T_s) * x2
        z = z_s + (z_e - z_s) * x2
        return np.sign(R_0) * np.sqrt(R_0 ** 2 + T_0 ** 2), z

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        def jacobian(x):
            x1 = x[0]
            x2 = x[1]
            c = np.ceil(x1).astype(int)
            f = np.floor(x1).astype(int)
            R_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
            R_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
            z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
            z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
            T_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
            T_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
            R_0 = R_s + (R_e - R_s) * x2
            T_0 = T_s + (T_e - T_s) * x2
            R = np.sign(R_0) * np.sqrt(R_0 ** 2 + T_0 ** 2)
            dR_0dx1 = (self.x_start[c] - self.x_start[f]) * (1 - x2) + (
                self.x_end[c] - self.x_end[f]
            ) * x2
            dR_0dx2 = R_e - R_s
            dT_0dx1 = (self.y_start[c] - self.y_start[f]) * (1 - x2) + (
                self.y_end[c] - self.y_end[f]
            ) * x2
            dT_0dx2 = T_e - T_s
            dzdx1 = (self.z_start[c] - self.z_start[f]) * (1 - x2) + (
                self.z_end[c] - self.z_end[f]
            ) * x2
            dzdx2 = z_e - z_s
            return [
                [
                    2 / R * (R_0 * dR_0dx1 + T_0 * dT_0dx1),
                    2 / R * (R_0 * dR_0dx2 + T_0 * dT_0dx2),
                ],
                [dzdx1, dzdx2],
            ]

        def forward(x):
            x_prime, z_prime = self.convert_to_Rz(x[0], x[1], 0)
            return [x_prime - R, z_prime - z]

        @np.vectorize
        def invert(R, z):
            """Perform a nonlinear-solve for an R-z pair."""
            # x = [R, z]
            result = root(forward, [0, 0.5], jac=jacobian)
            if not result.success:
                raise RuntimeWarning(
                    "Solver did not fully converge while inverting lines of sight."
                )
            return result.x[0], result.x[1]

        Rz = invert(R, z)
        return Rz[0], Rz[1]
        # # TODO: Consider if there is some way to invert this exactly,
        # # rather than rely on interpolation (which is necessarily
        # # inexact, as well as computationally expensive).
        # if not self.index_inversion:
        #     R_vals, z_vals, _ = self.convert_to_Rz()
        #     points = np.stack((np.ravel(R_vals), np.ravel(z_vals))).T
        #     index_vals = self.default_x1 * np.ones_like(self.default_x2)
        #     x2_vals = np.ones_like(self.default_x1) * self.default_x2
        #     interp2d = (
        #         LinearNDInterpolator if np.all(self.T_start == 0.0) and
        #         np.all(self.T_end == 0.0) else CloughTocher2DInterpolator
        #     )
        #     self.index_inversion = interp2d(points, np.ravel(index_vals))
        #     self.x2_inversion = interp2d(points, np.ravel(x2_vals))
        #     self.index_inversion = Rbf(R_vals, z_vals, index_vals)
        #     self.x2_inversion = Rbf(R_vals, z_vals, x2_vals)
        # assert self.x2_inversion is not None
        # x1 = self.index_inversion(R, z)
        # x2 = self.x2_inversion(R, z)
        # return x1, x2

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
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        R_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        R_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        T_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        T_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        R = R_s + (R_e - R_s) * x2
        T = T_s + (T_e - T_s) * x2
        z = z_s + (z_e - z_s) * x2
        spacings = np.sqrt(
            R.diff(direction) ** 2 + z.diff(direction) ** 2 + T.diff(direction) ** 2
        )
        result = zeros_like(R)
        result[{direction: slice(1, None)}] = spacings.cumsum(direction)
        return result


def _get_wall_intersection_distances(
    x_start: np.ndarray,
    z_start: np.ndarray,
    y_start: np.ndarray,
    x_end: np.ndarray,
    z_end: np.ndarray,
    y_end: np.ndarray,
    machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (1.83, 3.9),
        (-1.75, 2.0),
    ),
) -> np.ndarray:
    """
    Calculate the lenght needed for a line of sight to intersect with a wall of
    the Tokamak.

    Parameters
    ----------
    R_start
        1-D array of major radii of the start for each line-of-sight.
    z_start
        1-D array of vertical positions of the start for each line-of-sight.
    y_start
        1-D array of toroidal offset for the start of each line-of-sight.
    R_end
        1-D array of major radii of the end for each line-of-sight.
    z_end
        1-D array of vertical positions of the end for each line-of-sight.
    T_end
        1-D array of toroidal offset for the end of each line-of-sight.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``. Defaults to values for JET.

    Returns
    -------
    lengths
        The length of each line of sight for it to intersect a Tokamak wall.

    """
    opposite_x = np.where(
        x_end - x_start < 0, machine_dimensions[0][0], machine_dimensions[0][1]
    )
    opposite_z = np.where(
        z_end - z_start < 0, machine_dimensions[1][0], machine_dimensions[1][1]
    )
    # Calculate where LOS intersects opposite R-surface
    a = (x_end - x_start) ** 2 + (y_end - y_start) ** 2
    b = 2 * (x_start * (x_end - x_start) + y_start * (y_end - y_start))
    c = x_start ** 2 + y_start ** 2 - opposite_x ** 2
    factor = np.where(x_end - x_start < 0, -1, 1)
    mask = b ** 2 - 4 * a * c < 0
    # Check line of sight actually intersects the expected wall
    opposite_x[mask] = machine_dimensions[0][1]
    a[mask] = (x_end[mask] - x_start[mask]) ** 2 + (y_end[mask] - y_start[mask]) ** 2
    b[mask] = 2 * (
        x_start[mask] * (x_end[mask] - x_start[mask])
        + y_start[mask] * (y_end[mask] - y_start[mask])
    )
    c[mask] = x_start[mask] ** 2 + y_start[mask] ** 2 - opposite_x[mask] ** 2
    factor[mask] *= -1
    x2_trial = (-b + factor * np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    z_trial = z_start + x2_trial * (z_end - z_start)

    x2 = np.where(
        np.logical_and(
            machine_dimensions[1][0] <= z_trial, z_trial <= machine_dimensions[1][1]
        ),
        x2_trial,
        (opposite_z - z_start) / (z_end - z_start),
    )
    return x2 * np.sqrt(
        (x_start - x_end) ** 2 + (z_start - z_end) ** 2 + (y_start - y_end) ** 2
    )
