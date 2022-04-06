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
from .intersections import intersection
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
    from the x-y-z system is chosen as follows:

    - The x-grid ranges from ``min(x_start.min(), x_end.min())`` to
      ``max(x_start.max(), x_end.max())`` with ``num_points`` intervals.
    - The y-grid ranges from ``min(y_start.min(), y_end.min())`` to
      ``max(y_start.max(), y_end.max())`` with ``num_points`` intervals.
    - The z-grid ranges from ``min(z_start.min(), z_end.min())`` to
      ``max(z_start.max(), z_end.max())`` with ``num_points`` intervals.

    Parameters
    ----------
    x_start
        1-D array of x positions of the start for each line-of-sight.
    y_start
        1-D array of y positions for the start of each line-of-sight.
    z_start
        1-D array of z positions of the start for each line-of-sight.
    x_end
        1-D array of x positions of the end for each line-of-sight.
    y_end
        1-D array of y positions for the end of each line-of-sight.
    z_end
        1-D array of z positions of the end for each line-of-sight.
    name
        The name to refer to this coordinate system by, typically taken
        from the instrument it describes.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in xy-z space:
        ``((xymin, xymax), (zmin, zmax)`` where the machine is symmetric
        with respect to the z-axis. Defaults to values for JET.

    """

    def __init__(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        name: str,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
        dl: float = 0.01,
    ):

        # Find intersections with inner and outer wall
        start_coord, end_coord = _find_wall_intersections(
            origin, direction, machine_dimensions
        )
        x_start = np.array([start_coord[0]])
        y_start = np.array([start_coord[1]])
        z_start = np.array([start_coord[2]])
        x_end = np.array([end_coord[0]])
        y_end = np.array([end_coord[1]])
        z_end = np.array([end_coord[2]])
        # lengths = _get_wall_intersection_distances(
        #    x_start, y_start, z_start, x_end, y_end, z_end, machine_dimensions
        # )
        # new_length = max(lengths)
        # los_lengths = np.sqrt(
        #    (x_start - x_end) ** 2 + (y_start - y_end) ** 2 + (z_start - z_end) ** 2
        # )
        # factor = new_length / los_lengths
        self.x_start = DataArray(x_start)
        self.y_start = DataArray(y_start)
        self.z_start = DataArray(z_start)
        self._original_x_end = DataArray(x_end)
        self._original_y_end = DataArray(y_end)
        self._original_z_end = DataArray(z_end)
        self._machine_dims = machine_dimensions
        # self.x_end = DataArray(x_start + factor * (x_end - x_start))
        # self.y_end = DataArray(y_start + factor * (y_end - y_start))
        # self.z_end = DataArray(z_start + factor * (z_end - z_start))
        self.x_end = DataArray(x_end)
        self.y_end = DataArray(y_end)
        self.z_end = DataArray(z_end)
        self.index_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None
        self.x2_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None
        self.x1_name = name + "_coords"
        self.x2_name = name + "_los_position"

        # Set "dl" and "x2"
        self.set_dl(dl)
        # print(self.x2)
        # print(self.dl)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        result = result and np.all(self.x_start == other.x_start)
        result = result and np.all(self.y_start == other.y_start)
        result = result and np.all(self.z_start == other.z_start)
        result = result and np.all(self.x_end == other.x_end)
        result = result and np.all(self.y_end == other.y_end)
        result = result and np.all(self.z_end == other.z_end)
        result = result and self._machine_dims == other._machine_dims
        return result

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        x_0 = x_s + (x_e - x_s) * x2
        y_0 = y_s + (y_e - y_s) * x2
        z = z_s + (z_e - z_s) * x2
        return np.sqrt(x_0 ** 2 + y_0 ** 2), z

    def convert_to_xyz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        x_0 = x_s + (x_e - x_s) * x2
        y_0 = y_s + (y_e - y_s) * x2
        z = z_s + (z_e - z_s) * x2
        return x_0, y_0, z

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        def jacobian(x):
            x1 = x[0]
            x2 = x[1]
            c = np.ceil(x1).astype(int)
            f = np.floor(x1).astype(int)
            x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
            x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
            y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
            y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
            z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
            z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
            x_0 = x_s + (x_e - x_s) * x2
            y_0 = y_s + (y_e - y_s) * x2
            x = np.sign(x_0) * np.sqrt(x_0 ** 2 + y_0 ** 2)
            dx_0dx1 = (self.x_start[c] - self.x_start[f]) * (1 - x2) + (
                self.x_end[c] - self.x_end[f]
            ) * x2
            dx_0dx2 = x_e - x_s
            dy_0dx1 = (self.y_start[c] - self.y_start[f]) * (1 - x2) + (
                self.y_end[c] - self.y_end[f]
            ) * x2
            dy_0dx2 = y_e - y_s
            dzdx1 = (self.z_start[c] - self.z_start[f]) * (1 - x2) + (
                self.z_end[c] - self.z_end[f]
            ) * x2
            dzdx2 = z_e - z_s
            return [
                [
                    2 / x * (x_0 * dx_0dx1 + y_0 * dy_0dx1),
                    2 / x * (x_0 * dx_0dx2 + y_0 * dy_0dx2),
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
        self, direction: str, x1: LabeledArray, x2: LabeledArray, t: LabeledArray,
    ) -> LabeledArray:
        """Implementation of calculation of physical distances between points
        in this coordinate system. This accounts for potential toroidal skew of
        lines.

        """
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        x = x_s + (x_e - x_s) * x2
        y = y_s + (y_e - y_s) * x2
        z = z_s + (z_e - z_s) * x2
        spacings = np.sqrt(
            x.diff(direction) ** 2 + y.diff(direction) ** 2 + z.diff(direction) ** 2
        )
        result = zeros_like(x)
        result[{direction: slice(1, None)}] = spacings.cumsum(direction)
        return result

    def set_dl(self, dl: float):
        # Convert to Cartesian
        x_start = self.x_start
        y_start = self.y_start
        z_start = self.z_start
        x_end = self.x_end
        y_end = self.y_end
        z_end = self.z_end
        d = np.sqrt(
            (x_end - x_start) ** 2 + (y_end - y_start) ** 2 + (z_end - z_start) ** 2
        )

        # Find the number of points
        npts = np.ceil(d.data / dl).astype(int)
        ind = np.linspace(0, 1, npts[0], dtype=float)

        # Set "x2" and "dl"
        self.x2 = DataArray(ind, dims=self.x2_name)
        self.dl = self.distance(self.x2_name, 0, self.x2, 0)[1]


def _get_wall_intersection_distances(
    x_start: np.ndarray,
    y_start: np.ndarray,
    z_start: np.ndarray,
    x_end: np.ndarray,
    y_end: np.ndarray,
    z_end: np.ndarray,
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
    x_start
        1-D array of x positions of the start for each line-of-sight.
    y_start
        1-D array of y positions for the start of each line-of-sight.
    z_start
        1-D array of z positions of the start for each line-of-sight.
    x_end
        1-D array of x positions of the end for each line-of-sight.
    y_end
        1-D array of y positions for the end of each line-of-sight.
    z_end
        1-D array of z positions of the end for each line-of-sight.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in xy-z space:
        ``((xymin, xymax), (zmin, zmax)``. Defaults to values for JET.

    Returns
    -------
    lengths
        The length of each line of sight for it to intersect a Tokamak wall.

    """
    # TODO: generalize for (x, y, z) plane looking for intersection with the inner column
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

    x2_opposite = np.zeros_like(x2_trial)
    if not np.array_equal(z_start, z_end):
        x2_opposite = (opposite_z - z_start) / (z_end - z_start)

    x2 = np.where(
        np.logical_and(
            machine_dimensions[1][0] <= z_trial, z_trial <= machine_dimensions[1][1]
        ),
        x2_trial,
        x2_opposite,
    )
    return x2 * np.sqrt(
        (x_start - x_end) ** 2 + (y_start - y_end) ** 2 + (z_start - z_end) ** 2
    )


def _find_wall_intersections(
    origin: np.ndarray,
    direction: np.ndarray,
    machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (1.83, 3.9),
        (-1.75, 2.0),
    ),
):

    # Define XYZ lines for LOS from origin and direction vectors
    length = 3.0
    x_line = np.array(
        [origin[0] - length * direction[0], origin[0] + length * direction[0]],
        dtype=float,
    )
    y_line = np.array(
        [origin[1] - length * direction[1], origin[1] + length * direction[1]],
        dtype=float,
    )

    # Define XYZ lines for inner and outer walls
    angles = np.linspace(0.0, 2 * np.pi, 1000)
    x_wall_inner = machine_dimensions[0][0] * np.cos(angles)
    y_wall_inner = machine_dimensions[0][0] * np.sin(angles)
    x_wall_outer = machine_dimensions[0][1] * np.cos(angles)
    y_wall_outer = machine_dimensions[0][1] * np.sin(angles)

    # Find intersections, to calculate R_start, z_start, T_start, R_end, z_end, T_end ...
    xx, yx, ix, jx = intersection(x_line, y_line, x_wall_outer, y_wall_outer)
    distance = np.sqrt(
        (xx - origin[0]) ** 2 + (yx - origin[1]) ** 2
    )  # Distance from intersections
    i_closest = np.argmin(distance)
    i_furthest = np.argmax(distance)
    x_start = xx[i_closest]
    y_start = yx[i_closest]
    z_start = origin[-1]
    x_end = xx[i_furthest]
    y_end = yx[i_furthest]
    z_end = origin[-1]

    # Find intersections with inner wall (if exists)
    xx, yx, ix, jx = intersection(x_line, y_line, x_wall_inner, y_wall_inner)
    if len(xx) > 0:
        distance = np.sqrt(
            (xx - x_line[0]) ** 2 + (yx - y_line[0]) ** 2
        )  # Distance from intersections
        i_closest = np.argmin(distance)
        x_end = xx[i_closest]
        y_end = yx[i_closest]
        z_end = origin[-1]

    return (x_start, y_start, z_start), (x_end, y_end, z_end)
