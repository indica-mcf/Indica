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
from .flux_surfaces import FluxSurfaceCoordinates


class LinesOfSightTransform(CoordinateTransform):
    """Coordinate system for data collected along a number of lines-of-sight.

    The first coordinate in this system is an index indicating which
    line-of-site a location is on. The second coordinate ranges from 0
    to 1 (inclusive) and indicates the position of a location along
    the line-of-sight. Note that diagnostic using this coordinate
    system will usually only be indexed in the first coordinate, as
    the measurements were integrated along the line-of-sight.

    If not passed to the constructor, the default grid for converting
    from the x-z system is chosen as follows:

    - The x-grid ranges from ``min(x_start.min(), x_end.min())`` to
      ``max(x_start.max(), x_end.max())`` with ``num_points`` intervals.
    - The z-grid ranges from ``min(z_start.min(), z_end.min())`` to
      ``max(z_start.max(), z_end.max())`` with ``num_points`` intervals.

    Parameters
    ----------
    origin_x
        1-D array of x positions of the start for each line-of-sight.
    origin_y
        1-D array of z positions of the start for each line-of-sight.
    origin_z
        1-D array of y positions for the start of each line-of-sight.
    direction_x
        1-D array of x positions of the end for each line-of-sight.
    direction_y
        1-D array of z positions of the end for each line-of-sight.
    direction_z
        1-D array of y positions for the end of each line-of-sight.
    name
        The name to refer to this coordinate system by, typically taken
        from the instrument it describes.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in x-z space:
        ``((xmin, xmax), (zmin, zmax)``. Defaults to values for JET.

    """

    def __init__(
        self,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        direction_x: float,
        direction_y: float,
        direction_z: float,
        name: str,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
        dell: float = 0.01,
    ):

        # Origin and Direction vectors
        origin: Tuple[float, float, float] = (origin_x, origin_y, origin_z)
        direction: Tuple[float, float, float] = (direction_x, direction_y, direction_z)

        # Calculate x_start, y_start, z_start, x_end, y_end and z_end
        start_coords, end_coords = _find_wall_intersections(origin, direction, machine_dimensions=machine_dimensions)
        x_start = start_coords[0]
        y_start = start_coords[1]
        z_start = start_coords[2]
        x_end = end_coords[0]
        y_end = end_coords[1]
        z_end = end_coords[2]

        self.x_start = DataArray(x_start)
        self.z_start = DataArray(z_start)
        self.y_start = DataArray(y_start)
        self._original_x_end = DataArray(x_end)
        self._original_z_end = DataArray(z_end)
        self._original_y_end = DataArray(y_end)
        self._machine_dims = machine_dimensions
        self.x_end = DataArray(x_end)
        self.z_end = DataArray(z_end)
        self.y_end = DataArray(y_end)
        self.index_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None
        self.x2_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None
        self.x1_name = name + "_coords"
        self.x2_name = name + "_los_position"

        # Set "dell"
        self.dell_target = dell
        x2, dell_new = self.set_dl(dell)
        self.x2 = x2
        self.dell = dell_new

        # Set x, y, z
        self.x = self.x_start + (self.x_end - self.x_start) * x2
        self.y = self.y_start + (self.y_end - self.y_start) * x2
        self.z = self.z_start + (self.z_end - self.z_start) * x2

        # Calculate r, theta (cylindrical coordinates)
        self.R = np.sqrt(self.x**2 + self.y**2)
        # self.theta = np.arctan2(self.y, self.x)

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
        x_0 = self.x_start + (self.x_end - self.x_start) * x2
        y_0 = self.y_start + (self.y_end - self.y_start) * x2
        z = self.z_start + (self.z_end - self.z_start) * x2
        return np.sign(x_0) * np.sqrt(x_0**2 + y_0**2), z

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        def jacobian(x):
            x1 = x[0]
            x2 = x[1]
            x_0 = self.x_start + (self.x_end - self.x_start) * x2
            y_0 = self.y_start + (self.y_end - self.y_start) * x2
            x = np.sign(x_0) * np.sqrt(x_0**2 + y_0**2)
            dx_0dx1 = 0.0
            dx_0dx2 = self.x_end - self.x_start
            dy_0dx1 = 0.0
            dy_0dx2 = self.y_end - self.y_start
            dzdx1 = 0.0
            dzdx2 = self.z_end - self.z_start
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
        x = self.x_start + (self.x_end - self.x_start) * x2
        y = self.y_start + (self.y_end - self.y_start) * x2
        z = self.z_start + (self.z_end - self.z_start) * x2
        spacings = np.sqrt(
            x.diff(direction) ** 2 + z.diff(direction) ** 2 + y.diff(direction) ** 2
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

        # Set dl, calculate dl
        ind = np.linspace(0, 1, npts, dtype=float)
        x2 = DataArray(ind, dims=self.x2_name)
        dell_temp = self.distance(self.x2_name, 0, x2, 0)
        dell = dell_temp[1]

        return x2, dell

    def assign_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        self.flux_transform = flux_transform

    def convert_to_rho(self):
        self.rho = self.flux_transform.convert_from_Rz(self.R, self.z)


def _find_wall_intersections(
    origin: Tuple,
    direction: Tuple,
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


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0] - 1
    n2 = x2.shape[0] - 1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.NaN

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T

    indii = ii[in_range] + T[0, in_range]
    indjj = jj[in_range] + T[1, in_range]

    return xy0[:, 0], xy0[:, 1], indii, indjj
