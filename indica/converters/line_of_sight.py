"""Coordinate system representing a collection of lines of sight.
"""

from typing import cast
from typing import Tuple

import numpy as np
from scipy.optimize import root
import xarray as xr
from xarray import DataArray
from xarray import zeros_like

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .flux_surfaces import FluxSurfaceCoordinates
from ..numpy_typing import LabeledArray


class LineOfSightTransform(CoordinateTransform):
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
        A float giving x position for the origin of the line-of-sight.
    origin_y
        A float giving y position for the origin of the line-of-sight.
    origin_z
        A float giving z position for the origin of the line-of-sight.
    direction_x
        A float giving x position for the direction of the line-of-sight.
    direction_y
        A float giving y position for the direction of the line-of-sight.
    direction_z
        A float giving z position for the direction of the line-of-sight.
    name
        The name to refer to this coordinate system by, typically taken
        from the instrument it describes.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in x-z space:
        ``((xmin, xmax), (zmin, zmax)``. Defaults to values for JET.
    dl
        A float giving the distance between coordinates along the
        line-of-sight. Default to 0.01 metres.

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
        dl: float = 0.01,
    ):

        # Origin and Direction vectors
        origin: Tuple[float, float, float] = (origin_x, origin_y, origin_z)
        direction: Tuple[float, float, float] = (direction_x, direction_y, direction_z)

        # Calculate x_start, y_start, z_start, x_end, y_end and z_end
        start_coords, end_coords = _find_wall_intersections(
            origin, direction, machine_dimensions=machine_dimensions
        )
        x_start = start_coords[0]
        y_start = start_coords[1]
        z_start = start_coords[2]
        x_end = end_coords[0]
        y_end = end_coords[1]
        z_end = end_coords[2]

        self.name = f"{name}_line_of_sight_transform"
        self.x_start = DataArray(x_start)
        self.z_start = DataArray(z_start)
        self.y_start = DataArray(y_start)
        self._machine_dims = machine_dimensions
        self.x_end = DataArray(x_end)
        self.z_end = DataArray(z_end)
        self.y_end = DataArray(y_end)
        self.x1_name = "channel"
        self.x2_name = "los_position"

        self.x1 = np.array(0)

        # Set "dl"
        self.dl_target = dl
        x2, dl_new = self.set_dl(dl)
        self.x2 = x2
        self.dl = dl_new

        # Set x, y, z, r
        self.x, self.y = self.convert_to_xy(0, x2, 0)
        self.R, self.z = self.convert_to_Rz(0, x2, 0)

        # Calculate r, phi (cylindrical coordinates)
        self.phi = np.arctan2(self.y, self.x)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        result = cast(bool, result and np.all(self.x_start == other.x_start))
        result = cast(bool, result and np.all(self.z_start == other.z_start))
        result = cast(bool, result and np.all(self.y_start == other.y_start))
        result = cast(bool, result and np.all(self.x_end == other.x_end))
        result = cast(bool, result and np.all(self.z_end == other.z_end))
        result = cast(bool, result and np.all(self.y_end == other.y_end))
        result = cast(bool, result and np.all(self.dl == other.dl))
        result = cast(bool, result and np.all(self.x2 == other.x2))
        result = cast(bool, result and np.all(self.R == other.R))
        result = cast(bool, result and np.all(self.phi == other.phi))
        result = result and self._machine_dims == other._machine_dims
        return result

    def convert_to_xy(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Tuple:
        x = self.x_start + (self.x_end - self.x_start) * x2
        y = self.y_start + (self.y_end - self.y_start) * x2
        return x, y

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        x = self.x_start + (self.x_end - self.x_start) * x2
        y = self.y_start + (self.y_end - self.y_start) * x2
        z = self.z_start + (self.z_end - self.z_start) * x2
        return np.sign(x) * np.sqrt(x**2 + y**2), z

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        def jacobian(x):
            # x1 = x[0]
            x2 = x[1]
            x = self.x_start + (self.x_end - self.x_start) * x2
            y = self.y_start + (self.y_end - self.y_start) * x2
            x = np.sign(x) * np.sqrt(x**2 + y**2)
            dxdx1 = 0.0
            dxdx2 = self.x_end - self.x_start
            dydx1 = 0.0
            dydx2 = self.y_end - self.y_start
            dzdx1 = 0.0
            dzdx2 = self.z_end - self.z_start
            return [
                [
                    2 / x * (x * dxdx1 + y * dydx1),
                    2 / x * (x * dxdx2 + y * dydx2),
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

    def distance(
        self,
        direction: str,
        x1: LabeledArray,
        x2: LabeledArray,
        t: LabeledArray,
    ) -> np.ndarray:
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
        return result.values

    def set_dl(self, dl: float):
        # Convert to Cartesian
        x_start = self.x_start
        y_start = self.y_start
        z_start = self.z_start
        x_end = self.x_end
        y_end = self.y_end
        z_end = self.z_end
        los_length = np.sqrt(
            (x_end - x_start) ** 2 + (y_end - y_start) ** 2 + (z_end - z_start) ** 2
        )

        # Find the number of points
        npts = np.ceil(los_length.data / dl).astype(int)

        # Set dl, calculate dl
        ind = np.linspace(0, 1, npts, dtype=float)
        x2 = DataArray(ind, dims=self.x2_name)
        dl = self.distance(self.x2_name, 0, x2[0:2], 0)[1]

        return x2, dl

    def set_flux_transform(
        self, flux_transform: FluxSurfaceCoordinates, force: bool = False
    ):
        """
        Set flux surface transform to perform remapping from physical to flux space
        """
        if not hasattr(self, "flux_transform") or force:
            self.flux_transform = flux_transform
        elif self.flux_transform != flux_transform:
            raise Exception("Attempt to set flux surface transform twice.")

    def convert_to_rho(self, t: LabeledArray = None) -> Coordinates:
        """
        Convert R, z to rho given the flux surface transform
        """
        if not hasattr(self, "flux_transform"):
            raise Exception("Set flux transform to convert (R,z) to rho")
        if not hasattr(self.flux_transform, "equilibrium"):
            raise Exception("Set equilibrium in flux transform to convert (R,z) to rho")

        rho, theta = self.flux_transform.convert_from_Rz(self.R, self.z, t=t)
        rho = DataArray(rho, coords=[("t", t), (self.x2_name, self.x2)])
        theta = DataArray(theta, coords=[("t", t), (self.x2_name, self.x2)])
        rho = xr.where(rho >= 0, rho, 0.0)

        self.rho = rho
        self.theta = theta

        return rho, theta

    def map_to_los(
        self,
        profile_1d: DataArray,
        t: LabeledArray = None,
        limit_to_sep=True,
    ):
        """
        Map 1D profile to LOS
        TODO: extend for 2D interpolation to (R, z) instead of rho
        Parameters
        ----------
        profile_1d
            DataArray of the 1D profile to integrate
        t
            Time for interpolation
        limit_to_sep
            Set to True if values outside of separatrix are to be set to 0

        Returns
        -------
        Interpolation of the input profile along the LOS
        """
        self.check_flux_transform()
        if not hasattr(self, "rho"):
            self.convert_to_rho(t=t)

        if t is not None:
            rho = self.rho.interp(t=t, method="linear")
        else:
            rho = self.rho
        along_los = profile_1d.interp(rho_poloidal=rho)
        if limit_to_sep:
            along_los = xr.where(
                rho <= 1,
                along_los,
                0,
            )

        return along_los

    def integrate_on_los(
        self,
        profile_1d: DataArray,
        t: LabeledArray = None,
        limit_to_sep=True,
        passes: int = 1,
    ):
        """
        Integrate 1D profile along LOS
        Parameters
        ----------
        profile_1d
            DataArray of the 1D profile to integrate
        t
            Time for interpolation
        limit_to_sep
            Set to True if values outside of separatrix are to be set to 0
        passes
            Number of passes across the plasma (e.g. typical interferometer passes=2)

        Returns
        -------
        Line of sight integral along the LOS
        """
        along_los = self.map_to_los(
            profile_1d,
            t=t,
            limit_to_sep=limit_to_sep,
        )

        los_integral = passes * along_los.sum(self.x2_name) * self.dl

        return los_integral, along_los

    def check_flux_transform(self):
        if not hasattr(self, "flux_transform"):
            raise Exception("Missing flux surface transform")
        if not hasattr(self.flux_transform, "equilibrium"):
            raise Exception("Missing equilibrium in flux surface transform")


def _find_wall_intersections(
    origin: Tuple,
    direction: Tuple,
    machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (1.83, 3.9),
        (-1.75, 2.0),
    ),
):
    """Function for calculating "start" and "end" positions of the line-of-sight
    given the machine dimensions.

    The end coordinate is calculated by finding the intersections with the
    machine dimensions. If the intersection hits the inner column, the first
    intersection point becomes the "end" position of the line-of-sight. If the
    intersection misses the inner column, the last intersection point with the
    outer column becomes the "end" position.

    Parameters
    ----------
    origin
        A Tuple (1x3) giving the X, Y and Z origin positions of the line-of-sight
    direction
        A Tuple (1x3) giving the X, Y and Z direction of the line-of-sight
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in x-z space:
        ``((xmin, xmax), (zmin, zmax)``. Defaults to values for JET.

    Returns
    -------
    start_coordinates
        A Tuple (1x3) giving the X, Y and Z start positions of the line-of-sight
    end_coordinates
        A Tuple (1x3) giving the X, Y and Z end positions of the line-of-sight
    """

    # Define XYZ lines for LOS from origin and direction vectors
    length = 10.0
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

    # Find intersections, to calculate
    # R_start, z_start, T_start, R_end, z_end, T_end ...
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
        x,y, ix, iy=intersection(x1,y1,x2,y2)
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
        except np.linalg.LinAlgError:
            T[:, i] = np.NaN

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T

    indii = ii[in_range] + T[0, in_range]
    indjj = jj[in_range] + T[1, in_range]

    return xy0[:, 0], xy0[:, 1], indii, indjj
