"""Coordinate system representing a collection of lines of sight.
"""

from typing import Any
from typing import cast
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray
from xarray import Dataset
from xarray import zeros_like

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .abstractconverter import find_wall_intersections
from ..numpy_typing import LabeledArray
from ..numpy_typing import OnlyArray


class LineOfSightTransform(CoordinateTransform):
    """Coordinate system for data collected along a number of lines-of-sight.

    The first coordinate x1 is the channel number, the second x2 is the
    position along the line-of-sight and ranges from [0, 1].

    Parameters
    ----------
    origin_x
        x positions for the LOS origin in (m).
    origin_y
        y positions for the LOS origin in (m).
    origin_z
        z positions for the LOS origin in (m).
    direction_x
        delta x for the LOS direction in (m).
    direction_y
        delta y for the LOS direction in (m).
    direction_z
        delta z for the LOS direction in (m).
    name
        A string identifier (e.g. the name of the instrument).
    machine_dimensions
        The boundaries of the Tokamak in R-z space in (m):
        ``((Rmin, Rmax), (zmin, zmax)``.
    dl
        The spatial precision along the LOS in (m).
    passes
        Number of passes across the plasma (e.g. interferometer
        with corner cube has passes=2)
    beamlets_method
        Select option for method for distributing beamlets in
        the LOS.
        "simple" - beamlets distributed using numpy linspace
            e.g. linspace(-width/2, width/2, n_beamlets_x)
        "adaptive" (recommended) -  beamlets distributed evenly
            in both x and y directions.
    n_beamlets
        Number of beamlets in the LOS spot cross-section.
        Currently only works with quadratic sequence (e.g.
        1^2 = 1, 2^2 = 4, 3^2 = 9, 4^2 = 16, etc...)
    spot_width
        Width of the LOS spot in (m).
    spot_height
        Height of the LOS spot in (m).
    spot_shape
        Shape of the spot. e.g. "round", "square" or "rectangular"
    focal_length
        Focal length of the LOS in (m).
    plot_beamlets
        True/False flag to plot beamlet distribution across
        the spot

    Examples:
        Pinhole Cameras
            origin - position vector of the detector
            direction - direction vector from detector to the
                pinhole
            spot_shape - the spot is defined as the detector
                element cross-section, typically rectangular
            spot_width - width of the detector element
            spot_height - height of the detector element
            focal_length - distance from the centre of the
                detector element to the centre of the pinhole
        Spectrometers
            origin - position vector at a reference position
                beyond the optics e.g. the front of a lens or
                mirror position
            direction - direction vector of the LOS
            spot_shape - the spot is defined as the cross-section
                of the LOS, typically round
            spot_width - width of the spot
            spot_height - height of the spot
            focal_length - 'effective' distance from the spot to a
                virtual focal point to define the cone shape,
                typically a negative value (behind the lens)
    """

    # ToDo: Implement divergence
    def __init__(
        self,
        origin_x: OnlyArray,
        origin_y: OnlyArray,
        origin_z: OnlyArray,
        direction_x: OnlyArray,
        direction_y: OnlyArray,
        direction_z: OnlyArray,
        name: str = "",
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
        dl: float = 0.01,
        passes: int = 1,
        beamlets_method: str = "simple",
        n_beamlets: int = 1,
        spot_width: float = 0.0,
        spot_height: float = 0.0,
        spot_shape: str = "square",
        focal_length: float = -1000.0,
        plot_beamlets: bool = False,
        # div_width: float = 0.0,
        # div_height: float = 0.0,
        **kwargs: Any,
    ):

        self.instrument_name: str = name
        self.name = f"{self.instrument_name}_line_of_sight_transform"
        self.x1_name = "channel"
        self.x2_name = "los_position"
        self._machine_dims = machine_dimensions
        self.passes = passes

        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z
        self.direction_x = direction_x
        self.direction_y = direction_y
        self.direction_z = direction_z

        # Spot info
        self.spot_width = spot_width
        self.spot_height = spot_height
        self.spot_shape = spot_shape
        self.beamlets_method = beamlets_method
        self.n_beamlets_x = int(np.sqrt(n_beamlets))
        self.n_beamlets_y = int(np.sqrt(n_beamlets))
        self.focal_length = focal_length
        self.distribute_beamlets(debug=plot_beamlets)

        # self.div_width = div_width
        # self.spot_height = spot_height

        # Channel number
        self.x1: list = list(np.arange(0, len(origin_x)))

        # Calculate LOS coordinates
        self.set_dl(dl)

    @property
    def origin(self):
        self._origin = np.array(
            [self.origin_x, self.origin_y, self.origin_z]
        ).transpose()
        return self._origin

    @property
    def direction(self):
        self._direction = np.array(
            [self.direction_x, self.direction_y, self.direction_z]
        ).transpose()
        return self._direction

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
    ) -> Coordinates:
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        x = x_s + (x_e - x_s) * x2
        y = y_s + (y_e - y_s) * x2

        return x, y

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """
        Convert to (R,z) the position along the LOS x2 for channel(s) x1.
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

        return np.sqrt(x**2 + y**2), z

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a 'convert_from_Rz' "
            "method."
        )

    def distance(
        self,
        direction: str,
        x1: LabeledArray,
        x2: LabeledArray,
        t: LabeledArray,
    ) -> np.ndarray:
        """
        Calculate the physical distances between points along x2.
        """
        x = self.x_start[x1] + (self.x_end[x1] - self.x_start[x1]) * x2
        y = self.y_start[x1] + (self.y_end[x1] - self.y_start[x1]) * x2
        z = self.z_start[x1] + (self.z_end[x1] - self.z_start[x1]) * x2
        spacings = np.sqrt(
            x.diff(direction) ** 2 + z.diff(direction) ** 2 + y.diff(direction) ** 2
        )
        result = zeros_like(x)
        result[{direction: slice(1, None)}] = spacings.cumsum(direction)
        return result

    # Rotation function
    def rotate(self, x, y, xo, yo, theta):
        xr = np.cos(theta) * (x - xo) - np.sin(theta) * (y - yo) + xo
        yr = np.sin(theta) * (x - xo) + np.cos(theta) * (y - yo) + yo
        return np.array([xr, yr])

    # Function for adaptive beamlets
    def adaptive_beamlets(self, debug=False):
        # The size of each filament box
        delta = self.spot_width / float(self.n_beamlets_x)

        # The number of filament rows in the vertical dimension
        # Such that delta_x = delta_y - e.g. square
        m_dash = self.spot_height / delta
        m = int(np.ceil(m_dash))

        # Check "m" is odd or even
        if m % 2 == 0:  # even
            m = m + 1  # iterate to the next odd number
        else:  # odd
            pass

        param = (m * delta - self.spot_height) / 2
        param2 = delta / 2

        if debug:
            print(f"self.spot_width = {self.spot_width}")
            print(f"self.spot_height = {self.spot_height}")
            print(f"self.n_beamlets_x = {self.n_beamlets_x}")
            print(f"self.n_beamlets_y = {self.n_beamlets_y}")
            print(f"delta = {delta}")
            print(f"m_dash = {m_dash}")
            print(f"m = {m}")
            print(f"param = {param}")
            print(f"param2 = {param2}")
            print(" ")

        return param, param2, m

    # Function to define beamlets grid points
    def define_beamlets_simple(self):
        # Set-up beamlets grid
        grid_w = np.linspace(
            -self.spot_width / 2,
            self.spot_width / 2,
            self.n_beamlets_x,
            dtype=float,
        )
        grid_v = np.linspace(
            -self.spot_height / 2,
            self.spot_height / 2,
            self.n_beamlets_y,
            dtype=float,
        )
        W, V = np.meshgrid(grid_w, grid_v)
        w = W.flatten()
        v = V.flatten()

        return w, v

    # Function to define beamlets grid points
    def define_beamlets_adaptive(self):
        # Calculate parameters
        param, param2, m = self.adaptive_beamlets()

        count = 0
        while param > param2:
            # Go to next odd number
            self.n_beamlets_x = self.n_beamlets_x + 2

            # Calculate parameters
            param, param2, m = self.adaptive_beamlets()

            # Iterate
            count += 1
            if count == 9:
                raise ValueError("Beamlets solution not found")

        self.n_beamlets_y = m

        grid_w = np.linspace(
            -self.spot_width / 2,
            self.spot_width / 2,
            self.n_beamlets_x * 2 + 1,
            dtype=float,
        )
        grid_v = np.linspace(
            -self.spot_height / 2,
            self.spot_height / 2,
            self.n_beamlets_y * 2 + 1,
            dtype=float,
        )
        grid_w = grid_w[1::2]
        grid_v = grid_v[1::2]
        W, V = np.meshgrid(grid_w, grid_v)
        w = W.flatten()
        v = V.flatten()

        return w, v

    def distribute_beamlets(self, debug=False):
        """
        Distribute beamlets using information on spot size and divergence.
        """

        # Define beamlets grid
        if self.n_beamlets_x > 1:

            if self.beamlets_method == "adaptive":
                w, v = self.define_beamlets_adaptive()
            elif self.beamlets_method == "simple":
                w, v = self.define_beamlets_simple()
            else:
                raise ValueError(
                    f"Invalid 'beamlets_method' option: {self.beamlets_method}"
                )
        else:
            w = np.array([0.0])
            v = np.array([0.0])

        # Set number of beamlets
        self.beamlets = int(self.n_beamlets_x * self.n_beamlets_y)

        # Draw spot
        if self.spot_shape == "round":

            if self.n_beamlets_x > 1:

                # Draw ellipse
                a = self.spot_width / 2
                b = self.spot_height / 2
                spot_w = np.linspace(-a, a, 500, dtype=float)
                spot_y = (b / a) * np.sqrt(a**2 - spot_w**2)
                spot_w = np.append(spot_w, np.flip(spot_w))
                spot_y = np.append(spot_y, np.flip(-spot_y))

                # Find beamlets outside of the round shape
                # and remove them
                val = (w**2 / a**2) + (v**2 / b**2)
                inside = val <= 1
                self.beamlets = int(np.sum(inside))
                w = w[inside]
                v = v[inside]

            else:

                spot_w = np.nan
                spot_y = np.nan

        elif self.spot_shape == "square":

            spot_w = np.array(
                [
                    -0.5 * self.spot_width,
                    0.5 * self.spot_width,
                    0.5 * self.spot_width,
                    -0.5 * self.spot_width,
                    -0.5 * self.spot_width,
                ]
            )
            spot_y = np.array(
                [
                    -0.5 * self.spot_height,
                    -0.5 * self.spot_height,
                    0.5 * self.spot_height,
                    0.5 * self.spot_height,
                    -0.5 * self.spot_height,
                ]
            )

            if self.spot_width != self.spot_height:
                raise ValueError("spot_width does not equal spot_height")

        elif self.spot_shape == "rectangular":

            spot_w = np.array(
                [
                    -0.5 * self.spot_width,
                    0.5 * self.spot_width,
                    0.5 * self.spot_width,
                    -0.5 * self.spot_width,
                    -0.5 * self.spot_width,
                ]
            )
            spot_y = np.array(
                [
                    -0.5 * self.spot_height,
                    -0.5 * self.spot_height,
                    0.5 * self.spot_height,
                    0.5 * self.spot_height,
                    -0.5 * self.spot_height,
                ]
            )

        else:
            raise ValueError("Spot shape not available")

        if debug:
            plt.figure()
            plt.plot(spot_w, spot_y, "k")
            plt.scatter(w, v, c="r")
            plt.axis("equal")
            plt.show()

        # Set weightings
        self.weightings = np.ones_like(w)

        # Find distance to virtual focus position
        distance = self.focal_length

        # Build beamlets
        beamlet_origin_x = np.zeros((len(self.direction_x), self.beamlets))
        beamlet_origin_y = np.zeros((len(self.direction_x), self.beamlets))
        beamlet_origin_z = np.zeros((len(self.direction_x), self.beamlets))
        beamlet_direction_x = np.zeros((len(self.direction_x), self.beamlets))
        beamlet_direction_y = np.zeros((len(self.direction_x), self.beamlets))
        beamlet_direction_z = np.zeros((len(self.direction_x), self.beamlets))
        for i_los in range(len(self.direction_x)):

            # Direction coordinates
            dir_x = self.direction_x[i_los]
            dir_y = self.direction_y[i_los]
            dir_z = self.direction_z[i_los]

            # Origin coordinates
            orig_x = self.origin_x[i_los]
            orig_y = self.origin_y[i_los]
            orig_z = self.origin_z[i_los]

            # Find the normal of the direction
            normal = np.array([-dir_y, dir_x])
            ang_norm = np.arctan2(normal[1], normal[0])

            # Calculate virtual system reference coordinate
            system_x = orig_x + dir_x * distance
            system_y = orig_y + dir_y * distance
            system_z = orig_z + dir_z * distance

            # Iterate over each beamlet
            for i_beamlet in range(self.beamlets):
                # Move origin along plane normal to the direction
                beamlet_origin_x[i_los, i_beamlet] = orig_x + w[i_beamlet] * np.cos(
                    ang_norm
                )
                beamlet_origin_y[i_los, i_beamlet] = orig_y + w[i_beamlet] * np.sin(
                    ang_norm
                )
                beamlet_origin_z[i_los, i_beamlet] = orig_z + v[i_beamlet]

                # Direction
                if distance < 0:  # fibre optics
                    dir_vec = np.array(
                        [
                            beamlet_origin_x[i_los, i_beamlet] - system_x,
                            beamlet_origin_y[i_los, i_beamlet] - system_y,
                            beamlet_origin_z[i_los, i_beamlet] - system_z,
                        ]
                    )
                else:  # Pinhole cameras and neutral beams
                    dir_vec = np.array(
                        [
                            system_x - beamlet_origin_x[i_los, i_beamlet],
                            system_y - beamlet_origin_y[i_los, i_beamlet],
                            system_z - beamlet_origin_z[i_los, i_beamlet],
                        ]
                    )
                dir_vec_norm = dir_vec / np.linalg.norm(dir_vec)
                beamlet_direction_x[i_los, i_beamlet] = dir_vec_norm[0]
                beamlet_direction_y[i_los, i_beamlet] = dir_vec_norm[1]
                beamlet_direction_z[i_los, i_beamlet] = dir_vec_norm[2]

                # if self.div_width > 0:
                #     # Divergence
                #     if distance < 0:  # fibre optics
                #         div_angle_xy = self.div_width * (
                #             -grid_w[i_w] * 2 / self.spot_width
                #         )
                #         div_angle_rz = self.div_width * (
                #             grid_v[i_v] * 2 / self.spot_height
                #         )
                #     else:  # Pinhole cameras and neutral beams
                #         div_angle_xy = self.div_width * (
                #             -grid_w[i_w] * 2 / self.spot_width
                #         )
                #         div_angle_rz = self.div_width * (
                #             -grid_v[i_v] * 2 / self.spot_height
                #         )
                #
                #     # Calculate the projected distance in the XY plane
                #     r = np.sqrt(dir_vec_norm[0] ** 2 + dir_vec_norm[1] ** 2)
                #
                #     # Rotate direction vector in the XY plane with the angle
                #     # of divergence for each beamlet
                #     dir_vec_div_xy = self.rotate(
                #         dir_vec_norm[0], dir_vec_norm[1], 0.0, 0.0, div_angle_xy
                #     )
                #
                #     # Calculate new beamlet angle in the RZ plane
                #     theta_v = np.arctan2(
                #         dir_vec_norm[2],
                #         np.sqrt(dir_vec_norm[0] ** 2 + dir_vec_norm[1] ** 2),
                #     )
                #
                #     # Calculate the projected vertical distance,
                #     # while conserving distance of the LOS = 1.0 meter
                #     zdash = np.sin(theta_v + div_angle_rz)
                #
                #     # Therefore,
                #     # calculate the new projected distance in the XY plane
                #     rdash = np.sqrt(1.0 - zdash**2)
                #
                #     # Rescale the rotated direction vector in the XY plane
                #     dir_vec_div_xy = dir_vec_div_xy * (rdash / r)
                #
                #     # Set the new direction vector for each beamlet,
                #     # the vector remains a unit vector
                #     dir_vec_div = np.array(
                #         [dir_vec_div_xy[0], dir_vec_div_xy[1], zdash]
                #     )
                #
                #     dir_vec_norm = dir_vec_div
                #
                #     beamlet_direction_x[i_los, count] = dir_vec_norm[0]
                #     beamlet_direction_y[i_los, count] = dir_vec_norm[1]
                #     beamlet_direction_z[i_los, count] = dir_vec_norm[2]
                #
                # else:
                #
                #     beamlet_direction_x[i_los, count] = dir_vec_norm[0]
                #     beamlet_direction_y[i_los, count] = dir_vec_norm[1]
                #     beamlet_direction_z[i_los, count] = dir_vec_norm[2]

            if debug:
                plt.figure()
                plt.plot(orig_x, orig_y, "kx")
                for i_beamlet in range(self.beamlets):
                    x_ = np.array(
                        [
                            beamlet_origin_x[i_los, i_beamlet],
                            beamlet_origin_x[i_los, i_beamlet]
                            + 1.0 * beamlet_direction_x[i_los, i_beamlet],
                        ]
                    )
                    y_ = np.array(
                        [
                            beamlet_origin_y[i_los, i_beamlet],
                            beamlet_origin_y[i_los, i_beamlet]
                            + 1.0 * beamlet_direction_y[i_los, i_beamlet],
                        ]
                    )
                    plt.plot(x_, y_)
                plt.show()

        self.beamlet_origin_x = beamlet_origin_x
        self.beamlet_origin_y = beamlet_origin_y
        self.beamlet_origin_z = beamlet_origin_z
        self.beamlet_direction_x = beamlet_direction_x
        self.beamlet_direction_y = beamlet_direction_y
        self.beamlet_direction_z = beamlet_direction_z

    # def set_weightings(
    #     self, weightings: LabeledArray,
    # ):
    #     """
    #
    #     Parameters
    #     ----------
    #     weightings
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     self.weightings = weightings

    def set_dl(
        self,
        dl: float,
    ):
        """
        Set spatial resolutions of the lines of sight, and calculate spatial
        coordinates along the LOS

        Parameters
        ----------
        dl
            Spatial resolution (m)
        """

        if hasattr(self, "rhop"):
            delattr(self, "rhop")

        # Calculate start and end coordinates, R, z and phi for all LOS
        x_start: list = []
        y_start: list = []
        z_start: list = []
        x_end: list = []
        y_end: list = []
        z_end: list = []
        for channel in self.x1:
            x_start.append([])
            y_start.append([])
            z_start.append([])
            x_end.append([])
            y_end.append([])
            z_end.append([])
            for beamlet in range(self.beamlets):
                # origin = (
                #     self.origin_x[channel] + self.delta_origin_x[channel, beamlet],
                #     self.origin_y[channel] + self.delta_origin_y[channel, beamlet],
                #     self.origin_z[channel] + self.delta_origin_z[channel, beamlet],
                # )
                # direction = (
                #     self.direction_x[channel]
                #     + self.delta_direction_x[channel, beamlet],
                #     self.direction_y[channel]
                #     + self.delta_direction_y[channel, beamlet],
                #     self.direction_z[channel]
                #     + self.delta_direction_z[channel, beamlet],
                # )
                origin = (
                    self.beamlet_origin_x[channel, beamlet],
                    self.beamlet_origin_y[channel, beamlet],
                    self.beamlet_origin_z[channel, beamlet],
                )
                direction = (
                    self.beamlet_direction_x[channel, beamlet],
                    self.beamlet_direction_y[channel, beamlet],
                    self.beamlet_direction_z[channel, beamlet],
                )
                _start, _end = find_wall_intersections(
                    origin, direction, machine_dimensions=self._machine_dims
                )
                x_start[channel].append(_start[0])
                y_start[channel].append(_start[1])
                z_start[channel].append(_start[2])
                x_end[channel].append(_end[0])
                y_end[channel].append(_end[1])
                z_end[channel].append(_end[2])

        self.x_start = DataArray(
            x_start,
            coords=[(self.x1_name, self.x1), ("beamlet", np.arange(0, self.beamlets))],
        )
        self.y_start = DataArray(
            y_start,
            coords=[(self.x1_name, self.x1), ("beamlet", np.arange(0, self.beamlets))],
        )
        self.z_start = DataArray(
            z_start,
            coords=[(self.x1_name, self.x1), ("beamlet", np.arange(0, self.beamlets))],
        )

        x_end = DataArray(
            x_end,
            coords=[(self.x1_name, self.x1), ("beamlet", np.arange(0, self.beamlets))],
        )
        y_end = DataArray(
            y_end,
            coords=[(self.x1_name, self.x1), ("beamlet", np.arange(0, self.beamlets))],
        )
        z_end = DataArray(
            z_end,
            coords=[(self.x1_name, self.x1), ("beamlet", np.arange(0, self.beamlets))],
        )

        # Fix identical length of all lines of sight
        los_lengths = np.sqrt(
            (x_end - self.x_start) ** 2
            + (y_end - self.y_start) ** 2
            + (z_end - self.z_start) ** 2
        )
        length = np.max(los_lengths)
        npts = int(np.ceil(length / dl))
        length = float(npts * dl)
        factor = length / los_lengths
        self.x_end = self.x_start + factor * (x_end - self.x_start)
        self.z_end = self.z_start + factor * (z_end - self.z_start)
        self.y_end = self.y_start + factor * (y_end - self.y_start)

        # Calculate coordinates, set to Nan values beyond nominal length
        x: list = []
        y: list = []
        z: list = []
        R: list = []
        phi: list = []
        _x2 = np.linspace(0, 1, npts, dtype=float)
        x2 = DataArray(_x2, coords=[(self.x2_name, _x2)])
        for x1 in self.x1:

            _x, _y = self.convert_to_xy(x1, x2, 0)
            _R, _z = self.convert_to_Rz(x1, x2, 0)
            dist = self.distance(self.x2_name, x1, x2, 0)

            x.append(_x)
            y.append(_y)
            z.append(_z)
            R.append(_R)
            _phi = np.arctan2(_y, _x)
            phi.append(_phi)

        # TODO: Loop over DataArray to NaN where los length > LOS distance
        for x1 in self.x1:

            for beamlet in range(self.beamlets):
                dist = self.distance(self.x2_name, x1, x2, 0)
                _x = x[x1].sel(beamlet=beamlet)
                _y = y[x1].sel(beamlet=beamlet)
                _z = z[x1].sel(beamlet=beamlet)
                _R = R[x1].sel(beamlet=beamlet)
                _phi = phi[x1].sel(beamlet=beamlet)
                x[x1][beamlet] = xr.where(
                    dist[beamlet, :] <= los_lengths[x1, beamlet].values, _x, np.nan
                )
                y[x1][beamlet] = xr.where(
                    dist[beamlet, :] <= los_lengths[x1, beamlet].values, _y, np.nan
                )
                z[x1][beamlet] = xr.where(
                    dist[beamlet, :] <= los_lengths[x1, beamlet].values, _z, np.nan
                )
                R[x1][beamlet] = xr.where(
                    dist[beamlet, :] <= los_lengths[x1, beamlet].values, _R, np.nan
                )
                phi[x1][beamlet] = xr.where(
                    dist[beamlet, :] <= los_lengths[x1, beamlet].values, _phi, np.nan
                )

        # Reset end coordinates to values intersecting the machine walls
        self.x_end = x_end
        self.y_end = y_end
        self.z_end = z_end

        self.x2 = x2
        self.dl = float(dist[0, 1] - dist[0, 0])
        self.x = xr.concat(x, "channel")
        self.y = xr.concat(y, "channel")
        self.z = xr.concat(z, "channel")
        self.phi = xr.concat(phi, "channel")
        self.R = np.sqrt(self.x**2 + self.y**2)
        self.impact_parameter = self.calc_impact_parameter()

    def check_rho_and_profile(
        self, profile_to_map: DataArray, t: LabeledArray = None, calc_rho: bool = False
    ) -> DataArray:
        """
        Check requested time
        """

        time = np.array(t)
        if time.size == 1:
            time = float(time)

        equil_t = self.equilibrium.rhop.t
        equil_ok = (np.min(time) >= np.min(equil_t)) * (np.max(time) <= np.max(equil_t))
        if not equil_ok:
            print(f"Available equilibrium time {np.array(equil_t)}")
            raise ValueError(
                f"Inserted time {time} is not available in Equilibrium object"
            )

        # Make sure rhop.t == requested time
        if not hasattr(self, "rhop") or calc_rho:
            self.convert_to_rho_theta(t=time)
        else:
            if not np.array_equal(self.rhop.t, time):
                self.convert_to_rho_theta(t=time)

        # Check profile
        if not hasattr(profile_to_map, "t"):
            profile = profile_to_map.expand_dims({"t": time})  # type: ignore
        else:
            profile = profile_to_map

        if np.size(time) == 1:
            if np.isclose(profile.t, time, rtol=1.0e-4):
                if "t" in profile_to_map.dims:
                    profile = profile.sel(t=time, method="nearest")
            else:
                raise ValueError("Profile does not include requested time")
        else:
            prof_t = profile.t
            range_ok = (np.min(time) >= np.min(prof_t)) * (
                np.max(time) <= np.max(prof_t)
            )
            if range_ok:
                profile = profile.interp(t=time)
            else:
                raise ValueError("Profile does not include requested time")

        return profile

    def map_profile_to_los(
        self,
        profile_to_map: DataArray,
        t: LabeledArray = None,
        limit_to_sep: bool = True,
        calc_rho: bool = False,
    ) -> DataArray:
        """
        Map profile to lines-of-sight

        Parameters
        ----------
        profile_to_map
            DataArray of the profile to integrate
        t
            Time for interpolation
        limit_to_sep
            Set to True if values outside of separatrix are to be set to 0
        calc_rho
            Calculate rho for specified time-points

        Returns
        -------
            Interpolation of the input profile along the LOS
        """
        self.check_equilibrium()
        profile = self.check_rho_and_profile(profile_to_map, t, calc_rho)

        dims = profile_to_map.dims
        along_los: DataArray
        if "R" in dims and "z" in dims:
            R_ = self.R
            z_ = self.z

            along_los = profile_to_map.interp(R=R_, z=z_).T
        elif "rhop" in dims:
            _rhop = self.rhop
            if "theta" in dims:
                theta_ = self.theta
                along_los = profile.interp(rhop=_rhop, theta=theta_)
            else:
                along_los = profile.interp(rhop=_rhop)

            if limit_to_sep:
                along_los = xr.where(
                    _rhop <= 1,
                    along_los,
                    np.nan,
                )
        else:
            raise NotImplementedError("Coordinates not recognized...")

        drop_dims = [dim for dim in dims if dim != "t"]
        along_los = along_los.drop_vars(drop_dims)
        self.along_los = along_los
        self.profile_to_map = profile_to_map

        return along_los

    def integrate_on_los(
        self,
        profile_to_map: DataArray,
        t: LabeledArray = None,
        limit_to_sep=True,
        calc_rho=False,
        sum_beamlets=True,
    ) -> DataArray:
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

        Returns
        -------
        Line of sight integral along the LOS
        """
        along_los = self.map_profile_to_los(
            profile_to_map,
            t=t,
            limit_to_sep=limit_to_sep,
            calc_rho=calc_rho,
        )

        if sum_beamlets:
            los_integral = (
                self.passes
                * along_los.sum(["los_position", "beamlet"], skipna=True)
                * self.dl
                / float(self.beamlets)
            )
        else:
            los_integral = (
                self.passes * along_los.sum(["los_position"], skipna=True) * self.dl
            )

        if len(los_integral.channel) == 1:
            los_integral = los_integral.sel(channel=0)

        self.los_integral = los_integral

        return los_integral

    def calc_impact_parameter(self):
        """Calculate the impact parameter in Cartesian space"""
        impact = []
        index = []
        x = []
        y = []
        z = []
        R = []
        for ch in self.x1:
            x_mean = self.x.sel(channel=ch).mean("beamlet")
            y_mean = self.y.sel(channel=ch).mean("beamlet")
            z_mean = self.z.sel(channel=ch).mean("beamlet")
            distance = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
            _index = np.unravel_index(distance.argmin(), distance.shape)
            _index_temp = distance.argmin()
            index.append(_index_temp)
            impact.append(distance[_index])
            x.append(x_mean[_index])
            y.append(y_mean[_index])
            z.append(z_mean[_index])
            R.append(np.sqrt(x_mean[_index] ** 2 + y_mean[_index] ** 2))

        impact = Dataset(
            {
                "index": xr.concat(index, "channel"),
                "value": xr.concat(impact, "channel"),
                "x": xr.concat(x, "channel"),
                "y": xr.concat(y, "channel"),
                "z": xr.concat(z, "channel"),
                "R": xr.concat(R, "channel"),
            }
        )

        return impact
