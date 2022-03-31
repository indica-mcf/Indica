from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
from xarray import DataArray

from ..converters.line_of_sight import LinesOfSightTransform
from ..numpy_typing import LabeledArray


analytical_beam_defaults = {
    "element": "H",
    "amu": int(1),
    "energy": 25.0 * 1e3,
    "power": 500 * 1e3,
    "fractions": (0.7, 0.1, 0.2, 0.0),
    "divergence": (14 * 1e-3, 14e-3),
    "width": (0.025, 0.025),
    "location": (-0.3446, -0.9387, 0.0),
    "direction": (0.707, 0.707, 0.0),
}


class NeutralBeam:
    def __init__(self, name: str, use_defaults=True, **kwargs):
        """Set beam parameters, initialisation"""
        self.name = name  # Beam name

        # Use beam defaults
        if use_defaults:
            for (prop, default) in analytical_beam_defaults.items():
                setattr(self, prop, kwargs.get(prop, default))
        else:
            for (prop, default) in analytical_beam_defaults.items():
                setattr(self, prop, None)
            return

        # Set line_of_sight transform for centre of beam-line
        self.set_los_transform()

        # Set Attenuator
        self.attenuator = None

        # Print statements for debugging
        print(f"Beam = {self.name}")
        print(f"Energy = {self.energy} electron-volts")
        print(f"Power = {self.power} watts")
        print(f"Fractions (full, 1/2, 1/3, imp) = {self.fractions} %")
        print(f"divergence (x, y) = {self.divergence} rad")
        print(f"width (x, y) = {self.width} metres")

    def set_los_transform(
        self,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.175, 1.0),
            (-2.0, 2.0),
        ),
    ):
        self.transform = LinesOfSightTransform(
            self.location[0],
            self.location[1],
            self.location[2],
            self.direction[0],
            self.direction[1],
            self.direction[2],
            name=f"{self.name}_los",
            dl=0.01,
            machine_dimensions=machine_dimensions,
        )

    def run_BBNBI(self):
        print("Add code to run BBNBI")

    def run_analytical_beam(
        self,
        x_dash: LabeledArray = DataArray(np.linspace(-0.25, 0.25, 101, dtype=float)),
        y_dash: LabeledArray = DataArray(np.linspace(-0.25, 0.25, 101, dtype=float)),
    ):
        """Analytical beam based on double gaussian formula"""

        # Calculate Attenuator
        x2 = self.transform.x2
        attenuation_factor = np.ones_like(
            x2
        )  # Replace with Attenuation Object in future, interact with plasma

        # Calculate beam velocity
        # v_beam = self.beam_velocity()
        # e = 1.602 * 1e-19

        # Neutral beam
        n_x = 101
        n_y = 101
        n_z = len(attenuation_factor)
        nb_dash = np.zeros((n_z, n_x, n_y), dtype=float)
        # for i_z in range(n_z):
        #   for i_y in range(n_y):
        #       y_here = y_dash[i_y]
        #       exp_factor = np.exp(
        #           -(x_dash**2 / self.width[0]**2) - (y_here**2 / self.width[1]**2)
        #       )
        #       nb_dash[i_z, :, i_y] = \
        #           self.power * attenuation_factor[i_z] * exp_factor / \
        #           (np.pi * self.energy * e * np.prod(self.width) * v_beam)

        # mesh-grid of beam cross section coordinates
        z_dash = x2
        X_dash, Y_dash, Z_dash = np.meshgrid(x_dash, y_dash, z_dash)
        R_dash = np.sqrt(X_dash**2 + Y_dash**2)
        T_dash = np.arctan2(Y_dash, X_dash)

        x_transform = self.transform.x
        y_transform = self.transform.y
        z_transform = self.transform.z
        r_transform = np.sqrt(self.transform.x**2 + self.transform.y**2)
        theta_transform = np.arctan2(
            y_transform[1] - y_transform[0], x_transform[1] - x_transform[0]
        )
        phi_transform = np.arctan2(
            z_transform[1] - z_transform[0], r_transform[1] - r_transform[0]
        )
        theta_n = theta_transform + (np.pi / 2)
        phi_n = phi_transform + (np.pi / 2)

        xd = np.zeros_like(X_dash)
        yd = np.zeros_like(Y_dash)
        zd = np.zeros_like(Z_dash)
        for i_z in range(len(x2)):
            delta_X_dash = R_dash[:, :, i_z] * np.cos(T_dash[:, :, i_z])
            delta_Y_dash = R_dash[:, :, i_z] * np.sin(T_dash[:, :, i_z])

            xd[:, :, i_z] = x_transform[i_z].data + delta_X_dash * np.cos(
                theta_n.data
            )  # + X_dash[:, :, i_z]*np.tan(theta_n)
            yd[:, :, i_z] = y_transform[i_z].data + delta_X_dash * np.sin(
                theta_n.data
            )  # + Y_dash[:, :, i_z]*np.tan(theta_n)
            zd[:, :, i_z] = z_transform[i_z].data + delta_Y_dash * np.sin(phi_n.data)

        plt.figure()
        for i_z in range(len(x2)):
            plt.plot(xd[:, :, i_z].flatten(), yd[:, :, i_z].flatten(), "r.")
        plt.plot(self.transform.x, self.transform.y, "k")
        plt.axis("equal")
        plt.show(block=True)

        # print(X_dash)
        # print(np.shape(X_dash))
        # print("aa" ** 2)

        # # Interpolate over tok-grid
        # x_tok = DataArray(np.linspace(-1.0, 1.0, 101, dtype=float))
        # y_tok = DataArray(np.linspace(-1.0, 1.0, 101, dtype=float))
        # z_tok = DataArray(np.linspace(-0.5, 0.5, 51, dtype=float))
        # points = (z_tok.data, x_tok.data, y_tok.data)

        # X_tok, Y_tok, Z_tok = np.meshgrid(x_tok, y_tok, z_tok)
        # X_tok = X_tok.flatten()
        # Y_tok = Y_tok.flatten()
        # Z_tok = Z_tok.flatten()
        # nb_tok = np.zeros_like(X_tok)
        # for i in range(len(X_tok)):
        #     print(f'{i}out of {len(X_tok)}')
        #     point = np.array([Z_tok[i], X_tok[i], Y_tok[i]])
        #     nb_tok[i] = interpn(points, nb_dash, point, method='linear')

        # print("aa" ** 2)

        if True:

            plt.figure()
            plt.plot(x_dash, np.sum(nb_dash[0, :, :], axis=1))

            plt.figure()
            plt.contour(x_dash, y_dash, nb_dash[0, :, :], 100)
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.title("Initial beam cross section")
            plt.show(block=True)

    def beam_velocity(self):
        return 4.38 * 1e5 * np.sqrt(self.energy * 1e-3 / float(self.amu))

    def set_energy(self, energy: float):
        self.energy = energy

    def set_power(self, power: float):
        self.power = power

    def set_divergence(self, divergence: tuple):
        self.divergence = divergence

    def set_fractions(self, fractions: tuple):
        self.fractions = fractions

    def set_width(self, width: tuple):
        self.width = width

    def set_element(self, element: str):
        self.element = element
        if self.element == "H":
            self.amu = int(1)
        elif self.element == "D":
            self.amu = int(2)
        else:
            raise ValueError

    def set_location(self, location: tuple):
        self.location = location

    def set_direction(self, direction: tuple):
        self.direction = direction
