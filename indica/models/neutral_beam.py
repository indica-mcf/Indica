from ..numpy_typing import LabeledArray
import numpy as np
from scipy.optimize import root
from xarray import DataArray
from matplotlib import pyplot as plt
from typing import Tuple

from ..converters.line_of_sight import LinesOfSightTransform


analytical_beam_defaults = {
    "element": "H",
    "amu": int(1),
    "energy": 25.0*1e3,
    "power": 500*1e3,
    "fractions": (0.7, 0.1, 0.2, 0.0),
    "divergence": (14*1e-3, 14e-3),
    "width": (0.025, 0.025),
    "location": (-0.3446, -0.9387, 0.0),
    "direction": (0.707, 0.707, 0.0)
}

class NeutralBeam:

    def __init__(self, name: str, use_defaults=True, **kwargs):
        '''Set beam parameters, initialisation'''
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
        print(f'Beam = {self.name}')
        print(f'Energy = {self.energy} electron-volts')
        print(f'Power = {self.power} watts')
        print(f'Fractions (full, 1/2, 1/3, imp) = {self.fractions} %')
        print(f'divergence (x, y) = {self.divergence} rad')
        print(f'width (x, y) = {self.width} metres')

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
            name=f'{self.name}_los',
            dl=0.01,
            machine_dimensions=machine_dimensions
        )

    def run_BBNBI(self):
        print('Add code to run BBNBI')

    def run_analytical_beam(
            self,
            x_dash: LabeledArray = DataArray(np.linspace(-0.25, 0.25, 101, dtype=float)),
            y_dash: LabeledArray = DataArray(np.linspace(-0.25, 0.25, 101, dtype=float))
    ):
        '''Analytical beam based on double gaussian formula'''

        # Calculate Attenuator
        x2 = self.transform.x2
        attenuation_factor = np.ones_like(x2)  # Replace with Attenuation Object in future, interact with plasma

        # Calculate beam velocity
        v_beam = self.beam_velocity()
        e = 1.602 * 1e-19

        # Neutral beam
        n_x = len(x_dash)
        n_y = len(y_dash)
        n_z = len(attenuation_factor)
        nb_dash = np.zeros((n_z, n_x, n_y), dtype=float)
        for i_z in range(n_z):
            for i_y in range(n_y):
                exp_factor = np.exp(-(x_dash**2 / self.width[0]**2) - (y_dash[i_y]**2 / self.width[1]**2))
                nb_dash[i_z, :, i_y] = self.power * attenuation_factor[i_z] * exp_factor / (np.pi * self.energy * e * np.prod(self.width) * v_beam)

        if True:

            plt.figure()
            plt.plot(x_dash, np.sum(nb_dash[0, :, :], axis=1))

            plt.figure()
            plt.contour(x_dash, y_dash, nb_dash[0, :, :], 100)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Initial beam cross section')
            plt.show(block=True)


    def beam_velocity(self):
        return 4.38*1e5 * np.sqrt(self.energy * 1e-3 / float(self.amu))

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
