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

        # Print statements for debugging
        print(f'Beam = {self.name}')
        print(f'Energy = {self.energy} electron-volts')
        print(f'Power = {self.power} watts')
        print(f'Fractions (full, 1/2, 1/3, imp) = {self.fractions} %')
        print(f'divergence (x, y) = {self.divergence} rad')
        print(f'width (x, y) = {self.width} metres')


    def run_BBNBI(self):
        print('Add code to run BBNBI')

    def run_analytical_beam(
            self,
            x_dash: LabeledArray = DataArray(np.linspace(-1.0, 1.0, 501, dtype=float)),
            y_dash: LabeledArray = DataArray(np.linspace(-1.0, 1.0, 501, dtype=float)),
            z_dash: LabeledArray = DataArray(np.linspace(0.0, 1.0, 1001, dtype=float)),
    ):
        '''Analytical beam based on double gaussian formula'''

        # Calculate beam velocity
        v_beam = self.beam_velocity()
        

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
