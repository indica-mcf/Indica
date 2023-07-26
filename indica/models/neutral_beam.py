from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run


class NeutralBeam(DiagnosticModel):
    def __init__(
        self,
        name: str = 'rfx',
        element: str = "d",
        amu: float = 2.014,
        energy: float = 24.0 * 1e3,
        power: float = 500.0 * 1e3,
        fractions: tuple = (0.62, 0.24, 0.14, 0.0),
        div_x: float = 14 * 1e-3,
        div_y: float = 14 * 1e-3,
        width_x: float = 0.025,
        width_y: float = 0.025,
        debug: bool = True,
    ):
        """Set beam parameters, initialisation"""
        self.name = name  # Beam name
        self.element = element  # Element
        self.amu = amu  # Atomic mass
        self.energy = energy  # Beam energy (eV)
        self.power = power  # Beam power (W)
        self.fractions = fractions  # Beam fractions (%)
        self.div_x = div_x  # Beam divergence in x (rad)
        self.div_y = div_y  # Beam divergence in y (rad)
        self.width_x = width_x  # Beam 1/e width in x (metres)
        self.width_y = width_y  # Beam 1/e width in y (metres)

        # Set Attenuator
        self.attenuator = None

        if debug:
            # Print statements for debugging
            print(f"Beam = {self.name}")
            print(f"Energy = {self.energy} electron-volts")
            print(f"Power = {self.power} watts")
            print(f"Fractions (full, 1/2, 1/3, imp) = {self.fractions} %")
            print(f"divergence (x, y) = ({self.div_x},{self.div_y}) rad")
            print(f"width (x, y) = ({self.width_x},{self.width_y}) metres")

    def _build_bckc_dictionary(self):
        print('To be implemented!')
        return

    def __call__(
            self,
    ):
        print('To be implemented!')
        return


if __name__ == '__main__':

    # Generate neutral beam model
    beam = NeutralBeam()

    # Set plasma
    plasma = example_run()
    beam.set_plasma(plasma)
