from typing import Callable

from indica.defaults.load_defaults import load_default_objects
from indica.models import ChargeExchangeSpectrometer
from indica.models import EquilibriumReconstruction
from indica.models import HelikeSpectrometer
from indica.models import Interferometer
from indica.models import PinholeCamera
from indica.models import ThomsonScattering


class TestModels:
    """Test that the model calls run without error"""

    def setup_class(self):
        machine = "st40"
        self.machine = machine
        self.transforms = load_default_objects(machine, "geometry")
        self.equilibrium = load_default_objects(machine, "equilibrium")
        self.plasma = load_default_objects(machine, "plasma")
        self.plasma.set_equilibrium(self.equilibrium)

    def run_model(self, instrument: str, model: Callable):
        """
        Make sure model runs without errors
        """

        model = model(instrument)
        if instrument in self.transforms.keys():
            transform = self.transforms[instrument]
            if hasattr(transform, "set_equilibrium") and instrument != "efit":
                transform.set_equilibrium(self.equilibrium)
            model.set_transform(transform)
        model.set_plasma(self.plasma)

        bckc = model(sum_beamlets=False)

        assert type(bckc) == dict

    def test_interferometer(self):
        self.run_model("smmh", Interferometer)

    def test_thomson_scattering(self):
        self.run_model("ts", ThomsonScattering)

    def test_charge_exchange(self):
        self.run_model("cxff_pi", ChargeExchangeSpectrometer)

    def test_equilibrium(self):
        self.run_model("efit", EquilibriumReconstruction)

    def test_bolometer(self):
        self.run_model("blom_xy1", PinholeCamera)

    def test_helike_spectroscopy(self):
        self.run_model("xrcs", HelikeSpectrometer)
