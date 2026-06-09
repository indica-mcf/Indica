from typing import Callable

import numpy as np

from indica.defaults.load_defaults import load_default_objects
from indica.models import ChargeExchangeSpectrometer
from indica.models import EquilibriumReconstruction
from indica.models import HelikeSpectrometer
from indica.models import Interferometer
from indica.models import PinholeCamera
from indica.models import ThomsonScattering
from indica.models.passive_spectrometer import format_pecs
from indica.models.passive_spectrometer import PassiveSpectrometer
from indica.models.passive_spectrometer import read_adf15s
from indica.operators.atomic_data import default_atomic_data


class TestModels:
    """Test that the model calls run without error"""

    def setup_class(self):
        machine = "st40"
        self.machine = machine
        self.transforms = load_default_objects(machine, "geometry")
        self.equilibrium = load_default_objects(machine, "equilibrium")
        self.plasma = load_default_objects(machine, "plasma")
        self.plasma.set_equilibrium(self.equilibrium)
        self.fz, self.power_loss = default_atomic_data(self.plasma.elements)

    def run_model(self, instrument: str, model: Callable, *args, **kwargs):
        """
        Make sure model runs without errors
        """

        _model = model(instrument, *args, **kwargs)
        if instrument in self.transforms.keys():
            transform = self.transforms[instrument]
            if hasattr(transform, "set_equilibrium") and instrument != "efit":
                transform.set_equilibrium(self.equilibrium)
            _model.set_transform(transform)
        _model.set_plasma(self.plasma)

        bckc = _model(sum_beamlets=False)

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

        self.run_model("blom_xy1", PinholeCamera, self.power_loss)

    def test_helike_spectroscopy(self):
        self.run_model("xrcs", HelikeSpectrometer)

    def test_passive_spectroscopy(self):
        pecs = read_adf15s(
            ["c"],
        )
        pec_database = format_pecs(pecs, wavelength_bounds=slice(100, 200))

        window = np.linspace(100, 200, 100)
        self.run_model("xrcs", PassiveSpectrometer, window=window, pecs=pec_database)

    def test_bolometer_poisson_noise(self):
        model = PinholeCamera("blom_xy1", self.power_loss)
        transform = self.transforms["blom_xy1"]
        if hasattr(transform, "set_equilibrium"):
            transform.set_equilibrium(self.equilibrium)
        model.set_transform(transform)
        model.set_plasma(self.plasma)

        clean_bckc = model(sum_beamlets=False)
        noisy_bckc = model(
            sum_beamlets=False,
            noise_model="poisson",
            noise_config={
                "target_quantity": "brightness",
                "typical_counts": 2500,
                "rng": np.random.default_rng(12345),
            },
        )

        clean = clean_bckc["brightness"].values
        noisy = noisy_bckc["brightness"].values
        raw = noisy_bckc["brightness_clean"].values

        assert np.allclose(clean, raw, equal_nan=True)

        finite_positive = np.isfinite(clean) & np.isfinite(noisy) & (clean > 0)
        assert np.any(finite_positive)

        rel_change = np.abs(
            (noisy[finite_positive] - clean[finite_positive]) / clean[finite_positive]
        )

        assert np.any(~np.isclose(noisy[finite_positive], clean[finite_positive]))
        assert rel_change.mean() > 1e-4
        assert rel_change.mean() < 0.2
