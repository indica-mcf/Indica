from indica.available_quantities import READER_QUANTITIES
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.utilities import build_dataarrays
from indica.utilities import check_time_present


class EquilibriumReconstruction(AbstractDiagnostic):
    """Object representing observations from a magnetic reconstruction"""

    def __init__(
        self,
        name: str,
        instrument_method="get_equilibrium",
        noise_model: str | None = None,
        noise_config: dict | None = None,
    ):
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = READER_QUANTITIES[self.instrument_method]
        self.noise_model = noise_model
        self.noise_config = {} if noise_config is None else dict(noise_config)
        self._call_noise_model = self.noise_model
        self._call_noise_config = self.noise_config

    def _build_bckc_dictionary(self):
        self.bckc = {}
        bckc = {
            "t": self.t,
            "wp": self.wp,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)
        if self._call_noise_model is not None:
            self.apply_noise(
                noise_model=self._call_noise_model,
                noise_config=self._call_noise_config,
            )

    def __call__(
        self, t=None, noise_model: str | None = None, noise_config: dict | None = None
    ):
        """Add docs"""
        if self.plasma is None:
            raise ValueError("plasma object is needed")

        if t is None:
            t = self.plasma.time_to_calculate

        check_time_present(t, self.plasma.wp.t)

        self.t = t
        self.wp = self.plasma.wp.sel(t=t)
        self._call_noise_model = (
            self.noise_model if noise_model is None else noise_model
        )
        self._call_noise_config = (
            self.noise_config if noise_config is None else noise_config
        )
        self._build_bckc_dictionary()
        return self.bckc
