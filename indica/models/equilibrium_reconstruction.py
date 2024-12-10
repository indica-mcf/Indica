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
    ):
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = READER_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}
        bckc = {
            "t": self.t,
            "wp": self.wp,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
        self,
        t=None,
        **kwargs,
    ):
        """Add docs"""
        if self.plasma is None:
            raise ValueError("plasma object is needed")

        if t is None:
            t = self.plasma.time_to_calculate

        check_time_present(t, self.plasma.wp.t)

        self.t = t
        self.wp = self.plasma.wp.sel(t=t)
        self._build_bckc_dictionary()
        return self.bckc
