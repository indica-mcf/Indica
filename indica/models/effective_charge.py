import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.available_quantities import READER_QUANTITIES
from indica.converters import LineOfSightTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
from indica.utilities import build_dataarrays
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_rcparams


class EffectiveCharge(AbstractDiagnostic):
    """
    Object representing a diagnostic to measure effective charge of the plasma
    """

    Zeff: DataArray
    los_integral_zeff: DataArray

    def __init__(self, name: str, instrument_method="get_zeff"):
        self.transform: LineOfSightTransform
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = READER_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        bckc = {
            "t": self.t,
            "channel": np.arange(len(self.transform.x1)),
            "location": self.transform.origin,
            "direction": self.transform.direction,
            "zeff_avrg": self.zeff_remapped,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
        self,
        Zeff: DataArray = None,
        t: LabeledArray = None,
        calc_rho=False,
        **kwargs,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Zeff
            Plasma effective charge profile
        t

        Returns
        -------

        """
        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Zeff = 1 + (
                (
                    self.plasma.ion_density
                    * self.plasma.meanz
                    * (self.plasma.meanz - 1)
                    / self.plasma.electron_density
                )
                .interp(t=t)
                .sum("element")
            )
        if Zeff is None:
            raise ValueError("Give inputs or assign plasma class!")
        self.t: DataArray = t
        self.Zeff: DataArray = Zeff

        zeff_remapped = self.transform.map_profile_to_los(
            Zeff,
            t=self.t,
            calc_rho=calc_rho,
        )
        self.zeff_remapped = zeff_remapped.mean("los_position")

        self._build_bckc_dictionary()
        self.bckc["zeff_avrg"] = self.bckc["zeff_avrg"].assign_coords(
            {"channel": zeff_remapped.channel}
        )
        return self.bckc

    def plot(self, nplot: int = 1):
        set_plot_rcparams("profiles")
        if len(self.bckc) == 0:
            print("No model results to plot")
            return

        # Line-of-sight information
        self.transform.plot()
        plt.figure()
        _value = self.bckc["dphi"]
        if "beamlet" in _value.dims:
            plt.fill_between(
                _value.t,
                _value.max("beamlet"),
                _value.min("beamlet"),
                alpha=0.5,
            )
            value = _value.mean("beamlet")
        else:
            value = _value
        value.plot()
        set_axis_sci()
        plt.title(self.name.upper())
        plt.xlabel("Time (s)")
        plt.ylabel("$$Z_eff$$")
        plt.legend()
