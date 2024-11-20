from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.available_quantities import READER_QUANTITIES
from indica.converters import TransectCoordinates
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
from indica.utilities import build_dataarrays
from indica.utilities import set_plot_rcparams


class ChargeExchangeSpectrometer(AbstractDiagnostic):
    """
    Object representing a CXRS diagnostic
    """

    def __init__(
        self,
        name: str,
        element: str = "c",
        instrument_method="get_charge_exchange",
    ):
        self.transform: TransectCoordinates
        self.name = name
        self.element = element
        self.instrument_method = instrument_method
        self.quantities = READER_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}
        bckc = {
            "t": self.t,
            "channel": np.arange(len(self.transform.x1)),
            "x": self.transform.x,
            "y": self.transform.y,
            "z": self.transform.y,
            "R": self.transform.R,
            "vtor": self.Vtor_at_channels,
            "ti": self.Ti_at_channels,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
        self,
        Ti: DataArray = None,
        Vtor: DataArray = None,
        t: LabeledArray = None,
        calc_rho: bool = False,
        **kwargs,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ti
            Ion temperature profile (dims = "rho", "t")
        Vtor
            Toroidal rotation profile (dims = "rho", "t")

        Returns
        -------
        Dictionary of back-calculated quantities (identical to datareader.py)

        """
        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Ti = self.plasma.ion_temperature.interp(t=t)
            Vtor = self.plasma.toroidal_rotation.interp(t=t)
        else:
            if Ti is None or Vtor is None:
                raise ValueError("Give inputs or assign plasma class!")

        self.t = t
        self.Vtor = Vtor
        self.Ti = Ti

        Ti_at_channels = self.transform.map_profile_to_rho(Ti, t=t, calc_rho=calc_rho)
        Vtor_at_channels = self.transform.map_profile_to_rho(
            Vtor, t=t, calc_rho=calc_rho
        )

        self.Ti_at_channels = Ti_at_channels
        self.Vtor_at_channels = Vtor_at_channels

        self._build_bckc_dictionary()

        return self.bckc

    def plot(self, nplot: int = 1):
        set_plot_rcparams("profiles")

        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(self.t), dtype=float))

        self.transform.plot()

        # Plot back-calculated profiles
        plt.figure()
        for i, t in enumerate(self.t):
            if i % nplot:
                continue
            Vtor = self.bckc["vtor"].sel(t=t, method="nearest")
            rho = Vtor.transform.rhop.sel(t=t, method="nearest")
            plt.scatter(
                rho,
                Vtor,
                color=cols_time[i],
                marker="o",
                alpha=0.7,
                label=f"t={t:1.2f} s",
            )
        plt.xlabel("Channel")
        plt.ylabel("Measured toroidal rotation (rad/s)")
        plt.legend()

        plt.figure()
        for i, t in enumerate(self.t):
            if i % nplot:
                continue
            Ti = self.bckc["ti"].sel(t=t, method="nearest")
            rho = Ti.transform.rhop.sel(t=t, method="nearest")
            plt.scatter(rho, Ti, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("Channel")
        plt.ylabel("Measured ion temperature (eV)")
        plt.legend()
        plt.show()
