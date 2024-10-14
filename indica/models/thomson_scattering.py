import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.available_quantities import READER_QUANTITIES
from indica.converters import TransectCoordinates
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
from indica.utilities import build_dataarrays


class ThomsonScattering(AbstractDiagnostic):
    """
    Object representing a Thomson scattering diagnostic
    """

    def __init__(
        self,
        name: str,
        instrument_method="get_thomson_scattering",
    ):
        self.transform: TransectCoordinates
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = READER_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        bckc = {
            "t": self.t,
            "channel": np.arange(len(self.transform.x1)),
            "ne": self.Ne_at_channels,
            "te": self.Te_at_channels,
            "x": self.transform.x,
            "y": self.transform.y,
            "z": self.transform.z,
            "R": self.transform.R,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
        self,
        Ne: DataArray = None,
        Te: DataArray = None,
        t: LabeledArray = None,
        calc_rho: bool = False,
        **kwargs,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile (dims = "rho", "t")
        Te
            Electron temperature profile (dims = "rho", "t")

        Returns
        -------
        Dictionary of back-calculated quantities (identical to abstractreader.py)

        """
        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Ne = self.plasma.electron_density.interp(t=t)
            Te = self.plasma.electron_temperature.interp(t=t)
        else:
            if Ne is None or Te is None:
                raise ValueError("Give inputs of assign plasma class!")

        self.t = t
        self.Ne = Ne
        self.Te = Te

        Ne_at_channels = self.transform.map_profile_to_rho(
            Ne,
            t=self.t,
            calc_rho=calc_rho,
        )
        Ne_at_channels = Ne_at_channels.assign_coords(
            R=("channel", self.transform.R.data)
        )
        Ne_at_channels = Ne_at_channels.assign_coords(
            z=("channel", self.transform.z.data)
        )
        Te_at_channels = self.transform.map_profile_to_rho(
            Te,
            t=self.t,
            calc_rho=calc_rho,
        )
        Te_at_channels = Te_at_channels.assign_coords(
            R=("channel", self.transform.R.data)
        )
        Te_at_channels = Te_at_channels.assign_coords(
            z=("channel", self.transform.z.data)
        )

        self.Ne_at_channels = Ne_at_channels
        self.Te_at_channels = Te_at_channels

        self._build_bckc_dictionary()

        return self.bckc

    def plot(self):
        if len(self.bckc) == 0:
            print("No model results to plot")
            return

        # Back-calculated profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, np.size(self.t), dtype=float))
        plt.figure()
        for i, t in enumerate(self.t):
            Ne = self.bckc["ne"].sel(t=t, method="nearest")
            rho = Ne.transform.rho.sel(t=t, method="nearest")
            plt.scatter(
                rho,
                Ne,
                color=cols_time[i],
                marker="o",
                alpha=0.7,
                label=f"t={t:1.2f} s",
            )
        plt.xlabel("Channel")
        plt.ylabel("Measured electron density (m^-3)")
        plt.legend()

        plt.figure()
        for i, t in enumerate(self.t):
            Te = self.bckc["te"].sel(t=t, method="nearest")
            rho = Te.transform.rho.sel(t=t, method="nearest")
            plt.scatter(
                rho,
                Te,
                color=cols_time[i],
                marker="o",
                alpha=0.7,
                label=f"t={t:1.2f} s",
            )
        plt.xlabel("Channel")
        plt.ylabel("Measured electron temperature (eV)")
        plt.legend()
