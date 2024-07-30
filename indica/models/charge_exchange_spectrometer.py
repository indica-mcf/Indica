from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters import TransectCoordinates
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.defaults.load_defaults import load_default_objects
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.utilities import assign_datatype, set_plot_rcparams


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
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "vtor":
                quantity = quant
                self.bckc[quantity] = self.Vtor_at_channels
            elif quant == "ti":
                quantity = quant
                self.bckc[quantity] = self.Ti_at_channels
            elif quant == "spectra":
                # Placeholder
                continue
            elif quant == "fit":
                # Placeholder
                continue
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

            error = xr.full_like(self.bckc[quantity], 0.0)
            stdev = xr.full_like(self.bckc[quantity], 0.0)
            self.bckc[quantity].attrs = {
                "transform": self.transform,
                "error": error,
                "stdev": stdev,
                "provenance": str(self),
            }
            assign_datatype(self.bckc[quantity], datatype)

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
        Dictionary of back-calculated quantities (identical to abstractreader.py)

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

        Ti_at_channels = self.transform.map_profile_to_rho(
            Ti, t=t, calc_rho=calc_rho
        )
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
            rho = Vtor.transform.rho.sel(t=t, method="nearest")
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
            rho = Ti.transform.rho.sel(t=t, method="nearest")
            plt.scatter(rho, Ti, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("Channel")
        plt.ylabel("Measured ion temperature (eV)")
        plt.legend()
        plt.show()

