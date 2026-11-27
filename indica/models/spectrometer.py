import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr

from indica.available_quantities import READER_QUANTITIES
from indica.converters import LineOfSightTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.utilities import build_dataarrays
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_rcparams


class Spectrometer(AbstractDiagnostic):
    """
    Base class factoring out universal(!?) properties of spectrmeters to be inherited
    by passive_spectrometer and beam_spectrometer (and other future spectrometers...)
    """

    def __init__(
        self,
        name: str,
        window: np.array,
        instrument_method="spectrometer",
    ):

        self.transform: LineOfSightTransform
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = READER_QUANTITIES[self.instrument_method]
        self.window = xr.DataArray(window, {"window": window})

    def __call__(self):
        """
        Returns spectrometer geometry

        """
        return self.bckc

    def _build_bckc_dictionary(self):
        bckc = {
            "channel": np.arange(len(self.transform.x1)),
            "wavelength": self.window.data,
            "location": self.transform.origin,
            "direction": self.transform.direction,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def plot(self):
        set_plot_rcparams("profiles")

        self.transform.plot(np.mean(self.t))

        plt.figure()
        channels = self.transform.x1
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(self.t), dtype=float))
        if "spectra" in self.bckc.keys():
            spectra = self.bckc["spectra"]
            if "channel" in spectra.dims:
                spectra = spectra.sel(channel=int(np.median(channels)))
            for i, t in enumerate(np.array(self.t, ndmin=1)):
                plt.plot(
                    spectra.wavelength,
                    spectra.sel(t=t),
                    color=cols_time[i],
                    label=f"t={t:1.2f} s",
                )
            plt.ylabel("Emissivity (photon/m^2/nm/s)")
            plt.xlabel("Wavelength (nm)")
            plt.legend()
            set_axis_sci()

        plt.show(block=True)
