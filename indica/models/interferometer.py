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


class Interferometer(AbstractDiagnostic):
    """
    Object representing an interferometer diagnostics
    """

    Ne: DataArray
    los_integral_ne: DataArray

    def __init__(
        self,
        name: str,
        instrument_method="get_interferometry",
    ):
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
            "ne": self.los_integral_ne,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
        self, Ne: DataArray = None, t: LabeledArray = None, calc_rho=False, **kwargs
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile
        t

        Returns
        -------

        """
        # TODO: decide whether to select nearest time-points or interpolate in time!!
        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Ne = self.plasma.electron_density.interp(t=t)
        else:
            if Ne is None:
                raise ValueError("Give inputs or assign plasma class!")
        self.t: DataArray = t
        self.Ne: DataArray = Ne

        los_integral_ne = self.transform.integrate_on_los(
            Ne,
            t=self.t,
            calc_rho=calc_rho,
        )
        self.los_integral_ne = los_integral_ne

        self._build_bckc_dictionary()
        return self.bckc

    def plot(self, nplot: int = 1):
        set_plot_rcparams("profiles")
        if len(self.bckc) == 0:
            print("No model results to plot")
            return

        # Line-of-sight information
        self.transform.plot()
        plt.figure()
        _value = self.bckc["ne"]
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
        plt.ylabel("Measured LOS-integrated density (m^-2)")
        plt.legend()
