import matplotlib.pylab as plt
import xarray as xr
from xarray import DataArray

from indica.converters import LineOfSightTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
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
        self.los_transform: LineOfSightTransform
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "ne":
                quantity = quant
                self.bckc[quantity] = self.los_integral_ne
                error = xr.full_like(self.bckc[quantity], 0.0)
                stdev = xr.full_like(self.bckc[quantity], 0.0)
                self.bckc[quantity].attrs = {
                    "datatype": datatype,
                    "transform": self.los_transform,
                    "error": error,
                    "stdev": stdev,
                    "long_name": "Ne",
                    "units": "$m^{-2}$",
                }
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

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

        los_integral_ne = self.los_transform.integrate_on_los(
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
