import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.transect import TransectCoordinates
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


class ChargeExchange(DiagnosticModel):
    """
    Object representing a CXRS diagnostic
    """

    def __init__(
        self,
        name: str,
        element: str = "c",
        instrument_method="get_charge_exchange",
    ):

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
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

            error = xr.full_like(self.bckc[quantity], 0.0)
            stdev = xr.full_like(self.bckc[quantity], 0.0)
            self.bckc[quantity].attrs = {
                "datatype": datatype,
                "transform": self.transect_transform,
                "error": error,
                "stdev": stdev,
                "provenance": str(self),
            }

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

        if "element" in Vtor.dims:
            Vtor = Vtor.sel(element=self.element)
        if "element" in Ti.dims:
            Ti = Ti.sel(element=self.element)

        self.t = t
        self.Vtor = Vtor
        self.Ti = Ti

        Ti_at_channels = self.transect_transform.map_to_rho(
            Ti,
            t=t,
            calc_rho=calc_rho
        )
        Vtor_at_channels = self.transect_transform.map_to_rho(
            Vtor,
            t=t,
            calc_rho=calc_rho
        )

        self.Ti_at_channels = Ti_at_channels
        self.Vtor_at_channels = Vtor_at_channels

        self._build_bckc_dictionary()

        return self.bckc


def example_run(
    pulse:int=None,
    diagnostic_name: str = "cxrs",
    plasma=None,
    plot=False,
):

    # TODO: LOS sometimes crossing bad EFIT reconstruction

    if plasma is None:
        plasma = example_plasma(pulse=pulse)

    # Create new interferometers diagnostics
    nchannels = 5
    x_positions = np.linspace(0.2, 0.8, nchannels)
    y_positions = np.linspace(0.0, 0.0, nchannels)
    z_positions = np.linspace(0.0, 0.0, nchannels)

    transect_transform = TransectCoordinates(
        x_positions,
        y_positions,
        z_positions,
        diagnostic_name,
        machine_dimensions=plasma.machine_dimensions,
    )
    transect_transform.set_equilibrium(plasma.equilibrium)
    model = ChargeExchange(
        diagnostic_name,
    )
    model.set_transect_transform(transect_transform)
    model.set_plasma(plasma)

    bckc = model()

    if plot:
        it = int(len(plasma.t) / 2)
        tplot = plasma.t[it]

        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))

        model.transect_transform.plot_los(tplot, plot_all=True)

        # Plot back-calculated profiles
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.toroidal_rotation.sel(t=t, element=model.element).plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Vtor = bckc["vtor"].sel(t=t, method="nearest")
            plt.scatter(
                Vtor.rho_poloidal, Vtor, color=cols_time[i], marker="o", alpha=0.7
            )
        plt.xlabel("Channel")
        plt.ylabel("Measured toroidal rotation (rad/s)")
        plt.legend()

        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.ion_temperature.sel(t=t, element=model.element).plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Ti = bckc["ti"].sel(t=t, method="nearest")
            plt.scatter(Ti.rho_poloidal, Ti, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("Channel")
        plt.ylabel("Measured ion temperature (eV)")
        plt.legend()
        plt.show()

    return plasma, model, bckc

if __name__ == "__main__":
    example_run(plot=True)