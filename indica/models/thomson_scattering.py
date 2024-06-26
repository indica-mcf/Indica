import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters import TransectCoordinates
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_plasma
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.utilities import assign_datatype


class ThomsonScattering(DiagnosticModel):
    """
    Object representing a Thomson scattering diagnostic
    """

    def __init__(
        self,
        name: str,
        instrument_method="get_thomson_scattering",
    ):
        self.transect_transform: TransectCoordinates
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "ne":
                quantity = quant
                self.bckc[quantity] = self.Ne_at_channels
            elif quant == "te":
                quantity = quant
                self.bckc[quantity] = self.Te_at_channels
            elif quant == "chi2":
                # Placeholder
                continue
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

            error = xr.full_like(self.bckc[quantity], 0.0)
            stdev = xr.full_like(self.bckc[quantity], 0.0)
            self.bckc[quantity].attrs = {
                "transform": self.transect_transform,
                "error": error,
                "stdev": stdev,
                "provenance": str(self),
            }
            assign_datatype(self.bckc[quantity], datatype)

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

        Ne_at_channels = self.transect_transform.map_profile_to_rho(
            Ne,
            t=self.t,
            calc_rho=calc_rho,
        )
        Ne_at_channels = Ne_at_channels.assign_coords(
            R=("channel", self.transect_transform.R.data)
        )
        Ne_at_channels = Ne_at_channels.assign_coords(
            z=("channel", self.transect_transform.z.data)
        )
        Te_at_channels = self.transect_transform.map_profile_to_rho(
            Te,
            t=self.t,
            calc_rho=calc_rho,
        )
        Te_at_channels = Te_at_channels.assign_coords(
            R=("channel", self.transect_transform.R.data)
        )
        Te_at_channels = Te_at_channels.assign_coords(
            z=("channel", self.transect_transform.z.data)
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


def ts_transform_example(nchannels):
    x_positions = np.linspace(0.2, 0.8, nchannels)
    y_positions = np.linspace(0.0, 0.0, nchannels)
    z_positions = np.linspace(0.0, 0.0, nchannels)
    transform = TransectCoordinates(
        x_positions,
        y_positions,
        z_positions,
        "ts",
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
    )
    return transform


def example_run(
    pulse: int = None,
    diagnostic_name: str = "ts",
    plasma=None,
    plot=False,
):

    # TODO: LOS sometime crossing bad EFIT reconstruction and crashing...

    if plasma is None:
        from indica.equilibrium import fake_equilibrium

        plasma = example_plasma(pulse=pulse)
        machine_dims = plasma.machine_dimensions
        equilibrium = fake_equilibrium(
            tstart=plasma.tstart,
            tend=plasma.tend,
            dt=plasma.dt / 2.0,
            machine_dims=machine_dims,
        )
        plasma.set_equilibrium(equilibrium)

    transect_transform = ts_transform_example(11)
    transect_transform.set_equilibrium(plasma.equilibrium)
    model = ThomsonScattering(
        diagnostic_name,
    )
    model.set_transform(transect_transform)
    model.set_plasma(plasma)

    bckc = model()

    if plot:
        it = int(len(plasma.t) / 2)
        tplot = plasma.t[it].values

        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))

        model.transect_transform.plot(tplot)

        # Plot back-calculated profiles
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.electron_density.sel(t=t).plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Ne = bckc["ne"].sel(t=t, method="nearest")
            rho = Ne.transform.rho.sel(t=t, method="nearest")
            plt.scatter(rho, Ne, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("Channel")
        plt.ylabel("Measured electron density (m^-3)")
        plt.legend()

        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.electron_temperature.sel(t=t).plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Te = bckc["te"].sel(t=t, method="nearest")
            rho = Te.transform.rho.sel(t=t, method="nearest")
            plt.scatter(rho, Te, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("Channel")
        plt.ylabel("Measured electron temperature (eV)")
        plt.legend()

    return plasma, model, bckc


if __name__ == "__main__":
    plt.ioff()
    example_run(plot=True)
    plt.show()
