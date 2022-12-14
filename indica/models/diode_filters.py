import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight_multi import LineOfSightTransform
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
import indica.physics as ph
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


class Bremsstrahlung_filtered_diode(DiagnosticModel):
    """
    Object representing an diode filter diagnostic measuring
    Bremsstrahlung radiation in a specified wavelength range

    TODO: currently working only for Bremsstrahlung emission!!!
    """

    transform: LineOfSightTransform
    los_integral: DataArray

    def __init__(
        self,
        name: str,
        filter_wavelength: float = 530.0,
        filter_fwhm: float = 10,
        filter_shape: str = "tophat",
        etendue: float = 1.0,
        calibration: float = 2.0e-5,
        instrument_method="get_diode_filters",
    ):
        self.name = name
        self.filter_wavelength = filter_wavelength
        self.filter_fwhm = filter_fwhm
        self.filter_shape = filter_shape
        self.etendue = etendue
        self.calibration = calibration
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "brems":
                quantity = quant
                self.bckc[quantity] = self.los_integral
                error = xr.full_like(self.bckc[quantity], 0.0)
                stdev = xr.full_like(self.bckc[quantity], 0.0)
                self.bckc[quantity].attrs = {
                    "datatype": datatype,
                    "transform": self.transform,
                    "error": error,
                    "stdev": stdev,
                    "provenance": str(self),
                }
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

    def __call__(
        self,
        Te: DataArray = None,
        Ne: DataArray = None,
        Zeff: DataArray = None,
        t: LabeledArray = None,
        calc_rho: bool = False,
    ):
        """
        Calculate Bremsstrahlung emission and model measurement

        TODO: add set of spectral lines to model different line diodes

        Parameters
        ----------
        Te
            electron temperature
        Ne
            electron density
        Zeff
            Total effective charge
        t
            time
        """

        if self.plasma is not None:
            if t is None:
                t = self.plasma.t
            Ne = self.plasma.electron_density.interp(t=t)
            Te = self.plasma.electron_temperature.interp(t=t)
            Zeff = self.plasma.zeff.interp(t=t)
        else:
            if Ne is None or Te is None or Zeff is None:
                raise ValueError("Give inputs of assign plasma class!")

        self.t = t
        self.Te = Te
        self.Ne = Ne
        self.Zeff = Zeff

        emission = ph.zeff_bremsstrahlung(Te, Ne, self.filter_wavelength, zeff=Zeff)
        self.emission = emission

        los_integral = self.transform.integrate_on_los(
            emission.sum("element"),
            t=t,
            calc_rho=calc_rho,
        )

        self.los_integral = los_integral

        self._build_bckc_dictionary()

        return self.bckc


def example_run(plasma=None, plot: bool = False):
    if plasma is None:
        plasma = example_plasma()

    # Create new interferometers diagnostics
    diagnostic_name = "diode_brems"
    los_start = np.array([[0.8, 0, 0], [0.8, 0, -0.1], [0.8, 0, -0.2]])
    los_end = np.array([[0.17, 0, 0], [0.17, 0, -0.25], [0.17, 0, -0.2]])
    origin = los_start
    direction = los_end - los_start
    transform = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name=diagnostic_name,
        machine_dimensions=plasma.machine_dimensions,
        passes=1,
    )
    transform.set_equilibrium(plasma.equilibrium)
    model = Bremsstrahlung_filtered_diode(
        diagnostic_name,
    )
    model.set_transform(transform)
    model.set_plasma(plasma)
    bckc = model()

    if plot:
        it = int(len(plasma.t) / 2)
        tplot = plasma.t[it]

        plt.figure()
        plasma.equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
            levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        )
        channels = model.transform.x1
        cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(channels), dtype=float))
        for chan in channels:
            plt.plot(
                model.transform.R[chan],
                model.transform.z[chan],
                linewidth=3,
                color=cols[chan],
                alpha=0.7,
                label=f"CH{chan}",
            )

        plt.xlim(0, 1.0)
        plt.ylim(-0.6, 0.6)
        plt.axis("scaled")
        plt.legend()

        # Plot LOS mapping on equilibrium
        plt.figure()
        for chan in channels:
            model.transform.rho[chan].sel(t=tplot, method="nearest").plot(
                color=cols[chan],
                label=f"CH{chan}",
            )
        plt.xlabel("Path along the LOS")
        plt.ylabel("Rho-poloidal")
        plt.legend()

        # Plot back-calculated values
        plt.figure()
        for chan in channels:
            bckc["brems"].sel(channel=chan).plot(label=f"CH{chan}", color=cols[chan])
        plt.xlabel("Time (s)")
        plt.ylabel("Bremsstrahlung LOS-integrals (W/m^2)")
        plt.legend()

        # Plot the profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plt.plot(
                model.emission.sum("element").rho_poloidal,
                model.emission.sum("element").sel(t=t),
                color=cols_time[i],
                label=f"t={t:1.2f} s",
            )
        plt.xlabel("rho")
        plt.ylabel("Bremsstrahlung emission (W/m^3)")
        plt.legend()

    return plasma, model, bckc
