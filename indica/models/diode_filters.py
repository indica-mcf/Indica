from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
import indica.physics as ph
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


class BremsstrahlungDiode(DiagnosticModel):
    """
    Object representing an diode filter diagnostic measuring
    Bremsstrahlung radiation in a specified wavelength range

    TODO: currently working only for Bremsstrahlung emission!!!
    """

    los_integral: DataArray

    def __init__(
        self,
        name: str,
        filter_wavelength: float = 530.0,
        filter_fwhm: float = 10,
        filter_type: str = "boxcar",
        etendue: float = 1.0,
        calibration: float = 2.0e-5,
        instrument_method="get_diode_filters",
    ):
        """
        Filtered diode diagnostic measuring Bremsstrahlung
        TODO: does not account for contaminating spectral lines

        Parameters
        ----------
        name
            Diagnostic name
        filter_wavelength
            Central wavelength of transmission filter
        filter_fwhm
            FWHM of transmission filter
        filter_type
            Function describing filter shape
        etendue
            Etendue of system
        calibration
            Absolute calibration to convert from Watts -> counts
        instrument_method
            Name of indica reader method to read experimental diagnostic data
        """
        self.name = name
        self.filter_wavelength = filter_wavelength
        self.filter_fwhm = filter_fwhm
        self.filter_type = filter_type
        self.etendue = etendue
        self.calibration = calibration
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "brightness":
                quantity = quant
                self.bckc[quantity] = self.los_integral
                error = xr.full_like(self.bckc[quantity], 0.0)
                stdev = xr.full_like(self.bckc[quantity], 0.0)
                self.bckc[quantity].attrs = {
                    "datatype": datatype,
                    "transform": self.los_transform,
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
            Zeff = self.plasma.zeff.interp(t=t).sum("element")
        else:
            if Ne is None or Te is None or Zeff is None:
                raise ValueError("Give inputs of assign plasma class!")

        self.t: DataArray = t
        self.Te: DataArray = Te
        self.Ne: DataArray = Ne
        self.Zeff: DataArray = Zeff

        # Wavelength axis
        wavelength = np.linspace(
            self.filter_wavelength - self.filter_fwhm * 2,
            self.filter_wavelength + self.filter_fwhm * 2,
        )
        self.wavelength = DataArray(wavelength, coords=[("wavelength", wavelength)])

        # Transmission filter function
        self.transmission = ph.make_window(
            wavelength,
            self.filter_wavelength,
            self.filter_fwhm,
            window=self.filter_type,
        )

        # Bremsstrahlung emission for each time, radial position and wavelength
        wlength = deepcopy(self.wavelength)
        for dim in Ne.dims:
            wlength = wlength.expand_dims(dim={dim: self.Ne[dim]})
        self.emission = ph.zeff_bremsstrahlung(Te, Ne, wlength, zeff=Zeff)

        los_integral = self.los_transform.integrate_on_los(
            (self.emission * self.transmission).integrate("wavelength"),
            t=t,
            calc_rho=calc_rho,
        )

        self.los_integral = los_integral

        self._build_bckc_dictionary()

        return self.bckc


def example_run(pulse: int = None, plasma=None, plot: bool = False):
    if plasma is None:
        plasma = example_plasma(pulse=pulse)

    # Create new interferometers diagnostics
    diagnostic_name = "diode_brems"
    los_start = np.array([[0.8, 0, 0], [0.8, 0, -0.1], [0.8, 0, -0.2]])
    los_end = np.array([[0.17, 0, 0], [0.17, 0, -0.25], [0.17, 0, -0.2]])
    origin = los_start
    direction = los_end - los_start
    los_transform = LineOfSightTransform(
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
    los_transform.set_equilibrium(plasma.equilibrium)
    model = BremsstrahlungDiode(
        diagnostic_name,
    )
    model.set_los_transform(los_transform)
    model.set_plasma(plasma)
    bckc = model()

    if plot:
        it = int(len(plasma.t) / 2)
        tplot = plasma.t[it]

        model.los_transform.plot_los(tplot, plot_all=True)

        # Plot back-calculated values
        plt.figure()
        cols_chan = cm.gnuplot2(
            np.linspace(0.1, 0.75, len(model.los_transform.x1), dtype=float)
        )
        for chan in model.los_transform.x1:
            bckc["brightness"].sel(channel=chan).plot(
                label=f"CH{chan}", color=cols_chan[chan]
            )
        plt.xlabel("Time (s)")
        plt.ylabel("Bremsstrahlung LOS-integrals (W/m^2)")
        plt.legend()

        # Plot the profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plt.plot(
                model.emission.rho_poloidal,
                model.emission.sel(t=t).integrate("wavelength"),
                color=cols_time[i],
                label=f"t={t:1.2f} s",
            )
        plt.xlabel("rho")
        plt.ylabel("Bremsstrahlung emission (W/m^3)")
        plt.legend()

    return plasma, model, bckc


if __name__ == "__main__":
    plt.ioff()
    example_run(plot=True)
    plt.show()
