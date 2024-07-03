from copy import deepcopy

from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters import LineOfSightTransform
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_plasma
from indica.numpy_typing import LabeledArray
import indica.physics as ph
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.utilities import assign_datatype
from indica.utilities import format_coord


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
        filter_wavelength: float = 531.5,  # 532
        filter_fwhm: float = 1,  # 1
        filter_type: str = "boxcar",
        etendue: float = 1.0,
        calibration: float = 1,
        instrument_method="get_diode_filters",
        channel_mask: slice = None,  # =slice(18, 28),
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
        self.los_transform: LineOfSightTransform
        self.name = name
        self.filter_wavelength = filter_wavelength
        self.filter_fwhm = filter_fwhm
        self.filter_type = filter_type
        self.etendue = etendue
        self.calibration = calibration
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]
        self.channel_mask = channel_mask

        wavelength = np.linspace(
            self.filter_wavelength - self.filter_fwhm * 2,
            self.filter_wavelength + self.filter_fwhm * 2,
        )
        self.wavelength = format_coord(wavelength, "wavelength")

        # Transmission filter function
        transmission = ph.make_window(
            wavelength,
            self.filter_wavelength,
            self.filter_fwhm,
            window=self.filter_type,
        )
        self.transmission = DataArray(transmission, coords={"wavelength": wavelength})

    def integrate_spectra(self, spectra: DataArray, fit_background: bool = True):
        """
        Apply the diode transmission function to an input spectra

        Parameters
        ----------
        spectra
            Spectra with dimensions (channel, wavelength, time) in any order,
            and with units of W/m**2
        fit_background
            If True - background fitted and then integrated
            If False - spectra integrated using filter without any fitting

        Returns
        -------
        Background emission & integral of spectra using the filter transmission
        TODO: uncertainty on fit not calculated
        TODO: move spectral fitting to separate method outside of the class
        """

        # Interpolate transmission filter on spectral wavelength & restrict to > 0
        _transmission = self.transmission.interp(wavelength=spectra.wavelength)
        transmission = _transmission.where(_transmission > 1.0e-3, drop=True)
        wavelength_slice = slice(
            np.min(transmission.wavelength), np.max(transmission.wavelength)
        )

        _spectra = spectra.sortby("wavelength").sel(wavelength=wavelength_slice)
        if hasattr(_spectra, "error"):
            _spectra_err = spectra.error.sortby("wavelength").sel(
                wavelength=wavelength_slice
            )
        else:
            _spectra_err = xr.full_like(_spectra, 0.0)

        # Fit spectra to calculate background emission, filter and integrate
        if fit_background:
            fit = _spectra.polyfit("wavelength", 0)
            _spectra_fit = fit.polyfit_coefficients.sel(degree=0)
            spectra_to_integrate = _spectra_fit.expand_dims(
                dim={"wavelength": _spectra.wavelength}
            )
            spectra_to_integrate_err = xr.full_like(spectra_to_integrate, 0.0)
        else:
            spectra_to_integrate = _spectra
            spectra_to_integrate_err = _spectra_err

        spectra_to_integrate.attrs["error"] = spectra_to_integrate_err

        integral = (spectra_to_integrate * transmission).sum("wavelength")
        integral_err = (np.sqrt((spectra_to_integrate_err * transmission) ** 2)).sum(
            "wavelength"
        )
        integral.attrs["error"] = integral_err

        return spectra_to_integrate, integral

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
                    "transform": self.los_transform,
                    "error": error,
                    "stdev": stdev,
                    "provenance": str(self),
                }
                assign_datatype(self.bckc[quantity], datatype)
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
        **kwargs,
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
        TODO: emission needs a new name as it's in units [W m**-2 nm**-1]
        """

        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Ne = self.plasma.electron_density.sel(t=t)
            Te = self.plasma.electron_temperature.sel(t=t)
            Zeff = self.plasma.zeff.sel(t=t).sum("element")
        else:
            if Ne is None or Te is None or Zeff is None:
                raise ValueError("Give inputs of assign plasma class!")

        self.t: DataArray = t
        self.Te: DataArray = Te
        self.Ne: DataArray = Ne
        self.Zeff: DataArray = Zeff

        # Bremsstrahlung emission for each time, radial position and wavelength
        wlength = deepcopy(self.wavelength)
        for dim in Ne.dims:
            wlength = wlength.expand_dims(dim={dim: self.Ne[dim]})
        self.emission = ph.zeff_bremsstrahlung(Te, Ne, wlength, zeff=Zeff)
        self.emissivity = (self.emission * self.transmission).integrate("wavelength")
        los_integral = self.los_transform.integrate_on_los(
            self.emissivity,
            t=t,
            calc_rho=calc_rho,
        )
        if self.channel_mask is not None:
            los_integral = los_integral.where(
                (los_integral.channel > self.channel_mask.start)
                & (los_integral.channel < self.channel_mask.stop)
            )

        self.los_integral = los_integral

        self._build_bckc_dictionary()
        return self.bckc


def example_geometry(nchannels: int = 12):

    los_end = np.full((nchannels, 3), 0.0)
    los_end[:, 0] = 0.0
    los_end[:, 1] = np.linspace(-0.2, -1, nchannels)
    los_end[:, 2] = 0.0
    los_start = np.array([[1.5, 0, 0]] * los_end.shape[0])
    origin = los_start
    direction = los_end - los_start

    return origin, direction


def example_run(
    pulse: int = None, nchannels: int = 12, plasma=None, plot: bool = False
):
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

    # Create new interferometers diagnostics
    diagnostic_name = "diode_brems"
    origin, direction = example_geometry(nchannels=nchannels)
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
    model.set_transform(los_transform)
    model.set_plasma(plasma)
    bckc = model()

    if plot:
        it = int(len(plasma.t) / 2)
        tplot = plasma.t[it].values

        model.los_transform.plot(tplot)

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
                model.emissivity.rho_poloidal,
                model.emissivity.sel(t=t),
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
