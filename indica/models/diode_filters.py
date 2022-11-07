from indica.converters.lines_of_sight import LinesOfSightTransform
from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters import FluxSurfaceCoordinates
from indica.numpy_typing import LabeledArray
from xarray import DataArray
import indica.physics as ph


class Diode_filter:
    """
    Object representing an diode filter diagnostic measuring
    in a specified spectral range

    TODO: currently working only for Bremsstrahlung emission!!!
    """

    def __init__(
        self,
        name: str,
        filter_wavelength: float = 530.0,
        filter_fwhm: float = 10,
        filter_shape: str = "tophat",
        etendue: float = 1.0,
        calibration: float = 2.0e-5,
    ):
        self.name = name
        self.filter_wavelength = filter_wavelength
        self.filter_fwhm = filter_fwhm
        self.filter_shape = filter_shape
        self.etendue = etendue
        self.calibration = calibration

    def set_los_transform(self, transform: LinesOfSightTransform):
        """
        Parameters
        ----------
        transform
            line of sight transform of the modelled diagnostic
        passes
            number of passes along the line of sight
        """
        self.los_transform = transform

    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        """
        set flux surface transform for flux mapping of the line of sight
        """
        self.los_transform.set_flux_transform(flux_transform)

    def calculate_emission(
        self, Te: DataArray, Ne: DataArray, Zeff: DataArray,
    ):
        """
        Calculate Bremsstrahlung emission

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

        Returns
        -------

        """

        emission = ph.zeff_bremsstrahlung(Te, Ne, self.filter_wavelength, zeff=Zeff)
        self.emission = emission

        return emission

    def map_to_los(self, t: LabeledArray = None):
        """
        Map emission to LOS

        Parameters
        ----------
        t
            time (s)

        Returns
        -------
        Return emission along line of sight

        """
        if not hasattr(self, "emission"):
            raise Exception("Calculate emission characteristics before mapping to LOS")

        along_los = self.los_transform.map_to_los(self.emission, t=t)

        self.along_los = along_los
        return along_los

    def integrate_on_los(self, t: LabeledArray = None):
        """
        Calculate the integral along the line of sight
        For line intensities, the units are W sterad^-1 m^-2

        Parameters
        ----------
        t
            time (s)

        Returns
        -------
        Return line integral and interpolated density along the line of sight

        """
        if not hasattr(self, "emission"):
            raise Exception("Calculate emission characteristics before mapping to LOS")

        along_los = {}
        los_integral, along_los = self.los_transform.integrate_on_los(
            self.emission, t=t
        )
        los_integral = los_integral * self.etendue * self.calibration

        self.along_los = along_los
        self.los_integral = los_integral
        return los_integral, along_los
