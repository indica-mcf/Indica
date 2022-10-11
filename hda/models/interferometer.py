from indica.converters.lines_of_sight import LinesOfSightTransform
from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters import FluxSurfaceCoordinates
from indica.numpy_typing import LabeledArray


class Interferometer:
    """
    Object representing an interferometer diagnostics
    """

    def __init__(self, name: str):
        self.name = name

    def set_los_transform(self, transform: LinesOfSightTransform, passes:int=2):
        """
        Parameters
        ----------
        transform
            line of sight transform of the modelled diagnostic
        passes
            number of passes along the line of sight
        """
        self.los_transform = transform
        self.passes = passes

    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        """
        set flux surface transform for flux mapping of the line of sight
        """
        self.los_transform.set_flux_transform(flux_transform)

    def map_to_los(self, Ne: LabeledArray, t: LabeledArray = None):
        """
        Map interferometer measurements along line of sight

        Parameters
        ----------
        Ne
            1D profile of the electron density
        t
            time (s)

        Returns
        -------
        Return line integral and interpolated density along the line of sight

        """
        along_los = {}
        along_los["ne"] = self.los_transform.map_to_los(Ne, t=t)
        return along_los

    def integrate_on_los(self, Ne: LabeledArray, t: LabeledArray = None):
        """
        Calculate the integral of the interferometer measurement along the line of sight

        Parameters
        ----------
        Ne
            1D profile of the electron density
        t
            time (s)

        Returns
        -------
        Return line integral and interpolated density along the line of sight

        """
        along_los = {}
        los_integral = {}
        los_integral["ne"], along_los["ne"] = self.los_transform.integrate_on_los(Ne, t=t, passes=self.passes)

        return los_integral, along_los

    def line_integrated_phase_shift(self, Ne: LabeledArray, t: LabeledArray = None):
        raise NotImplementedError("Calculation of phase shift still not implemented")
