from indica.converters import lines_of_sight, line_of_sight
from indica.converters import FluxSurfaceCoordinates
from indica.numpy_typing import LabeledArray


class Interferometer:
    """
    Object representing an interferometer diagnostics
    """

    def __init__(self, name: str):
        self.name = name

    def set_los_transform(self, transform: lines_of_sight, passes:int=2):
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

    def map_los(self, t: LabeledArray = None, npts=100):
        """
        Map line of sight on equilibrium reconstruction

        Parameters
        ----------
        t
            time (s)
        npts
            number of points along the line of sight
        """
        self.los_transform.remap_los(t=t, npts=npts)

    def line_integrated_density(self, Ne: LabeledArray, t: LabeledArray = None):
        """
        Calculate the integral of the electron density along the line of sight

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
        los_integral, along_los = self.los_transform.integrate_on_los(Ne, t=t, passes=self.passes)
        return los_integral, along_los

    def line_integrated_phase_shift(self, Ne: LabeledArray, t: LabeledArray = None):
        raise NotImplementedError("Calculation of phase shift still not implemented")
