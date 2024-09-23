import numpy as np
import xarray as xr

from indica.profilers.profiler_base import ProfilerBase


class ProfilerBasis(ProfilerBase):
    """
    Class to build pca profiles from basis functions

    Parameters
    ----------
    radial_grid
        normalised radial grid [0, 1]  on which profile is to be built
    """

    def __init__(
        self,
        basis_functions: np.ndarray,
        bias: np.ndarray,
        ncomps: int = 2,
        radial_grid: np.ndarray = None,
        coord="rho_poloidal",
        parameters: dict = None,
    ):
        super().__init__(parameters)
        self.basis_functions = basis_functions
        self.bias = bias
        self.ncomps = ncomps
        self.radial_grid = radial_grid
        self.coord = coord

        # Weights have to be dynamically assigned as attributes, so they can be accessed
        # / changed by the profiler
        for icomp in range(self.ncomps):
            param_name = f"weight_{icomp + 1}"
            self.parameters[param_name] = 0
            setattr(self, param_name, 0)

    def construct_profile(
        self,
    ):
        weights = np.stack(
            [weight for weight_name, weight in self.parameters.items()], axis=-1
        ).T
        return np.dot(weights, self.basis_functions) + self.bias

    def __call__(
        self,
    ):
        """
        Builds the profile from basis functions using the parameters set
        """
        y = self.construct_profile()
        self.ydata = xr.DataArray(y, coords=[(self.coord, self.radial_grid.data)])
        return self.ydata
