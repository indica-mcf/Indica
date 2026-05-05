from abc import ABC
from abc import abstractmethod

from indica import Plasma
from indica.converters import CoordinateTransform

import numpy as np

class AbstractDiagnostic(ABC):
    name: str = ""
    bckc: dict = {}
    plasma: Plasma
    transform: CoordinateTransform

    def set_transform(self, transform: CoordinateTransform):
        """
        Line-of-sight or Transect coordinate transform
        """
        # TODO: types attribute set during initialisation!
        self.transform = transform

    def set_plasma(self, plasma: Plasma):
        """
        Assign Plasma class to use for computation of forward model
        """
        self.plasma = plasma

    def set_parameters(self, **kwargs):
        """
        Set any model kwargs
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def _build_bckc_dictionary(self):
        """
        Calculate back-calculated expected values that the diagnostic will
        be measuring. This can be directly compared to the data read in by
        the Indica Reader methods.
        """
        raise NotImplementedError(
            "{} does not implement a "
            "'__call__' method.".format(self.__class__.__name__)
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> dict:
        """
        Call the model and return back-calculated values
        """
        raise NotImplementedError(
            "{} does not implement a "
            "'__call__' method.".format(self.__class__.__name__)
        )


    def add_poisson_noise(self, data, typical_counts=50, reference=None, rng=None):
        """
        Add signal-dependent Poisson noise to continuous positive data.

        Parameters
        ----------
        data : np.ndarray or xarray.DataArray
            Positive clean signal, e.g. shape (n_times, n_channels).
        typical_counts : float
            Effective counts at the reference signal level.
            Smaller = noisier. Try 10, 100, 1000, 10000.
        reference : float or None
            Signal value corresponding to `typical_counts`.
            If None, use mean(data).
        rng : np.random.Generator or None
            Optional random generator.

        Returns
        -------
        noisy : np.ndarray or xarray.DataArray
            Noisified data in original units. If `data` is an xarray.DataArray,
            the output preserves its dimensions and coordinates.
        """
        if rng is None:
            rng = np.random.default_rng()

        input_data = data
        data = np.asarray(input_data)

        if np.any(data < 0):
            raise ValueError("Poisson noise requires non-negative data.")

        if reference is None:
            reference = np.mean(data)

        if reference <= 0:
            raise ValueError("reference must be positive.")

        # Convert physical signal to effective Poisson counts
        lam = typical_counts * data / reference

        # Draw noisy counts
        noisy_counts = rng.poisson(lam)

        # Convert back to original units
        noisy = noisy_counts * reference / typical_counts

        # Preserve xarray metadata when available.
        if hasattr(input_data, "dims") and hasattr(input_data, "coords") and hasattr(input_data, "copy"):
            return input_data.copy(data=noisy)

        return noisy
