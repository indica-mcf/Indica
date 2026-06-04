from abc import ABC
from abc import abstractmethod

from indica import Plasma
from indica.converters import CoordinateTransform
from indica.operators.noise import get_noise_model


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
    def _build_bckc_dictionary(
        self,
        noise: str | None = None,
        noise_config: dict | None = None,
    ):
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

    def apply_noise(self, noise: str, noise_config: dict | None = None):
        """
        Apply noise to the back-calculated values.
        The noise is applied to the quantity specified by
        'target_quantity' in noise_config.
        Preserves the original data in a new key with suffix
          '_raw', e.g. 'brightness_raw' for 'brightness'.
        """
        if noise_config is None:
            noise_config = {}
        if not isinstance(noise_config, dict):
            raise TypeError("noise_config must be a dictionary.")

        noise_model = get_noise_model(noise)
        config = dict(noise_config)
        target_quantity = config.pop("target_quantity", None)

        if target_quantity is None:
            raise ValueError(
                "noise_config must include 'target_quantity', e.g. "
                "noise_config={'target_quantity': 'brightness', ...}."
            )
        if target_quantity not in self.bckc:
            raise KeyError(
                f"'{target_quantity}' not found in model output keys: "
                f"{list(self.bckc.keys())}"
            )

        clean = self.bckc[target_quantity]
        self.bckc[f"{target_quantity}_raw"] = clean
        self.bckc[target_quantity] = noise_model(clean, **config)
