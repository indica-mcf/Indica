from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters import LineOfSightTransform
from indica.converters.abstractconverter import CoordinateTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
from indica.plasma import Plasma
from indica.utilities import build_dataarrays


class EmissionType:
    """Base class for defining ways to represent emission"""

    def __init__(self, coords: List):
        self.coords = coords

    name: str = ""

    def set_coord_names(self, *args):
        for coord in args:
            if coord not in self.coords:
                self.coords = list(set(self.coords.append(coord)))


class LocalEmissionModel(ABC):
    """Base class for calculating emission from specific process at a particular
    point in the plasma
    """

    name: str = ""
    plasma: Plasma
    emission_type: EmissionType

    def set_plasma(self, plasma: Plasma):
        """Assign plasma class"""
        self.plasma = plasma

    def set_parameters(self, **kwargs):
        """
        Set any model kwargs
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> DataArray:
        """Calculate the local emission in units of W.m-3.(...)"""
        raise NotImplementedError(
            "{} does not implement a" "'__cal__' method.".format(self.__class_.__name__)
        )


class LoSEmission:
    """Calculate line-integrated emission"""

    def __init__(
        self,
        name: str,
    ):
        self.name = name
        plasma: Plasma
        transform: LineOfSightTransform
        local_emission_model: LocalEmissionModel

    def set_plasma(self, plasma: Plasma):
        """
        Assign Plasma class to use for computation of forward model
        """

        self.plasma = plasma
        if hasattr(self, "local_emission_model"):
            self.local_emission_model.set_plasma(self.plasma)

    def set_transform(
        self,
        transform: CoordinateTransform,
        integration_method: str,
    ):
        """
        Line-of-sight coordinate transform
        """
        self.transform = transform
        self.calculate_spatial_integral = getattr(self.transform, integration_method)

    def set_local_emission_model(
        self,
        local_emission_model: LocalEmissionModel,
    ):

        # Add plasma to emission model
        if hasattr(self, "plasma"):
            local_emission_model.set_plasma(self.plasma)

        # Add emission model
        self.local_emission_model = local_emission_model

    def __call__(
        self,
        t: LabeledArray = None,
        local_emission_args: List = [],
        local_emission_kwargs: Dict = {},
        integrator_args: List = [],
        integrator_kwargs: Dict = {},
    ) -> xr.DataArray:
        """Calculate sightline-integrated emission - W.m-2 (with or without nm)"""

        local_emission = self.local_emission_model(
            *local_emission_args,
            **local_emission_kwargs,
        )
        self.los_emission = self.calculate_spatial_integral(
            local_emission,
            *integrator_args,
            **integrator_kwargs,
        )
        return self.los_emission


class BeamSpectrometer(AbstractDiagnostic):
    """
    Base class factoring out universal(!?) properties of spectrmeters to be inherited
    by passive_spectrometer and beam_spectrometer (and other future spectrometers...)
    """

    def __init__(
        self,
        name: str,
        spectrometer_data: Dict[str, DataArray],
    ):

        self.name = name
        self.instrument_method = "spectrometer"
        self.los_emission_models: Dict[str, LoSEmission] = {}

        self.emission_type: EmissionType

        self.transform: LineOfSightTransform
        self.channel: DataArray
        self.location: DataArray
        self.direction: DataArray
        self.index: DataArray
        self.wavelength: DataArray
        self.kernel_index: DataArray
        self.instrument_function: DataArray
        self.sensitivity: DataArray
        self.spectra_raw: DataArray
        self.spectra: DataArray

        self._data = spectrometer_data

        # Assign all spectrometer data as class attributes
        for k, v in self._data.items():
            setattr(self, k, v)

    def set_plasma(self, plasma: Plasma):
        """Assign plasma class to all emission models"""
        self.plasma = plasma
        for name, los_emission_model in self.los_emission_models.items():
            los_emission_model.set_plasma(self.plasma)

    def set_los_emission_model(
        self,
        local_emission_model: LocalEmissionModel,
        integration_method: str = "integrate_on_los",
    ):
        """Assign an emission model"""

        if not hasattr(self, "emission_type"):
            self.emission_type = local_emission_model.emission_type

        # Check if emission type matches spectrometer
        if local_emission_model.emission_type != self.emission_type:
            message = (
                "Cannot assign line of sight emission of type "
                + f"{local_emission_model.emission_type.name} to spectrometer "
                + f"expecting emission type {self.emission_type.name}."
            )
            raise TypeError(message)

        # Set up los emission model
        los_emission_model = LoSEmission("LoS_" + local_emission_model.name)
        los_emission_model.set_local_emission_model(local_emission_model)
        los_emission_model.set_plasma(self.plasma)
        los_emission_model.set_transform(self.transform, integration_method)

        # Add emission model to spectrometer
        self.los_emission_models.update({los_emission_model.name: los_emission_model})

    def make_los_emission(
        self,
        t: LabeledArray = None,
        local_emission_args: Dict[str, List] = {},
        local_emission_kwargs: Dict[str, Dict[str, Any]] = {},
        integrator_args: Dict[str, List] = {},
        integrator_kwargs: Dict[str, Dict[str, Any]] = {},
    ) -> Dict[str, DataArray]:

        # Initialise output
        los_emission = {}

        # Consider each emission model in turn
        for name, los_emission_model in self.los_emission_models.items():

            # Run the models
            los_emission[name] = los_emission_model(
                t=t,
                local_emission_args=local_emission_args.get(name, []),
                local_emission_kwargs=local_emission_kwargs.get(name, {}),
                integrator_args=integrator_args.get(name, []),
                integrator_kwargs=integrator_kwargs.get(name, {}),
            )

        self.los_emission = los_emission
        return self.los_emission

    def los_emission_to_counts(
        self,
    ):
        """Transform ideal emission, W.m-2.sr-1.nm-1, to spectrometer
        counts.
        """

        self.pixel_counts = {}
        for name, _spectra in self.ideal_spectra.items():
            _pixel_emission = self._pixel_integration(_spectra)
            _infnc_emission = self._convolve_instrument_function(_pixel_emission)
            self.pixel_counts[name] = self._scale_to_counts(_infnc_emission)

        return self.pixel_counts

    def contract_spectra(self, spectrum: xr.DataArray) -> xr.DataArray:
        """Sum a spectrum data array accross all dimensions except channel
        and wavelength
        """

        no_sum_dims = ["channel", "wavelength", "time"]
        sum_list = [d for d in spectrum.dims if d not in no_sum_dims]
        return spectrum.sum(dims=sum_list)

    def __call__(
        self,
        t: LabeledArray = None,
        local_emission_args: Dict[str, List] = {},
        local_emission_kwargs: Dict[str, Dict[str, Any]] = {},
        integrator_args: Dict[str, List] = {},
        integrator_kwargs: Dict[str, Dict[str, Any]] = {},
    ):
        """
        Returns spectrometer geometry

        """

        # Make fully decomposed spectra
        self.make_los_emission(
            t=t,
            local_emission_args=local_emission_args,
            local_emission_kwargs=local_emission_kwargs,
            integrator_args=integrator_args,
            integrator_kwargs=integrator_kwargs,
        )

        # Sum spectra
        self.total_emission = self.emission_type.make_empty_spectra(self)
        for name, emission in self.ideal_spectra.items():
            self.total_emission.data += self.contract_spectra(emission)

        # Convert into spectrometer counts
        if self.emission_type.wavelength_resolved:
            self.ideal_spectra_to_counts()
            self.total_counts = self.emission_type.make_empty_counts(self)
            for name, counts in self.pixel_counts.items():
                self.total_counts.data += self.contract_spectra(counts)

        return self._build_bckc_dictionary()

    def _convolve_instrument_function(self):
        """Not yet implemented

        Convolve a wavelength-resolved quantity with an instrument function

        """
        pass

    def _pixel_integration(self):
        """Not yet implemented

        Integrate wavelength-resolved spectrum into pixel bins
        The wavelength-resolved spectrum should be on a global wavelength axis
        Each channel may have a unique set of pixel bins

        """
        pass

    def _scale_to_counts(self):
        """Not yet implemented

        Scale emissivity (W.m-2.sr-1) to counts according to sensitivity and
        exposure duration

        """
        pass

    def _build_bckc_dictionary(self):
        bckc = {
            "channel": np.arange(len(self.transform.x1)),
            "wavelength": self.window.data,
            "location": self.transform.origin,
            "direction": self.transform.direction,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)
