from copy import deepcopy

import numpy as np
import scipy.constants as constants
from xarray import DataArray

from indica import physics
from indica.available_quantities import READER_QUANTITIES
from indica.converters import LineOfSightTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
from indica.utilities import build_dataarrays


class BremsstrahlungSpectrometer(AbstractDiagnostic):
    """
    Model measured Bremsstrahlung emission along a line-of-sight

    Calculated in ph / cm^2 / s / sr
    """

    Bremsstrahlung: DataArray
    los_integral_bremsstrahlung: DataArray

    def __init__(
        self,
        name: str,
        central_wavelength: float,
        instrument_method="get_spectrometer",
    ):
        self.transform: LineOfSightTransform
        self.name = name
        self.central_wavelength = central_wavelength
        self.instrument_method = instrument_method
        self.quantities = deepcopy(READER_QUANTITIES[self.instrument_method])

    def _build_bckc_dictionary(self):
        bckc = {
            "t": self.t,
            "channel": np.arange(len(self.transform.x1)),
            "wavelength": np.array(self.central_wavelength, ndmin=1),
            "location": self.transform.origin,
            "direction": self.transform.direction,
            "spectra": self.los_integral_bremsstrahlung,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
        self,
        Ne: DataArray = None,
        Te: DataArray = None,
        Zeff: DataArray = None,
        t: LabeledArray = None,
        calc_rho=False,
        gaunt_approx="callahan",
        **kwargs,
    ):
        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Ne = self.plasma.electron_density.interp(t=t)
            Te = self.plasma.electron_temperature.interp(t=t)
            Zeff = 1 + (
                (
                    self.plasma.ion_density
                    * self.plasma.meanz
                    * (self.plasma.meanz - 1)
                    / self.plasma.electron_density
                )
                .interp(t=t)
                .sum("element")
            )
        elif Ne is None or Te is None or Zeff is None:
            raise ValueError("Give inputs or assign plasma class!")

        self.t: DataArray = t
        self.Ne: DataArray = Ne
        self.Te: DataArray = Te
        self.Zeff: DataArray = Zeff

        self.Bremsstrahlung = physics.zeff_bremsstrahlung(
            Te=Te,
            Ne=Ne,
            wavelength=self.central_wavelength,
            zeff=Zeff,
            gaunt_approx=gaunt_approx,
        ).expand_dims(
            {"wavelength": np.array(self.central_wavelength, ndmin=1)}, axis=-1
        )  # W / m^3

        self.los_integral_bremsstrahlung = self.transform.integrate_on_los(
            (
                self.Bremsstrahlung
                / ((constants.h * constants.c) / (self.central_wavelength * 1e-9))
            ),
            t=self.t,
            calc_rho=calc_rho,
        ) * ((1e-4) / (4 * np.pi))

        self._build_bckc_dictionary()
        return self.bckc
