from abc import ABC
from abc import abstractmethod

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.configs.readers.adasconf import ADF11
from indica.readers import ADASReader
from indica.readers.adas import DEFAULT_PATH
from indica.utilities import format_dataarray
from indica.utilities import get_element_info
from indica.utilities import set_plot_colors

CM, COLS = set_plot_colors()


class FractionalAbundance(ABC):
    """Abstract class to calculate ionisation balance using ADAS rate coefficients"""

    def __init__(
        self,
        element: str,
        scd_year: str = None,
        acd_year: str = None,
        ccd_year: str = None,
        adas_path: str = DEFAULT_PATH,
        **kwargs,
    ):
        """
        Get element information, ADAS file names and adf11 data

        element
            element symbol

        ..._year
            ADAS adf11 "year" (see https://open.adas.ac.uk/adf11) for ionisation (scd),
            recombination (acd) and thermal charge exchange (ccd)
        """
        self.adas_reader = ADASReader(path=adas_path)

        _element_info = get_element_info(element)
        self.element = element
        self.element_info = {
            "Z": _element_info[0],
            "A": _element_info[1],
            "name": _element_info[2],
            "symbol": _element_info[3],
        }

        self.scd_year = scd_year
        self.acd_year = acd_year
        self.ccd_year = ccd_year
        for key in ["acd", "scd", "ccd"]:
            # Year
            if getattr(self, f"{key}_year") is None:
                _year = ADF11[element][key]
                setattr(self, f"{key}_year", _year)
            year = getattr(self, f"{key}_year")

            # Filename
            filename = f"{key}{year}_{element}.dat"
            setattr(self, f"{key}_file", filename)

            # Data
            data = self.adas_reader.get_adf11(key, self.element, year)
            setattr(self, key, data)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_parameters(self, **kwargs):
        """
        Set any model kwargs
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __call__(
        self,
        Te: DataArray,
        Ne: DataArray,
        Nn: DataArray = None,
        **kwargs,
    ) -> dict:
        """
        Run fractional abundance code

        kwargs
            for the prepare phase which may change depending on the code used
        """
        self.Te = Te
        self.Ne = Ne
        if Nn is None:
            _Nn = xr.full_like(Ne, 0.0)
            Nn = format_dataarray(_Nn, "thermal_neutral_density", dict(Ne.coords))
        self.Nn = Nn

        """Prepare code input data structure"""
        self.prepare(**kwargs)

        """Run code"""
        self.run()

        """Reorganise code output to return Indica-native results"""
        result = self.refactor_output()

        return result

    @abstractmethod
    def prepare(self, **kwargs):
        raise NotImplementedError(
            "Implement to reorganise input data to feed to FractionalAbundance code"
        )

    @abstractmethod
    def run(self):
        raise NotImplementedError(
            "Implement this method to run FractionalAbundance code and return results"
        )

    @abstractmethod
    def refactor_output(self):
        """
        Fractional abundance data structure must be identical for all codes
        and match the attributes in the Plasma class for seamless mapping

        result = DataArray("t", "rhop", "ion_charge)
        """
        raise NotImplementedError(
            "Implement this method to reorganise FractionalAbundance code output"
        )

    def plot(self, xlim: tuple = (0, 1.2), title: str = None):
        cols = CM(np.linspace(0.1, 0.75, len(self.ion_charge), dtype=float))

        for iq in np.int_(self.ion_charge):
            self.F_z_t.sel(ion_charge=iq).plot(color=cols[iq], alpha=0.8, label=iq)
        plt.legend()
        plt.xlim(xlim)
        if title is None:
            title = f"{self.element.title()} fractional abundance"
        plt.title(title)
