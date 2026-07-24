from abc import ABC
from abc import abstractmethod

from xarray import DataArray

from indica.configs.readers.adasconf import ADF11
from indica.readers import ADASReader
from indica.utilities import get_element_info


class FractionalAbundance(ABC):
    """Abstract class to calculate ionisation balance using ADAS rate coefficients"""

    def __init__(
        self,
        element: str,
        scd_year: str = None,
        acd_year: str = None,
        ccd_year: str = None,
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
        self.adas_reader = ADASReader()

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

    def plot(self, **kwargs):
        self.plot_fractional_abundance(**kwargs)

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

    @abstractmethod
    def plot_fractional_abundance(self, **kwargs):
        raise NotImplementedError(
            "Implement this method to plot FractionalAbundance code output"
        )
