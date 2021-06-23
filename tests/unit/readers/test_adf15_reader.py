"""Unit tests for the functions in routines.py
"""
import unittest

import numpy as np
from prov.model import ProvEntity

from indica.readers import ADASReader


class Testadf15(unittest.TestCase):
    """Provides unit tests for the reverse() function"""

    def setUp(self):
        """Setup list for testing"""
        self.reader = ADASReader()
        self.element = "ne"
        self.charge = "9"
        self.file_type = "pju"
        self.year = "96"

    def test_read(self):
        """Checks reading does not give exceptions"""
        try:
            _ = self.reader.get_adf15(
                self.element, self.charge, self.file_type, year=self.year
            )
        except Exception as e:
            raise e

    def test_false_input(self):
        """Checks reading with wrong inputs raises errors"""
        element = self.element + ";"
        try:
            _ = self.reader.get_adf15(
                element, self.charge, self.file_type, year=self.year
            )
        except Exception as e:
            assert e

    def test_index(self):
        """Checks index is float and values withing specified bounds"""
        data = self.reader.get_adf15(
            self.element, self.charge, self.file_type, year=self.year
        )

        assert all([type(index) is np.float64 for index in data.index.values])
        assert all(data.index >= 0)
        assert all(data.index <= np.max(data.shape))

    def test_electron_temperature(self):
        """Checks electron temperature is float and values withing specified bounds"""
        data = self.reader.get_adf15(
            self.element, self.charge, self.file_type, year=self.year
        )

        assert all(
            [
                type(electron_temperature) is np.float64
                for electron_temperature in data.electron_temperature.values
            ]
        )
        assert all(data.electron_temperature >= 0)
        assert all(data.electron_temperature < 1.0e6)

    def test_electron_density(self):
        """Checks electron density is float and values withing specified bounds"""
        data = self.reader.get_adf15(
            self.element, self.charge, self.file_type, year=self.year
        )

        assert all(
            [
                type(electron_density) is np.float64
                for electron_density in data.electron_density.values
            ]
        )
        assert all(data.electron_density >= 0)
        assert all(data.electron_density < 1.0e30)

    def test_wavelength(self):
        """Checks wavelengths is a float and values withing specified bounds"""
        data = self.reader.get_adf15(
            self.element, self.charge, self.file_type, year=self.year
        )

        assert all(
            [type(wavelength) is np.float64 for wavelength in data.wavelength.values]
        )
        assert all(data.wavelength >= 0)
        assert all(data.wavelength <= 1.0e6)

    def test_transition(self):
        """Checks transition is a string and of lenght > 0"""
        data = self.reader.get_adf15(
            self.element, self.charge, self.file_type, year=self.year
        )

        assert all(
            [type(transition) is np.str_ for transition in data.transition.values]
        )
        assert all([len(transition) > 0 for transition in data.transition.values])

    def test_transition_type(self):
        """Checks transition type is a string and of lenght > 0"""
        data = self.reader.get_adf15(
            self.element, self.charge, self.file_type, year=self.year
        )

        assert all(
            [type(transition_type) is np.str_ for transition_type in data.type.values]
        )
        assert all([len(transition_type) > 0 for transition_type in data.type.values])

    def test_datatype(self):
        """Checks datatype is string and as expected"""
        data = self.reader.get_adf15(
            self.element, self.charge, self.file_type, year=self.year
        )

        assert type(data.attrs["datatype"]) is tuple
        assert len(data.attrs["datatype"]) == 2
        assert all([type(datatype) is str for datatype in data.attrs["datatype"]])
        assert data.attrs["datatype"] == (
            f"photon_emissivity_coefficients_{self.file_type}",
            self.element,
        )

    def test_provenance(self):
        """Checks datatype is string and as expected"""
        data = self.reader.get_adf15(
            self.element, self.charge, self.file_type, year=self.year
        )

        assert type(data.attrs["provenance"]) is ProvEntity


class Testadf15_c5(Testadf15):
    def setUp(self):
        """Setup list for testing"""
        self.reader = ADASReader()
        self.element = "c"
        self.charge = "5"
        self.file_type = "bnd"
        self.year = "96"


class Testadf15_he0(Testadf15):
    def setUp(self):
        """Setup list for testing"""
        self.reader = ADASReader()
        self.element = "he"
        self.charge = "0"
        self.file_type = "pju"
        self.year = "93"


class Testadf15_b2(Testadf15):
    def setUp(self):
        """Setup list for testing"""
        self.reader = ADASReader()
        self.element = "b"
        self.charge = "2"
        self.file_type = "llu"
        self.year = "93"


class Testadf15_o5(Testadf15):
    def setUp(self):
        """Setup list for testing"""
        self.reader = ADASReader()
        self.element = "o"
        self.charge = "5"
        self.file_type = "pjr"
        self.year = "96"


class Testadf15_ar10(Testadf15):
    def setUp(self):
        """Setup list for testing"""
        self.reader = ADASReader()
        self.element = "ar"
        self.charge = "10"
        self.file_type = "ic"
        self.year = "40"


class Testadf15_transport(Testadf15):
    def setUp(self):
        """Setup list for testing"""
        self.reader = ADASReader()
        self.element = "ar"
        self.charge = "16"
        self.file_type = "llu"
        self.year = "transport"


if __name__ == "__main__":
    unittest.main()

reader = ADASReader()
element = "ar"
charge = "10"
file_type = "ic"
year = "40"
data = reader.get_adf15(element, charge, file_type, year=year)
