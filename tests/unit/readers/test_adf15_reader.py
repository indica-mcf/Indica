"""Unit tests for the functions in routines.py
"""
import numpy as np

from indica.readers import ADASReader


class Testadf15:
    """Provides unit tests for the ADF15 reader"""

    reader = ADASReader()
    element = "ne"
    charge = "9"
    file_type = "pju"
    year = "96"

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


class Testadf15_c5(Testadf15):
    """Provides unit tests for the ADF15 reader"""

    reader = ADASReader()
    element = "c"
    charge = "5"
    file_type = "bnd"
    year = "96"


class Testadf15_he0(Testadf15):
    """Provides unit tests for the ADF15 reader"""

    reader = ADASReader()
    element = "he"
    charge = "0"
    file_type = "pju"
    year = "93"


class Testadf15_b2(Testadf15):
    """Provides unit tests for the ADF15 reader"""

    reader = ADASReader()
    element = "b"
    charge = "2"
    file_type = "llu"
    year = "93"


class Testadf15_o5(Testadf15):
    """Provides unit tests for the ADF15 reader"""

    reader = ADASReader()
    element = "o"
    charge = "5"
    file_type = "pjr"
    year = "96"


class Testadf15_ar10(Testadf15):
    """Provides unit tests for the ADF15 reader"""

    reader = ADASReader()
    element = "ar"
    charge = "10"
    file_type = "ic"
    year = "40"


class Testadf15_transport(Testadf15):
    """Provides unit tests for the ADF15 reader"""

    reader = ADASReader()
    element = "ar"
    charge = "16"
    file_type = "llu"
    year = "transport"
