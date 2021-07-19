import unittest

import numpy as np
from xarray import DataArray

from indica.datatypes import ELEMENTS_BY_SYMBOL
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.mean_charge import MeanCharge
from indica.readers.adas import ADASReader


class Exception_Mean_Charge_Test_Case(unittest.TestCase):
    """Test case for testing assertion, type and value errors in
    MeanCharge call.
    """

    def __init__(self):
        pass

    def frac_abund_obj_type_check(self, F_z_t, element):
        """Test type error for MeanCharge call"""
        with self.assertRaises(TypeError):
            example_check = MeanCharge()
            example_check(F_z_t, element)

    def element_type_check(self, F_z_t, element):
        """Test type error for MeanCharge call"""
        with self.assertRaises(TypeError):
            example_check = MeanCharge()
            example_check(F_z_t, element)

    def element_value_check(self, F_z_t, element):
        """Test value error for MeanCharge call"""
        with self.assertRaises(ValueError):
            example_check = MeanCharge()
            example_check(F_z_t, element)

    def ionisation_assertion_check(self, F_z_t, element):
        """Test assertion error for MeanCharge call"""
        with self.assertRaises(AssertionError):
            example_check = MeanCharge()
            example_check(F_z_t, element)


def test_mean_charge():
    """Test MeanCharge.__call__."""
    ADAS_file = ADASReader()

    element = "be"

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    input_Ne = np.logspace(19.0, 16.0, 10)
    input_Te = np.logspace(4.6, 2.0, 10)

    input_Te = DataArray(
        data=input_Te, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )
    input_Ne = DataArray(
        data=input_Ne, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )

    example_frac_abundance = FractionalAbundance(SCD, ACD, input_Ne, input_Te)

    F_z_t0 = np.real(example_frac_abundance.F_z_t0)
    F_z_t0 = F_z_t0.expand_dims("t", axis=-1)

    element_name = ELEMENTS_BY_SYMBOL.get(element)

    input_check = Exception_Mean_Charge_Test_Case()

    input_check.frac_abund_obj_type_check(F_z_t0.data, element_name)

    input_check.element_type_check(F_z_t0, 4)

    input_check.element_value_check(F_z_t0, "xy")

    input_check.ionisation_assertion_check(F_z_t0[0:3], element_name)

    example_mean_charge = MeanCharge()

    result = example_mean_charge(F_z_t0, element_name)

    assert np.all(result == 0)

    F_z_tinf = example_frac_abundance.F_z_tinf

    F_z_tinf = F_z_tinf.expand_dims("t", axis=-1)

    example_mean_charge = MeanCharge()

    result = example_mean_charge(F_z_tinf, element_name)
    expected = np.zeros((*F_z_tinf.shape[1:],))
    expected += 4.0

    assert np.allclose(result, expected, rtol=1e-2)
