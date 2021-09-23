import unittest

import numpy as np
from xarray import DataArray

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.mean_charge import MeanCharge
from indica.operators.spline_fit import Spline
from indica.readers.adas import ADASReader
from indica.utilities import broadcast_spline


class Exception_Mean_Charge_Test_Case(unittest.TestCase):
    """Test case for testing assertion, type and value errors in
    MeanCharge call.
    """

    def __init__(self):
        pass

    def call_type_check(self, F_z_t, element):
        """Test TypeError for MeanCharge call."""
        with self.assertRaises(TypeError):
            example_mean_charge = MeanCharge()
            example_mean_charge(F_z_t, element)

    def call_value_check(self, F_z_t, element):
        """Test ValueError for MeanCharge call."""
        with self.assertRaises(ValueError):
            example_mean_charge = MeanCharge()
            example_mean_charge(F_z_t, element)

    def call_assertion_check(self, F_z_t, element):
        """Test AssertionError for MeanCharge call."""
        with self.assertRaises(AssertionError):
            example_mean_charge = MeanCharge()
            example_mean_charge(F_z_t, element)


def test_mean_charge():
    """Test MeanCharge.__call__."""
    ADAS_file = ADASReader()

    element = "be"

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    t = np.linspace(75.0, 80.0, 5)
    rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])

    input_Ne = DataArray(
        data=np.tile(np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19]), (len(t), 1)).T,
        coords=[("rho", rho_profile), ("t", t)],
        dims=["rho", "t"],
    )

    input_Te = DataArray(
        data=np.tile(np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]), (len(t), 1)).T,
        coords=[("rho", rho_profile), ("t", t)],
        dims=["rho", "t"],
    )

    rho = DataArray(
        data=np.linspace(0.0, 1.0, 20),
        coords=[("rho", np.linspace(0.0, 1.05, 20))],
        dims=["rho"],
    )

    dummy_coordinates = FluxSurfaceCoordinates("poloidal")

    input_Ne_spline = Spline(input_Ne, "rho", dummy_coordinates)
    input_Ne = broadcast_spline(
        input_Ne_spline.spline,
        input_Ne_spline.spline_dims,
        input_Ne_spline.spline_coords,
        rho,
    )

    input_Te_spline = Spline(input_Te, "rho", dummy_coordinates)
    input_Te = broadcast_spline(
        input_Te_spline.spline,
        input_Te_spline.spline_dims,
        input_Te_spline.spline_coords,
        rho,
    )

    example_frac_abundance = FractionalAbundance(SCD, ACD)

    example_frac_abundance.interpolate_rates(
        Ne=input_Ne.isel(t=0), Te=input_Te.isel(t=0)
    )

    example_frac_abundance.calc_ionisation_balance_matrix(Ne=input_Ne.isel(t=0))

    example_frac_abundance.calc_F_z_tinf()

    example_frac_abundance.calc_eigen_vals_and_vecs()

    example_frac_abundance.calc_eigen_coeffs()

    F_z_t0 = np.real(example_frac_abundance.F_z_t0)
    F_z_t0 = F_z_t0.expand_dims("t", axis=-1)

    input_check = Exception_Mean_Charge_Test_Case()

    input_check.call_type_check(F_z_t0.data, element)
    input_check.call_value_check(F_z_t0 * -1, element)
    input_check.call_value_check(F_z_t0 * -np.inf, element)
    input_check.call_value_check(F_z_t0 * np.inf, element)
    input_check.call_value_check(F_z_t0 * np.nan, element)

    input_check.call_type_check(F_z_t0, 4)

    input_check.call_value_check(F_z_t0, "xy")

    input_check.call_assertion_check(F_z_t0[0:3], element)

    example_mean_charge = MeanCharge()

    result = example_mean_charge(F_z_t0, element)

    assert np.all(result == 0)

    F_z_tinf = example_frac_abundance.F_z_tinf

    F_z_tinf = F_z_tinf.expand_dims("t", axis=-1)

    example_mean_charge = MeanCharge()

    result = example_mean_charge(F_z_tinf, element)
    expected = np.zeros((*F_z_tinf.shape[1:],))
    expected += 4.0

    assert np.allclose(result, expected, rtol=1e-2)
