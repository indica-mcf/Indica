import copy
import unittest

import numpy as np
import pytest
from xarray import DataArray

from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader


class Exception_Frac_Abund_Test_Case(unittest.TestCase):
    """Test case for testing assertion and value errors in
    FractionalAbundance initialisations and FractionalAbundance
    functions.
    """

    def __init__(
        self,
        FracAbundObj: FractionalAbundance,
        Ne: DataArray = None,
        Te: DataArray = None,
        Nh: DataArray = None,
        tau: LabeledArray = None,
        F_z_t0: DataArray = None,
    ):
        self.SCD = FracAbundObj.SCD
        self.ACD = FracAbundObj.ACD
        self.CCD = FracAbundObj.CCD

        self.Ne = FracAbundObj.Ne if Ne is None else Ne
        self.Te = FracAbundObj.Te if Te is None else Te
        self.Nh = FracAbundObj.Nh if Nh is None else Nh
        self.tau = FracAbundObj.tau if tau is None else tau
        self.F_z_t0 = FracAbundObj.F_z_t0 if F_z_t0 is None else F_z_t0

        self.FracAbundObj = FracAbundObj

        self.nominal_inputs = [
            self.SCD,
            self.ACD,
            self.CCD,
            self.Ne,
            self.Te,
            self.Nh,
            self.tau,
            self.F_z_t0,
        ]

    def init_type_check(
        self,
        SCD=None,
        ACD=None,
        CCD=None,
    ):
        inputs = [SCD, ACD, CCD]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        SCD, ACD, CCD = inputs

        """Test type errors are raised for FractionalAbundance initialisation."""
        with self.assertRaises(TypeError):
            FractionalAbundance(SCD, ACD, CCD)

    def init_value_error_check(
        self,
        SCD=None,
        ACD=None,
        CCD=None,
    ):
        inputs = [SCD, ACD, CCD]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        SCD, ACD, CCD = inputs

        """Test value errors are raised for FractionalAbundance initialisation."""
        with self.assertRaises(ValueError):
            FractionalAbundance(SCD, ACD, CCD)

    def interpolation_type_check(self, Ne=None, Te=None):
        """Test type errors are raised for interpolation_rates function"""
        inputs = [Ne, Te]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 3]

        Ne, Te = inputs

        with self.assertRaises(TypeError):
            self.FracAbundObj.interpolate_rates(Ne, Te)

    def interpolation_value_check(self, Ne=None, Te=None):
        """Test value errors are raised for interpolation_rates function"""
        inputs = [Ne, Te]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 3]

        Ne, Te = inputs

        with self.assertRaises(ValueError):
            self.FracAbundObj.interpolate_rates(Ne, Te)

    def calc_ionisation_balance_type_check(self, Ne=None, Nh=None):
        """Test type errors are raised for calc_ionisation_balance_matrix function."""
        nominal_inputs = [self.Ne, self.Nh]

        inputs = [Ne, Nh]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = nominal_inputs[i]

        Ne, Nh = inputs

        with self.assertRaises(TypeError):
            self.FracAbundObj.calc_ionisation_balance_matrix(Ne, Nh)

    def calc_ionisation_balance_value_check(self, Ne=None, Nh=None):
        """Test value errors are raised for calc_ionisation_balance_matrix function."""
        nominal_inputs = [self.Ne, self.Nh]

        inputs = [Ne, Nh]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = nominal_inputs[i]

        Ne, Nh = inputs

        with self.assertRaises(ValueError):
            self.FracAbundObj.calc_ionisation_balance_matrix(Ne, Nh)

    def calc_ionisation_balance_partial_input_value_check(self, Ne=None, Nh=None):
        nominal_inputs = [self.Ne, self.Nh]

        inputs = [Ne, Nh]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = nominal_inputs[i]

        Ne, Nh = inputs

        with self.assertRaises(ValueError):
            self.FracAbundObj.calc_ionisation_balance_matrix(Ne, Nh)

    def calc_eigen_coeffs_type_check(self, F_z_t0=None):
        """Test type errors are raised for calc_eigen_coeffs function."""
        if F_z_t0 is None:
            F_z_t0 = self.F_z_t0

        with self.assertRaises(TypeError):
            self.FracAbundObj.calc_eigen_coeffs(F_z_t0)

    def calc_eigen_coeffs_value_check(self, F_z_t0=None):
        """Test value errors are raised for calc_eigen_coeffs function."""
        if F_z_t0 is None:
            F_z_t0 = self.F_z_t0

        with self.assertRaises(ValueError):
            self.FracAbundObj.calc_eigen_coeffs(F_z_t0)

    def call_type_check(self, Ne=None, Te=None, Nh=None, tau=None):
        """Test type errors are raised for FractionalAbundance call."""
        inputs = [Ne, Te, Nh, tau]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 3]

        Ne, Te, Nh, tau = inputs

        with self.assertRaises(TypeError):
            self.FracAbundObj(Ne, Te, Nh, tau=tau, full_run=False)

    def call_value_check(self, Ne=None, Te=None, Nh=None, tau=None):
        """Test value errors are raised for FractionalAbundance call."""
        inputs = [Ne, Te, Nh, tau]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 3]

        Ne, Te, Nh, tau = inputs

        with self.assertRaises(ValueError):
            self.FracAbundObj(Ne, Te, Nh, tau=tau, full_run=False)


class Exception_Power_Loss_Test_Case(unittest.TestCase):
    """Test case for testing assertion and value errors in
    PowerLoss initialisations.
    """

    def __init__(
        self,
        PowerLossObj: PowerLoss,
        Ne: DataArray = None,
        Te: DataArray = None,
        Nh: DataArray = None,
        F_z_t: DataArray = None,
    ):
        self.PLT = PowerLossObj.PLT
        self.PRB = PowerLossObj.PRB
        self.PRC = PowerLossObj.PRC

        self.Ne = PowerLossObj.Ne if Ne is None else Ne
        self.Te = PowerLossObj.Te if Te is None else Te
        self.Nh = PowerLossObj.Nh if Nh is None else Nh
        self.F_z_t = PowerLossObj.F_z_t if F_z_t is None else F_z_t

        self.PowerLossObj = PowerLossObj

        self.nominal_inputs = [
            self.PLT,
            self.PRB,
            self.PRC,
            self.Ne,
            self.Te,
            self.Nh,
            self.F_z_t,
        ]

    def init_type_check(self, PLT=None, PRB=None, PRC=None):
        inputs = [PLT, PRB, PRC]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        PLT, PRB, PRC = inputs

        """Test type errors are raised for PowerLoss initialisation."""
        with self.assertRaises(TypeError):
            PowerLoss(PLT, PRB, PRC)

    def init_value_error_check(self, PLT=None, PRB=None, PRC=None):
        inputs = [PLT, PRB, PRC]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        PLT, PRB, PRC = inputs

        """Test value errors are raised for PowerLoss initialisation."""
        with self.assertRaises(ValueError):
            PowerLoss(PLT, PRB, PRC)

    def interpolation_type_check(self, Ne=None, Te=None):
        """Test type errors are raise for interpolation_power function"""
        inputs = [Ne, Te]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 3]

        Ne, Te = inputs

        with self.assertRaises(TypeError):
            self.PowerLossObj.interpolate_power(Ne, Te)

    def interpolation_value_check(self, Ne=None, Te=None):
        """Test value errors are raised for interpolation_power function."""
        inputs = [Ne, Te]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 3]

        Ne, Te = inputs

        with self.assertRaises(ValueError):
            self.PowerLossObj.interpolate_power(Ne, Te)

    def call_type_check(self, Ne=None, Te=None, Nh=None, F_z_t=None):
        """Test type errors are raised for PowerLoss call."""
        inputs = [Ne, Te, Nh, F_z_t]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 3]

        Ne, Te, Nh, F_z_t = inputs

        with self.assertRaises(TypeError):
            self.PowerLossObj(Ne, Te, Nh, F_z_t, full_run=False)

    def call_value_check(self, Ne=None, Te=None, Nh=None, F_z_t=None):
        """Test value errors are raised for PowerLoss call."""
        inputs = [Ne, Te, Nh, F_z_t]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 3]

        Ne, Te, Nh, F_z_t = inputs

        with self.assertRaises(ValueError):
            self.PowerLossObj(Ne, Te, Nh, F_z_t, full_run=False)

    def partial_input_value_check(self, Ne=None, Nh=None, F_z_t=None):
        nominal_inputs = [self.Ne, self.Nh, self.F_z_t]

        inputs = [Ne, Nh, F_z_t]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = nominal_inputs[i]

        Ne, Nh, F_z_t = inputs

        with self.assertRaises(ValueError):
            self.PowerLossObj.calculate_power_loss(Ne, Nh, F_z_t)


def input_error_check(
    invalid_input_name, invalid_input, error_check, test_case, func_name
):
    """Helper function for test_case checks"""

    # Initialization invalid input checks
    if func_name == "__init__":
        invalid_input_dict = {
            "SCD": {"SCD": invalid_input},
            "ACD": {"ACD": invalid_input},
            "CCD": {"CCD": invalid_input},
            "PLT": {"PLT": invalid_input},
            "PRC": {"PRC": invalid_input},
            "PRB": {"PRB": invalid_input},
        }.get(invalid_input_name)

        if invalid_input_dict:
            {
                TypeError: test_case.init_type_check,
                ValueError: test_case.init_value_error_check,
            }.get(error_check)(**invalid_input_dict)

    # interpolate_rates invalid input checks
    if func_name == "interpolate_rates" or func_name == "interpolate_power":
        invalid_input_dict = {
            "Ne": {"Ne": invalid_input},
            "Te": {"Te": invalid_input},
        }.get(invalid_input_name)

        if invalid_input_dict:
            {
                TypeError: test_case.interpolation_type_check,
                ValueError: test_case.interpolation_value_check,
            }.get(error_check)(**invalid_input_dict)

    # calc_ionisation_balance_matrix invalid input checks
    if func_name == "calc_ionisation_balance_matrix":
        invalid_input_dict = {
            "Ne": {"Ne": invalid_input},
            "Nh": {"Nh": invalid_input},
        }.get(invalid_input_name)

        if invalid_input_dict:
            {
                TypeError: test_case.calc_ionisation_balance_type_check,
                ValueError: test_case.calc_ionisation_balance_value_check,
            }.get(error_check)(**invalid_input_dict)

    if func_name == "partial_calc_ionisation_balance_matrix":
        invalid_input_dict = {
            "Ne": {"Ne": invalid_input},
            "Nh": {"Nh": invalid_input},
        }.get(invalid_input_name)

        if invalid_input_dict:
            {
                ValueError: test_case.calc_ionisation_balance_partial_input_value_check
            }.get(error_check)(**invalid_input_dict)

    if func_name == "partial_powerloss_inputs":
        invalid_input_dict = {
            "Ne": {"Ne": invalid_input},
            "Nh": {"Nh": invalid_input},
            "F_z_t": {"F_z_t": invalid_input},
        }.get(invalid_input_name)

        if invalid_input_dict:
            {ValueError: test_case.partial_input_value_check}.get(error_check)(
                **invalid_input_dict
            )

    # calc_eigen_coeffs invalid input checks
    if func_name == "calc_eigen_coeffs":
        invalid_input_dict = {
            "F_z_t0": {"F_z_t0": invalid_input},
        }.get(invalid_input_name)

        if invalid_input_dict:
            {
                TypeError: test_case.calc_eigen_coeffs_type_check,
                ValueError: test_case.calc_eigen_coeffs_value_check,
            }.get(error_check)(**invalid_input_dict)

    # FractionalAbundance call invalid input checks
    if func_name == "FracAbund__call__":
        invalid_input_dict = {
            "Ne": {"Ne": invalid_input},
            "Te": {"Te": invalid_input},
            "Nh": {"Nh": invalid_input},
            "tau": {"tau": invalid_input},
        }.get(invalid_input_name)

        if invalid_input_dict:
            {
                TypeError: test_case.call_type_check,
                ValueError: test_case.call_value_check,
            }.get(error_check)(**invalid_input_dict)

    if func_name == "PowerLoss__call__":
        invalid_input_dict = {
            "Ne": {"Ne": invalid_input},
            "Te": {"Te": invalid_input},
            "Nh": {"Nh": invalid_input},
            "F_z_t": {"F_z_t": invalid_input},
        }.get(invalid_input_name)

        if invalid_input_dict:
            {
                TypeError: test_case.call_type_check,
                ValueError: test_case.call_value_check,
            }.get(error_check)(**invalid_input_dict)


@pytest.fixture
def test_fractional_abundance_init():
    """Test initialisation of FractionalAbundance class."""
    ADAS_file = ADASReader()

    element = "be"

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")
    CCD = ADAS_file.get_adf11("ccd", element, "89")

    try:
        example_frac_abundance = FractionalAbundance(
            SCD,
            ACD,
            CCD=CCD,
        )
    except Exception as e:
        raise e

    # Test omission of optional inputs, CCD and Nh
    try:
        example_frac_abundance_no_optional = FractionalAbundance(
            SCD,
            ACD,
        )
    except Exception as e:
        raise e
    assert example_frac_abundance_no_optional.CCD is None

    test_case = Exception_Frac_Abund_Test_Case(example_frac_abundance)

    # Tests for correct raising of ValueError when CCD is provided without Nh.
    # Needs moving to check for calc_ionisation_balance_matrix
    # test_case.init_partial_input_value_check(
    #     SCD,
    #     ACD,
    #     CCD,
    # )

    init_func_name = test_case.FracAbundObj.__init__.__name__

    # SCD checks

    SCD_invalid = copy.deepcopy(SCD.data)
    input_error_check("SCD", SCD_invalid, TypeError, test_case, init_func_name)

    SCD_invalid = SCD.copy(deep=True)
    SCD_invalid.data = -1 * copy.deepcopy(SCD_invalid.data)
    input_error_check("SCD", SCD_invalid, ValueError, test_case, init_func_name)

    SCD_invalid.data = np.nan * copy.deepcopy(SCD.data)
    input_error_check("SCD", SCD_invalid, ValueError, test_case, init_func_name)

    SCD_invalid.data = np.inf * copy.deepcopy(SCD.data)
    input_error_check("SCD", SCD_invalid, ValueError, test_case, init_func_name)

    SCD_invalid.data = -np.inf * copy.deepcopy(SCD.data)
    input_error_check("SCD", SCD_invalid, ValueError, test_case, init_func_name)

    SCD_invalid = SCD.copy(deep=True)
    SCD_invalid = SCD_invalid.expand_dims("blank")
    input_error_check("SCD", SCD_invalid, ValueError, test_case, init_func_name)

    # ACD checks

    ACD_invalid = copy.deepcopy(ACD.data)
    input_error_check("ACD", ACD_invalid, TypeError, test_case, init_func_name)

    ACD_invalid = ACD.copy(deep=True)
    ACD_invalid.data = -1 * copy.deepcopy(ACD_invalid.data)
    input_error_check("ACD", ACD_invalid, ValueError, test_case, init_func_name)

    ACD_invalid.data = np.nan * copy.deepcopy(ACD.data)
    input_error_check("ACD", ACD_invalid, ValueError, test_case, init_func_name)

    ACD_invalid.data = np.inf * copy.deepcopy(ACD.data)
    input_error_check("ACD", ACD_invalid, ValueError, test_case, init_func_name)

    ACD_invalid.data = -np.inf * copy.deepcopy(ACD.data)
    input_error_check("ACD", ACD_invalid, ValueError, test_case, init_func_name)

    ACD_invalid = ACD.copy(deep=True)
    ACD_invalid = ACD_invalid.expand_dims("blank")
    input_error_check("ACD", ACD_invalid, ValueError, test_case, init_func_name)

    # CCD checks

    CCD_invalid = copy.deepcopy(CCD.data)
    input_error_check("CCD", CCD_invalid, TypeError, test_case, init_func_name)

    CCD_invalid = CCD.copy(deep=True)
    CCD_invalid.data = -1 * copy.deepcopy(CCD_invalid.data)
    input_error_check("CCD", CCD_invalid, ValueError, test_case, init_func_name)

    CCD_invalid.data = np.nan * copy.deepcopy(CCD.data)
    input_error_check("CCD", CCD_invalid, ValueError, test_case, init_func_name)

    CCD_invalid.data = np.inf * copy.deepcopy(CCD.data)
    input_error_check("CCD", CCD_invalid, ValueError, test_case, init_func_name)

    CCD_invalid.data = -np.inf * copy.deepcopy(CCD.data)
    input_error_check("CCD", CCD_invalid, ValueError, test_case, init_func_name)

    CCD_invalid = CCD.copy(deep=True)
    CCD_invalid = CCD_invalid.expand_dims("blank")
    input_error_check("CCD", CCD_invalid, ValueError, test_case, init_func_name)

    return example_frac_abundance, example_frac_abundance_no_optional


@pytest.fixture
def test_interpolate_rates(test_fractional_abundance_init):
    """Test interpolate_rates() function in FractionalAbundance class."""
    (
        example_frac_abundance,
        example_frac_abundance_no_optional,
    ) = test_fractional_abundance_init

    input_Ne = np.logspace(19.0, 16.0, 10)
    input_Ne = DataArray(
        data=input_Ne,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    input_Te = np.logspace(4.6, 2, 10)
    input_Te = DataArray(
        data=input_Te,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    test_case = Exception_Frac_Abund_Test_Case(
        example_frac_abundance, Ne=input_Ne, Te=input_Te
    )

    interp_func_name = test_case.FracAbundObj.interpolate_rates.__name__

    # Electron density checks
    input_Ne_invalid = copy.deepcopy(input_Ne.data)
    input_error_check("Ne", input_Ne_invalid, TypeError, test_case, interp_func_name)

    input_Ne_invalid = np.logspace(30.0, 16.0, 10)
    input_Ne_invalid = DataArray(
        data=input_Ne_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    input_error_check("Ne", input_Ne_invalid, ValueError, test_case, interp_func_name)

    input_Ne_invalid.data = np.logspace(19.0, 5.0, 10)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case, interp_func_name)

    input_Ne_invalid.data = np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case, interp_func_name)

    input_Ne_invalid.data = -np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case, interp_func_name)

    input_Ne_invalid.data = -1 + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case, interp_func_name)

    input_Ne_invalid.data = np.nan + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case, interp_func_name)

    input_Ne_invalid = input_Ne.copy(deep=True)
    input_Ne_invalid = input_Ne_invalid.expand_dims("blank")
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case, interp_func_name)

    # Electron temperature check

    input_Te_invalid = copy.deepcopy(input_Te.data)
    input_error_check("Te", input_Te_invalid, TypeError, test_case, interp_func_name)

    input_Te_invalid = np.logspace(5, 2, 10)
    input_Te_invalid = DataArray(
        data=input_Te_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check("Te", input_Te_invalid, ValueError, test_case, interp_func_name)

    input_Te_invalid.data = np.logspace(4.6, -1, 10)
    input_error_check("Te", input_Te_invalid, ValueError, test_case, interp_func_name)

    input_Te_invalid.data = np.inf + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case, interp_func_name)

    input_Te_invalid.data = -np.inf + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case, interp_func_name)

    input_Te_invalid.data = -1 + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case, interp_func_name)

    input_Te_invalid.data = np.nan + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case, interp_func_name)

    input_Te_invalid = input_Te.copy(deep=True)
    input_Te_invalid = input_Te_invalid.expand_dims("blank")
    input_error_check("Te", input_Te_invalid, ValueError, test_case, interp_func_name)

    try:
        (
            SCD_spec,
            ACD_spec,
            CCD_spec,
            num_of_ion_charges,
        ) = example_frac_abundance_no_optional.interpolate_rates(input_Ne, input_Te)
    except Exception as e:
        raise e

    assert SCD_spec.shape == (4, 10)
    assert ACD_spec.shape == (4, 10)
    assert CCD_spec is None

    assert np.all(np.logical_not(np.isnan(SCD_spec)))
    assert np.all(np.logical_not(np.isnan(ACD_spec)))

    assert np.all(np.logical_not(np.isinf(SCD_spec)))
    assert np.all(np.logical_not(np.isinf(ACD_spec)))

    try:
        (
            SCD_spec,
            ACD_spec,
            CCD_spec,
            num_of_ion_charges,
        ) = example_frac_abundance.interpolate_rates(input_Ne, input_Te)
    except Exception as e:
        raise e

    assert num_of_ion_charges == 5

    assert SCD_spec.shape == (4, 10)
    assert ACD_spec.shape == (4, 10)
    assert CCD_spec.shape == (4, 10)

    assert np.all(np.logical_not(np.isnan(SCD_spec)))
    assert np.all(np.logical_not(np.isnan(ACD_spec)))
    assert np.all(np.logical_not(np.isnan(CCD_spec)))

    assert np.all(np.logical_not(np.isinf(SCD_spec)))
    assert np.all(np.logical_not(np.isinf(ACD_spec)))
    assert np.all(np.logical_not(np.isinf(CCD_spec)))

    return example_frac_abundance, example_frac_abundance_no_optional


@pytest.fixture
def test_calc_ionisation_balance_matrix(test_interpolate_rates):
    """Test calc_ionisation_balance_matrix() function in FractionalAbundance class."""
    example_frac_abundance, example_frac_abundance_no_optional = test_interpolate_rates

    input_Ne = np.logspace(19.0, 16.0, 10)
    input_Ne = DataArray(
        data=input_Ne,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    input_Nh = 1e-5 * input_Ne
    input_Nh = DataArray(
        data=input_Nh,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    test_case = Exception_Frac_Abund_Test_Case(example_frac_abundance, Ne=input_Ne)

    partial_input_func_name = "partial_calc_ionisation_balance_matrix"

    input_error_check("Nh", None, ValueError, test_case, partial_input_func_name)

    test_case = Exception_Frac_Abund_Test_Case(
        example_frac_abundance_no_optional, Ne=input_Ne, Nh=input_Nh
    )

    input_error_check("Nh", input_Nh, ValueError, test_case, partial_input_func_name)

    test_case = Exception_Frac_Abund_Test_Case(
        example_frac_abundance, Ne=input_Ne, Nh=input_Nh
    )

    ionisation_func_name = (
        test_case.FracAbundObj.calc_ionisation_balance_matrix.__name__
    )

    # Electron density checks
    input_Ne_invalid = copy.deepcopy(input_Ne.data)
    input_error_check(
        "Ne", input_Ne_invalid, TypeError, test_case, ionisation_func_name
    )

    input_Ne_invalid = input_Ne.copy(deep=True)

    input_Ne_invalid.data = np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, ionisation_func_name
    )

    input_Ne_invalid.data = -np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, ionisation_func_name
    )

    input_Ne_invalid.data = -1 + np.zeros(input_Ne_invalid.data.shape)
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, ionisation_func_name
    )

    input_Ne_invalid.data = np.nan + np.zeros(input_Ne_invalid.data.shape)
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, ionisation_func_name
    )

    input_Ne_invalid = input_Ne.copy(deep=True)
    input_Ne_invalid = input_Ne_invalid.expand_dims("blank")
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, ionisation_func_name
    )

    # Thermal hydrogen density check

    input_Nh_invalid = copy.deepcopy(input_Nh.data)
    input_error_check(
        "Nh", input_Nh_invalid, TypeError, test_case, ionisation_func_name
    )

    input_Nh_invalid = np.inf + np.zeros(input_Nh_invalid.data.shape)
    input_Nh_invalid = DataArray(
        data=input_Nh_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check(
        "Nh", input_Nh_invalid, ValueError, test_case, ionisation_func_name
    )

    input_Nh_invalid.data = -np.inf + np.zeros(input_Nh_invalid.data.shape)
    input_error_check(
        "Nh", input_Nh_invalid, ValueError, test_case, ionisation_func_name
    )

    input_Nh_invalid.data = -1 + np.zeros(input_Nh_invalid.data.shape)
    input_error_check(
        "Nh", input_Nh_invalid, ValueError, test_case, ionisation_func_name
    )

    input_Nh_invalid.data = np.nan + np.zeros(input_Nh_invalid.data.shape)
    input_error_check(
        "Nh", input_Nh_invalid, ValueError, test_case, ionisation_func_name
    )

    input_Nh_invalid = input_Nh.copy(deep=True)
    input_Nh_invalid = input_Nh_invalid.expand_dims("blank")
    input_error_check(
        "Nh", input_Nh_invalid, ValueError, test_case, ionisation_func_name
    )

    try:
        ionisation_balance_matrix = (
            example_frac_abundance_no_optional.calc_ionisation_balance_matrix(
                input_Ne,
            )
        )
    except Exception as e:
        raise e

    assert ionisation_balance_matrix.shape == (5, 5, 10)

    assert np.all(np.logical_not(np.isnan(ionisation_balance_matrix)))

    assert np.all(np.logical_not(np.isinf(ionisation_balance_matrix)))

    try:
        ionisation_balance_matrix = (
            example_frac_abundance.calc_ionisation_balance_matrix(input_Ne, input_Nh)
        )
    except Exception as e:
        raise e

    assert ionisation_balance_matrix.shape == (5, 5, 10)

    assert np.all(np.logical_not(np.isnan(ionisation_balance_matrix)))

    assert np.all(np.logical_not(np.isinf(ionisation_balance_matrix)))

    return example_frac_abundance, example_frac_abundance_no_optional


@pytest.fixture
def test_calc_F_z_tinf(test_calc_ionisation_balance_matrix):
    """Test calc_F_z_tinf function in in FractionalAbundance class."""
    (
        example_frac_abundance,
        example_frac_abundance_no_optional,
    ) = test_calc_ionisation_balance_matrix

    try:
        F_z_tinf = example_frac_abundance_no_optional.calc_F_z_tinf()
    except Exception as e:
        raise e

    assert F_z_tinf.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(F_z_tinf)))

    assert np.all(np.logical_not(np.isinf(F_z_tinf)))

    rho = example_frac_abundance_no_optional.x1_coord
    ionisation_balance_matrix = (
        example_frac_abundance_no_optional.ionisation_balance_matrix
    )

    for irho in range(rho.size):
        test_null = np.dot(ionisation_balance_matrix[:, :, irho], F_z_tinf[:, irho])
        assert np.allclose(test_null, np.zeros(test_null.shape))

        test_normalization = np.sum(F_z_tinf[:, irho])
        assert np.allclose(test_normalization, 1.0, rtol=1e-2)

    try:
        F_z_tinf = example_frac_abundance.calc_F_z_tinf()
    except Exception as e:
        raise e

    assert F_z_tinf.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(F_z_tinf)))

    assert np.all(np.logical_not(np.isinf(F_z_tinf)))

    rho = example_frac_abundance.x1_coord
    ionisation_balance_matrix = example_frac_abundance.ionisation_balance_matrix

    for irho in range(rho.size):
        test_null = np.dot(ionisation_balance_matrix[:, :, irho], F_z_tinf[:, irho])
        assert np.allclose(test_null, np.zeros(test_null.shape))

        test_normalization = np.sum(F_z_tinf[:, irho])
        assert np.allclose(test_normalization, 1.0)

    return example_frac_abundance, example_frac_abundance_no_optional


@pytest.fixture
def test_calc_eigen_vals_and_vecs(test_calc_F_z_tinf):
    """Test calc_eigen_vals_and_vecs() function in FractionalAbundance class."""
    example_frac_abundance, example_frac_abundance_no_optional = test_calc_F_z_tinf

    try:
        (
            eig_vals,
            eig_vecs,
        ) = example_frac_abundance_no_optional.calc_eigen_vals_and_vecs()
    except Exception as e:
        raise e

    assert eig_vals.shape == (5, 10)
    assert eig_vecs.shape == (5, 5, 10)

    assert np.all(np.logical_not(np.isnan(eig_vals)))
    assert np.all(np.logical_not(np.isinf(eig_vals)))

    assert np.all(np.logical_not(np.isnan(eig_vecs)))
    assert np.all(np.logical_not(np.isinf(eig_vecs)))

    rho = example_frac_abundance_no_optional.x1_coord
    ionisation_balance_matrix = (
        example_frac_abundance_no_optional.ionisation_balance_matrix
    )

    for irho in range(rho.size):
        for ieig in range(example_frac_abundance_no_optional.num_of_ion_charges):
            test_eigen = np.dot(
                ionisation_balance_matrix[:, :, irho], eig_vecs[:, ieig, irho]
            ) - np.dot(eig_vals[ieig, irho], eig_vecs[:, ieig, irho])

            assert np.allclose(test_eigen, np.zeros(test_eigen.shape))

    try:
        eig_vals, eig_vecs = example_frac_abundance.calc_eigen_vals_and_vecs()
    except Exception as e:
        raise e

    assert eig_vals.shape == (5, 10)
    assert eig_vecs.shape == (5, 5, 10)

    assert np.all(np.logical_not(np.isnan(eig_vals)))
    assert np.all(np.logical_not(np.isinf(eig_vals)))

    assert np.all(np.logical_not(np.isnan(eig_vecs)))
    assert np.all(np.logical_not(np.isinf(eig_vecs)))

    rho = example_frac_abundance.x1_coord
    ionisation_balance_matrix = example_frac_abundance.ionisation_balance_matrix

    for irho in range(rho.size):
        for ieig in range(example_frac_abundance.num_of_ion_charges):
            test_eigen = np.dot(
                ionisation_balance_matrix[:, :, irho], eig_vecs[:, ieig, irho]
            ) - np.dot(eig_vals[ieig, irho], eig_vecs[:, ieig, irho])

            assert np.allclose(test_eigen, np.zeros(test_eigen.shape))

    return example_frac_abundance, example_frac_abundance_no_optional


@pytest.fixture
def test_calc_eigen_coeffs(test_calc_eigen_vals_and_vecs):
    """Test calc_eigen_coeffs() function in FractionalAbundance class."""
    (
        example_frac_abundance,
        example_frac_abundance_no_optional,
    ) = test_calc_eigen_vals_and_vecs

    # Stick with default F_z_t0=None assignment
    test_case = Exception_Frac_Abund_Test_Case(
        example_frac_abundance,
    )

    eigen_coeffs_func_name = test_case.FracAbundObj.calc_eigen_coeffs.__name__

    # Initial fractional abundance checks
    F_z_t0_invalid = [1.0, 0.0, 0.0, 0.0, 0.0]
    input_error_check(
        "F_z_t0", F_z_t0_invalid, TypeError, test_case, eigen_coeffs_func_name
    )

    F_z_t0_invalid = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    input_error_check(
        "F_z_t0", F_z_t0_invalid, TypeError, test_case, eigen_coeffs_func_name
    )

    F_z_t0_invalid = DataArray(
        data=np.array([[-1.0, 0.0, 0.0, 0.0, 0.0]]).T,
        coords=[("ion_charges", np.linspace(0, 4, 5)), ("rho_poloidal", [0.5])],
        dims=["ion_charges", "rho_poloidal"],
    )
    input_error_check(
        "F_z_t0", F_z_t0_invalid, ValueError, test_case, eigen_coeffs_func_name
    )

    F_z_t0_invalid.data = np.array([np.zeros(5) + np.nan]).T
    input_error_check(
        "F_z_t0", F_z_t0_invalid, ValueError, test_case, eigen_coeffs_func_name
    )

    F_z_t0_invalid.data = np.array([np.zeros(5) + np.inf]).T
    input_error_check(
        "F_z_t0", F_z_t0_invalid, ValueError, test_case, eigen_coeffs_func_name
    )

    F_z_t0_invalid.data = np.array([np.zeros(5) - np.inf]).T
    input_error_check(
        "F_z_t0", F_z_t0_invalid, ValueError, test_case, eigen_coeffs_func_name
    )

    F_z_t0_invalid = DataArray(
        data=np.array([[[1.0, 0.0, 0.0, 0.0, 0.0]]]).T,
        coords=[
            ("ion_charges", np.linspace(0, 4, 5)),
            ("rho_poloidal", [0.5]),
            ("t", [77.5]),
        ],
        dims=["ion_charges", "rho_poloidal", "t"],
    )
    input_error_check(
        "F_z_t0", F_z_t0_invalid, ValueError, test_case, eigen_coeffs_func_name
    )

    try:
        # Stick with default F_z_t0=None assignment
        eig_coeffs, F_z_t0 = example_frac_abundance_no_optional.calc_eigen_coeffs()
    except Exception as e:
        raise e

    assert eig_coeffs.shape == (5, 10)
    assert F_z_t0.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(eig_coeffs)))
    assert np.all(np.logical_not(np.isinf(eig_coeffs)))

    assert np.all(np.logical_not(np.isnan(F_z_t0)))
    assert np.all(np.logical_not(np.isinf(F_z_t0)))

    try:
        # Stick with default F_z_t0=None assignment
        eig_coeffs, F_z_t0 = example_frac_abundance.calc_eigen_coeffs()
    except Exception as e:
        raise e

    assert eig_coeffs.shape == (5, 10)
    assert F_z_t0.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(eig_coeffs)))
    assert np.all(np.logical_not(np.isinf(eig_coeffs)))

    assert np.all(np.logical_not(np.isnan(F_z_t0)))
    assert np.all(np.logical_not(np.isinf(F_z_t0)))

    return example_frac_abundance, example_frac_abundance_no_optional


def test_frac_abund_call(test_calc_eigen_coeffs):
    """Test FractionalAbundance class __call__()."""
    example_frac_abundance, example_frac_abundance_no_optional = test_calc_eigen_coeffs

    tau = 1e-16

    test_case = Exception_Frac_Abund_Test_Case(example_frac_abundance, tau=tau)

    call_func_name = "FracAbund" + test_case.FracAbundObj.__call__.__name__

    invalid_tau = "tau"
    input_error_check("tau", invalid_tau, TypeError, test_case, call_func_name)

    invalid_tau = -1
    input_error_check("tau", invalid_tau, ValueError, test_case, call_func_name)

    invalid_tau = np.inf
    input_error_check("tau", invalid_tau, ValueError, test_case, call_func_name)

    invalid_tau = -np.inf
    input_error_check("tau", invalid_tau, ValueError, test_case, call_func_name)

    invalid_tau = np.nan
    input_error_check("tau", invalid_tau, ValueError, test_case, call_func_name)

    input_Ne = np.logspace(19.0, 16.0, 10)
    input_Ne = DataArray(
        data=input_Ne,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    input_Te = np.logspace(4.6, 2, 10)
    input_Te = DataArray(
        data=input_Te,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    input_Nh = 1e-5 * input_Ne
    input_Nh = DataArray(
        data=input_Nh,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    try:
        F_z_t = example_frac_abundance_no_optional(
            input_Ne,
            input_Te,
            tau=tau,
            full_run=False,
        )
    except Exception as e:
        raise e

    assert F_z_t.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(F_z_t, example_frac_abundance_no_optional.F_z_t0, atol=1e-4)

    tau = 1e2

    try:
        F_z_t = example_frac_abundance_no_optional(
            input_Ne,
            input_Te,
            tau=tau,
            full_run=False,
        )
    except Exception as e:
        raise e

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(F_z_t, example_frac_abundance_no_optional.F_z_tinf, atol=2e-2)

    rho = example_frac_abundance_no_optional.x1_coord

    for irho in range(rho.size):
        test_normalization = np.sum(F_z_t[:, irho])
        assert np.abs(test_normalization - 1.0) <= 2e-2

    # Testing tau as a profile of rho_poloidal.
    tau = np.linspace(1.0, 1.0e-10, 10)
    tau = DataArray(
        data=tau,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    try:
        F_z_t = example_frac_abundance_no_optional(
            input_Ne,
            input_Te,
            tau=tau,
            full_run=False,
        )
    except Exception as e:
        raise e

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(
        F_z_t[:, 0], example_frac_abundance_no_optional.F_z_tinf[:, 0], atol=2e-2
    )
    assert np.allclose(
        F_z_t[:, -1], example_frac_abundance_no_optional.F_z_t0[:, -1], atol=1e-4
    )

    for irho in range(rho.size):
        test_normalization = np.sum(F_z_t[:, irho])
        assert np.abs(test_normalization - 1.0) <= 2e-2

    tau = 1e-16
    try:
        F_z_t = example_frac_abundance(
            input_Ne,
            input_Te,
            input_Nh,
            tau=tau,
            full_run=False,
        )
    except Exception as e:
        raise e

    assert F_z_t.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(F_z_t, example_frac_abundance.F_z_t0, atol=1e-5)

    tau = 1e2

    try:
        F_z_t = example_frac_abundance(
            input_Ne,
            input_Te,
            input_Nh,
            tau=tau,
            full_run=False,
        )
    except Exception as e:
        raise e

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(F_z_t, example_frac_abundance.F_z_tinf, atol=2e-2)

    rho = example_frac_abundance.x1_coord

    for irho in range(rho.size):
        test_normalization = np.sum(F_z_t[:, irho])
        assert np.abs(test_normalization - 1.0) <= 2e-2

    # Testing tau as a profile of rho_poloidal.
    tau = np.linspace(1.0, 1.0e-10, 10)
    tau = DataArray(
        data=tau,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    try:
        F_z_t = example_frac_abundance(
            input_Ne,
            input_Te,
            input_Nh,
            tau=tau,
            full_run=False,
        )
    except Exception as e:
        raise e

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(F_z_t[:, 0], example_frac_abundance.F_z_tinf[:, 0], atol=2e-2)
    assert np.allclose(F_z_t[:, -1], example_frac_abundance.F_z_t0[:, -1], atol=1e-4)

    for irho in range(rho.size):
        test_normalization = np.sum(F_z_t[:, irho])
        assert np.abs(test_normalization - 1.0) <= 2e-2


@pytest.fixture
def test_power_loss_init():
    """Test initialisation of PowerLoss class."""
    ADAS_file = ADASReader()

    element = "be"

    PLT = ADAS_file.get_adf11("plt", element, "89")
    PRC = ADAS_file.get_adf11("prc", element, "89")
    PRB = ADAS_file.get_adf11("prb", element, "89")

    try:
        example_power_loss = PowerLoss(PLT, PRB, PRC=PRC)
    except Exception as e:
        raise e

    # Test omission of optional inputs, PRC and Nh
    try:
        example_power_loss_no_optional = PowerLoss(PLT, PRB)
    except Exception as e:
        raise e

    test_case = Exception_Power_Loss_Test_Case(example_power_loss)

    init_func_name = example_power_loss.__init__.__name__

    # PLT checks

    PLT_invalid = copy.deepcopy(PLT.data)
    input_error_check("PLT", PLT_invalid, TypeError, test_case, init_func_name)

    PLT_invalid = PLT.copy(deep=True)
    PLT_invalid.data = -1 * copy.deepcopy(PLT_invalid.data)
    input_error_check("PLT", PLT_invalid, ValueError, test_case, init_func_name)

    PLT_invalid.data = np.nan * copy.deepcopy(PLT_invalid)
    input_error_check("PLT", PLT_invalid, ValueError, test_case, init_func_name)

    PLT_invalid.data = np.inf * copy.deepcopy(PLT_invalid)
    input_error_check("PLT", PLT_invalid, ValueError, test_case, init_func_name)

    PLT_invalid.data = -np.inf * copy.deepcopy(PLT_invalid)
    input_error_check("PLT", PLT_invalid, ValueError, test_case, init_func_name)

    PLT_invalid = PLT.copy(deep=True)
    PLT_invalid = PLT_invalid.expand_dims("blank")
    input_error_check("PLT", PLT_invalid, ValueError, test_case, init_func_name)

    # PRB checks

    PRB_invalid = copy.deepcopy(PRB.data)
    input_error_check("PRB", PRB_invalid, TypeError, test_case, init_func_name)

    PRB_invalid = PRB.copy(deep=True)
    PRB_invalid.data = -1 * copy.deepcopy(PRB_invalid.data)
    input_error_check("PRB", PRB_invalid, ValueError, test_case, init_func_name)

    PRB_invalid.data = np.nan * copy.deepcopy(PRB_invalid)
    input_error_check("PRB", PRB_invalid, ValueError, test_case, init_func_name)

    PRB_invalid.data = np.inf * copy.deepcopy(PRB_invalid)
    input_error_check("PRB", PRB_invalid, ValueError, test_case, init_func_name)

    PRB_invalid.data = -np.inf * copy.deepcopy(PRB_invalid)
    input_error_check("PRB", PRB_invalid, ValueError, test_case, init_func_name)

    PRB_invalid = PRB.copy(deep=True)
    PRB_invalid = PRB_invalid.expand_dims("blank")
    input_error_check("PRB", PRB_invalid, ValueError, test_case, init_func_name)

    # PRC checks

    PRC_invalid = copy.deepcopy(PRC.data)
    input_error_check("PRC", PRC_invalid, TypeError, test_case, init_func_name)

    PRC_invalid = PRC.copy(deep=True)
    PRC_invalid.data = -1 * copy.deepcopy(PRC_invalid.data)
    input_error_check("PRC", PRC_invalid, ValueError, test_case, init_func_name)

    PRC_invalid.data = np.nan * copy.deepcopy(PRC_invalid)
    input_error_check("PRC", PRC_invalid, ValueError, test_case, init_func_name)

    PRC_invalid.data = np.inf * copy.deepcopy(PRC_invalid)
    input_error_check("PRC", PRC_invalid, ValueError, test_case, init_func_name)

    PRC_invalid.data = -np.inf * copy.deepcopy(PRC_invalid)
    input_error_check("PRC", PRC_invalid, ValueError, test_case, init_func_name)

    PRC_invalid = PRC.copy(deep=True)
    PRC_invalid = PRC_invalid.expand_dims("blank")
    input_error_check("PRC", PRC_invalid, ValueError, test_case, init_func_name)

    return example_power_loss, example_power_loss_no_optional


@pytest.fixture
def test_interpolate_power(test_power_loss_init):
    """Test interpolate_power() function in PowerLoss class."""
    example_power_loss, example_power_loss_no_optional = test_power_loss_init

    input_Ne = np.logspace(19.0, 16.0, 10)

    input_Ne = DataArray(
        data=input_Ne,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    input_Te = np.logspace(4.6, 2, 10)

    input_Te = DataArray(
        data=input_Te,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    test_case = Exception_Power_Loss_Test_Case(
        example_power_loss, Ne=input_Ne, Te=input_Te
    )

    interpolate_func_name = example_power_loss.interpolate_power.__name__

    # Electron density checks

    input_Ne_invalid = copy.deepcopy(input_Ne.data)
    input_error_check(
        "Ne", input_Ne_invalid, TypeError, test_case, interpolate_func_name
    )

    input_Ne_invalid = np.logspace(30.0, 16.0, 10)
    input_Ne_invalid = DataArray(
        data=input_Ne_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Ne_invalid.data = np.logspace(19.0, 5.0, 10)
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Ne_invalid.data = np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Ne_invalid.data = -np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Ne_invalid.data = -1 + np.zeros(input_Ne_invalid.data.shape)
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Ne_invalid.data = np.nan + np.zeros(input_Ne_invalid.data.shape)
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Ne_invalid = input_Ne.copy(deep=True)
    input_Ne_invalid = input_Ne_invalid.expand_dims("blank")
    input_error_check(
        "Ne", input_Ne_invalid, ValueError, test_case, interpolate_func_name
    )

    # Electron temperature check

    input_Te_invalid = copy.deepcopy(input_Te.data)
    input_error_check(
        "Te", input_Te_invalid, TypeError, test_case, interpolate_func_name
    )

    input_Te_invalid = np.logspace(5, 2, 10)
    input_Te_invalid = DataArray(
        data=input_Te_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check(
        "Te", input_Te_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Te_invalid.data = np.logspace(4.6, -1, 10)
    input_error_check(
        "Te", input_Te_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Te_invalid.data = np.inf + np.zeros(input_Te_invalid.data.shape)
    input_error_check(
        "Te", input_Te_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Te_invalid.data = -np.inf + np.zeros(input_Te_invalid.data.shape)
    input_error_check(
        "Te", input_Te_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Te_invalid.data = -1 + np.zeros(input_Te_invalid.data.shape)
    input_error_check(
        "Te", input_Te_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Te_invalid.data = np.nan + np.zeros(input_Te_invalid.data.shape)
    input_error_check(
        "Te", input_Te_invalid, ValueError, test_case, interpolate_func_name
    )

    input_Te_invalid = input_Te.copy(deep=True)
    input_Te_invalid = input_Te_invalid.expand_dims("blank")
    input_error_check(
        "Te", input_Te_invalid, ValueError, test_case, interpolate_func_name
    )

    try:
        (
            PLT_spec,
            PRC_spec,
            PRB_spec,
            _,
        ) = example_power_loss_no_optional.interpolate_power(Ne=input_Ne, Te=input_Te)
    except Exception as e:
        raise e

    assert PLT_spec.shape == (4, 10)
    assert PRB_spec.shape == (4, 10)
    assert PRC_spec is None

    assert np.all(np.logical_not(np.isnan(PLT_spec)))
    assert np.all(np.logical_not(np.isnan(PRB_spec)))

    assert np.all(np.logical_not(np.isinf(PLT_spec)))
    assert np.all(np.logical_not(np.isinf(PRB_spec)))

    try:
        PLT_spec, PRC_spec, PRB_spec, _ = example_power_loss.interpolate_power(
            Ne=input_Ne, Te=input_Te
        )
    except Exception as e:
        raise e

    assert PLT_spec.shape == (4, 10)
    assert PRC_spec.shape == (4, 10)
    assert PRB_spec.shape == (4, 10)

    assert np.all(np.logical_not(np.isnan(PLT_spec)))
    assert np.all(np.logical_not(np.isnan(PRC_spec)))
    assert np.all(np.logical_not(np.isnan(PRB_spec)))

    assert np.all(np.logical_not(np.isinf(PLT_spec)))
    assert np.all(np.logical_not(np.isinf(PRC_spec)))
    assert np.all(np.logical_not(np.isinf(PRB_spec)))

    return example_power_loss, example_power_loss_no_optional


def test_power_loss_call(test_interpolate_power):
    """Test PowerLoss class __call__()."""
    example_power_loss, example_power_loss_no_optional = test_interpolate_power

    ADAS_file = ADASReader()

    element = "be"

    input_Ne = example_power_loss.Ne
    input_Te = example_power_loss.Te
    input_Nh = 1e-5 * input_Ne

    input_Nh = DataArray(
        data=input_Nh,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")
    CCD = ADAS_file.get_adf11("ccd", element, "89")
    try:
        example_frac_abundance = FractionalAbundance(
            SCD,
            ACD,
            CCD=CCD,
        )

        example_frac_abundance.interpolate_rates(Ne=input_Ne, Te=input_Te)
        example_frac_abundance.calc_ionisation_balance_matrix(Ne=input_Ne, Nh=input_Nh)
        F_z_tinf = np.real(example_frac_abundance.calc_F_z_tinf())
    except Exception as e:
        raise e

    test_case = Exception_Power_Loss_Test_Case(example_power_loss, F_z_t=F_z_tinf)

    partial_input_func_name = "partial_powerloss_inputs"

    input_error_check("Nh", None, ValueError, test_case, partial_input_func_name)

    test_case = Exception_Power_Loss_Test_Case(
        example_power_loss_no_optional, Nh=input_Nh, F_z_t=F_z_tinf
    )

    input_error_check("Nh", input_Nh, ValueError, test_case, partial_input_func_name)

    test_case = Exception_Power_Loss_Test_Case(
        example_power_loss, Nh=input_Nh, F_z_t=F_z_tinf
    )

    call_func_name = "PowerLoss" + example_power_loss.__call__.__name__

    # Thermal hydrogen density check

    input_Nh_invalid = copy.deepcopy(input_Nh.data)
    input_error_check("Nh", input_Nh_invalid, TypeError, test_case, call_func_name)

    input_Nh_invalid = np.inf + np.zeros(input_Nh_invalid.data.shape)
    input_Nh_invalid = DataArray(
        data=input_Nh_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case, call_func_name)

    input_Nh_invalid.data = -np.inf + np.zeros(input_Nh_invalid.data.shape)
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case, call_func_name)

    input_Nh_invalid.data = -1 + np.zeros(input_Nh_invalid.data.shape)
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case, call_func_name)

    input_Nh_invalid.data = np.nan + np.zeros(input_Nh_invalid.data.shape)
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case, call_func_name)

    input_Nh_invalid = input_Nh.copy(deep=True)
    input_Nh_invalid = input_Nh_invalid.expand_dims("blank")
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case, call_func_name)

    # Inputted fractional abundance checks

    F_z_t_invalid = [1.0, 0.0, 0.0, 0.0, 0.0]
    input_error_check("F_z_t", F_z_t_invalid, TypeError, test_case, call_func_name)

    F_z_t_invalid = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    input_error_check("F_z_t", F_z_t_invalid, TypeError, test_case, call_func_name)

    F_z_t_invalid = DataArray(
        data=-1.0 * np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, ValueError, test_case, call_func_name)

    F_z_t_invalid = DataArray(
        data=np.nan + np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, ValueError, test_case, call_func_name)

    F_z_t_invalid = DataArray(
        data=np.inf + np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, ValueError, test_case, call_func_name)

    F_z_t_invalid = DataArray(
        data=-np.inf + np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, ValueError, test_case, call_func_name)

    F_z_t_invalid = DataArray(
        data=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, ValueError, test_case, call_func_name)

    try:
        cooling_factor = example_power_loss_no_optional(
            Ne=input_Ne, Te=input_Te, F_z_t=F_z_tinf, full_run=False
        )
    except Exception as e:
        raise e

    assert cooling_factor.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(cooling_factor)))
    assert np.all(np.logical_not(np.isinf(cooling_factor)))

    try:
        cooling_factor = example_power_loss(
            Ne=input_Ne, Te=input_Te, Nh=input_Nh, F_z_t=F_z_tinf, full_run=False
        )
    except Exception as e:
        raise e

    assert cooling_factor.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(cooling_factor)))
    assert np.all(np.logical_not(np.isinf(cooling_factor)))
