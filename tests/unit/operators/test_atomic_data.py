import unittest

import numpy as np
import pytest
from xarray import DataArray

from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader


class Exception_Frac_Abund_Test_Case(unittest.TestCase):
    """Test case for testing assertion and value errors in
    FractionalAbundance initializations and FractionalAbundance
    call (due to tau being a user input).
    """

    def __init__(
        self,
        SCD,
        ACD,
        CCD,
        Ne,
        Nh,
        Te,
    ):
        self.SCD = SCD
        self.ACD = ACD
        self.CCD = CCD
        self.Ne = Ne
        self.Nh = Nh
        self.Te = Te

        self.nominal_inputs = [self.SCD, self.ACD, self.CCD, self.Ne, self.Nh, self.Te]

    def init_assert_check(
        self, SCD=None, ACD=None, CCD=None, Ne=None, Nh=None, Te=None, F_z_t0=None
    ):
        inputs = [SCD, ACD, CCD, Ne, Nh, Te]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        SCD, ACD, CCD, Ne, Nh, Te = inputs

        """Test assert errors are raised for FractionalAbundance initialization."""
        with self.assertRaises(AssertionError):
            FractionalAbundance(SCD, ACD, Ne, Te, F_z_t0, Nh, CCD, True)

    def init_type_check(
        self, SCD=None, ACD=None, CCD=None, Ne=None, Nh=None, Te=None, F_z_t0=None
    ):
        inputs = [SCD, ACD, CCD, Ne, Nh, Te]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        SCD, ACD, CCD, Ne, Nh, Te = inputs

        """Test type errors are raised for FractionalAbundance initialization."""
        with self.assertRaises(TypeError):
            FractionalAbundance(SCD, ACD, Ne, Te, F_z_t0, Nh, CCD, True)

    def init_value_error_check(
        self, SCD=None, ACD=None, CCD=None, Ne=None, Nh=None, Te=None, F_z_t0=None
    ):
        inputs = [SCD, ACD, CCD, Ne, Nh, Te]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        SCD, ACD, CCD, Ne, Nh, Te = inputs

        """Test value errors are raised for FractionalAbundance initialization."""
        with self.assertRaises(ValueError):
            FractionalAbundance(SCD, ACD, Ne, Te, F_z_t0, Nh, CCD, True)

    def tau_check(self, FracAbundObj: FractionalAbundance, tau):
        """Test assert errors are raised for FractionalAbundance call
        (concerning user input tau).
        """
        with self.assertRaises(AssertionError):
            FracAbundObj(tau)


class Exception_Power_Loss_Test_Case(unittest.TestCase):
    """Test case for testing assertion and value errors in
    PowerLoss initializations.
    """

    def __init__(
        self,
        PLT,
        PRC,
        PRB,
        Ne,
        Nh,
        Te,
        F_z_t,
    ):
        self.PLT = PLT
        self.PRC = PRC
        self.PRB = PRB
        self.Ne = Ne
        self.Nh = Nh
        self.Te = Te
        self.F_z_t = F_z_t

        self.nominal_inputs = [
            self.PRC,
            self.PRC,
            self.PRB,
            self.Ne,
            self.Nh,
            self.Te,
            self.F_z_t,
        ]

    def init_assert_check(
        self, PLT=None, PRC=None, PRB=None, Ne=None, Te=None, Nh=None, F_z_t=None
    ):
        inputs = [PLT, PRC, PRB, Ne, Nh, Te, F_z_t]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        PLT, PRC, PRB, Ne, Nh, Te, F_z_t = inputs

        """Test assert errors are raised for PowerLoss initialization."""
        with self.assertRaises(AssertionError):
            PowerLoss(PLT, PRB, Ne, Te, F_z_t, PRC, True, Nh)

    def init_type_check(
        self, PLT=None, PRC=None, PRB=None, Ne=None, Te=None, Nh=None, F_z_t=None
    ):
        inputs = [PLT, PRC, PRB, Ne, Nh, Te, F_z_t]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        PLT, PRC, PRB, Ne, Nh, Te, F_z_t = inputs

        """Test assert errors are raised for PowerLoss initialization."""
        with self.assertRaises(TypeError):
            PowerLoss(PLT, PRB, Ne, Te, F_z_t, PRC, True, Nh)

    def init_value_error_check(
        self, PLT=None, PRC=None, PRB=None, Ne=None, Te=None, Nh=None, F_z_t=None
    ):
        inputs = [PLT, PRC, PRB, Ne, Nh, Te, F_z_t]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        PLT, PRC, PRB, Ne, Nh, Te, F_z_t = inputs

        """Test assert errors are raised for PowerLoss initialization."""
        with self.assertRaises(ValueError):
            PowerLoss(PLT, PRB, Ne, Te, F_z_t, PRC, True, Nh)


def input_error_check(invalid_input_name, invalid_input, error_check, test_case):
    """Helper function for test_case checks"""
    invalid_input_dict = {
        "SCD": {"SCD": invalid_input},
        "ACD": {"ACD": invalid_input},
        "CCD": {"CCD": invalid_input},
        "Ne": {"Ne": invalid_input},
        "Nh": {"Nh": invalid_input},
        "Te": {"Te": invalid_input},
        "F_z_t0": {"F_z_t0": invalid_input},
        "PLT": {"PLT": invalid_input},
        "PRC": {"PRC": invalid_input},
        "PRB": {"PRB": invalid_input},
        "F_z_t": {"F_z_t": invalid_input},
    }.get(invalid_input_name)

    {
        TypeError: test_case.init_type_check,
        AssertionError: test_case.init_assert_check,
        ValueError: test_case.init_value_error_check,
    }.get(error_check)(**invalid_input_dict)


@pytest.fixture
def test_fractional_abundance_init():
    """Test initialization of FractionalAbundance class."""
    ADAS_file = ADASReader()

    element = "be"

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")
    CCD = ADAS_file.get_adf11("ccd", element, "89")

    input_Ne = np.logspace(19.0, 16.0, 10)
    input_Nh = 1e-5 * input_Ne
    input_Te = np.logspace(4.6, 2, 10)

    input_Te = DataArray(
        data=input_Te,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_Ne = DataArray(
        data=input_Ne,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_Nh = DataArray(
        data=input_Nh,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    try:
        example_frac_abundance = FractionalAbundance(
            SCD,
            ACD,
            Ne=input_Ne,
            Te=input_Te,
            Nh=input_Nh,
            CCD=CCD,
            unit_testing=True,
        )
    except Exception as e:
        raise e

    # Test omission of optional inputs, CCD and Nh
    try:
        example_frac_abundance_no_optional = FractionalAbundance(
            SCD,
            ACD,
            Ne=input_Ne,
            Te=input_Te,
            unit_testing=True,
        )
    except Exception as e:
        raise e
    assert example_frac_abundance_no_optional.Nh is None

    test_case = Exception_Frac_Abund_Test_Case(
        SCD, ACD, CCD, input_Ne, input_Nh, input_Te
    )

    # SCD checks

    SCD_invalid = np.copy(SCD.data)
    input_error_check("SCD", SCD_invalid, TypeError, test_case)

    SCD_invalid = SCD.copy()
    SCD_invalid.data = -1 * np.copy(SCD_invalid.data)
    input_error_check("SCD", SCD_invalid, ValueError, test_case)

    SCD_invalid.data = np.nan * np.copy(SCD.data)
    input_error_check("SCD", SCD_invalid, ValueError, test_case)

    SCD_invalid.data = np.inf * np.copy(SCD.data)
    input_error_check("SCD", SCD_invalid, ValueError, test_case)

    SCD_invalid.data = -np.inf * np.copy(SCD.data)
    input_error_check("SCD", SCD_invalid, ValueError, test_case)

    SCD_invalid = SCD.copy()
    SCD_invalid = SCD_invalid.expand_dims("blank")
    input_error_check("SCD", SCD_invalid, AssertionError, test_case)

    # ACD checks

    ACD_invalid = np.copy(ACD.data)
    input_error_check("ACD", ACD_invalid, TypeError, test_case)

    ACD_invalid = ACD.copy()
    ACD_invalid.data = -1 * np.copy(ACD_invalid.data)
    input_error_check("ACD", ACD_invalid, ValueError, test_case)

    ACD_invalid.data = np.nan * np.copy(ACD.data)
    input_error_check("ACD", ACD_invalid, ValueError, test_case)

    ACD_invalid.data = np.inf * np.copy(ACD.data)
    input_error_check("ACD", ACD_invalid, ValueError, test_case)

    ACD_invalid.data = -np.inf * np.copy(ACD.data)
    input_error_check("ACD", ACD_invalid, ValueError, test_case)

    ACD_invalid = ACD.copy()
    ACD_invalid = ACD_invalid.expand_dims("blank")
    input_error_check("ACD", ACD_invalid, AssertionError, test_case)

    # CCD checks

    CCD_invalid = np.copy(CCD.data)
    input_error_check("CCD", CCD_invalid, TypeError, test_case)

    CCD_invalid = CCD.copy()
    CCD_invalid.data = -1 * np.copy(CCD_invalid.data)
    input_error_check("CCD", CCD_invalid, ValueError, test_case)

    CCD_invalid.data = np.nan * np.copy(CCD.data)
    input_error_check("CCD", CCD_invalid, ValueError, test_case)

    CCD_invalid.data = np.inf * np.copy(CCD.data)
    input_error_check("CCD", CCD_invalid, ValueError, test_case)

    CCD_invalid.data = -np.inf * np.copy(CCD.data)
    input_error_check("CCD", CCD_invalid, ValueError, test_case)

    CCD_invalid = CCD.copy()
    CCD_invalid = CCD_invalid.expand_dims("blank")
    input_error_check("CCD", CCD_invalid, AssertionError, test_case)

    # Initial fractional abundance checks

    F_z_t0_invalid = [1.0, 0.0, 0.0, 0.0, 0.0]
    input_error_check("F_z_t0", F_z_t0_invalid, TypeError, test_case)

    F_z_t0_invalid = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    input_error_check("F_z_t0", F_z_t0_invalid, TypeError, test_case)

    F_z_t0_invalid = DataArray(
        data=np.array([[-1.0, 0.0, 0.0, 0.0, 0.0]]).T,
        coords=[("ion_charges", np.linspace(0, 4, 5)), ("rho_poloidal", [0.5])],
        dims=["ion_charges", "rho_poloidal"],
    )
    input_error_check("F_z_t0", F_z_t0_invalid, ValueError, test_case)

    F_z_t0_invalid.data = np.array([np.zeros(5) + np.nan]).T
    input_error_check("F_z_t0", F_z_t0_invalid, ValueError, test_case)

    F_z_t0_invalid.data = np.array([np.zeros(5) + np.inf]).T
    input_error_check("F_z_t0", F_z_t0_invalid, ValueError, test_case)

    F_z_t0_invalid.data = np.array([np.zeros(5) - np.inf]).T
    input_error_check("F_z_t0", F_z_t0_invalid, ValueError, test_case)

    F_z_t0_invalid = DataArray(
        data=np.array([[[1.0, 0.0, 0.0, 0.0, 0.0]]]).T,
        coords=[
            ("ion_charges", np.linspace(0, 4, 5)),
            ("rho_poloidal", [0.5]),
            ("t", [77.5]),
        ],
        dims=["ion_charges", "rho_poloidal", "t"],
    )
    input_error_check("F_z_t0", F_z_t0_invalid, AssertionError, test_case)

    # Electron density checks

    input_Ne_invalid = np.copy(input_Ne.data)
    input_error_check("Ne", input_Ne_invalid, TypeError, test_case)

    input_Ne_invalid = np.logspace(30.0, 16.0, 10)
    input_Ne_invalid = DataArray(
        data=input_Ne_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check("Ne", input_Ne_invalid, AssertionError, test_case)

    input_Ne_invalid.data = np.logspace(19.0, 5.0, 10)
    input_error_check("Ne", input_Ne_invalid, AssertionError, test_case)

    input_Ne_invalid.data = np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case)

    input_Ne_invalid.data = -np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case)

    input_Ne_invalid.data = -1 + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case)

    input_Ne_invalid.data = np.nan + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case)

    input_Ne_invalid = input_Ne.copy()
    input_Ne_invalid = input_Ne_invalid.expand_dims("blank")
    input_error_check("Ne", input_Ne_invalid, AssertionError, test_case)

    # Thermal hydrogen density check

    input_Nh_invalid = np.copy(input_Nh.data)
    input_error_check("Nh", input_Nh_invalid, TypeError, test_case)

    input_Nh_invalid = np.inf + np.zeros(input_Nh_invalid.data.shape)
    input_Nh_invalid = DataArray(
        data=input_Nh_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case)

    input_Nh_invalid.data = -np.inf + np.zeros(input_Nh_invalid.data.shape)
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case)

    input_Nh_invalid.data = -1 + np.zeros(input_Nh_invalid.data.shape)
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case)

    input_Nh_invalid.data = np.nan + np.zeros(input_Nh_invalid.data.shape)
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case)

    input_Nh_invalid = input_Nh.copy()
    input_Nh_invalid = input_Nh_invalid.expand_dims("blank")
    input_error_check("Nh", input_Nh_invalid, AssertionError, test_case)

    # Electron temperature check

    input_Te_invalid = np.copy(input_Te.data)
    input_error_check("Te", input_Te_invalid, TypeError, test_case)

    input_Te_invalid = np.logspace(5, 2, 10)
    input_Te_invalid = DataArray(
        data=input_Te_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check("Te", input_Te_invalid, AssertionError, test_case)

    input_Te_invalid.data = np.logspace(4.6, -1, 10)
    input_error_check("Te", input_Te_invalid, AssertionError, test_case)

    input_Te_invalid.data = np.inf + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case)

    input_Te_invalid.data = -np.inf + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case)

    input_Te_invalid.data = -1 + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case)

    input_Te_invalid.data = np.nan + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case)

    input_Te_invalid = input_Te.copy()
    input_Te_invalid = input_Te_invalid.expand_dims("blank")
    input_error_check("Te", input_Te_invalid, AssertionError, test_case)

    return example_frac_abundance, example_frac_abundance_no_optional


@pytest.fixture
def test_interpolate_rates(test_fractional_abundance_init):
    """Test interpolate_rates() function in FractionalAbundance class."""
    (
        example_frac_abundance,
        example_frac_abundance_no_optional,
    ) = test_fractional_abundance_init

    try:
        (
            SCD_spec,
            ACD_spec,
            CCD_spec,
            num_of_ion_charges,
        ) = example_frac_abundance_no_optional.interpolate_rates()
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
        ) = example_frac_abundance.interpolate_rates()
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

    try:
        ionisation_balance_matrix = (
            example_frac_abundance_no_optional.calc_ionisation_balance_matrix()
        )
    except Exception as e:
        raise e

    assert ionisation_balance_matrix.shape == (5, 5, 10)

    assert np.all(np.logical_not(np.isnan(ionisation_balance_matrix)))

    assert np.all(np.logical_not(np.isinf(ionisation_balance_matrix)))

    try:
        ionisation_balance_matrix = (
            example_frac_abundance.calc_ionisation_balance_matrix()
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

    try:
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


def test_fractional_abundance_call(test_calc_eigen_coeffs):
    """Test FractionalAbundance class __call__()."""
    example_frac_abundance, example_frac_abundance_no_optional = test_calc_eigen_coeffs

    test_case = Exception_Frac_Abund_Test_Case(
        example_frac_abundance_no_optional.SCD,
        example_frac_abundance_no_optional.ACD,
        example_frac_abundance_no_optional.CCD,
        example_frac_abundance_no_optional.Ne,
        example_frac_abundance_no_optional.Nh,
        example_frac_abundance_no_optional.Te,
    )

    tau = -1
    test_case.tau_check(example_frac_abundance_no_optional, tau)

    tau = np.inf
    test_case.tau_check(example_frac_abundance_no_optional, tau)

    tau = -np.inf
    test_case.tau_check(example_frac_abundance_no_optional, tau)

    tau = 1e-16
    try:
        F_z_t = example_frac_abundance_no_optional(tau)
    except Exception as e:
        raise e

    assert F_z_t.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(F_z_t, example_frac_abundance_no_optional.F_z_t0, atol=1e-4)

    tau = 1e2

    try:
        F_z_t = example_frac_abundance_no_optional(tau)
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
        F_z_t = example_frac_abundance_no_optional(tau)
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

    test_case = Exception_Frac_Abund_Test_Case(
        example_frac_abundance.SCD,
        example_frac_abundance.ACD,
        example_frac_abundance.CCD,
        example_frac_abundance.Ne,
        example_frac_abundance.Nh,
        example_frac_abundance.Te,
    )

    tau = -1
    test_case.tau_check(example_frac_abundance, tau)

    tau = np.inf
    test_case.tau_check(example_frac_abundance, tau)

    tau = -np.inf
    test_case.tau_check(example_frac_abundance, tau)

    tau = 1e-16
    try:
        F_z_t = example_frac_abundance(tau)
    except Exception as e:
        raise e

    assert F_z_t.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(F_z_t, example_frac_abundance.F_z_t0, atol=1e-5)

    tau = 1e2

    try:
        F_z_t = example_frac_abundance(tau)
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
        F_z_t = example_frac_abundance(tau)
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
    """Test initialization of PowerLoss class."""
    ADAS_file = ADASReader()

    element = "be"

    PLT = ADAS_file.get_adf11("plt", element, "89")
    PRC = ADAS_file.get_adf11("prc", element, "89")
    PRB = ADAS_file.get_adf11("prb", element, "89")

    input_Ne = np.logspace(19.0, 16.0, 10)
    input_Nh = 1e-5 * input_Ne
    input_Te = np.logspace(4.6, 2, 10)

    input_Te = DataArray(
        data=input_Te,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_Ne = DataArray(
        data=input_Ne,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
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
            Ne=input_Ne,
            Te=input_Te,
            Nh=input_Nh,
            CCD=CCD,
            unit_testing=False,
        )
    except Exception as e:
        raise e

    try:
        example_power_loss = PowerLoss(
            PLT,
            PRB,
            F_z_t=np.real(example_frac_abundance.F_z_tinf),
            Ne=input_Ne,
            Nh=input_Nh,
            Te=input_Te,
            PRC=PRC,
            unit_testing=True,
        )
    except Exception as e:
        raise e

    # Test omission of optional inputs, PRC and Nh
    try:
        example_power_loss_no_optional = PowerLoss(
            PLT,
            PRB,
            F_z_t=np.real(example_frac_abundance.F_z_tinf),
            Ne=input_Ne,
            Te=input_Te,
            unit_testing=True,
        )
    except Exception as e:
        raise e

    test_case = Exception_Power_Loss_Test_Case(
        PLT,
        PRC,
        PRB,
        input_Ne,
        input_Nh,
        input_Te,
        F_z_t=np.real(example_frac_abundance.F_z_tinf),
    )

    # PLT checks

    PLT_invalid = np.copy(PLT.data)
    input_error_check("PLT", PLT_invalid, TypeError, test_case)

    PLT_invalid = PLT.copy()
    PLT_invalid.data = -1 * np.copy(PLT_invalid.data)
    input_error_check("PLT", PLT_invalid, ValueError, test_case)

    PLT_invalid.data = np.nan * np.copy(PLT_invalid)
    input_error_check("PLT", PLT_invalid, ValueError, test_case)

    PLT_invalid.data = np.inf * np.copy(PLT_invalid)
    input_error_check("PLT", PLT_invalid, ValueError, test_case)

    PLT_invalid.data = -np.inf * np.copy(PLT_invalid)
    input_error_check("PLT", PLT_invalid, ValueError, test_case)

    PLT_invalid = PLT.copy()
    PLT_invalid = PLT_invalid.expand_dims("blank")
    input_error_check("PLT", PLT_invalid, AssertionError, test_case)

    # PRC checks

    PRC_invalid = np.copy(PRC.data)
    input_error_check("PRC", PRC_invalid, TypeError, test_case)

    PRC_invalid = PRC.copy()
    PRC_invalid.data = -1 * np.copy(PRC_invalid.data)
    input_error_check("PRC", PRC_invalid, ValueError, test_case)

    PRC_invalid.data = np.nan * np.copy(PRC_invalid)
    input_error_check("PRC", PRC_invalid, ValueError, test_case)

    PRC_invalid.data = np.inf * np.copy(PRC_invalid)
    input_error_check("PRC", PRC_invalid, ValueError, test_case)

    PRC_invalid.data = -np.inf * np.copy(PRC_invalid)
    input_error_check("PRC", PRC_invalid, ValueError, test_case)

    PRC_invalid = PRC.copy()
    PRC_invalid = PRC_invalid.expand_dims("blank")
    input_error_check("PRC", PRC_invalid, AssertionError, test_case)

    # PRB checks

    PRB_invalid = np.copy(PRB.data)
    input_error_check("PRB", PRB_invalid, TypeError, test_case)

    PRB_invalid = PRB.copy()
    PRB_invalid.data = -1 * np.copy(PRB_invalid.data)
    input_error_check("PRB", PRB_invalid, ValueError, test_case)

    PRB_invalid.data = np.nan * np.copy(PRB_invalid)
    input_error_check("PRB", PRB_invalid, ValueError, test_case)

    PRB_invalid.data = np.inf * np.copy(PRB_invalid)
    input_error_check("PRB", PRB_invalid, ValueError, test_case)

    PRB_invalid.data = -np.inf * np.copy(PRB_invalid)
    input_error_check("PRB", PRB_invalid, ValueError, test_case)

    PRB_invalid = PRB.copy()
    PRB_invalid = PRB_invalid.expand_dims("blank")
    input_error_check("PRB", PRB_invalid, AssertionError, test_case)

    # Inputted fractional abundance checks

    F_z_t_invalid = [1.0, 0.0, 0.0, 0.0, 0.0]
    input_error_check("F_z_t", F_z_t_invalid, TypeError, test_case)

    F_z_t_invalid = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    input_error_check("F_z_t", F_z_t_invalid, TypeError, test_case)

    F_z_t_invalid = DataArray(
        data=-1.0 * np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, ValueError, test_case)

    F_z_t_invalid = DataArray(
        data=np.nan + np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, ValueError, test_case)

    F_z_t_invalid = DataArray(
        data=np.inf + np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, ValueError, test_case)

    F_z_t_invalid = DataArray(
        data=-np.inf + np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, ValueError, test_case)

    F_z_t_invalid = DataArray(
        data=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        coords={"ion_charges": np.linspace(0, 4, 5, dtype=int)},
        dims=["ion_charges"],
    )
    input_error_check("F_z_t", F_z_t_invalid, AssertionError, test_case)

    # Electron density checks

    input_Ne_invalid = np.copy(input_Ne.data)
    input_error_check("Ne", input_Ne_invalid, TypeError, test_case)

    input_Ne_invalid = np.logspace(30.0, 16.0, 10)
    input_Ne_invalid = DataArray(
        data=input_Ne_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check("Ne", input_Ne_invalid, AssertionError, test_case)

    input_Ne_invalid.data = np.logspace(19.0, 5.0, 10)
    input_error_check("Ne", input_Ne_invalid, AssertionError, test_case)

    input_Ne_invalid.data = np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case)

    input_Ne_invalid.data = -np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case)

    input_Ne_invalid.data = -1 + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case)

    input_Ne_invalid.data = np.nan + np.zeros(input_Ne_invalid.data.shape)
    input_error_check("Ne", input_Ne_invalid, ValueError, test_case)

    input_Ne_invalid = input_Ne.copy()
    input_Ne_invalid = input_Ne_invalid.expand_dims("blank")
    input_error_check("Ne", input_Ne_invalid, AssertionError, test_case)

    # Thermal hydrogen density check

    input_Nh_invalid = np.copy(input_Nh.data)
    input_error_check("Nh", input_Nh_invalid, TypeError, test_case)

    input_Nh_invalid = np.inf + np.zeros(input_Nh_invalid.data.shape)
    input_Nh_invalid = DataArray(
        data=input_Nh_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case)

    input_Nh_invalid.data = -np.inf + np.zeros(input_Nh_invalid.data.shape)
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case)

    input_Nh_invalid.data = -1 + np.zeros(input_Nh_invalid.data.shape)
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case)

    input_Nh_invalid.data = np.nan + np.zeros(input_Nh_invalid.data.shape)
    input_error_check("Nh", input_Nh_invalid, ValueError, test_case)

    input_Nh_invalid = input_Nh.copy()
    input_Nh_invalid = input_Nh_invalid.expand_dims("blank")
    input_error_check("Nh", input_Nh_invalid, AssertionError, test_case)

    # Electron temperature check

    input_Te_invalid = np.copy(input_Te.data)
    input_error_check("Te", input_Te_invalid, TypeError, test_case)

    input_Te_invalid = np.logspace(5, 2, 10)
    input_Te_invalid = DataArray(
        data=input_Te_invalid,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )
    input_error_check("Te", input_Te_invalid, AssertionError, test_case)

    input_Te_invalid.data = np.logspace(4.6, -1, 10)
    input_error_check("Te", input_Te_invalid, AssertionError, test_case)

    input_Te_invalid.data = np.inf + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case)

    input_Te_invalid.data = -np.inf + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case)

    input_Te_invalid.data = -1 + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case)

    input_Te_invalid.data = np.nan + np.zeros(input_Te_invalid.data.shape)
    input_error_check("Te", input_Te_invalid, ValueError, test_case)

    input_Te_invalid = input_Te.copy()
    input_Te_invalid = input_Te_invalid.expand_dims("blank")
    input_error_check("Te", input_Te_invalid, AssertionError, test_case)

    return example_power_loss, example_power_loss_no_optional


@pytest.fixture
def test_interpolate_power(test_power_loss_init):
    """Test interpolate_power() function in PowerLoss class."""
    example_power_loss, example_power_loss_no_optional = test_power_loss_init

    try:
        (
            PLT_spec,
            PRC_spec,
            PRB_spec,
            _,
        ) = example_power_loss_no_optional.interpolate_power()
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
        PLT_spec, PRC_spec, PRB_spec, _ = example_power_loss.interpolate_power()
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

    try:
        cooling_factor = example_power_loss_no_optional()
    except Exception as e:
        raise e

    assert cooling_factor.shape == (10,)

    assert np.all(np.logical_not(np.isnan(cooling_factor)))
    assert np.all(np.logical_not(np.isinf(cooling_factor)))

    try:
        cooling_factor = example_power_loss()
    except Exception as e:
        raise e

    assert cooling_factor.shape == (10,)

    assert np.all(np.logical_not(np.isnan(cooling_factor)))
    assert np.all(np.logical_not(np.isinf(cooling_factor)))
