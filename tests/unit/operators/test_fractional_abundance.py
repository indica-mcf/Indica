import unittest

import numpy as np
import pytest
from xarray import DataArray

from indica.operators.fractional_abundance import FractionalAbundance
from indica.operators.fractional_abundance import PowerLoss
from indica.readers import ADASReader


class Assertion_Test_Case(unittest.TestCase):
    """Test case for testing assertion errors in FractionalAbundance and PowerLoss
    initializations and FractionalAbundance call (due to tau being a user input).
    """

    def init_fractional_abundance_assert_check(
        self, SCD, ACD, CCD, Ne, Nh, Te, F_z_t0=None
    ):
        """Test assert errors are raised for FractionalAbundance initialization."""
        with self.assertRaises(AssertionError):
            FractionalAbundance(SCD, ACD, Ne, Te, F_z_t0, Nh, CCD, True)

    def init_power_loss_assert_check(self, PLT, PRC, PRB, Ne, Nh, Te, F_z_t=None):
        """Test assert errors are raised for PowerLoss initialization."""
        with self.assertRaises(AssertionError):
            PowerLoss(PLT, PRB, Ne, Nh, Te, PRC, F_z_t, True)

    def tau_check(self, FracAbundObj: FractionalAbundance, tau):
        """Test assert errors are raised for FractionalAbundance call
        (concerning user input tau).
        """
        with self.assertRaises(AssertionError):
            FracAbundObj(tau)


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
        data=input_Te, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )
    input_Ne = DataArray(
        data=input_Ne, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )
    input_Nh = DataArray(
        data=input_Nh, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
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

    test_case = Assertion_Test_Case()

    F_z_t0_invalid = [1.0, 0.0, 0.0, 0.0, 0.0]
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t0=F_z_t0_invalid,
    )

    F_z_t0_invalid = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t0=F_z_t0_invalid,
    )

    F_z_t0_invalid = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t0=F_z_t0_invalid,
    )

    F_z_t0_invalid = np.zeros(5) + np.nan
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t0=F_z_t0_invalid,
    )

    F_z_t0_invalid = np.zeros(5) + np.inf
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t0=F_z_t0_invalid,
    )

    F_z_t0_invalid = np.zeros(5) - np.inf
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t0=F_z_t0_invalid,
    )

    input_Ne_invalid = np.logspace(30.0, 16.0, 10)
    input_Ne_invalid = DataArray(
        data=input_Ne_invalid, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )

    test_case.init_fractional_abundance_assert_check(
        SCD, ACD, CCD, Ne=input_Ne_invalid, Nh=input_Nh, Te=input_Te
    )

    input_Ne_invalid.data = np.logspace(19.0, 5.0, 10)
    test_case.init_fractional_abundance_assert_check(
        SCD, ACD, CCD, Ne=input_Ne_invalid, Nh=input_Nh, Te=input_Te
    )

    input_Ne_invalid.data = np.logspace(19.0, 16.0, 10)
    input_Nh_invalid = np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_Nh_invalid = DataArray(
        data=input_Nh_invalid, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )

    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne_invalid,
        Nh=input_Nh_invalid,
        Te=input_Te,
    )

    input_Nh_invalid.data = -np.inf + np.zeros(input_Ne_invalid.data.shape)
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne_invalid,
        Nh=input_Nh_invalid,
        Te=input_Te,
    )

    input_Nh_invalid.data = -1 + np.zeros(input_Ne_invalid.data.shape)
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne_invalid,
        Nh=input_Nh_invalid,
        Te=input_Te,
    )

    input_Nh_invalid.data = np.nan + np.zeros(input_Ne_invalid.data.shape)
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne_invalid,
        Nh=input_Nh_invalid,
        Te=input_Te,
    )

    input_Nh_invalid.data = 1e-5 * np.copy(input_Ne)

    input_Te_invalid = np.logspace(5, 2, 10)
    input_Te_invalid = DataArray(
        data=input_Te_invalid, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )

    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne_invalid,
        Nh=input_Nh,
        Te=input_Te_invalid,
    )

    input_Te_invalid.data = np.logspace(4.6, -1, 10)
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne_invalid,
        Nh=input_Nh,
        Te=input_Te_invalid,
    )

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
            num_of_stages,
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
            num_of_stages,
        ) = example_frac_abundance.interpolate_rates()
    except Exception as e:
        raise e

    assert num_of_stages == 5

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

    rho = example_frac_abundance_no_optional.Ne.coords["rho"]
    ionisation_balance_matrix = (
        example_frac_abundance_no_optional.ionisation_balance_matrix
    )

    for irho in range(rho.size):
        test_null = np.dot(ionisation_balance_matrix[:, :, irho], F_z_tinf[:, irho])
        assert np.allclose(test_null, np.zeros(test_null.shape))

        test_normalization = np.linalg.norm(F_z_tinf[:, irho])
        assert np.allclose(test_normalization, 1.0)

    try:
        F_z_tinf = example_frac_abundance.calc_F_z_tinf()
    except Exception as e:
        raise e

    assert F_z_tinf.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(F_z_tinf)))

    assert np.all(np.logical_not(np.isinf(F_z_tinf)))

    rho = example_frac_abundance.Ne.coords["rho"]
    ionisation_balance_matrix = example_frac_abundance.ionisation_balance_matrix

    for irho in range(rho.size):
        test_null = np.dot(ionisation_balance_matrix[:, :, irho], F_z_tinf[:, irho])
        assert np.allclose(test_null, np.zeros(test_null.shape))

        test_normalization = np.linalg.norm(F_z_tinf[:, irho])
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

    rho = example_frac_abundance_no_optional.Ne.coords["rho"]
    ionisation_balance_matrix = (
        example_frac_abundance_no_optional.ionisation_balance_matrix
    )

    for irho in range(rho.size):
        for ieig in range(example_frac_abundance_no_optional.num_of_stages):
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

    rho = example_frac_abundance.Ne.coords["rho"]
    ionisation_balance_matrix = example_frac_abundance.ionisation_balance_matrix

    for irho in range(rho.size):
        for ieig in range(example_frac_abundance.num_of_stages):
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

    test_case = Assertion_Test_Case()

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

    assert np.allclose(F_z_t, example_frac_abundance_no_optional.F_z_t0)

    tau = 1e2

    try:
        F_z_t = example_frac_abundance_no_optional(tau)
    except Exception as e:
        raise e

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(F_z_t, example_frac_abundance_no_optional.F_z_tinf, atol=2e-2)

    rho = example_frac_abundance_no_optional.Ne.coords["rho"]

    for irho in range(rho.size):
        test_normalization = np.linalg.norm(F_z_t[:, irho])
        assert np.abs(test_normalization - 1.0) <= 2e-2

    test_case = Assertion_Test_Case()

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

    assert np.allclose(F_z_t, example_frac_abundance.F_z_t0)

    tau = 1e2

    try:
        F_z_t = example_frac_abundance(tau)
    except Exception as e:
        raise e

    assert np.all(np.logical_not(np.isnan(F_z_t)))
    assert np.all(np.logical_not(np.isinf(F_z_t)))

    assert np.allclose(F_z_t, example_frac_abundance.F_z_tinf, atol=2e-2)

    rho = example_frac_abundance.Ne.coords["rho"]

    for irho in range(rho.size):
        test_normalization = np.linalg.norm(F_z_t[:, irho])
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
        data=input_Te, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )
    input_Ne = DataArray(
        data=input_Ne, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )
    input_Nh = DataArray(
        data=input_Nh, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )

    try:
        example_power_loss = PowerLoss(
            PLT,
            PRB,
            Ne=input_Ne,
            Nh=input_Nh,
            Te=input_Te,
            PRC=PRC,
            unit_testing=True,
        )
    except Exception as e:
        raise e

    # Test omission of optional input, PRC
    try:
        example_power_loss_no_PRC = PowerLoss(
            PLT,
            PRB,
            Ne=input_Ne,
            Nh=input_Nh,
            Te=input_Te,
            unit_testing=True,
        )
    except Exception as e:
        raise e

    test_case = Assertion_Test_Case()

    F_z_t_invalid = [1.0, 0.0, 0.0, 0.0, 0.0]
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t=F_z_t_invalid,
    )

    F_z_t_invalid = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t=F_z_t_invalid,
    )

    F_z_t_invalid = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t=F_z_t_invalid,
    )

    F_z_t_invalid = np.zeros(5) + np.nan
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t=F_z_t_invalid,
    )

    F_z_t_invalid = np.zeros(5) + np.inf
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t=F_z_t_invalid,
    )

    F_z_t_invalid = np.zeros(5) - np.inf
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        F_z_t=F_z_t_invalid,
    )

    input_Ne_invalid = np.logspace(30.0, 16.0, 10)
    input_Ne_invalid = DataArray(
        data=input_Ne_invalid, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )

    test_case.init_power_loss_assert_check(
        PLT, PRC, PRB, Ne=input_Ne_invalid, Nh=input_Nh, Te=input_Te
    )

    input_Ne_invalid.data = np.logspace(19.0, 5.0, 10)
    test_case.init_power_loss_assert_check(
        PLT, PRC, PRB, Ne=input_Ne_invalid, Nh=input_Nh, Te=input_Te
    )

    input_Ne_invalid.data = np.logspace(19.0, 16.0, 10)
    input_Nh_invalid = np.inf + np.zeros(input_Ne_invalid.data.shape)
    input_Nh_invalid = DataArray(
        data=input_Nh_invalid, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )

    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne_invalid,
        Nh=input_Nh_invalid,
        Te=input_Te,
    )

    input_Nh_invalid.data = -np.inf + np.zeros(input_Ne_invalid.data.shape)
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne_invalid,
        Nh=input_Nh_invalid,
        Te=input_Te,
    )

    input_Nh_invalid.data = -1 + np.zeros(input_Ne_invalid.data.shape)
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne_invalid,
        Nh=input_Nh_invalid,
        Te=input_Te,
    )

    input_Nh_invalid.data = np.nan + np.zeros(input_Ne_invalid.data.shape)
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne_invalid,
        Nh=input_Nh_invalid,
        Te=input_Te,
    )

    input_Nh_invalid.data = 1e-5 * np.copy(input_Ne)

    input_Te_invalid = np.logspace(5, 2, 10)
    input_Te_invalid = DataArray(
        data=input_Te_invalid, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )

    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne_invalid,
        Nh=input_Nh,
        Te=input_Te_invalid,
    )

    input_Te_invalid.data = np.logspace(4.6, -1, 10)
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne_invalid,
        Nh=input_Nh,
        Te=input_Te_invalid,
    )

    return example_power_loss, example_power_loss_no_PRC


@pytest.fixture
def test_interpolate_power(test_power_loss_init):
    """Test interpolate_power() function in PowerLoss class."""
    example_power_loss, example_power_loss_no_PRC = test_power_loss_init

    try:
        PLT_spec, PRC_spec, PRB_spec, _ = example_power_loss_no_PRC.interpolate_power()
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

    return example_power_loss, example_power_loss_no_PRC


def test_power_loss_call(test_interpolate_power):
    """Test PowerLoss class __call__()."""
    example_power_loss, example_power_loss_no_PRC = test_interpolate_power

    try:
        cooling_factor = example_power_loss_no_PRC()
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
