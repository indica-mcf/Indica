import unittest

import numpy as np
import pytest
from xarray import DataArray

from indica.operators.fractional_abundance import FractionalAbundance
from indica.operators.fractional_abundance import PowerLoss
from indica.readers import ADASReader


class Assertion_Test_Case(unittest.TestCase):
    def init_fractional_abundance_assert_check(
        self, SCD, ACD, CCD, Ne, Nh, Te, N_z_t0=None
    ):
        with self.assertRaises(AssertionError):
            FractionalAbundance(SCD, ACD, CCD, Ne, Nh, Te, N_z_t0, True)

    def init_power_loss_assert_check(self, PLT, PRC, PRB, Ne, Nh, Te, N_z_t=None):
        with self.assertRaises(AssertionError):
            PowerLoss(PLT, PRC, PRB, Ne, Nh, Te, N_z_t, True)

    def tau_check(self, FracAbundObj: FractionalAbundance, tau):
        with self.assertRaises(AssertionError):
            FracAbundObj(tau)


@pytest.fixture
def test_fractional_abundance_init():
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
            CCD,
            Ne=input_Ne,
            Nh=input_Nh,
            Te=input_Te,
            unit_testing=True,
        )
    except Exception as e:
        raise e

    test_case = Assertion_Test_Case()

    N_z_t0_invalid = [1.0, 0.0, 0.0, 0.0, 0.0]
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t0=N_z_t0_invalid,
    )

    N_z_t0_invalid = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t0=N_z_t0_invalid,
    )

    N_z_t0_invalid = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t0=N_z_t0_invalid,
    )

    N_z_t0_invalid = np.zeros(5) + np.nan
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t0=N_z_t0_invalid,
    )

    N_z_t0_invalid = np.zeros(5) + np.inf
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t0=N_z_t0_invalid,
    )

    N_z_t0_invalid = np.zeros(5) - np.inf
    test_case.init_fractional_abundance_assert_check(
        SCD,
        ACD,
        CCD,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t0=N_z_t0_invalid,
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

    return example_frac_abundance


@pytest.fixture
def test_interpolate_rates(test_fractional_abundance_init):
    try:
        (
            SCD_spec,
            ACD_spec,
            CCD_spec,
            num_of_stages,
        ) = test_fractional_abundance_init.interpolate_rates()
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

    return test_fractional_abundance_init


@pytest.fixture
def test_calc_ionisation_balance_matrix(test_interpolate_rates):
    try:
        ionisation_balance_matrix = (
            test_interpolate_rates.calc_ionisation_balance_matrix()
        )
    except Exception as e:
        raise e

    assert ionisation_balance_matrix.shape == (5, 5, 10)

    assert np.all(np.logical_not(np.isnan(ionisation_balance_matrix)))

    assert np.all(np.logical_not(np.isinf(ionisation_balance_matrix)))

    return test_interpolate_rates


@pytest.fixture
def test_calc_N_z_tinf(test_calc_ionisation_balance_matrix):
    try:
        N_z_tinf = test_calc_ionisation_balance_matrix.calc_N_z_tinf()
    except Exception as e:
        raise e

    assert N_z_tinf.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(N_z_tinf)))

    assert np.all(np.logical_not(np.isinf(N_z_tinf)))

    rho = test_calc_ionisation_balance_matrix.Ne.coords["rho"]
    ionisation_balance_matrix = (
        test_calc_ionisation_balance_matrix.ionisation_balance_matrix
    )

    for irho in range(rho.size):
        test_null = np.dot(ionisation_balance_matrix[:, :, irho], N_z_tinf[:, irho])
        assert np.allclose(test_null, np.zeros(test_null.shape))

        test_normalization = np.linalg.norm(N_z_tinf[:, irho])
        assert np.allclose(test_normalization, 1.0)

    return test_calc_ionisation_balance_matrix


@pytest.fixture
def test_calc_eigen_vals_and_vecs(test_calc_N_z_tinf):
    try:
        eig_vals, eig_vecs = test_calc_N_z_tinf.calc_eigen_vals_and_vecs()
    except Exception as e:
        raise e

    assert eig_vals.shape == (5, 10)
    assert eig_vecs.shape == (5, 5, 10)

    assert np.all(np.logical_not(np.isnan(eig_vals)))
    assert np.all(np.logical_not(np.isinf(eig_vals)))

    assert np.all(np.logical_not(np.isnan(eig_vecs)))
    assert np.all(np.logical_not(np.isinf(eig_vecs)))

    rho = test_calc_N_z_tinf.Ne.coords["rho"]
    ionisation_balance_matrix = test_calc_N_z_tinf.ionisation_balance_matrix

    for irho in range(rho.size):
        for ieig in range(test_calc_N_z_tinf.num_of_stages):
            test_eigen = np.dot(
                ionisation_balance_matrix[:, :, irho], eig_vecs[:, ieig, irho]
            ) - np.dot(eig_vals[ieig, irho], eig_vecs[:, ieig, irho])

            assert np.allclose(test_eigen, np.zeros(test_eigen.shape))

    return test_calc_N_z_tinf


@pytest.fixture
def test_calc_eigen_coeffs(test_calc_eigen_vals_and_vecs):
    try:
        eig_coeffs, N_z_t0 = test_calc_eigen_vals_and_vecs.calc_eigen_coeffs()
    except Exception as e:
        raise e

    assert eig_coeffs.shape == (5, 10)
    assert N_z_t0.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(eig_coeffs)))
    assert np.all(np.logical_not(np.isinf(eig_coeffs)))

    assert np.all(np.logical_not(np.isnan(N_z_t0)))
    assert np.all(np.logical_not(np.isinf(N_z_t0)))

    return test_calc_eigen_vals_and_vecs


def test_fractional_abundance_call(test_calc_eigen_coeffs):
    test_case = Assertion_Test_Case()

    tau = -1
    test_case.tau_check(test_calc_eigen_coeffs, tau)

    tau = np.inf
    test_case.tau_check(test_calc_eigen_coeffs, tau)

    tau = -np.inf
    test_case.tau_check(test_calc_eigen_coeffs, tau)

    tau = 1e-16
    try:
        N_z_t = test_calc_eigen_coeffs(tau)
    except Exception as e:
        raise e

    assert N_z_t.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(N_z_t)))
    assert np.all(np.logical_not(np.isinf(N_z_t)))

    assert np.allclose(N_z_t, test_calc_eigen_coeffs.N_z_t0)

    tau = 1e2

    try:
        N_z_t = test_calc_eigen_coeffs(tau)
    except Exception as e:
        raise e

    assert np.all(np.logical_not(np.isnan(N_z_t)))
    assert np.all(np.logical_not(np.isinf(N_z_t)))

    assert np.allclose(N_z_t, test_calc_eigen_coeffs.N_z_tinf, atol=2e-2)

    rho = test_calc_eigen_coeffs.Ne.coords["rho"]

    for irho in range(rho.size):
        test_normalization = np.linalg.norm(N_z_t[:, irho])
        assert np.abs(test_normalization - 1.0) <= 2e-2

    return test_calc_eigen_coeffs


@pytest.fixture
def test_power_loss_init():
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
            PRC,
            PRB,
            Ne=input_Ne,
            Nh=input_Nh,
            Te=input_Te,
            unit_testing=True,
        )
    except Exception as e:
        raise e

    test_case = Assertion_Test_Case()

    N_z_t_invalid = [1.0, 0.0, 0.0, 0.0, 0.0]
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t=N_z_t_invalid,
    )

    N_z_t_invalid = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t=N_z_t_invalid,
    )

    N_z_t_invalid = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t=N_z_t_invalid,
    )

    N_z_t_invalid = np.zeros(5) + np.nan
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t=N_z_t_invalid,
    )

    N_z_t_invalid = np.zeros(5) + np.inf
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t=N_z_t_invalid,
    )

    N_z_t_invalid = np.zeros(5) - np.inf
    test_case.init_power_loss_assert_check(
        PLT,
        PRC,
        PRB,
        Ne=input_Ne,
        Nh=input_Nh,
        Te=input_Te,
        N_z_t=N_z_t_invalid,
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

    return example_power_loss


@pytest.fixture
def test_interpolate_power(test_power_loss_init):
    try:
        PLT_spec, PRC_spec, PRB_spec, _ = test_power_loss_init.interpolate_power()
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

    return test_power_loss_init


def test_power_loss_call(test_interpolate_power):
    try:
        cooling_factor = test_interpolate_power()
    except Exception as e:
        raise e

    assert cooling_factor.shape == (10,)

    assert np.all(np.logical_not(np.isnan(cooling_factor)))
    assert np.all(np.logical_not(np.isinf(cooling_factor)))
