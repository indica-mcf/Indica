import numpy as np
import pytest
from xarray import DataArray

from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader

ELEMENT = "be"
ADAS_FILE = ADASReader()
SCD = ADAS_FILE.get_adf11("scd", ELEMENT, "89")
ACD = ADAS_FILE.get_adf11("acd", ELEMENT, "89")
CCD = ADAS_FILE.get_adf11("ccd", ELEMENT, "89")


@pytest.fixture
def test_fractional_abundance_init():
    """Test initialisation of FractionalAbundance class."""

    try:
        example_frac_abundance = FractionalAbundance(
            SCD,
            ACD,
            CCD=CCD,
        )
    except Exception as e:
        raise e

    try:
        example_frac_abundance_no_optional = FractionalAbundance(
            SCD,
            ACD,
        )
    except Exception as e:
        raise e
    assert example_frac_abundance_no_optional.CCD is None

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

    example_frac_abundance.interpolate_rates(input_Ne, input_Te)

    example_frac_abundance_no_optional.interpolate_rates(input_Ne, input_Te)

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

    example_frac_abundance.calc_ionisation_balance_matrix(Ne=input_Ne)

    assert np.all(example_frac_abundance.Nh == 0)

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
        for ieig in range(example_frac_abundance_no_optional.num_of_ion_charge):
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
        for ieig in range(example_frac_abundance.num_of_ion_charge):
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


def test_frac_abund_full_run(test_fractional_abundance_init):
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

    tau = 1e-16

    try:
        F_z_t = example_frac_abundance_no_optional(
            Ne=input_Ne, Te=input_Te, tau=tau, full_run=True
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
            Ne=input_Ne, Te=input_Te, tau=tau, full_run=True
        )
    except Exception as e:
        raise e

    assert F_z_t.shape == (5, 10)

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
            input_Te,
            input_Ne,
            tau=tau,
            full_run=True,
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

    input_Nh = 1e-5 * input_Ne
    input_Nh = DataArray(
        data=input_Nh,
        coords={"rho_poloidal": np.linspace(0.0, 1.0, 10)},
        dims=["rho_poloidal"],
    )

    tau = 1e-16
    try:
        F_z_t = example_frac_abundance(
            input_Te,
            input_Ne,
            input_Nh,
            tau=tau,
            full_run=True,
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
            input_Te,
            input_Ne,
            input_Nh,
            tau=tau,
            full_run=True,
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
            input_Te,
            input_Ne,
            input_Nh,
            tau=tau,
            full_run=True,
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

    example_power_loss.calculate_power_loss(Ne=input_Ne, F_z_t=F_z_tinf)

    assert np.all(example_power_loss.Nh == 0)

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


def test_power_loss_full_run(test_power_loss_init):
    example_power_loss, example_power_loss_no_optional = test_power_loss_init

    ADAS_file = ADASReader()

    element = "be"

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

    try:
        cooling_factor = example_power_loss_no_optional(
            Ne=input_Ne, Te=input_Te, F_z_t=F_z_tinf, full_run=True
        )
    except Exception as e:
        raise e

    assert cooling_factor.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(cooling_factor)))
    assert np.all(np.logical_not(np.isinf(cooling_factor)))

    try:
        cooling_factor = example_power_loss(
            Ne=input_Ne, Te=input_Te, Nh=input_Nh, F_z_t=F_z_tinf, full_run=True
        )
    except Exception as e:
        raise e

    assert cooling_factor.shape == (5, 10)

    assert np.all(np.logical_not(np.isnan(cooling_factor)))
    assert np.all(np.logical_not(np.isinf(cooling_factor)))
