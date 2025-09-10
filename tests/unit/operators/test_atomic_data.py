import numpy as np
from xarray import DataArray

from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader
from indica.utilities import get_element_info

ELEMENT = "be"
Z, A, NAME, SYMBOL = get_element_info(ELEMENT)

ADAS_FILE = ADASReader()
SCD = ADAS_FILE.get_adf11("scd", ELEMENT, "89")
ACD = ADAS_FILE.get_adf11("acd", ELEMENT, "89")
CCD = ADAS_FILE.get_adf11("ccd", ELEMENT, "89")

PLT = ADAS_FILE.get_adf11("plt", ELEMENT, "89")
PRC = ADAS_FILE.get_adf11("prc", ELEMENT, "89")
PRB = ADAS_FILE.get_adf11("prb", ELEMENT, "89")

RHOP = np.linspace(0.0, 1.0, 10)
INPUT_NE = DataArray(data=np.logspace(19.0, 16.0, 10), coords={"rhop": RHOP})
INPUT_NH = DataArray(np.logspace(14.0, 16.0, 10), coords={"rhop": RHOP})
INPUT_TE = DataArray(data=np.logspace(4.6, 2, 10), coords={"rhop": RHOP})
INPUT_TAU = DataArray(data=np.logspace(0, -3, 10), coords={"rhop": RHOP})


def fractional_abundance_init():
    """Test initialisation of FractionalAbundance class."""

    fract_abu = FractionalAbundance(
        SCD,
        ACD,
        ccd=CCD,
        full_run=False,
    )
    fract_abu_no_optional = FractionalAbundance(
        SCD,
        ACD,
        full_run=False,
    )
    fract_abu_full_run = FractionalAbundance(
        SCD,
        ACD,
        ccd=CCD,
        full_run=True,
    )

    return (
        fract_abu,
        fract_abu_no_optional,
        fract_abu_full_run,
    )


def power_loss_init():
    """Test initialisation of FractionalAbundance class."""

    power_loss = PowerLoss(
        PLT,
        PRB,
        prc=PRC,
        full_run=False,
    )
    power_loss_no_optional = PowerLoss(
        PLT,
        PRB,
        full_run=False,
    )
    power_loss_full_run = PowerLoss(
        PLT,
        PRB,
        prc=PRC,
        full_run=True,
    )

    return (
        power_loss,
        power_loss_no_optional,
        power_loss_full_run,
    )


class TestFractionalAbundance:
    def setup_class(self):
        (
            fract_abu,
            fract_abu_no_optional,
            fract_abu_full_run,
        ) = fractional_abundance_init()
        self.fract_abu = fract_abu
        self.fract_abu_no_optional = fract_abu_no_optional
        self.fract_abu_full_run = fract_abu_full_run
        self.rhop = RHOP
        self.input_Ne = INPUT_NE
        self.input_Te = INPUT_TE
        self.input_Nh = INPUT_NH
        self.input_tau = INPUT_TAU

    def test_interpolate_rates(self):
        """Test interpolate_rates() function in FractionalAbundance class."""
        try:
            self.fract_abu.interpolate_rates(self.input_Ne, self.input_Te)
        except Exception as e:
            raise e

    def test_interpolate_rates_no_optional(self):
        """Test interpolate_rates() function in FractionalAbundance class."""
        try:
            self.fract_abu_no_optional.interpolate_rates(self.input_Ne, self.input_Te)
        except Exception as e:
            raise e

    def test_calc_ionisation_balance_matrix(self):
        """Test calc_ionisation_balance_matrix() in FractionalAbundance class."""
        try:
            self.fract_abu.interpolate_rates(self.input_Ne, self.input_Te)
            ionisation_balance_matrix = self.fract_abu.calc_ionisation_balance_matrix(
                self.input_Ne, self.input_Nh
            )
        except Exception as e:
            raise e

        assert ionisation_balance_matrix.shape == (
            Z + 1,
            Z + 1,
            self.fract_abu.coord.size,
        )
        assert np.all(np.logical_not(np.isnan(ionisation_balance_matrix)))
        assert np.all(np.logical_not(np.isinf(ionisation_balance_matrix)))

    def test_calc_ionisation_balance_matrix_no_optional(self):
        """Test calc_ionisation_balance_matrix() in FractionalAbundance class."""
        try:
            self.fract_abu_no_optional.interpolate_rates(self.input_Ne, self.input_Te)
            ionisation_balance_matrix = (
                self.fract_abu_no_optional.calc_ionisation_balance_matrix(self.input_Ne)
            )
        except Exception as e:
            raise e

        assert ionisation_balance_matrix.shape == (
            Z + 1,
            Z + 1,
            self.fract_abu_no_optional.coord.size,
        )
        assert np.all(np.logical_not(np.isnan(ionisation_balance_matrix)))
        assert np.all(np.logical_not(np.isinf(ionisation_balance_matrix)))

    def test_calc_F_z_tinf(self):
        """Test calc_F_z_tinf function in in FractionalAbundance class."""
        try:
            self.fract_abu.interpolate_rates(self.input_Ne, self.input_Te)
            ionisation_balance_matrix = self.fract_abu.calc_ionisation_balance_matrix(
                self.input_Ne, self.input_Nh
            )
            F_z_tinf = self.fract_abu.calc_F_z_tinf()
        except Exception as e:
            raise e

        assert F_z_tinf.shape == (Z + 1, self.fract_abu_no_optional.coord.size)
        assert np.all(np.logical_not(np.isnan(F_z_tinf)))
        assert np.all(np.logical_not(np.isinf(F_z_tinf)))

        for icoord in range(self.fract_abu.coord.size):
            test_null = np.dot(
                ionisation_balance_matrix[:, :, icoord], F_z_tinf[:, icoord]
            )
            assert np.allclose(test_null, np.zeros(test_null.shape))

            test_normalization = np.sum(F_z_tinf[:, icoord])
            assert np.allclose(test_normalization, 1.0, rtol=1e-2)

        assert F_z_tinf.shape == (Z + 1, self.fract_abu.coord.size)
        assert np.all(np.logical_not(np.isnan(F_z_tinf)))
        assert np.all(np.logical_not(np.isinf(F_z_tinf)))

    def test_calc_eigen_vals_and_vecs(self):
        """Test calc_eigen_vals_and_vecs() function in FractionalAbundance class."""
        try:
            self.fract_abu.interpolate_rates(self.input_Ne, self.input_Te)
            self.fract_abu.calc_ionisation_balance_matrix(self.input_Ne, self.input_Nh)
            self.fract_abu.calc_F_z_tinf()
            (
                eig_vals,
                eig_vecs,
            ) = self.fract_abu.calc_eigen_vals_and_vecs()
        except Exception as e:
            raise e

        assert eig_vals.shape == (Z + 1, self.fract_abu.coord.size)
        assert eig_vecs.shape == (Z + 1, Z + 1, self.fract_abu.coord.size)

        assert np.all(np.logical_not(np.isnan(eig_vals)))
        assert np.all(np.logical_not(np.isinf(eig_vals)))

        assert np.all(np.logical_not(np.isnan(eig_vecs)))
        assert np.all(np.logical_not(np.isinf(eig_vecs)))

        ionisation_balance_matrix = self.fract_abu.ionisation_balance_matrix

        for icoord in range(self.fract_abu.coord.size):
            for ieig in range(self.fract_abu.nq):
                test_eigen = np.dot(
                    ionisation_balance_matrix[:, :, icoord], eig_vecs[:, ieig, icoord]
                ) - np.dot(eig_vals[ieig, icoord], eig_vecs[:, ieig, icoord])

                assert np.allclose(test_eigen, np.zeros(test_eigen.shape))

        try:
            eig_vals, eig_vecs = self.fract_abu.calc_eigen_vals_and_vecs()
        except Exception as e:
            raise e

        assert eig_vals.shape == (Z + 1, self.fract_abu.coord.size)
        assert eig_vecs.shape == (Z + 1, Z + 1, self.fract_abu.coord.size)

        assert np.all(np.logical_not(np.isnan(eig_vals)))
        assert np.all(np.logical_not(np.isinf(eig_vals)))

        assert np.all(np.logical_not(np.isnan(eig_vecs)))
        assert np.all(np.logical_not(np.isinf(eig_vecs)))

        ionisation_balance_matrix = self.fract_abu.ionisation_balance_matrix

        for icoord in range(self.fract_abu.coord.size):
            for ieig in range(self.fract_abu.nq):
                test_eigen = np.dot(
                    ionisation_balance_matrix[:, :, icoord], eig_vecs[:, ieig, icoord]
                ) - np.dot(eig_vals[ieig, icoord], eig_vecs[:, ieig, icoord])

                assert np.allclose(test_eigen, np.zeros(test_eigen.shape))

    def test_calc_eigen_coeffs(self):
        """Test calc_eigen_coeffs() function in FractionalAbundance class."""
        try:
            # Stick with default F_z_t0=None assignment
            self.fract_abu.interpolate_rates(self.input_Ne, self.input_Te)
            self.fract_abu.calc_ionisation_balance_matrix(self.input_Ne, self.input_Nh)
            self.fract_abu.calc_F_z_tinf()
            self.fract_abu.calc_eigen_vals_and_vecs()
            (
                eig_coeffs,
                F_z_t0,
            ) = self.fract_abu.calc_eigen_coeffs()
        except Exception as e:
            raise e

        assert eig_coeffs.shape == (Z + 1, self.fract_abu.coord.size)
        assert F_z_t0.shape == (Z + 1, self.fract_abu.coord.size)

        assert np.all(np.logical_not(np.isnan(eig_coeffs)))
        assert np.all(np.logical_not(np.isinf(eig_coeffs)))

        assert np.all(np.logical_not(np.isnan(F_z_t0)))
        assert np.all(np.logical_not(np.isinf(F_z_t0)))

    def test_frac_abu_full_run(self):
        try:
            # Stick with default F_z_t0=None assignment
            F_z_t = self.fract_abu_full_run(
                self.input_Te, self.input_Ne, self.input_Nh, self.input_tau
            )
        except Exception as e:
            raise e

        assert F_z_t.shape == (Z + 1, self.fract_abu_full_run.coord.size)
        assert np.all(np.logical_not(np.isnan(F_z_t)))
        assert np.all(np.logical_not(np.isinf(F_z_t)))

        assert np.all(self.fract_abu_full_run.F_z_t0.sel(ion_charge=0) == 1)
        assert np.all(self.fract_abu_full_run.F_z_t0.sel(ion_charge=slice(1, Z)) == 0)
        for icoord in range(self.fract_abu_full_run.coord.size):
            test_normalization = np.sum(F_z_t[:, icoord])
            assert np.abs(test_normalization - 1.0) <= 2e-2


class TestPowerLoss:
    def setup_class(self):
        (
            power_loss,
            power_loss_no_optional,
            power_loss_full_run,
        ) = power_loss_init()
        self.power_loss = power_loss
        self.power_loss_no_optional = power_loss_no_optional
        self.power_loss_full_run = power_loss_full_run
        self.rhop = RHOP
        self.input_Ne = INPUT_NE
        self.input_Te = INPUT_TE
        self.input_Nh = INPUT_NH
        self.input_tau = INPUT_TAU

    def test_interpolate_power(self):
        """Test interpolate_power() function in PowerLoss class."""
        try:
            self.power_loss.interpolate_power(self.input_Ne, self.input_Te)
        except Exception as e:
            raise e

    def test_calc_power_loss(self):
        """Test calculate_power_loss() function in PowerLoss class."""
        fract_abu = FractionalAbundance(
            SCD,
            ACD,
            ccd=CCD,
            full_run=True,
        )
        F_z_t = fract_abu(self.input_Te, self.input_Ne, self.input_Nh, self.input_tau)

        try:
            self.power_loss.interpolate_power(self.input_Ne, self.input_Te)
            cooling_factor = self.power_loss.calculate_power_loss(
                self.input_Ne, F_z_t, self.input_Nh
            )
        except Exception as e:
            raise e

        assert cooling_factor.shape == (Z + 1, self.power_loss.coord.size)

    def test_calc_power_loss_no_optional(self):
        """Test calculate_power_loss() function in PowerLoss class."""
        fract_abu = FractionalAbundance(
            SCD,
            ACD,
            ccd=CCD,
            full_run=True,
        )
        F_z_t = fract_abu(self.input_Te, self.input_Ne)

        try:
            self.power_loss.interpolate_power(self.input_Ne, self.input_Te)
            cooling_factor = self.power_loss.calculate_power_loss(self.input_Ne, F_z_t)
        except Exception as e:
            raise e

        assert cooling_factor.shape == (Z + 1, self.power_loss.coord.size)

    def test_power_loss_full_run(self):
        fract_abu = FractionalAbundance(
            SCD,
            ACD,
            ccd=CCD,
            full_run=True,
        )
        F_z_t = fract_abu(self.input_Te, self.input_Ne, self.input_Nh, self.input_tau)
        try:
            cooling_factor = self.power_loss(self.input_Ne, F_z_t, self.input_Nh)
        except Exception as e:
            raise e

        assert cooling_factor.shape == (Z + 1, self.power_loss.coord.size)
