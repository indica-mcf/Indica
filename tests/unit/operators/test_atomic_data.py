from copy import deepcopy

import numpy as np
import pytest
from xarray import DataArray

from indica.configs.operators import AuroraConfig
from indica.defaults.load_defaults import load_default_objects
from indica.examples import plasma as example_plasma
from indica.operators import FractionalAbundanceAdas
from indica.operators import PowerLoss
from indica.utilities import get_element_info

try:
    from indica.operators import FractionalAbundanceAurora
except ImportError:
    pass

MAIN_ION = "d"
ELEMENT = "be"
Z, A, NAME, SYMBOL = get_element_info(ELEMENT)

SCD_YEAR = "89"
ACD_YEAR = "89"
CCD_YEAR = "89"

PLT_YEAR = "89"
PRB_YEAR = "89"
PRC_YEAR = "89"

RHOP = np.linspace(0.0, 1.0, 10)
INPUT_NE = DataArray(data=np.logspace(19.0, 16.0, 10), coords={"rhop": RHOP})
INPUT_NN = DataArray(np.logspace(14.0, 16.0, 10), coords={"rhop": RHOP})
INPUT_TE = DataArray(data=np.logspace(4.6, 2, 10), coords={"rhop": RHOP})
INPUT_TAU = DataArray(data=np.logspace(0, -3, 10), coords={"rhop": RHOP})

EQUILIBRIUM = load_default_objects(
    "st40",
    "equilibrium",
)


def fractional_abundance_init():
    """Test initialisation of FractionalAbundance class."""

    fract_abu = FractionalAbundanceAdas(ELEMENT)
    fract_abu_full_run = FractionalAbundanceAdas(
        ELEMENT,
        SCD_YEAR,
        ACD_YEAR,
    )
    fract_abu_no_optional = FractionalAbundanceAdas(
        ELEMENT,
        SCD_YEAR,
        ACD_YEAR,
        CCD_YEAR,
    )

    return (
        fract_abu,
        fract_abu_no_optional,
        fract_abu_full_run,
    )


def power_loss_init():
    """Test initialisation of FractionalAbundance class."""

    power_loss = PowerLoss(ELEMENT)
    power_loss_no_optional = PowerLoss(
        ELEMENT,
        PLT_YEAR,
        PRB_YEAR,
    )
    power_loss_full_run = PowerLoss(
        ELEMENT,
        PLT_YEAR,
        PRB_YEAR,
        PRC_YEAR,
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
        self.input_Nn = INPUT_NN
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
                self.input_Ne, self.input_Nn
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
                self.input_Ne, self.input_Nn
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
            self.fract_abu.calc_ionisation_balance_matrix(self.input_Ne, self.input_Nn)
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
            self.fract_abu.calc_ionisation_balance_matrix(self.input_Ne, self.input_Nn)
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
                self.input_Te, self.input_Ne, self.input_Nn, self.input_tau
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


class TestFractionalAbundanceAurora:
    """Test that the fractional abundance operator can be used in Aurora."""

    def setup_class(self):
        pytest.importorskip(
            "indica.operators.fractionalabundance_aurora",
            reason="Issues with Aurora installation",
        )
        self.plasma = example_plasma(aurora_run=True)
        self.plasma.set_equilibrium(EQUILIBRIUM)
        self.plasma.build_atomic_data()

    def fractional_abundance_aurora_init(
        self,
    ):
        pytest.importorskip(
            "indica.operators.fractionalabundance_aurora",
            reason="Issues with Aurora installation",
        )
        self.ne = self.plasma.electron_density
        self.Te = self.plasma.electron_temperature
        self.Nn = self.plasma.neutral_density
        self.D_z = self.plasma.diffusion_coefficient
        self.V_z = self.plasma.convection_coefficient
        self.config = deepcopy(AuroraConfig)
        self.operator = FractionalAbundanceAurora(
            element=ELEMENT,
            main_ion=MAIN_ION,
            aurora_config=self.config,
            equilibrium=EQUILIBRIUM,
        )

    def test_call_returns_non_zero_values(self):
        pytest.importorskip(
            "indica.operators.fractionalabundance_aurora",
            reason="Issues with Aurora installation",
        )
        self.fractional_abundance_aurora_init()
        fz_t = self.operator(
            Ne=self.ne,
            Te=self.Te,
            Nn=self.Nn,
            D_z=self.D_z,
            V_z=self.V_z,
        )
        assert np.any(fz_t != 0)

    def test_call_with_one_timepoint_returns_non_zero_values(self):
        pytest.importorskip(
            "indica.operators.fractionalabundance_aurora",
            reason="Issues with Aurora installation",
        )
        self.fractional_abundance_aurora_init()
        fz_t = self.operator(
            Ne=self.ne.isel({"t": 0}),
            Te=self.Te.isel({"t": 0}),
            Nn=self.Nn.isel({"t": 0}),
            D_z=self.D_z.isel({"t": 0}),
            V_z=self.V_z.isel({"t": 0}),
        )
        assert np.any(fz_t != 0)

    def test_call_with_zero_nh_and_cxr_flag_true(self):
        pytest.importorskip(
            "indica.operators.fractionalabundance_aurora",
            reason="Issues with Aurora installation",
        )
        self.fractional_abundance_aurora_init()
        self.config["cxr_flag"] = True
        with pytest.raises(ValueError):
            self.operator(
                Ne=self.ne,
                Te=self.Te,
                Nn=self.Nn * 0.0,
                D_z=self.D_z,
                V_z=self.V_z,
            )

    def test_call_with_non_zero_nh_and_cxr_flag_false(self):
        pytest.importorskip(
            "indica.operators.fractionalabundance_aurora",
            reason="Issues with Aurora installation",
        )
        self.fractional_abundance_aurora_init()
        self.config["cxr_flag"] = False
        fz_t = self.operator(
            Ne=self.ne,
            Te=self.Te,
            Nn=self.Nn + 1e13,  # add a small non-zero value to ensure the operator runs
            D_z=self.D_z,
            V_z=self.V_z,
        )
        assert np.any(fz_t != 0)


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
        self.input_Nn = INPUT_NN
        self.input_tau = INPUT_TAU

    def test_interpolate_power(self):
        """Test interpolate_power() function in PowerLoss class."""
        try:
            self.power_loss.interpolate_power(self.input_Ne, self.input_Te)
        except Exception as e:
            raise e

    def test_calc_power_loss(self):
        """Test calculate_power_loss() function in PowerLoss class."""
        fract_abu = FractionalAbundanceAdas(ELEMENT)
        F_z_t = fract_abu(self.input_Te, self.input_Ne, self.input_Nn, self.input_tau)

        try:
            self.power_loss.interpolate_power(self.input_Ne, self.input_Te)
            cooling_factor = self.power_loss.calculate_power_loss(
                self.input_Ne, F_z_t, self.input_Nn
            )
        except Exception as e:
            raise e

        assert cooling_factor.shape == (Z + 1, self.power_loss.coord.size)

    def test_calc_power_loss_no_optional(self):
        """Test calculate_power_loss() function in PowerLoss class."""
        fract_abu = FractionalAbundanceAdas(ELEMENT)
        F_z_t = fract_abu(self.input_Te, self.input_Ne)

        try:
            self.power_loss.interpolate_power(self.input_Ne, self.input_Te)
            cooling_factor = self.power_loss.calculate_power_loss(self.input_Ne, F_z_t)
        except Exception as e:
            raise e

        assert cooling_factor.shape == (Z + 1, self.power_loss.coord.size)

    def test_power_loss_full_run(self):
        fract_abu = FractionalAbundanceAdas(ELEMENT)
        F_z_t = fract_abu(self.input_Te, self.input_Ne, self.input_Nn, self.input_tau)
        print(F_z_t)
        try:
            cooling_factor = self.power_loss(self.input_Ne, F_z_t, self.input_Nn)
        except Exception as e:
            raise e

        assert cooling_factor.shape == (Z + 1, self.power_loss.coord.size)
