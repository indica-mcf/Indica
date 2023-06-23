from copy import deepcopy
from typing import get_args
from typing import Hashable
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from scipy.stats import norm
from xarray import DataArray
from xarray import zeros_like

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.operators.bolometry_derivation import BolometryDerivation
from indica.operators.centrifugal_asymmetry import AsymmetryParameter
from indica.operators.centrifugal_asymmetry import ToroidalRotation
from indica.operators.extrapolate_impurity_density import asymmetry_from_R_z
from indica.operators.extrapolate_impurity_density import asymmetry_from_rho_theta
from indica.operators.extrapolate_impurity_density import ExtrapolateImpurityDensity
from indica.operators.extrapolate_impurity_density import recover_threshold_rho
from indica.readers import ADASReader
from .KB5_Bolometry_data import example_bolometry_LoS
from ..test_equilibrium_single import equilibrium_dat_and_te


class Recover_Threshold_Rho_Test_Case(TestCase):
    """Test case for testing type and value errors in recover_threshold_rho()."""

    def __init__(self, truncation_threshold, electron_temperature):
        """Initialise the test case with a set of nominal inputs."""
        self.truncation_threshold = truncation_threshold
        self.electron_temperature = electron_temperature

        self.nominal_inputs = [self.truncation_threshold, self.electron_temperature]

    def type_check(self, truncation_threshold=None, electron_temperature=None):
        """Test TypeError for recover_threshold_rho()."""
        inputs = [truncation_threshold, electron_temperature]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (truncation_threshold, electron_temperature) = inputs

        with self.assertRaises(TypeError):
            recover_threshold_rho(*inputs)

    def value_check(self, truncation_threshold=None, electron_temperature=None):
        """Test ValueError for recover_threshold_rho()."""
        inputs = [truncation_threshold, electron_temperature]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (truncation_threshold, electron_temperature) = inputs

        with self.assertRaises(ValueError):
            recover_threshold_rho(*inputs)


class Asymmetry_From_R_z_Test_Case(TestCase):
    """Test case for testing type and value errors in asymmetry_from_R_z()."""

    def __init__(self, data_R_z, flux_surfaces, rho_arr, threshold_rho, t_arr):
        """Initialise the test case with a set of nominal inputs."""
        self.data_R_z = data_R_z
        self.flux_surfaces = flux_surfaces
        self.rho_arr = rho_arr
        self.threshold_rho = threshold_rho
        self.t_arr = t_arr

        self.nominal_inputs = [
            self.data_R_z,
            self.flux_surfaces,
            self.rho_arr,
            self.threshold_rho,
            self.t_arr,
        ]

    def type_check(
        self,
        data_R_z=None,
        flux_surfaces=None,
        rho_arr=None,
        threshold_rho=None,
        t_arr=None,
    ):
        """Test TypeError for asymmetry_from_R_z()."""
        inputs = [data_R_z, flux_surfaces, rho_arr, threshold_rho, t_arr]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (data_R_z, flux_surfaces, rho_arr, threshold_rho, t_arr) = inputs

        with self.assertRaises(TypeError):
            recover_threshold_rho(*inputs)

    def value_check(self, data_R_z, flux_surfaces, rho_arr, threshold_rho, t_arr):
        """Test ValueError for asymmetry_from_R_z()."""
        inputs = [data_R_z, flux_surfaces, rho_arr, threshold_rho, t_arr]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (data_R_z, flux_surfaces, rho_arr, threshold_rho, t_arr) = inputs

        with self.assertRaises(ValueError):
            recover_threshold_rho(*inputs)


class Asymmetry_From_Rho_Theta_Test_Case(TestCase):
    """Test case for testing type and value errors in asymmetry_from_rho_theta()."""

    def __init__(self, data_rho_theta, flux_surfaces, threshold_rho, t_arr):
        """Initialise the test case with a set of nominal inputs."""
        self.data_rho_theta = data_rho_theta
        self.flux_surfaces = flux_surfaces
        self.threshold_rho = threshold_rho
        self.t_arr = t_arr

        self.nominal_inputs = [
            self.data_rho_theta,
            self.flux_surfaces,
            self.threshold_rho,
            self.t_arr,
        ]

    def type_check(
        self, data_rho_theta=None, flux_surfaces=None, threshold_rho=None, t_arr=None
    ):
        """Test TypeError for asymmetry_from_rho_theta()."""
        inputs = [data_rho_theta, flux_surfaces, threshold_rho, t_arr]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (data_rho_theta, flux_surfaces, threshold_rho, t_arr) = inputs

        with self.assertRaises(TypeError):
            recover_threshold_rho(*inputs)

    def value_check(self, data_rho_theta, flux_surfaces, threshold_rho, t_arr):
        """Test ValueError for asymmetry_from_rho_theta()."""
        inputs = [data_rho_theta, flux_surfaces, threshold_rho, t_arr]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (data_rho_theta, flux_surfaces, threshold_rho, t_arr) = inputs

        with self.assertRaises(ValueError):
            recover_threshold_rho(*inputs)


class Exception_Impurity_Density_Test_Case(TestCase):
    """Test case for testing type and value errors in ExtrapolateImpurityDensity
    call.
    """

    def __init__(
        self,
        impurity_density_sxr,
        electron_density,
        electron_temperature,
        truncation_threshold,
        flux_surfaces,
        asymmetry_parameter,
        t,
    ):
        """Initialise the test case with a set of nominal inputs."""
        self.impurity_density_sxr = impurity_density_sxr
        self.electron_density = electron_density
        self.electron_temperature = electron_temperature
        self.truncation_threshold = truncation_threshold
        self.flux_surfaces = flux_surfaces
        self.asymmetry_parameter = asymmetry_parameter
        self.t = t

        self.nominal_inputs = [
            self.impurity_density_sxr,
            self.electron_density,
            self.electron_temperature,
            self.truncation_threshold,
            self.flux_surfaces,
            self.asymmetry_parameter,
            self.t,
        ]

    def call_type_check(
        self,
        impurity_density_sxr=None,
        electron_density=None,
        electron_temperature=None,
        truncation_threshold=None,
        flux_surfaces=None,
        asymmetry_parameter=None,
        t=None,
    ):
        """Test TypeError for ExtrapolateImpurityDensity call."""
        inputs = [
            impurity_density_sxr,
            electron_density,
            electron_temperature,
            truncation_threshold,
            flux_surfaces,
            asymmetry_parameter,
            t,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            impurity_density_sxr,
            electron_density,
            electron_temperature,
            truncation_threshold,
            flux_surfaces,
            asymmetry_parameter,
            t,
        ) = inputs

        with self.assertRaises(TypeError):
            example_ = ExtrapolateImpurityDensity()
            example_(*inputs)

    def call_value_check(
        self,
        impurity_density_sxr=None,
        electron_density=None,
        electron_temperature=None,
        truncation_threshold=None,
        flux_surfaces=None,
        asymmetry_parameter=None,
        t=None,
    ):
        """ "Test ValueError for ExtrapolateImpurityDensity call."""
        inputs = [
            impurity_density_sxr,
            electron_density,
            electron_temperature,
            truncation_threshold,
            flux_surfaces,
            asymmetry_parameter,
            t,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            impurity_density_sxr,
            electron_density,
            electron_temperature,
            truncation_threshold,
            flux_surfaces,
            asymmetry_parameter,
            t,
        ) = inputs

        with self.assertRaises(ValueError):
            example_ = ExtrapolateImpurityDensity()
            example_(*inputs)


def invalid_input_checks_extrapolate_impurity_density(
    test_case: TestCase,
    nominal_input_name: str,
    nominal_input,
    zero_check: bool = False,
):
    """Tests that the test_case correctly identifies invalid inputs.

    Parameters
    ----------
    test_case
        Object of type TestCase that should contain the relevant check functions.
    nominal_input_name
        String signifying the internal name of the input variable
        (needed for error messages in case the check functions fail.)
    nominal_input
        Nominal value of the input variable.
    zero_check
        Optional boolean signifying whether or not to perform a check where a
        zero value for the input variable is checked.
    """
    if isinstance(test_case, Exception_Impurity_Density_Test_Case):
        if not isinstance(nominal_input, Hashable):
            invalid_input = 1.0
            test_case.call_type_check(**{nominal_input_name: invalid_input})
        elif isinstance(nominal_input, get_args(LabeledArray)):
            # Type ignore due to mypy complaining about redefinition of invalid_input

            invalid_input = "test"  # type:ignore
            test_case.call_type_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -1
            test_case.call_value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.inf
            test_case.call_value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -np.inf
            test_case.call_value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.nan
            test_case.call_value_check(**{nominal_input_name: invalid_input})

            if zero_check:
                invalid_input = deepcopy(nominal_input)  # type:ignore
                invalid_input *= 0
                test_case.call_value_check(**{nominal_input_name: invalid_input})

            if isinstance(nominal_input, (np.ndarray, DataArray)):
                invalid_input = deepcopy(nominal_input[0])  # type:ignore
                test_case.call_value_check(**{nominal_input_name: invalid_input})


def invalid_input_checks(
    test_case: TestCase,
    nominal_input_name: str,
    nominal_input,
    zero_check: bool = False,
):
    """Tests that the test_case correctly identifies invalid inputs.
    (Same as invalid_input_checks_extrapolate_impurity_density but two functions are
    split to satisfy complexity requirements from Flake8.)

    Parameters
    ----------
    test_case
        Object of type TestCase that should contain the relevant check functions.
    nominal_input_name
        String signifying the internal name of the input variable
        (needed for error messages in case the check functions fail.)
    nominal_input
        Nominal value of the input variable.
    zero_check
        Optional boolean signifying whether or not to perform a check where a
        zero value for the input variable is checked.
    """
    if isinstance(test_case, Recover_Threshold_Rho_Test_Case):
        if isinstance(nominal_input, get_args(LabeledArray)):
            # Type ignore due to mypy complaining about redefinition of invalid_input

            invalid_input = "test"  # type:ignore
            test_case.type_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -1  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.inf  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -np.inf  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.nan  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            if zero_check:
                invalid_input = deepcopy(nominal_input)  # type:ignore
                invalid_input *= 0
                test_case.value_check(**{nominal_input_name: invalid_input})

            if isinstance(nominal_input, (np.ndarray, DataArray)):
                invalid_input = deepcopy(nominal_input[0])  # type:ignore
                test_case.value_check(**{nominal_input_name: invalid_input})
    elif isinstance(test_case, Asymmetry_From_R_z_Test_Case):
        if not isinstance(nominal_input, Hashable):
            invalid_input = 1.0  # type:ignore
            test_case.type_check(**{nominal_input_name: invalid_input})
        elif isinstance(nominal_input, get_args(LabeledArray)):
            # Type ignore due to mypy complaining about redefinition of invalid_input

            invalid_input = "test"  # type:ignore
            test_case.type_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -1
            test_case.value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.inf  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -np.inf  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.nan  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            if zero_check:
                invalid_input = deepcopy(nominal_input)  # type:ignore
                invalid_input *= 0
                test_case.value_check(**{nominal_input_name: invalid_input})

            if isinstance(nominal_input, (np.ndarray, DataArray)):
                invalid_input = deepcopy(nominal_input[0])  # type:ignore
                test_case.value_check(**{nominal_input_name: invalid_input})
    elif isinstance(test_case, Asymmetry_From_Rho_Theta_Test_Case):
        if not isinstance(nominal_input, Hashable):
            invalid_input = 1.0  # type:ignore
            test_case.type_check(**{nominal_input_name: invalid_input})
        elif isinstance(nominal_input, get_args(LabeledArray)):
            # Type ignore due to mypy complaining about redefinition of invalid_input

            invalid_input = "test"  # type:ignore
            test_case.type_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -1  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.inf  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -np.inf  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.nan  # type:ignore
            test_case.value_check(**{nominal_input_name: invalid_input})

            if zero_check:
                invalid_input = deepcopy(nominal_input)  # type:ignore
                invalid_input *= 0
                test_case.value_check(**{nominal_input_name: invalid_input})

            if isinstance(nominal_input, (np.ndarray, DataArray)):
                invalid_input = deepcopy(nominal_input[0])  # type:ignore
                test_case.value_check(**{nominal_input_name: invalid_input})


def fractional_abundance_setup(
    element: str,
    t: np.ndarray,
    input_Te: DataArray,
    input_Ne: DataArray,
) -> DataArray:
    """Calculate and output Fractional abundance at t=infinity for calculating
    the mean charge in test_impurity_concentration()

    Parameters
    ----------
    element
        String of the symbol of the element per ADAS notation
        e.g be for Beryllium
    t
        Time np.ndarray (used for expanding the dimensions of the output of
        the function to ensure that time is a dimension of the output.)
    input_Te
        xarray.DataArray of electron temperature. Dimensions (rho, t)
    input_Ne
        xarray.DataArray of electron density. Dimensions (rho, t)

    Returns
    -------
    F_z_tinf
        Fractional abundance of the ionisation stages of the element at t=infinity.
        xarray.DataArray with dimensions (ion_charges, rho, t)
    """
    ADAS_file = ADASReader()

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    example_frac_abundance = FractionalAbundance(
        SCD,
        ACD,
    )

    example_frac_abundance.interpolate_rates(Ne=input_Ne, Te=input_Te)
    example_frac_abundance.calc_ionisation_balance_matrix(Ne=input_Ne)

    F_z_tinf = example_frac_abundance.calc_F_z_tinf()

    # ignore with mypy since this is testing and inputs are known
    F_z_tinf = F_z_tinf.expand_dims({"t": t.size}, axis=-1)  # type: ignore

    return F_z_tinf


def power_loss_setup(
    element: str,
    t: np.ndarray,
    input_Te: DataArray,
    input_Ne: DataArray,
) -> DataArray:
    """Calculate and output PowerLoss for calculating
    the extrapolated impurity density.

    Parameters
    ----------
    element
        String of the symbol of the element per ADAS notation
        e.g be for Beryllium
    t
        Time np.ndarray (used for expanding the dimensions of the output of
        the function to ensure that time is a dimension of the output.)
    input_Te
        xarray.DataArray of electron temperature. Dimensions (rho, t)
    input_Ne
        xarray.DataArray of electron density. Dimensions (rho, t)

    Returns
    -------
    power_loss
        Power loss of the element at t=infinity.
        xarray.DataArray with dimensions (ion_charges, rho, t).
    """
    ADAS_file = ADASReader()

    PLT = ADAS_file.get_adf11("plt", element, "89")
    PRB = ADAS_file.get_adf11("prc", element, "89")

    power_loss_obj = PowerLoss(PLT, PRB)

    frac_abund_inf = fractional_abundance_setup(
        element, t, input_Te.isel(t=0), input_Ne.isel(t=0)
    )

    power_loss = power_loss_obj(input_Ne.isel(t=0), input_Te.isel(t=0), frac_abund_inf)

    return power_loss


def input_data_setup():
    """Initial set-up for the majority of the data needed for ExtrapolateImpurityDensity.

    Returns
    -------
    input_Ne
        xarray.DataArray of electron density. Dimensions (rho, t)
    input_Te
        xarray.DataArray of electron temperature. Dimensions (rho, t)
    input_Ti
        xarray.DataArray of ion temperature. Dimensions (elements, rho, t)
    toroidal_rotations
        xarray.DataArray of toroidal rotations (needed for calculating the centrifugal
        asymmetry parameter). Dimensions (elements, rho, t)
    rho_arr
        xarray.DataArray of rho values, np.linspace(0, 1, 41). Dimensions (rho)
    theta_arr
        xarray.DataArray of theta values, np.linspace(-np.pi, np.pi, 21).
        Dimensions (theta)
    flux_surfs
        FluxSurfaceCoordinates object representing polar coordinate systems
        using flux surfaces for the radial coordinate.
    valid_truncation_threshold
        Truncation threshold (float) for the electron temperature
        (below this value soft-xray measurements are not valid)
    Zeff
        xarray.DataArray of the effective z(atomic number)-value for the plasma.
        Dimensions (rho, t)
    base_t
        xarray.DataArray of time values. Dimensions (t)
    R_derived
        Variable describing value of R in every coordinate on a (rho, theta) grid.
        xarray.DataArray with dimensions (rho, theta, t)
    R_lfs_values
        R_derived values at theta = 0 (ie low-field-side of the tokamak).
        xarray.DataArray with dimensions (rho, t)
    elements
        List of element symbols for all impurities.
    """
    base_rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])
    base_t = np.linspace(75.0, 80.0, 20)

    input_Te = np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3])
    input_Te = np.tile(input_Te, (len(base_t), 1)).T

    input_Te = DataArray(
        data=np.tile(np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]), (len(base_t), 1)).T,
        coords={"rho_poloidal": base_rho_profile, "t": base_t},
        dims=["rho_poloidal", "t"],
    )

    input_Ne = np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19])
    input_Ne = np.tile(input_Ne, (len(base_t), 1)).T

    input_Ne = DataArray(
        data=np.tile(
            np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19]), (len(base_t), 1)
        ).T,
        coords={"rho_poloidal": base_rho_profile, "t": base_t},
        dims=["rho_poloidal", "t"],
    )

    elements = ["be", "ne", "ni", "w"]

    input_Ti = np.array([2.0e3, 1.2e3, 0.5e3, 0.2e3, 0.1e3])
    input_Ti = np.tile(input_Ti, (len(elements), len(base_t), 1))
    input_Ti = np.swapaxes(input_Ti, 1, 2)

    input_Ti = DataArray(
        data=input_Ti,
        coords={"element": elements, "rho_poloidal": base_rho_profile, "t": base_t},
        dims=["element", "rho_poloidal", "t"],
    )

    toroidal_rotations = np.array([200.0e3, 170.0e3, 100.0e3, 30.0e3, 5.0e3])

    toroidal_rotations = np.tile(toroidal_rotations, (len(elements), len(base_t), 1))
    toroidal_rotations = np.swapaxes(toroidal_rotations, 1, 2)

    toroidal_rotations = DataArray(
        data=toroidal_rotations,
        coords=[
            ("element", elements),
            ("rho_poloidal", base_rho_profile),
            ("t", base_t),
        ],
        dims=["element", "rho_poloidal", "t"],
    )

    expanded_rho = np.linspace(base_rho_profile[0], base_rho_profile[-1], 41)

    rho_arr = expanded_rho
    theta_arr = np.linspace(-np.pi, np.pi, 21)

    rho_arr = DataArray(
        data=rho_arr, coords={"rho_poloidal": rho_arr}, dims=["rho_poloidal"]
    )
    theta_arr = DataArray(data=theta_arr, coords={"theta": theta_arr}, dims=["theta"])

    flux_surfs = FluxSurfaceCoordinates("poloidal")

    offset = MagicMock(return_value=0.02)
    equilib_dat, Te = equilibrium_dat_and_te()
    equilib = Equilibrium(equilib_dat, Te, sess=MagicMock(), offset_picker=offset)

    flux_surfs.set_equilibrium(equilib)

    R_derived, _ = flux_surfs.convert_to_Rz(
        DataArray(expanded_rho, {"rho_poloidal": expanded_rho}, dims=["rho_poloidal"]),
        theta_arr,
        base_t,
    )

    R_derived = R_derived.transpose("rho_poloidal", "theta", "t")

    R_lfs_values = R_derived.sel(theta=0, method="nearest")

    input_Te = input_Te.interp(rho_poloidal=expanded_rho, method="linear")
    input_Ne = input_Ne.interp(rho_poloidal=expanded_rho, method="linear")
    input_Ti = input_Ti.interp(rho_poloidal=expanded_rho, method="linear")
    toroidal_rotations = toroidal_rotations.interp(
        rho_poloidal=expanded_rho, method="linear"
    )

    toroidal_rotations /= R_lfs_values.data  # re-scale from velocity to frequency

    valid_truncation_threshold = 1.0e3

    Zeff = DataArray(
        data=1.85 * np.ones((*expanded_rho.shape, len(base_t))),
        coords=[("rho_poloidal", expanded_rho), ("t", base_t)],
        dims=["rho_poloidal", "t"],
    )

    return (
        input_Ne,
        input_Te,
        input_Ti,
        toroidal_rotations,
        rho_arr,
        theta_arr,
        flux_surfs,
        valid_truncation_threshold,
        Zeff,
        base_t,
        R_derived,
        R_lfs_values,
        elements,
    )


def gaussian_perturbation(gaussian_params):
    """Function to construct a signal that follows a Gaussian profile with
    three free parameters.

    Parameters
    ----------
    gaussian_params
        A list containing floats:
            amplitude
                Amplitude of the additional signal (Gaussian amplitude)
            standard_dev
                Standard deviation associated with the Gaussian construction
                (can be defined as FWHM/2.355 where FWHM is full-width at half maximum)
            position
                Position of the Gaussian. During optimization this is constrained to
                the extrapolated region of rho (ie. outside the SXR validity region).

    Returns
    -------
    sig
        xarray.DataArray containing the Gaussian signal with dimensions (rho,)
    """
    rho_arr, amplitude, standard_dev, position = gaussian_params

    gaussian_signal = norm(loc=position, scale=standard_dev)

    sig = gaussian_signal.pdf(rho_arr)

    sig = DataArray(data=sig, coords={"rho_poloidal": rho_arr}, dims=["rho_poloidal"])

    sig /= sig.max()

    sig *= amplitude

    return sig


def sxr_data_setup(input_data, sxr_truncation=True):
    """Set-up for soft-x-ray derived data.

    Parameters
    ----------
    input_data
        Output of input_data_setup().
    sxr_truncation
        Boolean to determine whether or not to assume constant asymmetry parameter
        outside the soft x-ray valid range.

    Returns
    -------
    input_sxr_density_asym_Rz
        Ground truth asymmetric impurity density on a (R, z) grid.
        xarray.DataArray with dimensions (R, z, t)
    input_sxr_density_asym_rho_theta
        Ground truth asymmetric impurity density on a (rho, theta) grid.
        xarray.DataArray with dimensions (rho, theta, t)
    additional_sig
        Gaussian signal to be added after extrapolation of the sxr_density
        outside of the valid SXR limits.
    example_asymmetry
        Asymmetry parameter, xarray.DataArray with dimensions (rho, t)
    asymmetry_modifier
        Asymmetry modifier used to transform a low-field side only rho-profile
        of a poloidally asymmetric quantity to a full poloidal cross-sectional
        profile ie. (rho, t) -> (rho, theta, t). Also can be defined as:
        exp(asymmetry_parameter * (R ** 2 - R_lfs ** 2)), where R is the major
        radius as a function of (rho, theta, t) and R_lfs is the low-field-side
        major radius as a function of (rho, t). xarray DataArray with dimensions
        (rho, theta, t)
    """
    (
        input_Ne,
        input_Te,
        input_Ti,
        toroidal_rotations,
        rho_arr,
        theta_arr,
        flux_surfs,
        valid_truncation_threshold,
        Zeff,
        base_t,
        R_derived,
        R_lfs_values,
        elements,
    ) = input_data

    R_arr = np.linspace(1.83, 3.9, 100)
    z_arr = np.linspace(-1.75, 2.0, 100)

    R_arr = DataArray(data=R_arr, coords={"R": R_arr}, dims=["R"])
    z_arr = DataArray(data=z_arr, coords={"z": z_arr}, dims=["z"])

    additional_sig = zeros_like(input_Ne)
    num_t = len(additional_sig.coords["t"].data)
    for ind_t, it in enumerate(additional_sig.coords["t"].data):
        amp = 1.0e17 - (ind_t / num_t) * 0.005e17
        width = 0.05 - (ind_t / num_t) * 0.0004
        pos = 0.9 - (ind_t / num_t) * 0.002
        additional_sig.loc[:, it] = gaussian_perturbation((rho_arr, amp, width, pos))

    sxr_density_data = 1e-3 * input_Ne.isel(t=0)  # 25.0e15 * np.exp(-rho_arr)

    sxr_density_data = np.tile(sxr_density_data, (len(base_t), len(theta_arr), 1))

    sxr_density_data = np.transpose(sxr_density_data, [2, 1, 0])

    input_sxr_density = DataArray(
        data=sxr_density_data,
        coords={"rho_poloidal": rho_arr, "theta": theta_arr, "t": base_t},
        dims=["rho_poloidal", "theta", "t"],
    )

    example_asymmetry_obj = AsymmetryParameter()
    example_asymmetry = example_asymmetry_obj(
        toroidal_rotations.copy(deep=True), input_Ti, "d", "w", Zeff, input_Te
    )
    example_asymmetry = example_asymmetry.transpose("rho_poloidal", "t")

    if sxr_truncation:
        threshold_rho = recover_threshold_rho(valid_truncation_threshold, input_Te)

        example_asymmetry.loc[threshold_rho[0] :, :] = example_asymmetry.loc[
            threshold_rho[0], :
        ]

    rho_derived, theta_derived = flux_surfs.convert_from_Rz(R_arr, z_arr, base_t)
    rho_derived = abs(rho_derived)

    rho_derived = rho_derived.transpose("R", "z", "t")
    theta_derived = theta_derived.transpose("R", "z", "t")

    asymmetry_modifier = np.exp(
        example_asymmetry * (R_derived**2 - R_lfs_values**2)
    )
    asymmetry_modifier = asymmetry_modifier.transpose("rho_poloidal", "theta", "t")

    input_sxr_density_lfs = input_sxr_density.sel(theta=0)  # + additional_sig
    additional_sig = additional_sig * asymmetry_modifier
    additional_sig = additional_sig.transpose("rho_poloidal", "theta", "t")

    input_sxr_density_asym_rho_theta = input_sxr_density_lfs * asymmetry_modifier
    input_sxr_density_asym_rho_theta = input_sxr_density_asym_rho_theta.transpose(
        "rho_poloidal", "theta", "t"
    )

    input_sxr_density_asym_R_z = input_sxr_density_asym_rho_theta.indica.interp2d(
        {"rho_poloidal": rho_derived, "theta": theta_derived}, method="linear"
    )
    input_sxr_density_asym_R_z = input_sxr_density_asym_R_z.fillna(0.0)
    input_sxr_density_asym_R_z = input_sxr_density_asym_R_z.transpose("R", "z", "t")

    return (
        input_sxr_density_asym_R_z,
        input_sxr_density_asym_rho_theta,
        additional_sig,
        example_asymmetry,
        asymmetry_modifier,
    )


def bolometry_input_data_setup(input_data):
    """Set-up for input data used for calculating bolometry data.

    Parameters
    ----------
    input_data
        Output of input_data_setup().

    Returns
    -------
    example_frac_abunds
        List of fractional abundances (an xarray.DataArray for each impurity)
        dimensions of each element in list are (ion_charges, rho, t).
    main_ion_power_loss
        Power loss associated with the main ion (eg. deuterium),
        xarray.DataArray with dimensions (rho, t)
    impurity_power_losses
        Power loss associated with all of the impurity elements,
        xarray.DataArray with dimensions (elements, rho, t)
    """
    initial_data = input_data

    base_t = initial_data[9]
    input_Te = initial_data[1]
    input_Ne = initial_data[0]
    elements = initial_data[12]

    main_ion_power_loss = power_loss_setup("h", base_t, input_Te, input_Ne)

    main_ion_power_loss = main_ion_power_loss.assign_coords(t=("t", base_t))

    main_ion_power_loss = main_ion_power_loss.sum(dim="ion_charges")

    example_frac_abunds = []
    impurity_power_losses = []

    for ielement in elements:
        example_frac_abund = fractional_abundance_setup(
            ielement, base_t, input_Te.isel(t=0), input_Ne.isel(t=0)
        )
        example_frac_abunds.append(example_frac_abund)

        impurity_power_loss = power_loss_setup(ielement, base_t, input_Te, input_Ne)
        impurity_power_loss = impurity_power_loss.assign_coords(t=("t", base_t))
        impurity_power_loss = impurity_power_loss.sum(dim="ion_charges")
        impurity_power_losses.append(impurity_power_loss.data)

    impurity_power_losses = np.array(impurity_power_losses)

    impurity_power_losses = DataArray(
        data=impurity_power_losses,
        coords=dict(**{"element": elements}, **impurity_power_loss.coords),
        dims=["element", *impurity_power_loss.dims],
    )

    return (example_frac_abunds, main_ion_power_loss, impurity_power_losses)


def test_extrapolate_impurity_density_call():
    """Test ExtrapolateImpurityDensity.__call__ and
    ExtrapolateImpurityDensity.optimize_perturbation
    """
    initial_data = input_data_setup()
    sxr_data = sxr_data_setup(initial_data)
    bolometry_input_data = bolometry_input_data_setup(initial_data)

    input_Ne = initial_data[0]
    input_Te = initial_data[1]
    valid_truncation_threshold = initial_data[7]
    flux_surfs = initial_data[6]
    base_t = initial_data[9]
    R_derived = initial_data[10]
    elements = initial_data[12]
    impurity_sxr_density_asym_Rz = sxr_data[0]
    additional_sig = sxr_data[2]
    orig_asymmetry_param = sxr_data[3]
    example_frac_abunds = bolometry_input_data[0]
    main_ion_power_loss = bolometry_input_data[1]
    impurity_power_loss = bolometry_input_data[2]

    example_extrapolate_impurity_density = ExtrapolateImpurityDensity()

    try:
        (
            example_result_R_z,
            example_result_rho_theta,
            t,
        ) = example_extrapolate_impurity_density(
            impurity_sxr_density_asym_Rz,
            input_Ne,
            input_Te,
            valid_truncation_threshold,
            flux_surfs,
        )
    except Exception as e:
        raise e

    perturbed_impurity_sxr_density_rho_theta = (
        example_result_rho_theta.copy(deep=True) + additional_sig
    )

    assert np.all(t == base_t)

    rho_profile = example_result_rho_theta.coords["rho_poloidal"].data
    theta_profile = example_result_rho_theta.coords["theta"].data

    beryllium_impurity_conc = np.tile(
        0.03 * input_Ne.data, (theta_profile.shape[0], 1, 1)
    )
    neon_impurity_conc = np.tile(0.02 * input_Ne.data, (theta_profile.shape[0], 1, 1))
    nickel_impurity_conc = np.tile(
        0.0002 * input_Ne.data, (theta_profile.shape[0], 1, 1)
    )

    beryllium_impurity_conc = np.transpose(beryllium_impurity_conc, (1, 0, 2))
    neon_impurity_conc = np.transpose(neon_impurity_conc, (1, 0, 2))
    nickel_impurity_conc = np.transpose(nickel_impurity_conc, (1, 0, 2))

    # be, ne, ni, w
    elements = ["be", "ne", "ni", "w"]

    impurity_densities = DataArray(
        data=np.ones(
            (len(elements), *rho_profile.shape, *theta_profile.shape, *t.shape)
        ),
        coords=[
            ("element", elements),
            ("rho_poloidal", rho_profile),
            ("theta", theta_profile),
            ("t", t.data),
        ],
        dims=["element", "rho_poloidal", "theta", "t"],
    )
    impurity_densities.data[0] = beryllium_impurity_conc
    impurity_densities.data[1] = neon_impurity_conc
    impurity_densities.data[2] = nickel_impurity_conc
    impurity_densities.data[3] = perturbed_impurity_sxr_density_rho_theta

    bolometry_args = [
        flux_surfs,
        example_bolometry_LoS,
        input_Ne.coords["t"],
        impurity_densities,
        example_frac_abunds,
        elements,
        input_Ne,
        main_ion_power_loss,
        impurity_power_loss,
    ]

    example_bolometry_derivation = BolometryDerivation(*bolometry_args)

    original_bolometry = example_bolometry_derivation(deriv_only=False, trim=False)
    example_bolometry_derivation(deriv_only=False, trim=True)

    original_bolometry = original_bolometry.rename({"channels": "bolo_kb5v_coords"})
    original_bolometry = original_bolometry.transpose("t", "bolo_kb5v_coords")

    optimized_impurity_density = (
        example_extrapolate_impurity_density.optimize_perturbation(
            example_result_rho_theta,
            original_bolometry.copy(deep=True),
            example_bolometry_derivation,
            "w",
            orig_asymmetry_param,
            threshold_rho=0.6,
            R_deriv=R_derived,
            time_correlation=True,
        )
    )

    # Rough check of entire rho_profile on the low-field-side.
    sum_of_residuals = np.abs(
        optimized_impurity_density.sel(theta=0)
        - perturbed_impurity_sxr_density_rho_theta.sel(theta=0)
    )
    sum_of_original = np.abs(perturbed_impurity_sxr_density_rho_theta.sel(theta=0))

    sum_of_original = sum_of_original.sum(["rho_poloidal"])
    sum_of_residuals = sum_of_residuals.sum(["rho_poloidal"])

    relative_fit_error = sum_of_residuals / sum_of_original

    try:
        assert np.max(relative_fit_error) < 0.1
    except AssertionError:
        raise AssertionError(
            f"Relative error across rho_poloidal is too high(maximum allowed is 0.1): \
                relative error = {relative_fit_error}"
        )

    # Constrained check of invalid SXR range of rho-profile on the low-field-side.
    sum_of_original = zeros_like(relative_fit_error)
    sum_of_residuals = zeros_like(relative_fit_error)

    for ind_t, it in enumerate(sum_of_original.t):
        sum_of_residuals_ = np.abs(
            optimized_impurity_density.sel(
                theta=0,
                rho_poloidal=slice(
                    example_extrapolate_impurity_density.threshold_rho.isel(
                        t=ind_t
                    ).data,
                    1.0,
                ),
            ).isel(t=ind_t)
            - perturbed_impurity_sxr_density_rho_theta.sel(
                theta=0,
                rho_poloidal=slice(
                    example_extrapolate_impurity_density.threshold_rho.isel(
                        t=ind_t
                    ).data,
                    1.0,
                ),
            ).isel(t=ind_t)
        )

        sum_of_original_ = np.abs(
            perturbed_impurity_sxr_density_rho_theta.sel(
                theta=0,
                rho_poloidal=slice(
                    example_extrapolate_impurity_density.threshold_rho.isel(
                        t=ind_t
                    ).data,
                    1.0,
                ),
            ).isel(t=ind_t)
        )

        sum_of_original.loc[it] = sum_of_original_.sum(["rho_poloidal"])
        sum_of_residuals.loc[it] = sum_of_residuals_.sum(["rho_poloidal"])

    relative_fit_error_constrained = sum_of_residuals / sum_of_original

    try:
        assert np.max(relative_fit_error_constrained) < 0.2
    except AssertionError:
        raise AssertionError(
            f"Relative error in invalid SXR region is too \
                high(maximum allowed is 0.2): relative error = \
                    {relative_fit_error_constrained}"
        )

    example_extrapolate_test_case = Exception_Impurity_Density_Test_Case(
        impurity_sxr_density_asym_Rz,
        input_Ne,
        input_Te,
        valid_truncation_threshold,
        flux_surfs,
        orig_asymmetry_param,
        t,
    )

    # Invalid SXR derived density checks

    invalid_input_checks_extrapolate_impurity_density(
        example_extrapolate_test_case,
        "impurity_density_sxr",
        impurity_sxr_density_asym_Rz,
    )

    # Invalid electron density checks

    invalid_input_checks_extrapolate_impurity_density(
        example_extrapolate_test_case, "electron_density", input_Ne
    )

    # Invalid electron temperature checks

    invalid_input_checks_extrapolate_impurity_density(
        example_extrapolate_test_case, "electron_temperature", input_Te, zero_check=True
    )

    # Invalid truncation threshold check

    invalid_input_checks_extrapolate_impurity_density(
        example_extrapolate_test_case,
        "truncation_threshold",
        valid_truncation_threshold,
        zero_check=True,
    )

    invalid_input_checks_extrapolate_impurity_density(
        example_extrapolate_test_case, "flux_surfaces", flux_surfs
    )

    invalid_input_checks_extrapolate_impurity_density(
        example_extrapolate_test_case, "asymmetry_parameter", orig_asymmetry_param
    )

    invalid_input_checks_extrapolate_impurity_density(
        example_extrapolate_test_case, "t", t
    )


def test_asymmetry_from_profile():
    """Test for asymmetry_from_R_z() and asymmetry_from_rho_theta()."""
    initial_data = input_data_setup()
    sxr_data = sxr_data_setup(initial_data, False)

    input_Te = initial_data[1]
    input_Ti = initial_data[2]
    orig_toroidal_rotations = initial_data[3]
    flux_surfs = initial_data[6]
    impurity_sxr_density_asym_Rz = sxr_data[0]
    impurity_sxr_density_asym_rho_theta = sxr_data[1]

    orig_toroidal_rotations = orig_toroidal_rotations.sel(element="w")

    rho_arr = impurity_sxr_density_asym_rho_theta.coords["rho_poloidal"]
    t_arr = impurity_sxr_density_asym_rho_theta.coords["t"]

    try:
        asymmetry_from_R_z_ = asymmetry_from_R_z(
            impurity_sxr_density_asym_Rz, flux_surfs, rho_arr, t_arr=t_arr
        )
    except Exception as e:
        raise e

    try:
        asymmetry_from_rho_theta_ = asymmetry_from_rho_theta(
            impurity_sxr_density_asym_rho_theta, flux_surfs, t_arr=t_arr
        )
    except Exception as e:
        raise e

    assert all(dim in asymmetry_from_R_z_.dims for dim in ["rho_poloidal", "t"])
    assert all(dim in asymmetry_from_rho_theta_.dims for dim in ["rho_poloidal", "t"])

    asymmetry_from_R_z_ = asymmetry_from_R_z_.expand_dims("element")
    asymmetry_from_R_z_ = asymmetry_from_R_z_.assign_coords({"element": ["w"]})

    asymmetry_from_rho_theta_ = asymmetry_from_rho_theta_.expand_dims("element")
    asymmetry_from_rho_theta_ = asymmetry_from_rho_theta_.assign_coords(
        {"element": ["w"]}
    )

    Zeff = DataArray(
        data=1.85 * np.ones((*rho_arr.shape, len(t_arr))),
        coords=[("rho_poloidal", rho_arr.data), ("t", t_arr.data)],
        dims=["rho_poloidal", "t"],
    )

    main_ion = "d"
    impurity = "w"

    derived_toroidal_rotation_rho_theta = ToroidalRotation()

    toroidal_rotation_results_rho_theta = derived_toroidal_rotation_rho_theta(
        asymmetry_from_rho_theta_, input_Ti, main_ion, impurity, Zeff, input_Te
    )

    assert np.allclose(
        orig_toroidal_rotations, toroidal_rotation_results_rho_theta, rtol=0.05
    )

    derived_toroidal_rotation_R_z = ToroidalRotation()

    toroidal_rotation_results_R_z = derived_toroidal_rotation_R_z(
        asymmetry_from_R_z_, input_Ti, main_ion, impurity, Zeff, input_Te
    )

    # Last 3 rho points don't match most likely due to converison from R-z to rho-theta,
    # not much that can be done about that.
    assert np.allclose(
        orig_toroidal_rotations[:-3, :], toroidal_rotation_results_R_z[:-3, :], rtol=0.1
    )

    valid_truncation_threshold = initial_data[7]

    valid_threshold_rho = recover_threshold_rho(valid_truncation_threshold, input_Te)

    example_asymmetry_from_R_z_test_case = Asymmetry_From_R_z_Test_Case(
        impurity_sxr_density_asym_Rz,
        flux_surfs,
        rho_arr,
        valid_threshold_rho,
        t_arr=t_arr,
    )

    invalid_input_checks(
        example_asymmetry_from_R_z_test_case,
        "data_R_z",
        impurity_sxr_density_asym_Rz,
        False,
    )

    invalid_input_checks(
        example_asymmetry_from_R_z_test_case,
        "flux_surfaces",
        flux_surfs,
        False,
    )

    invalid_input_checks(
        example_asymmetry_from_R_z_test_case, "rho_arr", rho_arr, False
    )

    invalid_input_checks(
        example_asymmetry_from_R_z_test_case,
        "threshold_rho",
        valid_threshold_rho,
        False,
    )

    invalid_input_checks(example_asymmetry_from_R_z_test_case, "t_arr", t_arr, False)

    example_asymmetry_from_rho_theta_test_case = Asymmetry_From_Rho_Theta_Test_Case(
        impurity_sxr_density_asym_rho_theta,
        flux_surfs,
        valid_threshold_rho,
        t_arr=t_arr,
    )

    invalid_input_checks(
        example_asymmetry_from_rho_theta_test_case,
        "data_rho_theta",
        impurity_sxr_density_asym_rho_theta,
        False,
    )

    invalid_input_checks(
        example_asymmetry_from_rho_theta_test_case,
        "flux_surfaces",
        flux_surfs,
        False,
    )

    invalid_input_checks(
        example_asymmetry_from_rho_theta_test_case,
        "threshold_rho",
        valid_threshold_rho,
        False,
    )

    invalid_input_checks(
        example_asymmetry_from_rho_theta_test_case, "t_arr", t_arr, False
    )


def test_recover_threshold_rho():
    initial_data = input_data_setup()
    valid_input_Te = initial_data[1]
    valid_truncation_threshold = initial_data[7]

    try:
        threshold_rho = recover_threshold_rho(
            valid_truncation_threshold, valid_input_Te
        )
    except Exception as e:
        raise e

    assert np.allclose(threshold_rho, 0.575)

    example_recover_threshold_rho_test_case = Recover_Threshold_Rho_Test_Case(
        valid_truncation_threshold, valid_input_Te
    )

    invalid_input_checks(
        example_recover_threshold_rho_test_case,
        "truncation_threshold",
        valid_truncation_threshold,
        True,
    )
    invalid_input_checks(
        example_recover_threshold_rho_test_case,
        "electron_temperature",
        valid_input_Te,
        True,
    )
