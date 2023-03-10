from copy import deepcopy
from typing import get_args
from typing import Hashable
from typing import Sequence
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from xarray import DataArray

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.operators.bolometry_derivation import BolometryDerivation
from indica.readers import OpenADASReader
from .KB5_Bolometry_data import example_bolometry_LoS
from ..test_equilibrium_single import equilibrium_dat_and_te


class Exception_Bolometry_Derivation(TestCase):
    """Test case for testing type and value and attribute errors in BolometryDerivation
    initialisation and __call__ functions.
    """

    def __init__(
        self,
        flux_surfs,
        LoS_bolometry_data,
        t_arr,
        impurity_densities,
        frac_abunds,
        impurity_elements,
        electron_density,
        main_ion_power_loss,
        impurities_power_loss,
        deriv_only,
        trim,
        t_val,
    ):
        """Initialise the test case with a set of nominal inputs."""
        self.flux_surfs = flux_surfs
        self.LoS_bolometry_data = LoS_bolometry_data
        self.t_arr = t_arr
        self.impurity_densities = impurity_densities
        self.frac_abunds = frac_abunds
        self.impurity_elements = impurity_elements
        self.electron_density = electron_density
        self.main_ion_power_loss = main_ion_power_loss
        self.impurities_power_loss = impurities_power_loss
        self.deriv_only = deriv_only
        self.trim = trim
        self.t_val = t_val

        self.nominal_inputs = [
            self.flux_surfs,
            self.LoS_bolometry_data,
            self.t_arr,
            self.impurity_densities,
            self.frac_abunds,
            self.impurity_elements,
            self.electron_density,
            self.main_ion_power_loss,
            self.impurities_power_loss,
            self.deriv_only,
            self.trim,
            self.t_val,
        ]

        self.nominal_bolometry_derivation_obj = BolometryDerivation(
            *self.nominal_inputs
        )

    def init_type_error_check(
        self,
        flux_surfs=None,
        LoS_bolometry_data=None,
        t_arr=None,
        impurity_densities=None,
        frac_abunds=None,
        impurity_elements=None,
        electron_density=None,
        main_ion_power_loss=None,
        impurities_power_loss=None,
    ):
        """Test type errors are raised for BolometryDerivation initialisation."""
        inputs = [
            flux_surfs,
            LoS_bolometry_data,
            t_arr,
            impurity_densities,
            frac_abunds,
            impurity_elements,
            electron_density,
            main_ion_power_loss,
            impurities_power_loss,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        with self.assertRaises(TypeError):
            BolometryDerivation(*inputs)

    def init_value_error_check(
        self,
        flux_surfs=None,
        LoS_bolometry_data=None,
        t_arr=None,
        impurity_densities=None,
        frac_abunds=None,
        impurity_elements=None,
        electron_density=None,
        main_ion_power_loss=None,
        impurities_power_loss=None,
    ):
        """Test value errors are raised for BolometryDerivation initialisation."""
        inputs = [
            flux_surfs,
            LoS_bolometry_data,
            t_arr,
            impurity_densities,
            frac_abunds,
            impurity_elements,
            electron_density,
            main_ion_power_loss,
            impurities_power_loss,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        with self.assertRaises(ValueError):
            BolometryDerivation(*inputs)

    def call_type_error_check(self, deriv_only=None, trim=None, t_val=None):
        """Test type errors are raised for BolometryDerivation call."""
        inputs = [deriv_only, trim, t_val]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 8]

        with self.assertRaises(TypeError):
            self.nominal_bolometry_derivation_obj(*inputs)

    def call_value_error_check(self, deriv_only=None, trim=None, t_val=None):
        """Test value errors are raised for BolometryDerivation call."""
        inputs = [deriv_only, trim, t_val]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i + 8]

        with self.assertRaises(ValueError):
            self.nominal_bolometry_derivation_obj(*inputs)


def input_data_setup():
    """Initial set-up for the majority of the data needed for ExtrapolateImpurityDensity.

    Returns
    -------
    input_Ne
        xarray.DataArray of electron density. Dimensions (rho, t)
    input_Te
        xarray.DataArray of electron temperature. Dimensions (rho, t)
    flux_surfs
        FluxSurfaceCoordinates object representing polar coordinate systems
        using flux surfaces for the radial coordinate.
    base_t
        xarray.DataArray of time values. Dimensions (t)
    elements
        List of element symbols for all impurities.
    rho_arr
        xarray.DataArray of rho values, np.linspace(0, 1, 41). Dimensions (rho)
    theta_arr
        xarray.DataArray of theta values, np.linspace(-np.pi, np.pi, 21).
        Dimensions (theta)
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

    elements = ["be", "ne", "ni"]

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

    input_Te = input_Te.interp(rho_poloidal=expanded_rho, method="linear")
    input_Ne = input_Ne.interp(rho_poloidal=expanded_rho, method="linear")

    return (
        input_Ne,
        input_Te,
        flux_surfs,
        base_t,
        elements,
        rho_arr,
        theta_arr,
    )


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
    ADAS_file = OpenADASReader()

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
    ADAS_file = OpenADASReader()

    PLT = ADAS_file.get_adf11("plt", element, "89")
    PRB = ADAS_file.get_adf11("prc", element, "89")

    power_loss_obj = PowerLoss(PLT, PRB)

    frac_abund_inf = fractional_abundance_setup(
        element, t, input_Te.isel(t=0), input_Ne.isel(t=0)
    )

    power_loss = power_loss_obj(input_Ne.isel(t=0), input_Te.isel(t=0), frac_abund_inf)

    return power_loss


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

    base_t = initial_data[3]
    input_Te = initial_data[1]
    input_Ne = initial_data[0]
    elements = initial_data[4]

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
        coords=dict(**{"elements": elements}, **impurity_power_loss.coords),
        dims=["elements", *impurity_power_loss.dims],
    )

    return (example_frac_abunds, main_ion_power_loss, impurity_power_losses)


def invalid_init_input_checks(
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
    if isinstance(test_case, Exception_Bolometry_Derivation):
        if not isinstance(nominal_input, Hashable):
            invalid_input = 1.0
            test_case.init_type_error_check(**{nominal_input_name: invalid_input})
        elif isinstance(nominal_input, Sequence):
            invalid_input = "test"  # type:ignore
            test_case.init_type_error_check(**{nominal_input_name: invalid_input})
        elif isinstance(nominal_input, get_args(LabeledArray)):
            # Type ignore due to mypy complaining about redefinition of invalid_input

            invalid_input = "test"  # type:ignore
            test_case.init_type_error_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -1
            test_case.init_value_error_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.inf
            test_case.init_value_error_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -np.inf
            test_case.init_value_error_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.nan
            test_case.init_value_error_check(**{nominal_input_name: invalid_input})

            if zero_check:
                invalid_input = deepcopy(nominal_input)  # type:ignore
                invalid_input *= 0
                test_case.init_value_error_check(**{nominal_input_name: invalid_input})

            if isinstance(nominal_input, (np.ndarray, DataArray)):
                invalid_input = deepcopy(nominal_input[0])  # type:ignore
                test_case.init_value_error_check(**{nominal_input_name: invalid_input})


def invalid_call_input_checks(
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
    if isinstance(test_case, Exception_Bolometry_Derivation):
        if not isinstance(nominal_input, Hashable):
            invalid_input = 1.0
            test_case.call_type_error_check(**{nominal_input_name: invalid_input})
        elif isinstance(nominal_input, get_args(LabeledArray)):
            # Type ignore due to mypy complaining about redefinition of invalid_input

            invalid_input = "test"  # type:ignore
            test_case.call_type_error_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -1
            test_case.call_value_error_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.inf
            test_case.call_value_error_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= -np.inf
            test_case.call_value_error_check(**{nominal_input_name: invalid_input})

            invalid_input = deepcopy(nominal_input)  # type:ignore
            invalid_input *= np.nan
            test_case.call_value_error_check(**{nominal_input_name: invalid_input})

            if zero_check:
                invalid_input = deepcopy(nominal_input)  # type:ignore
                invalid_input *= 0
                test_case.call_value_error_check(**{nominal_input_name: invalid_input})

            if isinstance(nominal_input, (np.ndarray, DataArray)):
                invalid_input = deepcopy(nominal_input[0])  # type:ignore
                test_case.call_value_error_check(**{nominal_input_name: invalid_input})


def test_bolometry_derivation():
    """Tests invalid inputs to BolometryDerivation initialisation and call.
    Output of BolometryDerivation call is tested in:
        tests/regression/operators/test_bolometry_derivation.py
    """
    initial_data = input_data_setup()
    bolometry_input_data = bolometry_input_data_setup(initial_data)

    flux_surfs = initial_data[2]
    LoS_bolometry_data = example_bolometry_LoS
    t_arr = initial_data[3]
    frac_abunds = bolometry_input_data[0]
    impurity_elements = initial_data[4]
    electron_density = initial_data[0]
    main_ion_power_loss = bolometry_input_data[1]
    impurities_power_loss = bolometry_input_data[2]
    rho_arr = initial_data[5]
    theta_arr = initial_data[6]

    beryllium_density = np.tile(
        0.03 * electron_density.data, (theta_arr.shape[0], 1, 1)
    )
    neon_density = np.tile(0.02 * electron_density.data, (theta_arr.shape[0], 1, 1))
    nickel_density = np.tile(0.0002 * electron_density.data, (theta_arr.shape[0], 1, 1))

    beryllium_density = np.transpose(beryllium_density, (1, 0, 2))
    neon_density = np.transpose(neon_density, (1, 0, 2))
    nickel_density = np.transpose(nickel_density, (1, 0, 2))

    impurity_densities = DataArray(
        data=np.ones(
            (len(impurity_elements), *rho_arr.shape, *theta_arr.shape, *t_arr.shape)
        ),
        coords=[
            ("elements", impurity_elements),
            ("rho_poloidal", rho_arr),
            ("theta", theta_arr),
            ("t", t_arr),
        ],
        dims=["elements", "rho_poloidal", "theta", "t"],
    )

    impurity_densities.data[0] = beryllium_density
    impurity_densities.data[1] = neon_density
    impurity_densities.data[2] = nickel_density

    deriv_only = False
    trim = True
    t_val = t_arr[0]

    t_arr = DataArray(data=t_arr, coords={"t": t_arr}, dims=["t"])

    example_bolometry_derivation = BolometryDerivation(
        flux_surfs,
        LoS_bolometry_data,
        t_arr,
        impurity_densities,
        frac_abunds,
        impurity_elements,
        electron_density,
        main_ion_power_loss,
        impurities_power_loss,
    )

    invalid_init_input_checks(example_bolometry_derivation, "flux_surfaces", flux_surfs)

    invalid_init_input_checks(
        example_bolometry_derivation, "LoS_bolometry_data", LoS_bolometry_data
    )

    invalid_init_input_checks(example_bolometry_derivation, "t_arr", t_arr)

    invalid_init_input_checks(
        example_bolometry_derivation, "impurity_densities", impurity_densities
    )

    invalid_init_input_checks(example_bolometry_derivation, "frac_abunds", frac_abunds)

    invalid_init_input_checks(
        example_bolometry_derivation, "impurity_elements", impurity_elements
    )

    invalid_init_input_checks(
        example_bolometry_derivation, "electron_density", electron_density
    )

    invalid_init_input_checks(
        example_bolometry_derivation, "main_ion_power_loss", main_ion_power_loss
    )

    invalid_init_input_checks(
        example_bolometry_derivation, "impurities_power_loss", impurities_power_loss
    )

    invalid_call_input_checks(example_bolometry_derivation, "deriv_only", deriv_only)

    invalid_call_input_checks(example_bolometry_derivation, "trim", trim)

    invalid_call_input_checks(example_bolometry_derivation, "t_val", t_val)
