import copy
from typing import get_args
from typing import Hashable
import unittest
from unittest.mock import MagicMock

import numpy as np
from xarray import DataArray

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.operators.centrifugal_asymmetry import AsymmetryParameter
from indica.operators.centrifugal_asymmetry import ToroidalRotation
from indica.operators.extrapolate_impurity_density import ExtrapolateImpurityDensity
from indica.readers import ADASReader
from ..test_equilibrium_single import equilibrium_dat_and_te

# import matplotlib.pyplot as plt


class Exception_Impurity_Density_Test_Case(unittest.TestCase):
    def __init__(
        self,
        impurity_density_sxr,
        electron_density,
        electron_temperature,
        truncation_threshold,
        flux_surfaces,
        frac_abund,
        impurity_elements,
        main_ion_power_loss,
        impurities_power_loss,
        t,
    ):
        self.impurity_density_sxr = impurity_density_sxr
        self.electron_density = electron_density
        self.electron_temperature = electron_temperature
        self.truncation_threshold = truncation_threshold
        self.flux_surfaces = flux_surfaces
        self.frac_abund = frac_abund
        self.impurity_elements = impurity_elements
        self.main_ion_power_loss = main_ion_power_loss
        self.impurities_power_loss = impurities_power_loss
        self.t = t

        self.nominal_inputs = [
            self.impurity_density_sxr,
            self.electron_density,
            self.electron_temperature,
            self.truncation_threshold,
            self.flux_surfaces,
            self.frac_abund,
            self.impurity_elements,
            self.main_ion_power_loss,
            self.impurities_power_loss,
            self.t,
        ]

    def call_type_check(
        self,
        impurity_density_sxr=None,
        electron_density=None,
        electron_temperature=None,
        truncation_threshold=None,
        flux_surfaces=None,
        frac_abund=None,
        impurity_elements=None,
        main_ion_power_loss=None,
        impurities_power_loss=None,
        t=None,
    ):
        inputs = [
            impurity_density_sxr,
            electron_density,
            electron_temperature,
            truncation_threshold,
            flux_surfaces,
            frac_abund,
            impurity_elements,
            main_ion_power_loss,
            impurities_power_loss,
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
            frac_abund,
            impurity_elements,
            main_ion_power_loss,
            impurities_power_loss,
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
        frac_abund=None,
        impurity_elements=None,
        main_ion_power_loss=None,
        impurities_power_loss=None,
        t=None,
    ):
        inputs = [
            impurity_density_sxr,
            electron_density,
            electron_temperature,
            truncation_threshold,
            flux_surfaces,
            frac_abund,
            impurity_elements,
            main_ion_power_loss,
            impurities_power_loss,
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
            frac_abund,
            impurity_elements,
            main_ion_power_loss,
            impurities_power_loss,
            t,
        ) = inputs

        with self.assertRaises(ValueError):
            example_ = ExtrapolateImpurityDensity()
            example_(*inputs)


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

    Returns
    -------
    F_z_tinf
        Fractional abundance of the ionisation stages of the element at t=infinity.
    """
    ADAS_file = ADASReader()

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    # t = np.linspace(75.0, 80.0, 5)
    # rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])

    # input_Te = DataArray(
    #     data=np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]),
    #     coords={"rho": rho_profile},
    #     dims=["rho"],
    # )

    # input_Ne = DataArray(
    #     data=np.array([5.0e19, 4e19, 3.0e19, 2.0e19, 1.0e19]),
    #     coords={"rho": rho_profile},
    #     dims=["rho"],
    # )

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
    rho_profile: np.ndarray,
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

    Returns
    -------
    power_loss
        Power loss of the element at t=infinity.
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
    base_rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])
    base_t = np.linspace(75.0, 80.0, 5)

    input_Te = np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3])
    input_Te = np.tile(input_Te, (len(base_t), 1)).T

    input_Te = DataArray(
        data=np.tile(np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]), (len(base_t), 1)).T,
        coords={"rho": base_rho_profile, "t": base_t},
        dims=["rho", "t"],
    )

    input_Ne = np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19])
    input_Ne = np.tile(input_Ne, (len(base_t), 1)).T

    input_Ne = DataArray(
        data=np.tile(
            np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19]), (len(base_t), 1)
        ).T,
        coords={"rho": base_rho_profile, "t": base_t},
        dims=["rho", "t"],
    )

    elements = ["be", "ne", "ni", "w"]

    input_Ti = np.array([2.0e3, 1.2e3, 0.5e3, 0.2e3, 0.1e3])
    input_Ti = np.tile(input_Ti, (len(elements), len(base_t), 1))
    input_Ti = np.swapaxes(input_Ti, 1, 2)

    input_Ti = DataArray(
        data=input_Ti,
        coords={"elements": elements, "rho": base_rho_profile, "t": base_t},
        dims=["elements", "rho", "t"],
    )

    toroidal_rotations = np.array([200.0e3, 170.0e3, 100.0e3, 30.0e3, 5.0e3])

    toroidal_rotations = np.tile(toroidal_rotations, (len(elements), len(base_t), 1))
    toroidal_rotations = np.swapaxes(toroidal_rotations, 1, 2)

    toroidal_rotations = DataArray(
        data=toroidal_rotations,
        coords=[("elements", elements), ("rho", base_rho_profile), ("t", base_t)],
        dims=["elements", "rho", "t"],
    )

    expanded_rho = np.linspace(base_rho_profile[0], base_rho_profile[-1], 41)

    rho_arr = expanded_rho
    theta_arr = np.linspace(-np.pi, np.pi, 21)

    rho_arr = DataArray(data=rho_arr, coords={"rho": rho_arr}, dims=["rho"])
    theta_arr = DataArray(data=theta_arr, coords={"theta": theta_arr}, dims=["theta"])

    flux_surfs = FluxSurfaceCoordinates("poloidal")

    offset = MagicMock(return_value=0.02)
    equilib_dat, Te = equilibrium_dat_and_te()
    equilib = Equilibrium(equilib_dat, Te, sess=MagicMock(), offset_picker=offset)

    flux_surfs.set_equilibrium(equilib)

    R_derived, _ = flux_surfs.convert_to_Rz(
        DataArray(expanded_rho, {"rho": expanded_rho}, dims=["rho"]), theta_arr, base_t
    )

    R_derived = R_derived.transpose("rho", "theta", "t")

    R_lfs_values = R_derived.interp(theta=0, method="linear")

    input_Te = input_Te.interp(rho=expanded_rho, method="cubic")
    input_Ne = input_Ne.interp(rho=expanded_rho, method="cubic")
    input_Ti = input_Ti.interp(rho=expanded_rho, method="cubic")
    toroidal_rotations = toroidal_rotations.interp(rho=expanded_rho, method="cubic")

    toroidal_rotations /= R_lfs_values.data  # re-scale from velocity to frequency

    valid_truncation_threshold = 1.0e3

    Zeff = DataArray(
        data=1.85 * np.ones((*expanded_rho.shape, len(base_t))),
        coords=[("rho", expanded_rho), ("t", base_t)],
        dims=["rho", "t"],
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


def sxr_data_setup():
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
    ) = input_data_setup()

    R_arr = np.linspace(1.83, 3.9, 40)
    z_arr = np.linspace(-1.75, 2.0, 40)

    R_arr = DataArray(data=R_arr, coords={"R": R_arr}, dims=["R"])
    z_arr = DataArray(data=z_arr, coords={"z": z_arr}, dims=["z"])

    sxr_density_data = 25.0e15 * np.exp(-rho_arr)

    sxr_density_data = np.tile(sxr_density_data, (len(base_t), len(theta_arr), 1))

    sxr_density_data = np.transpose(sxr_density_data, [2, 1, 0])

    input_sxr_density = DataArray(
        data=sxr_density_data,
        coords={"rho": rho_arr, "theta": theta_arr, "t": base_t},
        dims=["rho", "theta", "t"],
    )

    example_asymmetry_obj = AsymmetryParameter()
    example_asymmetry = example_asymmetry_obj(
        toroidal_rotations.copy(deep=True), input_Ti, "d", "w", Zeff, input_Te
    )
    example_asymmetry = example_asymmetry.transpose("rho", "t")

    rho_derived, theta_derived = flux_surfs.convert_from_Rz(R_arr, z_arr, base_t)
    rho_derived = abs(rho_derived)

    rho_derived = rho_derived.transpose("R", "z", "t")
    theta_derived = theta_derived.transpose("R", "z", "t")

    input_sxr_density_lfs = input_sxr_density.sel(theta=0)

    asymmetry_modifier = np.exp(
        example_asymmetry * (R_derived ** 2 - R_lfs_values ** 2)
    )
    asymmetry_modifier = asymmetry_modifier.transpose("rho", "theta", "t")

    input_sxr_density_asym = input_sxr_density_lfs * asymmetry_modifier
    input_sxr_density_asym = input_sxr_density_asym.transpose("rho", "theta", "t")

    input_sxr_density_asym = input_sxr_density_asym.indica.interp2d(
        {"rho": rho_derived, "theta": theta_derived}, method="linear"
    )
    input_sxr_density_asym = input_sxr_density_asym.fillna(0.0)
    input_sxr_density_asym = input_sxr_density_asym.transpose("R", "z", "t")

    return (input_sxr_density_asym, R_arr)


def fitting_data_setup():
    initial_data = input_data_setup()

    base_t = initial_data[9]
    input_Te = initial_data[1]
    input_Ne = initial_data[0]
    expanded_rho = initial_data[4].data

    example_frac_abund = fractional_abundance_setup(
        "w", base_t, input_Te.isel(t=0), input_Ne.isel(t=0)
    )

    main_ion_power_loss = power_loss_setup(
        "h", base_t, expanded_rho, input_Te, input_Ne
    )

    main_ion_power_loss = main_ion_power_loss.sum(dim="ion_charges")

    impurity_power_loss = power_loss_setup(
        "w", base_t, expanded_rho, input_Te, input_Ne
    )

    impurity_power_loss = impurity_power_loss.assign_coords(t=("t", base_t))
    impurity_power_loss = impurity_power_loss.sum(dim="ion_charges")

    impurity_power_loss = DataArray(
        data=impurity_power_loss.data[np.newaxis, :],
        coords=dict(**{"elements": ["w"]}, **impurity_power_loss.coords),
        dims=["elements", *impurity_power_loss.dims],
    )

    # Interpolation only for diagnostic purposes, will be deleted in the the future
    # input_sxr_density_asym = input_sxr_density_asym.interp(
    #     R=DataArray(
    #         np.linspace(1.83, 3.9, 100), {"R": np.linspace(1.83, 3.9, 100)}, ["R"]
    #     ),
    #     method="linear"
    # )
    # input_sxr_density_asym = input_sxr_density_asym.interp(
    #     z=DataArray(
    #         np.linspace(-1.75, 2.0, 100), {"z": np.linspace(-1.75, 2.0, 100)}, ["z"]
    #     ),
    #     method="linear"
    # )

    # input_sxr_density_asym.isel(t=0).transpose("z", "R").plot()
    # plt.axis("scaled")
    # plt.tight_layout()
    # plt.show()

    return (example_frac_abund, main_ion_power_loss, impurity_power_loss)


def invalid_input_tests(
    test_case: Exception_Impurity_Density_Test_Case,
    nominal_input_name: str,
    nominal_input,
    zero_check: bool = False,
):
    if not isinstance(nominal_input, Hashable):
        invalid_input = 1.0
        test_case.call_type_check(**{nominal_input_name: invalid_input})
    elif isinstance(nominal_input, get_args(LabeledArray)):
        # Type ignore due to mypy complaining about redefinition of invalid_input

        invalid_input = "test"  # type:ignore
        test_case.call_type_check(**{nominal_input_name: invalid_input})

        invalid_input = copy.deepcopy(nominal_input)  # type:ignore
        invalid_input *= -1
        test_case.call_value_check(**{nominal_input_name: invalid_input})

        invalid_input = copy.deepcopy(nominal_input)  # type:ignore
        invalid_input *= np.inf
        test_case.call_value_check(**{nominal_input_name: invalid_input})

        invalid_input = copy.deepcopy(nominal_input)  # type:ignore
        invalid_input *= -np.inf
        test_case.call_value_check(**{nominal_input_name: invalid_input})

        invalid_input = copy.deepcopy(nominal_input)  # type:ignore
        invalid_input *= np.nan
        test_case.call_value_check(**{nominal_input_name: invalid_input})

        if zero_check:
            invalid_input = copy.deepcopy(nominal_input)  # type:ignore
            invalid_input *= 0
            test_case.call_value_check(**{nominal_input_name: invalid_input})
    elif isinstance(nominal_input, (np.ndarray, DataArray)):
        invalid_input = copy.deepcopy(nominal_input[0])  # type:ignore
        test_case.call_value_check(**{nominal_input_name: invalid_input})


def test_extrapolate_impurity_density_call():
    initial_data = input_data_setup()
    sxr_data = sxr_data_setup()
    fitting_data = fitting_data_setup()

    input_Ne = initial_data[0]
    input_Te = initial_data[1]
    input_Ti = initial_data[2]
    valid_truncation_threshold = initial_data[7]
    flux_surfs = initial_data[6]
    base_t = initial_data[9]
    Zeff = initial_data[8]
    elements = initial_data[12]
    impurity_sxr_density_asym = sxr_data[0]
    example_frac_abund = fitting_data[0]
    main_ion_power_loss = fitting_data[1]
    impurity_power_loss = fitting_data[2]

    example_extrapolate_impurity_density = ExtrapolateImpurityDensity()

    try:
        (
            example_result,
            example_derived_asymmetry,
            t,
        ) = example_extrapolate_impurity_density(
            impurity_sxr_density_asym,
            input_Ne,
            input_Te,
            valid_truncation_threshold,
            flux_surfs,
            example_frac_abund,
            ["w"],
            main_ion_power_loss,
            impurity_power_loss,
        )
    except Exception as e:
        raise e

    example_derived_asymmetry = DataArray(
        data=example_derived_asymmetry.data[np.newaxis, :],
        coords={
            "elements": ["w"],
            "rho": example_derived_asymmetry.coords["rho"].data,
            "t": example_derived_asymmetry.coords["t"].data,
        },
        dims=["elements", "rho", "t"],
    )

    assert np.all(t == base_t)

    derived_toroidal_rotations = ToroidalRotation()
    derived_toroidal_rotations = derived_toroidal_rotations(
        example_derived_asymmetry, input_Ti, "d", "w", Zeff, input_Te
    )

    example_extrapolate_test_case = Exception_Impurity_Density_Test_Case(
        impurity_sxr_density_asym,
        input_Ne,
        input_Te,
        valid_truncation_threshold,
        flux_surfs,
        example_frac_abund,
        elements,
        main_ion_power_loss,
        impurity_power_loss,
        base_t,
    )

    # Invalid SXR derived density checks

    invalid_input_tests(
        example_extrapolate_test_case,
        "impurity_density_sxr",
        impurity_sxr_density_asym,
    )

    # Invalid electron density checks

    invalid_input_tests(example_extrapolate_test_case, "electron_density", input_Ne)

    # Invalid electron temperature checks

    invalid_input_tests(
        example_extrapolate_test_case, "electron_temperature", input_Te, zero_check=True
    )

    # Invalid truncation threshold check

    invalid_input_tests(
        example_extrapolate_test_case,
        "truncation_threshold",
        valid_truncation_threshold,
        zero_check=True,
    )
