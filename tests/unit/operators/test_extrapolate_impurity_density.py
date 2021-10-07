import copy
import unittest
from unittest.mock import MagicMock

import numpy as np
from xarray import DataArray

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.operators.extrapolate_impurity_density import ExtrapolateImpurityDensity
from ..test_equilibrium_single import equilibrium_dat_and_te


class Exception_Impurity_Density_Test_Case(unittest.TestCase):
    def __init__(
        self,
        impurity_density_sxr,
        electron_density,
        electron_temperature,
        truncation_threshold,
        t,
    ):
        self.impurity_density_sxr = impurity_density_sxr
        self.electron_density = electron_density
        self.electron_temperature = electron_temperature
        self.truncation_threshold = truncation_threshold
        self.t = t

        self.nominal_inputs = [
            self.impurity_density_sxr,
            self.electron_density,
            self.electron_temperature,
            self.truncation_threshold,
            self.t,
        ]

    def call_type_check(
        self,
        impurity_density_sxr=None,
        electron_density=None,
        electron_temperature=None,
        truncation_threshold=None,
        t=None,
    ):
        inputs = [
            impurity_density_sxr,
            electron_density,
            electron_temperature,
            truncation_threshold,
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
        t=None,
    ):
        inputs = [
            impurity_density_sxr,
            electron_density,
            electron_temperature,
            truncation_threshold,
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
            t,
        ) = inputs

        with self.assertRaises(ValueError):
            example_ = ExtrapolateImpurityDensity()
            example_(*inputs)


def test_extrapolate_impurity_density_call():
    base_rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])
    base_t = np.linspace(75.0, 80.0, 5)

    input_Te = DataArray(
        data=np.tile(np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]), (len(base_t), 1)).T,
        coords={"rho": base_rho_profile, "t": base_t},
        dims=["rho", "t"],
    )

    input_Ne = DataArray(
        data=np.tile(
            np.array([5.0e19, 4e19, 3.0e19, 2.0e19, 1.0e19]), (len(base_t), 1)
        ).T,
        coords={"rho": base_rho_profile, "t": base_t},
        dims=["rho", "t"],
    )

    valid_truncation_threshold = 1.0e3

    expanded_rho = np.linspace(base_rho_profile[0], base_rho_profile[-1], 100)

    input_Te = input_Te.interp(rho=expanded_rho, method="cubic")
    input_Ne = input_Ne.interp(rho=expanded_rho, method="cubic")

    R_arr = np.linspace(1.83, 3.9, 100)
    # R_arr = np.linspace(2.715, 3.05, 100)
    z_arr = np.linspace(-1.75, 2.0, 100)

    R_arr = DataArray(data=R_arr, coords={"R": R_arr}, dims=["R"])
    z_arr = DataArray(data=z_arr, coords={"z": z_arr}, dims=["z"])

    flux_surfs = FluxSurfaceCoordinates("poloidal")

    offset = MagicMock(return_value=0.02)
    equilib_dat, Te = equilibrium_dat_and_te()
    equilib = Equilibrium(equilib_dat, Te, sess=MagicMock(), offset_picker=offset)

    flux_surfs.set_equilibrium(equilib)

    example_extrapolate_impurity_density = ExtrapolateImpurityDensity()

    sxr_rho, sxr_theta = flux_surfs.convert_from_Rz(R_arr, z_arr, base_t)
    sxr_rho = np.abs(sxr_rho)
    sxr_rho = sxr_rho.transpose("R", "z", "t")

    sxr_theta = sxr_theta.transpose("R", "z", "t")
    rho_arr = np.linspace(0.0, 2.0, 50)
    theta_arr = np.linspace(np.min(sxr_theta), np.max(sxr_theta), 25)

    rho_arr = DataArray(data=rho_arr, coords={"rho": rho_arr}, dims=["rho"])
    theta_arr = DataArray(data=theta_arr, coords={"theta": theta_arr}, dims=["theta"])

    sxr_density_data = 25.0e15 * np.exp(-rho_arr)

    sxr_density_data = np.tile(sxr_density_data, (len(base_t), len(theta_arr), 1))

    sxr_density_data = np.transpose(sxr_density_data, [2, 1, 0])

    input_sxr_density = DataArray(
        data=sxr_density_data,
        coords={"rho": rho_arr, "theta": theta_arr, "t": base_t},
        dims=["rho", "theta", "t"],
    )

    rho_derived, theta_derived = flux_surfs.convert_from_Rz(R_arr, z_arr, base_t)
    rho_derived = np.abs(rho_derived)
    # rho_derived = rho_derived.where(rho_derived <= 1.0)

    rho_derived = rho_derived.transpose("R", "z", "t")
    theta_derived = theta_derived.transpose("R", "z", "t")

    input_sxr_density = input_sxr_density.indica.interp2d(
        {"rho": rho_derived, "theta": theta_derived}, method="cubic"
    )
    input_sxr_density = input_sxr_density.transpose("R", "z", "t")

    try:
        example_result, example_threshold_rho, t = example_extrapolate_impurity_density(
            input_sxr_density,
            input_Ne,
            input_Te,
            valid_truncation_threshold,
            flux_surfs,
        )
    except Exception as e:
        raise e

    assert np.all(t == base_t)

    example_extrapolate_test_case = Exception_Impurity_Density_Test_Case(
        input_sxr_density, input_Ne, input_Te, valid_truncation_threshold, base_t
    )

    # Invalid SXR derived density checks

    invalid_sxr_density = "invalid_test"

    example_extrapolate_test_case.call_type_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = DataArray(
        data=input_sxr_density[:, 0, :],
        coords={"R": R_arr, "t": base_t},
        dims=["R", "t"],
    )

    example_extrapolate_test_case.call_value_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = input_sxr_density.copy(deep=True)
    invalid_sxr_density = invalid_sxr_density * -1

    example_extrapolate_test_case.call_value_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = input_sxr_density.copy(deep=True)
    invalid_sxr_density = invalid_sxr_density * np.inf

    example_extrapolate_test_case.call_value_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = input_sxr_density.copy(deep=True)
    invalid_sxr_density = invalid_sxr_density * -np.inf

    example_extrapolate_test_case.call_value_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = input_sxr_density.copy(deep=True)
    invalid_sxr_density = invalid_sxr_density * np.nan

    example_extrapolate_test_case.call_value_check(
        impurity_density_sxr=invalid_sxr_density
    )

    # Invalid electron density checks

    invalid_Ne = "invalid_test"

    example_extrapolate_test_case.call_type_check(electron_density=invalid_Ne)

    invalid_Ne = DataArray(
        data=input_Ne[:, 0], coords={"rho": expanded_rho}, dims=["rho"]
    )

    example_extrapolate_test_case.call_value_check(electron_density=invalid_Ne)

    invalid_Ne = input_Ne.copy(deep=True)
    invalid_Ne = invalid_Ne * -1

    example_extrapolate_test_case.call_value_check(electron_density=invalid_Ne)

    invalid_Ne = input_Ne.copy(deep=True)
    invalid_Ne = invalid_Ne * np.inf

    example_extrapolate_test_case.call_value_check(electron_density=invalid_Ne)

    invalid_Ne = input_Ne.copy(deep=True)
    invalid_Ne = invalid_Ne * -np.inf

    example_extrapolate_test_case.call_value_check(electron_density=invalid_Ne)

    invalid_Ne = input_Ne.copy(deep=True)
    invalid_Ne = invalid_Ne * np.nan

    example_extrapolate_test_case.call_value_check(electron_density=invalid_Ne)

    # Invalid electron temperature checks

    invalid_Te = "invalid_test"

    example_extrapolate_test_case.call_type_check(electron_temperature=invalid_Te)

    invalid_Te = DataArray(
        data=input_Te[:, 0], coords={"rho": expanded_rho}, dims=["rho"]
    )

    example_extrapolate_test_case.call_value_check(electron_temperature=invalid_Te)

    invalid_Te = input_Te.copy(deep=True)
    invalid_Te = invalid_Te * 0

    example_extrapolate_test_case.call_value_check(electron_temperature=invalid_Te)

    invalid_Te = input_Te.copy(deep=True)
    invalid_Te = invalid_Te * -1

    example_extrapolate_test_case.call_value_check(electron_temperature=invalid_Te)

    invalid_Te = input_Te.copy(deep=True)
    invalid_Te = invalid_Te * np.inf

    example_extrapolate_test_case.call_value_check(electron_temperature=invalid_Te)

    invalid_Te = input_Te.copy(deep=True)
    invalid_Te = invalid_Te * -np.inf

    example_extrapolate_test_case.call_value_check(electron_temperature=invalid_Te)

    invalid_Te = input_Te.copy(deep=True)
    invalid_Te = invalid_Te * np.nan

    example_extrapolate_test_case.call_value_check(electron_temperature=invalid_Te)

    # Invalid truncation threshold check

    invalid_truncation_threshold = "invalid_test"

    example_extrapolate_test_case.call_type_check(
        truncation_threshold=invalid_truncation_threshold
    )

    invalid_truncation_threshold = valid_truncation_threshold * 0

    example_extrapolate_test_case.call_value_check(
        truncation_threshold=invalid_truncation_threshold
    )

    invalid_truncation_threshold = copy.deepcopy(valid_truncation_threshold)
    invalid_truncation_threshold = invalid_truncation_threshold * 0

    example_extrapolate_test_case.call_value_check(
        truncation_threshold=invalid_truncation_threshold
    )

    invalid_truncation_threshold = copy.deepcopy(valid_truncation_threshold)
    invalid_truncation_threshold = invalid_truncation_threshold * -1

    example_extrapolate_test_case.call_value_check(
        truncation_threshold=invalid_truncation_threshold
    )

    invalid_truncation_threshold = copy.deepcopy(valid_truncation_threshold)
    invalid_truncation_threshold = invalid_truncation_threshold * np.inf

    example_extrapolate_test_case.call_value_check(
        truncation_threshold=invalid_truncation_threshold
    )

    invalid_truncation_threshold = copy.deepcopy(valid_truncation_threshold)
    invalid_truncation_threshold = invalid_truncation_threshold * -np.inf

    example_extrapolate_test_case.call_value_check(
        truncation_threshold=invalid_truncation_threshold
    )

    invalid_truncation_threshold = copy.deepcopy(valid_truncation_threshold)
    invalid_truncation_threshold = invalid_truncation_threshold * np.nan

    example_extrapolate_test_case.call_value_check(
        truncation_threshold=invalid_truncation_threshold
    )
