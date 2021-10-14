import copy
import unittest
from unittest.mock import MagicMock

import numpy as np
from xarray import DataArray

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.operators.centrifugal_asymmetry import AsymmetryParameter
from indica.operators.extrapolate_impurity_density import ExtrapolateImpurityDensity
from ..test_equilibrium_single import equilibrium_dat_and_te

# import matplotlib.pyplot as plt


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

    expanded_rho = np.linspace(base_rho_profile[0], base_rho_profile[-1], 40)

    input_Te = input_Te.interp(rho=expanded_rho, method="linear")
    input_Ne = input_Ne.interp(rho=expanded_rho, method="linear")

    R_arr = np.linspace(1.83, 3.9, 40)
    z_arr = np.linspace(-1.75, 2.0, 40)

    R_arr = DataArray(data=R_arr, coords={"R": R_arr}, dims=["R"])
    z_arr = DataArray(data=z_arr, coords={"z": z_arr}, dims=["z"])

    flux_surfs = FluxSurfaceCoordinates("poloidal")

    offset = MagicMock(return_value=0.02)
    equilib_dat, Te = equilibrium_dat_and_te()
    equilib = Equilibrium(equilib_dat, Te, sess=MagicMock(), offset_picker=offset)

    flux_surfs.set_equilibrium(equilib)

    example_extrapolate_impurity_density = ExtrapolateImpurityDensity()

    sxr_rho, sxr_theta = flux_surfs.convert_from_Rz(R_arr, z_arr, base_t)
    sxr_rho = sxr_rho.transpose("R", "z", "t")

    sxr_theta = sxr_theta.transpose("R", "z", "t")
    rho_arr = np.linspace(0.0, 2.0, 40)
    theta_arr = np.linspace(np.min(sxr_theta), np.max(sxr_theta), 9)

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

    elements = ["be", "ne", "ni", "w"]

    input_Ti = np.array([2.0e3, 1.2e3, 0.5e3, 0.2e3, 0.1e3])
    input_Ti = np.tile(input_Ti, (len(elements), len(base_t), 1))
    input_Ti = np.swapaxes(input_Ti, 1, 2)

    input_Ti = DataArray(
        data=input_Ti,
        coords={"elements": elements, "rho": base_rho_profile, "t": base_t},
        dims=["elements", "rho", "t"],
    )
    input_Ti = input_Ti.interp({"rho": expanded_rho}, method="linear")

    toroidal_rotations = np.array([200.0e3, 170.0e3, 100.0e3, 30.0e3, 5.0e3])

    xr_rho_profile = DataArray(
        data=base_rho_profile, coords={"rho": base_rho_profile}, dims=["rho"]
    )

    R_lfs_values, _ = equilib.R_lfs(xr_rho_profile)

    toroidal_rotations /= R_lfs_values.data[0, :]  # re-scale from velocity to frequency

    toroidal_rotations = np.tile(toroidal_rotations, (len(elements), len(base_t), 1))
    toroidal_rotations = np.swapaxes(toroidal_rotations, 1, 2)

    toroidal_rotations = DataArray(
        data=toroidal_rotations,
        coords=[("elements", elements), ("rho", base_rho_profile), ("t", base_t)],
        dims=["elements", "rho", "t"],
    )

    toroidal_rotations = toroidal_rotations.interp(
        {"rho": expanded_rho}, method="linear"
    )

    Zeff = DataArray(
        data=1.85 * np.ones((*base_rho_profile.shape, len(base_t))),
        coords=[("rho", base_rho_profile), ("t", base_t)],
        dims=["rho", "t"],
    )
    Zeff = Zeff.interp({"rho": expanded_rho}, method="linear")

    example_asymmetry_obj = AsymmetryParameter()
    example_asymmetry = example_asymmetry_obj(
        toroidal_rotations.copy(deep=True), input_Ti, "d", "w", Zeff, input_Te
    )

    R_lfs_values = R_lfs_values.interp({"rho": expanded_rho}, method="linear")
    R_lfs_values = R_lfs_values.interp({"t": base_t}, method="linear")

    # theta_arr = np.linspace(np.min(input_sxr_density.coords["theta"]),
    # np.max(input_sxr_density.coords["theta"]), 100)

    # input_sxr_density = input_sxr_density.interp(theta=theta_arr, method="linear")

    # input_sxr_density["theta"] = (input_sxr_density["theta"] % (2.0 * np.pi))
    # input_sxr_density = input_sxr_density.sortby("theta")

    # input_sxr_density.isel(t=0).plot.pcolormesh("theta", "rho",
    # subplot_kws=dict(projection="polar"))
    # plt.show()

    rho_derived, theta_derived = flux_surfs.convert_from_Rz(R_arr, z_arr, base_t)
    rho_derived = np.abs(rho_derived)

    rho_derived = rho_derived.transpose("R", "z", "t")
    theta_derived = theta_derived.transpose("R", "z", "t")

    input_sxr_density_Rz = input_sxr_density.indica.interp2d(
        {"rho": rho_derived, "theta": theta_derived}, method="linear"
    )
    input_sxr_density_Rz = input_sxr_density_Rz.transpose("R", "z", "t")

    input_sxr_density_lfs = input_sxr_density_Rz.indica.interp2d(
        R=R_lfs_values,
        z=R_lfs_values.coords["z"],
        method="linear",
        assume_sorted=True,
    )

    R_derived, z_derived = flux_surfs.convert_to_Rz(
        DataArray(expanded_rho, {"rho": expanded_rho}, dims=["rho"]), theta_arr, base_t
    )

    R_lfs_values.transpose("rho", "t")
    R_derived = R_derived.transpose("rho", "theta", "t")
    example_asymmetry = example_asymmetry.transpose("rho", "t")

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

    # Interpolation only for diagnostic purposes, will be deleted in the the future
    # input_sxr_density_asym = input_sxr_density_asym.interp(
    #     R=DataArray(
    #         np.linspace(1.83, 3.9, 400), {"R": np.linspace(1.83, 3.9, 400)}, ["R"]
    #     ),
    #     method="linear"
    # )
    # input_sxr_density_asym = input_sxr_density_asym.interp(
    #     z=DataArray(
    #         np.linspace(-1.75, 2.0, 400), {"z": np.linspace(-1.75, 2.0, 400)}, ["z"]
    #     ),
    #     method="linear"
    # )

    # input_sxr_density_asym.isel(t=0).transpose("z", "R").plot()
    # plt.axis("scaled")
    # plt.tight_layout()
    # plt.show()

    try:
        example_result, example_threshold_rho, t = example_extrapolate_impurity_density(
            input_sxr_density_asym,
            input_Ne,
            input_Te,
            valid_truncation_threshold,
            flux_surfs,
        )
    except Exception as e:
        raise e

    assert np.all(t == base_t)

    example_extrapolate_test_case = Exception_Impurity_Density_Test_Case(
        input_sxr_density_asym, input_Ne, input_Te, valid_truncation_threshold, base_t
    )

    # Invalid SXR derived density checks

    invalid_sxr_density = "invalid_test"

    example_extrapolate_test_case.call_type_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = DataArray(
        data=input_sxr_density_asym[:, 0, :],
        coords={"R": R_arr, "t": base_t},
        dims=["R", "t"],
    )

    example_extrapolate_test_case.call_value_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = input_sxr_density_asym.copy(deep=True)
    invalid_sxr_density = invalid_sxr_density * -1

    example_extrapolate_test_case.call_value_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = input_sxr_density_asym.copy(deep=True)
    invalid_sxr_density = invalid_sxr_density * np.inf

    example_extrapolate_test_case.call_value_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = input_sxr_density_asym.copy(deep=True)
    invalid_sxr_density = invalid_sxr_density * -np.inf

    example_extrapolate_test_case.call_value_check(
        impurity_density_sxr=invalid_sxr_density
    )

    invalid_sxr_density = input_sxr_density_asym.copy(deep=True)
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
