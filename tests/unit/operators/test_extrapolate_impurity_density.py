import unittest

import numpy as np
from xarray.core.dataarray import DataArray

from indica.operators.extrapolate_impurity_density import ExtrapolateImpurityDensity


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

    input_Te = input_Te.interp(rho=expanded_rho)
    input_Ne = input_Ne.interp(rho=expanded_rho)

    input_sxr_density = DataArray(
        data=np.tile(12.0 * np.exp(-expanded_rho), (len(base_t), 1)).T,
        coords={"rho": expanded_rho, "t": base_t},
        dims=["rho", "t"],
    )

    example_extrapolate_impurity_density = ExtrapolateImpurityDensity()

    try:
        example_result, t = example_extrapolate_impurity_density(
            input_sxr_density, input_Ne, input_Te, valid_truncation_threshold
        )
    except Exception as e:
        raise e
