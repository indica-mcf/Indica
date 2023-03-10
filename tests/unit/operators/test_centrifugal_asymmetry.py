from typing import Dict
from typing import Union
import unittest
from unittest.mock import MagicMock

import numpy as np
from xarray.core.common import zeros_like
from xarray.core.dataarray import DataArray

from indica.equilibrium import Equilibrium
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.centrifugal_asymmetry import AsymmetryParameter
from indica.operators.centrifugal_asymmetry import ToroidalRotation
from indica.readers import OpenADASReader
from ..test_equilibrium_single import equilibrium_dat_and_te


class Exception_Asymmetry_Parameter_Test_Case(unittest.TestCase):
    """Test case for testing type and value errors in AsymmetryParameter call."""

    def __init__(
        self,
        toroidal_rotations,
        ion_temperature,
        main_ion,
        impurity,
        Zeff,
        electron_temp,
    ):
        self.toroidal_rotations = toroidal_rotations
        self.ion_temperature = ion_temperature
        self.main_ion = main_ion
        self.impurity = impurity
        self.Zeff = Zeff
        self.electron_temp = electron_temp

        self.nominal_inputs = [
            self.toroidal_rotations,
            self.ion_temperature,
            self.main_ion,
            self.impurity,
            self.Zeff,
            self.electron_temp,
        ]

    def call_type_check(
        self,
        toroidal_rotations=None,
        ion_temperature=None,
        main_ion=None,
        impurity=None,
        Zeff=None,
        electron_temp=None,
    ):
        """Test TypeError for AsymmetryParameter call."""
        inputs = [
            toroidal_rotations,
            ion_temperature,
            main_ion,
            impurity,
            Zeff,
            electron_temp,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            toroidal_rotations,
            ion_temperature,
            main_ion,
            impurity,
            Zeff,
            electron_temp,
        ) = inputs

        with self.assertRaises(TypeError):
            example_ = AsymmetryParameter()
            example_(*inputs)

    def call_value_check(
        self,
        toroidal_rotations=None,
        ion_temperature=None,
        main_ion=None,
        impurity=None,
        Zeff=None,
        electron_temp=None,
    ):
        """Test ValueError for AsymmetryParameter call."""
        inputs = [
            toroidal_rotations,
            ion_temperature,
            main_ion,
            impurity,
            Zeff,
            electron_temp,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            toroidal_rotations,
            ion_temperature,
            main_ion,
            impurity,
            Zeff,
            electron_temp,
        ) = inputs

        with self.assertRaises(ValueError):
            example_ = AsymmetryParameter()
            example_(*inputs)


class Exception_Toroidal_Rotation_Test_Case(unittest.TestCase):
    """Test case for testing type and value errors in AsymmetryParameter call."""

    def __init__(
        self,
        asymmetry_parameters,
        ion_temperature,
        main_ion,
        impurity,
        Zeff,
        electron_temp,
    ):
        self.asymmetry_parameters = asymmetry_parameters
        self.ion_temperature = ion_temperature
        self.main_ion = main_ion
        self.impurity = impurity
        self.Zeff = Zeff
        self.electron_temp = electron_temp

        self.nominal_inputs = [
            self.asymmetry_parameters,
            self.ion_temperature,
            self.main_ion,
            self.impurity,
            self.Zeff,
            self.electron_temp,
        ]

    def call_type_check(
        self,
        asymmetry_parameters=None,
        ion_temperature=None,
        main_ion=None,
        impurity=None,
        Zeff=None,
        electron_temp=None,
    ):
        """Test TypeError for ToroidalRotation call."""
        inputs = [
            asymmetry_parameters,
            ion_temperature,
            main_ion,
            impurity,
            Zeff,
            electron_temp,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            asymmetry_parameters,
            ion_temperature,
            main_ion,
            impurity,
            Zeff,
            electron_temp,
        ) = inputs

        with self.assertRaises(TypeError):
            example_ = ToroidalRotation()
            example_(*inputs)

    def call_value_check(
        self,
        asymmetry_parameters=None,
        ion_temperature=None,
        main_ion=None,
        impurity=None,
        Zeff=None,
        electron_temp=None,
    ):
        """Test ValueError for ToroidalRotation call."""
        inputs = [
            asymmetry_parameters,
            ion_temperature,
            main_ion,
            impurity,
            Zeff,
            electron_temp,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            asymmetry_parameters,
            ion_temperature,
            main_ion,
            impurity,
            Zeff,
            electron_temp,
        ) = inputs

        with self.assertRaises(ValueError):
            example_ = ToroidalRotation()
            example_(*inputs)


def fractional_abundance_setup(element: str, t: LabeledArray) -> DataArray:
    """Calculate and output Fractional abundance at t=infinity for calculating
    the mean charge in test_impurity_concentration()

    Parameters
    ----------
    element
        String of the symbol of the element per ADAS notation
        e.g be for Beryllium
    t
        Times at which to define input_Ne and input_Te (also used for the output)

    Returns
    -------
    F_z_tinf
        Fractional abundance of the ionisation stages of the element at t=infinity.
    """
    ADAS_file = OpenADASReader()

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    t = np.linspace(75.0, 80.0, 5)
    rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0], dtype=float)

    input_Te = DataArray(
        data=np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]),
        coords={"rho_poloidal": rho_profile},
        dims=["rho_poloidal"],
    )

    input_Ne = DataArray(
        data=np.array([5.0e19, 4e19, 3.0e19, 2.0e19, 1.0e19]),
        coords={"rho_poloidal": rho_profile},
        dims=["rho_poloidal"],
    )

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


def input_checking(
    var_name: str,
    test_case: Union[
        Exception_Asymmetry_Parameter_Test_Case, Exception_Toroidal_Rotation_Test_Case
    ],
    nominal_inputs: Dict,
    dim_check: bool = True,
):
    """Function to test inputs of type DataArray

    Parameters
    ----------
    var_name
        Name of variable to input
    test_case
        unittest.TestCase which will check that type and assertion errors are raised.
    nominal_inputs
        Dict of nominal inputs
    dim_check
        Boolean to signify whether or not to check errors from
        incorrect number of dimensions.
    """

    erroneous_input = {var_name: nominal_inputs[var_name].data}
    test_case.call_type_check(**erroneous_input)

    # all inputs must be positive except asymmetry_parameters
    if var_name != "asymmetry_parameters":
        erroneous_input = {var_name: nominal_inputs[var_name] * -1}
        test_case.call_value_check(**erroneous_input)

    erroneous_input = {var_name: nominal_inputs[var_name] * np.inf}
    test_case.call_value_check(**erroneous_input)

    erroneous_input = {var_name: nominal_inputs[var_name] * -np.inf}
    test_case.call_value_check(**erroneous_input)

    erroneous_input = {var_name: nominal_inputs[var_name] * np.nan}
    test_case.call_value_check(**erroneous_input)

    erroneous_input = {var_name: nominal_inputs[var_name] * np.nan}
    test_case.call_value_check(**erroneous_input)

    if dim_check:
        erroneous_input = {var_name: nominal_inputs[var_name].expand_dims("blank")}
        test_case.call_value_check(**erroneous_input)


def test_centrifugal_asymmetry():
    """Test AsymmetryParameter.__call__ and ToroidalRotation.__call__."""
    example_asymmetry = AsymmetryParameter()

    t = np.linspace(75.0, 80.0, 5)
    rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])

    example_equilib_dat, example_Te = equilibrium_dat_and_te()

    electron_temp = DataArray(
        data=np.tile(np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]), (len(t), 1)).T,
        coords=[("rho_poloidal", rho_profile), ("t", t)],
        dims=["rho_poloidal", "t"],
    )

    offset = MagicMock(return_value=0.02)

    example_equilibrium = Equilibrium(
        example_equilib_dat,
        example_Te,
        sess=MagicMock(),
        offset_picker=offset,
    )

    xr_rho_profile = DataArray(
        data=rho_profile, coords={"rho_poloidal": rho_profile}, dims=["rho_poloidal"]
    )

    R_lfs_values, _ = example_equilibrium.R_lfs(xr_rho_profile)

    # be, ne, ni, w
    elements = ["be", "ne", "ni", "w"]

    toroidal_rotations = np.array([200.0e3, 170.0e3, 100.0e3, 30.0e3, 5.0e3])

    toroidal_rotations /= R_lfs_values.data[0, :]

    toroidal_rotations = np.tile(toroidal_rotations, (len(elements), len(t), 1))
    toroidal_rotations = np.swapaxes(toroidal_rotations, 1, 2)

    toroidal_rotations = DataArray(
        data=toroidal_rotations,
        coords=[("element", elements), ("rho_poloidal", rho_profile), ("t", t)],
        dims=["element", "rho_poloidal", "t"],
    )

    ion_temperature = np.array([2.0e3, 1.2e3, 0.5e3, 0.2e3, 0.1e3])
    ion_temperature = np.tile(ion_temperature, (len(elements), len(t), 1))
    ion_temperature = np.swapaxes(ion_temperature, 1, 2)

    ion_temperature = DataArray(
        data=ion_temperature,
        coords=[("element", elements), ("rho_poloidal", rho_profile), ("t", t)],
        dims=["element", "rho_poloidal", "t"],
    )

    Zeff = DataArray(
        data=1.85 * np.ones((*rho_profile.shape, len(t))),
        coords=[("rho_poloidal", rho_profile), ("t", t)],
        dims=["rho_poloidal", "t"],
    )

    main_ion = "d"
    impurity = "be"

    # toroidal_rotations has to be deepcopied otherwise it gets modified when
    # passed to example_asymmetry.__call__
    nominal_inputs = {
        "toroidal_rotations": toroidal_rotations.copy(deep=True),
        "ion_temperature": ion_temperature,
        "main_ion": main_ion,
        "impurity": impurity,
        "Zeff": Zeff,
        "electron_temp": electron_temp,
    }

    # Checking outputs of AsymmetryParameter() and ToroidalRotation()
    asymmetry_parameters = zeros_like(toroidal_rotations)

    try:
        asymmetry_parameters.data[0] = example_asymmetry(**nominal_inputs)
    except Exception as e:
        raise e

    nominal_inputs["impurity"] = "ne"

    try:
        asymmetry_parameters.data[1] = example_asymmetry(**nominal_inputs)
    except Exception as e:
        raise e

    nominal_inputs["impurity"] = "ni"

    try:
        asymmetry_parameters.data[2] = example_asymmetry(**nominal_inputs)
    except Exception as e:
        raise e

    nominal_inputs["impurity"] = "w"

    try:
        asymmetry_parameters.data[3] = example_asymmetry(**nominal_inputs)
    except Exception as e:
        raise e

    example_toroidal_rotation = ToroidalRotation()

    del nominal_inputs["toroidal_rotations"]

    nominal_inputs["asymmetry_parameters"] = asymmetry_parameters

    try:
        output_toroidal_rotation = example_toroidal_rotation(**nominal_inputs)
    except Exception as e:
        raise e

    expected_toroidal_rotation = toroidal_rotations.sel(element="w")

    assert np.allclose(output_toroidal_rotation, expected_toroidal_rotation)

    # Checking inputs for AsymmetryParameter
    nominal_inputs = {
        "toroidal_rotations": toroidal_rotations,
        "ion_temperature": ion_temperature,
        "main_ion": main_ion,
        "impurity": impurity,
        "Zeff": Zeff,
        "electron_temp": electron_temp,
    }

    test_case_asymmetry = Exception_Asymmetry_Parameter_Test_Case(**nominal_inputs)

    for k, v in nominal_inputs.items():
        if k == "impurity" or k == "main_ion":
            continue

        input_checking(k, test_case_asymmetry, nominal_inputs)

    erroneous_input = {"impurity": 4}
    test_case_asymmetry.call_type_check(**erroneous_input)

    erroneous_input = {"impurity": "u"}
    test_case_asymmetry.call_value_check(**erroneous_input)

    erroneous_input = {"main_ion": 4}
    test_case_asymmetry.call_type_check(**erroneous_input)

    erroneous_input = {"main_ion": "u"}
    test_case_asymmetry.call_value_check(**erroneous_input)

    # Checking inputs for ToroidalRotation
    nominal_inputs = {
        "asymmetry_parameters": asymmetry_parameters,
        "ion_temperature": ion_temperature,
        "main_ion": main_ion,
        "impurity": impurity,
        "Zeff": Zeff,
        "electron_temp": electron_temp,
    }

    test_case_toroidal = Exception_Toroidal_Rotation_Test_Case(**nominal_inputs)

    for k, v in nominal_inputs.items():
        if k == "impurity" or k == "main_ion":
            continue

        input_checking(k, test_case_toroidal, nominal_inputs)

    erroneous_input = {"impurity": 4}
    test_case_toroidal.call_type_check(**erroneous_input)

    erroneous_input = {"impurity": "u"}
    test_case_toroidal.call_value_check(**erroneous_input)

    erroneous_input = {"main_ion": 4}
    test_case_toroidal.call_type_check(**erroneous_input)

    erroneous_input = {"main_ion": "u"}
    test_case_toroidal.call_value_check(**erroneous_input)
