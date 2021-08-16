from typing import Any
from typing import get_args
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from xarray.core.dataarray import DataArray

from indica.datatypes import ELEMENTS_BY_ATOMIC_NUMBER
from indica.datatypes import ELEMENTS_BY_MASS
from indica.datatypes import ELEMENTS_BY_SYMBOL
from indica.numpy_typing import LabeledArray
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType


class ToroidalRotation(Operator):
    """Calculate the toroidal rotation from asymmetry parameter.

    Parameters
    ----------

    Returns
    -------
    toroidal_rotation
        xarray.DataArray containing toroidal rotation for a given impurity element

    Methods
    -------
    input_check(
        var_name,
        var_to_check,
        var_type,
        ndim_to_check,
        greater_than_or_equal_zero
    )
        Checks the inputted var_to_check to ensure that it is valid.

    __call__(
        asymmetry_parameters,
        ion_temperature,
        main_ion,
        impurity,
        Zeff,
        electron_temp,
    )
        Calculates the toroidal_rotation from the asymmetry_parameter.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("toroidal_rotation", "plasma"),
    ]

    def __init__(self, sess: session.Session = session.global_session):
        super().__init__(sess=sess)

    def return_types(self, *args: DataType) -> Tuple[Any, ...]:
        return super().return_types(*args)

    def input_check(
        self,
        var_name: str,
        var_to_check,
        var_type: type,
        ndim_to_check: Optional[int] = None,
        greater_than_or_equal_zero: Optional[bool] = None,
    ):
        """Check validity of inputted variable - type check and
        various value checks(no infinities, greather than (or equal to) 0 or NaNs)

        Parameters
        ----------
        var_name
            Name of variable to check.
        var_to_check
            Variable to check.
        var_type
            Type to check variable against, eg. DataArray
        ndim_to_check
            Integer to check the number of dimensions of the variable.
        greater_than_or_equal_zero
            Boolean to check values in variable > 0 or >= 0.
        """
        try:
            assert isinstance(var_to_check, var_type)
        except AssertionError:
            raise TypeError(f"{var_name} must be of type {var_type}.")

        if greater_than_or_equal_zero is not None:
            try:
                if not greater_than_or_equal_zero:
                    # Mypy will ignore this line since even though var_to_check
                    # is type checked earlier it still doesn't explicitly
                    # know what type var_to_check
                    assert np.all(var_to_check > 0)  # type: ignore
                else:
                    # Mypy will ignore this line since even though var_to_check
                    # is type checked earlier it still doesn't explicitly
                    # know what type var_to_check
                    assert np.all(var_to_check >= 0)  # type: ignore
            except AssertionError:
                raise ValueError(f"Cannot have any negative values in {var_name}")

        if var_type in get_args(LabeledArray):
            try:
                assert np.all(var_to_check != np.nan)
            except AssertionError:
                raise ValueError(f"{var_name} cannot contain any NaNs.")

            try:
                assert np.all(np.abs(var_to_check) != np.inf)
            except AssertionError:
                raise ValueError(f"{var_name} cannot contain any infinities.")

        if ndim_to_check is not None and var_type in [np.ndarray, DataArray]:
            try:
                # Mypy will ignore this line since even though var_to_check
                # is type checked earlier it still doesn't explicitly
                # know what type var_to_check
                assert var_to_check.ndim == ndim_to_check  # type: ignore
            except AssertionError:
                raise ValueError(f"{var_name} must have {ndim_to_check} dimensions.")

    def __call__(  # type: ignore
        self,
        asymmetry_parameters: DataArray,
        ion_temperature: DataArray,
        main_ion: str,
        impurity: str,
        Zeff: DataArray,
        electron_temp: DataArray,
    ):
        """Calculates the toroidal rotation from the asymmetry parameter.

        Parameters
        ----------
        asymmetry_parameters
            xarray.DataArray containing asymmetry parameters data.
        ion_temperature
            xarray.DataArray containing ion temperature data.
        main_ion
            Element symbol of main ion.
        impurity
            Element symbol of chosen impurity element.
        Zeff
            xarray.DataArray containing Z-effective data from diagnostics.
        electron_temp
            xarray.DataArray containing electron temperature data.

        Returns
        -------
        toroidal_rotation
            xarray.DataArray containing data for toroidal rotations
            for the given impurity element
        """
        self.input_check(
            "asymmetry_parameters", asymmetry_parameters, DataArray, 3, True
        )

        self.input_check("ion_temperature", ion_temperature, DataArray, 3, False)

        self.input_check("main_ion", main_ion, str)

        try:
            assert main_ion in list(ELEMENTS_BY_SYMBOL.keys())
        except AssertionError:
            raise ValueError(
                f"main_ion must be one of {list(ELEMENTS_BY_SYMBOL.keys())}"
            )

        self.input_check("impurity", impurity, str)

        try:
            assert impurity in list(ELEMENTS_BY_SYMBOL.keys())
        except AssertionError:
            raise ValueError(
                f"impurity must be one of {list(ELEMENTS_BY_SYMBOL.keys())}"
            )

        self.input_check("Zeff", Zeff, DataArray, 2, True)

        self.input_check("electron_temp", electron_temp, DataArray, 2, False)

        asymmetry_parameter = asymmetry_parameters.sel(elements=impurity)

        impurity_name = ELEMENTS_BY_SYMBOL[impurity]
        main_ion_name = ELEMENTS_BY_SYMBOL[main_ion]

        impurity_mass_int = list(ELEMENTS_BY_MASS.keys())[
            list(ELEMENTS_BY_MASS.values()).index(impurity_name)
        ]

        unified_atomic_mass_unit = 1.660539066e-27
        impurity_mass = float(impurity_mass_int) * unified_atomic_mass_unit

        mean_charge = list(ELEMENTS_BY_ATOMIC_NUMBER.keys())[
            list(ELEMENTS_BY_ATOMIC_NUMBER.values()).index(impurity_name)
        ]

        main_ion_mass_int = list(ELEMENTS_BY_MASS.keys())[
            list(ELEMENTS_BY_MASS.values()).index(main_ion_name)
        ]

        main_ion_mass = float(main_ion_mass_int) * unified_atomic_mass_unit

        ion_temperature = ion_temperature.sel(elements=impurity)

        # mypy on the github CI suggests that * is an Unsupported operand type
        # between float and DataArray, don't know how to fix yet so for now ignored
        toroidal_rotation = 2.0 * ion_temperature * asymmetry_parameter  # type: ignore
        toroidal_rotation /= impurity_mass * (
            1.0
            - (mean_charge * main_ion_mass * Zeff * electron_temp)  # type: ignore
            / (impurity_mass * (ion_temperature + Zeff * electron_temp))
        )

        toroidal_rotation = toroidal_rotation ** 0.5

        return toroidal_rotation


class AsymmetryParameter(Operator):
    """Calculate the asymmetry parameter from toroidal rotation.

    Parameters
    ----------

    Returns
    -------
    asymmetry_parameter
        xarray.DataArray containing asymmetry_parameter for a given impurity element

    Methods
    -------
    input_check(
        var_name,
        var_to_check,
        var_type,
        ndim_to_check,
        greater_than_or_equal_zero
    )
        Checks the inputted var_to_check to ensure that it is valid.

    __call__(
        toroidal_rotation,
        ion_temperature,
        main_ion,
        impurity,
        Zeff,
        electron_temp,
    )
        Calculates the asymmetry parameter from the toroidal rotation.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    def __init__(self, sess: session.Session = session.global_session):
        super().__init__(sess=sess)

    def return_types(self, *args: DataType) -> Tuple[Any, ...]:
        return super().return_types(*args)

    def input_check(
        self,
        var_name: str,
        var_to_check,
        var_type: type,
        ndim_to_check: Optional[int] = None,
        greater_than_or_equal_zero: Optional[bool] = None,
    ):
        """Check validity of inputted variable - type check and
        various value checks(no infinities, greather than (or equal to) 0 or NaNs)

        Parameters
        ----------
        var_name
            Name of variable to check.
        var_to_check
            Variable to check.
        var_type
            Type to check variable against, eg. DataArray
        ndim_to_check
            Integer to check the number of dimensions of the variable.
        greater_than_or_equal_zero
            Boolean to check values in variable > 0 or >= 0.
        """
        try:
            assert isinstance(var_to_check, var_type)
        except AssertionError:
            raise TypeError(f"{var_name} must be of type {var_type}.")

        if greater_than_or_equal_zero is not None:
            try:
                if not greater_than_or_equal_zero:
                    # Mypy will ignore this line since even though var_to_check
                    # is type checked earlier it still doesn't explicitly
                    # know what type var_to_check
                    assert np.all(var_to_check > 0)  # type: ignore
                else:
                    # Mypy will ignore this line since even though var_to_check
                    # is type checked earlier it still doesn't explicitly
                    # know what type var_to_check
                    assert np.all(var_to_check >= 0)  # type: ignore
            except AssertionError:
                raise ValueError(f"Cannot have any negative values in {var_name}")

        if var_type in get_args(LabeledArray):
            try:
                assert np.all(var_to_check != np.nan)
            except AssertionError:
                raise ValueError(f"{var_name} cannot contain any NaNs.")

            try:
                assert np.all(np.abs(var_to_check) != np.inf)
            except AssertionError:
                raise ValueError(f"{var_name} cannot contain any infinities.")

        if ndim_to_check is not None and var_type in [np.ndarray, DataArray]:
            try:
                # Mypy will ignore this line since even though var_to_check
                # is type checked earlier it still doesn't explicitly
                # know what type var_to_check
                assert var_to_check.ndim == ndim_to_check  # type: ignore
            except AssertionError:
                raise ValueError(f"{var_name} must have {ndim_to_check} dimensions.")

    def __call__(  # type: ignore
        self,
        toroidal_rotations: DataArray,
        ion_temperature: DataArray,
        main_ion: str,
        impurity: str,
        Zeff: DataArray,
        electron_temp: DataArray,
    ):
        """Calculates the asymmetry parameter from the toroidal rotation.

        Parameters
        ----------
        toroidal_rotations
            xarray.DataArray containing toroidal rotations data.
        ion_temperature
            xarray.DataArray containing ion temperature data.
        main_ion
            Element symbol of main ion.
        impurity
            Element symbol of chosen impurity element.
        Zeff
            xarray.DataArray containing Z-effective data from diagnostics.
        electron_temp
            xarray.DataArray containing electron temperature data.

        Returns
        -------
        asymmetry_parameter
            xarray.DataArray containing data for asymmetry parameters
            for the given impurity element
        """
        self.input_check("toroidal_rotations", toroidal_rotations, DataArray, 3, True)

        self.input_check("ion_temperature", ion_temperature, DataArray, 3, False)

        self.input_check("main_ion", main_ion, str)

        try:
            assert main_ion in list(ELEMENTS_BY_SYMBOL.keys())
        except AssertionError:
            raise ValueError(
                f"main_ion must be one of {list(ELEMENTS_BY_SYMBOL.keys())}"
            )

        self.input_check("impurity", impurity, str)

        try:
            assert impurity in list(ELEMENTS_BY_SYMBOL.keys())
        except AssertionError:
            raise ValueError(
                f"impurity must be one of {list(ELEMENTS_BY_SYMBOL.keys())}"
            )

        self.input_check("Zeff", Zeff, DataArray, 2, True)

        self.input_check("electron_temp", electron_temp, DataArray, 2, False)

        toroidal_rotations = toroidal_rotations.sel(elements=impurity)

        impurity_name = ELEMENTS_BY_SYMBOL[impurity]
        main_ion_name = ELEMENTS_BY_SYMBOL[main_ion]

        impurity_mass_int = list(ELEMENTS_BY_MASS.keys())[
            list(ELEMENTS_BY_MASS.values()).index(impurity_name)
        ]

        unified_atomic_mass_unit = 1.660539066e-27
        impurity_mass = float(impurity_mass_int) * unified_atomic_mass_unit

        mean_charge = list(ELEMENTS_BY_ATOMIC_NUMBER.keys())[
            list(ELEMENTS_BY_ATOMIC_NUMBER.values()).index(impurity_name)
        ]

        main_ion_mass_int = list(ELEMENTS_BY_MASS.keys())[
            list(ELEMENTS_BY_MASS.values()).index(main_ion_name)
        ]

        main_ion_mass = float(main_ion_mass_int) * unified_atomic_mass_unit

        ion_temperature = ion_temperature.sel(elements=impurity)

        # mypy on the github CI suggests that * is in an Unsupported operand type
        # between float and DataArray, don't know how to fix yet so for now ignored
        asymmetry_parameter = (
            impurity_mass * (toroidal_rotations ** 2) / (2.0 * ion_temperature)  # type: ignore  # noqa: E501
        )
        asymmetry_parameter *= 1.0 - (
            mean_charge * main_ion_mass * Zeff * electron_temp  # type: ignore
        ) / (impurity_mass * (ion_temperature + Zeff * electron_temp))

        return asymmetry_parameter
