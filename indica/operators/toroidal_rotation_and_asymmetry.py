from typing import Any
from typing import get_args
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from xarray.core.dataarray import DataArray

from indica.numpy_typing import LabeledArray
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType


class ToroidalRotation(Operator):

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("toroidal_rotation", "plasma"),
    ]

    def __init__(self, sess: session.Session = session.global_session):
        super().__init__(sess=sess)

    def return_types(self, *args: DataType) -> Tuple[DataArray, ...]:
        return super().return_types(*args)

    def input_check(
        self,
        var_name: str,
        var_to_check,
        var_type: type,
        ndim_to_check: Optional[int] = None,
        greater_than_or_equal_zero: Optional[bool] = None,
    ):
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
        main_ion_mass: float,
        impurity_masses: DataArray,
        mean_charges: DataArray,
        Zeff_diag: DataArray,
        electron_temp: DataArray,
        impurity_element: str,
    ):
        asymmetry_parameter = asymmetry_parameters.sel(elements=impurity_element)
        impurity_mass = impurity_masses.sel(elements=impurity_element)
        mean_charge = mean_charges.sel(elements=impurity_element)

        toroidal_rotation = 2.0 * ion_temperature * asymmetry_parameter
        toroidal_rotation /= impurity_mass * (
            1.0
            - (mean_charge * main_ion_mass * Zeff_diag * ion_temperature)
            / (impurity_mass * (ion_temperature + Zeff_diag * electron_temp))
        )

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
        main_ion_mass,
        impurity_masses,
        mean_charges,
        Zeff_diag,
        electron_temp,
        impurity_element,
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
        """Checks the inputted var_to_check to ensure that it is valid.

        Parameters
        ----------
        var_name
            Name of variable to check, mostly used for error messages
        var_to_check
            Actual variable to check.
        var_type
            The type of the variable for type checking.
        ndim_to_check
            Optional: Number of dimensions the variable should have
            (if multi-dimensional)
        greater_than_or_equal_zero
            Optional: If True then the function checks that the variable is >= 0
            if False then the function checks that the variable is > 0
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
        toroidal_rotation: DataArray,
        ion_temperature: DataArray,
        main_ion_mass: float,
        impurity_masses: DataArray,
        mean_charges: DataArray,
        Zeff_diag: DataArray,
        electron_temp: DataArray,
        impurity_element: str,
    ):
        """Calculates the asymmetry parameter from the toroidal rotation.

        Parameters
        ----------
        toroidal_rotation
            xarray.DataArray containing toroidal rotation data.
        ion_temperature
            xarray.DataArray containing ion temperature data.
        main_ion_mass
            xarray.DataArray containing main ion mass data.
        impurity_masses
            xarray.DataArray containing data of the masses of all impurity elements.
        mean_charges
            xarray.DataArray containing data of the mean chgarges of all
            impurity elements.
        Zeff_diag
            xarray.DataArray containing Z-effective data from diagnostics.
        electron_temp
            xarray.DataArray containing electron temperature data.
        impurity_element
            Full name of the impurity element.
        """
        impurity_mass = impurity_masses.sel(elements=impurity_element)
        mean_charge = mean_charges.sel(elements=impurity_element)

        asymmetry_parameter = (
            impurity_mass * toroidal_rotation ** 2 / (2.0 * ion_temperature)
        )
        asymmetry_parameter *= 1.0 - (
            mean_charge * main_ion_mass * Zeff_diag * electron_temp
        ) / (impurity_mass * (ion_temperature + Zeff_diag * electron_temp))

        return asymmetry_parameter
