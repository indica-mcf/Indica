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
        main_ion_mass,
        impurity_masses,
        mean_charges,
        Zeff_diag,
        electron_temp,
        impurity_element,
    )
        Calculates the toroidal_rotation from the asymmetry_parameter.
    """

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
        main_ion_mass: float,
        impurity_masses: DataArray,
        mean_charges: DataArray,
        Zeff_diag: DataArray,
        electron_temp: DataArray,
        impurity_element: str,
    ):
        """Calculates the toroidal rotation from the asymmetry parameter.

        Parameters
        ----------
        asymmetry_parameters
            xarray.DataArray containing asymmetry parameters data.
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

        self.input_check(
            "main_ion_mass", main_ion_mass, float, greater_than_or_equal_zero=False
        )

        self.input_check("impurity_masses", impurity_masses, DataArray, 1, False)

        self.input_check("mean_charges", mean_charges, DataArray, 3, True)

        self.input_check("Zeff_diag", Zeff_diag, DataArray, 2, True)

        self.input_check("electron_temp", electron_temp, DataArray, 2, False)

        self.input_check("impurity_element", impurity_element, str)

        asymmetry_parameter = asymmetry_parameters.sel(elements=impurity_element)
        impurity_mass = impurity_masses.sel(elements=impurity_element)
        mean_charge = mean_charges.sel(elements=impurity_element)
        ion_temperature = ion_temperature.sel(elements=impurity_element)

        toroidal_rotation = 2.0 * ion_temperature * asymmetry_parameter
        toroidal_rotation /= impurity_mass * (
            1.0
            - (mean_charge * main_ion_mass * Zeff_diag * electron_temp)
            / (impurity_mass * (ion_temperature + Zeff_diag * electron_temp))
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
        toroidal_rotations
            xarray.DataArray containing toroidal rotations data.
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

        Returns
        -------
        asymmetry_parameter
            xarray.DataArray containing data for asymmetry parameters
            for the given impurity element
        """
        self.input_check("toroidal_rotations", toroidal_rotations, DataArray, 3, True)

        self.input_check("ion_temperature", ion_temperature, DataArray, 3, False)

        self.input_check(
            "main_ion_mass", main_ion_mass, float, greater_than_or_equal_zero=False
        )

        self.input_check("impurity_masses", impurity_masses, DataArray, 1, False)

        self.input_check("mean_charges", mean_charges, DataArray, 3, True)

        self.input_check("Zeff_diag", Zeff_diag, DataArray, 2, True)

        self.input_check("electron_temp", electron_temp, DataArray, 2, False)

        self.input_check("impurity_element", impurity_element, str)

        impurity_mass = impurity_masses.sel(elements=impurity_element)
        mean_charge = mean_charges.sel(elements=impurity_element)
        toroidal_rotations = toroidal_rotations.sel(elements=impurity_element)
        ion_temperature = ion_temperature.sel(elements=impurity_element)

        asymmetry_parameter = (
            impurity_mass * (toroidal_rotations ** 2) / (2.0 * ion_temperature)
        )
        asymmetry_parameter *= 1.0 - (
            mean_charge * main_ion_mass * Zeff_diag * electron_temp
        ) / (impurity_mass * (ion_temperature + Zeff_diag * electron_temp))

        return asymmetry_parameter
