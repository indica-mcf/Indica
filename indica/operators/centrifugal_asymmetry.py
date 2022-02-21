from typing import Any
from typing import List
from typing import Tuple
from typing import Union

from xarray.core.dataarray import DataArray

from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType
from ..datatypes import ELEMENTS
from ..utilities import input_check


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
        input_check(
            "asymmetry_parameters",
            asymmetry_parameters,
            DataArray,
            ndim_to_check=3,
            greater_than_or_equal_zero=True,
        )

        input_check(
            "ion_temperature",
            ion_temperature,
            DataArray,
            ndim_to_check=3,
            greater_than_or_equal_zero=False,
        )

        input_check("main_ion", main_ion, str)

        try:
            assert main_ion in list(ELEMENTS.keys())
        except AssertionError:
            raise ValueError(f"main_ion must be one of {list(ELEMENTS.keys())}")

        input_check("impurity", impurity, str)

        try:
            assert impurity in list(ELEMENTS.keys())
        except AssertionError:
            raise ValueError(f"impurity must be one of {list(ELEMENTS.keys())}")

        input_check(
            "Zeff", Zeff, DataArray, ndim_to_check=2, greater_than_or_equal_zero=True
        )

        input_check(
            "electron_temp",
            electron_temp,
            DataArray,
            ndim_to_check=2,
            greater_than_or_equal_zero=False,
        )

        asymmetry_parameter = asymmetry_parameters.sel(elements=impurity)

        impurity_mass_int = ELEMENTS[impurity][1]

        unified_atomic_mass_unit = 1.660539066e-27
        impurity_mass = float(impurity_mass_int) * unified_atomic_mass_unit

        mean_charge = ELEMENTS[impurity][0]

        main_ion_mass_int = ELEMENTS[main_ion][1]

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

        toroidal_rotation = toroidal_rotation**0.5

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
        input_check(
            "toroidal_rotations",
            toroidal_rotations,
            DataArray,
            ndim_to_check=3,
            greater_than_or_equal_zero=True,
        )

        input_check(
            "ion_temperature",
            ion_temperature,
            DataArray,
            ndim_to_check=3,
            greater_than_or_equal_zero=False,
        )

        input_check("main_ion", main_ion, str)

        try:
            assert main_ion in list(ELEMENTS.keys())
        except AssertionError:
            raise ValueError(f"main_ion must be one of {list(ELEMENTS.keys())}")

        input_check("impurity", impurity, str)

        try:
            assert impurity in list(ELEMENTS.keys())
        except AssertionError:
            raise ValueError(f"impurity must be one of {list(ELEMENTS.keys())}")

        input_check(
            "Zeff", Zeff, DataArray, ndim_to_check=2, greater_than_or_equal_zero=True
        )

        input_check(
            "electron_temp",
            electron_temp,
            DataArray,
            ndim_to_check=2,
            greater_than_or_equal_zero=False,
        )

        toroidal_rotations = toroidal_rotations.sel(elements=impurity)

        impurity_mass_int = ELEMENTS[impurity][1]

        unified_atomic_mass_unit = 1.660539066e-27
        impurity_mass = float(impurity_mass_int) * unified_atomic_mass_unit

        mean_charge = ELEMENTS[impurity][0]

        main_ion_mass_int = ELEMENTS[main_ion][1]

        main_ion_mass = float(main_ion_mass_int) * unified_atomic_mass_unit

        ion_temperature = ion_temperature.sel(elements=impurity)

        # mypy on the github CI suggests that * is in an Unsupported operand type
        # between float and DataArray, don't know how to fix yet so for now ignored
        asymmetry_parameter = (
            impurity_mass * (toroidal_rotations**2) / (2.0 * ion_temperature)
        )
        asymmetry_parameter *= 1.0 - (
            mean_charge * main_ion_mass * Zeff * electron_temp  # type: ignore
        ) / (impurity_mass * (ion_temperature + Zeff * electron_temp))

        return asymmetry_parameter
