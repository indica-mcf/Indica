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

    Attributes
    ----------
    ARGUMENT_TYPES: List[DataType]
        Ordered list of the types of data expected for each argument of the
        operator.
    RESULT_TYPES: List[DataType]
        Ordered list of the types of data returned by the operator.
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
        """Calculates the toroidal rotation frequency from the asymmetry parameter.

        Parameters
        ----------
        asymmetry_parameters
            xarray.DataArray containing asymmetry parameters data. In units of m^-2.
        ion_temperature
            xarray.DataArray containing ion temperature data. In units of eV.
        main_ion
            Element symbol of main ion.
        impurity
            Element symbol of chosen impurity element.
        Zeff
            xarray.DataArray containing Z-effective data from diagnostics.
        electron_temp
            xarray.DataArray containing electron temperature data. In units of eV.

        Returns
        -------
        toroidal_rotation
            xarray.DataArray containing data for toroidal rotation frequencies
            for the given impurity element
        """
        input_check(
            "asymmetry_parameters",
            asymmetry_parameters,
            DataArray,
            ndim_to_check=3,
            positive=False,
            strictly_positive=False,
        )

        input_check(
            "ion_temperature",
            ion_temperature,
            DataArray,
            ndim_to_check=3,
            strictly_positive=True,
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

        input_check("Zeff", Zeff, DataArray, ndim_to_check=2, strictly_positive=False)

        input_check(
            "electron_temp",
            electron_temp,
            DataArray,
            ndim_to_check=2,
            strictly_positive=True,
        )

        asymmetry_parameter = asymmetry_parameters.sel(element=impurity)

        impurity_mass_int = ELEMENTS[impurity][1]

        unified_atomic_mass_unit = 931.4941e6  # in eV/c^2
        impurity_mass = float(impurity_mass_int) * unified_atomic_mass_unit

        mean_charge = ELEMENTS[impurity][0]

        main_ion_mass_int = ELEMENTS[main_ion][1]

        main_ion_mass = float(main_ion_mass_int) * unified_atomic_mass_unit

        ion_temperature = ion_temperature.sel(element=impurity)

        # mypy on the github CI suggests that * is an Unsupported operand type
        # between float and DataArray, don't know how to fix yet so for now ignored
        toroidal_rotation = 2.0 * ion_temperature * asymmetry_parameter  # type: ignore
        toroidal_rotation /= impurity_mass * (
            1.0
            - (mean_charge * main_ion_mass * Zeff * electron_temp)  # type: ignore
            / (impurity_mass * (ion_temperature + Zeff * electron_temp))
        )

        toroidal_rotation = toroidal_rotation**0.5

        c = 3.0e8  # speed of light in vacuum
        toroidal_rotation *= c

        return toroidal_rotation


class AsymmetryParameter(Operator):
    """Calculate the asymmetry parameter from toroidal rotation.

    Parameters
    ----------

    Returns
    -------
    asymmetry_parameter
        xarray.DataArray containing asymmetry_parameter for a given impurity element

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
        """Calculates the asymmetry parameter from the toroidal rotation frequency.

        Parameters
        ----------
        toroidal_rotations
            xarray.DataArray containing toroidal rotation frequencies data.
            In units of ms^-1.
        ion_temperature
            xarray.DataArray containing ion temperature data. In units of eV.
        main_ion
            Element symbol of main ion.
        impurity
            Element symbol of chosen impurity element.
        Zeff
            xarray.DataArray containing Z-effective data from diagnostics.
        electron_temp
            xarray.DataArray containing electron temperature data. In units of eV.

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
            strictly_positive=False,
        )

        input_check(
            "ion_temperature",
            ion_temperature,
            DataArray,
            ndim_to_check=3,
            strictly_positive=True,
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

        input_check("Zeff", Zeff, DataArray, ndim_to_check=2, strictly_positive=True)

        input_check(
            "electron_temp",
            electron_temp,
            DataArray,
            ndim_to_check=2,
            strictly_positive=True,
        )

        toroidal_rotations = toroidal_rotations.sel(element=impurity)

        impurity_mass_int = ELEMENTS[impurity][1]

        unified_atomic_mass_unit = 931.4941e6  # in eV/c^2
        impurity_mass = float(impurity_mass_int) * unified_atomic_mass_unit

        mean_charge = ELEMENTS[impurity][0]

        main_ion_mass_int = ELEMENTS[main_ion][1]

        main_ion_mass = float(main_ion_mass_int) * unified_atomic_mass_unit

        ion_temperature = ion_temperature.sel(element=impurity)

        c = 3.0e8  # speed of light in m/s
        toroidal_rotations /= c

        # mypy on the github CI suggests that * is in an Unsupported operand type
        # between float and DataArray, don't know how to fix yet so for now ignored
        asymmetry_parameter = (
            impurity_mass
            * (toroidal_rotations**2)  # type: ignore
            / (2.0 * ion_temperature)  # type: ignore
        )
        asymmetry_parameter *= 1.0 - (
            mean_charge * main_ion_mass * Zeff * electron_temp  # type: ignore
        ) / (impurity_mass * (ion_temperature + Zeff * electron_temp))

        return asymmetry_parameter
