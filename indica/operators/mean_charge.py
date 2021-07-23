"""Operator calculating the mean charge from the fractional abundance of all
ionisation stages of a given element.
"""

from typing import get_args
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from xarray.core.common import zeros_like
from xarray.core.dataarray import DataArray

from indica.datatypes import ELEMENTS_BY_ATOMIC_NUMBER
from indica.numpy_typing import LabeledArray
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType


class MeanCharge(Operator):
    """Calculate mean charge for a given element from its fractional abundance.

    Parameters
    ----------

    Returns
    -------
    mean_charge
        numpy.ndarray of mean charge of the given element.

    Methods
    -------
    __call__(FracAbundObj, element)
        Calculates the mean charge.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("mean_charge", "impurity_element"),
    ]

    def __init__(self, sess: session.Session = session.global_session):
        super().__init__(sess=sess)

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
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

    def __call__(self, FracAbundObj: DataArray, element: str):  # type: ignore
        """Function to calculate the mean charge.

        Parameters
        ----------
        FracAbundObj
            numpy.ndarray describing the fractional abundance of the given element.
            The first axis must correspond to the ionisation stages of the element.
        element
            Name of the element for which the mean charge is desired.

        Returns
        -------
        mean_charge
            numpy.ndarray of mean charge of the given element.
        """

        self.input_check("FracAbundObj", FracAbundObj, DataArray, 3, True)
        self.input_check("element", element, str)

        element_atomic_number_tmp = [
            k for k, v in ELEMENTS_BY_ATOMIC_NUMBER.items() if v == element
        ]
        try:
            assert len(element_atomic_number_tmp) == 1
        except AssertionError:
            raise ValueError(
                f"Please input a single valid element from list:\
                {list(ELEMENTS_BY_ATOMIC_NUMBER.values())}"
            )

        element_atomic_number = element_atomic_number_tmp[0]

        ionisation_stages = np.linspace(
            0, element_atomic_number, element_atomic_number + 1, dtype=int
        )

        try:
            assert ionisation_stages.shape[0] == FracAbundObj.shape[0]
        except AssertionError:
            raise AssertionError(
                f"Number of ionisation stages in the \
                FractionalAbundance object do not match the expected number for the \
                    element provided, {element}"
            )

        mean_charge = zeros_like(FracAbundObj)
        mean_charge = mean_charge.isel(stages=0)
        mean_charge.drop_vars("stages")

        mean_charge = np.sum(
            ionisation_stages[:, np.newaxis, np.newaxis] * FracAbundObj, axis=0
        )

        return mean_charge
