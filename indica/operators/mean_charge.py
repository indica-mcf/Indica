"""Operator calculating the mean charge from the fractional abundance of all
ionisation stages of a given element.
"""

from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from xarray.core.common import zeros_like
from xarray.core.dataarray import DataArray

from indica.datatypes import ELEMENTS_BY_ATOMIC_NUMBER
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

        try:
            assert isinstance(FracAbundObj, DataArray)
        except AssertionError:
            raise TypeError(
                "Please ensure that the inputted FracAbundObj \
                argument is of type xarray.DataArray."
            )

        try:
            assert isinstance(element, str)
        except AssertionError:
            raise TypeError(
                "Please ensure that the inputted element \
                argument is of type string."
            )

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
