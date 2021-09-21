"""Operator calculating the mean charge from the fractional abundance of all
ionisation charges of a given element.
"""

from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from xarray.core.common import zeros_like
from xarray.core.dataarray import DataArray

from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType
from ..datatypes import ELEMENTS
from ..utilities import input_check


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
            The first axis must correspond to the ionisation charges of the element.
        element
            Symbol of the element for which the mean charge is desired.

        Returns
        -------
        mean_charge
            numpy.ndarray of mean charge of the given element.
        """

        input_check(
            "FracAbundObj",
            FracAbundObj,
            DataArray,
            ndim_to_check=3,
            greater_than_or_equal_zero=True,
        )
        input_check("element", element, str)

        try:
            assert element in ELEMENTS.keys()
        except AssertionError:
            raise ValueError(
                f"Please input a single valid element from list:\
                {list(ELEMENTS.keys())}"
            )

        element_atomic_number = ELEMENTS[element][0]

        ionisation_charges = np.linspace(
            0,
            element_atomic_number,
            element_atomic_number + 1,  # type: ignore
            dtype=int,
        )

        try:
            assert ionisation_charges.shape[0] == FracAbundObj.shape[0]
        except AssertionError:
            raise AssertionError(
                f"Number of ionisation charges in the \
                FractionalAbundance object do not match the expected number for the \
                    element provided, {element}"
            )

        mean_charge = zeros_like(FracAbundObj)
        mean_charge = mean_charge.isel(ion_charges=0)
        mean_charge.drop_vars("ion_charges")

        mean_charge = np.sum(
            ionisation_charges[:, np.newaxis, np.newaxis] * FracAbundObj, axis=0
        )

        return mean_charge
