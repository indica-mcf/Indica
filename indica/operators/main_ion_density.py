"""Operator calculating the main ion density given the densities of impurities.
"""

from typing import get_args
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from xarray.core.dataarray import DataArray

from indica.datatypes import DataType
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..numpy_typing import LabeledArray


class MainIonDensity(Operator):

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("main_ion", "number_density"),
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
        impurity_densities: DataArray,
        electron_density: DataArray,
        mean_charge: DataArray,
    ):
        # no ndim check since impurity densities can have coords:
        # [elements, rho, t] or [elements, R, z, t]
        self.input_check(
            "impurity_densities",
            impurity_densities,
            DataArray,
            greater_than_or_equal_zero=True,
        )

        self.input_check("electron_density", electron_density, DataArray, 2, True)

        self.input_check("mean_charge", mean_charge, DataArray, 3, True)

        main_ion_density = electron_density - (mean_charge * impurity_densities).sum(
            "elements"
        )

        return main_ion_density
