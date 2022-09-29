"""Operator calculating the main ion density given the densities of impurities.
"""

from typing import List
from typing import Tuple
from typing import Union

from xarray.core.dataarray import DataArray

from indica.datatypes import DataType
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..utilities import input_check


class MainIonDensity(Operator):
    """Calculates the main ion density from given impurity densities and mean charge.

    Attributes
    ----------
    ARGUMENT_TYPES: List[DataType]
        Ordered list of the types of data expected for each argument of the
        operator.
    RESULT_TYPES: List[DataType]
        Ordered list of the types of data returned by the operator.

    Returns
    -------
    main_ion_density
        xarray.DataArray of the main ion density.

    Methods
    -------
    __call__(impurity_densities, electron_density, mean_charge)
        Calculates the main ion density from given impurity densities and mean charge.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("main_ion", "number_density"),
    ]

    def __init__(self, sess: session.Session = session.global_session):
        super().__init__(sess=sess)

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        return super().return_types(*args)

    def __call__(  # type: ignore
        self,
        impurity_densities: DataArray,
        electron_density: DataArray,
        mean_charge: DataArray,
    ):
        """Calculates the main ion density from given impurity densities
        and mean charge.

        Parameters
        ----------
        impurity_densities
            xarray.DataArray of impurity densities for all impurity elements
            of interest.
        electron_density
            xarray.DataArray of electron density
        mean_charge
            xarray.DataArray of mean charge of all impurity elements of interest.
            This can be provided manually
            (with dimensions of ["element", "rho_poloidal", "t]),
            or can be passed as the results of MeanCharge.__call__

        Returns
        -------
        main_ion_density
            xarray.DataArray of the main ion density.
        """
        # no ndim check since impurity densities can have coords:
        # [elements, rho, t] or [elements, R, z, t]
        input_check(
            "impurity_densities",
            impurity_densities,
            DataArray,
            strictly_positive=False,
        )

        input_check(
            "electron_density",
            electron_density,
            DataArray,
            strictly_positive=False,
        )

        input_check(
            "mean_charge",
            mean_charge,
            DataArray,
            strictly_positive=False,
        )

        main_ion_density = electron_density - (mean_charge * impurity_densities).sum(
            "element"
        )

        return main_ion_density
