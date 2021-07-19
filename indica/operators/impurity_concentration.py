"""Operator calculating the impurity concentration
of a given element.
"""

from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from xarray import DataArray
from xarray.core.common import zeros_like

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType


class ImpurityConcentration(Operator):
    """Calculate impurity concentration of a given element.

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
        ("impurity_concentration", "impurity_element"),
        ("time", "impurity_element"),
    ]

    def __init__(self, sess: session.Session = session.global_session):
        super().__init__(sess=sess)

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        return super().return_types(*args)

    def __call__(  # type: ignore
        self,
        element: str,
        Zeff_diag: DataArray,
        impurity_densities: DataArray,
        electron_density: DataArray,
        mean_charge: DataArray,
        flux_surfaces: FluxSurfaceCoordinates,
        t: DataArray = None,
    ):

        try:
            assert isinstance(element, str)
        except AssertionError:
            raise TypeError(
                "Please ensure that the inputted element \
                argument is of type string."
            )

        elements_list = impurity_densities.coords["elements"]

        try:
            assert element in elements_list
        except AssertionError:
            raise ValueError(
                f"Please input a single valid element from list:\
                {elements_list}"
            )

        if t is None:
            t = Zeff_diag.t

        Zeff_diag = Zeff_diag.interp(t=t, method="nearest")

        transform = Zeff_diag.attrs["transform"]
        x1_name = transform.x1_name
        x2_name = transform.x2_name

        x1 = Zeff_diag.attrs[x1_name]
        x2_arr = np.linspace(0, 1, 300)
        x2 = DataArray(data=x2_arr, dims=[x2_name])

        R_arr, z_arr = transform.convert_to_Rz(x1, x2, t)

        rho, _ = flux_surfaces.convert_from_Rz(R_arr, z_arr, t)

        if isinstance(R_arr, (DataArray, np.ndarray)):
            R_arr = R_arr.squeeze()

        if isinstance(rho, (DataArray, np.ndarray)):
            rho = rho.squeeze()
            if isinstance(rho, DataArray):
                rho = rho.drop_vars("t")
                rho = rho.drop_vars("R")
                rho = rho.drop_vars("z")

        impurity_densities = impurity_densities.indica.interp2d(
            rho=rho,
            R=R_arr,
            method="cubic",
            assume_sorted=True,
        )

        impurity_densities = impurity_densities.interp(
            t=t, method="nearest", assume_sorted=True
        )

        # impurity_densities = impurity_densities.indica.interp2d(
        #     rho=rho.data,
        #     R=R_arr.data,
        #     method="cubic",
        #     assume_sorted=True
        # )

        electron_density = electron_density.interp(
            rho=rho, method="cubic", assume_sorted=True
        )

        electron_density = electron_density.interp(
            t=t, method="nearest", assume_sorted=True
        )

        mean_charge = mean_charge.interp(rho=rho, method="cubic", assume_sorted=True)

        mean_charge = mean_charge.interp(t=t, method="nearest", assume_sorted=True)

        dl = transform.distance(x2_name, DataArray(0), x2[0:2], 0)
        dl = dl[1]
        LoS_length = dl * 300

        concentration = zeros_like(Zeff_diag)

        term_1 = LoS_length * (Zeff_diag - 1)

        term_2 = zeros_like(term_1)
        for k, kdens in enumerate(impurity_densities.coords["elements"]):
            if element == kdens:
                continue

            term2_integrand = (impurity_densities[k] / electron_density) * (
                mean_charge[k] ** 2 - mean_charge[k]
            )

            term_2 += term2_integrand.sum(x2_name) * dl

        term_3 = (mean_charge[k - 1] ** 2 - mean_charge[k - 1]).sum(x2_name) * dl

        concentration = (term_1 - term_2) / term_3

        return concentration, t
