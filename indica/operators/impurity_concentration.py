"""Operator calculating the impurity concentration
of a given element.
"""

from typing import get_args
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from xarray import DataArray
from xarray.core.common import zeros_like

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.numpy_typing import LabeledArray
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

    Returns
    -------
    concentration
        xarray.DataArray containing the impurity concentration for the
        given impurity element.
    t
        If ``t`` was not specified as an argument for the __call__ function,
        return the time the results are given for.
        Otherwise return the argument.

    Methods
    -------
    __call__(
        element, Zeff_diag, impurity_densities, electron_density,
        mean_charge, flux_surfaces, t
    )
        Calculates the impurity concentration for the inputted element.
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
        element: str,
        Zeff_diag: DataArray,
        impurity_densities: DataArray,
        electron_density: DataArray,
        mean_charge: DataArray,
        flux_surfaces: FluxSurfaceCoordinates,
        t: DataArray = None,
    ):
        """Calculates the impurity concentration for the inputted element.

        Parameters
        ----------
        element
            String specifying the element for which the impurity concentration
            is desired. It should be given in full lower-case, eg. "beryllium"
        Zeff_diag
            xarray.DataArray containing the Zeff value/s from a diagnostic such as,
            Bremsstrahlung (ZEFH/KS3)
        impurity_densities
            xarray.DataArray of impurity densities for all impurity elements
            of interest.
        electron_density
            xarray.DataArray of electron density
        mean_charge
            xarray.DataArray of mean charge of all impurity elements of interest.
            This can be provided manually (with dimensions of ["elements", "rho", "t]),
            or can be passed as the results of MeanCharge.__call__
        flux_surfaces
            FluxSurfaceCoordinates object that defines the flux surface geometry
            of the equilibrium of interest.
        t
            Optional, time at which the impurity concentration is to be calculated at.

        Returns
        -------
        concentration
            xarray.DataArray containing the impurity concentration for the
            given impurity element.
        t
            If ``t`` was not specified as an argument for the __call__ function,
            return the time the results are given for.
            Otherwise return the argument.
        """
        self.input_check(
            "impurity_densities",
            impurity_densities,
            DataArray,
            greater_than_or_equal_zero=True,
        )

        self.input_check("element", element, str)

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
        else:
            self.input_check("t", t, DataArray, 1, True)

        self.input_check("Zeff_diag", Zeff_diag, DataArray, 1, True)
        self.input_check("electron_density", electron_density, DataArray, 2, False)
        self.input_check("mean_charge", mean_charge, DataArray, 3, True)
        self.input_check("flux_surfaces", flux_surfaces, FluxSurfaceCoordinates)

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

        if set(["R", "z"]).issubset(set(list(impurity_densities.coords.keys()))):
            impurity_densities = impurity_densities.indica.interp2d(
                z=z_arr,
                R=R_arr,
                method="cubic",
                assume_sorted=True,
            )
        elif set(["rho"]).issubset(set(list(impurity_densities.coords.keys()))):
            impurity_densities = impurity_densities.interp(
                rho=rho, method="linear", assume_sorted=True
            )
        else:
            raise ValueError(
                'Inputted impurity densities does not have any compatible\
                    coordinates: ["rho"] or ["R", "z"]'
            )

        impurity_densities = impurity_densities.interp(
            t=t, method="linear", assume_sorted=True
        )

        electron_density = electron_density.interp(
            rho=rho, method="linear", assume_sorted=True
        )

        electron_density = electron_density.interp(
            t=t, method="linear", assume_sorted=True
        )

        mean_charge = mean_charge.interp(rho=rho, method="linear", assume_sorted=True)

        mean_charge = mean_charge.interp(t=t, method="linear", assume_sorted=True)

        dl = transform.distance(x2_name, DataArray(0), x2[0:2], 0)
        dl = dl[1]
        LoS_length = dl * 300

        concentration = zeros_like(Zeff_diag)

        term_1 = LoS_length * (Zeff_diag - 1)

        term_2 = zeros_like(term_1)
        for k, kdens in enumerate(impurity_densities.coords["elements"]):
            if element == kdens:
                term_3 = (mean_charge[k] ** 2 - mean_charge[k]).sum(x2_name) * dl
                continue

            term2_integrand = (impurity_densities[k] / electron_density) * (
                mean_charge[k] ** 2 - mean_charge[k]
            )

            term_2 += term2_integrand.sum(x2_name) * dl

        concentration = (term_1 - term_2) / term_3

        return concentration, t
