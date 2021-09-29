from typing import List
from typing import Tuple
from typing import Union

from xarray import concat
from xarray import DataArray

from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType
from ..utilities import input_check


class ExtrapolateImpurityDensity(Operator):
    """Extrapolate the impurity density beyond the limits of SXR (Soft X-ray)

    Attributes
    ----------
    ARGUMENT_TYPES: List[DataType]
        Ordered list of the types of data expected for each argument of the
        operator.
    RESULT_TYPES: List[DataType]
        Ordered list of the types of data returned by the operator.

    Returns
    -------
    impurity_density_full

    t
        If ``t`` was not specified as an argument for the __call__ function,
        return the time the results are given for.
        Otherwise return the argument.

    Methods
    -------
    __call__(
        element, ion_radiation_loss, impurity_radiation_losses, electron_density,
        mean_charge, flux_surfaces, t
    )
        Calculates the impurity concentration for the inputted element.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("impurity_density", "impurity_element"),
        ("time", "impurity_element"),
    ]

    def __init__(self, sess: session.Session = session.global_session):
        super().__init__(sess=sess)

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        return super().return_types(*args)

    def recover_rho(self, truncation_threshold: float, electron_temperature: DataArray):
        """Recover the rho value for a given electron temperature threshold, as in
        at what rho location does the electron temperature drop below the specified
        threshold.

        Parameters
        ----------
        truncation_threshold
            User-specified temperature truncation threshold.
        electron_temperature
            xarray.DataArray of the electron temperature profile (in rho).
        """
        input_check(
            "truncation_threshold",
            truncation_threshold,
            float,
            greater_than_or_equal_zero=False,
        )

        input_check(
            "electron_temperature",
            electron_temperature,
            DataArray,
            greater_than_or_equal_zero=False,
        )

        try:
            assert set(["rho"]).issubset(set(list(electron_temperature.coords.keys())))
        except AssertionError:
            raise AssertionError("Electron temperature must be a profile of rho.")

        threshold_rho = electron_temperature.where(
            (electron_temperature - truncation_threshold >= 0), drop=True
        ).min("rho")

        threshold_rho = electron_temperature.where(
            electron_temperature == threshold_rho, drop=True
        ).coords["rho"]

        return threshold_rho

    def __call__(  # type: ignore
        self,
        impurity_density_sxr: DataArray,
        electron_density: DataArray,
        electron_temperature: DataArray,
        truncation_threshold: float,
        t: DataArray = None,
    ):
        """Extrapolates the impurity density beyond the limits of SXR (Soft X-ray)

        Parameters
        ----------
        impurity_density_sxr
            xarray.DataArray of impurity density derived from soft X-ray emissivity.
        electron_density
            xarray.DataArray of electron density
        electron_temperature
            xarray.DataArray of electron temperature
        truncation_threshold
            Truncation threshold for the electron temperature
        t
            Optional, time at which the impurity concentration is to be calculated at.

        Returns
        -------
        impurity_density_full

        t
            If ``t`` was not specified as an argument for the __call__ function,
            return the time the results are given for.
            Otherwise return the argument.
        """

        input_check(
            "impurity_density_sxr",
            impurity_density_sxr,
            DataArray,
            ndim_to_check=2,
            greater_than_or_equal_zero=True,
        )

        input_check(
            "electron_density",
            electron_density,
            DataArray,
            ndim_to_check=2,
            greater_than_or_equal_zero=True,
        )

        input_check(
            "electron_temperature",
            electron_temperature,
            DataArray,
            ndim_to_check=2,
            greater_than_or_equal_zero=False,
        )

        input_check(
            "truncation_threshold",
            truncation_threshold,
            float,
            greater_than_or_equal_zero=False,
        )

        if t is None:
            t = electron_density.t
        else:
            input_check("t", t, DataArray, greater_than_or_equal_zero=True)

        threshold_rho = self.recover_rho(truncation_threshold, electron_temperature)

        # Discontinuity mitigation
        boundary_electron_density = electron_density.sel({"rho": threshold_rho})
        boundary_impurity_density_sxr = impurity_density_sxr.sel({"rho": threshold_rho})

        discontinuity_scale = boundary_impurity_density_sxr / boundary_electron_density
        discontinuity_scale = discontinuity_scale.data[0, 0]

        # Continue impurity_density_sxr following the shape of the electron density
        # profile.
        bounded_impurity_density_sxr = impurity_density_sxr.where(
            impurity_density_sxr.rho <= threshold_rho, drop=True
        )

        bounded_electron_density = electron_density.where(
            electron_density.rho > threshold_rho, drop=True
        )

        extrapolated_impurity_density = concat(
            (
                bounded_impurity_density_sxr,
                bounded_electron_density * discontinuity_scale,
            ),
            dim="rho",
        )

        return extrapolated_impurity_density, t
