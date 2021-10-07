from typing import List
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate.fitpack2 import UnivariateSpline
from xarray import concat
from xarray import DataArray

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
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

        threshold_temp = electron_temperature.where(
            (electron_temperature - truncation_threshold >= 0), drop=True
        ).min("rho")

        threshold_rho = (
            electron_temperature.where(
                electron_temperature == threshold_temp, drop=True
            )
            .coords["rho"]
            .data
        )

        return threshold_rho

    def __call__(  # type: ignore
        self,
        impurity_density_sxr: DataArray,
        electron_density: DataArray,
        electron_temperature: DataArray,
        truncation_threshold: float,
        flux_surfaces: FluxSurfaceCoordinates,
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
            ndim_to_check=3,
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

        # Transform impurity_density_sxr to rho, theta coordinates
        R_arr = impurity_density_sxr.coords["R"]
        z_arr = impurity_density_sxr.coords["z"]

        sxr_rho, sxr_theta = flux_surfaces.convert_from_Rz(
            R_arr,
            z_arr,
            t.data,
        )

        if isinstance(sxr_rho, DataArray):
            sxr_rho = np.abs(sxr_rho)
            sxr_rho = sxr_rho.transpose("R", "z", "t")

        if isinstance(sxr_theta, DataArray):
            sxr_theta = sxr_theta.transpose("R", "z", "t")

        rho_arr = electron_density.coords["rho"].values
        theta_arr = np.array([0, np.max(sxr_theta)])
        t_arr = t.values

        rho_arr = DataArray(data=rho_arr, coords={"rho": rho_arr}, dims=["rho"])
        theta_arr = DataArray(
            data=theta_arr, coords={"theta": theta_arr}, dims=["theta"]
        )

        R_deriv, z_deriv = flux_surfaces.convert_to_Rz(rho_arr, theta_arr, t_arr)

        if isinstance(R_deriv, DataArray):
            R_deriv = R_deriv.transpose("rho", "theta", "t")
        if isinstance(z_deriv, DataArray):
            z_deriv = z_deriv.transpose("rho", "theta", "t")

        impurity_density_sxr = impurity_density_sxr.indica.interp2d(
            {"R": R_deriv, "z": z_deriv}, method="cubic"
        )
        impurity_density_sxr = impurity_density_sxr.transpose("rho", "theta", "t")

        # Discontinuity mitigation
        boundary_electron_density = electron_density.sel({"rho": threshold_rho})
        boundary_impurity_density_sxr = impurity_density_sxr.sel({"rho": threshold_rho})

        discontinuity_scale = boundary_impurity_density_sxr / boundary_electron_density
        discontinuity_scale = discontinuity_scale.isel(rho=0)

        # Continue impurity_density_sxr following the shape of the electron density
        # profile.
        bounded_impurity_density_sxr = impurity_density_sxr.where(
            impurity_density_sxr.rho <= threshold_rho, drop=True
        )

        bounded_electron_density = electron_density.where(
            electron_density.rho >= threshold_rho, drop=True
        )

        index_to_drop = np.where(
            bounded_impurity_density_sxr.coords["rho"].data == threshold_rho
        )[0][0]

        bounded_impurity_density_sxr = bounded_impurity_density_sxr.drop_isel(
            rho=index_to_drop
        )

        extrapolated_impurity_density = concat(
            (
                bounded_impurity_density_sxr,
                bounded_electron_density * discontinuity_scale,
            ),
            dim="rho",
        )

        assert np.all(np.logical_not(np.isnan(extrapolated_impurity_density)))

        rho_array = extrapolated_impurity_density.coords["rho"]

        variance_extrapolated_impurity_density_lfs = extrapolated_impurity_density.isel(
            {"t": 0, "theta": 0}
        ).var("rho")

        # variance_extrapolated_impurity_density_hfs = \
        #     extrapolated_impurity_density.isel({"t": 0, "theta": 1}).var("rho")

        extrapolated_spline_lfs = UnivariateSpline(
            rho_array,
            extrapolated_impurity_density[:, 0, 0],
            k=5,
            s=0.001 * variance_extrapolated_impurity_density_lfs,
        )

        # extrapolated_spline_hfs = UnivariateSpline(
        #     rho_array, extrapolated_impurity_density[:, 1, 0], k=5,
        #     s=0.001 * variance_extrapolated_impurity_density_hfs
        # )

        first_derivative_comb_lfs = extrapolated_impurity_density.isel(
            {"t": 0, "theta": 0}
        ).differentiate(coord="rho")

        drho = np.mean(np.diff(rho_array.data))

        first_derivative_spline_lfs = np.gradient(
            extrapolated_spline_lfs(rho_array, 0), drho
        )

        plt.plot(
            rho_array,
            extrapolated_impurity_density[:, 0, 0],
            c="b",
        )
        plt.plot(
            rho_array,
            extrapolated_spline_lfs(rho_array, 0),
            c="r",
        )
        plt.show()

        plt.plot(
            rho_array,
            first_derivative_comb_lfs,
            c="b",
        )
        plt.plot(
            rho_array,
            first_derivative_spline_lfs,
            c="r",
        )
        plt.show()

        return extrapolated_impurity_density, threshold_rho, t
