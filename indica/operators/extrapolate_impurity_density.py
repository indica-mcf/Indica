from typing import cast
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from scipy.interpolate import UnivariateSpline
from xarray import concat
from xarray import DataArray
from xarray.core.common import zeros_like

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.operators.main_ion_density import MainIonDensity
from indica.operators.mean_charge import MeanCharge
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType
from ..utilities import input_check

# import matplotlib.pyplot as plt


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
        frac_abund: DataArray,
        impurity_elements: Sequence,
        main_ion_power_loss: DataArray,
        impurities_power_loss: DataArray,
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
        # theta_arr = np.array([0, np.max(sxr_theta)])
        theta_arr = np.array([0.0, np.pi])
        t_arr = t.values

        rho_arr = DataArray(data=rho_arr, coords={"rho": rho_arr}, dims=["rho"])
        theta_arr = DataArray(
            data=theta_arr, coords={"theta": theta_arr}, dims=["theta"]
        )
        t_arr = DataArray(data=t_arr, coords={"t": t_arr}, dims=["t"])

        R_deriv, z_deriv = flux_surfaces.convert_to_Rz(rho_arr, theta_arr, t_arr)

        if isinstance(R_deriv, DataArray):
            R_deriv = R_deriv.transpose("rho", "theta", "t")
        if isinstance(z_deriv, DataArray):
            z_deriv = z_deriv.transpose("rho", "theta", "t")

        impurity_density_sxr = impurity_density_sxr.indica.interp2d(
            {"R": R_deriv, "z": z_deriv}, method="linear", assume_sorted=True
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

        extrapolated_smooth_lfs = []
        extrapolated_smooth_hfs = []

        for ind_t, it in enumerate(extrapolated_impurity_density.coords["t"]):
            variance_extrapolated_impurity_density_lfs = (
                extrapolated_impurity_density.isel({"t": ind_t, "theta": 0}).var("rho")
            )

            variance_extrapolated_impurity_density_hfs = (
                extrapolated_impurity_density.isel({"t": ind_t, "theta": 1}).var("rho")
            )

            extrapolated_spline_lfs = UnivariateSpline(
                rho_arr,
                extrapolated_impurity_density[:, 0, ind_t],
                k=5,
                s=0.001 * variance_extrapolated_impurity_density_lfs,
            )

            extrapolated_spline_hfs = UnivariateSpline(
                rho_arr,
                extrapolated_impurity_density[:, 1, ind_t],
                k=5,
                s=0.001 * variance_extrapolated_impurity_density_hfs,
            )

            extrapolated_smooth_lfs.append(extrapolated_spline_lfs(rho_arr, 0))
            extrapolated_smooth_hfs.append(extrapolated_spline_hfs(rho_arr, 0))

        extrapolated_smooth_lfs = DataArray(
            data=extrapolated_smooth_lfs,
            coords={"t": t, "rho": rho_arr},
            dims=["t", "rho"],
        )

        extrapolated_smooth_hfs = DataArray(
            data=extrapolated_smooth_hfs,
            coords={"t": t, "rho": rho_arr},
            dims=["t", "rho"],
        )

        extrapolated_smooth_lfs = cast(DataArray, extrapolated_smooth_lfs).transpose(
            "rho", "t"
        )
        extrapolated_smooth_hfs = cast(DataArray, extrapolated_smooth_hfs).transpose(
            "rho", "t"
        )

        R_lfs_midplane = cast(DataArray, R_deriv).isel(theta=0)  # theta = 0.0
        R_hfs_midplane = cast(DataArray, R_deriv).isel(theta=1)  # theta = np.pi

        derived_asymmetry_parameter = np.log(
            cast(DataArray, extrapolated_smooth_hfs)
            / cast(DataArray, extrapolated_smooth_lfs)
        )

        asym_denominator = R_hfs_midplane ** 2 - R_lfs_midplane ** 2

        derived_asymmetry_parameter /= asym_denominator

        # Set constant asymmetry parameter for rho<0.1
        derived_asymmetry_parameter.loc[
            0:0.1, :  # type: ignore
        ] = derived_asymmetry_parameter.loc[0.1, :]

        theta_arr = np.linspace(-np.pi, np.pi, 41)
        theta_arr = DataArray(theta_arr, {"theta": theta_arr}, ["theta"])
        R_deriv, z_deriv = flux_surfaces.convert_to_Rz(rho_arr, theta_arr, t_arr)
        R_deriv = cast(DataArray, R_deriv).transpose("rho", "theta", "t")
        z_deriv = cast(DataArray, z_deriv).transpose("rho", "theta", "t")

        asymmetry_modifier = np.exp(
            derived_asymmetry_parameter * (R_deriv ** 2 - R_lfs_midplane ** 2)
        )

        asymmetry_modifier = asymmetry_modifier.transpose("rho", "theta", "t")

        extrapolated_smooth_density = extrapolated_smooth_lfs * asymmetry_modifier
        extrapolated_smooth_density = extrapolated_smooth_density.transpose(
            "rho", "theta", "t"
        )

        extrapolated_smooth_density = DataArray(
            data=extrapolated_smooth_density.data[np.newaxis, :],
            coords=dict(
                **{"elements": impurity_elements}, **extrapolated_smooth_density.coords
            ),
            dims=["elements", *extrapolated_smooth_density.dims],
        )

        mean_charges = zeros_like(extrapolated_smooth_density)
        mean_charges = mean_charges.isel(theta=0)

        for element in impurity_elements:
            mean_charge = MeanCharge()
            mean_charge = mean_charge(frac_abund, element)
            mean_charges.loc[element] = mean_charge

        main_ion_density_obj = MainIonDensity()
        main_ion_density = main_ion_density_obj(
            extrapolated_smooth_density, electron_density, mean_charges
        )

        main_ion_density = main_ion_density.transpose("rho", "theta", "t")

        derived_power_loss = electron_density * (main_ion_density * main_ion_power_loss)
        impurities_losses = zeros_like(main_ion_density)
        for element in impurity_elements:
            impurities_losses += (
                extrapolated_smooth_density.loc[element]
                * impurities_power_loss.loc[element]
            )
        impurities_losses *= electron_density
        derived_power_loss += impurities_losses

        R_arr = np.linspace(np.min(R_deriv[1:]), np.max(R_deriv[1:]), 40)
        z_arr = np.linspace(np.min(z_deriv), np.max(z_deriv), 40)

        R_arr = DataArray(R_arr, {"R": R_arr}, ["R"])
        z_arr = DataArray(z_arr, {"z": z_arr}, ["z"])

        rho_derived, theta_derived = flux_surfaces.convert_from_Rz(R_arr, z_arr, t_arr)
        rho_derived = cast(DataArray, rho_derived).transpose("R", "z", "t")
        theta_derived = cast(DataArray, theta_derived).transpose("R", "z", "t")
        rho_derived = abs(rho_derived)

        extrapolated_smooth_density = extrapolated_smooth_density.indica.interp2d(
            {"rho": rho_derived, "theta": theta_derived},
            method="linear",
            assume_sorted=True,
        )

        extrapolated_smooth_density = extrapolated_smooth_density.fillna(0.0)

        return extrapolated_smooth_density, derived_asymmetry_parameter, t
