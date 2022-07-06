from typing import cast
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
from scipy.stats import norm
from xarray import concat
from xarray import DataArray
from xarray.core.common import zeros_like

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.operators.bolometry_derivation import BolometryDerivation
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType
from ..utilities import input_check


def asymmetry_from_R_z(
    data_R_z: DataArray,
    flux_surfaces: FluxSurfaceCoordinates,
    rho_arr: DataArray,
    threshold_rho: DataArray = None,
    t_arr: DataArray = None,
):
    """Function to calculate an asymmetry parameter from a given density profile in
    (R, z, t) coordinates.

    Parameters
    ----------
    data_R_z
        High-z density profile which is to be used to calculate the asymmetry parameter.
        xarray.DataArray with dimensions (R, z, t)
    flux_surfaces
        FluxSurfaceCoordinates object representing polar coordinate systems
        using flux surfaces for the radial coordinate.
    rho_arr
        1D xarray.DataArray of rho from 0 to 1.
    threshold_rho
        rho value denoting the cutoff point beyond which soft x-ray diagnostics
        are invalid. It's also used in setting the derived asymmetry parameter to be
        flat in the invalid region.
        xarray.DataArray with dimensions (t)
    t_arr
        1D xarray.DataArray of t.

    Returns
    -------
    derived_asymmetry_parameter
        Derived asymmetry parameter. xarray.DataArray with dimensions (rho, t)
    """

    input_check("data_R_z", data_R_z, DataArray, 3, True)

    input_check("flux_surfaces", flux_surfaces, FluxSurfaceCoordinates)

    input_check("rho_arr", rho_arr, DataArray, 1, True)

    if threshold_rho is not None:
        input_check("threshold_rho", threshold_rho, DataArray, 1, True)

    if t_arr is None:
        t_arr = data_R_z.coords["t"]
    else:
        input_check("t_arr", t_arr, DataArray, 1, True)

    theta_arr_ = np.array([0.0, np.pi])
    theta_arr = DataArray(data=theta_arr_, coords={"theta": theta_arr_}, dims=["theta"])

    R_deriv, z_deriv = flux_surfaces.convert_to_Rz(rho_arr, theta_arr)
    R_deriv = cast(DataArray, R_deriv).interp(t=t_arr, method="linear")
    z_deriv = cast(DataArray, z_deriv).interp(t=t_arr, method="linear")

    R_deriv = cast(DataArray, R_deriv).transpose("rho_poloidal", "theta", "t")
    z_deriv = cast(DataArray, z_deriv).transpose("rho_poloidal", "theta", "t")

    data_rho_theta = data_R_z.indica.interp2d(
        {"R": R_deriv, "z": z_deriv}, method="linear", assume_sorted=True
    )
    data_rho_theta = data_rho_theta.transpose("rho_poloidal", "theta", "t")

    R_lfs_midplane = cast(DataArray, R_deriv).isel(theta=0)  # theta = 0.0
    R_hfs_midplane = cast(DataArray, R_deriv).isel(theta=1)  # theta = np.pi

    derived_asymmetry_parameter = np.log(
        data_rho_theta.isel(theta=1) / data_rho_theta.isel(theta=0)
    )

    derived_asymmetry_parameter /= R_hfs_midplane**2 - R_lfs_midplane**2

    # Set constant asymmetry parameter for rho<0.1
    derived_asymmetry_parameter = derived_asymmetry_parameter.where(
        derived_asymmetry_parameter.coords["rho_poloidal"] > 0.1,
        other=derived_asymmetry_parameter.sel({"rho_poloidal": 0.1}, method="nearest"),
    )

    derived_asymmetry_parameter = np.abs(derived_asymmetry_parameter)

    if threshold_rho is not None:
        for ind_t, it in enumerate(threshold_rho.coords["t"]):
            derived_asymmetry_parameter.loc[
                threshold_rho[ind_t] :, it  # type:ignore
            ] = derived_asymmetry_parameter.loc[threshold_rho[ind_t], it]

    return derived_asymmetry_parameter


def asymmetry_from_rho_theta(
    data_rho_theta: DataArray,
    flux_surfaces: FluxSurfaceCoordinates,
    threshold_rho: DataArray = None,
    t_arr: DataArray = None,
):
    """Function to calculate an asymmetry parameter from a given density profile in
    (rho_poloidal, theta, t) coordinates.

    Parameters
    ----------
    data_rho_theta
        High-z density profile which is to be used to calculate the asymmetry parameter.
        xarray.DataArray with dimensions (rho_poloidal, theta, t)
    flux_surfaces
        FluxSurfaceCoordinates object representing polar coordinate systems
        using flux surfaces for the radial coordinate.
    threshold_rho
        rho value denoting the cutoff point beyond which soft x-ray diagnostics
        are invalid. It's also used in setting the derived asymmetry parameter to be
        flat in the invalid region.
        xarray.DataArray with dimensions (t)
    t_arr
        1D xarray.DataArray of t.

    Returns
    -------
    derived_asymmetry_parameter
        Derived asymmetry parameter. xarray.DataArray with dimensions (rho, t)
    """

    input_check("data_rho_theta", data_rho_theta, DataArray, 3, True)

    input_check("flux_surfaces", flux_surfaces, FluxSurfaceCoordinates)

    if threshold_rho is not None:
        input_check("threshold_rho", threshold_rho, DataArray, 1, True)

    if t_arr is None:
        t_arr = data_rho_theta.coords["t"]
    else:
        input_check("t_arr", t_arr, DataArray, 1, True)

    rho_arr = data_rho_theta.coords["rho_poloidal"]
    theta_arr_ = np.array([0.0, np.pi])
    theta_arr = DataArray(data=theta_arr_, coords={"theta": theta_arr_}, dims=["theta"])

    R_deriv, z_deriv = flux_surfaces.convert_to_Rz(rho_arr, theta_arr)

    R_deriv = cast(DataArray, R_deriv).interp(t=t_arr, method="linear")
    z_deriv = cast(DataArray, z_deriv).interp(t=t_arr, method="linear")

    R_deriv = cast(DataArray, R_deriv).transpose("rho_poloidal", "theta", "t")
    z_deriv = cast(DataArray, z_deriv).transpose("rho_poloidal", "theta", "t")

    R_lfs_midplane = cast(DataArray, R_deriv).isel(theta=0)  # theta = 0.0
    R_hfs_midplane = cast(DataArray, R_deriv).isel(theta=1)  # theta = np.pi

    derived_asymmetry_parameter = np.log(
        data_rho_theta.interp(theta=np.pi, method="linear")
        / data_rho_theta.interp(theta=0.0, method="linear")
    )

    derived_asymmetry_parameter /= R_hfs_midplane**2 - R_lfs_midplane**2

    # Set constant asymmetry parameter for rho<0.1
    derived_asymmetry_parameter = derived_asymmetry_parameter.where(
        derived_asymmetry_parameter.coords["rho_poloidal"] > 0.1,
        other=derived_asymmetry_parameter.sel({"rho_poloidal": 0.1}, method="nearest"),
    )

    derived_asymmetry_parameter = np.abs(derived_asymmetry_parameter)

    if threshold_rho is not None:
        for ind_t, it in enumerate(threshold_rho.coords["t"]):
            derived_asymmetry_parameter.loc[
                threshold_rho[ind_t] :, it  # type:ignore
            ] = derived_asymmetry_parameter.loc[threshold_rho[ind_t], it]

    return derived_asymmetry_parameter


def recover_threshold_rho(truncation_threshold: float, electron_temperature: DataArray):
    """Recover the rho value for a given electron temperature threshold, as in
    at what rho location does the electron temperature drop below the specified
    temperature.

    Parameters
    ----------
    truncation_threshold
        User-specified (float) temperature truncation threshold.
    electron_temperature
        xarray.DataArray of the electron temperature profile,
        with dimensions (rho, t).

    Returns
    -------
    threshold_rho
        rho value beyond which the electron temperature falls below
        the truncation_threshold.
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
        2,
        greater_than_or_equal_zero=False,
    )

    try:
        assert set(["rho_poloidal"]).issubset(
            set(list(electron_temperature.coords.keys()))
        )
    except AssertionError:
        raise AssertionError("Electron temperature must be a profile of rho.")

    threshold_temp = electron_temperature.where(
        (electron_temperature - truncation_threshold >= 0), drop=True
    ).min("rho_poloidal")

    threshold_rho_ind = electron_temperature.where(
        electron_temperature >= threshold_temp, np.inf
    ).argmin("rho_poloidal")

    threshold_rho = electron_temperature.coords["rho_poloidal"].isel(
        rho_poloidal=threshold_rho_ind
    )

    return threshold_rho


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
    extrapolated_smooth_density_Rz
        Extrapolated and smoothed impurity density,
        xarray.DataArray with dimensions (R, z, t).
    extrapolated_smooth_density_rho_theta
        Extrapolated and smoothed impurity density,
        xarray.DataArray with dimensions (rho, theta, t).
    t
        If ``t`` was not specified as an argument for the __call__ function,
        return the time the results are given for.
        Otherwise return the argument.

    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("number_density", "impurity_element"),
        ("number_density", "impurity_element"),
        ("time", "impurity_element"),
    ]

    def __init__(
        self,
        sess: session.Session = session.global_session,
    ):
        """Initialises ExtrapolateImpurityDensity class."""
        super().__init__(sess=sess)

        self.halfwidthhalfmax_coeff = np.sqrt(2 * np.log(2))
        self.fullwidthhalfmax_coeff = 2 * self.halfwidthhalfmax_coeff

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        return super().return_types(*args)

    def transform_to_rho_theta(
        self,
        data_R_z: DataArray,
        flux_surfaces: FluxSurfaceCoordinates,
        rho_arr: DataArray,
        t_arr: DataArray = None,
    ):
        """Function to transform data from an (R, z) grid to a (rho_poloidal, theta) grid

        Parameters
        ----------
        data_R_z
            xarray.DataArray to be transformed. Dimensions (R, z, t)
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.
        rho_arr
            1D xarray.DataArray of rho_poloidal from 0 to 1.
        t_arr
            1D xarray.DataArray of t.

        Returns
        -------
        data_rho_theta
            Transformed xarray.DataArray. Dimensions (rho_poloidal, theta, t)
        R_deriv
            Variable describing value of R in every coordinate on a (rho, theta) grid.
            xarray.DataArray with dimensions (rho, theta, t)
        z_deriv
            Variable describing value of z in every coordinate on a (rho, theta) grid.
            xarray.DataArray with dimensions (rho, theta, t)
        """
        if t_arr is None:
            t_arr = data_R_z.coords["t"]

        theta_arr = np.linspace(-np.pi, np.pi, 21)
        # mypy doesn't like re-assignments which changes types.
        theta_arr = DataArray(  # type: ignore
            data=theta_arr, coords={"theta": theta_arr}, dims=["theta"]
        )

        R_deriv, z_deriv = flux_surfaces.convert_to_Rz(rho_arr, theta_arr, t_arr)

        R_deriv = cast(DataArray, R_deriv).transpose("rho_poloidal", "theta", "t")
        z_deriv = cast(DataArray, z_deriv).transpose("rho_poloidal", "theta", "t")

        data_rho_theta = data_R_z.indica.interp2d(
            {"R": R_deriv, "z": z_deriv}, method="linear", assume_sorted=True
        )
        data_rho_theta = data_rho_theta.transpose("rho_poloidal", "theta", "t")

        return data_rho_theta, R_deriv, z_deriv

    def basic_extrapolation(
        self,
        data_rho_theta: DataArray,
        electron_density: DataArray,
        threshold_rho: float,
    ):
        """Basic extrapolation which eliminates the data_rho_theta for
        rho > threshold_rho and joins on electron density from that point
        outwards (in rho). Also multiplies electron density to prevent a
        discontinuity at rho_poloidal=threshold_rho.

        Parameters
        ----------
        data_rho_theta
            xarray.DataArray to extrapolate. Dimensions (rho, theta, t)
        electron_density
            xarray.DataArray of Electron density (axisymmetric). Dimensions (rho, t)
        threshold_rho
            Threshold value (float) of rho beyond which SXR diagnostics cannot
            be used to accurately infer impurity density.

        Returns
        -------
        extrapolated_impurity_density
            xarray.DataArray of extrapolated impurity density.
            Dimensions (rho, theta, t)
        """

        theta_arr = np.array([0.0, np.pi])
        theta_arr = DataArray(  # type: ignore
            data=theta_arr, coords={"theta": theta_arr}, dims=["theta"]
        )

        boundary_electron_density = electron_density.sel(
            {"rho_poloidal": threshold_rho}
        ).squeeze()
        boundary_data = data_rho_theta.sel({"rho_poloidal": threshold_rho}).squeeze()

        discontinuity_scale = boundary_data / boundary_electron_density

        # Continue impurity_density_sxr following the shape of the electron density
        # profile.
        bounded_data = data_rho_theta.where(
            data_rho_theta.rho_poloidal <= threshold_rho, 0.0
        )

        bounded_electron_density = electron_density.where(
            electron_density.rho_poloidal > threshold_rho, 0.0
        )

        extrapolated_impurity_density = (
            bounded_data + bounded_electron_density * discontinuity_scale
        )

        return extrapolated_impurity_density

    def extrapolation_smoothing(
        self,
        extrapolated_data: DataArray,
        rho_arr: DataArray,
    ):
        """Function to smooth extrapolatd data. Extrapolated data may not have
        any 0th order discontinuity but 1st order discontinuities may exist.
        Smoothing is necessary to eliminate these higher order discontinuities.

        Parameters
        ----------
        extrapolated_data
            xarray.DataArray extrapolated data to be smoothed.
            Dimensions (rho, theta, t)
        rho_arr
            xarray.DataArray used to construct smoothing splines. Dimensions (rho)
            (Must be higher or the same resolution as the rho dimension
            of extrapolated_data)

        Returns
        -------
        extrapolated_smooth_lfs_arr
            Extrapolated smoothed data on low-field side (fixed theta = 0)
        extrapolated_smooth_hfs_arr
            Extrapolated smoothed data on high-field side (fixed theta = pi)
        """
        t = extrapolated_data.coords["t"]

        extrapolated_smooth_lfs = []
        extrapolated_smooth_hfs = []

        for ind_t, it in enumerate(extrapolated_data.coords["t"]):
            variance_extrapolated_data_lfs = extrapolated_data.isel(
                {"t": ind_t, "theta": 0}
            ).var("rho_poloidal")

            variance_extrapolated_data_hfs = extrapolated_data.isel(
                {"t": ind_t, "theta": 1}
            ).var("rho_poloidal")

            extrapolated_spline_lfs = UnivariateSpline(
                rho_arr,
                extrapolated_data.isel(t=ind_t).sel(theta=0),
                k=5,
                s=0.001 * variance_extrapolated_data_lfs,
            )

            extrapolated_spline_hfs = UnivariateSpline(
                rho_arr,
                extrapolated_data.isel(t=ind_t).sel(theta=np.pi),
                k=5,
                s=0.001 * variance_extrapolated_data_hfs,
            )

            extrapolated_smooth_lfs.append(extrapolated_spline_lfs(rho_arr, 0))
            extrapolated_smooth_hfs.append(extrapolated_spline_hfs(rho_arr, 0))

        extrapolated_smooth_lfs_arr = DataArray(
            data=extrapolated_smooth_lfs,
            coords={"t": t, "rho_poloidal": rho_arr},
            dims=["t", "rho_poloidal"],
        )

        extrapolated_smooth_hfs_arr = DataArray(
            data=extrapolated_smooth_hfs,
            coords={"t": t, "rho_poloidal": rho_arr},
            dims=["t", "rho_poloidal"],
        )

        extrapolated_smooth_lfs_arr = extrapolated_smooth_lfs_arr.transpose(
            "rho_poloidal", "t"
        )
        extrapolated_smooth_hfs_arr = extrapolated_smooth_hfs_arr.transpose(
            "rho_poloidal", "t"
        )

        # Following section is to ensure that near the rho_poloidal=0 region, the
        # extrapolated_smooth_data is constant (ie. with a first-order derivative of 0).
        inv_extrapolated_smooth_hfs = DataArray(
            data=np.flip(extrapolated_smooth_hfs_arr.data, axis=0),
            coords={
                "rho_poloidal": -1
                * np.flip(extrapolated_smooth_hfs_arr.coords["rho_poloidal"].data),
                "t": extrapolated_smooth_hfs_arr.coords["t"].data,
            },
            dims=["rho_poloidal", "t"],
        )

        inv_rho_arr = inv_extrapolated_smooth_hfs.coords["rho_poloidal"].data
        inv_del_val = inv_rho_arr[-1]

        inv_extrapolated_smooth_hfs = inv_extrapolated_smooth_hfs.drop_sel(
            rho_poloidal=inv_del_val
        )

        extrapolated_smooth_mid_plane_arr = concat(
            (inv_extrapolated_smooth_hfs, extrapolated_smooth_lfs_arr), "rho_poloidal"
        )

        rho_zero_ind = np.where(
            np.isclose(extrapolated_smooth_mid_plane_arr.rho_poloidal.data, 0.0)
        )[0][0]

        smooth_central_region = extrapolated_smooth_mid_plane_arr.isel(
            rho_poloidal=slice(rho_zero_ind - 2, rho_zero_ind + 3)
        )

        smooth_central_region.loc[:, :] = smooth_central_region.max(dim="rho_poloidal")

        extrapolated_smooth_mid_plane_arr.loc[
            extrapolated_smooth_mid_plane_arr.rho_poloidal.data[
                rho_zero_ind - 2
            ] : extrapolated_smooth_mid_plane_arr.rho_poloidal.data[rho_zero_ind + 2],
            :,
        ] = smooth_central_region

        inv_extrapolated_smooth_hfs = extrapolated_smooth_mid_plane_arr.isel(
            rho_poloidal=slice(0, rho_zero_ind + 1)
        )

        extrapolated_smooth_hfs_arr = DataArray(
            data=np.flip(inv_extrapolated_smooth_hfs.data, axis=0),
            coords=extrapolated_smooth_hfs_arr.coords,
            dims=extrapolated_smooth_hfs_arr.dims,
        )

        # Ignoring mypy warning since it seems to be unaware that the xarray .loc
        # method uses label-based indexing and slicing instead of integer-based.
        extrapolated_smooth_lfs_arr = extrapolated_smooth_mid_plane_arr.loc[
            0:  # type: ignore
        ]

        return extrapolated_smooth_lfs_arr, extrapolated_smooth_hfs_arr

    def apply_asymmetry(
        self,
        asymmetry_parameter: DataArray,
        extrapolated_smooth_hfs: DataArray,
        extrapolated_smooth_lfs: DataArray,
        R_deriv: DataArray,
    ):
        """Applying an asymmetry parameter to low-field-side data which
        will be extended over the poloidal extent to obtain an asymmetric
        extrapolated smoothed data on a (rho, theta) grid.

        Parameters
        ----------
        asymmetry_parameter
            Asymmetry parameter to apply.
            xarray.DataArray with dimensions (rho, t)
        extrapolated_smooth_hfs
            Extrapolated smoothed data on high-field side (fixed theta = pi).
            xarray.DataArray with dimensions (rho, t)
        extrapolated_smooth_lfs
            Extrapolated smoothed data on low-field side (fixed theta = 0).
            xarray.DataArray with dimensions (rho, t)
        R_deriv
            Variable describing value of R in every coordinate on a (rho, theta) grid.
            xarray.DataArray with dimensions (rho, theta, t)

        Returns
        -------
        extrapolated_smooth_data
            Extrapolated and smoothed data on full (rho, theta) grid.
            xarray.DataArray with dimensions (rho, theta, t)
        asymmetry_modifier
            Asymmetry modifier used to transform a low-field side only rho-profile
            of a poloidally asymmetric quantity to a full poloidal cross-sectional
            profile ie. (rho, t) -> (rho, theta, t). Also can be defined as:
            exp(asymmetry_parameter * (R ** 2 - R_lfs ** 2)), where R is the major
            radius as a function of (rho, theta, t) and R_lfs is the low-field-side
            major radius as a function of (rho, t). xarray DataArray with dimensions
            (rho, theta, t)
        """
        rho_arr = extrapolated_smooth_hfs.coords["rho_poloidal"]
        self.rho_arr = rho_arr

        theta_arr = np.linspace(-np.pi, np.pi, 21)
        theta_arr = DataArray(
            theta_arr, {"theta": theta_arr}, ["theta"]
        )  # type: ignore

        R_lfs_midplane = cast(DataArray, R_deriv).sel(theta=0, method="nearest")

        asymmetry_modifier = np.exp(
            asymmetry_parameter * (R_deriv**2 - R_lfs_midplane**2)
        )

        asymmetry_modifier = asymmetry_modifier.transpose("rho_poloidal", "theta", "t")

        extrapolated_smooth_data = extrapolated_smooth_lfs * asymmetry_modifier
        extrapolated_smooth_data = extrapolated_smooth_data.transpose(
            "rho_poloidal", "theta", "t"
        )

        return extrapolated_smooth_data, asymmetry_modifier

    def transform_to_R_z(
        self,
        R_deriv: DataArray,
        z_deriv: DataArray,
        extrapolated_smooth_data: DataArray,
        flux_surfaces: FluxSurfaceCoordinates,
    ):
        """Function to transform data from an (rho, theta) grid to a (R, z) grid

        Parameters
        ----------
        R_deriv
            Variable describing value of R in every coordinate on a (rho, theta) grid.
            xarray.DataArray with dimensions (rho, theta, t)
            (from derive_and_apply_asymmetry)
        z_deriv
            Variable describing value of z in every coordinate on a (rho, theta) grid.
            xarray.DataArray with dimensions (rho, theta, t)
            (from derive_and_apply_asymmetry)
        extrapolated_smooth_data
            Extrapolated and smoothed data to transform onto (R, z) grid.
            xarray.DataArray with dimensions (rho, theta, t)
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.

        Returns
        -------
        extrapolated_smooth_data
            Extrapolated and smoothed data on (R, z) grid.
            xarray.DataArray with dimensions (R, z, t)
        """
        R_arr = np.linspace(np.min(R_deriv[1:]), np.max(R_deriv[1:]), 40)
        z_arr = np.linspace(np.min(z_deriv), np.max(z_deriv), 40)

        R_arr = DataArray(R_arr, {"R": R_arr}, ["R"])  # type: ignore
        z_arr = DataArray(z_arr, {"z": z_arr}, ["z"])  # type: ignore

        t_arr = extrapolated_smooth_data.coords["t"]

        rho_derived, theta_derived = flux_surfaces.convert_from_Rz(R_arr, z_arr, t_arr)
        rho_derived = cast(DataArray, rho_derived).transpose("R", "z", "t")
        theta_derived = cast(DataArray, theta_derived).transpose("R", "z", "t")
        rho_derived = abs(rho_derived)

        extrapolated_smooth_data = extrapolated_smooth_data.indica.interp2d(
            {"rho_poloidal": rho_derived, "theta": theta_derived},
            method="linear",
            assume_sorted=True,
        )

        extrapolated_smooth_data = extrapolated_smooth_data.fillna(0.0)

        return extrapolated_smooth_data

    def fitting_function(
        self,
        amplitude: float,
        standard_dev: float,
        position: float,
    ):
        """Function to construct a signal that modifies the
        extrapolated smoothed impurity density. The signal is constructed
        using a Gaussian profile with the three free parameters.

        Parameters
        ----------
        amplitude
            Amplitude of the additional signal (Gaussian amplitude)
        standard_dev
            Standard deviation associated with the Gaussian construction
            (can be defined as FWHM/2.355 where FWHM is full-width at half maximum)
        position
            Position of the Gaussian. During optimization this is constrained to
            the extrapolated region of rho (ie. outside the SXR validity region).

        Returns
        -------
        sig
            xarray.DataArray containing the Gaussian signal with dimensions (rho)
        """
        rho_arr = self.rho_arr

        gaussian_signal = norm(loc=position, scale=standard_dev)

        sig = gaussian_signal.pdf(rho_arr)

        sig = DataArray(
            data=sig, coords={"rho_poloidal": rho_arr}, dims=["rho_poloidal"]
        )

        sig /= sig.max()

        sig *= amplitude

        return sig

    def optimize_perturbation(
        self,
        extrapolated_smooth_data: DataArray,
        orig_bolometry_data: DataArray,
        bolometry_obj: BolometryDerivation,
        impurity_element: str,
        asymmetry_modifier: DataArray,
        time_correlation: bool = True,
    ):
        """Optimizes a Gaussian-style perturbation to recover the over-density
        structure that is expected on the low-field-side of the plasma.

        Parameters
        ----------
        extrapolated_smooth_data
            Extrapolated and smoothed data which continues the impurity density
            beyond the soft x-ray threshold limit by using electron density as a
            guide. xarray DataArray with dimensions (rho, theta, t).
        orig_bolometry_data
            Original bolometry data that is used in the objective function to fit
            the perturbation. xarray DataArray with dimensions (channels, t).
        bolometry_obj
            BolometryDerivation object.
        impurity_element
            String of impurity element symbol.
        asymmetry_modifier
            Asymmetry modifier used to transform a low-field side only rho-profile
            of a poloidally asymmetric quantity to a full poloidal cross-sectional
            profile ie. (rho, t) -> (rho, theta, t). Also can be defined as:
            exp(asymmetry_parameter * (R ** 2 - R_lfs ** 2)), where R is the major
            radius as a function of (rho, theta, t) and R_lfs is the low-field-side
            major radius as a function of (rho, t). xarray DataArray with dimensions
            (rho, theta, t)
        time_correlation
            Boolean to indicate whether or not to use time correlated guesses during
            the optimization (ie. the result of the optimization for the previous
            time-step is used as a guess for the next time-step.)

        Returns
        -------
        fitted_density
            New density with an optimized perturbation on the low-field-side that
            matches the original bolometry data. xarray DataArray with dimensions
            (rho, theta, t)
        """

        input_check(
            "extrapolated_smooth_data",
            extrapolated_smooth_data,
            DataArray,
            ndim_to_check=3,
            greater_than_or_equal_zero=True,
        )

        input_check(
            "orig_bolometry_data",
            orig_bolometry_data,
            DataArray,
            ndim_to_check=2,
            greater_than_or_equal_zero=True,
        )

        input_check(
            "asymmetry_modifier",
            asymmetry_modifier,
            DataArray,
            ndim_to_check=3,
            greater_than_or_equal_zero=True,
        )

        rho_arr = self.rho_arr
        drho = np.max(np.diff(rho_arr))

        # Check whether bolometry_obj as been called at least once
        # (ie. does it have a trimmed variant of the LoS bolometry data.)
        if hasattr(bolometry_obj, "LoS_bolometry_data_trimmed"):
            trim = True
            orig_bolometry_trimmed = []
            for bolo_diag, bolo_los in zip(
                orig_bolometry_data.coords["bolo_kb5v_coords"],
                [i[6] for i in bolometry_obj.LoS_bolometry_data],
            ):
                LoS_bolometry_data_trimmed_labels = [
                    i[6] for i in bolometry_obj.LoS_bolometry_data_trimmed
                ]
                if bolo_los in LoS_bolometry_data_trimmed_labels:
                    orig_bolometry_trimmed.append(orig_bolometry_data.loc[:, bolo_diag])
            orig_bolometry = concat(
                orig_bolometry_trimmed, dim="bolo_kb5v_coords"
            ).assign_attrs(**orig_bolometry_data.attrs)
        else:
            trim = False
            orig_bolometry = orig_bolometry_data

        extrapolated_smooth_data_mean = np.mean(
            extrapolated_smooth_data.loc[self.threshold_rho[0] :, :, :]
        )

        def objective_func(objective_array: Sequence, time: float):
            """Objective function that is passed to scipy.optimize.least_squares

            Parameters
            ----------
            objective_array
                List of [amplitude, standard deviation and position] defining the
                Gaussian perturbation.
            time
                Float specifying the time point of interest.

            Returns
            -------
            abs_error
                Absolute error between the derived bolometry data (with a given
                Gaussian perturbation) and the original bolometry data (ground truth)
                for the selected time point.
                xarray.DataArray with dimensions (channels)
            """
            amplitude, standard_dev, position = objective_array

            perturbation_signal = self.fitting_function(
                amplitude, standard_dev, position
            )

            # trim perturbation_signal to only be valid within rho = 0.0 and rho = 1.0
            perturbation_signal = perturbation_signal.interp(
                rho_poloidal=rho_arr, method="linear"
            )

            perturbation_signal = perturbation_signal * asymmetry_modifier.sel(t=time)

            modified_density = (
                extrapolated_smooth_data.sel(t=time) + perturbation_signal
            )

            bolometry_obj.impurity_densities.loc[
                impurity_element, :, :, time
            ] = modified_density

            # mypy unable to determine the length of bolometry_args so is getting
            # confused about whether t_val is included in bolometry_args or not,
            # hence ignored
            modified_bolometry_data = bolometry_obj(  # type:ignore
                deriv_only=True, trim=trim, t_val=time
            )

            comparison_orig_bolometry_data = orig_bolometry.sel(
                t=time, method="nearest", drop=True
            )

            error = np.abs(
                modified_bolometry_data.data - comparison_orig_bolometry_data.data
            )

            error = np.nan_to_num(error)

            return error

        fitted_density = zeros_like(bolometry_obj.electron_density)

        lower_amp_bound = 0.1 * extrapolated_smooth_data_mean
        upper_amp_bound = 1e4 * extrapolated_smooth_data_mean

        lower_width_bound = (drho / self.halfwidthhalfmax_coeff) / 3
        upper_width_bound = (
            (1.1 - self.threshold_rho[0].data) / self.fullwidthhalfmax_coeff
        ) / 3

        lower_pos_bound = self.threshold_rho[0].data + drho
        upper_pos_bound = 1.1

        initial_guesses = np.array(
            [
                [
                    np.sqrt(lower_amp_bound * upper_amp_bound),
                    np.mean([lower_width_bound, upper_width_bound]),
                    np.mean([lower_pos_bound, upper_pos_bound]),
                ]
            ]
        )

        fitting_bounds = [
            np.array(
                [
                    lower_amp_bound,
                    lower_width_bound,
                    lower_pos_bound,
                ]
            ),
            np.array(
                [
                    upper_amp_bound,
                    upper_width_bound,
                    upper_pos_bound,
                ]
            ),
        ]

        result = least_squares(
            fun=objective_func,
            x0=initial_guesses[0],
            bounds=tuple(fitting_bounds),
            max_nfev=50,
            args=(extrapolated_smooth_data.coords["t"].data[0],),
            ftol=1e-15,
            xtol=1e-60,
            gtol=1e-60,
        )

        gaussian_params = result.x

        if time_correlation:
            initial_guesses = np.append(
                initial_guesses, np.array([gaussian_params]), axis=0
            )

        for ind_t, it in enumerate(extrapolated_smooth_data.coords["t"].data[1:]):
            upper_width_bound = (
                (1.1 - self.threshold_rho[ind_t].data) / self.fullwidthhalfmax_coeff
            ) / 3

            lower_pos_bound = self.threshold_rho[ind_t].data + drho

            fitting_bounds[0][2] = lower_pos_bound
            fitting_bounds[1][1] = upper_width_bound

            if time_correlation:
                result = least_squares(
                    fun=objective_func,
                    x0=initial_guesses[ind_t],
                    bounds=tuple(fitting_bounds),
                    max_nfev=50,
                    args=(it,),
                    ftol=1e-60,
                    xtol=1e-3,
                    gtol=1e-60,
                )

                gaussian_params = result.x

                initial_guesses = np.append(
                    initial_guesses, np.array([gaussian_params]), axis=0
                )
            else:
                result = least_squares(
                    fun=objective_func,
                    x0=initial_guesses[0],
                    bounds=tuple(fitting_bounds),
                    max_nfev=15,
                    args=(it,),
                    ftol=1e-15,
                    xtol=1e-60,
                    gtol=1e-60,
                )

                gaussian_params = result.x

        fitted_density = bolometry_obj.impurity_densities.loc[impurity_element, :, :, :]
        fitted_density = fitted_density.transpose("rho_poloidal", "theta", "t")

        return fitted_density

    def __call__(  # type: ignore
        self,
        impurity_density_sxr: DataArray,
        electron_density: DataArray,
        electron_temperature: DataArray,
        truncation_threshold: float,
        flux_surfaces: FluxSurfaceCoordinates,
        asymmetry_parameter: DataArray = None,
        t: DataArray = None,
    ):
        """Extrapolates the impurity density beyond the limits of SXR (Soft X-ray)

        Parameters
        ----------
        impurity_density_sxr
            xarray.DataArray of impurity density derived from soft X-ray emissivity.
            Dimensions (R, z, t)
        electron_density
            xarray.DataArray of electron density. Dimensions (rho ,t)
        electron_temperature
            xarray.DataArray of electron temperature. Dimensions (rho, t)
        truncation_threshold
            Truncation threshold (float) for the electron temperature
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.
        asymmetry_parameter
            Optional, asymmetry parameter (either externally sourced or pre-calculated),
            xarray.DataArray with dimensions (rho, t)
        t
            Optional float, time at which the impurity concentration is to
            be calculated at.

        Returns
        -------
        extrapolated_smooth_density_R_z
            Extrapolated and smoothed impurity density ((R, z) grid).
            xarray.DataArray with dimensions (R, z, t)
        extrapolated_smooth_density_rho_theta
            Extrapolated and smoothed impurity density ((rho, theta) grid).
            xarray.DataArray with dimensions (rho, theta, t).
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
            greater_than_or_equal_zero=False,
        )

        input_check(
            "truncation_threshold",
            truncation_threshold,
            float,
            greater_than_or_equal_zero=False,
        )

        input_check("flux_surfaces", flux_surfaces, FluxSurfaceCoordinates)

        if t is None:
            t = electron_density.t
        else:
            input_check("t", t, DataArray, greater_than_or_equal_zero=True)

        self.threshold_rho = recover_threshold_rho(
            truncation_threshold, electron_temperature
        )

        # Transform impurity_density_sxr to (rho, theta) coordinates
        rho_arr = electron_density.coords["rho_poloidal"]
        t_arr = t

        (
            impurity_density_sxr_rho_theta,
            R_deriv,
            z_deriv,
        ) = self.transform_to_rho_theta(
            impurity_density_sxr,
            flux_surfaces,
            rho_arr,
            t_arr=t_arr,
        )

        # Continue impurity_density_sxr following the shape of the electron density
        # profile and mitigate discontinuity.

        extrapolated_impurity_density = self.basic_extrapolation(
            impurity_density_sxr_rho_theta, electron_density, self.threshold_rho
        )

        assert np.all(np.logical_not(np.isnan(extrapolated_impurity_density)))

        # Smoothing extrapolated data at the discontinuity.
        # (There is still a discontinuity in the radial gradient.)

        extrapolated_smooth_lfs, extrapolated_smooth_hfs = self.extrapolation_smoothing(
            extrapolated_impurity_density, rho_arr
        )

        if asymmetry_parameter is None:
            asymmetry_parameter = asymmetry_from_R_z(
                impurity_density_sxr, flux_surfaces, rho_arr, self.threshold_rho, t_arr
            )
        else:
            input_check(
                "asymmetry_parameter",
                asymmetry_parameter,
                DataArray,
                ndim_to_check=2,
                greater_than_or_equal_zero=True,
            )

        # Applying the asymmetry parameter to extrapolated density.
        # Also extends the data beyond the hfs and lfs to be
        # the full poloidal angle range.

        (
            extrapolated_smooth_density_rho_theta,
            asymmetry_modifier,
        ) = self.apply_asymmetry(
            asymmetry_parameter,
            extrapolated_smooth_hfs,
            extrapolated_smooth_lfs,
            R_deriv,
        )

        # Transform extrapolated density back onto a (R, z) grid

        extrapolated_smooth_density_R_z = self.transform_to_R_z(
            R_deriv, z_deriv, extrapolated_smooth_density_rho_theta, flux_surfaces
        )

        self.asymmetry_modifier = asymmetry_modifier

        return (
            extrapolated_smooth_density_R_z,
            extrapolated_smooth_density_rho_theta,
            t,
        )
