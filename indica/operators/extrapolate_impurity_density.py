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

    def __init__(
        self,
        sess: session.Session = session.global_session,
    ):
        """Initialises ExtrapolateImpurityDensity class."""
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

        Returns
        -------
        threshold_rho
            rho value beyond which the electron temperature falls below
            the truncation_threshold.
        """

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

        self.threshold_rho = threshold_rho

        return threshold_rho

    def transform_to_rho_theta_reduced(
        self,
        data_R_z: DataArray,
        flux_surfaces: FluxSurfaceCoordinates,
        rho_arr: DataArray,
        t_arr: DataArray = None,
    ):
        """Function to transform data from an (R, z) grid to a (rho, theta) grid

        Parameters
        ----------
        data_R_z
            Data on (R, z) grid to be transformed.
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.
        rho_arr
            1D array of rho from 0 to 1.
        t_arr
            1D array of t.

        Returns
        -------
        data_rho_theta
            Data on (rho, theta) grid.
        R_deriv
            Variable describing value of R in every coordinate on a (rho, theta) grid.
            (Used in derive_and_apply_asymmetry hence is returned by this function.)
        derived_asymmetry_parameter
            Derived asymmetry parameter (needed for __call__ which returns it)
        """
        if t_arr is None:
            t_arr = data_R_z.coords["t"]

        theta_arr = np.array([0.0, np.pi])
        theta_arr = DataArray(
            data=theta_arr, coords={"theta": theta_arr}, dims=["theta"]
        )

        R_deriv, z_deriv = flux_surfaces.convert_to_Rz(rho_arr, theta_arr)
        R_deriv = cast(DataArray, R_deriv).interp(t=t_arr, method="linear")
        z_deriv = cast(DataArray, z_deriv).interp(t=t_arr, method="linear")

        R_deriv = cast(DataArray, R_deriv).transpose("rho", "theta", "t")
        z_deriv = cast(DataArray, z_deriv).transpose("rho", "theta", "t")

        data_rho_theta = data_R_z.indica.interp2d(
            {"R": R_deriv, "z": z_deriv}, method="linear", assume_sorted=True
        )
        data_rho_theta = data_rho_theta.transpose("rho", "theta", "t")

        derived_asymmetry_parameter = np.log(
            data_rho_theta.isel(theta=1) / data_rho_theta.isel(theta=0)
        )

        R_lfs_midplane = cast(DataArray, R_deriv).isel(theta=0)  # theta = 0.0
        R_hfs_midplane = cast(DataArray, R_deriv).isel(theta=1)  # theta = np.pi

        derived_asymmetry_parameter /= R_hfs_midplane**2 - R_lfs_midplane**2

        derived_asymmetry_parameter.loc[0, :] = 1.0
        derived_asymmetry_parameter.loc[1.0, :] = 0.0

        return data_rho_theta, R_deriv, derived_asymmetry_parameter

    def basic_extrapolation(
        self,
        data_rho_theta: DataArray,
        electron_density: DataArray,
        threshold_rho: float,
    ):
        """Basic extrapolation which eliminates the data_rho_theta for
        rho > threshold_rho and joins on electron density from that point
        outwards (in rho). Also multiplies electron density to prevent a
        discontinuity at rho=threshold_rho.

        Parameters
        ----------
        data_rho_theta
            Data to extrapolate on (rho, theta) grid.
        electron_density
            Electron density (axisymmetric)
        threshold_rho
            Threshold value of rho beyond which SXR diagnostics cannot
            be used to accurately infer impurity density.

        Returns
        -------
        extrapolated_impurity_density
            Extrapolated impurity density
        """

        theta_arr = np.array([0.0, np.pi])
        theta_arr = DataArray(
            data=theta_arr, coords={"theta": theta_arr}, dims=["theta"]
        )

        boundary_electron_density = electron_density.sel(
            {"rho": threshold_rho}
        ).squeeze()
        boundary_data = data_rho_theta.sel({"rho": threshold_rho}).squeeze()

        discontinuity_scale = boundary_data / boundary_electron_density

        # Continue impurity_density_sxr following the shape of the electron density
        # profile.
        bounded_data = data_rho_theta.where(
            data_rho_theta.rho <= threshold_rho, drop=True
        )

        bounded_electron_density = electron_density.where(
            electron_density.rho >= threshold_rho, drop=True
        )

        bounded_data_trim = bounded_data[:-1]

        # bounded_data = bounded_data.drop_isel(rho=index_to_drop)

        extrapolated_impurity_density = concat(
            (
                bounded_data_trim,
                bounded_electron_density * discontinuity_scale,
            ),
            dim="rho",
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
            Extrapolated data to be smoothed (on (rho, theta) grid).
        rho_arr
            rho array used to construct smoothing splines.
            (Must be higher resolution than the one obtained from extrapolated_data)

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
            ).var("rho")

            variance_extrapolated_data_hfs = extrapolated_data.isel(
                {"t": ind_t, "theta": 1}
            ).var("rho")

            extrapolated_spline_lfs = UnivariateSpline(
                rho_arr,
                extrapolated_data[:, 0, ind_t],
                k=5,
                s=0.001 * variance_extrapolated_data_lfs,
            )

            extrapolated_spline_hfs = UnivariateSpline(
                rho_arr,
                extrapolated_data[:, 1, ind_t],
                k=5,
                s=0.001 * variance_extrapolated_data_hfs,
            )

            extrapolated_smooth_lfs.append(extrapolated_spline_lfs(rho_arr, 0))
            extrapolated_smooth_hfs.append(extrapolated_spline_hfs(rho_arr, 0))

        extrapolated_smooth_lfs_arr = DataArray(
            data=extrapolated_smooth_lfs,
            coords={"t": t, "rho": rho_arr},
            dims=["t", "rho"],
        )

        extrapolated_smooth_hfs_arr = DataArray(
            data=extrapolated_smooth_hfs,
            coords={"t": t, "rho": rho_arr},
            dims=["t", "rho"],
        )

        extrapolated_smooth_lfs_arr = extrapolated_smooth_lfs_arr.transpose("rho", "t")
        extrapolated_smooth_hfs_arr = extrapolated_smooth_hfs_arr.transpose("rho", "t")

        inv_extrapolated_smooth_hfs = DataArray(
            data=np.flip(extrapolated_smooth_hfs_arr.data, axis=0),
            coords={
                "rho": -1 * np.flip(extrapolated_smooth_hfs_arr.coords["rho"].data),
                "t": extrapolated_smooth_hfs_arr.coords["t"].data,
            },
            dims=["rho", "t"],
        )

        inv_rho_arr = inv_extrapolated_smooth_hfs.coords["rho"].data
        inv_del_val = inv_rho_arr[-1]

        inv_extrapolated_smooth_hfs = inv_extrapolated_smooth_hfs.drop_sel(
            rho=inv_del_val
        )

        extrapolated_smooth_mid_plane_arr = concat(
            (inv_extrapolated_smooth_hfs, extrapolated_smooth_lfs_arr), "rho"
        )

        drho = np.mean(np.diff(extrapolated_smooth_mid_plane_arr.coords["rho"].data))
        smooth_central_region = extrapolated_smooth_mid_plane_arr.loc[
            -2 * drho : 2 * drho
        ]

        smooth_central_region.loc[:, :] = smooth_central_region.max(dim="rho")

        extrapolated_smooth_mid_plane_arr.loc[
            -2 * drho : 2 * drho
        ] = smooth_central_region

        inv_extrapolated_smooth_hfs = extrapolated_smooth_mid_plane_arr.loc[:drho]

        extrapolated_smooth_hfs_arr = DataArray(
            data=np.flip(inv_extrapolated_smooth_hfs.data, axis=0),
            coords=extrapolated_smooth_hfs_arr.coords,
            dims=extrapolated_smooth_hfs_arr.dims,
        )

        # Ignoring mypy warning since it seems to be unaware that the xarray .loc
        # method uses label-based indexing and slicing instead of integer-based.
        extrapolated_smooth_lfs_arr = extrapolated_smooth_mid_plane_arr.loc[
            0.0:  # type: ignore
        ]

        return extrapolated_smooth_lfs_arr, extrapolated_smooth_hfs_arr

    def derive_and_apply_asymmetry(
        self,
        R_deriv: DataArray,
        extrapolated_smooth_hfs: DataArray,
        extrapolated_smooth_lfs: DataArray,
        flux_surfaces: FluxSurfaceCoordinates,
    ):
        """Deriving asymmetry parameter from low-field-side and
        high-field-side data and applying it to low-field-side data which
        has been extended over the poloidal extent to obtain an asymmetric
        extrapolated smoothed data on a (rho, theta) grid.

        Parameters
        ----------
        R_deriv
            Variable describing value of R in every coordinate on a (rho, theta) grid.
            (from transform_to_rho_theta_reduced)
        extrapolated_smooth_hfs
            Extrapolated smoothed data on high-field side (fixed theta = pi)
        extrapolated_smooth_lfs
            Extrapolated smoothed data on low-field side (fixed theta = 0)
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.

        Returns
        -------
        extrapolated_smooth_data
            Extrapolated and smoothed data on full (rho, theta) grid.
        R_deriv
            Variable describing value of R in every coordinate on a (rho, theta) grid.
            (uniquely calculated by this function used for transform_to_R_z)
        z_deriv
            Variable describing value of z in every coordinate on a (rho, theta) grid.
            (uniquely calculated by this function used for transform_to_R_z)
        derived_asymmetry_parameter
            Derived asymmetry parameter (needed for __call__ which returns it)
        """
        rho_arr = extrapolated_smooth_hfs.coords["rho"]
        self.rho_arr = rho_arr
        t_arr = extrapolated_smooth_hfs.coords["t"]

        R_lfs_midplane = cast(DataArray, R_deriv).isel(theta=0)  # theta = 0.0
        R_hfs_midplane = cast(DataArray, R_deriv).isel(theta=1)  # theta = np.pi

        derived_asymmetry_parameter = np.log(
            cast(DataArray, extrapolated_smooth_hfs)
            / cast(DataArray, extrapolated_smooth_lfs)
        )

        asym_denominator = R_hfs_midplane**2 - R_lfs_midplane**2

        derived_asymmetry_parameter /= asym_denominator

        # Set constant asymmetry parameter for rho<0.1
        derived_asymmetry_parameter.loc[
            0:0.1, :  # type: ignore
        ] = derived_asymmetry_parameter.loc[0.1, :]

        theta_arr = np.linspace(-np.pi, np.pi, 21)
        theta_arr = DataArray(theta_arr, {"theta": theta_arr}, ["theta"])
        R_deriv_, z_deriv_ = flux_surfaces.convert_to_Rz(rho_arr, theta_arr)
        R_deriv = cast(DataArray, R_deriv_).interp(t=t_arr, method="linear")
        z_deriv = cast(DataArray, z_deriv_).interp(t=t_arr, method="linear")
        R_deriv = cast(DataArray, R_deriv).transpose("rho", "theta", "t")
        z_deriv = cast(DataArray, z_deriv).transpose("rho", "theta", "t")

        asymmetry_modifier = np.exp(
            derived_asymmetry_parameter * (R_deriv**2 - R_lfs_midplane**2)
        )

        asymmetry_modifier = asymmetry_modifier.transpose("rho", "theta", "t")

        extrapolated_smooth_data = extrapolated_smooth_lfs * asymmetry_modifier
        extrapolated_smooth_data = extrapolated_smooth_data.transpose(
            "rho", "theta", "t"
        )

        return extrapolated_smooth_data, R_deriv, z_deriv, derived_asymmetry_parameter

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
            (from derive_and_apply_asymmetry)
        z_deriv
            Variable describing value of z in every coordinate on a (rho, theta) grid.
            (from derive_and_apply_asymmetry)
        extrapolated_smooth_data
            Extrapolated and smoothed data to transform onto (R, z) grid.
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.

        Returns
        -------
        extrapolated_smooth_data
            Extrapolated and smoothed data on (R, z) grid.
        """
        R_arr = np.linspace(np.min(R_deriv[1:]), np.max(R_deriv[1:]), 40)
        z_arr = np.linspace(np.min(z_deriv), np.max(z_deriv), 40)

        R_arr = DataArray(R_arr, {"R": R_arr}, ["R"])
        z_arr = DataArray(z_arr, {"z": z_arr}, ["z"])

        t_arr = extrapolated_smooth_data.coords["t"]

        rho_derived, theta_derived = flux_surfaces.convert_from_Rz(R_arr, z_arr, t_arr)
        rho_derived = cast(DataArray, rho_derived).transpose("R", "z", "t")
        theta_derived = cast(DataArray, theta_derived).transpose("R", "z", "t")
        rho_derived = abs(rho_derived)

        extrapolated_smooth_data = extrapolated_smooth_data.indica.interp2d(
            {"rho": rho_derived, "theta": theta_derived},
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
            DataArray containing the Gaussian signal with dimensions (rho,)
        """
        rho_arr = self.rho_arr

        gaussian_signal = norm(loc=position, scale=standard_dev)

        sig = gaussian_signal.pdf(rho_arr)

        sig = DataArray(data=sig, coords={"rho": rho_arr}, dims=["rho"])

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

        # bolometry_args[4] = self.bolometry_setup(*bolometry_setup_args)

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
                Gaussian perturbation) and the original bolometry data (ground truth).
            """
            amplitude, standard_dev, position = objective_array

            perturbation_signal = self.fitting_function(
                amplitude, standard_dev, position
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
                deriv_only=True, trim=True, t_val=time
            )

            abs_error = np.abs(
                modified_bolometry_data - orig_bolometry_data.sel(t=time)
            )  # / orig_bolometry_data.sel(t=time)

            abs_error = np.nan_to_num(abs_error)

            return abs_error

        fitted_density = zeros_like(bolometry_obj.electron_density)

        initial_guesses = np.array([[0.2e17, 0.2, 0.8]])

        # Initial fitting for first time-step.
        result = least_squares(
            fun=objective_func,
            x0=initial_guesses[0],
            bounds=(
                np.array([0.1e17, 0.1, self.threshold_rho[0]]),
                np.array([2.0e17, 0.4, 0.95]),
            ),
            max_nfev=10,
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

        perturbation_signal = self.fitting_function(*gaussian_params)
        fitted_density.loc[:, extrapolated_smooth_data.coords["t"].data[0]] = (
            extrapolated_smooth_data.sel(
                theta=0,
                t=extrapolated_smooth_data.coords["t"].data[0],
                method="nearest",
            )
            + perturbation_signal
        )

        for ind_t, it in enumerate(extrapolated_smooth_data.coords["t"].data[1:]):
            if time_correlation:
                result = least_squares(
                    fun=objective_func,
                    x0=initial_guesses[ind_t],
                    bounds=(
                        np.array([0.1e17, 0.1, self.threshold_rho[0]]),
                        np.array([2.0e17, 0.4, 0.95]),
                    ),
                    max_nfev=50,
                    args=(it,),
                    ftol=1e-60,
                    xtol=5e-2,
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
                    bounds=(
                        np.array([0.1e17, 0.1, self.threshold_rho[0]]),
                        np.array([2.0e17, 0.4, 0.95]),
                    ),
                    max_nfev=10,
                    args=(it,),
                    ftol=1e-15,
                    xtol=1e-60,
                    gtol=1e-60,
                )

                gaussian_params = result.x

            perturbation_signal = self.fitting_function(*gaussian_params)
            fitted_density.loc[:, it] = (
                extrapolated_smooth_data.sel(theta=0, t=it, method="nearest")
                + perturbation_signal
            )

        fitted_density = fitted_density * asymmetry_modifier
        fitted_density = fitted_density.transpose("rho", "theta", "t")

        return fitted_density

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
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.
        t
            Optional, time at which the impurity concentration is to be calculated at.

        Returns
        -------
        extrapolated_smooth_density_Rz
            Extrapolated and smoothed impurity density ((R, z) grid).
        asym_par
            Derived asymmetry parameter.
        t
            If ``t`` was not specified as an argument for the __call__ function,
            return the time the results are given for.
            Otherwise return the argument.
        extrapolated_smooth_density
            Extrapolated and smoothed impurity density ((rho, theta) grid).
        asymmetry_modifier
            Asymmetry modifier used to transform a low-field side only rho-profile
            of a poloidally asymmetric quantity to a full poloidal cross-sectional
            profile ie. (rho, t) -> (rho, theta, t). Also can be defined as:
            exp(asymmetry_parameter * (R ** 2 - R_lfs ** 2)), where R is the major
            radius as a function of (rho, theta, t) and R_lfs is the low-field-side
            major radius as a function of (rho, t). xarray DataArray with dimensions
            (rho, theta, t)
        R_deriv
            Variable describing value of R in every coordinate on a (rho, theta) grid.
            (from derive_and_apply_asymmetry)
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

        threshold_rho = self.recover_rho(truncation_threshold, electron_temperature)

        # Transform impurity_density_sxr to (rho, theta) coordinates
        rho_arr = electron_density.coords["rho"]
        t_arr = t

        (
            impurity_density_sxr,
            R_deriv,
            derived_asymmetry_parameter_full,
        ) = self.transform_to_rho_theta_reduced(
            impurity_density_sxr,
            flux_surfaces,
            rho_arr,
            t_arr=t_arr,
        )

        # Continue impurity_density_sxr following the shape of the electron density
        # profile and mitigate discontinuity.

        extrapolated_impurity_density = self.basic_extrapolation(
            impurity_density_sxr, electron_density, threshold_rho
        )

        assert np.all(np.logical_not(np.isnan(extrapolated_impurity_density)))

        # Smoothing extrapolated data at the discontinuity.
        # (There is still a discontinuity in the radial gradient.)

        extrapolated_smooth_lfs, extrapolated_smooth_hfs = self.extrapolation_smoothing(
            extrapolated_impurity_density, rho_arr
        )

        # Deriving and applying the asymmetry parameter to extrapolated density.
        # Also extends the data beyond the hfs and lfs to be
        # the full poloidal angle range.

        (
            extrapolated_smooth_density,
            R_deriv,
            z_deriv,
            derived_asymmetry_parameter_sxr_only,
        ) = self.derive_and_apply_asymmetry(
            R_deriv, extrapolated_smooth_hfs, extrapolated_smooth_lfs, flux_surfaces
        )

        # Transform extrapolated density back onto a (R, z) grid

        extrapolated_smooth_density_Rz = self.transform_to_R_z(
            R_deriv, z_deriv, extrapolated_smooth_density, flux_surfaces
        )

        R_lfs = R_deriv.interp(theta=0, method="linear")

        # Switch determining whether to use asymmetry parameter derived from
        # valid range of the soft x-ray derived impurity density given on input (True)
        # or to use the asymmetry parameter derived from all of the soft x-ray
        # derived impurity density (False).
        asym_sxr_switch = True
        # asym_sxr_switch = False

        if asym_sxr_switch:
            asym_par = derived_asymmetry_parameter_sxr_only
        else:
            asym_par = derived_asymmetry_parameter_full

        asymmetry_modifier = np.exp(asym_par * (R_deriv**2 - R_lfs**2))
        asymmetry_modifier = asymmetry_modifier.transpose("rho", "theta", "t")

        return (
            extrapolated_smooth_density_Rz,
            asym_par,
            t,
            extrapolated_smooth_density,
            asymmetry_modifier,
            R_deriv,
        )
