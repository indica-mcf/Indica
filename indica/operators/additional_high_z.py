from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import scipy as sp
import xarray as xr

from indica import session
from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.datatypes import DataType
from indica.utilities import coord_array
from .abstractoperator import Operator
from .bolometry_derivation import BolometryDerivation
from .centrifugal_asymmetry import AsymmetryParameter
from .extrapolate_impurity_density import asymmetry_modifier_from_parameter


def bolo_los(bolo_diag_array: xr.DataArray) -> List[List[Union[List, str]]]:
    return [
        [
            np.array([bolo_diag_array.attrs["transform"].x_start.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].z_start.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].y_start.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].x_end.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].z_end.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].y_end.data[i].tolist()]),
            "bolo_kb5" + str(i.values),
        ]
        for i in bolo_diag_array.bolo_kb5v_coords
    ]


def calc_fsa_quantity(
    symmetric_component: xr.DataArray,
    asymmetry_parameter: xr.DataArray,
    flux_surfaces: FluxSurfaceCoordinates,
    ntheta: int = 12,
) -> xr.DataArray:
    """
    Calculate the flux surface average (FSA) quantity from the symmetric component and
    asymmetry parameter.

    Parameters
    ----------
    symmetric_component
        Symmetric component of the quantity to average. Dimensions (t, rho).
    asymmetry_parameter
        Asymmetric parameter for the quantity to average. Dimension (t, rho).
    flux_surfaces
        FluxSurfaceCoordinates object representing polar coordinate systems
        using flux surfaces for the radial coordinate.
    ntheta
        Number of angular position points to use while approximating FSA.

    Returns
    -------
    fsa_quantity
        Flux surface averaged quantity
    """
    rho = symmetric_component.coords["rho_poloidal"]
    theta = coord_array(np.linspace(0.0, 2.0 * np.pi, ntheta), "theta")

    R_mid, z_mid = flux_surfaces.convert_to_Rz(rho, theta)

    asymmetry_modifier = asymmetry_modifier_from_parameter(asymmetry_parameter, R_mid)
    # quantity evaluated at midpoint between low and high on flux surface
    quantity_mid = symmetric_component * asymmetry_modifier

    # TODO: consider using a surface area weighted average
    return quantity_mid.mean(dim="theta")

    # theta_low = theta - np.pi / ntheta
    # theta_high = theta + np.pi / ntheta
    #
    # R_low, z_low = flux_surfaces.convert_to_Rz(rho, theta_low)
    # R_high, z_high = flux_surfaces.convert_to_Rz(rho, theta_high)
    #
    # dl = np.hypot(R_high - R_low, z_high - z_low)
    #
    # # FSA is surface element weighted average
    # # 2*pi factor from rotation Z axis cancels
    # return (quantity_mid * dl * R_mid).sum(dim="theta") / (R_mid * dl).sum(
    #     dim="theta"
    # )


class AdditionalHighZ(Operator):
    def __init__(
        self,
        sess: session.Session = session.global_session,
    ):
        super().__init__(sess=sess)

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        return (("number_density", "impurity_element"),)

    def _calc_shape(
        n_high_z_midplane: xr.DataArray,
        n_high_z_asymmetry_parameter: xr.DataArray,
        q_high_z: xr.DataArray,
        q_additional_high_z: xr.DataArray,
        flux_surfaces: FluxSurfaceCoordinates,
    ) -> xr.DataArray:
        """
        Calculate the flux surface averaged additional high Z impurity density
        unnormalised shape.

        Implements equation 2.7 from the main paper.

        Parameters
        ----------
        n_high_z_midplane
            Density of the main high Z impurity element, usually Tungsten along
            the midplane. Dimensions (t, rho).
        n_high_z_asymmetry_parameter
            Asymmetry parameter for the main high Z impurity. Dimensions (t, rho).
        q_high_z
            Impurity charge for the main high Z impurity. Dimensions (t, rho).
        q_additional_high_z
            Impurity charge for the additional high Z impurity. Dimensions (t, rho).
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.

        Returns
        -------
        n_additional_high_z_unnormalised_fsa
            Flux surface averaged additional high Z impurity density.
            Dimensions (t, rho).
        """
        n_high_z_fsa = calc_fsa_quantity(
            n_high_z_midplane, n_high_z_asymmetry_parameter, flux_surfaces
        )

        # Use a spline to get derivative
        n_high_z_fsa_spline = sp.interpolate.CubicSpline(
            n_high_z_fsa.coords["rho_poloidal"],
            n_high_z_fsa,
            axis=1,
            bc_type="natural",
        )
        n_high_z_fsa_derivative = n_high_z_fsa_spline.derivative()

        rho = n_high_z_fsa.coords["rho_poloidal"]
        integrand_points = (n_high_z_fsa_derivative(rho) * q_additional_high_z) / (
            n_high_z_fsa * q_high_z
        )

        n_additional_high_z_unnormalised_fsa = np.exp(
            sp.integrate.cumulative_trapezoid(integrand_points, x=rho, initial=0)
        )

        n_additional_high_z_unnormalised_fsa = xr.DataArray(
            data=n_additional_high_z_unnormalised_fsa,
            coords=n_high_z_midplane.coords,
            dims=n_high_z_midplane.dims,
        )

        # fix density to 0 at rho=1 if present
        n_additional_high_z_unnormalised_fsa = (
            n_additional_high_z_unnormalised_fsa.where(
                n_additional_high_z_unnormalised_fsa.coords["rho_poloidal"] != 1,
                other=0,
            )
        )

        return n_additional_high_z_unnormalised_fsa

    def _calc_first_normalisation(
        n_additional_high_z_unnormalised_fsa: xr.DataArray,
        n_main_high_z_midplane: xr.DataArray,
        n_main_high_z_asymmetry_parameter: xr.DataArray,
        flux_surfaces: FluxSurfaceCoordinates,
    ) -> xr.DataArray:
        """
        Calculate the first normalisation to set the number of additional high Z ions
        equal to the number of main high Z ions.

        Implements equation 2.8 from the main paper.

        Parameters
        ----------
        n_additional_high_z_unnormalised_fsa
            Flux surface averaged additional high Z impurity density.
            Dimensions (t, rho).
        n_main_high_z_midplane
            Density of the main high Z impurity element, usually Tungsten along
            the midplane. Dimensions (t, rho).
        n_main_high_z_asymmetry_parameter
            Asymmetry parameter for the main high Z impurity. Dimensions (t, rho).
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.

        Returns
        -------
        n_additional_high_z_seminormalised_fsa
            Flux surface averaged additional high Z impurity density and partly
            normalised using the main high Z density.
            Dimensions (t, rho).
        """
        t = n_additional_high_z_unnormalised_fsa.coords["t"]
        rho = n_additional_high_z_unnormalised_fsa.coords["rho_poloidal"]

        n_main_high_z_fsa = calc_fsa_quantity(
            n_main_high_z_midplane, n_main_high_z_asymmetry_parameter, flux_surfaces
        )

        rho_arr = rho.data
        # get midpoints between rho points
        rho_mid_arr = (rho_arr[:-1] + rho_arr[1:]) / 2
        rho_mid_arr = np.append(rho_mid_arr, 1)
        rho_mid = coord_array(rho_mid_arr, "rho_poloidal")

        # find volume for each shell using diff from enclosed volume for each rho_mid
        cumulative_volumes, _, _ = flux_surfaces.equilibrium.enclosed_volume(rho_mid, t)
        rho_axis = cumulative_volumes.get_axis_num("rho_poloidal")
        volume_elems_arr = np.diff(cumulative_volumes, axis=rho_axis)
        # missing centre volume after diff, insert it
        volume_elems_arr = np.insert(
            volume_elems_arr,
            0,
            values=cumulative_volumes.isel(rho_poloidal=0),
            axis=rho_axis,
        )
        volume_elems = xr.DataArray(
            data=volume_elems_arr,
            dims=cumulative_volumes.dims,
            # shift volumes back onto main rho grid
            coords=n_main_high_z_fsa.coords,
        )

        # compute numbers of confined ions
        main_high_z_counts = (volume_elems * n_main_high_z_fsa).sum(dim="rho_poloidal")
        additional_high_z_counts = (
            volume_elems * n_additional_high_z_unnormalised_fsa
        ).sum(dim="rho_poloidal")

        return (
            (main_high_z_counts) / (additional_high_z_counts)
        ) * n_additional_high_z_unnormalised_fsa

    def _calc_seminormalised_additional_high_z_density(
        n_additional_high_z_unnormalised_fsa: xr.DataArray,
        toroidal_rotations: xr.DataArray,
        ion_temperature: xr.DataArray,
        main_ion: str,
        impurity: str,
        Zeff: xr.DataArray,
        electron_temp: xr.DataArray,
        flux_surfaces: FluxSurfaceCoordinates,
    ) -> xr.DataArray:
        """
        Calculate the unnormalised additional high Z density on the midplane
        and the asymmetry parameter.

        Uses equation 2.12 and implements 2.13.

        Parameters
        ----------
        n_additional_high_z_unnormalised_fsa
            Unnormalised additional high Z impurity density. Dimensions (t, rho).
        toroidal_rotations
            xarray.DataArray containing toroidal rotation frequencies data.
            In units of ms^-1.
        ion_temperature
            xarray.DataArray containing ion temperature data. In units of eV.
        main_ion
            Element symbol of main ion.
        impurity
            Element symbol of chosen impurity element.
        Zeff
            xarray.DataArray containing Z-effective data from diagnostics.
        electron_temp
            xarray.DataArray containing electron temperature data. In units of eV.
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.

        Returns
        -------
        n_additional_high_z_unnormalised_midplane
            Additional high Z impurity density along the midplane. Dimensions (t, rho).
        n_additional_high_z_asymmetry_parameter
            Additional high Z impurity asymmetry parameter. Dimensions (t, rho).
        """
        # first calculate asymmetry_parameter
        asym_operator = AsymmetryParameter()
        n_additional_high_z_asymmetry_parameter = asym_operator(
            toroidal_rotations,
            ion_temperature,
            main_ion,
            impurity,
            Zeff,
            electron_temp,
        )

        # second convert fsa density to midplane density
        rho = n_additional_high_z_unnormalised_fsa.coords["rho_poloidal"]
        # number of points to use for fsa to midplane conversion
        ntheta = 12
        theta = coord_array(np.linspace(0.0, 2.0 * np.pi, ntheta), "theta")

        R_mid, z_mid = flux_surfaces.convert_to_Rz(rho, theta)

        asymmetry_modifier = asymmetry_modifier_from_parameter(
            n_additional_high_z_asymmetry_parameter, R_mid
        )
        # TODO: check maths for fsa of asymmetry modifier
        asymmetry_modifier_fsa = calc_fsa_quantity(
            asymmetry_modifier, xr.ones_like(asymmetry_modifier), flux_surfaces, ntheta
        )
        return (
            n_additional_high_z_unnormalised_fsa / asymmetry_modifier_fsa,
            n_additional_high_z_asymmetry_parameter,
        )

    def _calc_normalised_additional_high_z_density(
        n_additional_high_z_seminormalised_midplane: xr.DataArray,
        n_additional_high_z_asymmetry_parameter: xr.DataArray,
        additional_high_z_element: str,
        flux_surfaces: FluxSurfaceCoordinates,
        bolometry_observation: xr.DataArray,
        bolometry_obj: BolometryDerivation,
    ) -> xr.DataArray:
        """
        Calculate the normalised additional high Z density using the bolometry data.

        Implements equations 2.9 and 2.10 from the main paper.

        Parameters
        ----------
        n_additional_high_z_seminormalised_midplane
            Additional high Z impurity density along the midplane after first
            normalisation. Dimensions (t, rho).
        n_additional_high_z_asymmetry_parameter
            Additional high Z impurity asymmetry parameter. Dimensions (t, rho).
        additional_high_z_element
            Symbol for additional high Z impurity element.
        flux_surfaces
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.
        bolometry_observation
            Measured bolometry data used for normalisation. Dimensions (t, channel).
        bolometry_obj
            BolometryDerivation object.

        Returns
        -------
        n_additional_high_z_midplane
            Normalised midplane additional high Z density. Dimensions (t, rho).
        """
        rho = n_additional_high_z_seminormalised_midplane.coords["rho_poloidal"]
        theta = bolometry_obj.impurity_densities.coords["theta"]
        R_mid, z_mid = flux_surfaces.convert_to_Rz(rho, theta)
        asymmetry_modifier = asymmetry_modifier_from_parameter(
            n_additional_high_z_asymmetry_parameter, R_mid
        )
        # construct density on a rho, theta grid
        n_additional_high_z = (
            n_additional_high_z_seminormalised_midplane * asymmetry_modifier
        )

        # ensure data is transposed correctly
        n_additional_high_z = n_additional_high_z.transpose(
            "rho_poloidal", "theta", "t"
        )
        # insert density into bolometry obj
        bolometry_obj.impurity_densities.loc[
            additional_high_z_element, :, :, :
        ] = n_additional_high_z

        bolometry_prediction = bolometry_obj(False, False, t_val=None)

        C = 1 / (bolometry_prediction / bolometry_observation).mean()

        return C * n_additional_high_z_seminormalised_midplane

    def __call__(self):
        raise NotImplementedError
