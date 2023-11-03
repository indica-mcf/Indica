from typing import Sequence
from typing import Tuple

import numpy as np
import scipy as sp
import xarray as xr

from indica import session
from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.datatypes import DataType
from indica.utilities import coord_array
from .abstractoperator import Operator
from .extrapolate_impurity_density import asymmetry_modifier_from_parameter


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
        n_additional_high_z_unnormalised_fsa_density
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

        n_additional_high_z_unnormalised_fsa_density = np.exp(
            sp.integrate.cumulative_trapezoid(integrand_points, x=rho, initial=0)
        )

        n_additional_high_z_unnormalised_fsa_density = xr.DataArray(
            data=n_additional_high_z_unnormalised_fsa_density,
            coords=n_high_z_midplane.coords,
            dims=n_high_z_midplane.dims,
        )

        # fix density to 0 at rho=1 if present
        n_additional_high_z_unnormalised_fsa_density = (
            n_additional_high_z_unnormalised_fsa_density.where(
                n_additional_high_z_unnormalised_fsa_density.coords["rho_poloidal"]
                != 1,
                other=0,
            )
        )

        return n_additional_high_z_unnormalised_fsa_density

    def _calc_unnormalised_additional_high_z_density(
        n_additional_high_z_unnormalised_fsa: xr.DataArray,
        toroidal_rotations: xr.DataArray,
        ion_temperature: xr.DataArray,
        main_ion: str,
        impurity: str,
        Zeff: xr.DataArray,
        electron_temp: xr.DataArray,
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

        Returns
        -------
        n_additional_high_z_unnormalised_midplane
            Additional high Z impurity density along the midplane. Dimensions (t, rho).
        additional_high_z_asymmetry_parameter
            Additional high Z impurity asymmetry parameter. Dimensions (t, rho).
        """
        raise NotImplementedError

    def _calc_normalised_additional_high_z_density(
        n_additional_high_z_unnormalised_midplane: xr.DataArray,
        additional_high_z_asymmetry_parameter: xr.DataArray,
        flux_surfs: FluxSurfaceCoordinates,
        LoS_bolometry_data: Sequence,
        # TODO: take bolometry object rather than impurity_densities
        impurity_densities: xr.DataArray,
        frac_abunds: xr.DataArray,
        impurity_elements: Sequence[str],
        electron_density: xr.DataArray,
        power_loss: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculate the normalised additional high Z density using the bolometry data.

        Implements equations 2.9 and 2.10 from the main paper.

        Parameters
        ----------
        flux_surfs
            FluxSurfaceCoordinates object representing polar coordinate systems
            using flux surfaces for the radial coordinate.
        LoS_bolometry_data
            Line-of-sight bolometry data in the same format as given in:
            tests/unit/operator/KB5_Bolometry_data.py
        impurity_densities
            Densities for all impurities. Dimensions (elements, t, rho, theta).
        frac_abunds
            Fractional abundances list of fractional abundances.
            Dimensions  (element, ion_charges, t, rho).
        impurity_elements
            List of element symbols(as strings) for all impurities.
        electron_density
            Electron density. Dimensions (t, rho)
        power_loss
            Power loss associated with each ion.
            Dimensions (elements, t, rho).

        Returns
        -------
        n_additional_high_z_midplane
            Normalised midplane additional high Z density. Dimensions (t, rho).
        """
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
