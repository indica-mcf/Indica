import xarray as xr

from .abstractoperator import Operator
from .. import session


class AdditionalHighZ(Operator):
    def __init__(
        self,
        sess: session.Session = session.global_session,
    ):
        super().__init__(sess=sess)

    def _calc_shape(
        n_high_z_midplane: xr.DataArray,
        high_z_asymmetry_parameter: xr.DataArray,
        q_high_z: xr.DataArray,
        q_additional_high_z,
    ) -> xr.DataArray:
        """
        Calculate the flux surface averaged additional high Z impurity density.

        Implements equation 2.7 from the main paper.

        Parameters
        ----------
        n_high_z_midplane
            Density of the main high Z impurity element, usually Tungsten along
            the midplane. Dimensions (rho, t).
        high_z_asymmetry_parameter
            Asymmetry parameter for the main high Z impurity. Dimensions (rho, t).
        q_high_z
            Impurity charge for the main high Z impurity. Dimensions (rho, t).
        q_additional_high_z
            Impurity charge for the additional high Z impurity. Dimensions (rho, t).

        Returns
        -------
        n_additional_high_z_unnormalised_fsa_density
            Flux surface averaged additional high Z impurity density.
            Dimensions (rho, t).
        """
        raise NotImplementedError

    def _calc_unnormalised_additional_high_z_density(
        n_additional_high_z_unnormalised_fsa: xr.DaraArray,
        toroidal_rotations: xr.DataArray,
        ion_temperature: xr.DataArray,
        main_ion: str,
        impurity: str,
        Zeff: xr.DataArray,
        electron_temp: xr.DataArray,
    ) -> xr.DaraArray:
        """
        Calculate the unnormalised additional high Z density on the midplane
        and the asymmetry parameter.

        Uses equation 2.12 and implements 2.13.

        Parameters
        ----------
        n_additional_high_z_unnormalised_fsa
            Unnormalised additional high Z impurity density. Dimensions (rho, t).
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
            Additional high Z impurity density along the midplane. Dimensions (rho, t).
        additional_high_z_asymmetry_parameter
            Additional high Z impurity asymmetry parameter. Dimensions (rho, t).
        """
        raise NotImplementedError

    def _calc_normalised_additional_high_z_density(
        n_additional_high_z_unnormalised_midplane: xr.DataArray,
        additional_high_z_asymmetry_parameter: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculate the normalised additional high Z density using the bolometry data.

        Implements equations 2.9 and 2.10 from the main paper.

        Parameters
        ----------

        """
        raise NotImplementedError
