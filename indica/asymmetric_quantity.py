import numpy as np
import xarray as xr

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium


def check_grids(a: xr.DataArray, b: xr.DataArray):
    if a.rho_poloidal != b.rho_poloidal:
        raise ValueError("Values must be on the same rho grid.")


def check_equilibria(a: Equilibrium, b: Equilibrium):
    if a != b:
        raise ValueError("Equilibria must be the same.")


def calc_R_square_diff(
    rho_grid: xr.DataArray, time_grid: xr.DataArray, equilibrium: Equilibrium
):
    flux_coords = FluxSurfaceCoordinates(kind="poloidal")
    flux_coords.set_equilibrium(equilibrium)
    R_lfs, _ = flux_coords.convert_to_Rz(
        rho_grid,
        xr.zeros_like(rho_grid),
        time_grid,
    )
    R_hfs, _ = flux_coords.convert_to_Rz(
        rho_grid,
        np.pi * xr.ones_like(rho_grid),
        time_grid,
    )

    return R_hfs**2 - R_lfs**2


def asymmetry_parameter_from_modifier(
    asymmetry_modifier: xr.DataArray, equilibrium: Equilibrium
):
    R_square_diff = calc_R_square_diff(
        asymmetry_modifier.rho_poloidal,
        asymmetry_modifier.t,
        equilibrium,
    )

    return np.log(asymmetry_modifier) / R_square_diff


def asymmetry_parameter_from_lfs_hfs(
    lfs_values: xr.DataArray,
    hfs_values: xr.DataArray,
    equilibrium: Equilibrium,
):
    return asymmetry_parameter_from_modifier(
        asymmetry_modifier_from_lfs_hfs(lfs_values, hfs_values),
        equilibrium,
    )


def asymmetry_modifier_from_lfs_hfs(lfs_values: xr.DataArray, hfs_values: xr.DataArray):
    check_grids(hfs_values, lfs_values)
    return hfs_values / lfs_values


class AsymmetricQuantity:
    """
    Class to hold and manipulate asymmetric quantities, defined as:
    n_s(rho, R; t) = n_s(rho, R_0; t) exp{lambda_s(rho; t)(R(rho; t)^2 - R_0(rho; t)^2)}

    Parameters
    ----------
    lfs_values: xr.DataArray
        The (low field side) values of n_s(rho, R_0; t) on coordinates (rho_poloidal, t)
    asymmetry_parameter: xr.DataArray
        The values of lambda_s(rho; t) on coordinates (rho_poloidal, t)
    equilibrium: Equilibrium
        The plasma equilibrium

    """

    def __init__(
        self,
        lfs_values: xr.DataArray,
        asymmetry_parameter: xr.DataArray,
        equilibrium: Equilibrium,
    ):
        self.lfs_values = lfs_values
        self.asymmetry_parameter = asymmetry_parameter

        check_grids(lfs_values, asymmetry_parameter)

        self.equilibrium = equilibrium

    @classmethod
    def from_midplane(cls, rho_midplane_values: xr.DataArray, equilibrium: Equilibrium):
        lfs_values = rho_midplane_values.sel(rho_poloidal=slice(0, None))
        hfs_values = rho_midplane_values.sel(rho_poloidal=slice(None, 0))

        hfs_values = hfs_values.assign_coords(-hfs_values.rho_poloidal)
        check_grids(lfs_values, hfs_values)

        return cls.from_lfs_hfs(lfs_values, hfs_values, equilibrium)

    @classmethod
    def from_lfs_hfs(
        cls,
        lfs_values: xr.DataArray,
        hfs_values: xr.DataArray,
        equilibrium: Equilibrium,
    ):
        asymmetry_parameter = asymmetry_parameter_from_lfs_hfs(
            lfs_values, hfs_values, equilibrium
        )
        return cls(lfs_values, asymmetry_parameter, equilibrium)

    @classmethod
    def from_lfs_asymmetry_modifier(
        cls,
        lfs_values: xr.DataArray,
        asymmetry_modifier: xr.DataArray,
        equilibrium,
    ):
        asymmetry_parameter = asymmetry_parameter_from_modifier(
            asymmetry_modifier, equilibrium
        )
        return cls(lfs_values, asymmetry_parameter, equilibrium)

    @classmethod
    def from_R_z(cls, values_R_z: xr.DataArray):
        raise NotImplementedError

    @classmethod
    def from_rho_theta(cls, values_rho_theta: xr.DataArray, equilibrium: Equilibrium):
        lfs_values = values_rho_theta.sel(theta=0, method="nearest")
        hfs_values = values_rho_theta.sel(theta=np.pi, method="nearest")
        return cls.from_lfs_hfs(lfs_values, hfs_values, equilibrium)

    @classmethod
    def from_rho_R(cls, values_rho_R: xr.DataArray):
        raise NotImplementedError

    def to_R_z(self):
        raise NotImplementedError

    def to_rho_theta(self):
        raise NotImplementedError

    def to_rho_R(self):
        raise NotImplementedError

    def asymmetry_modifier(self, R_square_diff):
        return np.exp(self.asymmetry_parameter * R_square_diff)

    def R_square_diff(self):
        flux_coords = FluxSurfaceCoordinates(kind="poloidal")
        flux_coords.set_equilibrium(self.equilibrium)
        R_lfs, _ = flux_coords.convert_to_Rz(
            self.lfs_values.rho_poloidal,
            xr.zeros_like(self.lfs_values.rho_poloidal),
            self.lfs_values.t,
        )
        R_hfs, _ = flux_coords.convert_to_Rz(
            self.lfs_values.rho_poloidal,
            np.pi * xr.ones_like(self.lfs_values.rho_poloidal),
            self.lfs_values.t,
        )
        return R_hfs**2 - R_lfs**2

    # TODO: check values on same rho grids or interpolate
    def __eq__(self, other) -> bool:
        check_grids(self.lfs_values, other.lfs_values)
        check_equilibria(self.equilibrium, other.equilibrium)
        return (
            self.lfs_values == other.lfs_values
            and self.asymmetry_parameter == self.asymmetry_parameter
        )

    def __add__(self, other):
        check_grids(self.lfs_values, other.lfs_values)
        check_equilibria(self.equilibrium, other.equilibrium)

        lfs_values = self.lfs_values + other.lfs_values
        R_square_diff = self.R_square_diff()
        asymmetry_parameter = (
            np.log(
                (
                    self.lfs_values * np.exp(self.asymmetry_parameter * R_square_diff)
                    + other.lfs_values
                    * np.exp(other.asymmetry_parameter * R_square_diff)
                )
                / (self.lfs_values + other.lfs_values)
            )
            / R_square_diff
        )
        return self.__class__(lfs_values, asymmetry_parameter, self.equilibrium)
