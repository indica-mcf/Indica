import numpy as np
import xarray as xr


class AsymmetricQuantity:
    """
    Class to hold and manipulate asymmetric quantities, defined as:
    n_s(rho, R; t) = n_s(rho, R_0; t) exp{lambda_s(rho; t)(R(rho; t)^2 - R_0(rho; t)^2)}

    Parameters
    ----------
    lfs_values: xr.DataArray
        The values of n_s(rho, R_0; t)
    asymmetry_parameter: xr.DataArray
        The values of lambda_s(rho; t)
    R_square_diff: xr.DataArray
        The values of R(rho; t)^2 - R_0(rho; t)^2

    """

    def __init__(
        self,
        lfs_values: xr.DataArray,
        asymmetry_parameter: xr.DataArray,
        R_square_diff: xr.DataArray,
    ):
        self.lfs_values = lfs_values
        self.asymmetry_parameter = asymmetry_parameter
        self.R_square_diff = R_square_diff

    @classmethod
    def from_midplane(
        cls, rho_midplane_values: xr.DataArray, R_square_diff: xr.DataArray
    ):
        lfs_values = rho_midplane_values.sel(rho_poloidal=slice(0, None))
        hfs_values = rho_midplane_values.sel(rho_poloidal=slice(None, 0))
        return cls.from_lfs_hfs(lfs_values, hfs_values, R_square_diff)

    @classmethod
    def from_lfs_hfs(
        cls,
        lfs_values: xr.DataArray,
        hfs_values: xr.DataArray,
        R_square_diff: xr.DataArray,
    ):
        asymmetry_parameter = cls.asymmetry_parameter_from_lfs_hfs(
            lfs_values, hfs_values, R_square_diff
        )
        return cls(lfs_values, asymmetry_parameter, R_square_diff)

    @classmethod
    def from_lfs_asymmetry_modifier(
        cls,
        lfs_values: xr.DataArray,
        asymmetry_modifier: xr.DataArray,
        R_square_diff: xr.DataArray,
    ):
        asymmetry_parameter = cls.asymmetry_parameter_from_modifier(
            asymmetry_modifier, R_square_diff
        )
        return cls(lfs_values, asymmetry_parameter, R_square_diff)

    @classmethod
    def from_R_z(cls, values_R_z: xr.DataArray):
        raise NotImplementedError

    @classmethod
    def from_rho_theta(
        cls, values_rho_theta: xr.DataArray, R_square_diff: xr.DataArray
    ):
        lfs_values = values_rho_theta.sel(theta=0, method="nearest")
        hfs_values = values_rho_theta.sel(theta=np.pi, method="nearest")
        return cls.from_lfs_hfs(lfs_values, hfs_values, R_square_diff)

    @classmethod
    def from_rho_R(cls, values_rho_R: xr.DataArray):
        raise NotImplementedError

    def to_R_z(self):
        raise NotImplementedError

    def to_rho_theta(self):
        raise NotImplementedError

    def to_rho_R(self):
        raise NotImplementedError

    def asymmetry_modifier(self):
        return np.exp(self.asymmetry_parameter * self.R_square_diff)

    @classmethod
    def asymmetry_parameter_from_modifier(
        cls, asymmetry_modifier: xr.DataArray, R_square_diff: xr.DataArray
    ):
        return np.log(asymmetry_modifier) / R_square_diff

    @classmethod
    def asymmetry_parameter_from_lfs_hfs(
        cls,
        lfs_values: xr.DataArray,
        hfs_values: xr.DataArray,
        R_square_diff: xr.DataArray,
    ):
        return cls.asymmetry_parameter_from_modifier(
            cls.asymmetry_modifier_from_lfs_hfs(lfs_values, hfs_values), R_square_diff
        )

    @staticmethod
    def asymmetry_modifier_from_lfs_hfs(
        lfs_values: xr.DataArray, hfs_values: xr.DataArray
    ):
        return hfs_values / lfs_values

    # TODO: check values on same rho grids or interpolate
    def __eq__(self, other) -> bool:
        return (
            self.lfs_values == other.lfs_values
            and self.asymmetry_parameter == self.asymmetry_parameter
        )

    def __add__(self, other):
        lfs_values = self.lfs_values + other.lfs_values
        asymmetry_parameter = (
            np.log(
                (
                    self.lfs_values
                    * np.exp(self.asymmetry_parameter * self.R_square_diff)
                    + other.lfs_values
                    * np.exp(other.asymmetry_parameter * other.R_square_diff)
                )
                / (self.lfs_values + other.lfs_values)
            )
            / self.R_square_diff
        )
        return self.__class__(lfs_values, asymmetry_parameter, self.R_square_diff)
