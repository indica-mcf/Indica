import matplotlib.pylab as plt
import hda.physics as ph
from typing import Hashable
from typing import Tuple
from typing import Union
from copy import deepcopy

import xarray as xr

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline, interp1d
from xarray import DataArray

from indica.converters import CoordinateTransform
from indica.numpy_typing import ArrayLike, LabeledArray

from indica.utilities import broadcast_spline

SingleBoundaryType = Union[str, Tuple[int, ArrayLike]]
BoundaryType = Union[str, Tuple[SingleBoundaryType, SingleBoundaryType]]


class Spline:
    """Callable class wrapping a `:class:scipy.interpolate.CubicSpline`
    object so it will work with DataArrays.

    Parameters
    ----------
    values : DataArray
        The values to interpolate.
    dim : Hashable
        The axis along which to interpolate.
    bounds : BoundaryType
        The boundary condition to pass to `:class:scipy.interpolate.CubicSpline`.

    """

    def __init__(
        self,
        values: DataArray,
        bounds: BoundaryType = "clamped",
        dim: str = "rho_poloidal",
    ):
        """
        Spline interpolation object

        Parameters
        ----------
        values
            DataArray to be used as basis for interpolation
        bounds
            Boundary conditions
        dim
            Dimension on which
        """

        self.values = values.transpose()
        self.coord = values.coords[dim]
        self.dim = dim
        self.bounds = bounds
        self.spline_dims = tuple(d for d in values.dims if d != dim)
        self.spline_coords = {
            k: np.asarray(v) for k, v in values.coords.items() if k != self.dim
        }
        self.transpose_order = (self.dim,) + self.spline_dims

    def __call__(self, interp_coord: LabeledArray,) -> DataArray:
        """Get the spline values at the locations given by the coordinates.

        Parameters
        ----------
        interp_coord
            The values of the coordinate on which interpolation is to be performed

        """

        if not hasattr(self, "cubicspline"):
            self.prepare()

        if type(interp_coord) != DataArray:
            interp_coord = DataArray(interp_coord, coords=[(self.dim, interp_coord)])

        result = broadcast_spline(
            self.cubicspline, self.spline_dims, self.spline_coords, interp_coord
        ).transpose(*self.values.dims)

        return result

    def prepare(self):
        rho_quad = np.unique(np.sort(np.append(np.linspace(self.coord[1]/3, self.coord[-1]))))
        values_quad = self.values.interp(rho_poloidal=rho_quad, method="quadratic")
        rho_lin = np.unique(np.sort(np.append(rho_quad, [0, 1.05])))
        values = self.values.interp(rho_poloidal=rho_lin, method="linear")
        values.loc[dict(rho_poloidal=values.rho_poloidal.max())] = 0.0

        self.cubicspline = CubicSpline(
            values_quad.coords[self.dim],
            values_quad.transpose(*self.transpose_order),
            0,
            self.bounds,
            False,
        )

    def scale(self, const: DataArray, dim_lim=()):
        """
        Scale values starting self.values by a defined factor and prepare spline object

        Parameters
        ----------
        const
            Constant by which self.values are to be scaled
        dim_lim
            Limits (inclusive) in self.dim to which the scaling is to be applied


        """
        if len(dim_lim) == 0:
            dim_lim = (self.coord.min().values, self.coord.max().values)

        self.values = xr.where(
            (self.coord >= dim_lim[0]) * (self.coord <= dim_lim[1]),
            self.values * const,
            self.values,
        ).transpose(*self.values.dims)

        self.prepare()


class Plasma_profs:
    def __init__(self, time):
        """
        Make fake plasma profiles for electron density, electron and ion
        temperature, toroidal rotation

        Initialization already calculates default profiles and saves
        them as class attributes

        Results are given as xarray.DataArray

        Parameters
        ----------
        time
            Time grid
        rho
            Radial grid
        """

        rho = [0.0, 0.6, 0.85, 0.95, 1.0]
        nt = len(time)
        dims = ("rho_poloidal", "t")
        el_temp = DataArray(
            [[2.0e3, 1.2e3, 0.2e3, 0.06e3, 0.02e3]] * nt,
            coords=[("t", time), ("rho_poloidal", rho)],
        ).transpose(*dims)
        ion_temp = deepcopy(el_temp)
        el_dens = DataArray(
            [[5.0e19, 4.5e19, 4e19, 3.0e19, 1.0e19]] * nt,
            coords=[("t", time), ("rho_poloidal", rho)],
        ).transpose(*dims)
        v_tor = DataArray(
            [[200.0e3, 180e3, 90.0e3, 40e3, 5e3]] * nt,
            coords=[("t", time), ("rho_poloidal", rho)],
        ).transpose(*dims)

        self.el_temp = Spline(el_temp)
        self.ion_temp = Spline(ion_temp)
        self.el_dens = Spline(el_dens)
        self.v_tor = Spline(v_tor)

def coord_array(coord_vals, coord_name):
    return DataArray(coord_vals, coords=[(coord_name, coord_vals)])
