"""
Utility functions for running Stan Bayesian analysis script
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from indica.converters import FluxMajorRadCoordinates
from indica.converters import FluxSurfaceCoordinates
from indica.converters import ImpactParameterCoordinates
from indica.converters.time import bin_to_time_labels


@dataclass
class LOSData:
    """
    Class to hold line of sight data required in Stan models
    """

    N_los: int
    # indices 0-based here
    rho_lower_indices: xr.DataArray
    rho_interp_lower_frac: xr.DataArray
    R_square_diff: xr.DataArray
    los_values: xr.DataArray
    los_errors: xr.DataArray


def create_LOSData(
    diagnostic_data: xr.DataArray,
    los_coord_name: str,
    flux_coords: FluxSurfaceCoordinates,
    rho: xr.DataArray,
    t: xr.DataArray,
    N_los_points: int,
) -> LOSData:
    """
    Take diagnostic from reader and output all quantities required by Stan
    for these lines of sight
    """
    rho_maj_radius = FluxMajorRadCoordinates(flux_coords)

    # bin camera data, drop excluded channels
    binned_camera = bin_to_time_labels(
        t.data, diagnostic_data.dropna(dim=los_coord_name)
    )

    x2 = np.linspace(0, 1, N_los_points)
    camera = xr.Dataset(
        {"camera": binned_camera},
        {binned_camera.attrs["transform"].x2_name: x2},
        {"transform": binned_camera.attrs["transform"]},
    )

    rho_los_points, R_los_points = camera.indica.convert_coords(rho_maj_radius)

    # TODO: work out if this is the best/right way to deal with values outside plasma:
    rho_los_points = rho_los_points.clip(0, 1).fillna(1)

    R_los_points_lfs_midplane, _ = flux_coords.convert_to_Rz(
        rho_los_points, xr.zeros_like(rho_los_points), rho_los_points.t
    )

    R_square_diff = R_los_points**2 - R_los_points_lfs_midplane**2

    # TODO: check this
    # since we clipped before, some values of R_los_points_lfs_midplane
    # don't match R_los_points - need to clip again
    R_square_diff = R_square_diff.clip(max=0)

    # find indices for interpolation, subtract 1 to get lower indices
    rho_lower_indices = rho.searchsorted(rho_los_points) - 1
    # clip indices because we subtracted 1 to select lower
    # clip indices because there should be a value above them
    rho_lower_indices = rho_lower_indices.clip(min=0, max=len(rho) - 2)

    rho_lower_indices = xr.DataArray(
        data=rho_lower_indices, dims=rho_los_points.dims, coords=rho_los_points.coords
    )

    rho_dropped = rho.drop("rho_poloidal")
    lower = rho_dropped[rho_lower_indices]
    upper = rho_dropped[rho_lower_indices + 1]

    rho_interp_lower_frac = (upper - rho_los_points) / (upper - lower)

    # weights:
    ip_coords = ImpactParameterCoordinates(
        camera.attrs["transform"], flux_coords, times=t
    )
    impact_param, _ = camera.indica.convert_coords(ip_coords)
    weights = camera.camera * (0.02 + 0.18 * np.abs(impact_param))

    return LOSData(
        N_los=len(binned_camera.sxr_v_coords),
        # Stan is 1-based
        rho_lower_indices=rho_lower_indices,
        rho_interp_lower_frac=rho_interp_lower_frac,
        R_square_diff=R_square_diff,
        los_values=binned_camera,
        # los_errors=binned_camera.error.isel(t=t_index),
        los_errors=weights,
    )
