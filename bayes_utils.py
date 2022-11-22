"""
Utility functions for running Stan Bayesian analysis script
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from enum import Enum
from enum import unique
from typing import Dict
from typing import List

import numpy as np
import xarray as xr

from indica.converters import FluxMajorRadCoordinates
from indica.converters import FluxSurfaceCoordinates
from indica.converters import ImpactParameterCoordinates
from indica.converters.time import bin_to_time_labels
from indica.operators import SplineFit
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader


@unique
class LOSType(Enum):
    BOLO = (auto(),)
    SXR = (auto(),)


def fit_electron_profiles(
    hrts_diagnostic: Dict[str, xr.DataArray],
    rho: xr.DataArray,
    t: xr.DataArray,
):
    """
    Fit the ne and te profiles to a given rho and t array
    """
    # set datatype as expected by spline fitter
    rho.attrs["datatype"] = ("norm_flux_pol", "plasma")

    knots_te = [0.0, 0.3, 0.6, 0.85, 0.9, 0.98, 1.0, 1.05]
    fitter_te = SplineFit(
        lower_bound=0.0,
        upper_bound=hrts_diagnostic["te"].max() * 1.1,
        knots=knots_te,
    )
    results_te = fitter_te(rho, t, hrts_diagnostic["te"])
    te = results_te[0]

    temp_ne = deepcopy(hrts_diagnostic["ne"])
    temp_ne.attrs["datatype"] = deepcopy(
        hrts_diagnostic["te"].attrs["datatype"]
    )  # TEMP for SplineFit checks
    knots_ne = [0.0, 0.3, 0.6, 0.85, 0.95, 0.98, 1.0, 1.05]
    fitter_ne = SplineFit(
        lower_bound=0.0, upper_bound=temp_ne.max() * 1.1, knots=knots_ne
    )
    results_ne = fitter_ne(rho, t, temp_ne)
    ne = results_ne[0]

    return ne, te


def get_power_loss(
    elements: List[str],
    flux_surface: FluxSurfaceCoordinates,
    ne: xr.DataArray,
    te: xr.DataArray,
    t: xr.DataArray,
    los_type: LOSType,
) -> xr.DataArray:
    """
    Calculate the power loss values at the points sampled for LOS calculations
    """
    # TODO: put on correct grid, not rho-theta or whatever
    # TODO: consider doing power loss calculations on rho grid then interpolating
    #       onto correct time, los_coord, los_point grid

    # deuterium and trititum are hydrogen
    elements = [
        "h" if element == "d" or element == "t" else element for element in elements
    ]

    adas = ADASReader()

    SCD = {
        element: adas.get_adf11("scd", element, year)
        for element, year in zip(elements, ["89"] * len(elements))
    }
    ACD = {
        element: adas.get_adf11("acd", element, year)
        for element, year in zip(elements, ["89"] * len(elements))
    }
    FA = {
        element: FractionalAbundance(SCD=SCD.get(element), ACD=ACD.get(element))
        for element in elements
    }

    if los_type == LOSType.SXR:
        # Read in SXR data filtered for SXR camera window
        sxr_adas = ADASReader("/home/elitherl/Analysis/SXR/indica/sxr_filtered_adf11/")

        PLT = {
            element: sxr_adas.get_adf11("pls", element, year)
            for element, year in zip(elements, ["5"] * len(elements))
        }
        PRB = {
            element: sxr_adas.get_adf11("prs", element, year)
            for element, year in zip(elements, ["5"] * len(elements))
        }
        PL = {
            element: PowerLoss(PLT=PLT.get(element), PRB=PRB.get(element))
            for element in elements
        }
    elif los_type == LOSType.BOLO:
        PLT = {
            element: adas.get_adf11("plt", element, year)
            for element, year in zip(elements, ["89"] * len(elements))
        }
        PRB = {
            element: adas.get_adf11("prb", element, year)
            for element, year in zip(elements, ["89"] * len(elements))
        }
        PL = {
            element: PowerLoss(PLT=PLT.get(element), PRB=PRB.get(element))
            for element in elements
        }
    else:
        raise ValueError("los_type unrecognised.")

    # %% Calculating power loss

    fzt = {
        elem: xr.concat(
            [
                xr.concat(
                    [
                        FA[elem](
                            Ne=ne.interp(t=time).interp(sxr_v_coords=los_coord),
                            Te=te.interp(t=time).interp(sxr_v_coords=los_coord),
                            tau=time,
                        ).expand_dims("sxr_v_coords", -1)
                        for los_coord in ne.sxr_v_coords
                    ],
                    dim="sxr_v_coords",
                )
                .assign_coords({"sxr_v_coords": ne.sxr_v_coords})
                .expand_dims("t", -1)
                for time in t.values
            ],
            dim="t",
        )
        .assign_coords({"t": t.values})
        .assign_attrs(transform=flux_surface)
        for elem in elements
    }

    power_loss = {
        elem: xr.concat(
            [
                xr.concat(
                    [
                        PL[elem](
                            Ne=ne.interp(t=time).sel(sxr_v_coords=los_coord),
                            Te=te.interp(t=time).sel(sxr_v_coords=los_coord),
                            F_z_t=fzt[elem].sel(t=time).sel(sxr_v_coords=los_coord),
                        ).expand_dims("sxr_v_coords", -1)
                        for los_coord in ne.sxr_v_coords
                    ],
                    dim="sxr_v_coords",
                )
                .assign_coords({"sxr_v_coords": ne.sxr_v_coords})
                .expand_dims("t", -1)
                for time in t.values
            ],
            dim="t",
        )
        .assign_coords({"t": t.values})
        .assign_attrs(transform=flux_surface)
        for elem in elements
    }

    power_loss = xr.concat(
        [val.sum("ion_charges") for key, val in power_loss.items()],
        dim="element",
    ).assign_coords({"element": [key for key in power_loss.keys()]})

    return power_loss


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
    # values to premultiply each los point by e.g. for SXR ne*cooling_function
    # dimensions [N_elements, N_los, N_los_points]
    premult_values: xr.DataArray
    los_values: xr.DataArray
    los_errors: xr.DataArray


def create_LOSData(
    los_diagnostic: xr.DataArray,
    los_coord_name: str,
    hrts_diagnostic: Dict[str, xr.DataArray],
    flux_coords: FluxSurfaceCoordinates,
    rho: xr.DataArray,
    t: xr.DataArray,
    N_los_points: int,
    elements: List[str],
    los_type: LOSType,
) -> LOSData:
    """
    Take diagnostic from reader and output all quantities required by Stan
    for these lines of sight
    """
    rho_maj_radius = FluxMajorRadCoordinates(flux_coords)

    # bin camera data, drop excluded channels
    binned_camera = bin_to_time_labels(
        t.data, los_diagnostic.dropna(dim=los_coord_name)
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

    ne, te = fit_electron_profiles(hrts_diagnostic, rho_los_points, t)

    power_loss = get_power_loss(
        elements,
        flux_coords,
        ne,
        te,
        t,
        los_type,
    )

    premult_values = ne * power_loss

    return LOSData(
        N_los=len(binned_camera.sxr_v_coords),
        # Stan is 1-based
        rho_lower_indices=rho_lower_indices,
        rho_interp_lower_frac=rho_interp_lower_frac,
        R_square_diff=R_square_diff,
        premult_values=premult_values,
        los_values=binned_camera,
        # los_errors=binned_camera.error.isel(t=t_index),
        los_errors=weights,
    )
