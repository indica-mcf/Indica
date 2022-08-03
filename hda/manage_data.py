import numpy as np
import xarray as xr
from xarray import DataArray
from hda.utils import print_like
from hda.models.plasma import Plasma
from indica.converters.time import bin_in_time_dt, get_tlabels_dt
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates
from indica.converters import lines_of_sight, line_of_sight
from copy import deepcopy


def initialize_bckc(diagnostic, quantity, data, bckc={}):
    """
    Initialise back-calculated data with all info as original data, apart
    from provenance and revision attributes

    Parameters
    ----------
    data
        DataArray of original data to be "cloned"

    Returns
    -------

    """
    if diagnostic not in bckc:
        bckc[diagnostic] = {}

    bckc_tmp = initialize_bckc_quantity(data[diagnostic], quantity)
    bckc[diagnostic][quantity] = bckc_tmp

    return bckc


def initialize_bckc_quantity(diagnostic_data: dict, quantity):
    data = diagnostic_data[quantity]
    bckc = xr.full_like(data, np.nan)
    attrs = bckc.attrs
    if type(bckc) == DataArray:
        if "error" in attrs.keys():
            attrs["error"] = xr.full_like(attrs["error"], np.nan)
        if "partial_provenance" in attrs.keys():
            attrs.pop("partial_provenance")
            attrs.pop("provenance")
    bckc.attrs = attrs

    return bckc


def bin_data_in_time(
    exp_data: dict,
    tstart: float,
    tend: float,
    dt: float,):
    """
    Bin raw experimental data on the desired time axis, assign equilibrium to
    transform objects

    Parameters
    ----------
    raw_data
        Experimental data as returned by Indica's abstractreader.py
    tstart
        Start of time range for which to get data.
    tend
        End of time range for which to get data.
    dt
        Time binning/interpolation

    Returns
    -------
    Data dictionary identical to input value but binned on new time axis
    """
    binned_data = {}
    for quant in exp_data.keys():
        data = deepcopy(exp_data[quant])
        attrs = data.attrs
        if "t" in data.coords:
            data = bin_in_time_dt(tstart, tend, dt, data)
        if "provenance" in attrs:
            data.attrs["provenance"] = attrs["provenance"]

        binned_data[quant] = data

    return binned_data


def map_on_equilibrium(
    diagnostic_data: dict,
    flux_transform: FluxSurfaceCoordinates,
):
    """
    Assign equilibrium and transform, map viewing LOS

    Parameters
    ----------
    diagnostic_data
        Experimental data of a specific instrument as returned by Indica's abstractreader.py
    flux_transform
        Indica's FluxSurfaceTransform object

    Returns
    -------
    Data dictionary identical to input with transforms set for remapping
    and remapping
    """

    data = diagnostic_data[list(diagnostic_data)[0]]
    if "transform" not in data.attrs:
        return data

    transform = data.attrs["transform"]
    if hasattr(flux_transform, "equilibrium"):
        transform.set_equilibrium(flux_transform.equilibrium, force=True)
        if "LinesOfSightTransform" in str(data.attrs["transform"]):
            transform.set_flux_transform(flux_transform)
            transform.remap_los(t=data.t)

    for quantity in diagnostic_data.keys():
        diagnostic_data[quantity].attrs["transform"] = transform

    return diagnostic_data


def apply_limits(
    data,
    diagnostic: str,
    quantity=None,
    val_lim=(np.nan, np.nan),
    err_lim=(np.nan, np.nan),
):
    """
    Set to Nan all data whose value or relative error aren't within specified limits
    """

    if quantity is None:
        quantity = list(data[diagnostic])
    else:
        quantity = list(quantity)

    for q in quantity:
        error = None
        value = data[diagnostic][q]
        if "error" in value.attrs.keys():
            error = data[diagnostic][q].attrs["error"]

        if np.isfinite(val_lim[0]):
            print(val_lim[0])
            value = xr.where(value >= val_lim[0], value, np.nan)
        if np.isfinite(val_lim[1]):
            print(val_lim[1])
            value = xr.where(value <= val_lim[1], value, np.nan)

        if error is not None:
            if np.isfinite(err_lim[0]):
                print(err_lim[0])
                value = xr.where((error / value) >= err_lim[0], value, np.nan)
            if np.isfinite(err_lim[1]):
                print(err_lim[1])
                value = xr.where((error / value) <= err_lim[1], value, np.nan)

        data[diagnostic][q].values = value.values

    return data
