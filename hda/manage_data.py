import numpy as np
import xarray as xr
from xarray import DataArray
from hda.utils import print_like
from hda.models.plasma import Plasma
from indica.converters.time import bin_in_time_dt
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

    data_tmp = data[diagnostic][quantity]
    bckc_tmp = xr.full_like(data_tmp, np.nan)
    attrs = bckc_tmp.attrs
    if type(bckc_tmp) == DataArray:
        if "error" in attrs.keys():
            attrs["error"] = xr.full_like(attrs["error"], np.nan)
        if "partial_provenance" in attrs.keys():
            attrs.pop("partial_provenance")
            attrs.pop("provenance")
    bckc_tmp.attrs = attrs

    bckc[diagnostic][quantity] = bckc_tmp

    return bckc


def remap_diagnostic(diag_data, flux_transform, npts=100):
    """
    Calculate maping on equilibrium for speccified diagnostic

    Returns
    -------

    """
    new_attrs = {}
    trans = diag_data.attrs["transform"]
    x1 = diag_data.coords[trans.x1_name]
    x2_arr = np.linspace(0, 1, npts)
    x2 = DataArray(x2_arr, dims=trans.x2_name)
    dl = trans.distance(trans.x2_name, DataArray(0), x2[0:2], 0)[1]
    new_attrs["x2"] = x2
    new_attrs["dl"] = dl
    new_attrs["R"], new_attrs["z"] = trans.convert_to_Rz(x1, x2, 0)

    dt_equil = flux_transform.equilibrium.rho.t[1] - flux_transform.equilibrium.rho.t[0]
    dt_data = diag_data.t[1] - diag_data.t[0]
    if dt_data > dt_equil:
        t = diag_data.t
    else:
        t = None
    rho_equil, _ = flux_transform.convert_from_Rz(new_attrs["R"], new_attrs["z"], t=t)
    rho = rho_equil.interp(t=diag_data.t, method="linear")
    rho = xr.where(rho >= 0, rho, 0.0)
    rho.coords[trans.x2_name] = x2
    new_attrs["rho"] = rho

    return new_attrs

def build_data(plasma: Plasma, data, instrument=""):
    """
    Bin raw experimental data on the desired time axis, assign equilibrium to
    transform objects

    Parameters
    ----------
    plasma
        Plasma class
    data
        Experimental data dictionary whose elements are the data structures
        returned by Indica's abstractreader, its keys = instrument identifiers
    instrument
        Build data only for specified instrument

    Returns
    -------

    """
    print_like("Building data class")
    binned_data = {}

    for kinstr in data.keys():
        if (len(instrument) > 0) and (kinstr != instrument):
            continue
        instrument_data = {}

        if type(data[kinstr]) != dict:
            value = deepcopy(data[kinstr])
            if np.size(value) > 1:
                value = bin_in_time_dt(plasma.tstart, plasma.tend, plasma.dt, value)
            binned_data[kinstr] = value
            continue

        for kquant in data[kinstr].keys():
            value = data[kinstr][kquant]
            if "t" in value.coords:
                value = bin_in_time_dt(plasma.tstart, plasma.tend, plasma.dt, value)

            if "transform" in data[kinstr][kquant].attrs and hasattr(plasma, "equilibrium"):
                value.attrs["transform"] = data[kinstr][kquant].transform
                value.transform.set_equilibrium(plasma.equilibrium, force=True)
                if "LinesOfSightTransform" in str(value.attrs["transform"]):
                    geom_attrs = remap_diagnostic(value, plasma.flux_coords)
                    for kattrs in geom_attrs:
                        value.attrs[kattrs] = geom_attrs[kattrs]

            if "provenance" in data[kinstr][kquant].attrs:
                value.attrs["provenance"] = data[kinstr][kquant].provenance

            instrument_data[kquant] = value

        binned_data[kinstr] = instrument_data
        if kinstr == instrument:
            break

    return binned_data

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
