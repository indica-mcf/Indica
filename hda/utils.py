from xarray import DataArray
from copy import deepcopy
import xarray as xr

def calc_los_int(experimental_data, profile_to_integrate, t=None):
    dl = experimental_data.attrs["dl"]
    rho = experimental_data.attrs["rho"]
    transform = experimental_data.attrs["transform"]

    x2_name = transform.x2_name

    value = profile_to_integrate.interp(rho_poloidal=rho)
    if t is not None:
        value = value.sel(t=t)
        rho = rho.sel(t=t)
    value = xr.where(rho <= 1, value, 0, )
    los_integral = value.sum(x2_name) * dl

    return los_integral


def assign_datatype(data_array: DataArray, datatype: tuple, unit=""):
    data_array.name = f"{datatype[1]}_{datatype[0]}"
    data_array.attrs["datatype"] = datatype
    if len(unit) > 0:
        data_array.attrs["unit"] = unit


def assign_data(data: DataArray, datatype: tuple, unit="", make_copy=True):
    if make_copy:
        new_data = deepcopy(data)
    else:
        new_data = data

    new_data.name = f"{datatype[1]}_{datatype[0]}"
    new_data.attrs["datatype"] = datatype
    if len(unit) > 0:
        new_data.attrs["unit"] = unit

    return new_data


def print_like(string):
    print(f"\n {string}")
