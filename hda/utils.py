from xarray import DataArray
from copy import deepcopy

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
