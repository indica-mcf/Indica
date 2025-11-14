from copy import deepcopy
from typing import Callable
from typing import Dict

import numpy as np
import xarray as xr
from xarray import DataArray

from indica.configs.readers.st40readerprocessorconf import ST40ReaderProcessorConf
from indica.converters.time import convert_in_time_dt
from indica.converters.time import get_tlabels_dt


class ReaderProcessor:
    """
    Takes raw data from a datareader and applies filtering and binning
    """

    def __init__(
        self,
        conf=ST40ReaderProcessorConf(),
    ):

        self.conf = conf
        self.raw_data: dict = None
        self.processed_data: dict = {}

    def reset_data(self):
        self.processed_data = {}

    def __call__(
        self,
        raw_data: dict,
        tstart: float = None,
        tend: float = None,
        dt: float = None,
        verbose: bool = False,
    ):

        self.reset_data()
        self.raw_data = raw_data

        self.t = get_tlabels_dt(tstart, tend, dt)

        self.processed_data = apply_filter(
            raw_data,
            filters=self.conf.filter_values,
            filter_func=value_condition,
            filter_func_name="value",
            verbose=verbose,
        )

        self.processed_data = apply_filter(
            self.processed_data,
            filters=self.conf.filter_coordinates,
            filter_func=coordinate_condition,
            filter_func_name="co-ordinate",
            verbose=verbose,
        )

        self.processed_data = bin_data_in_time(
            self.processed_data, tstart=tstart, tend=tend, dt=dt
        )
        return self.processed_data


def bin_data_in_time(
    raw_data: Dict[str, Dict[str, xr.DataArray]],
    tstart: float = 0.02,
    tend: float = 0.1,
    dt: float = 0.01,
    debug=False,
):
    binned_data = {}
    for instr in raw_data.keys():
        if debug:
            print(f"instr: {instr}")
        binned_quantities = {}
        for quant in raw_data[instr].keys():
            if debug:
                print(f"quant: {quant}")
            data_quant = deepcopy(raw_data[instr][quant])

            if "t" in data_quant.coords:
                data_quant = convert_in_time_dt(tstart, tend, dt, data_quant)
                # Using groupedby_bins always removes error from coords so adding it
                if "error" in raw_data[instr][quant].coords:
                    error = convert_in_time_dt(
                        tstart, tend, dt, raw_data[instr][quant].error
                    )
                    data_quant = data_quant.assign_coords(
                        error=(raw_data[instr][quant].dims, error.data)
                    )
                binned_quantities[quant] = data_quant
        binned_data[instr] = binned_quantities
    return binned_data


def apply_filter(
    data: Dict[str, Dict[str, xr.DataArray]],
    filters: Dict[str, Dict[str, tuple]],
    filter_func: Callable,
    filter_func_name="value",
    verbose=False,
):

    filtered_data = {}
    for instrument, quantities in data.items():
        if instrument not in filters.keys():
            if verbose:
                print(f"missing {filter_func_name} filter for {instrument}")
            filtered_data[instrument] = deepcopy(data[instrument])
            continue

        filtered_data[instrument] = {}
        for quantity_name, quantity in quantities.items():
            if quantity_name not in filters[instrument]:
                filtered_data[instrument][quantity_name] = deepcopy(
                    data[instrument][quantity_name]
                )
                continue

            filter_info = filters[instrument][quantity_name]
            filtered_data[instrument][quantity_name] = filter_func(
                quantity, filter_info
            )
    return filtered_data


def value_condition(data: DataArray, limits: tuple):
    condition = (data >= limits[0]) * (data < limits[1])
    filtered_data = xr.where(condition, data, np.nan)
    filtered_data.attrs = data.attrs
    return filtered_data


def coordinate_condition(data: DataArray, coord_info: tuple):
    coord_name: str = coord_info[0]
    coord_slice: tuple = coord_info[1]
    condition = (data.coords[coord_name] >= coord_slice[0]) * (
        data.coords[coord_name] < coord_slice[1]
    )
    filtered_data = data.where(condition, np.nan)
    filtered_data.attrs = data.attrs
    return filtered_data



#Not useable yet!
def transp_transform(psin, rhotor, profiles):



        #Required coordinate transform
        #Note that we still need to get all the inputs! This is just the transform

        from scipy.interpolate import interp1d
        #rhopol from PSIN. Can we assume we always have this? Or do I need to use FTOR or something else to get this?
        rhopol = np.sqrt(psin)

        #Order the grid so monotonic
        order = np.argsort(rhotor)
        rt = rhotor[order]
        rp = rhopol[order]

        #Strictly monotonic
        mask = np.concatenate(([True], np.diff(rt) > 1e-10))
        rt = rt[mask]
        rp = rp[mask]


        #Instad: use this 2d, and then just duplicate timepoints and mapping. D


        #Interpolators both ways
        f_rhotor_to_rhopol = interp1d(rt, rp, kind="linear", bounds_error=False, fill_value=np.nan)
        g_rhopol_to_rhotor = interp1d(rp, rt, kind="linear", bounds_error=False, fill_value=np.nan)

        #A target grid (uniform in rhopol). 100 arbitrary for now
        rhopol_target = np.linspace(0.0, 1.0, 100)

        #Map each target rhopol back to rhotor for evaluation of TRANSP profiles:
        #For each of these desired ρ_pol surfaces, what ρ_tor value corresponds to the same magnetic surface?
        rhotor_for_target = g_rhopol_to_rhotor(rhopol_target)

        #Then for every profile we want to convert:
        def convert_profile_to_rhopol(rhotor_grid, X_rhotor, rhotor_for_target, rhopol_target):
            # Interpolate X from its native rhotor grid to rhotor_for_target
            Xi = np.interp(
                rhotor_for_target,
                rhotor_grid,
                X_rhotor,
                left=np.nan, right=np.nan  
            )
            return rhopol_target, Xi


        converted = {}
        for name, arr in profiles.items():
            rp_tgt, X_on_rhopol = convert_profile_to_rhopol(rt, arr[order][mask], rhotor_for_target, rhopol_target)
            converted[name] = X_on_rhopol


