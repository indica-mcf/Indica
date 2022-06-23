"""
Utility functions for analysis workflow
"""
from typing import List
from typing import Union

import numpy as np
from xarray import concat
from xarray import DataArray

from indica.operators import BolometryDerivation


def bolo_los(bolo_diag_array: DataArray) -> List[List[Union[List, str]]]:
    return [
        [
            np.array([bolo_diag_array.attrs["transform"].x_start.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].z_start.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].y_start.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].x_end.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].z_end.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].y_end.data[i].tolist()]),
            "bolo_kb5" + str(i.values),
        ]
        for i in bolo_diag_array.bolo_kb5v_coords
    ]


def bolo_impact_parameter(bolo_derivation: BolometryDerivation, trimmed: bool = False):
    attr = "LoS_coords_trimmed" if trimmed is True else "LoS_coords"
    if not hasattr(bolo_derivation, attr):
        raise UserWarning("Bolometry derivation must be run to calculate coords")
    rho_outp = []
    for coord in getattr(bolo_derivation, attr, []):
        los_coord_name = [val for val in coord["rho_poloidal"].dims if val != "t"][0]
        rho_min = coord["rho_poloidal"].min(los_coord_name)
        theta = coord["theta"].where(coord["rho_poloidal"] == rho_min, drop=True)
        rho_outp.append(rho_min.where(np.abs(theta[0]) < np.pi / 2, other=-rho_min))
    return concat(rho_outp, dim="channel", coords="minimal", compat="override")
