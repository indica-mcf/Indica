from copy import deepcopy

import numpy as np
from xarray import DataArray

from indica.readers import ST40Reader
import indica.writers.mds_tree_structures as trees


def read_hda_data(
    pulse: int,
    revision: int,
    uid: str = "",
    instrument: str = "HDA",
    tstart: float = 0.0,
    tend: float = 0.2,
    verbose=False,
):
    """
    Read HDA data from MDS+

    Parameters
    ----------

    Returns
    -------
        Dictionary of quantities contained in HDA MDS+ database

    """
    reader = ST40Reader(pulse, tstart, tend)
    nodes = trees.hda()

    _time, dims, units, path = reader._get_data(uid, instrument, ":TIME", revision)
    time = DataArray(
        _time, coords=[("t", _time)], attrs={"long_name": "Time", "units": units}
    )
    _rhop, _, _, _ = reader._get_data(
        uid, instrument, ".PROFILES.PSI_NORM:RHOP", revision
    )
    rhop = DataArray(
        _rhop, coords=[("rho_poloidal", _rhop)], attrs={"long_name": "Rhop"}
    )
    _rpos, dims, units, path = reader._get_data(
        uid, instrument, ".PROFILES.R_MIDPLANE:RPOS", revision
    )
    rpos = DataArray(
        _rpos, coords=[("R", _rpos)], attrs={"long_name": "R", "units": units}
    )
    data_dict = {}
    for sub_path, quantities in nodes.items():
        for node_name, node_info in quantities.items():
            long_name = node_info[1].split(",")[0]
            quantity = f"{sub_path}:{node_name}"
            data, dims, units, _path = reader._get_data(
                uid, instrument, quantity, revision
            )
            attrs = {"long_name": long_name, "units": units}
            if verbose:
                print(_path)

            if (
                node_name == "RHOP"
                or node_name == "XPSN"
                or node_name == "TIME"
                or np.array_equal(data, "FAILED")
            ):
                continue

            try:
                data_to_write = deepcopy(data)
                if np.size(data) == 1:
                    coords = None
                    continue
                elif sub_path == ".GLOBAL":
                    coords = [("t", time)]
                elif sub_path == ".PROFILES.PSI_NORM":
                    coords = [("t", time), ("rho_poloidal", rhop)]
                elif sub_path == ".PROFILES.R_MIDPLANE":
                    coords = [("t", time), ("R", rpos)]

                data_dict[quantity] = DataArray(
                    data_to_write,
                    coords=coords,
                    attrs=attrs,
                )
                # Coordinate attributes long_name and units not preserved...
                if coords is not None:
                    for i, dim in enumerate(data_dict[quantity].dims):
                        data_dict[quantity].coords[dim].attrs = coords[i][1].attrs
            except ValueError:
                print(f"Error reading == {_path}")

    return data_dict
