from copy import deepcopy
import os
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Union

import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.available_quantities import READER_QUANTITIES
from indica.utilities import build_dataarrays
from indica.utilities import CACHE_DIR
from indica.utilities import format_dataarray
from indica.utilities import get_element_info

DEFAULT_PATH = Path("")
DEFAULT_PULSE = 11890
ELEMENTS = ["D", "C", "Ar", "Ne"]


class SOLPSReader:
    """
    Class for reading SOLPS output
    TODO: Currently standalone, should be converted to a child of the datareader class
    """

    def __init__(
        self,
        path: Union[str, Path] = DEFAULT_PATH,
        pulse: int = DEFAULT_PULSE,
    ):
        self.available_quantities = READER_QUANTITIES["get_solps"]
        path = Path(path)
        if path == DEFAULT_PATH:
            self.path = Path.home() / CACHE_DIR / "solps" / f"{pulse}"
        else:
            self.path = path

    def _read_solps_txt(self) -> Dict[str, np.array]:
        """
        Read file and return dictionary with numpy arrays
        """

        R, z, database_results = {}, {}, {}
        files = os.listdir(self.path)
        for _file in files:
            file_type = _file.split(".")[0]

            filepath = self.path / _file
            with filepath.open("r") as f:
                tmp = f.readline()
                while "r array" not in tmp:
                    tmp = f.readline()
                _str = f.readline()
                _R = np.array(_str.strip().split()).astype(float)

                while "z array" not in tmp:
                    tmp = f.readline()
                _str = f.readline()
                _z = np.array(_str.strip().split()).astype(float)

                while "value array" not in tmp:
                    tmp = f.readline()
                _list = []
                for iz in range(len(_z)):
                    _str = f.readline()
                    _no_nans = _str.replace("NaN", "-1")
                    _list.append(_no_nans.strip().split())

                _array = np.array(_list).astype(float)
                _data = np.where(_array > 0, _array, np.nan)

                R[file_type] = _R
                z[file_type] = _z
                database_results[file_type] = _data

        for k in R.keys():
            assert np.all(R[k] == _R)
            assert np.all(z[k] == _z)

        database_results["R"] = _R
        database_results["z"] = _z
        self.database_results = database_results

        return database_results

    def get(
        self, file_type: str = "txt", verbose: bool = False, time: float = 0.05
    ) -> Dict[str, DataArray]:
        """
        Temporary get method, similar to indica/readers/datareader

        Reformats SOLPS output in standard Indica DataArray format
        """
        if file_type == "txt":
            reader_method: Callable = self._read_solps_txt
        else:
            raise ValueError(f"File type {file_type} not yet supported")

        database_results = reader_method()

        # Combine element data into 1 matrixof shape (ion_charge, R, z)
        t = np.array([time])
        nion: list = []
        fz: dict = {}
        elements, element_z, element_a, element_symbol = [], [], [], []
        for elem in ELEMENTS:
            if f"n{elem}0" in database_results.keys():
                _z, _a, _name, _symbol = get_element_info(elem)
                element_z.append(_z)
                element_a.append(_a)
                element_symbol.append(_symbol)

                if elem == "D":
                    element = "h"
                else:
                    element = _symbol.lower()

                elements.append(element)
                _fz = []
                database_results[f"n{elem}"] = np.full_like(
                    database_results[f"n{elem}0"], 0.0
                )
                ion_charge = np.arange(_z + 1)
                for q in ion_charge:
                    _tmp = database_results[f"n{elem}{q}"]
                    _fz.append(_tmp)
                    database_results[f"n{elem}"] += _tmp
                    database_results.pop(f"n{elem}{q}")
                nion.append(database_results[f"n{elem}"])

                fz_coords = {
                    "ion_charge": ion_charge,
                    "z": database_results["z"],
                    "R": database_results["R"],
                }
                fz[f"{element}"] = format_dataarray(
                    np.array(_fz) / database_results[f"n{elem}"],
                    "fractional_abundance",
                    fz_coords,
                    make_copy=True,
                ).expand_dims(dim={"t": t})
        database_results["nion"] = np.array(nion)
        database_results["element"] = elements
        database_results["atomic_weight"] = element_a
        database_results["atomic_number"] = element_z

        data = build_dataarrays(
            database_results,
            self.available_quantities,
            include_error=False,
            verbose=verbose,
        )
        for k in data.keys():
            data[k] = data[k].expand_dims(dim={"t": t})

        data["fz"]: dict = fz
        self.data = data

        return data

    def plot_solps_output(
        self, key: str = "te", element: str = "c", ion_charge: int = 0
    ):
        if hasattr(self, "data"):
            title = ""
            data = deepcopy(self.data[key])
            if key == "fz":
                data = data[element]
                title += f"{element} "
            else:
                if "element" in data.dims:
                    data = data.sel(element=element)
                    title += f"{element} "
            if "ion_charge" in data.dims:
                data = data.sel(ion_charge=ion_charge)
                title += f"{ion_charge}+"

            plt.figure()
            data.plot()
            Rlim = (data.R.min(), data.R.max())
            zlim = (data.z.min(), data.z.max())
            plt.vlines(Rlim[0], zlim[0], zlim[1], color="k")
            plt.vlines(Rlim[1], zlim[0], zlim[1], color="k")
            plt.hlines(zlim[0], Rlim[0], Rlim[1], color="k")
            plt.hlines(zlim[1], Rlim[0], Rlim[1], color="k")
            plt.axis("equal")
            plt.title(title)
