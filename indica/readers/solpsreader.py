from pathlib import Path
from typing import Callable
from typing import Union

import matplotlib.pylab as plt
import numpy as np

from indica.utilities import build_dataarrays
from indica.utilities import CACHE_DIR

DEFAULT_PATH = Path("")
DEFAULT_FILENAME = "te_400_400.txt"


class SOLPSReader:
    """
    Class for reading SOLPS output

    TODO: Currently standalone, should be converted to a child of the datareader class
    """

    def __init__(
        self,
        path: Union[str, Path] = DEFAULT_PATH,
        filename: str = DEFAULT_FILENAME,
        datatype: str = "electron_temperature",
    ):
        path = Path(path)
        if path == DEFAULT_PATH:
            self.namespace = "openadas"
            self.path = Path.home() / CACHE_DIR / "solps"
        else:
            self.namespace = "localadas"
            self.path = path
        self.filename = filename
        self.datatype = datatype

    def _read_solps_txt(self):
        filepath = self.path / self.filename

        with filepath.open("r") as f:
            tmp = f.readline()
            while "r array" not in tmp:
                tmp = f.readline()
            _str = f.readline()
            R = np.array(_str.strip().split()).astype(float)

            while "z array" not in tmp:
                tmp = f.readline()
            _str = f.readline()
            z = np.array(_str.strip().split()).astype(float)

            while "value array" not in tmp:
                tmp = f.readline()
            _list = []
            for iz in range(len(z)):
                _str = f.readline()
                _no_nans = _str.replace("NaN", "-1")
                _list.append(_no_nans.strip().split())

            _array = np.array(_list).astype(float)
            values = np.where(_array > 0, _array, np.nan)

            database_results = {"R": R, "z": z, self.datatype: values}
            self.database_results = database_results
            return database_results

    def get(self, file_type: str = "txt", verbose: bool = False, plot: bool = False):
        if file_type == "txt":
            reader_method: Callable = self._read_solps_txt
            available_quantities = {
                "R": ("R", []),
                "z": ("z", []),
                self.datatype: (self.datatype, ["z", "R"]),
            }
        else:
            raise ValueError(f"File type {file_type} not yet supported")

        database_results = reader_method()

        data = build_dataarrays(
            database_results,
            available_quantities,
            include_error=False,
            verbose=verbose,
        )
        self.data = data

        if plot:
            self.plot_solps_output()

        return data

    def plot_solps_output(self):
        if hasattr(self, "data"):
            plt.figure()
            self.data[self.datatype].plot()
            plt.vlines(
                self.data["R"].min(),
                self.data["z"].min(),
                self.data["z"].max(),
                color="k",
            )
            plt.vlines(
                self.data["R"].max(),
                self.data["z"].min(),
                self.data["z"].max(),
                color="k",
            )
            plt.hlines(
                self.data["z"].min(),
                self.data["R"].min(),
                self.data["R"].max(),
                color="k",
            )
            plt.hlines(
                self.data["z"].max(),
                self.data["R"].min(),
                self.data["R"].max(),
                color="k",
            )
            plt.axis("equal")
