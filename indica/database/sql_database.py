# TODO: for the time being it's a mix between st40_database & indica
#       st40_database - used for static
#       indica - for everything else
from copy import deepcopy
from time import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

from indica.configs.readers import ST40Conf
from indica.readers import ReaderProcessor
from indica.readers import ST40Reader
import mysql.connector
import numpy as np
from st40_database import GetCodeRuns
from st40_phys_trends.settings import filter_conf
from st40_phys_trends.settings import GLOBAL_ALL
from st40_phys_trends.settings import GLOBAL_SYSTEM_SPECIFIC
from st40_phys_trends.settings import READ_PARAMS
from st40_phys_trends.settings import sql_credentials
from st40_phys_trends.settings import STATIC_ALL
from st40_phys_trends.settings import STATIC_SYSTEM_SPECIFIC
from st40_phys_trends.settings import TREE_NAMES
import xarray as xr
from xarray import DataArray


CONF = ST40Conf()
INSTRUMENT_METHODS = CONF.INSTRUMENT_METHODS


class SQLDatabase:
    """
    Instantiate class for 1 pulse & 1 code whose RUNs will be written to SQL
    """

    def __init__(
        self,
        pulse: int,
        tree_name: str,
        database_name: str = "st40_phys_trends_test",
        apply_data_q: bool = False,
        raise_exceptions: bool = False,
        debug: bool = False,
        write: bool = False,
    ):
        """
        pulse - pulse number
        tree - MDS+ tree to scrape and write to SQL
        database_name - SQL database name to write to
        apply_data_q - set to NaN data if data quality nodes < 1
        debug - print steps to terminal
        """
        if database_name != "st40_phys_trends_test":
            raise Exception("Don't write to official Trends *yet*!!!!!")

        self.pulse = pulse
        self.tree = tree_name.lower()
        self.database_name = database_name
        self.apply_data_q = apply_data_q
        self.raise_exceptions = raise_exceptions
        self.debug = debug
        self.write = write

        self.sql_credentials = sql_credentials[database_name]
        self.tlim = READ_PARAMS["tlim"]
        self.dt = READ_PARAMS["dt"]
        self.overlap = READ_PARAMS["overlap"]
        self._debug_log = ""
        self.t0_runtime = time()

        self.tree_type = INSTRUMENT_METHODS[self.tree]

        # Find out what to write to SQL
        self.static_to_write = STATIC_ALL
        if self.tree_type in STATIC_SYSTEM_SPECIFIC:
            self.static_to_write += STATIC_SYSTEM_SPECIFIC[self.tree_type]

        self.global_to_write = GLOBAL_ALL
        if self.tree_type in GLOBAL_SYSTEM_SPECIFIC:
            self.global_to_write += GLOBAL_SYSTEM_SPECIFIC[self.tree_type]

        # Instantiate configuration for data processing
        self.processing_conf = filter_conf()
        self.reader_processor = ReaderProcessor(conf=self.processing_conf)
        self.t_sql = self.reader_processor.get_tlabels_dt(self.tlim[0], self.tlim[1], self.dt)

    def instantiate_data_reader(self):
        try:
            self.st40 = ST40Reader(self.pulse, self.tlim[0] - self.dt * 2, self.tlim[1] + self.dt * 2, tree=self.tree)
        except Exception as e:
            self.debug_log(f"Error instantiating reader for {self.tree} - {e}")

    def get_runs_list(self):
        self.runs_list = []
        try:
            runs_list = GetCodeRuns(self.pulse, code_names=[self.tree]).get_code_runs()
            self.runs_list = [run.split("#")[-1].strip() for run in runs_list]
        except Exception as e:
            self.debug_log(f"Error reading runs list for {self.tree} - {e}")

    def get_best_run(self):
        # Works for both BEST and POST_BEST
        # Only 1 BEST is allowed --> len(_best_run_name) == 1
        self.best_run: str = ""
        best_run_name = [run for run in self.runs_list if "BEST" in run]
        if len(best_run_name) == 1 and "BEST" in best_run_name[0].upper():
            # There's no need to read BEST since you already know what it's pointing to
            self.runs_list.remove(best_run_name[0])

            # Continue even if data is unavailable
            try:
                _best_run, _ = self.st40.reader_utils.get_signal("", self.tree, ".BEST_RUN", best_run_name[0])
                self.best_run = str(_best_run)
                self.debug_log(f"OK reading BEST run for {self.tree}")
            except Exception as e:
                self.debug_log(f"Error reading BEST run for {self.tree} - {e}")
        else:
            self.debug_log(f"No or duplicate BEST run(s) for {self.tree}")

    def get_static_data(self):
        """Read static data
        TODO: add to Indica reading of static information so this function becomes redundant
        ...this should be a copy-paste of get_global_data()...
        """
        self.static_data: dict = {}
        for run in self.runs_list:
            data = {}
            for quantity in self.static_to_write:
                _key = quantity.split(".")[-1].split(":")[-1]

                # Continue even if data is unavailable
                data[_key] = ""
                try:
                    # TODO: currently implemented only for string static data
                    _data, _ = self.st40.reader_utils.get_signal("", self.tree, f".{quantity}", run)
                    _data = str(_data)
                    if strip_mds_path(quantity.lower()) == "datetime" and len(_data) > 0:
                        _data = _data[:-1]  # remove 'Z' at end of string
                    data[_key] = _data.strip()
                except Exception as e:
                    self.debug_log(f"Error reading static {quantity} for {self.tree}#{run} - {e}")

            self.static_data[run] = deepcopy(data)

    def get_raw_global(self):
        self.global_data_raw = {}

        # See if tree is available in reader methods
        if len(self.global_to_write) == 0:
            raise Exception(f"Error - no global quantities for {self.tree}")

        for run in self.runs_list:
            _data, data = {}, {}

            # Continue even if data is unavailable
            try:
                _data = self.st40.get(uid="", instrument=self.tree, revision=run)
                self.debug_log(f"OK reading {self.tree} global for {run}")
            except Exception as e:
                self.debug_log(f"Error reading global for {run} - {e}")
                continue

            # Retain only the data specified in the configuration files
            for quantity in self.global_to_write:
                if quantity in _data:
                    data[quantity] = _data[quantity]

            self.global_data_raw[run] = deepcopy(data)

    def process_global(self):
        self.global_data = {}
        for run in self.global_data_raw.keys():
            processed = self.reader_processor(
                {self.tree: self.global_data_raw[run]},
                tstart=self.tlim[0],
                tend=self.tlim[1],
                dt=self.dt,
                overlap=self.overlap,
                check_bounds=False,
            )[self.tree]

            data: Dict[DataArray] = {}
            for quantity in processed.keys():
                _data = processed[quantity]
                dims = _data.dims

                # Differentiate in time
                _data = _data.assign_coords(d_dt=(dims, _data.differentiate("t").data))

                # Set to NaN data if bad data quality
                mask = xr.full_like(_data, 1)
                if "data_q" in _data and "t" in dims and self.apply_data_q:
                    mask = xr.where(_data["data_q"] > 0.9, 1, np.nan)

                # Set masked data
                data[quantity] = _data * mask
                for coord in ["d_dt", "error", "stdev"]:
                    data[quantity] = data[quantity].assign_coords({coord: (dims, (_data.coords[coord] * mask).data)})

                self.global_data[run] = deepcopy(data)

    def write_to_sql_best_linkage(self):
        table_name = "best_linkage"

        data_list = [
            {"SQL_key": "pulseNo", "data": str(self.pulse)},
            {"SQL_key": f"{self.tree_type}_{self.tree}_best", "data": self.best_run},
        ]
        self.write_to_mysql(data_list, table_name)
        return data_list, table_name

    def write_to_sql_static(self):
        table_name = f"{self.tree_type}_static"

        data_dict = {}
        for run, _data in self.static_data.items():
            # Each entry must have info on pulse, tree and run
            data_list = [
                {"SQL_key": "pulseNo", "data": self.pulse},
                {"SQL_key": "tree_name", "data": self.tree},
                {"SQL_key": "run_name", "data": run},
            ]
            for _quantity, data in _data.items():
                quantity = _quantity.lower()
                data_list.append({"SQL_key": quantity, "data": data})
            data_dict[run] = data_list
            self.write_to_mysql(data_list, table_name)
        return data_dict, table_name

    def write_to_sql_global(self):
        table_name = f"{self.tree_type}_global"

        data_dict = {}
        for t in self.t_sql:
            data_dict[t] = {}
            for run, _data in self.global_data.items():
                data_dict[t][run] = {}
                data_list = [
                    {"SQL_key": "pulseNo", "data": self.pulse},
                    {"SQL_key": "tree_name", "data": self.tree},
                    {"SQL_key": "run_name", "data": run},
                    {"SQL_key": "time", "data": f"{t:.3f}"},
                ]  # type(time) = string to avoid floating point issues
                for _quantity, data in _data.items():
                    quantity = _quantity.lower()
                    # Each entry must have info on pulse, tree and run
                    data_list.append({"SQL_key": quantity, "data": float(data.sel(t=t).data)})
                    data_list.append({"SQL_key": f"{quantity}_error", "data": float(data.error.sel(t=t).data)})
                    data_list.append({"SQL_key": f"{quantity}_stdev", "data": float(data.stdev.sel(t=t).data)})
                    data_list.append({"SQL_key": f"d_{quantity}_dt", "data": float(data.d_dt.sel(t=t).data)})
                data_dict[t][run] = data_list
                self.write_to_mysql(data_list, table_name)
        return data_dict, table_name

    # def write_sql_query_global()

    def write_to_mysql(self, data_list: List[dict], table_name: str):
        if not self.write:
            return

        if "test" not in self.database_name:
            raise (Exception("Allowed to write only to test database for the time being!"))

        # Connect to MySQL
        mydb = mysql.connector.connect(**self.sql_credentials)

        comma_new_line = ",\n            "
        sql_query = f"""
            INSERT INTO {table_name} ({', '.join(data['SQL_key'] for data in data_list)})
            VALUES ({", ".join('%s' for _ in data_list)})
            ON DUPLICATE KEY UPDATE
                {comma_new_line.join(data['SQL_key'] + ' = VALUES(' + data['SQL_key'] + ')' for data in data_list)}
        """

        # The data needs to be a tuple of strings, with None if no data, e.g.
        # values = ("abc", "-2", None, None)
        values = tuple(f"{data['data']}" if data["data"] is not None else None for data in data_list)

        mycursor = mydb.cursor()
        mycursor.execute(sql_query, values)
        mydb.commit()
        mydb.close()

    def get_elapsed_time(self):
        return f"{time()-self.t0_runtime:.3f}"

    def debug_log(self, message: str):
        debug_message = f"{self.get_elapsed_time()} {message}"
        self._debug_log += f"\n {debug_message}"
        if self.debug:
            print(debug_message)

    def run_with_log(
        self,
        operator: Callable[..., Any],
        message: str | None,
        args: dict[Any, Any] | None = None,
    ) -> None:
        msg = ""
        if message is not None:
            msg += f"{message}"

        try:
            if args is None:
                ret_data = operator()
            else:
                ret_data = operator(**args)

            self.debug_log(msg)
            return ret_data
        except Exception as e:
            msg += f" - {e}"

            self.debug_log(msg)

            if self.debug:
                print(msg)

            if self.raise_exceptions:
                raise e(msg)

    def __call__(self):
        # Instantiate reader
        self.get_runs_list()
        self.instantiate_data_reader()

        # Get best run and write it to SQL
        self.get_best_run()
        self.write_to_sql_best_linkage()

        # Get static data and write it to SQL
        self.get_static_data()
        try:
            self.write_to_sql_static()
        except Exception as e:
            self.debug_log(f"Error writing static to SQL - {e}")

        # Get raw data, process it, and write it to SQL
        self.get_raw_global()
        self.process_global()
        try:
            self.write_to_sql_global()
        except Exception as e:
            self.debug_log(f"Error writing static to SQL - {e}")

        # Close MDS+ connection
        self.st40.close()

def scrape(
    pulse_numbers: List[int],
    tree_names: list = [],
    database_name: str = "st40_phys_trends_test",
    apply_data_q: bool = False,
    debug: bool = False,
    write_to_sql: bool = False,
):

    if len(tree_names) == 0:
        tree_names = TREE_NAMES

    for pulse in pulse_numbers:
        for tree in tree_names:
            controller = ST40PhysTrends(
                pulse, 
                tree, 
                database_name, 
                apply_data_q=apply_data_q, 
                debug=debug, 
                write=write_to_sql
            )
            controller()


def print_skip(tree_name: str, run: str):
    print(f"Error reading {tree_name} {run} - skipping")


def strip_mds_path(mds_path: str):
    return mds_path.split(".")[-1].split(":")[-1]
