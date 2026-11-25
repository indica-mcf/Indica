"""Core analysis workflow for reading pulse data and writing trend results to SQL."""

# TODO: for the time being it's a mix between st40_database & indica
#       st40_database - used for static
#       indica - for everything else
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from datetime import UTC
from time import time
from typing import Any

import mysql.connector
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.configs.readers import ST40Conf
from indica.database.settings import filter_conf
from indica.database.settings import GLOBAL_ALL
from indica.database.settings import GLOBAL_SYSTEM_SPECIFIC
from indica.database.settings import READ_PARAMS
from indica.database.settings import sql_credentials
from indica.database.settings import STATIC_ALL
from indica.database.settings import STATIC_SYSTEM_SPECIFIC
from indica.database.settings import TREE_NAMES
from indica.readers import ReaderProcessor
from indica.readers import ST40Reader

CONF = ST40Conf()
INSTRUMENT_METHODS = CONF.INSTRUMENT_METHODS


class SQLDatabase:
    """Process one pulse and one MDS+ tree, then write trend data to SQL."""

    def __init__(
        self,
        pulse: int,
        tree_name: str,
        database_name: str = "physics_trends",
        apply_data_q: bool = False,
        is_plasma_ip_threshold: float = 60.0e3,
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
        if database_name != "indica.database_test":
            raise Exception("Don't write to official Trends *yet*!!!!!")

        self.pulse = int(pulse)
        self.tree = tree_name.lower()
        self.database_name = database_name
        self.is_plasma_ip_threshold = is_plasma_ip_threshold
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
        self.t_sql = self.reader_processor.get_tlabels_dt(
            self.tlim[0], self.tlim[1], self.dt
        )

    def instantiate_data_reader(self):
        """Create the underlying ST40 reader used to fetch pulse data."""
        try:
            self.st40 = ST40Reader(
                self.pulse,
                self.tlim[0] - self.dt * 2,
                self.tlim[1] + self.dt * 2,
                tree=self.tree,
            )
        except Exception as e:
            self.debug_log(f"Error instantiating reader for {self.tree} - {e}")

    def get_runs_list(self):
        """
        TODO: implement this function in Indica
        List the available code runs for the target pulse and tree.
        """
        raise (NotImplementedError)

        # self.runs_list = []
        # try:
        #     runs_list = GetCodeRuns(self.pulse, code_names=[self.tree]).get_code_runs()
        #     self.runs_list = [run.split("#")[-1].strip() for run in runs_list]
        # except Exception as e:
        #     self.debug_log(f"Error reading runs list for {self.tree} - {e}")

    def get_best_run(self):
        """Identify the best run for the current pulse and tree."""
        # Works for both BEST and POST_BEST
        # Only 1 BEST is allowed --> len(_best_run_name) == 1
        self.best_run: str = ""
        best_run_name = [run for run in self.runs_list if "BEST" in run]
        if len(best_run_name) == 1 and "BEST" in best_run_name[0].upper():
            # There's no need to read BEST since you already know what it's pointing to
            self.runs_list.remove(best_run_name[0])

            # Continue even if data is unavailable
            try:
                _best_run, _ = self.st40.reader_utils.get_signal(
                    "", self.tree, ".BEST_RUN", best_run_name[0]
                )
                self.best_run = str(_best_run)
                self.debug_log(f"OK reading BEST run for {self.tree}")
            except Exception as e:
                self.debug_log(f"Error reading BEST run for {self.tree} - {e}")
        else:
            self.debug_log(f"No or duplicate BEST run(s) for {self.tree}")

    def get_static_data(self):
        """Read static metadata for each run in the pulse and tree."""
        self.static_data: dict = {}
        for run in self.runs_list:
            data = {}
            for quantity in self.static_to_write:
                _key = quantity.split(".")[-1].split(":")[-1]
                is_datetime = strip_mds_path(quantity.lower()) == "datetime"

                # Continue even if data is unavailable
                data[_key] = None if is_datetime else ""
                try:
                    # TODO: currently implemented only for string static data
                    _data, _ = self.st40.reader_utils.get_signal(
                        "", self.tree, f".{quantity}", run
                    )
                    data[_key] = (
                        format_sql_datetime(_data)
                        if is_datetime
                        else str(_data).strip()
                    )
                    if is_datetime and data[_key] is None:
                        self.debug_log(
                            f"Missing or invalid static {quantity} for {self.tree}#{run}: {_data!r}; using SQL NULL"
                        )
                except Exception as e:
                    self.debug_log(
                        f"Error reading static {quantity} for {self.tree}#{run} - {e}"
                    )

            self.static_data[run] = deepcopy(data)

    def get_raw_global(self):
        """Read the raw global data for all available runs."""
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
        """Process and align the raw global data on the common time grid."""
        self.global_data = {}
        for run in self.global_data_raw.keys():
            try:
                processed = self.reader_processor(
                    {self.tree: self.global_data_raw[run]},
                    tstart=self.tlim[0],
                    tend=self.tlim[1],
                    dt=self.dt,
                    overlap=self.overlap,
                    check_bounds=False,
                )[self.tree]
            except Exception:
                continue

            data: dict[DataArray] = {}
            for quantity in processed.keys():
                try:
                    _data = processed[quantity]
                    dims = _data.dims

                    # Calculate time derivative (d/dt) for trend analysis
                    _data = _data.assign_coords(
                        d_dt=(dims, _data.differentiate("t").data)
                    )

                    # Create quality mask: 1 for valid data, NaN for poor quality
                    # Only applied if data_q coordinate exists and apply_data_q flag is set
                    mask = xr.full_like(_data, 1)
                    if "data_q" in _data and "t" in dims and self.apply_data_q:
                        # Threshold: keep data where data_q > 0.9, mask others as NaN
                        mask = xr.where(_data["data_q"] > 0.9, 1, np.nan)

                    # Apply quality mask to data and all coordinate arrays
                    data[quantity] = _data * mask
                    for coord in ["d_dt", "error", "stdev"]:
                        data[quantity] = data[quantity].assign_coords(
                            {coord: (dims, (_data.coords[coord] * mask).data)}
                        )

                    self.global_data[run] = deepcopy(data)
                except Exception as e:
                    self.debug_log(f"Error post-processing {run} - {e}")

    def write_to_sql_pulse(self):
        """
        TODO: implement this function in Indica

        Write the pulse-level metadata row to the SQL pulse table.
        """

        raise (NotImplementedError)

        # had_plasma = 0

        # results = GetData(self.pulse, "PULSE#")
        # datetime = results.get("DATETIME")
        # pre_comment = results.get("PRE_COMMENT")
        # post_comment = results.get("POST_COMMENT")

        # if not (
        #     isinstance(datetime, DatabaseFailure)
        #     or isinstance(pre_comment, DatabaseFailure)
        #     or isinstance(post_comment, DatabaseFailure)
        # ):
        #     results = GetData(self.pulse, "PFIT#POST_BEST")
        #     ip = results.get("GLOBAL.IP")
        #     if not isinstance(ip, DatabaseFailure):
        #         ip_max = np.nanmax(ip)
        #         if not np.isnan(ip_max) and ip_max > self.is_plasma_ip_threshold:
        #             had_plasma = 1

        #     table_name = "pulse"
        #     data_list = [
        #         {"SQL_key": "pulseNo", "data": int(self.pulse)},
        #         {"SQL_key": "datetime", "data": format_sql_datetime(datetime)},
        #         {"SQL_key": "pre_comment", "data": pre_comment},
        #         {"SQL_key": "post_comment", "data": post_comment},
        #         {"SQL_key": "had_plasma", "data": had_plasma},
        #     ]
        #     self.pulse_sql_data_dict = {"pulse": data_list}
        #     self.write_to_mysql(data_list, table_name, debug_str=table_name)

        # return had_plasma

    def write_to_sql_best_linkage(self):
        """Write the best-run linkage for the current pulse and tree."""
        table_name = "best_linkage"

        data_list = [
            {"SQL_key": "pulseNo", "data": self.pulse},
            {"SQL_key": "code_name", "data": self.tree},
            {"SQL_key": "run_name", "data": self.best_run},
        ]

        # data_list = [
        #     {"SQL_key": "pulseNo", "data": str(self.pulse)},
        #     {"SQL_key": f"{self.tree_type}_{self.tree}_best", "data": self.best_run},
        # ]
        self.write_to_mysql(data_list, table_name)

    def write_to_sql_static(self):
        """Write the static metadata rows for each run to SQL."""
        table_name = f"{self.tree_type}_static"

        data_dict = {}
        for run, _data in self.static_data.items():
            # Each entry must have info on pulse, tree and run
            data_list = [
                {"SQL_key": "pulseNo", "data": self.pulse},
                {"SQL_key": "code_name", "data": self.tree},
                {"SQL_key": "run_name", "data": run},
            ]
            for _quantity, data in _data.items():
                quantity = _quantity.lower()
                data_list.append({"SQL_key": quantity, "data": data})
            data_dict[run] = data_list
            self.write_to_mysql(data_list, table_name)

    def write_to_sql_global(self):
        """Write the processed global trend rows for each time bin to SQL."""
        table_name = f"{self.tree_type}_global"

        data_dict = {}
        for run, _data in self.global_data.items():
            for _quantity, data in _data.items():
                quantity = _quantity.lower()
                for t in self.t_sql:
                    data_list = [
                        {"SQL_key": "pulseNo", "data": self.pulse},
                        {"SQL_key": "code_name", "data": self.tree},
                        {"SQL_key": "run_name", "data": run},
                        {
                            "SQL_key": "time",
                            "data": f"{t:.3f}",
                        },  # use str to avoid floating point issues
                    ]

                    # Skip timepoint
                    if (
                        np.isnan(float(data.sel(t=t).data))
                        or np.isnan(float(data.error.sel(t=t).data))
                        or np.isnan(float(data.stdev.sel(t=t).data))
                        or np.isnan(float(data.d_dt.sel(t=t).data))
                    ):
                        continue

                    # Each entry must have info on pulse, tree and run
                    data_list.append(
                        {"SQL_key": quantity, "data": float(data.sel(t=t).data)}
                    )
                    data_list.append(
                        {
                            "SQL_key": f"{quantity}_error",
                            "data": float(data.error.sel(t=t).data),
                        }
                    )
                    data_list.append(
                        {
                            "SQL_key": f"{quantity}_stdev",
                            "data": float(data.stdev.sel(t=t).data),
                        }
                    )
                    data_list.append(
                        {
                            "SQL_key": f"{quantity}_d_dt",
                            "data": float(data.d_dt.sel(t=t).data),
                        }
                    )
                    self.write_to_mysql(data_list, table_name)

    def write_to_mysql(
        self, data_list: list[dict], table_name: str, debug_str: str = ""
    ):
        """Insert or update one row of SQL data using the project table schema."""
        if not self.write:
            return

        if "test" not in self.database_name:
            raise (
                Exception("Allowed to write only to test database for the time being!")
            )

        # Connect to MySQL
        mydb = mysql.connector.connect(**self.sql_credentials)

        comma_new_line = ",\n            "
        # Use INSERT ... ON DUPLICATE KEY UPDATE for idempotent writes
        # (updates existing rows if all key fields match)
        sql_query = f"""
            INSERT INTO {table_name} ({", ".join(data["SQL_key"] for data in data_list)})
            VALUES({", ".join("%s" for _ in data_list)})
            ON DUPLICATE KEY UPDATE
                {comma_new_line.join(data["SQL_key"] + " = VALUES(" + data["SQL_key"] + ")" for data in data_list)}
        """

        # Convert data to tuple of values, preserving None for SQL NULL
        # Example: ("abc", "-2", None, None)
        values = tuple(
            f"{data['data']}" if data["data"] is not None else None
            for data in data_list
        )

        mycursor = mydb.cursor()
        mycursor.execute(sql_query, values)
        mydb.commit()
        mydb.close()

        self.debug_log(f"Writing {debug_str} \n {sql_query}")

    def get_elapsed_time(self):
        """Return the runtime since the controller was created."""
        return f"{time() - self.t0_runtime:.3f}"

    def debug_log(self, message: str):
        """Store and optionally print a timed debug message."""
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
        """Run an operation while recording timing and error messages."""
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
        """Run the full trend workflow for the selected pulse and tree."""
        # Write pulse info to SQL
        had_plasma = 0
        try:
            had_plasma = self.write_to_sql_pulse()
        except Exception as e:
            self.debug_log(f"Skipping {self.pulse} pulse table - {e}")
            return

        if not (had_plasma):
            self.debug_log(f"Skipping {self.pulse} as not a plasma")
            return

        # Instantiate reader
        try:
            self.get_runs_list()
            self.instantiate_data_reader()
        except Exception as e:
            self.debug_log(f"Skipping {self.pulse}, error instantiating reader - {e}")
            return

        # Get best run and write it to SQL
        try:
            self.get_best_run()
            self.write_to_sql_best_linkage()
        except Exception as e:
            self.debug_log(f"Skipping {self.pulse} best linkage table - {e}")

        # # Get static data and write it to SQL
        # try:
        #     self.get_static_data()
        #     self.write_to_sql_static()
        # except Exception as e:
        #     self.debug_log(f"Skipping {self.pulse} static table - {e}")

        # # Get raw data, process it, and write it to SQL
        # try:
        #     self.get_raw_global()
        #     self.process_global()
        #     self.write_to_sql_global()
        # except Exception as e:
        #     self.debug_log(f"Skipping {self.pulse} static table - {e}")

        # Close Indica reader MDS+ connection
        if hasattr(self, "st40"):
            self.st40.close()


def scrape(
    pulse_numbers: list[int],
    tree_names: list = [],
    database_name: str = "indica.database_test",
    apply_data_q: bool = False,
    debug: bool = False,
    write_to_sql: bool = False,
):
    """Batch process multiple pulses and trees through the trend workflow.

    Iterates through all combinations of pulse numbers and tree names,
    instantiating a controller for each and running the full workflow.

    Parameters
    ----------
    pulse_numbers : list[int]
        List of pulse numbers to process
    tree_names : list, optional
        List of MDS+ tree names. If empty, all configured trees are used
    database_name : str, optional
        Target database name (default: "indica.database_test")
    apply_data_q : bool, optional
        Apply data quality masking (default: False)
    debug : bool, optional
        Enable debug logging (default: False)
    write_to_sql : bool, optional
        Actually write results to database (default: False)
    """

    if len(tree_names) == 0:
        tree_names = TREE_NAMES

    for pulse in pulse_numbers:
        for tree in tree_names:
            controller = SQLDatabase(
                pulse,
                tree,
                database_name,
                apply_data_q=apply_data_q,
                debug=debug,
                write=write_to_sql,
            )
            controller()


def print_skip(tree_name: str, run: str):
    """Print a skip message for a failed data read."""
    print(f"Error reading {tree_name} {run} - skipping")


def strip_mds_path(mds_path: str):
    """Extract the final segment of an MDS+ node path.

    Removes all prefix and punctuation (dots and colons) to get
    just the node name. E.g., ".EFIT01:GFILE.DATETIME" → "DATETIME"

    Parameters
    ----------
    mds_path : str
        Full MDS+ path string

    Returns
    -------
    str
        The final node name component
    """
    return mds_path.split(".")[-1].split(":")[-1]


def format_sql_datetime(value: Any) -> str | None:
    """Return a MySQL datetime string, or None (SQL NULL) for missing or invalid values."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value == ".":
        return None

    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(value.replace("T", " "))
    except ValueError:
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed.isoformat(sep=" ", timespec="milliseconds")


def format_mdspath(mdspath: str):
    """Return node name only, stripping MDS+ path of all punctuation"""
    _path = deepcopy(mdspath)
    if ":" in _path:
        _path = _path.split(":")[-1]
    if "." in _path:
        _path = _path.split(".")[-1]
    return _path.lower()
