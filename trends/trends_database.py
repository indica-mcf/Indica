from copy import deepcopy
import json
import os
import glob
import pathlib
import pickle
import re

import numpy as np
from trends.info_dict import info_dict
import xarray as xr
from xarray import DataArray
from xarray import Dataset

from indica.readers import ST40Reader


class Database:
    """
    Read and manage Trends database

    ...add documentation on each method...

    test_flow()
        Example call and flow

    TODO: add the following quantities
    all NBI, li, betaP
    """

    def __init__(
        self,
        pulse_start: int = 8207,
        pulse_end: int = 10046,
        tlim: tuple = (-0.03, 0.3),
        dt: float = 0.01,
        overlap: float = 0.5,
        t_max: float = 0.02,
        ipla_min: float = 50.0e3,
        reload: bool = False,
        set_info: bool = False,
    ):
        """
        Initialise Trends database class

        If reload = True, the old version of the database is read in
        and all attributes assigned to present class

        Parameters
        ----------
        pulse_start
            first pulse of the series
        pulse_end
            last pulse of the series
        tlim
            time window for reading in the data
        dt
            time resolution of new time axis for binning
        overlap
            overlap for binning over new time axis (0.5 = 50% overlap)
        t_max
            time beyond which to search for maximum in plasma parameters
        ipla_min
            minimum plasma current (A) to include pulse in database
        reload
            set to True to reload version of the database written to file
        set_info
            set info dictionary from .py file
        """

        self.set_paths()
        if reload:
            self.reload_database(pulse_start, pulse_end)
        else:
            print(f"\n Building new database for pulse range {pulse_start}-{pulse_end}")

            self.pulse_start = pulse_start
            self.pulse_end = pulse_end

            if set_info:
                info = self.get_info_dict()
            else:
                info = self.get_info_json()
            self.info = info

            self.tlim = tlim
            self.dt = dt
            self.overlap = overlap
            self.time = np.arange(self.tlim[0], self.tlim[1], dt * overlap)
            self.t_max = t_max
            self.ipla_min = ipla_min

            self.initialize_structures()

    def __call__(self, write: bool = False):
        """
        Read all data for desired pulse range and write database to file

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        binned, max_val, min_val, pulses = self.read_data(
            self.pulse_start, self.pulse_end
        )
        self.binned = binned
        self.max_val = max_val
        self.min_val = min_val
        self.pulses = pulses

    def set_paths(
        self,
        path_data: str = None,
        file_info_json: str = None,
        file_info_py: str = None,
    ):
        """
        Set paths/file names for reading/saving the info dictionary, the database,
        plotting figures

        Parameters
        ----------
        path_data
            path where data and setup files will be saved
        file_info_json
            file name of JSON containing the information dictionary on what to
            read from MDS+
        """
        if path_data is None:
            path_data = f"{current_file_path()}/"
        if file_info_json is None:
            file_info_json = "info_dict.json"
        if file_info_py is None:
            file_info_py = "info_dict.py"

        self.path_data = path_data
        self.file_info_json = file_info_json
        self.file_info_py = file_info_py

    def database_filename(self, pulse_start: int = None, pulse_end: int = None, ext="pkl"):
        """
        Open Pickle file where Database has been saved
        If either of the two values is None, latest file will be searched for
        in the default directory

        Parameters
        ----------
        pulse_start
            Inital pulse in database
        pulse_end
            Last pulse in database

        Returns
        -------
            Name of file to be read

        """
        filename = None
        if pulse_start is None or pulse_end is None:
            path_file = f"{self.path_data}*trends_database.{ext}"
            file_list = glob.glob(path_file)
            if len(file_list) != 0:
                filename = max(file_list, key=os.path.getctime)
        else:
            filename = f"{pulse_start}_{pulse_end}_trends_database"

        if filename is None:
            print(f"\n File not found: {path_file}")
        else:
            print(f"\n Database read from file: {filename}")

        return filename

    def reload_database(
        self, pulse_start: int = None, pulse_end: int = None, source: str = "pickle"
    ):
        """
        Reload data from database

        Parameters
        ----------
        source
            Define source to read from (pickle, mysql, ...)

        Returns
        -------
            Database data/class as saved to file

        """

        if source.lower() == "pickle":
            _file = self.database_filename(pulse_start, pulse_end)
            _file = f"{self.path_data}{_file}.pkl"
            database = pickle.load(open(_file, "rb"))
            for k in list(database.__dict__):
                setattr(self, k, getattr(database, k))

            print(f"Trends database reloaded from {_file}")
        else:
            print("\n *Only* reading from Pickle file currently supported \n")
            raise ValueError

        return database

    def get_info_dict(self):
        """
        Get info dictionary using python function

        Returns
        -------
        info
            dictionary with information on what to read
        """
        info = info_dict()
        return info

    def get_info_json(self, file_info_json: str = None):
        """
        Get info dict from JSON file

        Returns
        -------
        info
            dictionary with information on what to read
        """
        if file_info_json is None:
            file_info_json = self.file_info_json

        _file = f"{self.path_data}{file_info_json}"
        with open(f"{_file}", "r") as f:
            info = json.load(f)

        return info

    def write_info_to_file(self, file_info_py: str = None, file_info_json: str = None):
        """
        Write info dictionary to .py and .json files
        """
        if file_info_py is None:
            file_info_py = self.file_info_py
        if file_info_json is None:
            file_info_json = self.file_info_json

        # JSON file
        _file = f"{self.path_data}{file_info_json}"
        _file_backup = f"{self.path_data}_{file_info_json}"
        try:
            info = self.get_info_json()
        except FileNotFoundError:
            info = {}

        if info != self.info:
            print(f"\n Updating {_file} file \n")
            rename_file(_file, _file_backup)
            _json = json.dumps(self.info)
            with open(_file, "w") as f:
                f.write(_json)
        else:
            print(f"\n {_file} file already up to date \n")

        # Python file
        _file = f"{self.path_data}{file_info_py}"
        _file_backup = f"{self.path_data}{file_info_py}"
        try:
            info = self.get_info_dict()
        except FileNotFoundError:
            info = {}

        if info != self.info:
            print(f"\n Updating {_file} file \n")
            rename_file(_file, _file_backup)
            with open(_file, "w") as f:
                f.write("def info_dict():\n")
                f.write(f"    info = {self.info} \n")
                f.write("    return info")
        else:
            print(f"\n {_file} file already up to date \n")

        # Re-read file and return info dictionary written within,
        # check that it's == class value
        info = self.get_info_json()
        assert info == self.info

        return info

    def initialize_structures(self):
        """
        Initialize common data structures to save data MDS+
        """
        constant = DataArray(np.nan)

        value = DataArray(np.nan)
        error = xr.full_like(value, np.nan)
        stdev = xr.full_like(value, np.nan)
        time = xr.full_like(value, np.nan)
        revision = xr.full_like(constant, np.nan)
        self.empty_max_val = Dataset(
            {
                "value": value,
                "error": error,
                "stdev": stdev,
                "time": time,
                "revision": revision,
            }
        )

        value = DataArray(np.full(self.time.shape, np.nan), coords=[("t", self.time)])
        error = xr.full_like(value, np.nan)
        stdev = xr.full_like(value, np.nan)
        gradient = xr.full_like(value, np.nan)
        cumul = xr.full_like(value, np.nan)
        revision = xr.full_like(constant, np.nan)
        self.empty_binned = Dataset(
            {
                "value": value,
                "error": error,
                "stdev": stdev,
                "gradient": gradient,
                "cumul": cumul,
                "revision": revision,
            }
        )

    def add_pulses(self, pulse_end: int):
        """
        Add data from newer pulses

        Parameters
        ----------
        pulse_end
            Last pulse to include in the analysis
        """
        pulse_start = self.pulse_end + 1
        if pulse_end < pulse_start:
            print("\n Only newer pulses can be added (...for the time being...) \n")
            return

        self.pulse_end = pulse_end
        binned, max_val, min_val, pulses = self.read_data(pulse_start, pulse_end,)

        if len(pulses) > 0:
            self.pulses = np.append(self.pulses, pulses)
            for k in self.info:
                self.binned[k] = xr.concat([self.binned[k], binned[k]], "pulse")
                self.max_val[k] = xr.concat([self.max_val[k], max_val[k]], "pulse")
                self.min_val[k] = xr.concat([self.min_val[k], max_val[k]], "pulse")

    def add_quantities(self, info: dict = None):
        """
        Add additional quantities to the database

        Temporary: define info structure here
        """
        if info is None:
            # TODO: add beam powers fo HNBI and RFX !!!
            print("No new items to add")
            return

        print(f"New items being added: {list(info)}")

        binned, max_val, min_val, pulses = self.read_data(
            self.pulse_start, self.pulse_end, info=info, pulse_list=self.pulses,
        )

        assert self.pulses == pulses

        for k in info.keys():
            self.info[k] = info[k]

        for k in binned.keys():
            self.binned[k] = binned[k]
            self.max_val[k] = max_val[k]
            self.min_val[k] = min_val[k]

    def read_data(
        self,
        pulse_start: int,
        pulse_end: int,
        info: dict = None,
        pulse_list: list = None,
    ):
        """
        Read data in time-range of interest

        Parameters
        ----------
        pulse_start
            first pulse of the series
        pulse_end
            last pulse of the series
        info
            database info dictionary
        pulse_list
            list or array of pulses to read
        """

        def append_empty(k, binned, max_val):
            binned[k].append(self.empty_binned)
            max_val[k].append(self.empty_max_val)
            min_val[k].append(self.empty_max_val)

        if info is None:
            info = self.info

        if pulse_list is None:
            pulse_list = np.arange(pulse_start, pulse_end + 1)

        binned: dict = {}
        max_val: dict = {}
        min_val: dict = {}
        for k in info.keys():
            binned[k] = []
            max_val[k] = []
            min_val[k] = []

        pulses = []
        for pulse in pulse_list:
            print(pulse)
            reader = ST40Reader(int(pulse), self.tlim[0], self.tlim[1],)

            proceed = pulse_ok(reader, self.tlim, ipla_min=self.ipla_min)
            if not proceed:
                print("Pulse not OK...")
                continue

            pulses.append(pulse)
            for k, v in info.items():
                # Add data structure to dictionaries only for quantities that have data
                if "uid" not in v.keys():
                    append_empty(k, binned, max_val)
                    continue

                data, dims = reader._get_data(v["uid"], v["diag"], v["node"], v["rev"])
                if np.array_equal(data, "FAILED") or np.array_equal(dims[0], "FAILED"):
                    append_empty(k, binned, max_val)
                    continue

                time = dims[0]
                if np.min(time) > np.max(self.tlim) or np.max(time) < np.min(self.tlim):
                    append_empty(k, binned, max_val)
                    continue

                err = None
                if v["err"] is not None:
                    err, _ = reader._get_data(v["uid"], v["diag"], v["err"], v["rev"])
                rev = v["rev"]
                if rev == 0:
                    rev = int(reader._get_revision(v["uid"], v["diag"], v["rev"]))

                binned_tmp, max_val_tmp, min_val_tmp = self.bin_in_time(
                    data, time, err,
                )
                binned_tmp.revision.values = rev
                max_val_tmp.revision.values = rev
                min_val_tmp.revision.values = rev

                binned[k].append(binned_tmp)
                max_val[k].append(max_val_tmp)
                min_val[k].append(min_val_tmp)

        if len(pulses) > 0:
            for k in binned.keys():
                binned[k] = xr.concat(binned[k], "pulse").assign_coords(
                    {"pulse": pulses}
                )
                max_val[k] = xr.concat(max_val[k], "pulse").assign_coords(
                    {"pulse": pulses}
                )
                min_val[k] = xr.concat(min_val[k], "pulse").assign_coords(
                    {"pulse": pulses}
                )

        return binned, max_val, min_val, pulses

    def bin_in_time(
        self, data: np.ndarray, time: np.ndarray, err: np.ndarray = None, sign=+1
    ):
        """
        TODO: account for texp for spectroscopy intensities if not already in MDS+

        Bin data in time, calculate maximum value in specified time range,
        create datasets for binned data and maximum values

        Parameters
        ----------
        data
            array of data to be binned
        time
            time axis of data to be binned
        err
            error of data to be binned

        Returns
        -------

        """
        time_new = self.time
        dt = self.dt
        binned = deepcopy(self.empty_binned)
        max_val = deepcopy(self.empty_max_val)
        min_val = deepcopy(self.empty_max_val)

        ifin = np.where(np.isfinite(data))[0]
        if len(ifin) < 2:
            return binned, max_val, min_val

        # Find max value of raw data in time range of analysis
        tind = np.where((time >= self.tlim[0]) * (time < self.tlim[1]))[0]
        max_ind = tind[np.nanargmax(data[tind])]
        max_val.value.values = data[max_ind]
        max_val.time.values = time[max_ind]

        min_ind = tind[np.nanargmin(data[tind])]
        min_val.value.values = data[min_ind]
        min_val.time.values = time[min_ind]
        if err is not None:
            max_val.error.values = err[max_ind]
            min_val.error.values = err[min_ind]

        # Bin in time following rules refined at initialzation
        time_binning = time_new[
            np.where((time_new >= np.nanmin(time)) * (time_new <= np.nanmax(time)))
        ]
        for t in time_binning:
            tind = (time >= t - dt / 2.0) * (time < t + dt / 2.0)
            tind_lt = time <= t

            if len(tind) > 0:
                data_tmp = data[tind]
                ifin = np.where(np.isfinite(data_tmp))[0]
                if len(ifin) >= 1:
                    binned.value.loc[dict(t=t)] = np.mean(data_tmp[ifin])
                    binned.cumul.loc[dict(t=t)] = np.sum(data[tind_lt]) * dt
                    # TODO: separate std from error propagation
                    # TODO: deviation from a linear evolution?
                    #  --> InMatlab: regress > error around linear reg
                    binned.stdev.loc[dict(t=t)] = np.std(data_tmp[ifin])
                    if err is not None:
                        err_tmp = err[tind]
                        binned.error.loc[dict(t=t)] = np.sqrt(
                            np.sum(err_tmp[ifin] ** 2)
                        ) / len(ifin)

        # TODO: calculate gradient uncertainty
        binned.gradient.values = binned.value.differentiate("t", edge_order=2)
        binned.cumul.values = xr.where(np.isfinite(binned.value), binned.cumul, np.nan)

        return binned, max_val, min_val


def write_to_pickle(database: Database, pkl_file: str = None):
    """
    Write database to local pickle file

    Parameters
    ----------
    database
        Database class to save to pickle
    """
    database.write_info_to_file()

    if pkl_file is None:
        pkl_file = f"{database.database_filename(database.pulse_start, database.pulse_end)}.pkl"
    _file = f"{database.path_data}{pkl_file}"
    print(f"Saving Trends database to \n {_file}")
    pickle.dump(database, open(_file, "wb"))


def write_to_mysql(database: Database, database_name: str = "st40_test"):
    """
    Write database to file(s), update info file(s)

    Parameters
    ----------
    database
        Database class to save to pickle
    """
    mysql_conn = pymysql.connect(
        user="marco.sertoli",
        password="Marco3142!",
        host="192.168.1.9",
        database="database_name",
        port=3306,
    )
    mysql_cursor = mysql_conn.cursor()


def pulse_ok(reader: ST40Reader, tlim: tuple, ipla_min: float = 50.0e3):
    """
    Check whether pulse meets requirements to be included in the database

    Parameters
    ----------
    reader
        ST40reader class already initialized for desired pulse
    tlim
        (start, end) time of desired time window (s)
    ipla_min
        minimum value of Ip to read in the data (A)


    Returns
    -------
    True or False whether the pulse is a good one or not

    """
    ipla_pfit, dims_pfit = reader._get_data(
        "", "pfit", ".post_best.results.global:ip", -1
    )
    (ipla_efit, dims_efit,) = reader._get_data("", "efit", ".constraints.ip:cvalue", 0)

    ok_pfit = not np.array_equal(ipla_pfit, "FAILED")
    ok_efit = not np.array_equal(ipla_efit, "FAILED")
    if ok_pfit or ok_efit:
        if ok_pfit:
            time_pfit = dims_pfit[0]
            ok_pfit = (
                (len(np.where(ipla_pfit > ipla_min)[0]) > 3)
                * (np.min(time_pfit) < np.max(tlim))
                * (np.max(time_pfit) > np.min(tlim))
            )
        if ok_efit:
            time_efit = dims_efit[0]
            ok_efit = (
                (len(np.where(ipla_efit > ipla_min)[0]) > 3)
                * (np.min(time_efit) < np.max(tlim))
                * (np.max(time_efit) > np.min(tlim))
            )

    return ok_pfit or ok_efit


def current_file_path():
    return str(pathlib.Path(__file__).parent.resolve())


def rename_file(_file: str, _file_backup: str):
    try:
        os.rename(_file, _file_backup)
    except FileNotFoundError:
        print(f"No backup required, {_file} does not exist")


def add_quantities(database: Database):
    info = {
        "p_hnbi1": {
            "uid": "nbi",
            "diag": "hnbi1",
            "node": ":pinj",
            "rev": "1",
            "err": None,
            "label": "P$_{HNBI1}$",
            "units": "(MW)",
            "const": 1.0,
        },
        "p_rfx": {
            "uid": "nbi",
            "diag": "rfx",
            "node": ":pinj",
            "rev": "1",
            "err": None,
            "label": "P$_{RFX}$",
            "units": "(MW)",
            "const": 1.0,
        },
    }
    database.add_quantities(info=info)

    write_to_pickle(database)

    database.write_info_to_file()

    return database


def fix_things(database: Database, assign: bool = True):
    """
    Place where to put temporary workflow to fix data in the database
    """
    return
    binned = deepcopy(database.binned)

    for k in binned.keys():
        binned[k].gradient.values = binned[k].value.differentiate("t", edge_order=2)

    if assign:
        database.binned = binned

    return database


def test_flow(
    pulse_start: int = 9770, pulse_end: int = 9790, pulse_add: int = 9800,
):
    # Initialize class
    database = Database(pulse_start=pulse_start, pulse_end=pulse_end,)
    # Read all data and save to class attributes
    database()

    # Add pulses to database
    database.add_pulses(pulse_add)

    # Write information and data to file
    write_to_pickle(database)

    return database


def run_workflow(
    pulse_start: int = 8207,
    pulse_end: int = 10046,
    set_info: bool = False,
    write: bool = False,
):
    """
    Run workflow to build Trends database from scratch
    """
    database = Database(
        pulse_start=pulse_start, pulse_end=pulse_end, set_info=set_info,
    )
    database()

    if write:
        write_to_pickle(database)

    return database
