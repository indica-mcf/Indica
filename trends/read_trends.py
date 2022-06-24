from copy import deepcopy
import glob
import json
import os
import pathlib
import pickle
import ast

import numpy as np
import pymysql
from trends.info_dict import info_dict
import xarray as xr
from xarray import DataArray
from xarray import Dataset

from trends.trends_database import Database
from indica.readers import ST40Reader


class Trends(Database):

    def __init__(self,
                 # pulse_start: int = None,
                 # pulse_end: int = None,
                 # pulse_No: int = None,
                 # key: str = None,
                 # variable: str = None,
                 # data_type: str = None,
                 reload: bool = False):
        # self.pulse_min = pulse_start
        # self.pulse_max = pulse_end
        # self.pulse_No = pulse_No
        # self.key = key
        # self.variable = variable
        # self.data_type = data_type

        self.username = 'marco.sertoli'
        self.password = 'Marco3142!'
        self.database = Database(reload=True)

    def login_to_mysql(self, database_name: str = "st40_test"):
        pymysql_connector = pymysql.connect(
            user=self.username,
            password=self.password,
            host='192.168.1.9',
            database=database_name,
            port=3306,
            cursorclass=pymysql.cursors.DictCursor
        )
        return pymysql_connector

    def read_whole_database(self):
        database_connector = self.login_to_mysql()
        with database_connector:
            with database_connector.cursor() as cursor:
                cursor.execute("SELECT * FROM `regression_database`"
                               )
                self.database = cursor.fetchall()
                return self.database

    def run_get_simple(self, field, condition):
        database_connector = self.login_to_mysql()

        with database_connector:
            with database_connector.cursor() as cursor:
                cursor.execute(
                    "SELECT " + str(field) + " FROM `regression_database` WHERE " + condition
                )
                self.database = cursor.fetchall()
                return self.database

    def run_get_datasets(self, variables, field, condition):
        database_connector = self.login_to_mysql()
        with database_connector:
            with database_connector.cursor() as cursor:
                if type(variables) == str:
                    cursor.execute(
                        "SELECT " + str(field) + " FROM `regression_database` WHERE " + condition
                    )
                    data = cursor.fetchall()

                    return np.array(json.loads(data[0]['data ->> "$.binned.ipla_efit.data"'])).astype('float')

                if type(variables) == list
                    for variable in variables:



    def get(self,
            pulse_start: int = None,
            pulse_end: int = None,
            pulse_No: int = None,
            key: str = None,
            variable: str = None,
            data_type: str = None):

        if pulse_start and pulse_end and pulse_No is not None:
            return ValueError("Invalid input. Can only read in a pulse range, all pulses, or from a specific pulse")

        if pulse_start is not None and pulse_end is not None and pulse_start > pulse_end:
            return ValueError("pulse_start cannot be greater than end pulse_end.")

        if data_type is not None and variable is None:
            return ValueError("Cannot input 'data_type' without specifying 'variable'")

        if pulse_No is not None and variable is not None and data_type is not None and key is None:
            return ValueError("You must input a 'key' (e.g., 'binned').")

        # all data between in a pulse range (list of dictionaries)
        if pulse_start is not None and pulse_end is not None and variable is None:
            return self.run_get_simple('data', "pulseNo BETWEEN " + str(pulse_start) + " AND "
                                       + str(pulse_end) + " ORDER BY pulseNo")

        # all data from just one pulse (dictionary)
        if pulse_start is None and pulse_end is None and pulse_No is not None and variable is None and data_type is None:
            return self.run_get_simple('data', " pulseNo = " + str(pulse_No))

        # a single data_type for a single variable from a single pulse (e.g., data or gradient as np.ndarray from all
        # possible keys)
        if data_type is not None and variable is not None and pulse_No is not None:
            return self.run_get_datasets(variable, 'data ->> "$.' + str(key) + '.' + str(variable) + '.' +
                                         str(data_type) + '"', "pulseNo = " + str(pulse_No))




        # a single data_type for particular pulse range
        if key is not None and variable is not None and data_type is not None and pulse_start is not None and \
                pulse_end is not None:

            pulse_nos = np.array(self.get_pulse_nos(pulse_start, pulse_end)).astype('float')

            data = self.run_get('data ->> "$.' + str(key) + '.' + str(variable) + '.' + str(data_type) + '"'
                                + ", pulseno", "pulseNo BETWEEN " + str(pulse_start) + " AND " + str(pulse_end) +
                                " ORDER BY pulseNo")

            data_list = []
            for pulseNo in range(0, len(data)):
                data_list.append(json.loads(data[pulseNo]['data ->> "$.' + str(key) + '.' + str(variable) + '.' +
                                                          str(data_type) + '"']))

            data_list = np.array(data_list).astype('float')
            return [pulse_nos, data_list]

    def get_pulse_nos(self,
                      pulse_start: int = None,
                      pulse_end: int = None,
                      pulse_No: int = None):

        database_connector = self.login_to_mysql()

        if pulse_No is None and pulse_start is not None and pulse_end is not None:

            with database_connector:
                with database_connector.cursor() as cursor:
                    cursor.execute(
                        "SELECT data ->> '$.static.pulseNo' FROM `regression_database` WHERE pulseNo BETWEEN " +
                        str(pulse_start) + " AND " + str(pulse_end)
                    )

                    pulse_dicts = cursor.fetchall()
                    pulse_list = []
                    for pulseNo in range(0, len(pulse_dicts)):
                        pulse_list.append(pulse_dicts[pulseNo]["data ->> '$.static.pulseNo'"])
                    return np.array(pulse_list).astype('float')

        else:
            with database_connector:
                with database_connector.cursor() as cursor:
                    cursor.execute(
                        "SELECT data ->> '$.static.pulseNo' FROM `regression_database`"
                    )
                    pulse_dicts = cursor.fetchall()
                    pulse_list = []
                    for pulseNo in range(0, len(pulse_dicts)):
                        pulse_list.append(pulse_dicts[pulseNo]["data ->> '$.static.pulseNo'"])
                    return np.array(pulse_list).astype('float')

    def run_filter_test(self):
        database_connector = self.login_to_mysql()
        print('yes')
        # all data in pulse range (dictionary)
        with database_connector:
            with database_connector.cursor() as cursor:
                cursor.execute(
                    # "SELECT data ->> '$.static.pulseNo' FROM `regression_database`")
                    "SELECT data ->> '$.binned.ipla_efit.data', pulseno FROM `regression_database` WHERE pulseno BETWEEN 9770 and 9771")

                results = cursor.fetchall()
                pulse_list = []
                data_list = []
                for pulseNo in range(0, len(results)):
                    pulse_list.append(results[pulseNo]["pulseno"])
                    data_list.append(json.loads(results[pulseNo]["data ->> '$.binned.ipla_efit.data'"]))
                return np.array(data_list).astype('float')
