import json

import numpy
import pymysql
from trends.info_dict import info_dict

def login_to_mysql(username: str = None, password: str = None, database_name: str = "st40_test"):
    pymysql_connector = pymysql.connect(
        user=username,
        password=password,
        host='192.168.1.9',
        database=database_name,
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )
    return pymysql_connector
login = login_to_mysql("marco.sertoli", 'Marco3142!')


def call_dict():
    """
    calls the dictionary, giving access to variable names and the corresponding quantities

    Returns
    -------
    dictionary
    """
    dictionary = info_dict()
    return dictionary


def translate_to_json(database):
    """
    Convert python's json to a string (json encoding)
    Note - python's default json package does not produce "correct" json
    ("NaN" is not in the json specification, but "null" is)

    Returns
    -------
    A list of JSON strings which can be pushed to mysql database
    """
    regr_data = database(reload=True)  # Load in database
    time = regr_data.time  # global time values

    param_dict = call_dict()  # Need the dictionary for specific parameters for each pulse (e.g., units)
    # print(param_dict)

    keys = []
    for key in param_dict:
        keys.append(key)

    # TODO: use all keys
    keys_test = []
    for i in range(0, 2):
        keys_test.append((keys[i]))

    stat_list_str = ['data', 'error_lower', 'error_upper', 'display_unit', 'display_const', 'label']
    bin_test_param_list = ['data', 'gradient', 'error_lower', 'error_upper', 'display_unit', 'display_const', 'label']

    pulseNos = regr_data.binned['ipla_efit'].to_dict()['coords']['pulse']['data']  # find all pulse numbers

    json_list = []  # a list of json string dictionaries

    for pulseNo in pulseNos:  # loop through all pulses and create json string
        # base JSON structure which we 'append' to the master json

        base_json = {
            'static': {
                'max_val': {},
                'min_val': {},
                'mc_charge#v': None,
                'had_plasma': True,
                'had_rfx': True,
                'had_hnbi1': True,
                'pulseNo': pulseNo,
                'datetime': None
            },
            'binned': {}}

        for key in keys_test:  # loop through all keys
            binned_data = regr_data.binned[key].sel(pulse=pulseNo)  # data for a specific pulse number

            # all relevant quantities for each key
            value = binned_data.value.values
            gradient = binned_data.gradient.values
            error_l = binned_data.error.values
            error_u = binned_data.error.values
            const = param_dict[key]['const']
            unit = param_dict[key]['units']
            label = param_dict[key]['label']

            stat_list_prop = [value, error_l, error_u, unit, const, label]
            bin_list_properties = [value, gradient, error_l, error_u, unit, const, label]

            """
            The following code adds the data to the 'max_val' and 'min_val'
            """

            base_json['static']['max_val'][key.replace('_', '#')] = {}
            base_json['static']['min_val'][key.replace('_', '#')] = {}

            for i in range(0, len(stat_list_prop)):
                if type(stat_list_prop[i]) == numpy.ndarray:
                    base_json['static']['max_val'][key.replace('_', '#')][stat_list_str[i]] = list(
                        stat_list_prop[i])
                    base_json['static']['min_val'][key.replace('_', '#')][stat_list_str[i]] = list(
                        stat_list_prop[i])

                else:
                    base_json['static']['max_val'][key.replace('_', '#')][stat_list_str[i]] = \
                        stat_list_prop[i]
                    base_json['static']['min_val'][key.replace('_', '#')][stat_list_str[i]] = \
                        stat_list_prop[i]

            """
            The following code adds the data to the 'binned' key
            """

            base_json['binned']['time'] = list(time)  # add time data
            base_json['binned']['pulseNo'] = [pulseNo] * len(time)  # add pulse * time
            base_json['binned'][key.replace('_', '#')] = {}  # insert json variable (e.g., ip#efit)

            for i in range(0, len(bin_list_properties)):
                if type(bin_list_properties[i]) == numpy.ndarray:
                    base_json['binned'][key.replace('_', '#')][bin_test_param_list[i]] = list(bin_list_properties[i])
                else:
                    base_json['binned'][key.replace('_', '#')][bin_test_param_list[i]] = bin_list_properties[i]

        # Convert python's json to a string (json encoding)
        # Note - python's default json package does not produce "correct" json
        # ("NaN" is not in the json specification, but "null" is)
        json_str = json.dumps(base_json).replace('NaN', 'null')
        json_list.append(json_str)

    return json_list


def create_my_sql():
    database_connector = login_to_mysql('marco.sertoli', 'Marco3142!')
    cursor = database_connector.cursor()
    cursor.execute("CREATE DATABASE `Trends`")


def commit_to_mysql(query: str,
                    entry):
    database_connector = login_to_mysql('marco.sertoli', 'Marco3142!')
    with database_connector:
        with database_connector.cursor() as cursor:
            cursor.execute(query, entry)
            database_connector.commit()


def write_to_mysql(json_list: list
                   ):
    """
    Writes json strings to the database

    Parameters
    ----------
    json_list

    Returns
    -------
    writes to pymysql
    """

    for json_str in json_list:
        pulseNo = json.loads(json_str)['static']['pulseNo']

        database_connector = login_to_mysql('marco.sertoli', 'Marco3142!')

        with database_connector:
            with database_connector.cursor() as cursor:
                # Delete this pulse from MySQL if it already exists
                query = "DELETE FROM regression_database WHERE pulseNo = " + str(pulseNo)
                commit_to_mysql(query, None)

                # Add into MySQL
                query = "INSERT INTO `regression_database` (pulseNo, data) VALUES (%s, %s)"
                val = (str(pulseNo), json_str)
                cursor.execute(query, val)
                database_connector.commit()
                return 'yes'



def delete_from_mysql():
    database_connector = login_to_mysql('marco.sertoli', 'Marco3142!')

    with database_connector:
        with database_connector.cursor() as cursor:
            cursor.execute("DELETE FROM `regression_database`")
            database_connector.commit()
            return read_from_mysql("SELECT * FROM `regression_database`")


# TODO: Able to extract just plasma current data from all pulses

def read_from_mysql(pulseNo: int = None,
                    key: str = None,
                    variable: str = None,
                    data_type: str = None,
                    value: str = None,
                    json_list: list = None):
    """
    Reads data from regression database

    Parameters
    ----------
    pulseNo
    key
    variable
    data_type
    value
    json_list

    Example structure:
    Database  = {'binned': {                (key)
                            'time': []      (variable)
                            'pulseNo': number
                            'ip#efit': {
                                        'data': [data]    (data_type)
                                        'gradient': [gradients]}},

                'static': {                 (key)
                            'max_val': {    (variable)
                                        'ip#efit': {      (data_type)
                                                    data': [data],              (value)
                                                    'error_lower': [errors]}},
                            'pulseNo': number}}
    """

    database_connector = login_to_mysql('marco.sertoli', 'Marco3142!')

    if pulseNo is not None:
        with database_connector:
            with database_connector.cursor() as cursor:
                cursor.execute("SELECT data FROM `regression_database` WHERE pulseNo = " + str(pulseNo))
                result = cursor.fetchall()
                database = json.loads(result[0]['data'])

                if key is None:
                    return database

                if key == 'static':
                    return call_static(database, variable, data_type, value)

                if key == 'binned':
                    return call_binned(database, variable, data_type)

    if pulseNo is None:


    # TODO: the dictionary structure for this example is missing some sections such as the 'max_val' contents
    # TODO: e.g. the 'static' key has an extra layer of depth compared to 'binned'


def update_mysql():
    return


def call_binned(database,
                variable: str = None,
                data_type: str = None):
    """
    Returns values from the 'binned' key

    Parameters
    ----------
    database
    variable
    data_type

    Returns
    -------
    The desired values of your input
    """
    if data_type is not None:
        return database['binned'][variable][data_type]

    elif data_type is None and variable is not None:
        return database['binned'][variable]

    elif variable is None:
        return database['binned']


# TODO: this does not work for this dataset as the max_val -> ip#efit is empty
def call_static(data: dict,
                variable: str = None,
                data_type: str = None,
                value: str = None):
    """
    Returns values from the 'static' key

    Parameters
    ----------
    data
    variable
    data_type
    value

    Returns
    -------

    """

    if value is not None:
        return data['static'][variable][data_type][value]
    elif value is None and data_type is not None and variable is not None:
        return data['static'][variable][data_type]
    elif value is None and data_type is None and variable is not None:
        return data['static'][variable]
    elif value is None and data_type is None and variable is None:
        return data['static']
