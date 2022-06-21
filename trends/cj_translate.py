import json

import numpy
import pymysql
from trends.info_dict import info_dict


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
    Generates a JSON dictionary from python dictionary which includes all pulses and corresponding data

    Returns
    -------
    JSON string which can be pushed to mysql
    """
    regr_data = database(reload=True)  # Load in database
    time = regr_data.time  # global time values

    param_dict = call_dict()  # Need the dictionary for specific parameters for each pulse (e.g., units)
    # print(param_dict)

    keys = []  # list of all keys (e.g., ipla_efit)
    for key in param_dict:
        keys.append(key)

    # TODO: use all keys
    keys_test = []
    for i in range(0, 2):
        keys_test.append((keys[i]))

    param_list = ['data', 'gradient', 'error_lower', 'error_upper', 'display_unit', 'display_const', 'label']

    # TODO: finalise these lists
    stat_test_param_list = ['data', 'error_lower', 'error_upper', 'display_unit', 'display_const', 'label']
    bin_test_param_list = ['display_unit', 'display_const', 'label']

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

            list_properties = [value, gradient, error_l, error_u, unit, const]  # list of all run properties
            temp_list_properties = [unit, const, label]

            """
            The following code adds the data to the 'max_val' and 'min_val'
            """

            base_json['static']['max_val'][key.replace('_', '#')] = {}
            base_json['static']['min_val'][key.replace('_', '#')] = {}

            for i in range(0, len(bin_test_param_list)):
                if type(temp_list_properties[i]) == numpy.ndarray:
                    base_json['static']['max_val'][key.replace('_', '#')][stat_test_param_list[i]] = list(
                        temp_list_properties[i])
                    base_json['static']['min_val'][key.replace('_', '#')][stat_test_param_list[i]] = list(
                        temp_list_properties[i])

                else:
                    base_json['static']['max_val'][key.replace('_', '#')][stat_test_param_list[i]] = \
                        temp_list_properties[i]
                    base_json['static']['min_val'][key.replace('_', '#')][stat_test_param_list[i]] = \
                        temp_list_properties[i]

            """
            The following code adds the data to the 'binned' key
            """

            base_json['binned']['time'] = list(time)  # add time data
            base_json['binned']['pulseNo'] = [pulseNo] * len(time)  # add pulse * time
            base_json['binned'][key.replace('_', '#')] = {}  # insert json variable (e.g., ip#efit)

            for i in range(0, len(bin_test_param_list)):
                if type(temp_list_properties[i]) == numpy.ndarray:
                    base_json['binned'][key.replace('_', '#')][bin_test_param_list[i]] = list(temp_list_properties[i])
                else:
                    base_json['binned'][key.replace('_', '#')][bin_test_param_list[i]] = temp_list_properties[i]

        # Convert python's json to a string (json encoding)
        # Note - python's default json package does not produce "correct" json
        # ("NaN" is not in the json specification, but "null" is)
        json_str = json.dumps(base_json).replace('NaN', 'null')
        json_list.append(json_str)

    return json_list


def create_my_sql():
    pymysql_connector = pymysql.connect(
        user="marco.sertoli",
        password='Marco3142!',
        host='192.168.1.9',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )
    cursor = pymysql_connector.cursor()
    cursor.execute("CREATE DATABASE `Trends`")


def write_to_mysql(json_list: list):
    """
    Writes json strings to the database

    Parameters
    ----------
    json_list

    Returns
    -------
    writes to pymysql
    """
    # pymysql_connector = pymysql.connect(
    #     user="marco.sertoli",
    #     password="Marco3142!",
    #     host="192.168.1.9",
    #     database='st40_test',
    #     port=3306,
    # )

    for json_str in json_list:
        pulseNo = json.loads(json_str)['static']['pulseNo']
        # print(pulseNo)

        pymysql_connector = pymysql.connect(
            user="marco.sertoli",
            password="Marco3142!",
            host="192.168.1.9",
            database='st40_test',
            port=3306,
            cursorclass=pymysql.cursors.DictCursor

        )
        # print(json_str)
        with pymysql_connector:
            with pymysql_connector.cursor() as cursor:
                # Delete this pulse from MySQL if it already exists
                # (nothing bad happens if pulse does not exist)
                sql = "DELETE FROM regression_database WHERE pulseNo = " + str(pulseNo)
                cursor.execute(sql)
                print('Removing duplicate pulse')

                # Add into MySQL
                sql = "INSERT INTO `regression_database` (pulseNo, data) VALUES (%s, %s)"
                val = (str(pulseNo), json_str)
                print(val)
                cursor.execute(sql, val)
                result = cursor.fetchall()
                print('Adding pulse to MySQL = ', result)
    return print('Done')


def read_from_mysql(query: str, key: str = None, variable: str = None, data_type: str = None, value: str = None):
    """
    Reads data from regression database

    Parameters
    ----------
    query
    key
    variable
    data_type
    value

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
    pymysql_connector = pymysql.connect(
        user='marco.sertoli',
        password='Marco3142!',
        host='192.168.1.9',
        database='st40_test',
        port=3306,
        cursorclass=pymysql.cursors.DictCursor
    )
    with pymysql_connector:
        with pymysql_connector.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            database = json.loads(result['data'])

            if key is None:
                return database

            if key != 'static' or 'binned':
                return ValueError("You have not specified an appropriate key. Must be 'binned' or 'static'")

            if key == 'static':
                return call_static(database, variable, data_type, value)

            if key == 'binned':
                return call_binned(database, variable, data_type)

    # TODO: the dictionary structure for this example is missing some sections such as the 'max_val' contents
    # TODO: e.g. the 'static' key has an extra layer of depth compared to 'binned'


def call_binned(database, variable: str = None, data_type: str = None):
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
    else:
        return database['binned'][variable]


# TODO: this does not work for this dataset as the max_val -> ip#efit is empty
def call_static(data: dict, variable: str, data_type: str, value: str = None):
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
    else:
        return data['static'][variable]
