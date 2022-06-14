import json
import pymysql.connector
import mysql
import numpy as np

import hda.regression_analysis as regr

# Read "regression database" data in
#regr_data = regr.Database(pulse_start=9169, pulse_end=9170, reload=False)
regr_data = regr.Database(pulse_start=9207, pulse_end=9229, reload=False)
regr_data()  # Simulating XRCS measurement for Te(0) re-scaling

# Connect to MySQL
# mysql_conn = mysql.connector.connect(
#     user='peter.buxton',
#     password='qwerty007a',
#     host='192.168.1.9',
#     database='st40_test',
#     port='3306'
# )
# mysql_cursor = mysql_conn.cursor()
mysql_conn = pymysql.connect(
    user="marco.sertoli",
    password='Marco3142!',
    host='192.168.1.9',
    database='st40_test',
    port=3306
)
mysql_cursor = mysql_conn.cursor()

# Extract the pulseNos from "regression database"
data_dict = regr_data.binned['ipla_efit'].to_dict()
pulseNos = data_dict['coords']['pulse']['data']

# Loop over each pulse and write to MySQL
print('writing to MySQL')
for iPulse, pulseNo in enumerate(pulseNos):
    data_dict = regr_data.binned['ipla_efit'].to_dict()
    time = data_dict['coords']['t']['data']
    Ip = data_dict['data_vars']['value']['data'][iPulse]
    Ip_gradient = data_dict['data_vars']['gradient']['data'][iPulse]
    Ip_error = data_dict['data_vars']['error']['data'][iPulse]

    data_dict = regr_data.binned['wp_efit'].to_dict()
    WMHD = data_dict['data_vars']['value']['data'][iPulse]
    WMHD_gradient = data_dict['data_vars']['gradient']['data'][iPulse]
    WMHD_error = data_dict['data_vars']['error']['data'][iPulse]

    # Te0__xrcs
    data_dict = regr_data.binned['te0'].to_dict()
    te0_xrcs = data_dict['data_vars']['value']['data'][iPulse]
    te0_gradient = data_dict['data_vars']['gradient']['data'][iPulse]
    te0_error = data_dict['data_vars']['error']['data'][iPulse]

    # Ti0__xrcs
    data_dict = regr_data.binned['ti0'].to_dict()
    ti0_xrcs = data_dict['data_vars']['value']['data'][iPulse]
    ti0_gradient = data_dict['data_vars']['gradient']['data'][iPulse]
    ti0_error = data_dict['data_vars']['error']['data'][iPulse]

    # Make the JSON structure
    JSON = {
        'static': {
            'max_val': {
                'ip#efit': None,
                'wmhd#efit': None
            },
            'mc_charge_v': None,
            'had_plasma': True,
            'had_rfx': True,
            'had_hnbi1': True,
            'pulseNo': pulseNo,
            'datetime': None
        },
        'binned': {
            'time': time,
            'pulseNo': [pulseNo]*len(time),
            'ip#efit': {
                'data': Ip,
                'gradient': Ip_gradient,
                'error_lower': Ip_error,
                'error_upper': Ip_error,
                'help': 'Plasma current',
                'unit': 'ampere',
                'display_const': 1e-3,
                'display_unit': 'kA',
            },
            'wmhd#efit': {
                'data': WMHD,
                'gradient': WMHD_gradient,
                'error_lower': WMHD_error,
                'error_upper': WMHD_error,
                'help': 'Stored energy',
                'unit': 'joule',
                'display_const': 1e-3,
                'display_unit': 'kJ',
            },
            # 'wdia#dialoop': {
                # 'data': WMHD,
                # 'gradient': WMHD_gradient,
                # 'error_lower': WMHD_error,
                # 'error_upper': WMHD_error,
                # 'help': '',
                # 'unit': 'joule'
            # },
            'ti0#xrcs': {
                'data': ti0_xrcs,
                'gradient': ti0_gradient,
                'error_lower': ti0_error,
                'error_upper': ti0_error,
                'help': 'Central ion temperature',
                'unit': 'electronvolt',
                'display_const': 1e-3,
                'display_unit': 'keV',
            },
            'te0#xrcs': {
                'data': te0_xrcs,
                'gradient': te0_gradient,
                'error_lower': te0_error,
                'error_upper': te0_error,
                'help': 'Central electron temperature',
                'unit': 'electronvolt',
                'display_const': 1e-3,
                'display_unit': 'keV',
            },
            'ne_int#nirh1': {
                'data': regr_data.binned['ne_nirh1'].to_dict()['data_vars']['value']['data'][iPulse],
                'gradient': regr_data.binned['ne_nirh1'].to_dict()['data_vars']['gradient']['data'][iPulse],
                'error_lower': regr_data.binned['ne_nirh1'].to_dict()['data_vars']['error']['data'][iPulse],
                'error_upper': regr_data.binned['ne_nirh1'].to_dict()['data_vars']['error']['data'][iPulse],
                'help': 'Line integrated electron density',
                'unit': 'metre**-2',
                'display_const': 1e-19,
                'display_unit': '10<sup>19</sup> m<sup>-2</sup>',
            },
            'ne_int#smmh1': {
                'data': regr_data.binned['ne_smmh1'].to_dict()['data_vars']['value']['data'][iPulse],
                'gradient': regr_data.binned['ne_smmh1'].to_dict()['data_vars']['gradient']['data'][iPulse],
                'error_lower': regr_data.binned['ne_smmh1'].to_dict()['data_vars']['error']['data'][iPulse],
                'error_upper': regr_data.binned['ne_smmh1'].to_dict()['data_vars']['error']['data'][iPulse],
                'help': 'Line integrated electron density',
                'unit': 'metre**-2',
                'display_const': 1e-19,
                'display_unit': '10<sup>19</sup> m<sup>-2</sup>',
            },
        }
    }

    # Convert python's json to a string (json encoding)
    # Note - python's default json package does not produce "correct" json
    # ("NaN" is not in the json specification, but "null" is)
    JSON_str = json.dumps(JSON).replace('NaN', 'null')

    # Delete this pulse from MySQL if it already exists
    # (nothing bad happens if pulse does not exist)
    sql = "DELETE FROM regression_database WHERE pulseno = " + str(pulseNo)
    mysql_cursor.execute(sql)

    # Add into MySQL
    sql = "INSERT INTO regression_database (pulseno, data) VALUES (%s, %s)"
    val = (str(pulseNo), JSON_str)
    mysql_cursor.execute(sql, val)

mysql_conn.commit()





