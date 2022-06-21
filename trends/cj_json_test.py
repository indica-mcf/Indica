# from trends.info_dict import info_dict
# from trends.trends_database import Database
# pulseNo = 10014
# ipla = regr_data.binned['ipla_efit'].sel(pulse=pulseNo)


time = 1
pulseNo = 2
Ip = 3
Ip_gradient = 4
Ip_error = 5

WMHD = 3
WMHD_gradient = 4
WMHD_error = 5

JSON = {
    'binned': {'time': time, 'pulseNo': [pulseNo], 'ip#efit': {'data': Ip}}}

# print(JSON)

# def test_json_translate()
# x = type(JSON)
# print(x)
"""
Accessing elemets of a dictionary
"""
# print(JSON['binned']['ip#efit']['data'])


"""
Deleting elements
"""
# del JSON['binned']['time']
# print(JSON)

"""
Appending element to nested dictionary
"""
# JSON['binned']['ip#efit'] = {'data': Ip, 'gradient': Ip_gradient}
# print(JSON)


###
# New test

# JSON = {
#         'static': {
#             'max_val': {
#                 'ip#efit': None,
#                 'wmhd#efit': None
#             },
#             'mc_charge_v': None,
#             'had_plasma': True,
#             'had_rfx': True,
#             'had_hnbi1': True,
#             'pulseNo': pulseNo,
#             'datetime': None
#         },
#         'binned': {
#             'time': time,
#             'pulseNo': pulseNo,
#             'ip#efit': {
#                 'data': Ip,
#                 'gradient': Ip_gradient,
#                 'error_lower': Ip_error,
#                 'error_upper': Ip_error,
#                 'help': 'Plasma current',
#                 'unit': 'ampere',
#                 'display_const': 1e-3,
#                 'display_unit': 'kA'}}}
#
# print('ORIGINAL', JSON)
#
# test_JSON = {
#         'static': {
#             'max_val': {
#                 'ip#efit': None,
#                 'wmhd#efit': None
#             },
#             'mc_charge_v': None,
#             'had_plasma': True,
#             'had_rfx': True,
#             'had_hnbi1': True,
#             'pulseNo': pulseNo,
#             'datetime': None
#         },
#         'binned': {
#         }}
#
# test_JSON['binned'] = {'time': time,
#             'pulseNo': pulseNo, 'ip#efit': {
#                 'data': Ip,
#                 'gradient': Ip_gradient,
#                 'error_lower': Ip_error,
#                 'error_upper': Ip_error,
#                 'help': 'Plasma current',
#                 'unit': 'ampere',
#                 'display_const': 1e-3,
#                 'display_unit': 'kA'}}
# print('NEW',test_JSON)
#
# if test_JSON == JSON:
#     print('TRUE')
# #

"""
New test
"""
variable = 0
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
        'pulseNo': pulseNo,
        'ip#efit': {
            'data': Ip,
            'gradient': Ip_gradient,
            'error_lower': Ip_error,
            'error_upper': Ip_error},

        'wmhd#efit': {
            'data': WMHD,
            'gradient': WMHD_gradient,
            'error_lower': WMHD_error,
            'error_upper': WMHD_error,
        }}}

test_JSON = {
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
        'pulseNo': pulseNo}}

# params = ['ip#efit', 'wmhd#efit']
# param_prop = ['data', 'gradient', 'error_lower', 'error_upper']
# other = ['help', 'unit', 'const', 'mult']
# value = [Ip, Ip_gradient, Ip_error, Ip_error]


# for i in range(0, len(parameter)):


# def appender(params):
#     regr_data = Database(reload=True)
#
#
#
#     data_dict = regr_data.binned['ipla_efit'].to_dict()
#     pulseNos = data_dict['coords']['pulse']['data']
#     time = regr_data.time
#
#
#     JSON = {
#         'static': {
#             'max_val': {
#                 'ip#efit': None,
#                 'wmhd#efit': None
#             },
#             'mc_charge_v': None,
#             'had_plasma': True,
#             'had_rfx': True,
#             'had_hnbi1': True,
#             'pulseNo': pulseNo,
#             'datetime': None
#         },
#         'binned': {
#             'time': time,
#             'pulseNo': pulseNo,
#             'ip#efit': {
#                 'data': Ip,
#                 'gradient': Ip_gradient,
#                 'error_lower': Ip_error,
#                 'error_upper': Ip_error},
#
#             'wmhd#efit': {
#                 'data': WMHD,
#                 'gradient': WMHD_gradient,
#                 'error_lower': WMHD_error,
#                 'error_upper': WMHD_error,
#             }}}
#
#     test_JSON = {
#         'static': {
#             'max_val': {
#                 'ip#efit': None,
#                 'wmhd#efit': None
#             },
#             'mc_charge_v': None,
#             'had_plasma': True,
#             'had_rfx': True,
#             'had_hnbi1': True,
#             'pulseNo': pulseNo,
#             'datetime': None
#         },
#         'binned': {
#             'time': time,
#             'pulseNo': pulseNo}}
#
#     # params = ['ip#efit', 'wmhd#efit']
#     param_prop = ['data', 'gradient', 'error_lower', 'error_upper']
#     other = ['help', 'unit', 'const', 'mult']
#     value = [Ip, Ip_gradient, Ip_error, Ip_error]
#     for i in range(0, len(params)):
#         test_JSON['binned'][params[i]] = {}
#
#         for j in range(0, len(param_prop)):
#             test_JSON['binned'][params[i]][param_prop[j]] = value[j]
#     if JSON == test_JSON:
#         print(True)
# lists = ['ip#efit', 'wmhd#efit']
# appender(lists)


# def call_dict():
#     dict_ = info_dict()
#     return dict_


master_jason = {}

base_JSON = {
    'static': {
        'max_val': {
            'ip#efit': None,
            'wmhd#efit': None
        },
        'mc_charge_v': None,
        'had_plasma': True,
        'had_rfx': True,
        'had_hnbi1': True,
        'datetime': None
    }}

base_JSON['static']['pulseNo'] = pulseNo
master_jason.update(base_JSON)

# print(master_jason)


#######################################

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
        'pulseNo': pulseNo,
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
        }}}

# for key in JSON:
#     [print(key)]

JSON = {
    'static0001': {
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
    'binned0001': {
        'time': time,
        'pulseNo': [pulseNo] * len(time),
        'ip#efit': {
            'data': Ip,
            'gradient': Ip_gradient,
            'error_lower': Ip_error,
            'error_upper': Ip_error,
            'unit': '(MA)',
            'const': 1e-3,

        },
        'wmhd#efit': {
            'data': WMHD,
            'gradient': WMHD_gradient,
            'error_lower': WMHD_error,
            'error_upper': WMHD_error,
            'unit': '(kJ)',
            'const': 1e-3,

        },
    'static0002': {
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
    'binned0002': {
        'time': time,
        'pulseNo': [pulseNo] * len(time),
        'ip#efit': {
            'data': Ip,
            'gradient': Ip_gradient,
            'error_lower': Ip_error,
            'error_upper': Ip_error,
            'unit': '(MA)',
            'const': 1e-6,

        },
        'wmhd#efit': {
            'data': WMHD,
            'gradient': WMHD_gradient,
            'error_lower': WMHD_error,
            'error_upper': WMHD_error,
            'unit': '(kJ)',
            'const': 1e-3,

        }}}}
