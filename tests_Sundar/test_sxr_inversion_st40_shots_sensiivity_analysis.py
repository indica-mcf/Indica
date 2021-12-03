import getpass
import itertools

import matplotlib.pyplot as plt
import numpy as np
import os
from xarray import DataArray

from scipy import interpolate

# from indica.equilibrium import Equilibrium
# from indica.operators import InvertRadiation
# from indica.readers import ST40Reader
# from indica.utilities import coord_array
# from indica.converters import bin_to_time_labels
# from indica.converters import FluxSurfaceCoordinates

import time as tt
import pickle

import st40_sxr_inversion as ss

import multiprocessing as mp

plots = True
save_plot = True
save_data = True

pulseNo = 9184
t1 = 0.05
t2 = 0.07
time = [t1,t2]

#SAVE DIRECTORY
save_directory_base = '/home/sundaresan.sridhar/Modules/sxr_inversion/sensitivity_analysis'

#FUNCTION TO GET SAVE DIRECTORY
def get_save_directory(folder = ''):
    os.chdir(save_directory_base)
    try:
        os.mkdir(folder)
    except:
        pass
    os.chdir(folder)
    return os.getcwd()

#DEFAULT INPUT DATA
input_data_default = dict(
    d_time = 5*1.e-3,
    angle=0,
    R_shift=0,
    z_shift=0,
    fit_asymmetry=False,
    compute_asymmetry=False,
    debug=True,
    plots=True,
    save_directory='',
    filenameSuffix='',
    )

#SWEEP VALUES
sweep_values = dict(
    angle = np.arange(-5,6),
    R_shift = np.arange(0,6)*1.e-2,
    z_shift = np.arange(-5,6)*1.e-2,
    d_time = np.array([3,5,10])*1.e-3,
    exclude_bad_points = [True,False],
    )

if __name__ == "__main__":
    #FIELDS
    # fields = list(sweep_values.keys())
    fields = ['R_shift','z_shift','d_time']
    #SWEEP OF FIELDS
    for field in fields:
        #SWEEP DATA
        sweep_value = sweep_values[field]
        #SWEEP OF VALUES
        for value in sweep_value:
            #INPUT DATA
            input_data = input_data_default.copy()
            #SAVE DIRECTORY
            input_data['save_directory'] = get_save_directory('sweep_'+field)
            if 'exclude' in field:
                input_data['filenameSuffix'] = field+'_'+str(value)
            else:
                if '_shift' in field:
                    SuffixValue = value * 1.e+2
                elif '_time' in field:
                    SuffixValue = value * 1.e+3
                else:
                    SuffixValue = value
                input_data['filenameSuffix'] = field+'_'+str(int(SuffixValue))
            input_data[field] = value
            ss.make_SXR_inversion(pulseNo,time,input_data)