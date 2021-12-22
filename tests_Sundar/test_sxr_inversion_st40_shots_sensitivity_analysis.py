import getpass
import itertools

import matplotlib.pyplot as plt
import numpy as np
import os
from xarray import DataArray

from scipy import interpolate

import time as tt
import pickle

import st40_sxr_inversion as ss
import SXR_inversion_plot as ss_plot

import multiprocessing as mp

plots = True
save_plot = True
save_data = True
version_control = False
version = 'v1_zshift_-1'

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
    z_shift=0 * 1.e-2,
    cameras = ['filter_4'],
    n_intervals = 65,
    datatype = 'sxr',
    knots = 6,
    runParallel = False,
    fit_asymmetry=False,
    compute_asymmetry=False,
    debug=True,
    )

#SWEEP VALUES
sweep_values = dict(
    angle = np.arange(-5,6),
    R_shift = np.arange(0,6)*1.e-2,
    z_shift = np.arange(-5,6)*1.e-2,
    d_time = np.array([3,5,10])*1.e-3,
    exclude_bad_points = [True,False],
    )


#FIELDS
fields = list(sweep_values.keys())
# fields = ['angle']

if __name__ == "__main__":
    #SWEEP OF FIELDS
    for field in fields:
        #SWEEP DATA
        sweep_value = sweep_values[field]
        #SAVE DIRECTORY
        if version_control:
            save_directory = get_save_directory('sweep_'+field+'_'+version)
        else:
            save_directory = get_save_directory('sweep_'+field)

        #SWEEP DATA DECLARATION
        sweep_data = {}
        #SWEEP OF VALUES
        for value in sweep_value:
            #INPUT DATA
            input_data = input_data_default.copy()
            if 'exclude' in field:
                filenameSuffix = str(pulseNo) + '_'+ field + '_' + str(value)
            else:
                if '_shift' in field:
                    SuffixValue = value * 1.e+2
                elif '_time' in field:
                    SuffixValue = value * 1.e+3
                else:
                    SuffixValue = value
                filenameSuffix = str(pulseNo) + '_'+ field + '_' + str(int(SuffixValue))
            input_data[field] = value
            sweep_data[filenameSuffix] = ss.make_SXR_inversion(pulseNo,time,input_data)
            #PLOT
            if plots:
                ss_plot.make_SXR_inversion_plots(sweep_data[filenameSuffix],saveFig=save_plot,save_directory=save_directory)
        #SAVING THE DATA
        if save_data:
            filename = save_directory+'/'+str(pulseNo)+'_sweep_'+field+'.p'
            with open(filename,'wb') as handle:
                pickle.dump(sweep_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
                print(filename+' saved successfully!')