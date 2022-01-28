import sys
try:
    sys.path.remove('/home/marco.sertoli/python/Indica')
except:
    pass
try:
    sys.path.append('/home/sundaresan.sridhar/Modules/st40/Indica')
except:
    pass

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
import st40_sxr_inversion_plots as ss_plot

import multiprocessing as mp

plots = True
save_plot = True
save_data = True
version_control = True
version = ''

pulseInfo = {
    '9408' : [15,135],
    # '9560' : [15,120],
    # '9229' : [15,120],
    # '9409' : [15,150],
    # '9411' : [15,150],
    # '9539' : [15,150],
    # '9537' : [15,150],    
    # '9184' : [15,150],
    # '9538' : [15,140],    
    }


#SAVE DIRECTORY
save_directory_base = '/home/sundaresan.sridhar/Modules/sxr_inversion/shots_tomo_1D'

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
    d_time = 2*1.e-3,
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
    EFIT_run = 1,
    method = 'tomo_1D',
    # method = 'indica',
    )

exception_pulses = []

if __name__ == "__main__":

    #INPUT DATA
    input_data = input_data_default.copy()
    
    #SWEEP OF SHOT NUMBERS
    for pulseNo,time in pulseInfo.items():
        
        #TIME
        time = list(np.array(time)*1.e-3)
        
        #PULSENO
        pulseNo = int(pulseNo)
                 
        #SAVE DIRECTORY
        save_directory = get_save_directory(str(pulseNo))
        
        #FILENAME SUFFIX
        filenameSuffix = str(pulseNo)
                
        try:
            #PERFORMING SXR INVERSION
            shot_data = ss.make_SXR_inversion(pulseNo,time,input_data)
            
            #PLOT
            if plots:
                ss_plot.make_SXR_inversion_plots(shot_data,saveFig=save_plot,save_directory=save_directory)
            
            #SAVING THE DATA
            if save_data:
                filename = save_directory+'/'+str(pulseNo)+'.p'
                with open(filename,'wb') as handle:
                    pickle.dump(shot_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
                    print(filename+' saved successfully!')
            
        except Exception as e:
            print(e)
            exception_pulses += [pulseNo]
            