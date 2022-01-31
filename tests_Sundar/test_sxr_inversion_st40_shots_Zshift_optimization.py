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
from xarray import DataArray,concat
import multiprocessing as mp
from scipy import interpolate
import time as tt
import pickle

import st40_sxr_inversion as ss
import st40_sxr_inversion_plots as ss_plot

invert = True
plots = True
save_plot = True
save_data = True
version_control = False
version = ''

pulseInfo = {
    '9408' : [15,135],
    '9560' : [15,120],
    '9229' : [15,120],
    '9409' : [15,150],
    '9411' : [15,150],
    '9539' : [15,150],
    '9537' : [15,150],    
    '9538' : [15,140],    
    '9184' : [15,150],
    }

sweepInfo = {
    '9408' : [-5,5],
    '9560' : [-0.5,5],
    '9229' : [-0.5,5],
    '9409' : [-1.5,5],
    '9411' : [-1.5,5],
    '9539' : [-0.5,5],
    '9537' : [-0.5,5],    
    '9184' : [-1.5,1.5],
    '9538' : [-0.5,5],    
    }

#SAVE DIRECTORY
save_directory_base = '/home/sundaresan.sridhar/Modules/sxr_inversion/shots_optimized_tomo_1D'

#FUNCTION TO GET SAVE DIRECTORY
def get_save_directory(folder,save_directory_base):
    os.chdir(save_directory_base)
    try:
        os.mkdir(folder)
    except:
        pass
    os.chdir(folder)
    return os.getcwd()

#DEFAULT INPUT DATA
input_data_default = dict(
    d_time = 3*1.e-3,
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
    exclude_bad_points = True,
    optimize_z_shift = True,    
    method = 'tomo_1D',
    EFIT_run = 1,
    )

# #Z_SHIFT VALUES
# z_range = [-2,2]
# z_shifts = np.linspace(z_range[0],z_range[1],41)

if __name__ == "__main__":

    #INPUT DATA
    input_data = input_data_default.copy()
    
    #SWEEP OF SHOT NUMBERS
    for pulseNo,time in pulseInfo.items():
        
        #DEBUG
        if input_data['debug']:
            print('SXR inversion module started for #'+str(pulseNo))
        
        #Z_SHIFT VALUES
        z_range = sweepInfo[str(pulseNo)]
        dz = 0.1 #cm
        no_z = int((z_range[1]-z_range[0])/dz)+1
        z_shifts = np.linspace(z_range[0],z_range[1],no_z)
        
        #TIME
        time = list(np.array(time)*1.e-3)
        
        #PULSENO
        pulseNo = int(pulseNo)
                 
        #SAVE DIRECTORY
        save_directory_shot = get_save_directory(str(pulseNo),save_directory_base)
        
        #FILENAME SUFFIX
        filenameSuffix = str(pulseNo)
        
        #EXCEPTIONS
        exceptions = []
              
        #CHANGING INPUT DATA
        input_data['z_shift'] = z_shifts * 1.e-2
             
        #MAKING SWEEP PLOTS
        save_directory_sweep = get_save_directory('sweep_plots',save_directory_shot)

        #PERFORMING SXR INVERSION               
        if invert:
            try:                
                #PERFORMING SXR INVERSION
                results = ss.make_SXR_inversion(pulseNo,time,input_data)            
                for iz,z_shift in enumerate(z_shifts):
                    #KEY
                    key = str(pulseNo) + '_z_shift_' + str(int(z_shift*10))+'_mm'
                    #SAVE DIRECTORY
                    save_directory = get_save_directory(key,save_directory_sweep)
                    #SWEEP PLOT
                    if plots:
                        plt.close('all')
                        sel_results = results['all_results']['sweep_value_'+str(iz+1)]
                        sel_results['input_data']['z_shift'] = z_shift * 1.e-2
                        ss_plot.make_SXR_inversion_plots(sel_results,saveFig=save_plot,save_directory=save_directory)
                    #SAVING THE DATA
                    if save_data:
                        filename = save_directory+'/'+key+'.p'
                        sel_results = results['all_results']['sweep_value_'+str(iz+1)]
                        with open(filename,'wb') as handle:
                            pickle.dump(sel_results,handle,protocol=pickle.HIGHEST_PROTOCOL)
                            print(filename+' saved successfully!')                
                    #OPTIMIZE PLOT
                    save_directory = get_save_directory('optimize_plots',save_directory_shot)
                    if plots:
                        plt.close('all')
                        ss_plot.make_SXR_inversion_plots(results,saveFig=save_plot,save_directory=save_directory,optimize_plot=True)
                    #SAVING THE DATA
                    if save_data:
                        filename = save_directory+'/optimized_'+str(pulseNo)+'.p'
                        with open(filename,'wb') as handle:
                            pickle.dump(results,handle,protocol=pickle.HIGHEST_PROTOCOL)
                            print(filename+' saved successfully!')    
                    #PLOTTING AND SAVING Z_SHIFTS
                    save_directory = get_save_directory('plots_z_shifts',save_directory_base)
                    ss_plot.plot_z_shifts(results,save_directory=save_directory,saveFig=True)
            except Exception as e:
                print(e)
                exceptions += [pulseNo]
                sweep_data = {}
        else:
            # pass
            # print('SXR inversion turned off for #'+str(pulseNo))
            #LOADING THE RESULTS
            save_directory = get_save_directory('optimize_plots',save_directory_shot)
            filename = save_directory+'/optimized_'+str(pulseNo)+'.p'
            with open(filename,'rb') as handle:
                results = pickle.load(handle)
                print(filename+' loaded successfully!') 
            #PLOTTING AND SAVING Z_SHIFTS
            save_directory = get_save_directory('plots_z_shifts',save_directory_base)
            ss_plot.plot_z_shifts(results,save_directory=save_directory,saveFig=True)
