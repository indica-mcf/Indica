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

import ST40_SXR_Inversion.st40_sxr_inversion as ss
import ST40_SXR_Inversion.st40_sxr_inversion_plots as ss_plot
import multiprocessing as mp

#9561,9562

#TIME INFORMATION OF THE PULSE
pulseTimeInfo = {
    '9408' : [15,135],
    '9560' : [15,120],
    '9229' : [15,120],
    '9409' : [15,150],
    '9411' : [15,150],
    '9539' : [15,150],
    '9537' : [15,150],    
    '9184' : [15,150],
    '9538' : [15,140],
    '9780' : [15,150],
    '9783' : [15,150],    
    '9619' : [15,70],
    '9622' : [15,70],
    '9623' : [15,70],
    '9624' : [15,70],
    '9626' : [15,70],
    }

#ZSHIFT SWEEP INFORMATION
zshiftSweepInfo = {
    '9408' : [-5,5],
    '9409' : [-1.5,5],
    '9560' : [-0.5,5],
    '9229' : [-0.5,5],
    '9411' : [-1.5,5],
    '9539' : [-0.5,5],
    '9537' : [-0.5,5],    
    '9184' : [-1.5,1.5],
    '9538' : [-0.5,5],    
    '9780' : [-1.5,5],
    '9783' : [-1.5,5],
    }

#DEFAULT DICTIONARY
defaultDict = dict(
    plot_all    = True,
    plot_opt    = True,
    plot_chi2   = True,
    save_plot   = True,
    save_data   = True,
    invert      = True,
    method      = 'tomo_1D',
    EFIT_run    = 1,
    SXR_run     = 1,
    optimize    = False,    
    )

#FUNCTION TO RUN SXR INVERSION
def run_SXR_inversion(pulseNos,dataDict={}):

    #EXCEPTION PULSES
    exception_pulses = []
    
    #PULSENOS
    if pulseNos is None:
        pulseNos = [int(x) for x in pulseTimeInfo.keys()]
    
    #DATA DICT
    for key,value in defaultDict.items():
        dataDict[key] = dataDict[key] if (key in dataDict) else value
    if dataDict['optimize']:
        dataDict['optimize_z_shift']    = True
    
    #BASE DIRECTORY
    opt_folder      = '_optimized' if dataDict['optimize'] else ''
    SXR_whichRun    = 'BEST' if dataDict['SXR_run']==0 else 'RUN'+str(dataDict['SXR_run']).zfill(2)
    EFIT_whichRun   = 'BEST' if dataDict['EFIT_run']==0 else 'RUN'+str(dataDict['EFIT_run']).zfill(2)
    if 'base_directory' not in dataDict:
        baseDirectory   =   'shots'+opt_folder+'_'+dataDict['method']+'_SXR_'+SXR_whichRun+'_EFIT_'+EFIT_whichRun
    else:
        baseDirectory   =   dataDict['base_directory']
    baseDirectory = '/home/sundaresan.sridhar/Modules/sxr_inversion/'+baseDirectory
    
    #SWEEP OF PULSE NUMBERS
    for ipulse,pulseNo in enumerate(pulseNos):
        
        #PRINTING THE STATUS]
        print(str(ipulse+1)+'/'+str(len(pulseNos))+'-#'+str(pulseNo))
        
        #Z SHIFTS
        if (dataDict['optimize'])&('z_shift' not in dataDict):
            z_range                 = [x for x in zshiftSweepInfo[str(pulseNo)]]
            dz                      = 0.1  if ('dz' not in dataDict) else dataDict['dz']
            no_z                    = int((z_range[1]-z_range[0])/dz)+1
            dataDict['z_shift']     = np.linspace(z_range[0],z_range[1],no_z)*1.e-2
        
        #TIME RANGE
        try:
            time = [x*1.e-3 for x in pulseTimeInfo[str(pulseNo)]]
            proceed = True
        except:
            print('Error in getting time interval for pulse #'+str(pulseNo))
        
        #PULSE DIRECTORY
        pulseDirectory = baseDirectory + '/' + str(pulseNo)
                        
        try:
        
            #PERFORMING SXR INVERSION
            if dataDict['invert']:
                shot_data = ss.make_SXR_inversion(pulseNo,time,dataDict)
                #SAVING THE DATA
                if dataDict['save_data']:
                    ss.make_directory(pulseDirectory)
                    filename = pulseDirectory+'/'+str(pulseNo)+'.p'
                    with open(filename,'wb') as handle:
                        pickle.dump(shot_data,handle,protocol=pickle.HIGHEST_PROTOCOL)
                        print(filename+' saved successfully!')
            else:
                handle = open(pulseDirectory+'/'+str(pulseNo)+'.p','rb')
                shot_data = pickle.load(handle)
        
            
            #NON OPTIMIZED PLOT
            if not dataDict['optimize']:
                #ALL PLOTS
                if dataDict['plot_all']:
                    #ALL FIT PLOTS
                    ss_plot.make_SXR_inversion_plots(shot_data,saveFig=dataDict['save_plot'],save_directory=pulseDirectory+'/'+'fit_plots')
            else: #OPTIMIZED PLOT                
                #ALL PLOTS
                if dataDict['plot_all']:
                    for iz,z_shift in enumerate(dataDict['z_shift']):
                        #SELECTED RESULTS
                        sel_results = shot_data['all_results']['sweep_value_'+str(iz+1)]
                        sel_results['input_data']['z_shift'] = z_shift
                        #ALL FIT PLOTS
                        save_directory = pulseDirectory+'/sweep_fit_plots/sweep_value_'+str(iz+1)
                        # ss_plot.make_SXR_inversion_plots(sel_results,saveFig=dataDict['save_plot'],save_directory=save_directory)
                #OPTIMUM PLOT
                if dataDict['plot_opt']:
                    #OPTIMUM FIT PLOTS
                    save_directory = pulseDirectory+'/optimum_fit_plots/'
                    ss_plot.make_SXR_inversion_plots(shot_data,saveFig=dataDict['save_plot'],save_directory=save_directory)
                #SAVING THE SWEEP DATA
                if dataDict['save_data']:
                    for iz,z_shift in enumerate(dataDict['z_shift']):
                        #SELECTED RESULTS
                        sel_results = shot_data['all_results']['sweep_value_'+str(iz+1)]
                        sel_results['input_data']['z_shift'] = z_shift
                        #ALL FIT PLOTS
                        save_directory = pulseDirectory+'/sweep_data/'
                        ss.make_directory(save_directory)
                        filename = save_directory+str(pulseNo)+'_sweep_'+str(iz+1)+'.p'
                        with open(filename,'wb') as handle:
                            pickle.dump(sel_results,handle,protocol=pickle.HIGHEST_PROTOCOL)
                            print(filename+' saved successfully!')
            
            #CHI2 EVOLUTION PLOT
            if dataDict['plot_chi2']:
                #SAVE DIRECTORY
                save_directory = baseDirectory + '/plots_chi2'
                #OPTIMIZATION PLOT
                if dataDict['optimize']:
                    #OPTIMIATION PLOT
                    ss_plot.plot_z_shifts(shot_data,save_directory=save_directory,saveFig=dataDict['save_plot'],results2={},methods=[])
                #NON-OPTIMIZATION PLOT
                else:
                    ss_plot.plot_chi2(shot_data,save_directory=save_directory,saveFig=dataDict['save_plot'])
        
        except Exception as e:
            print(e)
            exception_pulses += [pulseNo]   
            
