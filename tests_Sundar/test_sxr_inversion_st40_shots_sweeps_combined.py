import getpass
import itertools

import matplotlib.pyplot as plt
import numpy as np
import os
from xarray import DataArray

from indica.equilibrium import Equilibrium
from indica.operators import InvertRadiation
from indica.readers import ST40Reader
from indica.utilities import coord_array
from indica.converters import bin_to_time_labels
from indica.converters import FluxSurfaceCoordinates

import time as tt
import pickle

import st40_sxr_inversion as ss

plots = True
save_plot = True
save_data = True

#SAVE DIRECTORY
# save_directory = '/home/sundaresan.sridhar/Modules/sxr_inversion/sweep_shots'
save_directory = '/home/sundaresan.sridhar/Modules/sxr_inversion/sweep_Marco'



#ZSHIFTS
z_shifts = np.arange(0,6) * 1.e-2

#ALL DATA
all_data = {}

if 'Marco' not in save_directory:
    
    #PULSE NUMBERS
    data_pulseNos = dict(
        ohmic = [9408,9409,9411],
        NBI = [9539,9560],    
        )
    
    #TIMES
    data_times = dict(
        ohmic = 90,
        NBI   = 70,    
        )
    
    #INTEGRATION TIME
    dt = 5 * 1.e-3
    
    #PULSE NOS AND TIMES
    pulseNos = []
    times = []
    for key in data_pulseNos.keys():
        pulseNos += data_pulseNos[key]
        times    += list(np.tile(data_times[key]*1.e-3,len(data_pulseNos[key])))

    
    #COMBINING THE DATA
    for i_pulse,pulseNo in enumerate(pulseNos):
        #SWEEP OF Z SHIFTS
        for z_shift in z_shifts:
            #OPENING THE TUPLE
            filename =save_directory + '/'+str(pulseNo)+'_zshift_'+str(int(z_shift*100))+'.p'
            key = str(pulseNo) + '_'+str(int(z_shift*100))
            with open(filename, 'rb') as handle:
                all_data[key] = pickle.load(handle)
                
    #COMBINED PLOTS
    for pulseNo in pulseNos:
        #SWEEP OF TIMES
        for i_time,time in enumerate(all_data[str(pulseNo)+'_0']['t'].data):
            #FIGURE DECLARATION
            plt.close('all')
            fig,ax = plt.subplots(nrows=2,ncols=3,squeeze=True,figsize=(16,10),sharex=True,sharey=True)
            #SWEEP OF Z_SHIFTS
            for i,z_shift in enumerate(z_shifts):
                #CHI2 VALUE
                key = str(pulseNo) + '_'+str(int(z_shift*100))
                chi2 = all_data[key]['back_integral']['chi2'][i_time]
                #TITLE
                title = 'z_shift = '+str(int(z_shift*100))+', chi2 = '+str(np.round(chi2,2))
                #SELECTED DATA
                sel_data = all_data[key]['back_integral']
                #I AND J
                if i<3:
                    i1 = 0
                    i2 = i
                elif i<6:
                    i1 = 1
                    i2 = i - 3
                else:
                    i1 = 2
                    i2 = i - 6
                #PLOT
                ax[i1,i2].plot(sel_data['channel_no'][all_data[key]['channels_considered']],sel_data['data_theory'][i_time,all_data[key]['channels_considered']]/1.e+3,color='b')
                ax[i1,i2].scatter(sel_data['channel_no'][all_data[key]['channels_considered']],sel_data['data_experiment'][i_time,all_data[key]['channels_considered']]/1.e+3,color='k')
                ax[i1,i2].scatter(sel_data['channel_no'][np.logical_not(all_data[key]['channels_considered'])],sel_data['data_experiment'][i_time,np.logical_not(all_data[key]['channels_considered'])]/1.e+3,color='r',marker='v')
                #TITLE
                ax[i1,i2].set_title(title)
            #DELETING THE LAST SUBPLOT
            # fig.delaxes(ax[2][3])
            # fig.text(0.5, 0.04, 'Impact parameter [rho poloidal]', ha='center',fontsize=25)
            fig.text(0.5, 0.04, 'Channel number', ha='center',fontsize=25)
            fig.text(0.08, 0.5, 'Line emission [Kw/m-2]', va='center', rotation='vertical',fontsize=25)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False,labelsize=50)
            #SAVING THE PLOT
            filenameFig = save_directory+'/combined/'+str(pulseNo)+'_t_'+str(i_time+1)+'.png'
            plt.savefig(filenameFig)
            plt.close('all')
    
    
    #VARIATION PLOTS
    chi2_data = {}
    for pulseNo in pulseNos:
        #SWEEP OF TIMES
        for i_time,time in enumerate(all_data[str(pulseNo)+'_0']['t'].data):
            #LABEL
            label = str(pulseNo)+' @ '+str(np.round(time*1.e+3,1))+' ms'
            #SWEEP OF Z_SHIFTS
            chi2 = np.nan * np.ones(len(z_shifts))
            for i,z_shift in enumerate(z_shifts):
                #CHI2 VALUE
                key = str(pulseNo) + '_'+str(int(z_shift*100))
                chi2[i] = all_data[key]['back_integral']['chi2'][i_time]
            #CHI2 VALUES
            chi2_data[label] = chi2
    #FIGURE DECLARATION
    plt.close('all')
    plt.figure(figsize=(16,10))
    for label,data in chi2_data.items():
        plt.scatter(z_shifts*100,data)
        plt.plot(z_shifts*100,data,label=label)
    plt.legend(fontsize=15)   
    plt.xlabel('z_shift [cm]',fontsize=25)
    plt.ylabel('chi2 [no unit]',fontsize=25)
    plt.tick_params(axis='both',labelsize=20)
    #SAVING THE PLOT
    filenameFig = save_directory+'/combined/variation_plot.png'
    plt.savefig(filenameFig)
    plt.close('all')

else:
    pulseNos = [9229]
    i_time = 0
    #BEFORE AND AFTER CRASH
    fields = ['before_crash','after_crash']
    for field in fields:
        all_data[field] = {}
        #SWEEP OF PULSES
        for i_pulse,pulseNo in enumerate(pulseNos):
            #SWEEP OF Z SHIFTS
            for z_shift in z_shifts:
                #OPENING THE TUPLE
                filename =save_directory + '/' + field + '/'+str(pulseNo)+'_zshift_'+str(int(z_shift*100))+'.p'
                key = str(pulseNo) + '_'+str(int(z_shift*100))
                with open(filename, 'rb') as handle:
                    all_data[field][key] = pickle.load(handle)
                
    #COMBINED PLOTS
    for pulseNo in pulseNos:
        #SWEEP OF FIELDS
        for field in fields:
            #FIGURE DECLARATION
            plt.close('all')
            fig,ax = plt.subplots(nrows=2,ncols=3,squeeze=True,figsize=(16,10),sharex=True,sharey=True)
            #SWEEP OF Z_SHIFTS
            for i,z_shift in enumerate(z_shifts):
                #CHI2 VALUE
                key = str(pulseNo) + '_'+str(int(z_shift*100))
                chi2 = all_data[field][key]['back_integral']['chi2'][i_time]
                #TITLE
                title = 'z_shift = '+str(int(z_shift*100))+', chi2 = '+str(np.round(chi2,2))
                #SELECTED DATA
                sel_data = all_data[field][key]['back_integral']
                #I AND J
                if i<3:
                    i1 = 0
                    i2 = i
                elif i<6:
                    i1 = 1
                    i2 = i - 3
                else:
                    i1 = 2
                    i2 = i - 6
                #PLOT
                ax[i1,i2].plot(sel_data['channel_no'][all_data[field][key]['channels_considered']],sel_data['data_theory'][i_time,all_data[field][key]['channels_considered']]/1.e+3,color='b')
                ax[i1,i2].scatter(sel_data['channel_no'][all_data[field][key]['channels_considered']],sel_data['data_experiment'][i_time,all_data[field][key]['channels_considered']]/1.e+3,color='k')
                ax[i1,i2].scatter(sel_data['channel_no'][np.logical_not(all_data[field][key]['channels_considered'])],sel_data['data_experiment'][i_time,np.logical_not(all_data[field][key]['channels_considered'])]/1.e+3,color='r',marker='v')
                #TITLE
                ax[i1,i2].set_title(title)
            #DELETING THE LAST SUBPLOT
            # fig.delaxes(ax[2][3])
            # fig.text(0.5, 0.04, 'Impact parameter [rho poloidal]', ha='center',fontsize=25)
            fig.text(0.5, 0.04, 'Channel number', ha='center',fontsize=25)
            fig.text(0.08, 0.5, 'Line emission [Kw/m-2]', va='center', rotation='vertical',fontsize=25)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False,labelsize=50)
            #SAVING THE PLOT
            filenameFig = save_directory+'/combined/'+str(pulseNo)+'_'+field+'.png'
            plt.savefig(filenameFig)
            plt.close('all')
    
    
    #VARIATION PLOTS
    chi2_data = {}
    #SWEEP OF PULSE NUMBERS
    for pulseNo in pulseNos:
        #SWEEP OF FIELDS
        for field in fields:
            #LABEL
            label = str(pulseNo)+' @ '+str(np.round(all_data[field][str(pulseNo)+'_0']['t'].data[i_time]*1.e+3,1))+' ms - '+field
            #SWEEP OF Z_SHIFTS
            chi2 = np.nan * np.ones(len(z_shifts))
            for i,z_shift in enumerate(z_shifts):
                #CHI2 VALUE
                key = str(pulseNo) + '_'+str(int(z_shift*100))
                chi2[i] = all_data[field][key]['back_integral']['chi2'][i_time]
            #CHI2 VALUES
            chi2_data[label] = chi2
    #FIGURE DECLARATION
    plt.close('all')
    plt.figure(figsize=(16,10))
    for label,data in chi2_data.items():
        plt.scatter(z_shifts*100,data)
        plt.plot(z_shifts*100,data,label=label)
    plt.legend(fontsize=15)   
    plt.xlabel('z_shift [cm]',fontsize=25)
    plt.ylabel('chi2 [no unit]',fontsize=25)
    plt.tick_params(axis='both',labelsize=20)
    #SAVING THE PLOT
    filenameFig = save_directory+'/combined/variation_plot.png'
    plt.savefig(filenameFig)
    plt.close('all')