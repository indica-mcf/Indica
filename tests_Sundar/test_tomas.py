#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:53:18 2022

@author: sundaresan.sridhar
"""
import pickle
from tomo_1D import *
import matplotlib.pyplot as plt
plt.close('all')

base_directory = '/home/sundaresan.sridhar/Modules/sxr_inversion/shots_tomo_1D/'

pulseNo = 9229

#FIELDS
fields = ['Indica','tomo_1D']

#INVERSION DATA
data_inv = {}    
#SWEEP OF FIELDS
for field in fields:
    #FILENAME
    filename = base_directory + str(pulseNo)+'_'+field+'/'+str(pulseNo)+'.p'
    #LOADING THE PICKEL FILE
    with open(filename,'rb') as handle:
        shot_data = pickle.load(handle)
    data_inv[field] = shot_data['filter_4']

#PLOT
colors = ['red','blue']
for it in range(0,len(data_inv[fields[0]]['t'])):
    plt.figure(figsize=(16,10))
    for ifield,field in enumerate(fields):
        #CHANNELS CONSIDERED
        valid = data_inv[field]['channels_considered']
        #PLOT DATA
        x_data  = data_inv[field]['back_integral']['channel_no']
        y_data1 = data_inv[field]['back_integral']['data_experiment'][it,:]
        y_data2 = data_inv[field]['back_integral']['data_theory'][it,:]
        chi2 = data_inv[field]['back_integral']['chi2'][it]
        #PLOT
        plt.scatter(x_data,y_data1,color=colors[ifield])
        plt.plot(x_data,y_data2,color=colors[ifield],label=field+'-chi2='+str(np.round(chi2,2)))
    #LEGEND
    plt.legend()
    #SAVING THE FIGURE
    filename = base_directory+str(pulseNo)+'_comparison/'+str(pulseNo)+'_t_'+str(it)+'.png'
    plt.savefig(filename)
    plt.close()
    
