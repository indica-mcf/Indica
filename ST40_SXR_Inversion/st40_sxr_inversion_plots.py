#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:31:42 2021

@author: sundaresan.sridhar
"""
#MODULE TO PLOT
import matplotlib.pyplot as plt
import numpy as np
import ST40_SXR_Inversion.st40_sxr_inversion as ss
from MDSplus import *
from scipy import interpolate

#MDSPLUS IP ADDRESS
MDSplus_IP_address = '192.168.1.7:8000'

#CLOSING ALL THE PLOTS
plt.close('all') 

#PLOT ATTRIBUTES
ticksize    =   15
labelsize   =   20

save_directory = ''
saveFig = True

#FUNCTION TO PLOT SXR INVERSION
def make_SXR_inversion_plots(return_data,saveFig=False,save_directory=''):
    #RETURN FIGURE DATA
    return_Fig_data = {}
    #SWEEP OF CAMERAS
    for camera in return_data['input_data']['cameras']:
        #CAMERA DATA
        data_camera = return_data[camera]
        #SWEEP OF TIMES
        for i,time in enumerate(data_camera['t']):
            
            #FIGURE DECLARATION
            fig,ax = plt.subplots(nrows=2,ncols=3,squeeze=True,figsize=(16,10))
            gs = ax[0, 2].get_gridspec()
                  
            #BASE TITLE - pulseNo & time
            baseTitle  = '#'+str(return_data['pulseNo']) #pulseNo
            baseTitle += '-'+camera #NAME OF THE CAMERA
            baseTitle += ' @ t = '+str(np.round((time-(return_data['input_data']['d_time'])/2)*1.e+3,1))
            baseTitle += ' - '+str(np.round((time+(return_data['input_data']['d_time'])/2)*1.e+3,1))+' s, '
            #ANGLE
            baseTitle += 'angle = '+str(int(return_data['input_data']['angle']))+' [degrees], '
            #SHIFTS
            shifts = ['R_shift','z_shift']
            for shift in shifts:
                baseTitle += shift + ' = ' + str(int(return_data['input_data'][shift]*1.e+2))+' cm, '
            #CHI2
            baseTitle += 'chi2 = '+str(np.round(data_camera['back_integral']['chi2'][i],2))
            
            #TITLE OF THE PLOT
            fig.suptitle(baseTitle)
        
            #SUBPLOT 1 - EMISSIVITY PROFILE
            axx = ax[0,0]
            zdata = data_camera['emissivity_2D']['data'][:,:,i]/1.e+3
            z_min = np.nanmin(zdata)
            z_max = np.nanmax(zdata)
            heatmap = axx.pcolormesh(data_camera['emissivity_2D']['R'], data_camera['emissivity_2D']['z'], zdata, cmap='RdBu', vmin=z_min, vmax=z_max)
            axx.set_xlabel("R (m)")
            axx.set_ylabel("z (m)")
            axx.set_xlim(0, 1)
            axx.set_ylim(-0.5, 0.5)
            fig.colorbar(heatmap, ax=axx)
    
            #SUBPLOT 2 - BACKINTEGRAL
            axx = ax[0,1]
            sel_channels = return_data[camera]['channels_considered']
            data_exp = return_data[camera]['back_integral']['data_experiment'][i,:]
            data_the = return_data[camera]['back_integral']['data_theory'    ][i,:]
            x_data   = return_data[camera]['back_integral']['channel_no'     ]
            axx.scatter(x_data[sel_channels],data_exp[sel_channels]/1.e+3,color='k')
            axx.scatter(x_data[np.logical_not(sel_channels)],data_exp[np.logical_not(sel_channels)]/1.e+3,color='r',marker='v')
            axx.plot(x_data[sel_channels],data_the[sel_channels]/1.e+3,color='b')        
            axx.set_xlabel('Channel Number')
            axx.set_ylabel('back integral [kW/m-2]')
    
            #SUBPLOT 3 - SYMMETRIC EMISSIVITY
            axx = ax[1,0]
            x_data  = return_data[camera]['profile']['rho_poloidal'][i,:]
            pro_sym = return_data[camera]['profile']['sym_emissivity'][i,:]
            axx.plot(x_data,pro_sym/1.e+3)        
            axx.set_xlabel('rho poloidal')
            axx.set_ylabel('symmetric emissivity [kW/m3]')        
        
            #SUBPLOT 4 - ASYMMETRIC EMISSIVITY
            axx = ax[1,1]
            x_data  = return_data[camera]['profile']['rho_poloidal'][i,:]
            par_asy = return_data[camera]['profile']['asym_parameter'][i,:]
            axx.plot(x_data,par_asy)        
            axx.set_xlabel('rho poloidal')
            axx.set_ylabel('asymmetric parameter')   
            axx.set_ylim(-0.1,1)
        
            #SUBPLOT 5 - DIRECTION
            ax[0,2].remove()
            ax[1,2].remove()                
            axbig = fig.add_subplot(gs[0:, -1])
            #HEATMAP PLOT
            heatmap = axbig.pcolormesh(data_camera['emissivity_2D']['R'], data_camera['emissivity_2D']['z'], zdata, cmap='RdBu', vmin=z_min, vmax=z_max)
            #PROJECTION PLOT
            R_data = return_data[camera]['projection']['R']
            z_data = return_data[camera]['projection']['z']
            for j,ch_considered in enumerate(sel_channels):
                if ch_considered:
                    axbig.plot(R_data[j,:], z_data[j,:], color='k')
                else:
                    axbig.plot(R_data[j,:], z_data[j,:], color='r',linestyle=':')
            axbig.set_xlabel("R (m)")
            axbig.set_ylabel("z (m)")
            axbig.set_xlim(0, 1)
            axbig.set_ylim(-0.5, 0.5)
            axbig.vlines(0.17, -0.5, 0.5, label="Inner column", color="black")
            
            #FILENAME
            fileName  = str(return_data['pulseNo']) #pulseNo
            fileName += '_'+camera #CAMERA NAME
            fileName += '_t_'+str(i) #time
            fileName += '_angle_'+str(int(return_data['input_data']['angle'])) #angle
            shifts = ['R_shift','z_shift']
            for shift in shifts:
                fileName += '_'+ shift + '_' + str(int(return_data['input_data'][shift]*1.e+2))
            
            #FIGURE DATA
            return_Fig_data[fileName] = fig
            
            #SAVING THE PLOT
            if saveFig:
                #SAVE DIRECTORY
                ss.make_directory(save_directory)
                fileName = save_directory + '/' + fileName + '.png'
                #SAVING THE PLOT
                plt.savefig(fileName)
                print(fileName+' is saved')
                plt.close()
    
    #RETURNING THE FIGURES
    return return_Fig_data

#FUNCTION TO PLOT Z-SHIFT
def plot_z_shifts(results,save_directory='',saveFig=True,results2={},methods=[]):
    #ALL RESULTS
    if results2!={}:
        results_all = (results,results2,)
    else:
        results_all = (results,)
    #SUBPLOT DECLARATION
    plt.close('all')    
    #COLORS OF Z_SHIFTS
    colors_zshift = ['red','blue']
    #SWEEP OF RESULTS
    k = 0
    data_zshifts = {}
    for iresults,results in enumerate(results_all):
        #METHOD
        if len(results_all)==1:
            method = 'set1'
        else:
            method = methods[iresults]
        #Z_SHIFT OPTIMIZATION PLOT
        t = results['filter_4']['t']*1.e+3
        chi2_optimum = results['filter_4']['back_integral']['chi2']
        chi2_all     = results['results_optimize']['chi2_evolution']
        chi2_0       = results['all_results']['sweep_value_'+str(np.where(results['results_optimize']['z_shifts']==0)[0][0]+1)]['filter_4']['back_integral']['chi2']
        z_shifts     = results['filter_4']['z_shift']      
        data_zshifts[method] = dict(
            time = t,
            data = z_shifts,
            color = colors_zshift[iresults],
            method = method,
            )
        #SUBPLOTS DECLARATION
        if iresults==0:
            if len(results_all)==1:
                nrows = 3
            else:
                nrows = 4
            fig,ax = plt.subplots(nrows=nrows,ncols=1,squeeze=True,figsize=(16,10),sharex=True)            
        #SUBPLOT 1 - ALL CHI2 OF THE SWEEP
        #ALL CHI2
        for i in range(0,np.size(chi2_all,1)):
            ax[k].plot(t,chi2_all[:,i],color='orange')
        #CHI2 AT Z_SHIFT=0
        ax[k].plot(t,chi2_0,color='blue',label='chi2 at z_shift=0cm')
        #OPTIMAL CHI2
        ax[k].plot(t,chi2_optimum,color='k',label='chi2 at optimum z_shifts')
        #LEGEND
        ax[k].legend(ncol=3,fontsize=ticksize)
        #YLABEL
        if len(results_all)==1:
            ax[k].set_ylabel('chi2',size=labelsize)
        else:
            ax[k].set_ylabel('chi2('+method+')',size=labelsize)
        #TICK SIZE
        ax[k].tick_params(axis='both', labelsize=ticksize)
        #TITLE
        if iresults==0:
            fig.suptitle('#'+str(results['pulseNo'])+' - z_shift optimization',fontsize=labelsize*1.5)
        #INCREMENT
        k += 1
    
    #SUBPLOT - OPTIMUM Z_SHIFT
    for key,value in data_zshifts.items():
        t = value['time']
        z_shifts = value['data']
        if len(results_all)==1:
            ax[k].plot(t,z_shifts*100,color=value['color'],label='z_shift (optimum)')
        else:
            ax[k].plot(t,z_shifts*100,color=value['color'],label='z_shift (optimum,'+value['method']+')')
    #TICK SIZE
    ax[k].tick_params(axis='both', labelsize=ticksize)
    #YLABEL
    ax[k].set_ylabel('z_shift[cm]',size=labelsize)
    #LEGEND
    if len(results_all)>1:
        ax[k].legend(ncol=2,fontsize=ticksize)
    #k INCREMENT
    k += 1
    
    #LOADING ZMAG FROM EFIT
    results = results_all[0]
    whichRun = 'BEST' if (results['input_data']['EFIT_run']==0) else 'RUN'+str(results['input_data']['EFIT_run']).zfill(2)
    pulseNo = results['pulseNo']
    conn = Connection(MDSplus_IP_address)
    conn.openTree('EFIT',pulseNo)
    node = whichRun+'.GLOBAL:ZMAG'
    r_zmag         = conn.get(node).data()
    r_t_zmag       = conn.get('dim_of('+node+')').data()    
    #SUBPLOT - OPTIMUM Z_SHIFT
    label = 'Zmag - z_shift'
    for iresults,sel_data in enumerate(data_zshifts.values()):
        #LABEL
        if len(results_all)>1:
            label_plot = label + '('+sel_data['method']+')'
        else:
            label_plot = label
        #DATA
        t = sel_data['time']
        data = sel_data['data']
        col  = sel_data['color']        
        #ZMAG FROM EFIT
        zmag = interpolate.interp1d(r_t_zmag,r_zmag,bounds_error=False)(t*1.e-3)
        #ZMAG PLOT
        if iresults==0:
            ax[k].plot(t,zmag*100,color='black',label='Zmag (EFIT)')
        #ZMAG - ZSHIFT PLOT
        ax[k].plot(t,(zmag - data)*100,color=col,label=label_plot)
    #LEGEND
    ax[k].legend(ncol=1,fontsize=ticksize)
    #TICK SIZE
    ax[k].tick_params(axis='both', labelsize=ticksize)
    #YLABEL
    ax[k].set_ylabel('Zmag[cm]',size=labelsize)
    #XLABEL
    ax[k].set_xlabel('Time [ms]',size=labelsize)
    #FILENAME
    fileName = str(results['pulseNo'])+'_z_shift_optimization'    
    if len(results_all)>1:
        fileName += '_comparison'
    #SUBPLOTS ADJUST
    plt.subplots_adjust(top=0.9,wspace=0.05, hspace=0.1)      
    #SAVING THE PLOT
    if saveFig:
        #SAVE DIRECTORY
        ss.make_directory(save_directory)
        fileName = save_directory + '/' + fileName + '.png'
        #SAVING THE PLOT
        plt.savefig(fileName)
        print(fileName+' is saved')
        plt.close()
    #RETURNING THE FIGURE DATA
    return fig

#FUNCTION TO PLOT CHI2 EVOLUTION
def plot_chi2(results,save_directory='',saveFig=True):
    #SUBPLOT DECLARATION
    plt.close('all')    
    #CHI2 DATA
    t_chi2      =   results['filter_4']['t'] * 1.e+3
    chi2        =   results['filter_4']['back_integral']['chi2']
    #EFIT DATA
    whichRun = 'BEST' if (results['input_data']['EFIT_run']==0) else 'RUN'+str(results['input_data']['EFIT_run']).zfill(2)
    pulseNo = results['pulseNo']
    conn = Connection(MDSplus_IP_address)
    conn.openTree('EFIT',pulseNo)
    node = whichRun+'.GLOBAL:ZMAG'
    zmag         = conn.get(node).data()
    t_zmag       = conn.get('dim_of('+node+')').data()*1.e+3
    sel_map      = (t_zmag>=np.nanmin(t_chi2))&(t_zmag<=np.nanmax(t_chi2))
    t_zmag       = t_zmag[sel_map]
    zmag         = zmag[sel_map]
    
    #SUBPLOT DECLARATION
    fig,ax = plt.subplots(nrows=2,ncols=1,squeeze=True,figsize=(16,10),sharex=True)            
    #SUBPLOT 1 - CHI2 DATA
    ax[0].plot(t_chi2,chi2,color='blue')
    ax[0].set_ylabel('chi2',size=labelsize)
    #SUBPLOT 2 - ZMAG DATA
    ax[1].plot(t_zmag,zmag*1.e+2,color='red',label='Zmag from EFIT#'+whichRun)
    ax[1].set_ylabel('Zmag [cm]',size=labelsize)
    #SWEEP OF SUBPLOTS
    for k in range(0,2):
        #TICK SIZE
        ax[k].tick_params(axis='both', labelsize=ticksize)
        #LEGEND
        ax[k].legend(ncol=1,fontsize=ticksize)
        #XLABEL
        ax[k].set_xlabel('Time [ms]',size=labelsize)    
    #TITLE
    fig.suptitle('#'+str(results['pulseNo'])+' - chi2 evolution',fontsize=labelsize*1.5)
    #SUBPLOTS ADJUST
    plt.subplots_adjust(top=0.9,wspace=0.05, hspace=0.1)      
    #SAVING THE PLOT
    if saveFig:
        #SAVE DIRECTORY
        ss.make_directory(save_directory)
        #FILENAME
        fileName = str(results['pulseNo'])+'_chi2_evolution.png'
        fileName = save_directory + '/' + fileName + '.png'
        #SAVING THE PLOT
        plt.savefig(fileName)
        print(fileName+' is saved')
        plt.close()
    #RETURNING THE FIGURE DATA
    return fig