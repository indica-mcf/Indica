#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:31:42 2021

@author: sundaresan.sridhar
"""
#MODULE TO PLOT
import matplotlib.pyplot as plt
import numpy as np

#CLOSING ALL THE PLOTS
plt.close('all') 

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
            x_data  = return_data[camera]['profile']['sym_emissivity'].coords['rho_poloidal'].data
            pro_sym = return_data[camera]['profile']['sym_emissivity'].data[i,:]
            par_asy = return_data[camera]['profile']['asym_parameter'].data[i,:]
            axx.plot(x_data,pro_sym/1.e+3)        
            axx.set_xlabel('rho poloidal')
            axx.set_ylabel('symmetric emissivity [kW/m3]')        
        
            #SUBPLOT 4 - ASYMMETRIC EMISSIVITY
            axx = ax[1,1]
            x_data  = return_data[camera]['profile']['asym_parameter'].coords['rho_poloidal'].data
            par_asy = return_data[camera]['profile']['asym_parameter'].data[i,:]
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
                if save_directory=='':
                    save_directory = os.getcwd()
                fileName = save_directory + '/' + fileName + '.png'
                #SAVING THE PLOT
                plt.savefig(fileName)
                print(fileName+' is saved')
                plt.close()
    
    #RETURNING THE FIGURES
    return return_Fig_data
