import getpass
import itertools

import matplotlib.pyplot as plt
import numpy as np
import os
from xarray import DataArray

from indica.equilibrium import Equilibrium
from indica.operators import InvertRadiationST40
from indica.readers import ST40Reader
from indica.utilities import coord_array
from indica.converters import bin_to_time_labels
from indica.converters import FluxSurfaceCoordinates

import time as tt
import pickle

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
    exclude_bad_points = True,
    )

#FUNCTION TO MAKE SXR INVERSION
def make_SXR_inversion(pulseNo,time,input_data={}):
    #INPUT DATA
    for key,value in input_data_default.items():
        if key not in input_data.keys():
            input_data[key] = value
    
    #INITIAL TIME OF EXECUTION
    if input_data['debug']:
        starting_time = tt.time()
        st = starting_time
    
    #DEFAULT VALUES
    knots = 6
    cameras = ['filter_4']
    
    #ST40 COORDINATES
    MACHINE_DIMS = ((0.15, 0.8), (-0.75, 0.75))
    R = coord_array(np.linspace(MACHINE_DIMS[0][0],MACHINE_DIMS[0][1], 100), "R")
    z = coord_array(np.linspace(MACHINE_DIMS[1][0],MACHINE_DIMS[1][1], 100), "z")
    
    #TIME DATA ARRAY
    t  = coord_array(np.arange(time[0],time[1],input_data['d_time']), "t")
    
    #ST40 READER INITIALIZATION
    t1_ST40 = time[0] - (input_data['d_time'])
    t2_ST40 = time[1] + (input_data['d_time'])
    reader = ST40Reader(pulseNo,tstart=t1_ST40,tend=t2_ST40)
    #DEBUG TIME
    if input_data['debug']:
        print('Reading ST40 reader. It took '+str(tt.time()-st)+' seconds')
        st = tt.time()
    
    #EQUILIBRIUM DATA
    equilib_dat = reader.get("", "efit", 0)
    equilibrium = Equilibrium(equilib_dat,R_shift=input_data['R_shift'],z_shift=input_data['z_shift'])  
    #DEBUG TIME
    if input_data['debug']:
        print('Reading equilibrium. It took '+str(tt.time()-st)+' seconds')
        st = tt.time()
            
    #READING SXR DATA
    reader.angle = input_data['angle']
    sxr = reader.get("sxr", "diode_arrays", 1, cameras)
    plt.close('all')
    #DEBUG TIME
    if input_data['debug']:
        print('Reading SXR data. It took '+str(tt.time()-st)+' seconds')
        st = tt.time()
        
    #REMAPPING THE SXR DATA
    flux_coords = FluxSurfaceCoordinates("poloidal")
    flux_coords.set_equilibrium(equilibrium)
    npts = 100
    for k, data in sxr.items():
        trans = data.attrs["transform"]
        x1 = data.coords[trans.x1_name]
        x2_arr = np.linspace(0, 1, npts)
        x2 = DataArray(x2_arr, dims=trans.x2_name)
        dl = trans.distance(trans.x2_name, DataArray(0), x2[0:2], 0)[1]
        data.attrs["x2"] = x2
        data.attrs["dl"] = dl
        data.attrs["x"], data.attrs["y"], data.attrs["z"] = trans.convert_to_xyz(x1, x2, 0)
        data.attrs["R"], data.attrs["z"] = trans.convert_to_Rz(x1, x2, 0)
        rho_equil, _ = flux_coords.convert_from_Rz(
            data.attrs["R"], data.attrs["z"]
        )
        rho = rho_equil.interp(t=data.t, method="linear")
        data.attrs["rho"] = rho
        sxr[k] = data
    for data in itertools.chain( equilib_dat.values(), sxr.values()):
        if hasattr(data.attrs["transform"], "equilibrium"):
            del data.attrs["transform"].equilibrium
    for data in itertools.chain( sxr.values()):
        data.indica.equilibrium = equilibrium
    #DEBUG TIME
    if input_data['debug']:
        print('Remapping SXR data. It took '+str(tt.time()-st)+' seconds')
        st = tt.time()
    
    #GETTING THE AASYMMETRY PARAMETER    
    #FUNCTION TO ESTIMATE ASYMMETRY PARAMETER
    def get_asymmetry_parameter(pulseNo,t):
        #FUNCTION TO GET GAUSSIAN PROFILE
        def get_gaussian(sigma = 0.35,center=0):
            #X DATA
            x = np.linspace(0,1,100)
            #Y DATA
            y_term = (-1/2) * (((x-center)/sigma)**2)
            return x,np.exp(y_term)
        # #X AND Y GAUSSIAN
        # p,y_gaussian = get_gaussian()
        # #PROFILES
        # Ti = 4 * 1.e+3 * y_gaussian #keV
        # Te = 4 * 1.e+3 * y_gaussian #keV
        # Zeff = 2 * y_gaussian       #no unit
        # wphi = 1 * 1.e+3 * y_gaussian     #Hz
        # qs = 2 * y_gaussian
        # mi = y_gaussian
        # ms = 12 * y_gaussian
        # #ASYMETTERY PARAMETER
        # term_1 = (ms * (wphi**2)) / (2*Ti)
        # term_2_1 = qs/ms
        # term_2_2 = (mi*Zeff*Te)/(Ti+(Zeff*Te))
        # term_2 = term_2_1 * term_2_2
        # lambda_s = term_1 * (1-term_2)
        p,asym = get_gaussian(sigma=0.35,center=0.8)
        asymmetry_parameter = DataArray(np.repeat(np.array([asym]),len(t),axis=0), dims = ["t","p"], coords={"t":t,"p":p})
        return asymmetry_parameter
    #DEBUG TIME
    if input_data['debug']:
        print('Asymmetry parameter estimation. It took '+str(tt.time()-st)+' seconds')
        st = tt.time()
        
    #EMISSIVITY PROFILE
    inverter = InvertRadiationST40(len(cameras), "sxr", knots)
    inverter.fit_asymmetry = input_data['fit_asymmetry']
    if input_data['compute_asymmetry']:
        inverter.asymmetry_parameter = get_asymmetry_parameter(pulseNo,t)
    if not input_data['exclude_bad_points']:
        inverter.exclude_bad_points = False
    emissivity, emiss_fit, *camera_results = inverter(R, z, t, *(sxr[c] for c in cameras))
    #DEBUG TIME
    if input_data['debug']:
        print('SXR inversion. It took '+str(tt.time()-st)+' seconds')
        st = tt.time()
    
    #GATHERING THE DATA
    return_data = dict(
        pulseNo = pulseNo,
        R_shift = input_data['R_shift'],
        z_shift = input_data['z_shift'],
        t       = t,
        dt      = input_data['d_time'],
        channels_considered = camera_results[0].has_data.data,
          # results = dict(
          #     emissivity = emissivity,
          #     emiss_fit = emiss_fit,
          #     camera_results = camera_results,
          #     ),
         input_data = dict(
             cameras = cameras,
             R = R,
             z = z,
             t = t,
              knots = knots,
             # sxr = sxr,
             ),
         emissivity_2D = dict(
             R = emissivity.R.data, 
             z = emissivity.z.data,
             data = emissivity.data.T,
             ),
         sxr_data = dict(
             time =  sxr[cameras[0]].t.data,
             data =  sxr[cameras[0]].data,
             no_channel = np.size(sxr[cameras[0]].data,1),        
             ),
        back_integral = dict(
            p_impact        = np.round(camera_results[0].attrs['impact_parameters'].rho_min.data,2).T,
            data_experiment = camera_results[0]['camera'].data,
            data_theory     = camera_results[0]['back_integral'].data,
            channel_no      = np.arange(1,np.size(sxr[cameras[0]].data,1)+1),
            ),
        profile = dict(
              rho_poloidal = emiss_fit.rho_poloidal.data,
              sym_emissivity = emiss_fit.symmetric_emissivity.data,
              asym_parameter = emiss_fit.asymmetry_parameter.data,
              ),
        projection = dict(
            R = sxr[cameras[0]].R.data,
            z = sxr[cameras[0]].z.data,
            ),
        )
    
    #ESTIMATING THE CHI2
    data_exp = return_data['back_integral']['data_experiment'][:,return_data['channels_considered']]
    data_the = return_data['back_integral']['data_theory'    ][:,return_data['channels_considered']]
    chi2 = np.sqrt(np.nansum(((data_exp-data_the)**2) / (data_exp**2),axis=1))
    return_data['back_integral']['chi2']= chi2
    
    #PLOTS
    if input_data['plots']:
        #CLOSING ALL THE PLOTS
        plt.close('all')        
        #SWEEP OF TIMES
        for i,time in enumerate(t):
            #FIGURE DECLARATION
            fig,ax = plt.subplots(nrows=2,ncols=3,squeeze=True,figsize=(16,10))
            gs = ax[0, 2].get_gridspec()
                      
            #BASE TITLE
            baseTitle = '#'+str(pulseNo)+' @ t='+str(np.round((time.data-(input_data['d_time']/2))*1e+3,2))+'-'+str(np.round((time.data+(input_data['d_time']/2))*1e+3,2))+' ms, angle = '+str(input_data['angle'])+' [degree]'+', knots = '+str(int(knots))+', chi2 = '+str(np.round(chi2[i],2))+', R_shift = '+str(np.round(input_data['R_shift']*1.e+2,0))+' cm, z_shift = '+str(np.round(input_data['z_shift']*1.e+2,0))+' cm'
    
            #TITLE OF THE PLOT
            fig.suptitle(baseTitle)
            
            #SUBPLOT 1 - EMISSIVITY PROFILE
            axx = ax[0,0]
            zdata = emissivity.sel(t=time).data.T/1.e+3
            z_min = np.nanmin(zdata)
            z_max = np.nanmax(zdata)
            heatmap = axx.pcolormesh(emissivity.R.data, emissivity.z.data, zdata, cmap='RdBu', vmin=z_min, vmax=z_max)
            axx.set_xlabel("R (m)")
            axx.set_ylabel("z (m)")
            axx.set_xlim(0, 1)
            axx.set_ylim(-0.5, 0.5)
            fig.colorbar(heatmap, ax=axx)
    
            #SUBPLOT 2 - BACKINTEGRAL
            axx = ax[0,1]
            data_exp = camera_results[0]['camera'].sel(t=time).data
            data_the = camera_results[0]['back_integral'].sel(t=time).data
            p_impact = np.arange(1,len(data_exp)+1)
            axx.scatter(p_impact[return_data['channels_considered']],data_exp[return_data['channels_considered']]/1.e+3,color='k')
            axx.plot(p_impact[return_data['channels_considered']],data_the[return_data['channels_considered']]/1.e+3,color='b')        
            axx.scatter(p_impact[np.logical_not(return_data['channels_considered'])],data_exp[np.logical_not(return_data['channels_considered'])]/1.e+3,color='r',marker='v')
            axx.set_xlabel('Channel Number')
            axx.set_ylabel('back integral [kW/m-2]')
    
            #SUBPLOT 3 - SYMMETRIC EMISSIVITY
            axx = ax[1,0]
            axx.plot(emiss_fit.rho_poloidal.data,emiss_fit.symmetric_emissivity.sel(t=time)/1.e+3)        
            axx.set_xlabel('rho poloidal')
            axx.set_ylabel('symmetric emissivity [kW/m3]')        
            
            #SUBPLOT 4 - ASYMMETRIC EMISSIVITY
            axx = ax[1,1]
            axx.plot(emiss_fit.rho_poloidal.data,emiss_fit.asymmetry_parameter.sel(t=time))        
            axx.set_xlabel('rho poloidal')
            axx.set_ylabel('asymmetric parameter')        
            
            #SUBPLOT 5 - DIRECTION
            ax[0,2].remove()
            ax[1,2].remove()                
            axbig = fig.add_subplot(gs[0:, -1])
            #HEATMAP PLOT
            heatmap = axbig.pcolormesh(emissivity.R.data, emissivity.z.data, zdata, cmap='RdBu', vmin=z_min, vmax=z_max)
            f4 = sxr["filter_4"]
            for j in range(f4.R.shape[0]):
                if return_data['channels_considered'][j]:
                    axbig.plot(f4.R[j,:], f4.z[j, :],color='k')
                else:
                    axbig.plot(f4.R[j,:], f4.z[j, :],color='r',linestyle=':')
            axbig.set_xlabel("R (m)")
            axbig.set_ylabel("z (m)")
            axbig.set_xlim(0, 1)
            axbig.set_ylim(-0.5, 0.5)
            axbig.vlines(0.17, -0.5, 0.5, label="Inner column", color="black")
            
            #SAVING THE PLOT
            if input_data['save_directory']!='':
                filename = input_data['save_directory'] + '/' + str(pulseNo)+'_t_'+str(i+1)+'_'+input_data['filenameSuffix']+'.png'
                plt.savefig(filename)
                print(filename+' is saved')
                plt.close()
                    
    #SAVING THE DATA
    if input_data['save_directory']!='':
        filename =input_data['save_directory'] + '/'+str(pulseNo)+'_'+input_data['filenameSuffix']+'.p'
        with open(filename, 'wb') as handle:
            pickle.dump(return_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #TOTAL TIME ELAPSED
    if input_data['debug']:
        print('Total time. It elapsed '+str(tt.time()-starting_time)+' seconds')
    
    #RETURNING THE DATA
    return return_data