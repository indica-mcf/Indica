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

starting_time = tt.time()

baseFilenameDir = '/home/sundaresan.sridhar/Modules/sxr_inversion/sweep_default'




angle = 0
R_shift = 0
z_shift = 0
knots = 6
cameras = ['filter_4']

plots = True
save_plot = True
save_data = True

#PULSE NUMBERS
data_pulseNos = dict(
    ohmic = [9408],#,9409,9411],
    NBI = [],#[9539,9560],    
    )

#TIMES
data_times = dict(
    ohmic = 90,
    NBI   = 70,    
    )

#PULSE NOS AND TIMES
pulseNos = []
times = []
for key in data_pulseNos.keys():
    pulseNos += data_pulseNos[key]
    times    += list(np.tile(data_times[key]*1.e-3,len(data_pulseNos[key])))

#INTEGRATION TIME
dt = 0.005

#SXR CAMERA TIMES
t_1 = 50 * 1.e-3
t_2 = 100 * 1.e-3

#ST40 COORDINATES
MACHINE_DIMS = ((0.15, 0.8), (-0.8, 0.8))
R = coord_array(np.linspace(MACHINE_DIMS[0][0],MACHINE_DIMS[0][1], 100), "R")
z = coord_array(np.linspace(MACHINE_DIMS[1][0],MACHINE_DIMS[1][1], 100), "z")

#SAVE DATA
if save_data:
    return_data = dict(
        R = R,
        z = z,
        pulseNos = pulseNos,
        angle = angle,
        z_shift = z_shift,
        R_shift = R_shift,
        dt = dt,
        )

#SWEEP OF PULSE NUMBERS
for i,pulseNo in enumerate(pulseNos):
    #TIMES
    t1 = times[i]
    t2 = times[i] + (2*dt)
    t = coord_array(np.arange(t1,t2,dt), "t")
    #SAVING TIME DATA
    if save_data:
        return_data[str(pulseNo)+'_t'] = t
    
    #ST40 READER INITIATION
    reader = ST40Reader(pulseNo,tstart=t_1,tend=t_2)

    #EQUILIBRIUM DATA
    equilib_dat = reader.get("", "efit", 0)
    equilibrium = Equilibrium(equilib_dat,R_shift=R_shift,z_shift=z_shift)  
        
    # #READING SXR DATA
    reader.angle = angle
    sxr = reader.get("sxr", "diode_arrays", 0)
    plt.close('all')
    
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

    
    #EMISSIVITY PROFILE
    inverter = InvertRadiation(len(cameras), "sxr", knots)
    emissivity, emiss_fit, *camera_results = inverter(R, z, t, *(sxr[c] for c in cameras))
    
    #CHI2 VALUE
    chi2_values = []
    for i in range(0,len(t.data)):
        D_exp = camera_results[0]['camera'].data[i,:]
        D_the = camera_results[0]['back_integral'].data[i,:]
        chi2 = np.sqrt(np.nansum((D_exp-D_the)**2/(D_exp**2)))
        chi2_values += [chi2]
        
        
    #PLOTS
    if plots:

        #CLOSING ALL THE PLOTS
        plt.close('all')
            
        #SWEEP OF TIMES
        for i,time in enumerate(t):
                    
            #FIGURE DECLARATION
            fig,ax = plt.subplots(nrows=2,ncols=3,squeeze=True,figsize=(16,10))
            gs = ax[0, 2].get_gridspec()
                      
            #BASE TITLE
            baseTitle = '#'+str(pulseNo)+' @ t='+str(np.round(time.data*1e+3,0))+' ms ,dt = '+str(int(dt*1.e+3))+'ms, angle = '+str(angle)+' [degree]'+', knots = '+str(int(knots))+', chi2 = '+str(np.round(chi2_values[i],2))+', R_shift = '+str(np.round(R_shift*1.e+2,0))+' cm'+' - '+', z_shift = '+str(np.round(z_shift*1.e+2,0))+' cm'
                
            #BASE FILENAME
            baseFilename = 'combined_'+str(pulseNo)+'_t_'+str(int(time.data*1.e+3))+'_ms_angle_'+str(np.round(angle,1))+'_knots_'+str(int(knots))+'_Rshift_'+str(int(R_shift*1.e+2))+'_Zshift_'+str(int(z_shift*1.e+2))+'.png'

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
            # p_impact = np.round(camera_results[0].attrs['impact_parameters'].rho_min.sel(t=time).data,2)
            data_exp = camera_results[0]['camera'].sel(t=time).data
            data_the = camera_results[0]['back_integral'].sel(t=time).data
            p_impact = np.arange(1,len(data_exp)+1)
            axx.scatter(p_impact,data_exp/1.e+3,color='r')
            axx.plot(p_impact,data_the/1.e+3,color='b')
            # axx.set_xlabel('Impact parameters [m]')
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
            for i in range(f4.R.shape[0]):
                axbig.plot(f4.R[i,:], f4.z[i, :])
            axbig.set_xlabel("R (m)")
            axbig.set_ylabel("z (m)")
            axbig.set_xlim(0, 1)
            axbig.set_ylim(-0.5, 0.5)
            axbig.vlines(0.17, -0.5, 0.5, label="Inner column", color="black")
            # axbig.hlines(0, 0, 1, label="Inner column", color="black",linestyles='dashed')
        
#             #SAVING THE PLOT
#             if save_plot:
#                 os.chdir(baseFilenameDir)
#                 plt.savefig(baseFilename)
#                 print(baseFilenameDir+baseFilename+' is saved')
#                 plt.close()

#         #SAVING THE DATA
#         if save_data:
#             for i,time in enumerate(t):
#                 #KEY
#                 key = 'combined_'+str(pulseNo)+'_t_'+str(int(time.data*1.e+3))+'_ms_angle_'+str(np.round(angle,1))+'_knots_'+str(int(knots))+'_Rshift_'+str(int(R_shift*1.e+2))+'_Zshift_'+str(int(z_shift*1.e+2))

#                 #DECLARING THE KEY
#                 return_data[key] = {}
                
#                 #EMISSIVITY 2D DATA
#                 return_data[key]['emissivity_2D'] = dict(
#                     R = emissivity.R.data, 
#                     z = emissivity.z.data,
#                     data = emissivity.sel(t=time).data.T,
#                     )
#                 #BACK INTEGRAL DATA
#                 return_data[key]['back_integral'] = dict(
#                     p_impact = np.round(camera_results[0].attrs['impact_parameters'].rho_min.sel(t=time).data,2),
#                     data_exp = camera_results[0]['camera'].sel(t=time).data,
#                     data_the = camera_results[0]['back_integral'].sel(t=time).data,
#                     )
#                 #SYMMETRIC EMISSIVITY DATA
#                 return_data[key]['symmetric_data'] = dict(
#                     rho_poloidal = emiss_fit.rho_poloidal.data,
#                     sym_emissivity = emiss_fit.symmetric_emissivity.sel(t=time),
#                     asym_parameter = emiss_fit.asymmetry_parameter.sel(t=time),
#                     )
#                 #POLOIDAL PROJECTION DATA
#                 return_data[key]['poloidal_projection'] = dict(
#                     R = sxr["filter_4"].R,
#                     z = sxr["filter_4"].z,
#                     )
#                 #CHI2 VALUE
#                 return_data[key]['chi2'] = chi2_values[i]
                
# #SAVING THE DATA
# filename = baseFilenameDir + '/processed_data.p'
# print(filename)
# with open(filename, 'wb') as handle:
#     pickle.dump(return_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print('It elapsed '+str(tt.time()-starting_time)+' seconds')