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

starting_time = tt.time()

pulseNo = 9229
# angles = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
angle = -0.5
knots = 6
cameras = ['filter_4']
plots_independent = False
plots = True
save_plot = True
t1 = 0.05
t2 = 0.051
# R_shifts = [-0.01,-0.02,-0.03,-0.04,-0.05,0,0.01,0.02,0.03,0.04,0.05]
R_shifts = [-0.01,-0.02,-0.03,-0.04,-0.05,0.06,0.07,0.08,0.09,0.1,-0.06,-0.07,-0.08,-0.09,-0.1]
# R_shift = 0

t_1 = 0.01
t_2 = 0.1

#ST40 COORDINATES
MACHINE_DIMS = ((0.15, 0.8), (-0.8, 0.8))
R = coord_array(np.linspace(MACHINE_DIMS[0][0],MACHINE_DIMS[0][1], 100), "R")
z = coord_array(np.linspace(MACHINE_DIMS[1][0],MACHINE_DIMS[1][1], 1000), "z")
t = coord_array(np.linspace(t1,t2, 2), "t")

#ST40 READER INITIATION
reader = ST40Reader(pulseNo,tstart=t_1,tend=t_2)

equilib_dat = reader.get("", "efit", 0)

# equilibrium = Equilibrium(equilib_dat,R_shift=R_shift)  

# for angle in angles:
for R_shift in R_shifts:
    
    equilibrium = Equilibrium(equilib_dat,R_shift=R_shift)  

    
    print('angle is '+str(angle))
        
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
    if plots_independent:

        #CLOSING ALL THE PLOTS
        plt.close('all')
        
        for time in t:
            if time.data==t1:
                
                #BASE TITLE
                baseTitle = '#'+str(pulseNo)+' @ t='+str(np.round(time.data*1.e+3,0))+' ms - angle = '+str(angle)+' [degree]'+', knots = '+str(int(knots))+', chi2 = '+str(np.round(chi2,2))+', R_shift = '+str(np.round(R_shift*1.e+2,0))+' cm'+' - '
                
                #BASE FILENAME
                baseFilenameDir = '/home/sundaresan.sridhar/Modules/sxr_inversion/'
                baseFilename = '_'+str(pulseNo)+'_t_'+str(int(time.data*1.e+3))+'_ms_angle_'+str(np.round(angle,1))+'_knots_'+str(int(knots))+'_Rshift_'+str(int(R_shift*1.e+2))+'.png'
                
                
                #PLOT 1 - EMISSIVITY PROFILE
                plt.figure(figsize=(16,10))
                plotTitle = 'emissivity_2D'
                #SWEEP OF Z VALUES
                for iz,zval in enumerate(emissivity.z.data):
                    plt.plot(emissivity.R.data,emissivity.data[0,:,iz],label='z = '+str(np.round(zval,2))+' m')
                
                plt.xlabel('R [m]')
                plt.ylabel('emissivity [W/m3]')
                plt.legend()
                plt.title(baseTitle+'Emissivity profile')
                os.chdir(baseFilenameDir)
                if save_plot:
                    plt.savefig(plotTitle+baseFilename)
                    print(baseFilenameDir+plotTitle+baseFilename+' is saved')
                    plt.close()
                    
                #PLOT 2 - SYMMETRIC EMISSIVITY
                plt.figure(figsize=(16,10))
                plotTitle = 'symmetric_emissivity'
                plt.plot(emiss_fit.rho_poloidal.data,emiss_fit.symmetric_emissivity.data[0,:])        
                plt.xlabel('rho poloidal')
                plt.ylabel('symmetric emissivity [W/m3]')        
                plt.title(baseTitle+'Symmetric emissivity')
                os.chdir(baseFilenameDir)
                if save_plot:
                    plt.savefig(plotTitle+baseFilename)
                    print(baseFilenameDir+plotTitle+baseFilename+' is saved')
                    plt.close()
                
                #PLOT 3 - ASYMMETRIC EMISSIVITY
                plt.figure(figsize=(16,10))
                plotTitle = 'asymmetric_parameter'
                plt.plot(emiss_fit.rho_poloidal.data,emiss_fit.asymmetry_parameter.data[0,:])        
                plt.xlabel('rho poloidal')
                plt.ylabel('asymmetric parameter')        
                plt.title(baseTitle+'Asymmetry parameter')
                os.chdir(baseFilenameDir)
                if save_plot:
                    plt.savefig(plotTitle+baseFilename)
                    print(baseFilenameDir+plotTitle+baseFilename+' is saved')
                    plt.close()
                
                #PLOT 4 - BACKINTEGRAL
                p_impact = np.round(camera_results[0].attrs['impact_parameters'].rho_min.sel(t=time).data,2)
                plt.figure(figsize=(16,10))
                plotTitle = 'back_integral'        
                for cam, cname in zip(camera_results, cameras):
                    data = cam["camera"].sel(t=time)
                    cam["back_integral"].sel(t=time).plot(label="From model")
                    data.plot.line("o", label="From camera")
                plt.xlabel('Impact parameters [m]')
                plt.title(baseTitle+'Back integral - chi2 = '+str(np.round(chi2,2)))
                plt.xticks(np.arange(0,len(p_impact)),p_impact)
                os.chdir(baseFilenameDir)
                if save_plot:
                    plt.savefig(plotTitle+baseFilename)
                    print(baseFilenameDir+plotTitle+baseFilename+' is saved')
                    plt.close()
                
        #PLOT 5 - DIRECTION
        plt.figure(figsize=(16,10))
        plotTitle = 'los'
        f4 = sxr["filter_4"]
        for i in range(f4.R.shape[0]):
            plt.plot(f4.R[i,:], f4.z[i, :])
        plt.xlabel("R (m)")
        plt.ylabel("z (m)")
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        plt.vlines(0.17, -0.5, 0.5, label="Inner column", color="black")
        plt.title(baseTitle+"SXR LOS in (R,z) plane")
        plt.legend()
        if save_plot:
            plt.savefig(plotTitle+baseFilename)
            print(baseFilenameDir+plotTitle+baseFilename+' is saved')
            plt.close()
        
    #PLOTS
    if plots:

        #CLOSING ALL THE PLOTS
        plt.close('all')
        
        for i,time in enumerate(t):
                
            #FIGURE DECLARATION
            fig,ax = plt.subplots(nrows=2,ncols=3,squeeze=True,figsize=(16,10))
            gs = ax[0, 2].get_gridspec()
                  
            #BASE TITLE
            baseTitle = '#'+str(pulseNo)+' @ t='+str(np.round(time.data*1e+3,0))+' ms , angle = '+str(angle)+' [degree]'+', knots = '+str(int(knots))+', chi2 = '+str(np.round(chi2_values[i],2))+', R_shift = '+str(np.round(R_shift*1.e+2,0))+' cm'+' - '
            print(baseTitle)
            
            #BASE FILENAME
            baseFilenameDir = '/home/sundaresan.sridhar/Modules/sxr_inversion/'
            baseFilename = 'combined_'+str(pulseNo)+'_t_'+str(int(time.data*1.e+3))+'_ms_angle_'+str(np.round(angle,1))+'_knots_'+str(int(knots))+'_Rshift_'+str(int(R_shift*1.e+2))+'.png'
            
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
            p_impact = np.round(camera_results[0].attrs['impact_parameters'].rho_min.sel(t=time).data,2)
            data_exp = camera_results[0]['camera'].sel(t=time).data
            data_the = camera_results[0]['back_integral'].sel(t=time).data
            axx.scatter(p_impact,data_exp/1.e+3,color='r')
            axx.plot(p_impact,data_the/1.e+3,color='b')
            axx.set_xlabel('Impact parameters [m]')
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
            f4 = sxr["filter_4"]
            for i in range(f4.R.shape[0]):
                axbig.plot(f4.R[i,:], f4.z[i, :])
            axbig.set_xlabel("R (m)")
            axbig.set_ylabel("z (m)")
            axbig.set_xlim(0, 1)
            axbig.set_ylim(-0.5, 0.5)
            axbig.vlines(0.17, -0.5, 0.5, label="Inner column", color="black")
            axbig.hlines(0, 0, 1, label="Inner column", color="black",linestyles='dashed')
        
            #SAVING THE PLOT
            if save_plot:
                os.chdir(baseFilenameDir)
                plt.savefig(baseFilename)
                print(baseFilenameDir+baseFilename+' is saved')
                plt.close()

print('It elapsed '+str(tt.time()-starting_time)+' seconds')