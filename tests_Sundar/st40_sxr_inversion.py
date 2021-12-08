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
from indica.converters import FluxMajorRadCoordinates
from indica.converters import ImpactParameterCoordinates


from xarray import Dataset

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
        debug_data = {'steps':{}}
        starting_time = tt.time()
        st = starting_time
    
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
        step = 'Reading ST40 reader'
        step_time = np.round(tt.time() - st,2)
        debug_data['steps'][step] = step_time
        print(step+'. It took '+str(step_time)+' seconds')
        st = tt.time()
    
    #EQUILIBRIUM DATA
    equilib_dat = reader.get("", "efit", 0)
    equilibrium = Equilibrium(equilib_dat,R_shift=input_data['R_shift'],z_shift=input_data['z_shift'])  
    #DEBUG TIME
    if input_data['debug']:
        step = 'Reading equilibrium'
        step_time = np.round(tt.time() - st,2)
        debug_data['steps'][step] = step_time
        print(step+'. It took '+str(step_time)+' seconds')
        st = tt.time()
            
    #READING SXR DATA
    reader.angle = input_data['angle']
    sxr = reader.get("sxr", "diode_arrays", 1, input_data['cameras'])
    plt.close('all')
    #DEBUG TIME
    if input_data['debug']:
        step = 'Reading SXR data'
        step_time = np.round(tt.time() - st,2)
        debug_data['steps'][step] = step_time
        print(step+'. It took '+str(step_time)+' seconds')
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
        sxr[k] = data
    for data in itertools.chain( equilib_dat.values(), sxr.values()):
        if hasattr(data.attrs["transform"], "equilibrium"):
            del data.attrs["transform"].equilibrium
    for data in itertools.chain( sxr.values()):
        data.indica.equilibrium = equilibrium

    #DEBUG TIME
    if input_data['debug']:
        step = 'Remapping SXR data'
        step_time = np.round(tt.time() - st,2)
        debug_data['steps'][step] = step_time
        print(step+'. It took '+str(step_time)+' seconds')
        st = tt.time()
    
    #BINNING THE SXR CAMERAS
    x2 = np.linspace(0.0, 1.0, input_data['n_intervals'])
    sxr_binned = {}
    for key,c in sxr.items():
        sxr_binned[key] = Dataset(
            {"camera": bin_to_time_labels(t.data, c)},
            {c.attrs["transform"].x2_name: x2},
            {"transform": c.attrs["transform"]},
        )
        sxr_binned[key].attrs['dl'] = c.attrs["dl"]
    #DEBUG TIME
    if input_data['debug']:
        step = 'Binning SXR data'
        step_time = np.round(tt.time() - st,2)
        debug_data['steps'][step] = step_time
        print(step+'. It took '+str(step_time)+' seconds')
        st = tt.time()

    
    rho_maj_rad = FluxMajorRadCoordinates(flux_coords)
    rho_max = 0.0
    #SWEEP OF BINNED VALUES
    for c in sxr_binned.values():
        if input_data['exclude_bad_points']:
            c["has_data"] = np.logical_not(np.isnan(c.camera.isel(t=0))) & (c.camera.isel(t=0) >= 1.e+3)
        else:
            
            c["has_data"] = np.logical_not(np.isnan(c.camera.isel(t=0)))
        c.attrs["nlos"] = int(np.sum(c["has_data"]))
        ip_coords = ImpactParameterCoordinates(
            c.attrs["transform"], flux_coords, times=t
        )
        c.attrs["impact_parameters"] = ip_coords
        rho_max = max(rho_max, ip_coords.rhomax())        
        c.coords['rho'], c.coords['R'] = c.indica.convert_coords(rho_maj_rad)
        impact_param, _ = c.indica.convert_coords(ip_coords)
        c["weights"] = c.camera * (0.02 + 0.18 * np.abs(impact_param))
        c["weights"].attrs["transform"] = c.camera.attrs["transform"]
        c["weights"].attrs["datatype"] = ("weighting", input_data['datatype'])
        c.coords["R_0"] = c.attrs["transform"].equilibrium.R_hfs(
            c.coords['rho'], c.coords["t"]
        )[0]   
   #DEBUG TIME
    if input_data['debug']:
        step = 'SXR weight estimation'
        step_time = np.round(tt.time() - st,2)
        debug_data['steps'][step] = step_time
        print(step+'. It took '+str(step_time)+' seconds')
        st = tt.time()

        
    # #GETTING THE AASYMMETRY PARAMETER    
    # #FUNCTION TO ESTIMATE ASYMMETRY PARAMETER
    # def get_asymmetry_parameter(pulseNo,t):
    #     #FUNCTION TO GET GAUSSIAN PROFILE
    #     def get_gaussian(sigma = 0.35,center=0):
    #         #X DATA
    #         x = np.linspace(0,1,100)
    #         #Y DATA
    #         y_term = (-1/2) * (((x-center)/sigma)**2)
    #         return x,np.exp(y_term)
    #     # #X AND Y GAUSSIAN
    #     # p,y_gaussian = get_gaussian()
    #     # #PROFILES
    #     # Ti = 4 * 1.e+3 * y_gaussian #keV
    #     # Te = 4 * 1.e+3 * y_gaussian #keV
    #     # Zeff = 2 * y_gaussian       #no unit
    #     # wphi = 1 * 1.e+3 * y_gaussian     #Hz
    #     # qs = 2 * y_gaussian
    #     # mi = y_gaussian
    #     # ms = 12 * y_gaussian
    #     # #ASYMETTERY PARAMETER
    #     # term_1 = (ms * (wphi**2)) / (2*Ti)
    #     # term_2_1 = qs/ms
    #     # term_2_2 = (mi*Zeff*Te)/(Ti+(Zeff*Te))
    #     # term_2 = term_2_1 * term_2_2
    #     # lambda_s = term_1 * (1-term_2)
    #     p,asym = get_gaussian(sigma=0.35,center=0.8)
    #     asymmetry_parameter = DataArray(np.repeat(np.array([asym]),len(t),axis=0), dims = ["t","p"], coords={"t":t,"p":p})
    #     return asymmetry_parameter
    # #DEBUG TIME
    # if input_data['debug']:
    #     print('Asymmetry parameter estimation. It took '+str(tt.time()-st)+' seconds')
    #     st = tt.time()
        
    #EMISSIVITY PROFILE
    inverter = InvertRadiationST40(len(input_data['cameras']), input_data["datatype"], input_data['knots'], input_data['runParallel'],input_data['fit_asymmetry'],input_data['debug'])
    # # if input_data['compute_asymmetry']:
    # #     inverter.asymmetry_parameter = get_asymmetry_parameter(pulseNo,t)
    #OTHER PARAMETER TO PASS TO THE INVERTER CLASS
    inverter.dim_name = "rho_" + flux_coords.flux_kind
    inverter.rho_max = rho_max
    #FITTING THE EMISSIVITY
    return_data = inverter(R, z, t, sxr_binned,flux_coords)
    #DEBUG TIME
    if input_data['debug']:
        step = 'SXR inversion'
        step_time = np.round(tt.time() - st,2)
        debug_data['steps'][step] = step_time
        print(step+'. It took '+str(step_time)+' seconds')
        st = tt.time()
        
    #UPDATING THE RETURN DATA
    return_data['input_data'] = input_data
    return_data['pulseNo'   ] = pulseNo
    return_data['time_range'] = time
    
    #TOTAL TIME ELAPSED
    if input_data['debug']:
        step = 'Total evaluation time'
        step_time = np.round(tt.time() - starting_time,2)
        debug_data[step] = step_time
        print(step+'. It took '+str(step_time)+' seconds')
        st = tt.time()
    
    #DEBUG DATA
    if input_data['debug']:
        return_data['debug_data'].update(debug_data)
    
    #RETURNING THE DATA
    return return_data