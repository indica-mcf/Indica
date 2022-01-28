import getpass
import itertools

import matplotlib.pyplot as plt
import numpy as np
import os
from xarray import DataArray
import multiprocessing as mp
from  scipy.interpolate import interp1d, RectBivariateSpline

from indica.equilibrium import Equilibrium
from indica.operators import InvertRadiationST40
from indica.readers import ST40Reader
from indica.utilities import coord_array
from indica.converters import bin_to_time_labels
from indica.converters import FluxSurfaceCoordinates
from indica.converters import FluxMajorRadCoordinates
from indica.converters import ImpactParameterCoordinates

import tomo_1D

from xarray import Dataset
import time as tt
import pickle

#DEFAULT INPUT DATA
input_data_default = dict(
    d_time = 5*1.e-3,
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
    optimize_z_shift = False,
    exclude_bad_points = True,
    EFIT_run = 0,
    method = 'indica',
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
    
    # #EQUILIBRIUM DATA
    # st = tt.time()
    # equilib_dat = reader.get("", "efit", 0)
    # equilibrium = Equilibrium(equilib_dat,R_shift=input_data['R_shift'],z_shift=z_shift)  
    # #DEBUG TIME
    # if input_data['debug']:
    #     step = 'Reading equilibrium'
    #     step_time = np.round(tt.time() - st,2)
    #     debug_data['steps'][step] = step_time
    #     print(step+'. It took '+str(step_time)+' seconds')
    #     st = tt.time()
       
    # #REMAPPING THE SXR DATA
    # flux_coords = FluxSurfaceCoordinates("poloidal")
    # flux_coords.set_equilibrium(equilibrium)
    # npts = 100
    # for k, data in sxr.items():
    #     trans = data.attrs["transform"]
    #     x1 = data.coords[trans.x1_name]
    #     x2_arr = np.linspace(0, 1, npts)
    #     x2 = DataArray(x2_arr, dims=trans.x2_name)
    #     dl = trans.distance(trans.x2_name, DataArray(0), x2[0:2], 0)[1]
    #     data.attrs["x2"] = x2
    #     data.attrs["dl"] = dl
    #     sxr[k] = data
    # for data in itertools.chain( equilib_dat.values(), sxr.values()):
    #     if hasattr(data.attrs["transform"], "equilibrium"):
    #         del data.attrs["transform"].equilibrium
    # for data in itertools.chain( sxr.values()):
    #     data.indica.equilibrium = equilibrium      
    # #DEBUG TIME
    # if input_data['debug']:
    #     step = 'Remapping SXR data'
    #     step_time = np.round(tt.time() - st,2)
    #     debug_data['steps'][step] = step_time
    #     print(step+'. It took '+str(step_time)+' seconds')
    #     st = tt.time()
      
    # #BINNING THE SXR CAMERAS
    # x2 = np.linspace(0.0, 1.0, input_data['n_intervals'])
    # sxr_binned = {}
    # for key,c in sxr.items():
    #     sxr_binned[key] = Dataset(
    #         {"camera": bin_to_time_labels(t.data, c)},
    #         {c.attrs["transform"].x2_name: x2},
    #         {"transform": c.attrs["transform"]},
    #     )
    #     sxr_binned[key].attrs['dl'] = c.attrs["dl"]
    # #DEBUG TIME
    # if input_data['debug']:
    #     step = 'Binning SXR data'
    #     step_time = np.round(tt.time() - st,2)
    #     debug_data['steps'][step] = step_time
    #     print(step+'. It took '+str(step_time)+' seconds')
    #     st = tt.time()

    # #WEIGHT ESTIMATION
    # rho_maj_rad = FluxMajorRadCoordinates(flux_coords)
    # rho_max = 0.0
    # #SWEEP OF BINNED VALUES
    # for c in sxr_binned.values():
    #     if input_data['exclude_bad_points']:
    #         c["has_data"] = np.logical_not(np.isnan(c.camera.isel(t=0))) & (c.camera.isel(t=0) >= 1.e+3)
    #     else:          
    #         c["has_data"] = np.logical_not(np.isnan(c.camera.isel(t=0)))
    #     c.attrs["nlos"] = int(np.sum(c["has_data"]))
    #     ip_coords = ImpactParameterCoordinates(
    #         c.attrs["transform"], flux_coords, times=t
    #     )
    #     c.attrs["impact_parameters"] = ip_coords
    #     c.coords['rho'], c.coords['R'] = c.indica.convert_coords(rho_maj_rad)
    #     impact_param, _ = c.indica.convert_coords(ip_coords)
    #     c["weights"] = c.camera * (0.02 + 0.18 * np.abs(impact_param))
    #     c["weights"].attrs["transform"] = c.camera.attrs["transform"]
    #     c["weights"].attrs["datatype"] = ("weighting", input_data['datatype'])
    #     c.coords["R_0"] = c.attrs["transform"].equilibrium.R_hfs(
    #         c.coords['rho'], c.coords["t"]
    #     )[0]  
    # #DEBUG TIME
    # if input_data['debug']:
    #     step = 'SXR weight estimation'
    #     step_time = np.round(tt.time() - st,2)
    #     debug_data['steps'][step] = step_time
    #     print(step+'. It took '+str(step_time)+' seconds')
    #     st = tt.time()
    
    
    #FUNCTION TO PERFORM SXR INVERSION WITH SWEEP OF Z_SHIFT
    def sweep_z_shift(z_shift,sxr):
        #EQUILIBRIUM DATA
        st = tt.time()
        equilib_dat = reader.get("", "efit", input_data['EFIT_run'])
        equilibrium = Equilibrium(equilib_dat,R_shift=input_data['R_shift'],z_shift=z_shift)  
        #DEBUG TIME
        if input_data['debug']:
            step = 'Reading equilibrium'
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
  
        #WEIGHT ESTIMATION
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
      
      #SXR INVERSION FROM INDICA
      if input_data['method'] = 'indica':
            
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
            
      else:
          
          #INPUT DATA OF TOMOGRAPHY
          #SWEEP OF CAMERAS
          data_all = {}
          for camera in input_data['cameras']:
              #SELECTING SXR AND BINNED DATA
              sel_sxr = sxr[camera]
              sel_bin = sxr_binned[camera]
              #INPUT DATA
              inp_data = dict(
                  brightness = sel_bin['camera'].data,
                  dl = sel_bin.dl.data,
                  t = sel_bin.t.data,
                  R = sel_bin.R.data,
                  z = sel_bin.z.data,
                  rho_equil = dict(
                      R = equilibrium.rho.R.data,
                      z = equilibrium.rho.z.data,
                      t = equilibrium.rho.t.data,
                      rho = equilibrium.rho.data,
                      ),
                  impact_parameters = sel_bin.attrs['impact_parameters'].rho_min.data.T,
                  debug = input_data['debug'],
                  )
              data_all[camera] = inp_data
              
          
          #TOMOGRAPHY CLASS INITIALIZATION
          tomo = tomo_1D.SXR_tomography(data)
          #PERFORMING INVERSION USING TOMOGRAOHIC ROUTINE FROM TOMAS            
          return_data = tomo()



            
            
            
#          #DEBUG TIME
#          if input_data['debug']:
#              step = 'SXR inversion'
#              step_time = np.round(tt.time() - st,2)
#              debug_data['steps'][step] = step_time
#              print(step+'. It took '+str(step_time)+' seconds')
#              st = tt.time()
            
#          #UPDATING THE RETURN DATA
#          return_data['input_data'] = input_data
#          return_data['pulseNo'   ] = pulseNo
#          return_data['time_range'] = time
        
#          #TOTAL TIME ELAPSED
#          if input_data['debug']:
#              step = 'Total evaluation time'
#              step_time = np.round(tt.time() - starting_time,2)
#              debug_data[step] = step_time
#              print(step+'. It took '+str(step_time)+' seconds')
#              st = tt.time()
        
#          #DEBUG DATA
#          if input_data['debug']:
#              return_data['debug_data'].update(debug_data)
        
#          #RETURNING THE DATA
#          return return_data
     
#      #RETURN DATA
#      if input_data['optimize_z_shift']:
#          #SWEEP OF Z_SHIFT
#          results = {}
#          for i,z_shift in enumerate(input_data['z_shift']):
#              #KEY
#              key = 'sweep_value_'+str(i+1)
#              #RESULTS
#              results[key] = sweep_z_shift(z_shift,sxr)
#          #DEBUG DATA
#          if input_data['debug']:
#              results['debug_data'] = dict(total_time_for_sweep=tt.time()-starting_time)
#          #OPTIMIZING THE RESULT
#          results = optimize_results(results,input_data['z_shift'])
#      else:
#          results = sweep_z_shift(input_data['z_shift'],sxr)
#      #RETURNING THE RESULTS
#      return results
         

# #FUNCTION TO OPTIMIZE THE RESULTS
# def optimize_results(dataI,z_shifts):
#     #OPTIMIZE RESULTS DECLARATION
#     results_optimize = {}
#     #CHI2 GROUPING
#     for i in range(0,len(z_shifts)):
#         temp_chi2 = np.array([dataI['sweep_value_'+str(i+1)]['filter_4']['back_integral']['chi2']]).T
#         if 'chi2_evolution' in results_optimize.keys():
#             results_optimize['chi2_evolution'] = np.append(results_optimize['chi2_evolution'],temp_chi2,axis=1)
#         else:
#             results_optimize['chi2_evolution'] = temp_chi2
#     #Z SHIFT VALUES
#     results_optimize['z_shifts'] = z_shifts
#     #OPTIMUM Z_SHIFT INDICES AND VALUES
#     results_optimize['opt_ind'] = np.array([])
#     results_optimize['opt_zshift'] = np.array([])
#     for it in range(0,np.size(results_optimize['chi2_evolution'],0)):
#         results_optimize['opt_ind']     = np.append(results_optimize['opt_ind'],np.where(results_optimize['chi2_evolution'][it,:]==np.nanmin(results_optimize['chi2_evolution'][it,:]))[0][0]).astype(int)
#         results_optimize['opt_zshift']  = np.append(results_optimize['opt_zshift'],results_optimize['z_shifts'][results_optimize['opt_ind'][it]])
#     #MERGING THE DATA
#     copy_once = ['input_data','pulseNo','time_range',
#                 ['filter_4','channels_considered'],
#                 ['filter_4','t'],
#                 ['filter_4','back_integral','channel_no'],
#                 ['filter_4','emissivity_2D','R'],
#                 ['filter_4','emissivity_2D','z'],
#                 ['filter_4','projection'],
#                 ['filter_4','profile','asym_parameter'],
#                 ['filter_4','profile','sym_emissivity'],
#                 ['filter_4','profile','rho_poloidal'],
#                 ]
#     merge_arrays = dict(
#         dim_1              =   [['filter_4','back_integral','chi2']], 
#         dim_2_axis_0       =   [['filter_4','back_integral','data_experiment'],
#                                 ['filter_4','back_integral','data_theory'],
#                                 ['filter_4','back_integral','p_impact']
#                                 ],
#         dim_3_axis_2       =   [['filter_4','emissivity_2D','data']],
#         )
#     group_and_merge = ['debug_data']
#     #FUNCTION TO GET DATA
#     def get_data(dataI,field):
#         if type(field)==str:
#             sel_data = dataI[field]
#         else:
#             #SWEEP OF FIELDS
#             for isub,subfield in enumerate(field):
#                 if isub==0:
#                     sel_data = dataI[subfield]
#                 else:
#                     sel_data = sel_data[subfield]
#         #RETURNING THE SELECTED DATA
#         return sel_data
#     #MERGING TWO DICTIONARIES
#     def mergedicts(dict1, dict2):
#         for k in set(dict1.keys()).union(dict2.keys()):
#             if k in dict1 and k in dict2:
#                 if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
#                     yield (k, dict(mergedicts(dict1[k], dict2[k])))
#                 else:
#                     # If one of the values is not a dict, you can't continue merging it.
#                     # Value from second dict overrides one in first and we move on.
#                     yield (k, dict2[k])
#                     # Alternatively, replace this with exception raiser to alert you of value conflicts
#             elif k in dict1:
#                 yield (k, dict1[k])
#             else:
#                 yield (k, dict2[k])
#     #FUNCTION TO PUT DATA
#     def put_data(dataO,data, field):
#         #TEMP DATA
#         temp_data = {}
#         if type(field)==str:
#             temp_data[field] = data
#         else:
#             #REVERSING THE FIELDS
#             field.reverse()
#             for isub,subfield in enumerate(field):
#                 if isub==0:
#                     temp_data = {subfield:data}
#                 else:
#                     temp_data = {subfield:temp_data}
#         #OUTPUT DATA
#         dataOut = dict(mergedicts(dataO,temp_data))
#         #RETURNING THE OUTPUT DATA
#         return dataOut
#     #COPYING ONCE
#     dataO = {}
#     for field in copy_once:
#         #SELECTED DATA TO COPY
#         sel_data = get_data(dataI['sweep_value_1'],field)
#         #PUTTING THE DATA
#         dataO = put_data(dataO,sel_data,field)
#     #MERGE ARRAY
#     for merge_request in merge_arrays.keys():
#         #SWEEP OF QUANTITIES
#         for quantity in merge_arrays[merge_request]:
#             #SWEEP OF VALUES
#             for it,zind in enumerate(results_optimize['opt_ind']):
#                 #GETTING THE DATA
#                 sel_data = get_data(dataI['sweep_value_'+str(zind+1)],quantity)
#                 #DECLARING THE RET DATA
#                 if (it==0):
#                     if (merge_request!='xarray'):
#                         ret_data = np.nan * np.ones(sel_data.shape)
#                     else:
#                         ret_data = np.nan * np.ones(sel_data.data.shape)
#                 #TIME DATA
#                 if merge_request=='dim_1':
#                     ret_data[it] = sel_data[it]
#                 elif merge_request=='dim_2_axis_0':
#                     ret_data[it,:] = sel_data[it,:]
#                 elif merge_request=='dim_3_axis_2':
#                     ret_data[:,:,it] = sel_data[:,:,it]
#                 elif merge_request=='xarray':
#                     ret_data[it,:] = sel_data.data[it,:] 
#             #WRITING THE DATA
#             if merge_request=='xarray':
#                 sel_data.data = ret_data
#                 ret_data = sel_data
#             dataO = put_data(dataO,ret_data,quantity)
#     #GROUP AND MERGE DATA
#     for field in group_and_merge:
#         #DECLARATION
#         dataO['sweep_'+field] = {}
#         #SWEEP OF ZSHIFTS
#         for i in range(0,len(results_optimize['z_shifts'])):
#             #KEY
#             key = 'sweep_value_'+str(i+1)
#             #DATA
#             dataO['sweep_'+field][key] = dataI[key][field]
#     #OPTIMIZATION DATA
#     dataO['results_optimize'] = results_optimize
#     #OPTIMIZED ZSHIFT VALUES
#     dataO['filter_4']['z_shift'] = results_optimize['opt_zshift']
#     #ALL RESULTS
#     dataO['all_results'] = dataI
#     #RETURNING THE DATA
#     return dataO  