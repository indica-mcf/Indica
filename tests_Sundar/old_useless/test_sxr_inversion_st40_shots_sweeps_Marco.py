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

#PULSE NUMBERS
data_pulseNos = dict(
    ohmic = [9229],
    NBI = [],    
    )

#TIMES
data_times = dict(
    ohmic = 77.5,
    NBI   = 70,    
    )

#INTEGRATION TIME
dt = 3 * 1.e-3


#PULSE NOS AND TIMES
pulseNos = []
times = []
for key in data_pulseNos.keys():
    pulseNos += data_pulseNos[key]
    times    += list(np.tile(data_times[key]*1.e-3,len(data_pulseNos[key])))

#SAVE DIRECTORY
save_directory = '/home/sundaresan.sridhar/Modules/sxr_inversion/sweep_Marco'

#ZSHIFTS
z_shifts = np.arange(0,6) * 1.e-2

#SWEEP OF PULSES
for i,pulseNo in enumerate(pulseNos):
    for z_shift in z_shifts:
        ss.make_SXR_inversion(pulseNo,[times[i],times[i]+(dt)],dt,angle=0,R_shift=0,z_shift=z_shift,debug=True,plots=True,save_directory=save_directory)
    
