import getpass
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from xarray import DataArray
from scipy import interpolate
import time as tt
import pickle

import st40_sxr_inversion as sxr_eval
import SXR_inversion_plot as sxr_plot

plots = True
save_plot = True
save_data = True

pulseNo = 9184
t1 = 0.05
t2 = 0.06
time = [t1,t2]

#SAVE DIRECTORY
save_directory_base = ''

#DEFAULT INPUT DATA
input_data = dict(
    d_time = 5*1.e-3,
    angle=0,
    R_shift=0,
    z_shift=0,
    fit_asymmetry=False,
    compute_asymmetry=False,
    debug=True,
    runParallel = False,
    datatype='sxr',
    knots = 6,
    cameras = ['filter_4'],
    n_intervals = 65,
    )

# if __name__ == "__main__":
#MAKING SXR INVERSION
return_data = sxr_eval.make_SXR_inversion(pulseNo,time,input_data)

#PLOTTING SXR INVERSION
return_data['Fig'] = sxr_plot.make_SXR_inversion_plots(return_data,saveFig=False,save_directory='')