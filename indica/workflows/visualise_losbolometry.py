# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
# ---

from functools import partial
import random
from typing import Callable
import warnings

from io import BytesIO
from deap import base
from deap import creator
from deap import tools
import matplotlib.pylab as plt
import numpy as np
import xarray
from xarray import DataArray
import pickle
from matplotlib.widgets import Slider
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from indica.defaults.load_defaults import load_default_objects
from indica.models import PinholeCamera
from indica.operators import tomo_1D
from indica.operators.atomic_data import default_atomic_data

from los_bolometry_radiation import get_solution, interactive_solution_timeslice_plot_from_list

warnings.simplefilter(action="ignore", category=FutureWarning)




from collections import deque


import numpy as np
def run_example_diagnostic_model(
    machine: str, instrument: str, model: Callable, plot: bool = False, **kwargs
):
    # Initialise plasma and diagnostic model
    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")

    plasma.set_equilibrium(equilibrium)
    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)
    transform.spot_shape = "square"
    transform.focal_length = -1000.0

    model = model(instrument, **kwargs)
    model.set_transform(transform)
    model.set_plasma(plasma)

    machine_r = transform._machine_dims[0][1]

    machine_x0 = transform._machine_dims[0][0]
    machine_x1= transform._machine_dims[0][1]
    machine_z0=transform._machine_dims[1][0]
    machine_z1=transform._machine_dims[1][1]
    bckc, phantom_emission = model(return_emissivity=True)

    filelist=["fullrunHOF_3_gens35.pkl",
              "fullrunHOF_4_gens28.pkl",
              "fullrunHOF_5_gens33.pkl",
              "fullrunHOF_6_gens48.pkl",
              "fullrunHOF_7_gens21.pkl",
              "fullrunHOF_8_gens13.pkl",
              "fullrunHOF_9_gens15.pkl",
              "fullrunHOF_10_gens27.pkl",
              "fullrunHOF_11_gens13.pkl",
              ]
            
    hofs=[]
    for filename in filelist:


        with open(f'/home/jussi.hakosalo/Indica/indica/workflows/jussitesting/{filename}','rb') as file:
                newhof=pickle.load(file)
                hofs.extend(newhof[:5])
        solutions=[]
    print(len(hofs))
    for sol in hofs:
        solutions.append(get_solution(sol,transform,model,phantom_emission,"sqrt"))
    #Sort list
    solutions = sorted(solutions, key=lambda x: x[-2])
    #take only top 20:
 
    solutions=solutions[:20]

    interactive_solution_timeslice_plot_from_list(solutions,init_solution=0)


def run_bm():

    machine = "st40"
    instrument = "blom_xy1"
    _model = PinholeCamera
    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])

    return run_example_diagnostic_model(
        machine, instrument, _model, plot=False, power_loss=power_loss
    )


run_bm()