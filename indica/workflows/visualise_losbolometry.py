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

from los_bolometry_radiation import get_solution, interactive_solution_timeslice_plot_from_list, assert_valid_impact_params

warnings.simplefilter(action="ignore", category=FutureWarning)




from collections import deque

import re
import os
import numpy as np


from collections import defaultdict
 
def group_hof_files_by_los(folder):
    grouped = defaultdict(list)
    pattern = re.compile(r'_.*?(\d+)los')
 
    for fname in os.listdir(folder):
        if "HOF" not in fname:
            continue
        match = pattern.search(fname)
        if match:
            los_num = int(match.group(1))
            grouped[los_num].append(os.path.join(folder, fname))
    return dict(grouped)



def prune_by_cosine(vectors, sim_thresh=0.95):
    """
    vectors: list of lists (or 2D array), shape (N, D)
    sim_thresh: float in (0,1], higher = stricter (fewer kept)
 
    Returns:
      keep_idxs: indices of selected representatives
      clusters:  list of lists of original indices grouped with each rep
    """
    X = np.asarray(vectors, dtype=float)
    # normalize rows (avoid div by zero)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
 
    keep_idxs = []
    clusters = []
 
    for i, xi in enumerate(Xn):
        if not keep_idxs:
            keep_idxs.append(i)
            clusters.append([i])
            continue
        # compute cosine sim to current reps
        reps = Xn[keep_idxs]                     # (K, D)
        sims = reps @ xi                         # (K,)
        k = np.argmax(sims)
        if sims[k] >= sim_thresh:
            clusters[k].append(i)                # assign to closest rep
        else:
            keep_idxs.append(i)                  # new cluster
            clusters.append([i])
 
    return keep_idxs, clusters


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


    #Get a grouped dict of all files that have the same los number
    filelist_grouped=group_hof_files_by_los("/home/jussi.hakosalo/Indica/indica/workflows/jussitesting/")
            
    hofs=[]

    for los_number in filelist_grouped.keys():
        filelist=filelist_grouped[los_number]
        all_solutions_from_same_los_run=[]

        for filename in filelist:


            with open(f'{filename}','rb') as file:
                    all_solutions_from_same_los_run.extend(pickle.load(file))
        print(len(all_solutions_from_same_los_run))
        keep_idx,_=prune_by_cosine(all_solutions_from_same_los_run,sim_thresh=0.95   )
        print(len(keep_idx))
        to_add=[all_solutions_from_same_los_run[i] for i in keep_idx]
        if len(to_add)>10:
            to_add=to_add[:10]
        hofs.extend([all_solutions_from_same_los_run[i] for i in keep_idx])

    solutions=[]

    #hofs=[[ 0.0, 6.0, 15.0, 18.0, 336.0, 356.0, 0.0, 0.0, 0.14285714, 0.14285714, -0.14285714, 0.0, -0.28571429,-0.7]]
    print(f"Cos pruned to a length of {len(hofs)}")
    for sol in hofs:
        solu=get_solution(sol,transform,model,phantom_emission,"sqrt")
        if solu:
            solutions.append(solu)
    print(f"Validity filtered to a length of {len(solutions)}")

    #Sort list: sort by best penalised
    sort_by_penalized=True
    if sort_by_penalized:
        solutions = sorted(solutions, key=lambda x: x[-2])
        print(solutions[0])
    else:

        solutions = sorted(solutions, key=lambda x: x[-3])



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