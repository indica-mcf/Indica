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
from indica.workflows.los_bolometry_geometry import (
    assert_valid_impact_params,
    assert_valid_maximum_impact,
    direction_from_polar_and_dir_offset,
    generate_valid_pair_pool,
    load_pair_pool_csv,
    make_individual_from_pool,
    obstacle_penalty_factor,
    origin_from_polar_angle,
    random_angle_avoiding_left,
    save_pair_pool_csv,
    update_los,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)




from collections import deque
 


        

 
 
 

 


class EarlyStopper:

    """

    Stop when BOTH:

      - best-of-gen hasn't improved for `patience` consecutive generations, and

      - rolling-mean(gen-average) over `win` hasn't improved for `patience` consecutive generations.

    Minimization: lower is better.

    """

    def __init__(self, patience=3, win=3, eps=1e-9):

        self.patience = patience

        self.win = win

        self.eps = eps
 
        self.best_prev = None

        self.best_stall = 0
 
        self.avg_buf = deque(maxlen=win)

        self.roll_prev = None

        self.roll_stall = 0
 
    def update(self, best_now, avg_now):

        # --- best-of-gen tracking ---

        if self.best_prev is None:

            self.best_prev = best_now

            # also seed rolling average

            self.avg_buf.append(avg_now)

            self.roll_prev = np.mean(self.avg_buf)

            return False
 
        best_improved = (self.best_prev - best_now) > self.eps

        self.best_stall = 0 if best_improved else self.best_stall + 1

        self.best_prev = best_now
 
        # --- rolling average of gen averages ---

        self.avg_buf.append(avg_now)

        roll_now = np.mean(self.avg_buf)

        roll_improved = (self.roll_prev - roll_now) > self.eps

        self.roll_stall = 0 if roll_improved else self.roll_stall + 1

        self.roll_prev = roll_now
 
        # stop only if BOTH stalled for `patience` consecutive gens
        print(f"Rolling average: {roll_now:.4e}")

        return (self.best_stall >= self.patience) and (self.roll_stall >= self.patience)
 






 


def random_offset():
    return random.uniform(-0.99, 0.99)

i=0
def make_feasible_individual(generator, evaluate, max_tries=500):
    """
    Keep sampling with `generator()` and testing with `evaluate()`
    until fitness is finite and below BIG. Assign fitness and return.
    """
    global i
    i+=1
    print(f"Created {i} feasible individuals in total")
    for _ in range(max_tries):
        ind = generator()
        fit = evaluate(ind)  # must return a tuple, e.g. (loss,)
        if isinstance(fit, tuple) and len(fit) >= 1:
            f0 = float(fit[0])
            if np.isfinite(f0) and f0 < BIG:
                ind.fitness.values = fit
                return ind
        # else: try again
    raise RuntimeError("Failed to sample a feasible individual after max_tries")

def define_ga(model, number_of_los, phantom_emission):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    preload=True
    if preload:
        pool = load_pair_pool_csv("/home/jussi.hakosalo/Indica/indica/workflows/valid_los_bolom_pairs.csv")
        _rng = np.random.default_rng()
        
        toolbox.register(
            "individual_raw",
            make_individual_from_pool,
            pair_pool=pool,
            n_los=number_of_los,
            
            rng=_rng,
        )
        
        # If you still want a feasibility wrapper, keep it; otherwise:
        toolbox.register("individual", toolbox.individual_raw)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) 
    else:

        toolbox.register("attr_angle", partial(random_angle_avoiding_left,transform=model.transform))
        toolbox.register("attr_offset", random_offset)


        toolbox.register(
            "individual_raw",
            tools.initCycle,
            creator.Individual,
            (toolbox.attr_angle,) * number_of_los + (toolbox.attr_offset,) * number_of_los,
            n=1,
        )


        
        toolbox.register(
            "individual",
            make_feasible_individual,
            generator=toolbox.individual_raw,
            evaluate=toolbox.evaluate,
            max_tries=500,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register(
        "evaluate",
        partial(
            evaluateIndividual,
            model=model,
            phantom_emission=phantom_emission,
        ),
    )



    toolbox.register("mate",tools.cxTwoPoint)
    toolbox.register("mutate",tools.mutFlipBit, indpb=0.05)
    toolbox.register("select",tools.selTournament,tournsize=3)

    return toolbox


def canonicalize_population(pop):
    n_pairs=int(len(pop[0])/2)
    for ind in pop:
        t = np.asarray(ind[:n_pairs], dtype=float) % 360.0
        s = np.asarray(ind[n_pairs:], dtype=float)
        order = np.lexsort((s, t))  # primary t, secondary s
        ind[:n_pairs] = t[order].tolist()
        ind[n_pairs:] = s[order].tolist()
        # invalidate fitness
        if getattr(ind.fitness, "valid", False):
            del ind.fitness.values

BIG=1e13
def evaluateIndividual(individual, model, phantom_emission):
    transform = model.transform

    try:
        # Change origin and direction of lines of sight
        N = len(individual) // 2
        los_angles = individual[:N]
        min_los_angle = np.min(los_angles)
        offsets = individual[N:]
        directions = []
        origins = []
        for i in range(N):
            new_origin_x, new_origin_z = origin_from_polar_angle(los_angles[i], model.transform)
            origins.append((new_origin_x, 0, new_origin_z))
            new_dir_x, new_dir_z = direction_from_polar_and_dir_offset(
                los_angles[i], offsets[i], transform
            )
            directions.append((new_dir_x, 0, new_dir_z))

        transform.set_origin(np.array(origins))
        transform.set_direction(np.array(directions))
        #rotate_all(transform, min_los_angle)
        update_los(transform)

        assert_valid_impact_params(transform)
        assert_valid_maximum_impact(transform)

        # Re-run model and calculate inversion
        bckc = model()
        downsampled_inverted = calculate_tomo_inversion(
            bckc["brightness"], transform, phantom_emission.rhop
        )

        mse, corr = reconstruction_metric(phantom_emission, downsampled_inverted)
        # print("Inidividual MSE: ",mse)

        #Impact parameter penalty for LOS overshooting the plasma
        imp2=transform.impact_rho.sel(t=0,method="nearest")
        #Find all that are larger than 1.1. Sum the overshoots and 3x that to add
        impact_penalty=imp2-1.1
        positive_imp=(3*np.sum(impact_penalty[impact_penalty>0])+1).values



        #Divertor rectangles
        rects=[(0.15,0.45,0.4,0.8),(0.15,0.45,-0.8,-0.4)]
        divertor_penalty=obstacle_penalty_factor(individual,transform,rects)
        if divertor_penalty==2.5:
            return (BIG,)
        else:
            return (float(mse)*positive_imp,)
    except ValueError:
        return (BIG,)
    except IndexError:
        return (BIG,)
    except AssertionError:
        return(BIG,)

def run_ga(number_of_los, model, phantom_emission, plot_gen=False):
    toolbox = define_ga(model, number_of_los, phantom_emission)
    pop = toolbox.population(n=70)
    # evaluate invalid only
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fits = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fits):
        ind.fitness.values = fit
    if plot_gen:  
        return pop 
    
    avg_hist=[]
    best_hist=[]
    best_ind=[]

    hof=tools.HallOfFame(maxsize=10)
    hof.update(pop)
    
    stopper=EarlyStopper()




    CXPB, MUTPB = 0.5, 0.2
    gens = 0
    while gens < 50 :
        gens = gens + 1
        print("-- Generation %i --" % gens)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # variation (no deletes here)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
        
        # canonicalize and invalidate
        canonicalize_population(offspring)  # this does the deletion

        # evaluate invalid only
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        
        pop[:]=offspring
        hof.update(pop)


        fitslist=[ind.fitness.values[0] for ind in pop if ind.fitness.valid]
        print(f"Generation average: {np.format_float_scientific(np.mean(fitslist),precision=3)}")
        print(f"Best so far: {np.format_float_scientific(toolbox.evaluate(hof[0])[0],precision=3)}")
        avg_hist.append(float(np.mean(fitslist)))
        best_hist.append(toolbox.evaluate(hof[0])[0])
        best_ind.append(toolbox.clone(tools.selBest(pop,1)[0]))


        best_now = min(fitslist)

        avg_now  = sum(fitslist) / len(fitslist)
    
        if stopper.update(best_now, avg_now):

            print(f"Early stop at gen {gens}: best={best_now:.6g}, avg={avg_now:.6g}")

            break
 
    """
    #plotting
    gens = np.arange(len(avg_hist))
    plt.subplot(1, 2, 1)
    plt.plot(gens,avg_hist)
    plt.title("Average fitness of gen")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    plt.subplot(1, 2, 2)
    plt.plot(gens,best_hist)
    plt.title("Best individual so far")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()
    """

    return hof,best_ind

















def reconstruction_metric(emissivity, downsampled_inverted):
    # Difference

    diff = emissivity - downsampled_inverted
    mse = (diff**2).mean(dim=("t", "rhop"))

    var_mse=(diff**2).mean(dim="rhop").var(dim="t",ddof=0)
    cv2=var_mse/(mse**2+1e-12)


    corr = xarray.corr(
        emissivity.stack(points=("t", "rhop")),
        downsampled_inverted.stack(points=("t", "rhop")),
        dim="points",
    )

    #Normalized variance. Alpha 0.1 is tiny variance minimization, alpha=1 is consistency equally as important
    alpha=0.1
    loss=mse*(1+alpha*cv2)

    return loss, corr


def calculate_tomo_inversion(
    brightness, transform, rhop_out, reg_level_guess: float = 0.6
):
    has_data = np.logical_not(np.isnan(brightness.isel(t=0).data))
    rho_equil = transform.equilibrium.rhop.interp(t=brightness.t)
    input_dict = dict(
        brightness=brightness.data,
        dl=transform.dl,
        t=brightness.t.data,
        R=transform.R.mean("beamlet").data,
        z=transform.z.mean("beamlet").data,
        rho_equil=dict(
            R=rho_equil.R.data,
            z=rho_equil.z.data,
            t=rho_equil.t.data,
            rho=rho_equil.data,
        ),
        has_data=has_data,
        debug=False,
    )

    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=reg_level_guess)
    tomo()

    inverted_emissivity = DataArray(
        tomo.emiss, coords=[("t", tomo.tvec), ("rhop", tomo.rho_grid_centers)]
    )
    inverted_error = DataArray(
        tomo.emiss_err,
        coords=[("t", tomo.tvec), ("rhop", tomo.rho_grid_centers)],
    )
    inverted_emissivity = inverted_emissivity.assign_coords(
        error=(inverted_emissivity.dims, inverted_error.data)
    )

    # This has 100 radials
    # print(inverted_emissivity)
    # print(inverted_emissivity.shape)
    # This has 41 radials
    # print(emissivity)
    # print(emissivity.shape)

    # Interpolate the desired output radial grid
    downsampled_inverted = inverted_emissivity.interp(rhop=rhop_out)
    # print(downsampled_inverted.shape)
    return downsampled_inverted


 
def interactive_timeslice_plot(phantom_emission, downsampled_inverted, artist_fn):

    """

    Interactive plot with slider on the LEFT, custom artist on the RIGHT.
 
    Parameters

    ----------

    phantom_emission : xarray.DataArray

        True emission, with dims including 't' and 'rhop'.

    downsampled_inverted : xarray.DataArray

        Reconstructed emission, with dims including 't' and 'rhop'.

    artist_fn : callable

        A function taking a Matplotlib Axes as input, and drawing whatever

        you like into it (e.g., artist_fn(ax2)).

    """

    t_vals = np.asarray(phantom_emission.t)

    nT = len(t_vals)
 
    i0 = 0

    t0 = t_vals[i0]
 
    # two subplots side by side with equal width

    fig, (ax_left, ax_right) = plt.subplots(

        1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]}

    )

    plt.subplots_adjust(bottom=0.18)
 
    # --- left panel: timeslice plot ---

    (line_phantom,) = ax_left.plot(

        phantom_emission.rhop,

        phantom_emission.sel(t=t0, method="nearest"),

        label="Phantom",

    )

    (line_recon,) = ax_left.plot(

        downsampled_inverted.rhop,

        downsampled_inverted.sel(t=t0, method="nearest"),

        linestyle="dashed",

        label="Reconstructed",

    )

    ax_left.set_xlabel("rhop")

    ax_left.set_ylabel("emission")

    ax_left.legend()
 
    # Global ylim with margin

    ymin = float(np.nanmin([phantom_emission.min(), downsampled_inverted.min()]))

    ymax = float(np.nanmax([phantom_emission.max(), downsampled_inverted.max()]))

    ax_left.set_ylim(ymin, ymax * 1.05)

    ax_left.set_title(f"t = {t0}")
 
    # --- slider under the left panel ---

    ax_slider = fig.add_axes([0.15, 0.08, 0.35, 0.04])  # narrower to fit under left half

    s_t = Slider(ax=ax_slider, label="t index", valmin=0, valmax=nT - 1,

                 valinit=i0, valstep=1)
 
    def update(idx):

        idx = int(idx)

        tt = t_vals[idx]

        y_p = phantom_emission.sel(t=tt, method="nearest")

        y_r = downsampled_inverted.sel(t=tt, method="nearest")

        line_phantom.set_ydata(y_p)

        line_recon.set_ydata(y_r)

        ax_left.set_title(f"t = {tt}")

        fig.canvas.draw_idle()
 
    s_t.on_changed(update)
 
    # --- right panel: user-supplied artist ---

    artist_fn(ax_right)
 
    plt.show()


    """
    A 2-slider browser:
      - Slider 1 (left, bottom-left): time index for current solution
      - Slider 2 (right, bottom-right): solution index
 
    Parameters
    ----------
    get_solution : callable
        get_solution(sol_idx) -> (phantom_emission, downsampled_inverted, artist_fn)
        - phantom_emission, downsampled_inverted: xarray.DataArray with dims ('t','rhop')
        - artist_fn: callable(ax) that draws the geometry (rasterized image, etc.) into ax
                     (You can use your grab_figure_as_image(...) factory to build these.)
    n_solutions : int
        Number of solutions available (discrete).
    init_solution : int
        Initial solution index.
    """
 
    # --- cache for loaded solutions ---
    cache = {}
    def load(sol_idx):
        if sol_idx not in cache:
            cache[sol_idx] = get_solution(sol_idx)
        return cache[sol_idx]
 
    # --- initial data ---
    phantom, recon, artist_fn = load(init_solution)
    t_vals = np.asarray(phantom.t)
    i0 = 0
    t0 = t_vals[i0]
 
    # --- figure & axes ---
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]}
    )
    plt.subplots_adjust(bottom=0.20)  # a bit more room for two sliders
 
    # --- left panel (lines) ---
    (line_phantom,) = ax_left.plot(
        np.asarray(phantom.rhop),
        np.asarray(phantom.sel(t=t0, method="nearest")),
        label="Phantom",
    )
    (line_recon,) = ax_left.plot(
        np.asarray(recon.rhop),
        np.asarray(recon.sel(t=t0, method="nearest")),
        linestyle="dashed",
        label="Reconstructed",
    )
    ax_left.set_xlabel("rhop"); ax_left.set_ylabel("emission"); ax_left.legend()
    # per-solution y-limits
    def set_left_ylim(p, r):
        ymin = float(np.nanmin([p.min(), r.min()]))
        ymax = float(np.nanmax([p.max(), r.max()]))
        ax_left.set_ylim(ymin, ymax * 1.05)
    set_left_ylim(phantom, recon)
    ax_left.set_title(f"t = {t0}")
 
    # --- right panel (geometry image) ---
    ax_right.cla()
    artist_fn(ax_right)            # draw current solution’s geometry
    ax_right.set_axis_off()        # keep it as an image panel
 
    # --- sliders: time (left) and solution (right) ---
    ax_slider_time = fig.add_axes([0.12, 0.10, 0.35, 0.05])
    s_time = Slider(ax=ax_slider_time, label="t index", valmin=0, valmax=len(t_vals)-1,
                    valinit=i0, valstep=1)
 
    ax_slider_sol  = fig.add_axes([0.57, 0.10, 0.30, 0.05])
    s_sol  = Slider(ax=ax_slider_sol,  label="solution", valmin=0, valmax=n_solutions-1,
                    valinit=init_solution, valstep=1)
 
    # --- state holder (so callbacks always see the latest objects) ---
    state = {
        "phantom": phantom,
        "recon":   recon,
        "t_vals":  t_vals,
        "artist_fn": artist_fn,
        "time_slider": s_time,
        "time_ax": ax_slider_time,
        "current_t_idx": i0,
    }
 
    # --- callbacks ---
    def update_time(idx):
        idx = int(idx)
        state["current_t_idx"] = idx
        tt = state["t_vals"][idx]
        p  = state["phantom"].sel(t=tt, method="nearest")
        r  = state["recon"].sel(t=tt,   method="nearest")
 
        # update x and y (rhop may differ per solution)
        line_phantom.set_xdata(np.asarray(state["phantom"].rhop))
        line_phantom.set_ydata(np.asarray(p))
        line_recon.set_xdata(np.asarray(state["recon"].rhop))
        line_recon.set_ydata(np.asarray(r))
 
        ax_left.set_title(f"t = {tt}")
        fig.canvas.draw_idle()
 
    def change_solution(sol_idx):
        sol_idx = int(sol_idx)
        p, r, art = load(sol_idx)
 
        # update state
        state["phantom"] = p
        state["recon"]   = r
        state["artist_fn"] = art
        state["t_vals"]  = np.asarray(p.t)
 
        # (1) possibly rebuild the time slider if length changed
        new_len = len(state["t_vals"])
        old_slider = state["time_slider"]
        need_rebuild = int(old_slider.valmax) != (new_len - 1)
        # clamp current t to new range
        new_t_idx = min(state["current_t_idx"], new_len - 1)
        if need_rebuild:
            # remove old slider axes
            state["time_ax"].remove()
            # create new slider with updated range
            ax_new = fig.add_axes([0.12, 0.10, 0.35, 0.05])
            s_new  = Slider(ax=ax_new, label="t index", valmin=0, valmax=new_len-1,
                            valinit=new_t_idx, valstep=1)
            s_new.on_changed(update_time)
            state["time_ax"] = ax_new
            state["time_slider"] = s_new
        else:
            # just set its value (this triggers update_time)
            state["time_slider"].set_val(new_t_idx)
 
        # (2) update left y-limits for the new solution
        set_left_ylim(p, r)
 
        # (3) update right panel image
        ax_right.cla()
        state["artist_fn"](ax_right)
        ax_right.set_axis_off()
 
        # (4) force a redraw (update_time is called via slider if rebuilt)
        if not need_rebuild:
            update_time(new_t_idx)
        fig.canvas.draw_idle()
 
    # wire callbacks
    s_time.on_changed(update_time)
    s_sol.on_changed(change_solution)
 
    plt.show()
 
def interactive_solution_timeslice_plot_from_list(solutions, init_solution=0, show_penalties=True):



    """
    Two-slider browser over a LIST of solutions.
 
    Parameters
    ----------
    solutions : list of (phantom, recon, artist_fn)
        - phantom, recon: xarray.DataArray with dims ('t','rhop')
        - artist_fn: callable(ax) -> draws right-panel geometry for that solution
    init_solution : int
        Index of the initial solution to show.
    """



    if not solutions:
        raise ValueError("`solutions` must be a non-empty list.")
    if not (0 <= init_solution < len(solutions)):
        raise ValueError("`init_solution` out of range.")
 
    def _y_limits(p, r):
        ymin = float(np.nanmin([p.min(), r.min()]))
        ymax = float(np.nanmax([p.max(), r.max()]))
        return ymin, ymax * 1.05
 
    # --- pull initial solution ---
    if not show_penalties:
        phantom, recon, artist_fn, mse = solutions[init_solution]
    else:
        phantom, recon, artist_fn, mse, pen_mse, n_los = solutions[init_solution]

    t_vals = np.asarray(phantom.t)
    i0 = 0
    t0 = t_vals[i0]
 
    # --- figure & axes ---
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]}
    )
    plt.subplots_adjust(bottom=0.20)

    supt=fig.suptitle(f"Solution {init_solution},LOS: {n_los},\npenalized MSE= {pen_mse:.4e}, MSE= {mse:.4e} ")
 
    # --- left panel (lines) ---
    (line_phantom,) = ax_left.plot(
        np.asarray(phantom.rhop),
        np.asarray(phantom.sel(t=t0, method="nearest")),
        label="Phantom",
    )
    (line_recon,) = ax_left.plot(
        np.asarray(recon.rhop),
        np.asarray(recon.sel(t=t0, method="nearest")),
        linestyle="dashed",
        label="Reconstructed",
    )
    ax_left.set_xlabel("rhop"); ax_left.set_ylabel("emission"); ax_left.legend()
    ymin, ymax = _y_limits(phantom, recon)
    ax_left.set_ylim(ymin, ymax)
    ax_left.set_title(f"t = {t0}")
 
    # --- right panel (geometry for initial solution) ---
    ax_right.cla()
    artist_fn(ax_right)
    ax_right.set_axis_off()
 
    # --- sliders: time (left) and solution (right) ---
    ax_slider_time = fig.add_axes([0.12, 0.10, 0.35, 0.05])
    s_time = Slider(ax=ax_slider_time, label="t index",
                    valmin=0, valmax=len(t_vals)-1, valinit=i0, valstep=1)
 
    ax_slider_sol  = fig.add_axes([0.57, 0.10, 0.30, 0.05])
    s_sol  = Slider(ax=ax_slider_sol,  label="solution",
                    valmin=0, valmax=len(solutions)-1, valinit=init_solution, valstep=1)
 
    # --- state ---
    state = {
        "phantom": phantom,
        "recon": recon,
        "t_vals": t_vals,
        "artist_fn": artist_fn,
        "mse": mse,
        "pen_mse": pen_mse,
        "n_los": n_los,
        "suptitle":supt,
        "time_slider": s_time,
        "time_ax": ax_slider_time,
        "current_t_idx": i0,
        "current_sol_idx": init_solution,
    }
 
    # --- callbacks ---
    def update_time(idx):
        idx = int(idx)
        state["current_t_idx"] = idx
        tt = state["t_vals"][idx]
 
        p  = state["phantom"].sel(t=tt, method="nearest")
        r  = state["recon"].sel(t=tt,   method="nearest")
 
        line_phantom.set_xdata(np.asarray(state["phantom"].rhop))
        line_phantom.set_ydata(np.asarray(p))
        line_recon.set_xdata(np.asarray(state["recon"].rhop))
        line_recon.set_ydata(np.asarray(r))
 
        ax_left.set_title(f"t = {tt}")
        fig.canvas.draw_idle()
 
    def change_solution(sol_idx):
        sol_idx = int(sol_idx)
        if sol_idx == state["current_sol_idx"]:
            return
        if show_penalties:
            phantom_i, recon_i, artist_fn_i, mse_i, pen_mse_i, n_los = solutions[sol_idx]
        else:
            phantom_i, recon_i, artist_fn_i, mse_i = solutions[sol_idx]
 
        # update state
        state["phantom"] = phantom_i
        state["recon"]   = recon_i
        state["artist_fn"] = artist_fn_i
        state["mse"]= mse_i
        state["pen_mse"]= pen_mse_i
        state["n_los"]=n_los
        state["suptitle"].set_text(f"Solution {sol_idx},LOS: {n_los},\npenalized MSE= {pen_mse_i:.4e}, MSE= {mse_i:.4e} ")
        state["t_vals"]  = np.asarray(phantom_i.t)
        state["current_sol_idx"] = sol_idx
 
        # rebuild time slider if length changed
        new_len = len(state["t_vals"])
        old_len = int(state["time_slider"].valmax) + 1
        new_t_idx = min(state["current_t_idx"], new_len - 1)
 
        if new_len != old_len:
            # remove old slider axes
            state["time_ax"].remove()
            # new time slider
            ax_new = fig.add_axes([0.12, 0.10, 0.35, 0.05])
            s_new  = Slider(ax=ax_new, label="t index",
                            valmin=0, valmax=new_len-1, valinit=new_t_idx, valstep=1)
            s_new.on_changed(update_time)
            state["time_ax"] = ax_new
            state["time_slider"] = s_new
        else:
            # trigger update_time via setting value
            state["time_slider"].set_val(new_t_idx)
 
        # refresh left y-limits for this solution
        ymin, ymax = _y_limits(phantom_i, recon_i)
        ax_left.set_ylim(ymin, ymax)
 
        # refresh right panel geometry
        ax_right.cla()
        state["artist_fn"](ax_right)
        ax_right.set_axis_off()
 
        # ensure left panel data also updated (if we didn't rebuild slider)
        if new_len == old_len:
            update_time(new_t_idx)
 
        fig.canvas.draw_idle()
 
    # wire
    s_time.on_changed(update_time)
    s_sol.on_changed(change_solution)
 
    plt.show()



from io import BytesIO

 


def grab_figure_as_image(callable_plotter, *, pick=None, dpi=200):

    was_interactive = plt.isinteractive()

    plt.ioff()

    try:

        before = set(plt.get_fignums())

        callable_plotter()

        new_ids = [n for n in plt.get_fignums() if n not in before]

        if not new_ids:

            raise RuntimeError("No new figures were created.")

        new_figs = [plt.figure(n) for n in new_ids]
 
        # choose figure

        if pick is None:

            fig = new_figs[-1]

        elif isinstance(pick, int):

            fig = new_figs[pick]

        else:

            matches = [f for f in new_figs if pick(f)]

            if not matches:

                raise RuntimeError("No figure matched the predicate.")

            fig = matches[0]
 
        # optional: tighten layout inside the fig before saving (helps some cases)

        try:

            fig.tight_layout()

        except Exception:

            pass
 
        # save tightly-cropped PNG to memory

        buf = BytesIO()

        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2, dpi=dpi)

        buf.seek(0)

        img = plt.imread(buf)

    finally:

        # close all temp figs

        for f in plt.get_fignums():

            if f in new_ids:

                plt.close(f)

        if was_interactive:

            plt.ion()
 
    def artist_fn(ax):

        # Fill the entire right-hand axes with the image

        ax.imshow(img, aspect="auto", extent=[0, 1, 0, 1])

        ax.set_axis_off()

        return ax
 
    return artist_fn


def get_solution(individual, transform, model, phantom_emission,los_penalty=None,return_transform_object=False):
    try:
        N = len(individual) // 2
        los_angles = individual[:N]
        min_los_angle = np.min(los_angles)
        offsets = individual[N:]
        directions = []
        origins = []
        for i in range(N):
            new_origin_x, new_origin_z = origin_from_polar_angle(los_angles[i], transform)
            origins.append((new_origin_x, 0, new_origin_z))
            new_dir_x, new_dir_z = direction_from_polar_and_dir_offset(
                los_angles[i], offsets[i], transform
            )
            directions.append((new_dir_x, 0, new_dir_z))

        transform.set_origin(np.array(origins))
        transform.set_direction(np.array(directions))
        #rotate_all(transform, min_los_angle)
        update_los(transform)
        assert_valid_impact_params(transform)
        assert_valid_maximum_impact(transform)
        # Re-run model and calculate inversion
        bckc = model()
        downsampled_inverted = calculate_tomo_inversion(
            bckc["brightness"], transform, phantom_emission.rhop
        )
        def pick_geom(fig):
            return any(ax.get_xlabel() == "R [m]" for ax in fig.axes)
        

        
        geom_R_artist = grab_figure_as_image(lambda: transform.plot(), pick=pick_geom)
        



        mse, corr = reconstruction_metric(phantom_emission, downsampled_inverted)


        if los_penalty=="sqrt":
            mse_penalized=(np.sqrt(N))*mse
            return (phantom_emission,downsampled_inverted,geom_R_artist,mse,mse_penalized,N)
        else:
            return (phantom_emission,downsampled_inverted,geom_R_artist,mse,)
    except ValueError:
        print("Nan slice somwhere")
        return None
    except AssertionError:
        print("Impact param overlap")
        return None




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


    pair_generation=False
    if pair_generation:
        rects = [
            (0.15, 0.45,  0.4,  0.8),   # upper-left block
            (0.15, 0.45, -0.8, -0.4),   # lower-left block
        ]
        pairs = generate_valid_pair_pool(
            transform=model.transform,
            rects=rects,
            angle_step_deg=1.0,
            offsets_per_angle= 15,     # e.g., 9 or 13
            offset_kind="grid",          # or "random"
            max_pairs=50000,             # optional cap
        )
        save_pair_pool_csv(pairs, "/home/jussi.hakosalo/Indica/indica/workflows/valid_los_bolom_pairs.csv")
        print("Saved", pairs.shape[0], "valid (angle, offset) pairs.")


    # Run model and inversion
    bckc, phantom_emission = model(return_emissivity=True)

    #Plot the initial pop
    plot_gen=True

    for los_count in range(6,10):
        for runs in range(3):

            savepickle=True
            if savepickle:

                if plot_gen:
                    sols=[]
                    best =run_ga(los_count,model,phantom_emission,plot_gen)
                    for item in best:
                        sol=get_solution(item,transform,model,phantom_emission,"sqrt")
                        if sol:
                            sols.append(sol)
                    interactive_solution_timeslice_plot_from_list(sols)
                    ata
                else:
                    hof,bestPerGen=run_ga(los_count,model,phantom_emission)

                                

                gens=len(bestPerGen)
                with open(f'/home/jussi.hakosalo/Indica/indica/workflows/jussitesting/fullrunHOF_{los_count}los{gens}_gens_run{runs}.pkl', 'wb') as file:
                    # Dump data with highest protocol for best performance
                    pickle.dump(hof, file, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f'/home/jussi.hakosalo/Indica/indica/workflows/jussitesting/fullrunBESTOFGEN_{los_count}los_{gens}gens_run{runs}.pkl', 'wb') as file:
                    # Dump data with highest protocol for best performance
                    pickle.dump(bestPerGen, file, protocol=pickle.HIGHEST_PROTOCOL) 

            else:
                with open('/home/jussi.hakosalo/Indica/indica/workflows/jussitesting/fullrunHOF_12_gens36.pkl','rb') as file:
                        hof=pickle.load(file)
            

    return
    solutions=[]
    #for sol in bestPerGen:
    for sol in hof:
        solutions.append(get_solution(sol,transform,model,phantom_emission))

    interactive_solution_timeslice_plot_from_list(solutions,init_solution=0)
    
    """

    #Best individual to a transform
    print(f"Best individual, to be applied: {best}")
    individual=best

    N = len(individual) // 2
    los_angles = individual[:N]
    min_los_angle = np.min(los_angles)
    offsets = individual[N:]
    directions = []
    origins = []
    for i in range(N):
        new_origin_x, new_origin_z = origin_from_polar_angle(los_angles[i], transform)
        origins.append((new_origin_x, 0, new_origin_z))
        new_dir_x, new_dir_z = direction_from_polar_and_dir_offset(
            los_angles[i], offsets[i]
        )
        directions.append((new_dir_x, 0, new_dir_z))

    transform.set_origin(np.array(origins))
    transform.set_direction(np.array(directions))
    #rotate_all(transform, min_los_angle)
    update_los(transform)

    # Re-run model and calculate inversion
    bckc = model()
    downsampled_inverted = calculate_tomo_inversion(
        bckc["brightness"], transform, phantom_emission.rhop
    )




    # choose the figure whose Axes has xlabel "R [m]"

    
    interactive_timeslice_plot(
        phantom_emission,
        downsampled_inverted,
        artist_fn=geom_R_artist
)
    #transform.plot()
    #plt.show()

    r=1
    for t in phantom_emission.t:
        plt.subplot(3,3,r)
        plt.plot(phantom_emission.rhop, phantom_emission.sel(t=t), label="Phantom")
        plt.plot(
            downsampled_inverted.rhop,
            downsampled_inverted.sel(t=t),
            linestyle="dashed",
            label="Reconstructed",
        )
        plt.legend()
        r+=1
    plt.show()
    """

    mse, corr = reconstruction_metric(phantom_emission, downsampled_inverted)
    print("MSE: ", mse)

    

    return plasma, model, phantom_emission


def run_bm():

    machine = "st40"
    instrument = "blom_xy1"
    _model = PinholeCamera
    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])

    return run_example_diagnostic_model(
        machine, instrument, _model, plot=False, power_loss=power_loss
    )

if __name__ == "__main__":
    run_bm()
