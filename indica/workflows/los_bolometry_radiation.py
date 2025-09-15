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

warnings.simplefilter(action="ignore", category=FutureWarning)



def random_angle():
    return random.uniform(0.0, 360.0)

def _ray_intersects_rect_2d(origin, direction, rect):
    """
    Ray (origin + t*dir, t>=0) vs axis-aligned rectangle in x–z.
    origin: (x,z), direction: (dx,dz), rect: (xmin,xmax,zmin,zmax)
    """
    ox, oz = float(origin[0]), float(origin[1])
    dx, dz = float(direction[0]), float(direction[1])
    xmin, xmax, zmin, zmax = rect
 
    # If starting inside, count as intersecting
    if (xmin <= ox <= xmax) and (zmin <= oz <= zmax):
        return True
 
    inv_dx = np.inf if dx == 0.0 else 1.0 / dx
    inv_dz = np.inf if dz == 0.0 else 1.0 / dz
 
    t1x = (xmin - ox) * inv_dx
    t2x = (xmax - ox) * inv_dx
    tmin_x, tmax_x = (min(t1x, t2x), max(t1x, t2x))
 
    t1z = (zmin - oz) * inv_dz
    t2z = (zmax - oz) * inv_dz
    tmin_z, tmax_z = (min(t1z, t2z), max(t1z, t2z))
 
    t_enter = max(tmin_x, tmin_z)
    t_exit  = min(tmax_x, tmax_z)
    return (t_exit >= max(t_enter, 0.0))


def _any_los_hits_rects(origins_xz, dirs_xz, rects):
    for (ox, oz), (dx, dz) in zip(origins_xz, dirs_xz):
        for rect in rects:
            if _ray_intersects_rect_2d((ox, oz), (dx, dz), rect):
                return True
    return False


def obstacle_penalty_factor(individual, transform, rects):
    """
    Genome: first half = angles (deg), second half = dir_offsets [-1,1].
    rects: list of (xmin,xmax,zmin,zmax) rectangles in x–z.
    Returns 1.5 if any LOS intersects any rect, else 1.0.
    """
    g = np.asarray(individual, dtype=float)
    n = g.size // 2
    angles = (g[:n] % 360.0 + 360.0) % 360.0
    offsets = np.clip(g[n:], -1.0, 1.0)
 
    origins = np.empty((n, 2), dtype=float)
    dirs    = np.empty((n, 2), dtype=float)
    for i, (ang, off) in enumerate(zip(angles, offsets)):
        ox, oz = origin_from_polar_angle(ang, transform)           # (x,z)
        dx, dz = direction_from_polar_and_dir_offset(ang, off)     # (dx,dz)
        origins[i] = (ox, oz)
        dirs[i]    = (dx, dz)
 
    return 1.5 if _any_los_hits_rects(origins, dirs, rects) else 1.0


def obstacle_penalty_factor(individual, transform, rects):
    """
    Genome: first half = angles (deg), second half = dir_offsets [-1,1].
    rects: list of (xmin,xmax,zmin,zmax) rectangles in x–z.
    Returns 1.5 if any LOS intersects any rect, else 1.0.
    """
    g = np.asarray(individual, dtype=float)
    n = g.size // 2
    angles = (g[:n] % 360.0 + 360.0) % 360.0
    offsets = np.clip(g[n:], -1.0, 1.0)
 
    origins = np.empty((n, 2), dtype=float)
    dirs    = np.empty((n, 2), dtype=float)
    for i, (ang, off) in enumerate(zip(angles, offsets)):
        ox, oz = origin_from_polar_angle(ang, transform)           # (x,z)
        dx, dz = direction_from_polar_and_dir_offset(ang, off)     # (dx,dz)
        origins[i] = (ox, oz)
        dirs[i]    = (dx, dz)
 
    return 1.5 if _any_los_hits_rects(origins, dirs, rects) else 1.0

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
        "evaluate",
        partial(
            evaluateIndividual,
            model=model,
            phantom_emission=phantom_emission,
        ),
    )
    
    toolbox.register(
        "individual",
        make_feasible_individual,
        generator=toolbox.individual_raw,
        evaluate=toolbox.evaluate,
        max_tries=500,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)



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

        mse, corr = reconstruction_metric(phantom_emission, downsampled_inverted)
        # print("Inidividual MSE: ",mse)


        rects=[(0.15,0.45,0.8,0.4),(0.15,0.45,-0.4,-0.8)]
        divertor_penalty=obstacle_penalty_factor(individual,transform,rects)
        return (float(mse)*divertor_penalty,)
    except ValueError:
        return (BIG,)

def run_ga(number_of_los, model, phantom_emission):
    toolbox = define_ga(model, number_of_los, phantom_emission)
    pop = toolbox.population(n=40)
    # evaluate invalid only
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fits = list(map(toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fits):
        ind.fitness.values = fit   

    avg_hist=[]
    best_hist=[]
    best_ind=[]

    hof=tools.HallOfFame(maxsize=5)
    hof.update(pop)
    


    CXPB, MUTPB = 0.5, 0.2
    gens = 0
    while gens < 25 :
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

    return hof,best_ind


def rotate_all(transform, t_min_deg):
    """
    Rotate arrays of origins (N,3) and directions (N,3) in the XY-plane
    by -t_min_deg degrees. Z is left unchanged.

    Parameters
    ----------
    origins : array-like, shape (N,3)
        Array of origins, each (x,y,0).
    directions : array-like, shape (N,3)
        Array of directions, each (dx,dy,0).
    t_min_deg : float
        Minimum angle in degrees; rotation is by -t_min_deg CCW.

    Returns
    -------
    origins_rot : np.ndarray, shape (N,3)
    directions_rot : np.ndarray, shape (N,3)
    """
    origins = transform.origin
    directions = transform.direction

    phi = np.deg2rad(-t_min_deg)  # CCW rotation by -t_min
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])  # 2x2 rotation matrix

    # apply rotation to x,y parts
    origins_xy = origins[:, :2] @ R.T
    dirs_xy = directions[:, :2] @ R.T

    # reassemble with z untouched
    origins_rot = np.column_stack([origins_xy, origins[:, 2]])
    directions_rot = np.column_stack([dirs_xy, directions[:, 2]])

    transform.set_origin(origins_rot)
    transform.set_direction(directions_rot)


def random_angle_test(transform):


    los_angles = np.array(
        [random_angle_avoiding_left(transform) for _ in range(8)]
    )
    los_angles=[59.63209549196224, 95.6216568731703, 266.8815031970812, 268.4670419152264, 308.5250168863993, 341.2164113018966, 342.5982764331459, 357.7043321523542]
    offsets=[-0.9581814903561077, 0.3645514201473765, 0.0, -0.1651885848622241, 0.3344766602479097, -0.41803296599972817, 0.18487535897572593, -0.0416023700388628]
    dirs=[]
    for angle, offset in zip(los_angles,offsets):
        dirs.append(direction_from_polar_and_dir_offset(angle,offset))

    #min_los_angle = np.min(los_angles)
    origin = transform.origin
    direction = transform.direction
    origin = np.delete(origin, [0, 1, 2, 3, 4, 5, 6, 7], axis=0)
    transform.set_origin(origin)
    direction = np.delete(direction, [0, 1, 2, 3, 4, 5, 6, 7], axis=0)
    transform.set_direction(direction)

    a=0
    for angle in los_angles:
        new_origin_x, new_origin_z = origin_from_polar_angle(angle, transform)
        transform.add_origin((new_origin_x, 0, new_origin_z))

        #new_dir_x, new_dir_z = random_feasible_direction_from_polar_angle(
        #    angle
        #)
        new_dir_x,new_dir_z=dirs[a]
        transform.add_direction((new_dir_x, 0, new_dir_z))
        a+=1

    #rotate_all(transform, min_los_angle)

    update_los(transform)


def _rect_center_and_extents(transform):
    x0 = transform._machine_dims[0][0]
    x1 = transform._machine_dims[0][1]
    z0 = transform._machine_dims[1][0]
    z1 = transform._machine_dims[1][1]
    cx = 0.5 * (x0 + x1)
    cz = 0.5 * (z0 + z1)
    ax = 0.5 * (x1 - x0)  # half-width x
    az = 0.5 * (z1 - z0)  # half-height z
    return cx, cz, ax, az

def _ray_to_rect_boundary(angle_deg, transform):
    cx, cz, ax, az = _rect_center_and_extents(transform)
    th = np.deg2rad(angle_deg)
    ux, uz = np.cos(th), np.sin(th)  # unit direction in x–z
    eps = 1e-12
    tx = ax / (abs(ux) + eps)
    tz = az / (abs(uz) + eps)
    t = min(tx, tz)
    return cx + t * ux, cz + t * uz


def random_angle_avoiding_left(transform, ):

    # Half-extents
    x0, x1 = transform._machine_dims[0]
    z0, z1 = transform._machine_dims[1]
    ax = 0.5 * abs(x1 - x0)  # half-width
    az = 0.5 * abs(z1 - z0)  # half-height
 
    # Angular half-width of the left-edge exclusion
    alpha = np.degrees(np.arctan2(az, ax))  # deg
    start = 180.0 - alpha
    end   = 180.0 + alpha
 
    # Allowed set: [0, start) U (end, 360)
    width1 = max(0.0, start - 0.0)
    width2 = max(0.0, 360.0 - end)
    total = width1 + width2
 
    if random.random() < (width1 / total):
        angle = random.uniform(0.0, start)
    else:
        angle = random.uniform(end, 360.0)
    return angle

def random_feasible_direction_from_polar_angle(angle):

    inward_direction = (angle + 180.0) % 360.0
    direction_angle = inward_direction + random.uniform(-65.0, 65.0)
    th = np.deg2rad(direction_angle)
    return np.cos(th), np.sin(th)  # (dx, dz)

def direction_from_polar_and_dir_offset(angle, dir_offset):

    inward_direction = (angle + 180.0) % 360.0
    direction_angle = inward_direction + 90.0 * float(dir_offset)
    th = np.deg2rad(direction_angle)
    return np.cos(th), np.sin(th)  # (dx, dz)

def origin_from_polar_angle(angle, transform):

    return _ray_to_rect_boundary(angle, transform)  # (x, z)


def update_los(transform):

    transform.x1 = list(np.arange(0, len(transform.origin)))

    transform.distribute_beamlets(debug=False)
    transform.set_dl(0.01)


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
 
def interactive_solution_timeslice_plot_from_list(solutions, init_solution=0):



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
    phantom, recon, artist_fn, mse = solutions[init_solution]
    t_vals = np.asarray(phantom.t)
    i0 = 0
    t0 = t_vals[i0]
 
    # --- figure & axes ---
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]}
    )
    plt.subplots_adjust(bottom=0.20)

    supt=fig.suptitle(f"Solution {init_solution}, MSE= {mse:.4e}")
 
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
        phantom_i, recon_i, artist_fn_i, mse_i = solutions[sol_idx]
 
        # update state
        state["phantom"] = phantom_i
        state["recon"]   = recon_i
        state["artist_fn"] = artist_fn_i
        state["mse"]= mse_i
        state["suptitle"].set_text(f"Solution {sol_idx},MSE={mse_i:.4e}")
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


def apply_individual_to_transform(individual, transform):
    """
    Genome layout: first half = direction offsets in [-1,1],
                   second half = angles in degrees [0,360).
    Recomputes ALL origins and directions and writes them to `transform`.
    """
    g = np.asarray(individual, dtype=float)
    n = g.size // 2
    dir_offsets = np.clip(g[:n], -1.0, 1.0)
    angles = (g[n:] % 360.0 + 360.0) % 360.0
 
    # Build origins (x,0,z) from angles
    origins = np.empty((n, 3), dtype=float)
    for i, ang in enumerate(angles):
        x, z = origin_from_polar_angle(ang, transform)
        origins[i] = (x, 0.0, z)
 
    # Build directions (dx,0,dz) from (angle, offset)
    directions = np.empty((n, 3), dtype=float)
    for i, (ang, off) in enumerate(zip(angles, dir_offsets)):
        dx, dz = direction_from_polar_and_dir_offset(ang, off)
        directions[i] = (dx, 0.0, dz)
 
    # Replace all LOS and update
    transform.set_origin(origins)
    transform.set_direction(directions)
    update_los(transform)
 
    return transform, directions, origins


 


    """

    Interactive plot with a slider to explore timeslices.
 
    Parameters

    ----------

    phantom_emission : xarray.DataArray

        True emission, with dims including 't' and 'rhop'.

    downsampled_inverted : xarray.DataArray

        Reconstructed emission, with dims including 't' and 'rhop'.

    """

    # Extract time coordinates as a NumPy array

    t_vals = np.asarray(phantom_emission.t)

    nT = len(t_vals)
 
    # Initial index/time

    i0 = 0

    t0 = t_vals[i0]
 
    # Create figure and axis

    fig, ax = plt.subplots()

    plt.subplots_adjust(bottom=0.18)  # leave space for slider
 
    # Initial plot

    (line_phantom,) = ax.plot(

        phantom_emission.rhop,

        phantom_emission.sel(t=t0, method="nearest"),

        label="Phantom",

    )

    (line_recon,) = ax.plot(

        downsampled_inverted.rhop,

        downsampled_inverted.sel(t=t0, method="nearest"),

        linestyle="dashed",

        label="Reconstructed",

    )

    ax.set_xlabel("rhop")

    ax.set_ylabel("emission")

    ax.legend()
 
    # Fix initial y-limits (optional)

    ymin = float(np.nanmin([phantom_emission.min(), downsampled_inverted.min()]))
    ymax = float(np.nanmax([phantom_emission.max(), downsampled_inverted.max()]))
    ax.set_ylim(ymin, ymax * 1.05)  # +5% headroom

    ax.set_title(f"t = {t0}")
 
    # Add slider

    ax_slider = fig.add_axes([0.15, 0.08, 0.7, 0.04])

    s_t = Slider(

        ax=ax_slider, label="t index",

        valmin=0, valmax=nT - 1,

        valinit=i0, valstep=1

    )
 
    # Update function

    def update(idx):

        idx = int(idx)

        tt = t_vals[idx]

        y_p = phantom_emission.sel(t=tt, method="nearest")

        y_r = downsampled_inverted.sel(t=tt, method="nearest")

        line_phantom.set_ydata(y_p)

        line_recon.set_ydata(y_r)

        ax.set_title(f"t = {tt}")

        fig.canvas.draw_idle()
 
    s_t.on_changed(update)
 
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


def get_solution(individual, transform, model, phantom_emission):
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
    def pick_geom(fig):
        return any(ax.get_xlabel() == "R [m]" for ax in fig.axes)
    
    geom_R_artist = grab_figure_as_image(lambda: transform.plot(), pick=pick_geom)
    mse, corr = reconstruction_metric(phantom_emission, downsampled_inverted)
    return (phantom_emission,downsampled_inverted,geom_R_artist,mse)


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




    # Run model and inversion
    bckc, phantom_emission = model(return_emissivity=True)
    los_count=3
    hof,bestPerGen=run_ga(los_count,model,phantom_emission)
    with open(f'fullrunHOF_{los_count}.pkl', 'wb') as file:
        # Dump data with highest protocol for best performance
        pickle.dump(hof, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'fullrunBESTOFGEN_{los_count}los.pkl', 'wb') as file:
        # Dump data with highest protocol for best performance
        pickle.dump(bestPerGen, file, protocol=pickle.HIGHEST_PROTOCOL)
    """
    with open('fullrunHOF.pkl','rb') as file:
        hof=pickle.load(file)
    best=hof[0]
    """
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


run_bm()
