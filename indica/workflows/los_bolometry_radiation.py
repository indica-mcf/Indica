from functools import partial
import random
from typing import Callable
import warnings

from deap import base
from deap import creator
from deap import tools
import matplotlib.pylab as plt
import numpy as np
import xarray
from xarray import DataArray

from indica.defaults.load_defaults import load_default_objects
from indica.models import PinholeCamera
from indica.operators import tomo_1D
from indica.operators.atomic_data import default_atomic_data

warnings.simplefilter(action="ignore", category=FutureWarning)


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
        rotate_all(transform, min_los_angle)
        update_los(transform)

        # Re-run model and calculate inversion
        bckc = model()
        downsampled_inverted = calculate_tomo_inversion(
            bckc["brightness"], transform, phantom_emission.rhop
        )

        mse, corr = reconstruction_metric(phantom_emission, downsampled_inverted)
        # print("Inidividual MSE: ",mse)
        return (float(mse),)
    except ValueError:
        print("The error")
        return (1e13,)


def random_angle():
    return random.uniform(0.0, 360.0)


def random_offset():
    return random.uniform(-0.99, 0.9)


def define_ga(model, number_of_los, phantom_emission):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_angle", partial(random_angle_avoiding_left,transform=model.transform))
    toolbox.register("attr_offset", random_offset)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_angle,) * number_of_los + (toolbox.attr_offset,) * number_of_los,
        n=1,
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



def run_ga(number_of_los, model, phantom_emission):
    toolbox = define_ga(model, number_of_los, phantom_emission)
    pop = toolbox.population(n=50)
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
    while gens < 20 :
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

    return hof[0]


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

    corr = xarray.corr(
        emissivity.stack(points=("t", "rhop")),
        downsampled_inverted.stack(points=("t", "rhop")),
        dim="points",
    )
    return mse, corr


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

    best=run_ga(8,model,phantom_emission)


    #Best individual to a transform
    print(f"Best individual, to be applied: {best}")
    transform=apply_individual_to_transform(best,model.transform)
        # Re-run model and calculate inversion
    bckc = model()

    downsampled_inverted = calculate_tomo_inversion(
        bckc["brightness"], transform, phantom_emission.rhop
    )


    #then set and recalculate?
    


 #   random_angle_test(transform)

    transform.plot()
    plt.show()


    for t in phantom_emission.t:
        plt.plot(phantom_emission.rhop, phantom_emission.sel(t=t), label="Phantom")
        plt.plot(
            downsampled_inverted.rhop,
            downsampled_inverted.sel(t=t),
            linestyle="dashed",
            label="Reconstructed",
        )
        plt.legend()
        plt.show()

    mse, corr = reconstruction_metric(phantom_emission, downsampled_inverted)
    print("MSE: ", mse)


    if plot and hasattr(model, "plot"):
        plt.ioff()
        model.plot()
        plt.show()

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
