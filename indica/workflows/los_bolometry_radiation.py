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


def evaluateIndividual(individual, model, machine_r, phantom_emission):
    transform = model.transform

    # Change origin and direction of lines of sight
    N = len(individual) // 2
    los_angles = individual[:N]
    min_los_angle = np.min(los_angles)
    offsets = individual[N:]
    directions = []
    origins = []
    for i in range(N):
        new_origin_x, new_origin_y = origin_from_polar_angle(los_angles[i], machine_r)
        origins.append((new_origin_x, new_origin_y, 0))
        new_dir_x, new_dir_y = direction_from_polar_and_dir_offset(
            los_angles[i], machine_r, offsets[i]
        )
        directions.append((new_dir_x, new_dir_y, 0))
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
    return float(mse)


def random_angle():
    return random.uniform(0.0, 360.0)


def random_offset():
    return random.uniform(-0.8, 0.8)


def define_ga(model, machine_r, number_of_los, phantom_emission):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_angle", random_angle)
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
            machine_r=machine_r,
            phantom_emission=phantom_emission,
        ),
    )
    return toolbox


def run_ga(number_of_los, machine_r, model, phantom_emission):
    toolbox = define_ga(model, machine_r, number_of_los, phantom_emission)
    pop = toolbox.population(n=30)
    fitnesses = list(map(toolbox.evaluate, pop))
    print(fitnesses)

    CXPB, MUTPB = 0.5, 0.2
    gens = 0
    while gens < 10:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

    ata


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


def random_angle_test(transform, machine_r):
    los_angles = np.array(
        360
        * np.random.rand(
            10,
        )
    )
    min_los_angle = np.min(los_angles)
    origin = transform.origin
    direction = transform.direction
    origin = np.delete(origin, [0, 1, 2, 3, 4, 5, 6, 7], axis=0)
    transform.set_origin(origin)
    direction = np.delete(direction, [0, 1, 2, 3, 4, 5, 6, 7], axis=0)
    transform.set_direction(direction)

    for angle in los_angles:
        new_origin_x, new_origin_y = origin_from_polar_angle(angle, machine_r)
        transform.add_origin((new_origin_x, new_origin_y, 0))

        new_dir_x, new_dir_y = random_feasible_direction_from_polar_angle(
            angle, machine_r
        )
        transform.add_direction((new_dir_x, new_dir_y, 0))

    rotate_all(transform, min_los_angle)

    update_los(transform)


def random_feasible_direction_from_polar_angle(angle, machine_r):
    inward_direction = (angle + 180) % 360
    direction_angle = inward_direction + random.uniform(-85, 85)
    return np.cos(np.deg2rad(direction_angle)), np.sin(np.deg2rad(direction_angle))


def direction_from_polar_and_dir_offset(angle, machine_r, dir_offset):
    inward_direction = (angle + 180) % 360
    direction_angle = inward_direction + 90 * dir_offset
    return np.cos(np.deg2rad(direction_angle)), np.sin(np.deg2rad(direction_angle))


def origin_from_polar_angle(angle, machine_r):
    return machine_r * np.cos(np.deg2rad(angle)), machine_r * np.sin(np.deg2rad(angle))


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

    # Run model and inversion
    bckc, phantom_emission = model(return_emissivity=True)

    ##ga_instance=define_ga()
    ##ga_instance.run()

    # run_ga(8,machine_r,model,phantom_emission)

    # random_angle_test(transform,machine_r)
    # Fitness should be: redefine the transform and origin, run update transform function, then calculate tomo inversion,
    # fitness is the reconstruction metric.

    downsampled_inverted = calculate_tomo_inversion(
        bckc["brightness"], transform, phantom_emission.rhop
    )

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
    transform.plot(0.02)
    plt.show()

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


# run_bm()
