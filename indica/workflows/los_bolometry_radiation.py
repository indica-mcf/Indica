from indica.defaults.load_defaults import load_default_objects
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data
from indica.operators import tomo_1D
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_parameter, centrifugal_asymmetry_2d_map
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools

import xarray
from xarray import DataArray
from typing import Callable

import matplotlib.pylab as plt



def define_ga():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)


def rotate_origin_direction(origin,direction,min_angle):
    ox, oy, _ = origin
    dx, dy, _ = direction
 
    phi = np.deg2rad(-min_angle) 
    c, s = np.cos(phi), np.sin(phi)
 
    ox_new = c * ox - s * oy
    oy_new = s * ox + c * oy
 
    dx_new = c * dx - s * dy
    dy_new = s * dx + c * dy
 
    origin_rot = np.array([ox_new, oy_new, 0.0])
    direction_rot = np.array([dx_new, dy_new, 0.0])
 
    return origin_rot, direction_rot


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
    R = np.array([[c, -s],
                  [s,  c]])  # 2x2 rotation matrix
 
    # apply rotation to x,y parts
    origins_xy = origins[:, :2] @ R.T
    dirs_xy = directions[:, :2] @ R.T
 
    # reassemble with z untouched
    origins_rot = np.column_stack([origins_xy, origins[:, 2]])
    directions_rot = np.column_stack([dirs_xy, directions[:, 2]])
 
    transform.set_origin(origins_rot)
    transform.set_direction(directions_rot)




def random_angle_test(transform,machine_r):
    los_angles=np.array(360*np.random.rand(10,))
    min_los_angle=np.min(los_angles)
    origin=transform.origin
    direction=transform.direction
    origin=np.delete(origin,[0,1,2,3,4,5,6,7],axis=0)
    print(origin.shape)
    transform.set_origin(origin)
    direction=np.delete(direction,[0,1,2,3,4,5,6,7],axis=0)
    transform.set_direction(direction)


    for angle in los_angles:
        new_origin_x,new_origin_y=origin_from_polar_angle(angle,machine_r)
        transform.add_origin((new_origin_x,new_origin_y,0))

        new_dir_x,new_dir_y=random_feasible_direction_from_polar_angle(angle,machine_r)
        transform.add_direction((new_dir_x,new_dir_y,0))

    rotate_all(transform,min_los_angle)

    update_los(transform)

def random_feasible_direction_from_polar_angle(angle,machine_r):
    inward_direction=(angle+180)%360
    direction_angle=inward_direction+random.uniform(-60,60)
    print("dir ang:",direction_angle)
    return np.cos(np.deg2rad(direction_angle)),np.sin(np.deg2rad(direction_angle))


def origin_from_polar_angle(angle,machine_r):
    return machine_r*np.cos(np.deg2rad(angle)),machine_r*np.sin(np.deg2rad(angle))

def example_poloidal_asymmetry(plasma, equilibrium):
    PLASMA=plasma
    asymmetry_parameter = centrifugal_asymmetry_parameter(
        PLASMA.ion_density,
        PLASMA.ion_temperature,
        PLASMA.electron_temperature,
        PLASMA.toroidal_rotation,
        PLASMA.meanz,
        PLASMA.zeff,
        PLASMA.main_ion,
    )

    ion_density_2d = centrifugal_asymmetry_2d_map(
        PLASMA.ion_density,
        asymmetry_parameter,
        equilibrium,
    )

    return ion_density_2d

def update_los(transform):

    transform.x1=list(np.arange(0,len(transform.origin)))

    transform.distribute_beamlets(debug=False)
    transform.set_dl(0.01)

def reconstruction_metric(emissivity, downsampled_inverted):
        #Difference
    diff=emissivity-downsampled_inverted
    mse=(diff**2).mean(dim=("t","rhop"))

    corr=xarray.corr(emissivity.stack(points=("t","rhop")),downsampled_inverted.stack(points=("t","rhop")),dim="points")
    return mse,corr


def calculate_tomo_inversion(transform,plasma,phantom_emission,emissivity):
    PLASMA=plasma
    los_transform = transform
    los_transform.set_equilibrium(PLASMA.equilibrium)
    los_integral = los_transform.integrate_on_los(
        phantom_emission, phantom_emission.t.values
    )
    #print("phantom em")
    #print(phantom_emission.shape)

    z = los_transform.z.mean("beamlet")
    R = los_transform.R.mean("beamlet")
    dl = los_transform.dl
    has_data = np.logical_not(np.isnan(los_integral.isel(t=0).data))
    rho_equil = los_transform.equilibrium.rhop.interp(t=los_integral.t)
    input_dict = dict(
        brightness=los_integral.data.T,
        dl=dl,
        t=los_integral.t.data,
        R=R.data,
        z=z.data,
        rho_equil=dict(
            R=rho_equil.R.data,
            z=rho_equil.z.data,
            t=rho_equil.t.data,
            rho=rho_equil.data,
        ),
        has_data=has_data,
        debug=False,
    )
    if emissivity is not None:
        input_dict["emissivity"] = emissivity


    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=0.8)
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

    data_tomo = los_integral
    bckc_tomo = DataArray(tomo.backprojection.T, coords=data_tomo.coords)


    #This has 100 radials
    #print(inverted_emissivity)
    #print(inverted_emissivity.shape)
    #This has 41 radials
    #print(emissivity)
    #print(emissivity.shape)

    #Interpolate the 100 to 41
    downsampled_inverted=inverted_emissivity.interp(rhop=emissivity["rhop"])
    #print(downsampled_inverted.shape)
    return inverted_emissivity,downsampled_inverted


def get_phantom_emission(bckc,plasma,equilibrium,emissivity):

    los_integral=bckc["brightness"]
    #print(los_integral)
    PLASMA=plasma
    element: str = "ar"
    asymmetric_profile=False
    if asymmetric_profile:
        ion_density_2d = example_poloidal_asymmetry(plasma,equilibrium)
        emissivity = None
    else:
        rho_2d = PLASMA.equilibrium.rhop.interp(t=PLASMA.t.values)
        ion_density_2d = PLASMA.ion_density.interp(rhop=rho_2d)
        emissivity = emissivity
    imp_density_2d = ion_density_2d.sel(element=element).drop_vars("element")
    el_density_2d = PLASMA.electron_density.interp(rhop=imp_density_2d.rhop)
    lz_tot_2d = (
        PLASMA.lz_tot[element].sum("ion_charge").interp(rhop=imp_density_2d.rhop)
    )
    phantom_emission = el_density_2d * imp_density_2d * lz_tot_2d
    return phantom_emission




def run_example_diagnostic_model(
    machine: str, instrument: str, model: Callable, plot: bool = False, **kwargs
):
    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")

    plasma.set_equilibrium(equilibrium)
    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)

    model = model(instrument, **kwargs)
    model.set_transform(transform)
    model.set_plasma(plasma)

    machine_r=transform._machine_dims[0][1]
    

    #Need to add to defaults
    transform.spot_shape="square"
    transform.focal_length= -1000.0

    bckc, emissivity = model(
        sum_beamlets=False,
        return_emissivity=True
    )



    phantom_emission=get_phantom_emission(bckc,plasma,equilibrium,emissivity)

    
    ##ga_instance=define_ga()
    ##ga_instance.run()

    random_angle_test(transform,machine_r)
    #Fitness should be: redefine the transform and origin, run update transform function, then calculate tomo inversion,
    #fitness is the reconstruction metric.
    

    inverted,downsampled_inverted=calculate_tomo_inversion(transform,plasma,phantom_emission,emissivity)






    mse,corr=reconstruction_metric(emissivity,downsampled_inverted)
    print("MSE: ",mse)
    transform.plot(0.02)
    plt.show()






    if plot and hasattr(model, "plot"):
        plt.ioff()
        model.plot()
        plt.show()

    return plasma, model, emissivity

def run_bm():

    machine = "st40"
    instrument = "blom_xy1"
    _model = PinholeCamera
    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])

    return run_example_diagnostic_model(
        machine, instrument, _model, plot=False, power_loss=power_loss
    )

run_bm()