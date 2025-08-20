from indica.defaults.load_defaults import load_default_objects
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data
from indica.operators import tomo_1D
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_parameter, centrifugal_asymmetry_2d_map
import numpy as np

import xarray
from xarray import DataArray
from typing import Callable

import matplotlib.pylab as plt

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
    #plt.close()
    #ax2=inverted_emissivity[2].plot()
    #plt.savefig("/home/jussi.hakosalo/Indica/indica/workflows/jussitesting/emissivity2.png")

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


    #Need to add to defaults
    transform.spot_shape="square"
    transform.focal_length= -1000.0

    bckc, emissivity = model(
        sum_beamlets=False,
        return_emissivity=True
    )

    #ax=emissivity[2].plot()
    #plt.savefig("/home/jussi.hakosalo/Indica/indica/workflows/jussitesting/emissivity.png")

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

    

    inverted,downsampled_inverted=calculate_tomo_inversion(transform,plasma,phantom_emission,emissivity)

    mse,corr=reconstruction_metric(emissivity,downsampled_inverted)





    #Now for the LOS change

    transform.add_origin((0.425,-1.245,0))
    transform.add_direction((-0.18,0.98,0))

    transform.add_origin((0.105,-1.045,0))
    transform.add_direction((0.18,0.98,0))


    transform.add_origin((0.115,-1.045,0))
    transform.add_direction((-0.18,0.98,0))

    transform.x1=list(np.arange(0,len(transform.origin)))

    #transform.set_dl(0.01)

    transform.distribute_beamlets(debug=False)
    transform.set_dl(0.01)


    inverted,downsampled_inverted=calculate_tomo_inversion(transform,plasma,phantom_emission,emissivity)

    mse2,corr2=reconstruction_metric(emissivity,downsampled_inverted)

    print(mse,mse2)
    print(corr,corr2)

    transform.plot(np.mean(0.02))
    plt.show()
    
    ata

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