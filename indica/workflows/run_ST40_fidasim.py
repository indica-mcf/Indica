# To run this function, please first be on smaug 2, and then...
#
# git checkout jmwood94-0/107_add_neutral_beam in your Indica repo
#
# cd to this script
#
# set_global_conda
# conda activate env_Indica
# source /home/theory/ascot/ascot5.3_env
# [Edit path_to_indica to your path]
# python run_ST40_neutral_beam
#

from matplotlib import pyplot as plt
import numpy as np
import sys

# Add Indica to python path
path_to_indica = '/home/jonathan.wood/git_home/Indica/'
sys.path.append(path_to_indica)

# Import Indica modules
from indica.models import NeutralBeam, ChargeExchange
from indica.models.plasma import example_run
from indica.models.plasma import Plasma
from indica.converters import LineOfSightTransform

# Import MDS+
from MDSplus import *

# Import package - ToDo: move into the Indica
PATH_TO_CODE = '/home/jonathan.wood/git_home/te-fidasim'
import sys
sys.path.append(PATH_TO_CODE)
import fidasim_ST40_indica


# Function for running TE-fidasim forward model from Indica framework
def driver(
    cxspec: ChargeExchange,
    beam: NeutralBeam,
    beam_on: np.ndarray,
    run_fidasim: bool = True,
    which_spectrometer: str = "Princeton",
    quiet: bool = False
):

    # Pulse number
    pulse = beam.plasma.pulse

    # Build beam configuration
    nbiconfig = {
        "name": beam.name,
        "einj": beam.energy * 1e-3,
        "pinj": beam.power * 1e-6,
        "current_fractions": [beam.fractions[0], beam.fractions[1], beam.fractions[2]],
        "ab": beam.amu
    }

    # Build spectrometer configuration
    if which_spectrometer == "Princeton":
        specconfig = {
            "name": which_spectrometer,
            "chord_IDs": ["M3", "M4", "M5", "M6", "M7", "M8"],
            "cross_section_corr": True
        }
    elif which_spectrometer == "Chers_new":
        specconfig = {
            "name": which_spectrometer,
            "chord_IDs": ["M1", "M2", "M3"],
            "cross_section_corr": False
        }
    else:
        raise ValueError(f'{which_spectrometer} is not available')

    # Add geom_dict and chord_IDs to specconfig
    origin = cxspec.los_transform.origin
    direction = cxspec.los_transform.direction
    chord_ids = [f"M{i+1}" for i in range(np.shape(direction)[0])]
    geom_dict = dict()
    for i_chord, id in enumerate(chord_ids):
        geom_dict[id] = {}
        geom_dict[id]["origin"] = origin[i_chord, :]
        geom_dict[id]["diruvec"] = direction[i_chord, :]
    specconfig["chord_IDs"] = chord_ids
    specconfig["geom_dict"] = geom_dict

    # Atomic mass of plasma ion
    if beam.plasma.main_ion == 'h':
        plasma_ion_amu = 1.00874
    elif beam.plasma.main_ion == 'd':
        plasma_ion_amu = 2.014
    else:
        raise ValueError('Plasma ion must be Hydrogen "h" or Deuterium "d"')

    # Times to analyse
    times = beam.plasma.t.data

    Ti = np.zeros((len(specconfig["chord_IDs"]), len(times))) * np.nan
    Ti_err = np.zeros((len(specconfig["chord_IDs"]), len(times))) * np.nan
    vtor = np.zeros((len(specconfig["chord_IDs"]), len(times))) * np.nan
    vtor_err = np.zeros((len(specconfig["chord_IDs"]), len(times))) * np.nan
    for i_time, time in enumerate(times):
        if beam_on[i_time]:
            # Extract data from plasma / equilibrium objects
            # profiles
            rho_1d = beam.plasma.ion_temperature.coords["rho_poloidal"]
            ion_temperature = beam.plasma.ion_temperature.sel(element='c', t=time).values
            electron_temperature = beam.plasma.electron_temperature.sel(t=time).values
            electron_density = beam.plasma.electron_density.sel(t=time).values
            neutral_density = beam.plasma.neutral_density.sel(t=time).values
            toroidal_rotation = beam.plasma.toroidal_rotation.sel(element='c', t=time).values
            zeffective = beam.plasma.zeff.sum("element").sel(t=time).values

            # magnetic data
            # rho poloidal
            rho_2d = beam.plasma.equilibrium.rho.interp(
                t=time,
                method="nearest"
            )

            # rho toroidal
            rho_tor, _ = beam.plasma.equilibrium.convert_flux_coords(rho_2d, t=time)
            rho_tor = rho_tor.values  # NaN's is this going to be an issue?

            # radius
            R = beam.plasma.equilibrium.rho.coords["R"].values

            # vertical position
            z = beam.plasma.equilibrium.rho.coords["z"].values

            # meshgrid
            R_2d, z_2d = np.meshgrid(R, z)

            # Br
            br, _ = beam.plasma.equilibrium.Br(
                beam.plasma.equilibrium.rho.coords["R"],
                beam.plasma.equilibrium.rho.coords["z"],
                t=time
            )
            br = br.values

            # Bz
            bz, _ = beam.plasma.equilibrium.Bz(
                beam.plasma.equilibrium.rho.coords["R"],
                beam.plasma.equilibrium.rho.coords["z"],
                t=time
            )
            bz = bz.values

            # Bt, ToDo: returns NaNs!!
            #bt, _ = beam.plasma.equilibrium.Bt(
            #    beam.plasma.equilibrium.rho.coords["R"],
            #    beam.plasma.equilibrium.rho.coords["z"],
            #    t=time
            #)
            #bt = bt.values

            # Bypass bug -> irod = 2*pi*R * BT / mu0_fiesta;
            irod = 3.0 * 1e6
            bt = irod * (4 * np.pi * 1e-7) / (2 * np.pi * R_2d)

            # rho
            rho = rho_2d.values

            if False:
                # Plot magnetic fields
                plt.figure()
                plt.subplot(131)
                plt.contourf(
                    beam.plasma.equilibrium.rho.coords["R"],
                    beam.plasma.equilibrium.rho.coords["z"],
                    br,
                )
                plt.subplot(132)
                plt.contourf(
                    beam.plasma.equilibrium.rho.coords["R"],
                    beam.plasma.equilibrium.rho.coords["z"],
                    br,
                )
                plt.subplot(133)
                plt.contourf(
                    beam.plasma.equilibrium.rho.coords["R"],
                    beam.plasma.equilibrium.rho.coords["z"],
                    br,
                )
                plt.show(block=True)

            plasmaconfig = {
                "R": R_2d,
                "z": z_2d,
                "rho_1d": rho_1d,
                "rho": rho,
                "rho_t": rho_tor,
                "br": br,
                "bz": bz,
                "bt": bt,
                "ti": ion_temperature,
                "te": electron_temperature,
                "nn": neutral_density,
                "ne": electron_density,
                "omegator": toroidal_rotation,
                "zeff": zeffective,
                "plasma_ion_amu": plasma_ion_amu,
            }

            if not quiet:
                # Print statements
                print(f'pulse = {pulse}')
                print(f'time = {time}')
                print(f'nbiconfig = {nbiconfig}')
                print(f'specconfig = {specconfig}')
                print(f'plasmaconfig = {plasmaconfig}')
                print(f'plasma_ion_amu = {plasma_ion_amu}')
                print(f'run_fidasim = {run_fidasim}')

            # Run TE-fidasim
            results = fidasim_ST40_indica.main(
                pulse,
                time,
                nbiconfig,
                specconfig,
                plasmaconfig,
                num_cores=3,
                user="jonathan.wood",
                force_run_fidasim=run_fidasim,
                save_dir="/home/jonathan.wood/fidasim_output",
            )
            Ti[:, i_time] = results["Ti"]
            Ti_err[:, i_time] = results["Ti_err"]
            vtor[:, i_time] = results["vtor"]
            vtor_err[:, i_time] = results["vtor_err"]

    outdata = dict()
    outdata['time'] = times
    outdata['Ti'] = Ti
    outdata['Ti_err'] = Ti_err
    outdata['vtor'] = vtor
    outdata['vtor_err'] = vtor_err

    return outdata


# Test
if __name__ == "__main__":

    # Inputs
    pulse = 10014
    tstart = 0.04
    tend = 0.08
    dt = 0.01

    # Get plasma
    plasma = example_run(tstart=tstart, tend=tend, dt=dt, pulse=pulse)
    print(f'plasma = {plasma}')
    print(f'plasma.machine_dimensions = {plasma.machine_dimensions}')

    # Read CX spectrometer geometry
    which_spectrometer = 'Princeton'

    if which_spectrometer == 'Princeton':
        geom_pulse = 99000001
        tree = 'CAL_PI'
    elif which_spectrometer == 'Chers_new':
        geom_pulse = 99000001
        tree = 'CAL_TWS_C'  # this is P2.3 data!!
    else:
        raise ValueError('Whooops')

    t = Tree(tree, geom_pulse)
    origin = t.getNode('GEOMETRY.BEST.LOCATION').data()
    direction = t.getNode('GEOMETRY.BEST.DIRECTION').data()
    t.close()

    if which_spectrometer == 'Princeton':
        i_fibres = np.arange(21, 27, 1, dtype=int)
    elif which_spectrometer == 'Chers_new':
        i_fibres = np.arange(0, 3, 1, dtype=int)
    else:
        raise ValueError('Im pretty sure it is impossible to get here ;)')

    origin = origin[i_fibres, :]
    direction = direction[i_fibres, :]

    # make LineOfSightTransform
    los = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name=which_spectrometer,
        machine_dimensions=plasma.machine_dimensions,
    )

    # Create CX Spectrometer class
    cxspec = ChargeExchange(name=which_spectrometer)
    cxspec.set_los_transform(los)

    # Generate neutral beam model
    beam = NeutralBeam()
    beam.set_plasma(plasma)

    # Add beam input data here?
    beam_on = np.ones(len(beam.plasma.t.data))

    # Run driver function
    run_fidasim = False
    results = driver(
        cxspec,
        beam,
        beam_on,
        run_fidasim=run_fidasim,
        which_spectrometer=which_spectrometer,
    )

    print(results)


