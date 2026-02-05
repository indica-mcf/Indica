import copy
from typing import cast
from typing import List
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
from numpy.core.numeric import zeros_like
from pandas import DataFrame
import scipy
import xarray as xr
from xarray import DataArray


import subprocess
import json
import numpy as np
import os
import shutil

# Add path to Indica
import sys
sys.path.append("/home/jussi.hakosalo/Indica")

# Import fidasim module
import TE_fidasim_indica

PATH_TO_TE_FIDASIM = os.path.dirname(os.path.realpath(__file__))
print(f'PATH_TO_TE_FIDASIM = {PATH_TO_TE_FIDASIM}')


from indica.configs.readers.adasconf import ADF11
from indica.numpy_typing import LabeledArray
from indica.profilers.profiler_gauss import ProfilerGauss
from indica.readers.adas import ADASReader
from indica.utilities import DATA_PATH
from indica.utilities import set_plot_colors
from .abstractoperator import Operator




def run_tefida(
        shot_number: int,
        time: float,
        nbiconfig: dict,
        specconfig: dict,
        plasmaconfig: dict,
        num_cores=3,
        user="jussi.hakosalo",
        force_run_fidasim=False,
        save_dir="/home/jussi.hakosalo/fidasim_output",
):

    # Print inputs
    print(f'shot_number = {shot_number}')
    print(f'time = {time}')
    print(f'num_cores = {num_cores}')
    print(f'spec = {specconfig["name"]}')
    print(f'beam = {nbiconfig["name"]}')
    print(f'user = {user}')
    print(f'force_run_fidasim = {force_run_fidasim}')

    # Variables
    beam = nbiconfig["name"]

    # File paths
    beam_save_dir = save_dir + f'/{shot_number}/t_{time:0.6f}/{beam.lower()}'
    fidasim_out = save_dir + f'/{shot_number}/t_{time:0.6f}/{beam.lower()}/{user}_inputs.dat'

    # Remove the existing folder if re-running fidasim
    if force_run_fidasim:
        try:
            path_to_fidasim = save_dir + f'/{shot_number}/t_{time:0.6f}/{beam.lower()}'
            shutil.rmtree(path_to_fidasim)
            print('Remove ' + save_dir + f'/{shot_number}/t_{time:0.6f}/{beam.lower()}')
        except FileNotFoundError:
            print('No file ' + save_dir + f'/{shot_number}/t_{time:0.6f}/{beam.lower()}')

    # Run pre-processing code
    TE_fidasim_indica.prepare_fidasim(
        shot_number,
        time,
        nbiconfig,
        specconfig,
        plasmaconfig,
        save_dir=save_dir,
        plot_geo=False,
        fine_MC_res=True,
    )

    # Run FIDASIM (only on first instance - or if path provided)
    # -> /home/theory/FIDASIM/fidasim /home/bart.lomanowski/te-fidasim/output/9780/t_0.080000/rfx/bart.lomanowski_inputs.dat 8
    # /home/jonathan.wood/git_home/te-fidasim/output/10013/t_0.065000/rfx/jonathan.wood_spectra.h5

    # Check if this directory exists
    #fida_weights = beam_save_dir+f'/{user}_fida_weights.h5'
    #fidasim_exists = os.path.exists(fida_weights)
    #if (not fidasim_exists) or force_run_fidasim:

    print("=== Subprocess user check ===")
    subprocess.run(["whoami"])
    subprocess.run(["id"])
    print("=== End check ===")
    if force_run_fidasim:
        print('...     FIDASIM')
        subprocess.run(
            [
                "/home/jussi.hakosalo/fidasim/FIDASIM-2.0.0/fidasim",
                fidasim_out,
                f"{num_cores}"
            ]
        )

    # Run post-processing code
    results = TE_fidasim_indica.postproc_fidasim(
        shot_number,
        time,
        nbiconfig,
        specconfig,
        plasmaconfig,
        save_dir=save_dir,
        debug=False,
    )

    return results





class NBIOperator(Operator):

    """This operator should be operating on a standard plasma+profiles, and spit out
        fast neutral density and fast particle pressure.

        Now apart from the plasma, we need something to store the configs with.

        But how do we want to use this?
    """


    def __init__(
        self,
        plasma
    ):
        self.Plasma=plasma

    def __call__(self, a: DataArray, b: DataArray) -> DataArray:
        
        # Set-up FIDASIM run
        # Build beam configuration
        nbiconfig = {
            "name": "hnbi",
            "einj": 52.0,  # keV
            "pinj": 0.5,   # MW
            "current_fractions": [
                0.5,
                0.35,
                0.15
            ],
            "ab": 2.014
        }

        # specconfig
        chord_ids = [f"M{i + 1}" for i in range(np.shape(direction)[0])]
        geom_dict = dict()
        for i_chord, id in enumerate(chord_ids):
            geom_dict[id] = {}
            geom_dict[id]["origin"] = list(origin[i_chord, :] * 1e2)
            geom_dict[id]["diruvec"] = list(direction[i_chord, :])
        specconfig = {
            "chord_IDs": chord_ids,
            "geom_dict": geom_dict,
            "name": "TriWaSp_P2p4",
            "cross_section_corr": False,
        }

    # Loop over time
    for i_time, time in enumerate(plasma.t.data):
        rho_1d = plasma.ion_temperature.rhop.values
        ion_temperature = plasma.ion_temperature.sel(t=time).values
        electron_temperature = plasma.electron_temperature.sel(t=time).values
        electron_density = plasma.electron_density.sel(t=time).values
        neutral_density = plasma.neutral_density.sel(t=time).values
        toroidal_rotation = plasma.toroidal_rotation.sel(t=time).values
        zeffective = plasma.zeff.sum("element").sel(t=time).values

        print(f"rho_1d = {rho_1d}")
        print(f"ion_temperature = {ion_temperature}")
        print(f"electron_temperature = {electron_temperature}")
        print(f"electron_density = {electron_density}")
        print(f"neutral_density = {neutral_density}")
        print(f"toroidal_rotation = {toroidal_rotation}")
        print(f"zeffective = {zeffective}")

        # rho poloidal
        rho_2d = plasma.equilibrium.rhop.interp(
            t=time,
            method="nearest"
        )

        # rho toroidal
        rho_tor = plasma.equilibrium.convert_flux_coords(rho_2d, t=time)
        rho_tor = rho_tor[0].values

        # radius
        R = plasma.equilibrium.rhop.R.values
        z = plasma.equilibrium.rhop.z.values
        R_2d, z_2d = np.meshgrid(R, z)

        # Br
        br, _ = plasma.equilibrium.Br(
            plasma.equilibrium.rhop.R,
            plasma.equilibrium.rhop.z,
            t=time
        )
        br = br.values

        # Bz
        bz, _ = plasma.equilibrium.Bz(
            plasma.equilibrium.rhop.R,
            plasma.equilibrium.rhop.z,
            t=time
        )
        bz = bz.values

        # Bt
        # bt, _ = plasma.equilibrium.Bt(
        #     plasma.equilibrium.rhop.R,
        #     plasma.equilibrium.rhop.z,
        #     t=time
        # )
        # bt = bt.values  # NaN values an issue??
        irod = 3.0 * 1e6
        bt = irod * (4 * np.pi * 1e-7) / (2 * np.pi * R_2d)

        # rho
        rho = rho_2d.values

        # plasmaconfig
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

        print(f"plasmaconfig = {plasmaconfig}")

        # Run TE-fidasim
        run_fidasim = True
        path_to_code = "/home/jussi.hakosalo/te-fidasim"
        sys.path.append(path_to_code)

        results = run_tefida(
            pulse,
            time,
            nbiconfig,
            specconfig,
            plasmaconfig,
            num_cores=3,
            user="jussi.hakosalo",
            force_run_fidasim=run_fidasim,
            save_dir="/home/jussi.hakosalo/fidasim_output"
        )





