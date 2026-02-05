# 17/02/23, Jon Wood
#
# set_global_conda
# source activate env_Indica
#
# &
#
# git checkout marcosertoli/st40

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


def main(
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
