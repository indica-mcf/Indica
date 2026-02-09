import copy
from typing import cast
from typing import List
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
from numpy.core.numeric import zeros_like
from pandas import DataFrame
import scipy
from indica.operators import nbi_utils
import xarray as xr
from xarray import DataArray


import pickle
import subprocess
import json
import numpy as np
import os
import shutil
import sys
import pwd
import h5py as h5



PATH_TO_TE_FIDASIM = os.path.dirname(os.path.realpath(__file__))
print(f'PATH_TO_TE_FIDASIM = {PATH_TO_TE_FIDASIM}')


from indica.configs.readers.adasconf import ADF11
from indica.numpy_typing import LabeledArray
from indica.profilers.profiler_gauss import ProfilerGauss
from indica.readers.adas import ADASReader
from indica.utilities import DATA_PATH
from indica.utilities import set_plot_colors
from .abstractoperator import Operator
from .nbi_configs import FIDASIM_BIN_PATH
from .nbi_configs import FIDASIM_OUTPUT_DIR
from .nbi_configs import GEOMETRY_PKL_PATH
from .nbi_configs import NBI_USER
from .nbi_configs import TE_FIDASIM_CODE_PATH

# Flow/Config map:
# 1) NBIOperator takes a transform + nbispecs (=beam + spectrometer spec).
# 2) nbispecs (from nbi_configs.DEFAULT_NBI_SPECS or test overrides) supplies
#    beam operating params (einj/pinj/current_fractions/ab) and spec JSON path.
# 3) nbi_utils.prepare_fidasim builds FIDASIM inputs by combining:
#    - nbispecs (beam params; also picks beam name for geometry),
#    - specconfig JSON (diagnostic chords, so spectroscopy config. This should actually come with the CXS spec class.),
#    - plasmaconfig (equilibrium + profiles),
#    - global settings in nbi_configs.py (paths, MC settings, grids, switches),
#    - beam geometry from get_hnbi_geo/get_rfx_geo via create_st40_beam_grid.
# 4) Resulting inputs are written to FIDASIM_OUTPUT_DIR and run.



#From h5 to xarray. Still WIP.It doesn’t preserve original dimension names (only creates generic ones).
def _h5_to_xarray_dataset(h5_path: str) -> xr.Dataset:
    data_vars = {}
    with h5.File(h5_path, "r") as h5f:
        root_attrs = dict(h5f.attrs)

        #Visitor function. Check if object is a dataset, get the data,
        #Build a safe variable name
        def _visit(name, obj):
            if isinstance(obj, h5.Dataset):
                data = obj[()]
                var_name = name.replace("/", "__")
                #Synthetic dumension names
                dims = tuple(
                    f"{var_name}_dim_{i}" for i in range(getattr(data, "ndim", 0))
                )
                #Store the data to the xarray with the correct attrivbutes
                data_vars[var_name] = xr.DataArray(
                    data, dims=dims, attrs=dict(obj.attrs)
                )

        #Recursive visitor function
        h5f.visititems(_visit)
    #Build the actual dataset
    return xr.Dataset(data_vars=data_vars, attrs=root_attrs)


class NBIOperator(Operator):

    """This operator should be operating on a standard plasma+profiles, and spit out
        fast neutral density and fast particle pressure. I believe it does.
    """


    def __init__(
        self,
        transform,
        nbispecs,
    ):
        #Initialized with beam related info, so transform and geam parameters. Beam geometry comes later and 
        #through the configs instead.
        self.transform = transform


        #NBI and spectroscopy config
        self.nbispecs = nbispecs
        self.name = nbispecs.get("name")
        self.einj = nbispecs.get("einj")
        self.pinj = nbispecs.get("pinj")
        self.current_fractions = nbispecs.get("current_fractions")
        self.ab = nbispecs.get("ab")



        self.plasma_ion_amu=2.014






        origin = self.transform.origin
        direction = self.transform.direction
        x_pos = self.transform.origin_x
        y_pos = self.transform.origin_y



        #Spectroscopy config formatting using the trasnform
        chord_ids = [f"M{i + 1}" for i in range(np.shape(direction)[0])]
        geom_dict = {}
        for i_chord, id in enumerate(chord_ids):
            geom_dict[id] = {}
            geom_dict[id]["origin"] = list(origin[i_chord, :] * 1e2)
            geom_dict[id]["diruvec"] = list(direction[i_chord, :])
        self.specconfig = {
            "chord_IDs": chord_ids,
            "geom_dict": geom_dict,
            "name": nbispecs.get("spec_name"),
            "cross_section_corr": False,
            "spec_json_path": nbispecs.get("spec_json_path"),
        }





    def __call__(self, profiles, eqdata,pulse) -> dict:
 

        # Set-up FIDASIM run


        # Loop over time
        neutrals_by_time = {}
        for i_time, time in enumerate(profiles["t"].data):
            rho_1d = profiles["ion_temperature"].rhop.values
            ion_temperature = profiles["ion_temperature"].sel(t=time).values
            electron_temperature = profiles["electron_temperature"].sel(t=time).values
            electron_density = profiles["electron_density"].sel(t=time).values
            neutral_density = profiles["neutral_density"].sel(t=time).values
            toroidal_rotation = profiles["toroidal_rotation"].sel(t=time).values
            zeffective = profiles["zeff"].sum("element").sel(t=time).values




            # rho poloidal
            rho_2d = eqdata["rhop"].interp(
                t=time,
                method="nearest"
            )

            # rho toroidal
            rho_tor = eqdata["convert_flux_coords"](rho_2d, t=time)
            rho_tor = rho_tor[0].values

            # radius
            R = eqdata["rhop"].R.values
            z = eqdata["rhop"].z.values
            R_2d, z_2d = np.meshgrid(R, z)

            # Br
            br, _ = eqdata["Br"](
                eqdata["rhop"].R,
                eqdata["rhop"].z,
                t=time
            )
            br = br.values

            # Bz
            bz, _ = eqdata["Bz"](
                eqdata["rhop"].R,
                eqdata["rhop"].z,
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
                "plasma_ion_amu": self.plasma_ion_amu,
            }

            print(f"plasmaconfig = {plasmaconfig}")

            # Run TE-fidasim
            run_fidasim = True
            sys.path.append(TE_FIDASIM_CODE_PATH)


            # Print inputs
            print(f'shot_number = {pulse}')
            print(f'time = {time}')
            print('num_cores = 3')
            print(f'spec = {self.specconfig["name"]}')
            print(f'beam = {self.nbispecs["name"]}')
            print(f'user = {NBI_USER}')
            print(f'force_run_fidasim = {run_fidasim}')

            # Variables
            beam = self.nbispecs["name"]

            # File paths
            save_dir = FIDASIM_OUTPUT_DIR
            user = NBI_USER
            num_cores = 3
            fidasim_out = (
                save_dir
                + f'/{pulse}/t_{time:0.6f}/{beam.lower()}/{user}_inputs.dat'
            )

            # Remove the existing folder if re-running fidasim
            if run_fidasim:
                try:
                    path_to_fidasim = (
                        save_dir + f'/{pulse}/t_{time:0.6f}/{beam.lower()}'
                    )
                    shutil.rmtree(path_to_fidasim)
                    print(
                        'Remove ' + save_dir + f'/{pulse}/t_{time:0.6f}/{beam.lower()}'
                    )
                except FileNotFoundError:
                    print(
                        'No file ' + save_dir + f'/{pulse}/t_{time:0.6f}/{beam.lower()}'
                    )

            # Run pre-processing code
            #This takes in pulse nuber, time, the nbi configuration, the spectroscopy configuration, and plasma.
            nbi_utils.prepare_fidasim(
                pulse,
                time,
                self.nbispecs,
                self.specconfig,
                plasmaconfig,
                save_dir=save_dir,
                plot_geo=False,
                fine_MC_res=True,
            )

            print("ready to go after preprocessing")


            print("=== Subprocess user check ===")
            subprocess.run(["whoami"])
            subprocess.run(["id"])
            print("=== End check ===")
            if run_fidasim:
                print('...     FIDASIM')
                subprocess.run(
                    [
                        FIDASIM_BIN_PATH,
                        fidasim_out,
                        f"{num_cores}"
                    ]
                )


            runid = pwd.getpwuid(os.getuid())[0]
            time_str = "t_{:8.6f}".format(time)
            run_dir = save_dir + "/" + str(pulse) + "/" + time_str
            beam_save_dir = run_dir + "/" + beam
            neut_file = beam_save_dir + "/" + runid + "_neutrals.h5"

            if not os.path.exists(neut_file):
                raise FileNotFoundError(f"Neutrals file not found: {neut_file}")

            neutrals_by_time[float(time)] = {
                "path": neut_file,
                "data": _h5_to_xarray_dataset(neut_file),
            }



        return neutrals_by_time
