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


def _h5_to_xarray_dataset(h5_path: str) -> xr.Dataset:
    data_vars = {}
    with h5.File(h5_path, "r") as h5f:
        root_attrs = dict(h5f.attrs)

        def _visit(name, obj):
            if isinstance(obj, h5.Dataset):
                data = obj[()]
                dims = tuple(f"dim_{i}" for i in range(getattr(data, "ndim", 0)))
                var_name = name.replace("/", "__")
                data_vars[var_name] = xr.DataArray(
                    data, dims=dims, attrs=dict(obj.attrs)
                )

        h5f.visititems(_visit)

    return xr.Dataset(data_vars=data_vars, attrs=root_attrs)


class NBIOperator(Operator):

    """This operator should be operating on a standard plasma+profiles, and spit out
        fast neutral density and fast particle pressure.

        Now apart from the plasma, we need something to store the configs with.

        But how do we want to use this?
    """

    #Todo: clarify init vs call arguments and processing

    def __init__(
        self,
        plasma,
        transform,
    
    ):
        self.plasma=plasma
        self.transform=transform

        # Plasma ion mass
        self.plasma_ion_amu = 2.014

    def __call__(self, pulse) -> dict:
        plasma=self.plasma
 
        tws_geom = pickle.load(open(GEOMETRY_PKL_PATH, 'rb'))

        #TODO: all this needs to be in the init, through transform
        focal_length = -0.03995269  # meter
        spot_width = 1.1 * 1e-3  # meter
        spot_height = 1.1 * 1e-3  # meter
        origin = tws_geom['origin']
        direction = tws_geom['direction']
        x_pos = tws_geom['x_pos']
        y_pos = tws_geom['y_pos']
        # Set-up FIDASIM run
        # Build beam configuration

        """
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
        """

        # Loop over time
        neutrals_by_time = {}
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
            sys.path.append(TE_FIDASIM_CODE_PATH)

            # Print inputs
            print(f'shot_number = {pulse}')
            print(f'time = {time}')
            print('num_cores = 3')
            print(f'spec = {specconfig["name"]}')
            print(f'beam = {nbiconfig["name"]}')
            print(f'user = {NBI_USER}')
            print(f'force_run_fidasim = {run_fidasim}')

            # Variables
            beam = nbiconfig["name"]

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
            nbi_utils.prepare_fidasim(
                pulse,
                time,
                nbiconfig,
                specconfig,
                plasmaconfig,
                save_dir=save_dir,
                plot_geo=False,
                fine_MC_res=True,
            )

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
