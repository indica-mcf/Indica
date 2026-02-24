import copy
import json
import os
import pickle
import pwd
import shutil
import subprocess
import sys
from typing import List
from typing import Tuple
from typing import cast

import h5py as h5
import matplotlib.pylab as plt
import numpy as np
import scipy
import xarray as xr
from numpy.core.numeric import zeros_like
from pandas import DataFrame
from xarray import DataArray

from indica.operators import nbi_utils



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
from .nbi_configs import NBI_USER
from .nbi_configs import TE_FIDASIM_CODE_PATH

# Flow/Config map:
# 1) NBIOperator takes a transform + nbispecs (=beam specs).
# 2) nbispecs (from nbi_configs.DEFAULT_NBI_SPECS or test overrides) supplies
#    beam operating params (einj/pinj/current_fractions/ab).
# 3) nbi_utils.prepare_fidasim builds FIDASIM inputs by combining:
#    - nbispecs (beam params; also picks beam name for geometry),
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
        name: str,
        einj: float,
        pinj: float,
        current_fractions: List[float],
        ab: float,
        selected_model: str = "FIDASIM",
        pulse: int = None,
        plasma_ion_amu: float = 2.014,
    ):
        # Initialized with beam related info; transform is set later.
        self.transform = None
        self.selected_model = selected_model
        self.pulse = pulse

        # NBI config
        self.name = name
        self.einj = einj
        self.pinj = pinj
        self.current_fractions = current_fractions
        self.ab = ab
        self.nbispecs = {
            "name": self.name,
            "einj": self.einj,
            "pinj": self.pinj,
            "current_fractions": self.current_fractions,
            "ab": self.ab,
        }

        self.plasma_ion_amu = plasma_ion_amu

    def __call__(
        self,
        ion_temperature=None,
        electron_temperature=None,
        electron_density=None,
        neutral_density=None,
        toroidal_rotation=None,
        zeff=None,
        t=None,
        pulse: int = None,
        plasma=None,
    ) -> dict:
        

        
        # Set-up FIDASIM run.
        if plasma is not None:
            self.plasma = plasma
        if self.plasma is not None:
            if ion_temperature is None:
                ion_temperature = self.plasma.ion_temperature
            if electron_temperature is None:
                electron_temperature = self.plasma.electron_temperature
            if electron_density is None:
                electron_density = self.plasma.electron_density
            if neutral_density is None:
                neutral_density = self.plasma.neutral_density
            if toroidal_rotation is None:
                toroidal_rotation = self.plasma.toroidal_rotation
            if zeff is None:
                zeff = self.plasma.zeff
            if t is None:
                t = getattr(self.plasma, "time_to_calculate", None)
                if t is None:
                    t = self.plasma.t

        if (
            ion_temperature is None
            or electron_temperature is None
            or electron_density is None
            or neutral_density is None
            or toroidal_rotation is None
            or zeff is None
        ):
            raise ValueError("Give inputs or assign plasma class!")

        if t is None:
            t = ion_temperature.t
        t_values = np.atleast_1d(getattr(t, "data", t))

        if pulse is None:
            pulse = self.pulse
        if pulse is None:
            raise ValueError("pulse is required (set it on init or pass to __call__)")

        if self.transform is None:
            raise ValueError("transform is required (set it before calling)")
        if not hasattr(self.transform, "equilibrium") or self.transform.equilibrium is None:
            raise ValueError("transform is missing equilibrium data")
        eq = self.transform.equilibrium

        profiles = {
            "t": t_values,
            "ion_temperature": ion_temperature,
            "electron_temperature": electron_temperature,
            "electron_density": electron_density,
            "neutral_density": neutral_density,
            "toroidal_rotation": toroidal_rotation,
            "zeff": zeff,
        }
        eqdata = {
            "rhop": eq.rhop,
            "convert_flux_coords": eq.convert_flux_coords,
            "Br": eq.Br,
            "Bz": eq.Bz,
        }

        neutrals_by_time = {}
        for i_time, time in enumerate(profiles["t"]):
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
            # equilibrium too (convert_flux_coordinates func)
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

            # this comes from eq inside transform. transform.eq.bfield


            irod = 3.0 * 1e6
            bt = irod * (4 * np.pi * 1e-7) / (2 * np.pi * R_2d)

            # rho
            # comes from eq too
            rho = rho_2d.values
            



            # From this point on, everything is FIDASIM specific. I should make more general!
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
            print(f'beam = {self.nbispecs["name"]}')
            print(f'user = {NBI_USER}')
            print(f'force_run_fidasim = {run_fidasim}')

            # Variables
            beam = self.nbispecs["name"]


            # this should be in the preparation
            # and generalizable to other beam models

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
            #This takes in pulse number, time, the nbi configuration, and plasma.
            nbi_utils.prepare_fidasim(
                pulse,
                time,
                self.nbispecs,
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
