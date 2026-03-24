import numpy as np

import os
from indica.converters import LineOfSightTransform
from indica.operators import NbiOperator
from indica.utilities import time_to_ms
from indica.configs.operators.fidasim_configs import (
    FIDASIM_OUTPUT_DIR,
    PLASMA_INTERP_GRID_SETTINGS,
    FIDASIM_FI_DIST_FILE,
    FIDASIM_BASE_DIR,
    SIMULATION_SWITCHES,
    MC_SETTINGS_FINE,
    WAVELENGTH_GRID_SETTINGS,
    WEIGHT_FUNCTION_SETTINGS,
    FIDASIM_BIN_PATH,
)
import shutil
import xarray as xr

import subprocess

import fidasim
from fidasim.utils import beam_grid
from fidasim.utils import rz_grid
import h5py as h5

SHAPE_MAP = {
    "rectangular": 1,
    "rectangle": 1,
    "square": 1,
    "round": 2,
    "circular": 2,
}
SHAPE_MAP_DEFAULT = 2


class NbiFidasim(NbiOperator):
    def prepare(
        self,
        fi_dist_file: str = FIDASIM_FI_DIST_FILE,
        save_dir: str = FIDASIM_OUTPUT_DIR,
        fida_dir: str = FIDASIM_BASE_DIR,
        mc_settings: dict = MC_SETTINGS_FINE,
        wavelength_grid_settings: dict = WAVELENGTH_GRID_SETTINGS,
        weight_function_settings: dict = WEIGHT_FUNCTION_SETTINGS,
        simulation_switches: dict = SIMULATION_SWITCHES,
    ):
        """
        Prepare NBI code input files translating Indica native
        quantities to what Fidasim expects
        """

        # File names and paths
        run_prefix = (
            f"{str(self.file_name).strip()}_{time_to_ms(self.t)}_ms_{self.name}"
        )
        run_dir = os.path.join(save_dir, run_prefix)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        beam_save_dir = os.path.join(run_dir, self.name)
        if not os.path.exists(beam_save_dir):
            os.makedirs(beam_save_dir)

        fidasim_out = os.path.join(beam_save_dir, f"{run_prefix}_inputs.dat")
        neut_file = os.path.join(beam_save_dir, f"{run_prefix}_neutrals.h5")

        # Store information necessary for run phase
        self.fidasim_out = fidasim_out
        self.neut_file = neut_file

        # Remove the existing folder if re-running fidasim
        # TODO: do we need a safety net if we don't want to overwrite?
        try:
            shutil.rmtree(run_dir)
        except FileNotFoundError:
            pass

        # Define plasma interpolation grid bounds and create beam grid from transform
        rmin = self.transform._machine_dims[0][0]
        rmax = self.transform._machine_dims[0][1]
        zmin = self.transform._machine_dims[1][0]
        zmax = self.transform._machine_dims[1][1]
        nr = PLASMA_INTERP_GRID_SETTINGS["nr"]
        nz = PLASMA_INTERP_GRID_SETTINGS["nz"]
        grid = rz_grid(rmin, rmax, nr, zmin, zmax, nz)
        bgrid, beam_cfg = create_grids(self.transform)

        # Map all quantities to the Fidasim 2D (R,z) grid
        equilibrium = self.transform.equilibrium
        _R = grid["r2d"][0, :]
        _z = grid["z2d"][:, 0]
        R = xr.DataArray(_R, coords={"R": _R})
        z = xr.DataArray(_z, coords={"z": _z})
        rhop_2d = equilibrium.rhop.interp(t=self.t).interp(R=R, z=z)
        rhot_2d, _ = self.equilibrium.convert_flux_coords(rhop_2d, t=self.t)
        br_2d, bz_2d, bt_2d, _ = self.equilibrium.Bfield(R, z, t=self.t, full_Rz=True)

        # Mask where plasma profiles are available
        max_rhop_profiles = np.max(self.Te.rhop)
        mask = xr.full_like(rhop_2d, 1)
        mask = xr.where(rhop_2d <= max_rhop_profiles, mask, 0)

        plasma = {
            "data_source": "Indica",
            "time": self.t,
            "zeff": self.Zeff.sel(rhop=rhop_2d).data,
            "ti": self.Ti.sel(rhop=rhop_2d).data * 1.0e-03,
            "te": self.Te.sel(rhop=rhop_2d).data * 1.0e-03,
            "denn": self.Nn.sel(rhop=rhop_2d).data * 1.0e-06,
            "dene": self.Ne.sel(rhop=rhop_2d).data * 1.0e-06,
            "vr": np.zeros_like(rhop_2d),
            "vz": np.zeros_like(rhop_2d),
            "vt": self.Vtor.sel(rhop=rhop_2d).data * 100.0,
            "mask": mask.data,
            "plasma_ion_amu": self.target_element_info["A"],
            "grid": grid,
            "flux": rhop_2d,
            "bgrid": bgrid,
        }

        # Add midplane profiles to plasma dictionary
        zmag = xr.full_like(R, self.equilibrium.zmag.sel(t=self.t).data)
        rhop_midplane = self.equilibrium.flux_coords(R, zmag, t=self.t)
        profiles_midplane = {
            "ti": self.Ti.interp(rhop=rhop_midplane).data * 1.0e-03,
            "te": self.Te.interp(rhop=rhop_midplane).data * 1.0e-03,
            "dene": self.Ne.interp(rhop=rhop_midplane).data * 1.0e-06,
            "denn": self.Nn.interp(rhop=rhop_midplane).data * 1.0e-06,
            "vt": self.Vtor.interp(rhop=rhop_midplane).data * 100.0,
            "rho": rhop_midplane,
            "r_omp": self.equilibrium.R.data * 100,
        }
        plasma["profiles"] = profiles_midplane

        # Create equilibrium dictionary
        # TODO: Electric field currently set to 0
        # TODO: Equilibrium class should have provenance info
        #       (e.g. EFIT) for "data_source"!!
        equil = {
            "time": self.t,
            "data_source": "Provenance to be implemented!",
            "br": br_2d,
            "bz": bz_2d,
            "bt": bt_2d,
            "er": np.zeros_like(br_2d),
            "ez": np.zeros_like(br_2d),
            "et": np.zeros_like(br_2d),
            "mask": xr.full_like(rhop_2d, 1),
        }

        # Read the dummy fast-ion distribution
        # TODO: this should be refactored to something sensible,
        #       but can leave this for later
        _fi_dist = h5.File(fi_dist_file, "r")
        fi_dist = dict()
        fi_dist["type"] = int(_fi_dist["type"][()])
        fi_dist["time"] = _fi_dist["time"][()]
        fi_dist["nenergy"] = int(_fi_dist["nenergy"][()])
        fi_dist["energy"] = _fi_dist["energy"][()]
        fi_dist["npitch"] = int(_fi_dist["npitch"][()])
        fi_dist["pitch"] = _fi_dist["pitch"][()]
        fbm_grid = np.zeros((fi_dist["nenergy"], fi_dist["npitch"], nr, nz))
        fi_dist["f"] = fbm_grid
        fi_dist["denf"] = _fi_dist["denf"][()]
        fi_dist["data_source"] = str(_fi_dist["data_source"][()])

        general_settings = {
            "device": self.machine,
            "shot": self.pulse,
            "time": self.t,
            "runid": run_prefix,
            "comment": "test",
            "result_dir": beam_save_dir,
            "tables_file": fida_dir + "/tables/atomic_tables.h5",
        }

        nbi_settings = {
            "einj": self.energy,
            "pinj": self.power,
            "current_fractions": np.array(self.current_fractions),
            "ab": self.nbi_element_info["A"],
        }
        plasma_settings = {
            "ai": self.target_element_info["A"],
            "impurity_charge": np.mean(self.MeanZ),
        }

        inputs = dict(general_settings)
        inputs.update(simulation_switches)
        inputs.update(mc_settings)
        inputs.update(nbi_settings)
        inputs.update(plasma_settings)
        inputs.update(wavelength_grid_settings)
        inputs.update(weight_function_settings)
        inputs.update(bgrid)

        fidasim.prefida(inputs, grid, beam_cfg, plasma, equil, fi_dist)

    def run(self, num_cores: int = 3):
        """
        Run beam code
        """

        subprocess.run(
            [
                FIDASIM_BIN_PATH,
                self.fidasim_out,
                f"{num_cores}",
            ]
        )

    def refactor_output(self) -> xr.Dataset:
        """
        This is the old fidasim_out_to_xarray_dataset function,
        changed to general name that matches the abstract Nbi operator

        # TODO: but what is this output exactly? Why to DataSet and
                not a DataArrays? what are its coordinates/dimensions?
                we'd like it as indica native as possible
        # TODO: make the output of this == dict as explained in the
                abstract method of the ABC class
        """

        if not os.path.exists(self.neut_file):
            raise FileNotFoundError(f"Neutrals file not found: {self.neut_file}")

        data_vars = {}
        with h5.File(self.neut_file, "r") as h5f:
            root_attrs = dict(h5f.attrs)

            # Visitor function. Check if object is a dataset, get the data,
            # build a safe variable name.
            def _visit(name, obj):
                if isinstance(obj, h5.Dataset):
                    data = obj[()]
                    var_name = name.replace("/", "__")
                    # Synthetic dimension names
                    dims = tuple(
                        f"{var_name}_dim_{i}" for i in range(getattr(data, "ndim", 0))
                    )
                    # Store the data to the xarray with the correct attributes
                    data_vars[var_name] = xr.DataArray(
                        data, dims=dims, attrs=dict(obj.attrs)
                    )

            # Recursive visitor function
            h5f.visititems(_visit)

        # Build the actual dataset
        neutrals = xr.Dataset(data_vars=data_vars, attrs=root_attrs)

        return neutrals


def create_grids(
    transform: LineOfSightTransform,
    delta_src=0.0,
    delta_ang=0.0,
):
    """
    Starting from Indica transforms create Fidasim beam grid
    TODO: Indica transform currently has only 1 focal length
    """
    _axis = np.array()
    norm = np.linalg.norm(_axis)
    if norm <= 0.0:
        raise ValueError("transform direction vector has zero norm")
    axis = _axis / norm

    try:
        shape = SHAPE_MAP[transform.spot_shape.lower()]
    except KeyError:
        shape = SHAPE_MAP_DEFAULT

    beam_cfg = {
        "data_source": "",
        "shape": shape,
        "src": 100 * np.array(transform.origin[0]),
        "axis": axis,
        "widy": 100 * transform.spot_width,
        "widz": 100 * transform.spot_height,
        "divy": transform.div_width[0],
        "divz": transform.div_height[0],
        "focy": 100.0 * transform.focal_length,
        "focz": 100.0 * transform.focal_length,
        "naperture": 0,  # Default for now
    }

    if beam_cfg["naperture"] > 0:
        raise NotImplementedError(
            "LineOfSightTransform doesn't have the necessary attributes"
        )

    # Optional source offset normal to beam axis in XY plane.
    if delta_src != 0.0:
        norm_angle = np.arctan2(beam_cfg["axis"][1], beam_cfg["axis"][0]) + np.pi / 2
        beam_cfg["src"][0] = beam_cfg["src"][0] + delta_src * np.cos(norm_angle) * 100.0
        beam_cfg["src"][1] = beam_cfg["src"][1] + delta_src * np.sin(norm_angle) * 100.0

    # Optional in-plane axis rotation.
    if delta_ang != 0.0:
        axis_angle = np.arctan2(beam_cfg["axis"][1], beam_cfg["axis"][0]) + delta_ang
        beam_cfg["axis"] = np.array(
            [np.cos(axis_angle), np.sin(axis_angle), beam_cfg["axis"][2]]
        )
        beam_cfg["axis"] = beam_cfg["axis"] / np.linalg.norm(beam_cfg["axis"])

    # TODO: Check that these make sense!!!
    rstart = transform._machine_dims[0][1]  # [cm]
    bgrid = beam_grid(
        beam_cfg,
        rstart,
        length=rstart * 2.5,
        width=rstart * 2.5,
        height=rstart / 2.0,
        dv=2.0,
    )

    return bgrid, beam_cfg
