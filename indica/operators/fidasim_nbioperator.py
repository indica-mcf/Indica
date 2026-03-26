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
from fidasim.utils import uvw_to_xyz
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

        # FIDASIM naming convention from preprocessing.py/check_inputs:
        #   <runid>_inputs.dat, <runid>_neutrals.h5, <runid>_distribution.h5
        # These are the same artifacts used directly in the legacy
        # TriWaSp workflow (forward_model_triwasp_P2p4.py).
        fidasim_out = os.path.join(beam_save_dir, f"{run_prefix}_inputs.dat")
        neut_file = os.path.join(beam_save_dir, f"{run_prefix}_neutrals.h5")
        distribution_file = os.path.join(beam_save_dir, f"{run_prefix}_distribution.h5")

        # Store information necessary for run/refactor phase
        self.fidasim_out = fidasim_out
        self.neut_file = neut_file
        self.distribution_file = distribution_file

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




        # Persist beam-grid transform terms (alpha/beta/gamma/origin) produced
        # by create_grids/FIDASIM beam_grid.
        self._beam_grid = dict(bgrid)
        # Precompute coordinate/refinement mappings now, while all geometry and
        # equilibrium context is already in-memory (TriWaSp workflow principle).
        # This keeps refactor_output focused on reading outputs + binning only.
        self._cache_refactor_mappings(grid=grid, bgrid=bgrid)

        

        # Map all quantities to the Fidasim 2D (R,z) grid
        equilibrium = self.transform.equilibrium
        _R = grid["r2d"][0, :]
        _z = grid["z2d"][:, 0]
        R = xr.DataArray(_R, coords={"R": _R})
        z = xr.DataArray(_z, coords={"z": _z})
        rhop_2d = equilibrium.rhop.interp(t=self.t).interp(R=R, z=z)
        rhot_2d, _ = self.transform.equilibrium.convert_flux_coords(rhop_2d, t=self.t)
        br_2d, bz_2d, bt_2d, _ = self.transform.equilibrium.Bfield(R, z, t=self.t, full_Rz=True)

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
        zmag = xr.full_like(R, self.transform.equilibrium.zmag.sel(t=self.t).data)
        rhop_midplane = self.transform.equilibrium.flux_coords(R, zmag, t=self.t)
        profiles_midplane = {
            "ti": self.Ti.interp(rhop=rhop_midplane).data * 1.0e-03,
            "te": self.Te.interp(rhop=rhop_midplane).data * 1.0e-03,
            "dene": self.Ne.interp(rhop=rhop_midplane).data * 1.0e-06,
            "denn": self.Nn.interp(rhop=rhop_midplane).data * 1.0e-06,
            "vt": self.Vtor.interp(rhop=rhop_midplane).data * 100.0,
            "rho": rhop_midplane,
            "r_omp": self.transform.equilibrium.R.data * 100,
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





    @staticmethod
    def _cell_widths(coord_1d: np.ndarray) -> np.ndarray:
        """Approximate cell widths from cell-center coordinates.

        FIDASIM outputs store cell-center axes (x, y, z), so we reconstruct
        local widths via neighbor distances for volume-weighted binning.
        """
        coord = np.asarray(coord_1d, dtype=float)
        if coord.size == 1:
            return np.ones(1, dtype=float)

        widths = np.empty_like(coord)
        widths[1:-1] = 0.5 * (coord[2:] - coord[:-2])
        widths[0] = coord[1] - coord[0]
        widths[-1] = coord[-1] - coord[-2]
        return np.abs(widths)

    @staticmethod
    def _rhop_bin_average(
        values: np.ndarray,
        rhop_values: np.ndarray,
        weights: np.ndarray,
        rhop_centers: np.ndarray,
    ) -> np.ndarray:
        """Volume-weighted average of values on rhop bins.

        This is the core reduction step that turns FIDASIM grid fields into
        the Indica-native 1D profile expected by the NBI abstract interface.
        """
        rhop = np.asarray(rhop_centers, dtype=float)
        if rhop.size == 1:
            out = np.full(1, np.nan, dtype=float)
            mask = (
                np.isfinite(values)
                & np.isfinite(rhop_values)
                & np.isfinite(weights)
                & (weights > 0.0)
            )
            if np.any(mask):
                out[0] = np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])
            return out

        mids = 0.5 * (rhop[1:] + rhop[:-1])
        edges = np.empty(rhop.size + 1, dtype=float)
        edges[1:-1] = mids
        edges[0] = rhop[0] - (mids[0] - rhop[0])
        edges[-1] = rhop[-1] + (rhop[-1] - mids[-1])

        idx = np.digitize(rhop_values, edges) - 1
        mask = (
            np.isfinite(values)
            & np.isfinite(rhop_values)
            & np.isfinite(weights)
            & (weights > 0.0)
            & (idx >= 0)
            & (idx < rhop.size)
        )

        num = np.bincount(
            idx[mask], weights=values[mask] * weights[mask], minlength=rhop.size
        )
        den = np.bincount(idx[mask], weights=weights[mask], minlength=rhop.size)

        out = np.full(rhop.size, np.nan, dtype=float)
        valid = den > 0.0
        out[valid] = num[valid] / den[valid]
        return out

    def _cache_refactor_mappings(self, grid: dict, bgrid: dict):
        """Precompute geometry->rhop mappings used by refactor_output.

        Provenance
        ----------
        This follows the legacy TriWaSp workflow intent: geometry transforms are
        determined once, and output files are then reduced using that mapping.
        """
        rhop = np.asarray(self.Nn.coords["rhop"].data, dtype=float)
        self._rhop_refactor = rhop

        eq_rhop = self.transform.equilibrium.rhop.interp(t=self.t)

        # Beam-grid (uvw in cm) -> machine (R,z in m) for neutral density maps.
        x_cm = np.linspace(float(bgrid["xmin"]), float(bgrid["xmax"]), int(bgrid["nx"]))
        y_cm = np.linspace(float(bgrid["ymin"]), float(bgrid["ymax"]), int(bgrid["ny"]))
        z_cm = np.linspace(float(bgrid["zmin"]), float(bgrid["zmax"]), int(bgrid["nz"]))
        z3, y3, x3 = np.meshgrid(z_cm, y_cm, x_cm, indexing="ij")
        uvw = np.vstack((x3.ravel(), y3.ravel(), z3.ravel()))

        xyz = uvw_to_xyz(
            float(bgrid["alpha"]),
            float(bgrid["beta"]),
            float(bgrid["gamma"]),
            uvw,
            origin=np.asarray(bgrid["origin"], dtype=float),
        )
        r_m = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2) * 1.0e-2
        z_m = xyz[2] * 1.0e-2
        self._neutral_rhop_flat = eq_rhop.interp(
            R=xr.DataArray(r_m, dims=("point",)),
            z=xr.DataArray(z_m, dims=("point",)),
        ).data

        dx_cm = self._cell_widths(x_cm)
        dy_cm = self._cell_widths(y_cm)
        dz_cm = self._cell_widths(z_cm)
        self._neutral_weights_flat = (
            dz_cm[:, None, None] * dy_cm[None, :, None] * dx_cm[None, None, :]
        ).ravel() * 1.0e-6

        # Interpolation grid (R,z) -> rhop for fast-ion density denf map.
        r2d = np.asarray(grid["r2d"], dtype=float)
        z2d = np.asarray(grid["z2d"], dtype=float)
        self._fast_rhop_flat = eq_rhop.interp(
            R=xr.DataArray((r2d * 1.0e-2).ravel(), dims=("point",)),
            z=xr.DataArray((z2d * 1.0e-2).ravel(), dims=("point",)),
        ).data

        r_axis = np.asarray(grid["r"], dtype=float)
        z_axis = np.asarray(grid["z"], dtype=float)
        dr_m = self._cell_widths(r_axis) * 1.0e-2
        dz_m = self._cell_widths(z_axis) * 1.0e-2

        if r2d.shape == (z_axis.size, r_axis.size):
            area = dz_m[:, None] * dr_m[None, :]
        elif r2d.shape == (r_axis.size, z_axis.size):
            area = dr_m[:, None] * dz_m[None, :]
        else:
            # Fallback for unexpected orientation; use mean cell sizes.
            area = np.full_like(r2d, float(np.mean(dr_m) * np.mean(dz_m)))

        self._fast_weights_flat = (np.abs(r2d) * 1.0e-2 * area).ravel()

    def _neutral_density_profile(self) -> np.ndarray:
        """Map FIDASIM neutral densities to rhop using precomputed mapping.

        Provenance
        ----------
        Mirrors the legacy TriWaSp usage pattern:
        - use fdens/hdens/tdens ground state (index 0),
        - convert cm^-3 -> m^-3,
        - reduce with precomputed geometry->rhop weights.
        """
        with h5.File(self.neut_file, "r") as h5f:
            fdens = np.asarray(h5f["fdens"][...], dtype=float)
            hdens = np.asarray(h5f["hdens"][...], dtype=float)
            tdens = np.asarray(h5f["tdens"][...], dtype=float)

        neutral_m3 = (fdens[..., 0] + hdens[..., 0] + tdens[..., 0]) * 1.0e6
        return self._rhop_bin_average(
            neutral_m3.ravel(),
            np.asarray(self._neutral_rhop_flat, dtype=float),
            np.asarray(self._neutral_weights_flat, dtype=float),
            np.asarray(self._rhop_refactor, dtype=float),
        )

    def _fast_ion_density_profile(self) -> np.ndarray:
        """Map FIDASIM denf to rhop using precomputed interpolation-grid mapping."""
        if not os.path.exists(self.distribution_file):
            return np.full(self._rhop_refactor.size, np.nan, dtype=float)

        with h5.File(self.distribution_file, "r") as h5f:
            if "denf" not in h5f:
                return np.full(self._rhop_refactor.size, np.nan, dtype=float)
            denf_m3 = np.asarray(h5f["denf"][...], dtype=float) * 1.0e6

        return self._rhop_bin_average(
            denf_m3.ravel(),
            np.asarray(self._fast_rhop_flat, dtype=float),
            np.asarray(self._fast_weights_flat, dtype=float),
            np.asarray(self._rhop_refactor, dtype=float),
        )

    def refactor_output(self) -> dict:
        """
        Return Indica-native NBI results following the abstract operator contract.

        Implementation notes
        --------------------
        - Contract source: abstract_nbioperator.NbiOperator.refactor_output docs.
        - Geometry mappings are precomputed in prepare(); this stage only reads
          output densities and applies cached reductions to rhop profiles.
        """
        rhop = np.asarray(self._rhop_refactor, dtype=float)
        t_coord = np.array([float(self.t)], dtype=float)

        neutral_profile = self._neutral_density_profile()
        fast_ion_profile = self._fast_ion_density_profile()
        pressure_placeholder = np.full(rhop.size, np.nan, dtype=float)

        def _to_da(name: str, profile: np.ndarray, status: str) -> xr.DataArray:
            return xr.DataArray(
                profile[np.newaxis, :],
                coords={"t": t_coord, "rhop": rhop},
                dims=("t", "rhop"),
                name=name,
                attrs={
                    "data_source": "FIDASIM",
                    "status": status,
                    "neutrals_file": self.neut_file,
                },
            )

        return {
            "neutral_density": _to_da(
                "neutral_density",
                neutral_profile,
                "mapped_from_fidasim_neutrals_h5",
            ),
            "fast_ion_density": _to_da(
                "fast_ion_density",
                fast_ion_profile,
                "mapped_from_fidasim_distribution_denf",
            ),
            "parallel_fast_ion_pressure": _to_da(
                "parallel_fast_ion_pressure",
                pressure_placeholder,
                "placeholder_distribution_moment_not_implemented",
            ),
            "perpendicular_fast_ion_pressure": _to_da(
                "perpendicular_fast_ion_pressure",
                pressure_placeholder,
                "placeholder_distribution_moment_not_implemented",
            ),
        }

def create_grids(
    transform: LineOfSightTransform,
    delta_src=0.0,
    delta_ang=0.0,
):
    """
    Starting from Indica transforms create Fidasim beam grid
    TODO: Indica transform currently has only 1 focal length
    """
    #_axis = np.array()
    _axis = np.asarray(transform.direction[0], dtype=float)

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
