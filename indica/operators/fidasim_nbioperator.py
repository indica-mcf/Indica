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
        # FIDASIM rz_grid uses cm, while transform machine dimensions are in m.
        rmin = self.transform._machine_dims[0][0] * 100.0
        rmax = self.transform._machine_dims[0][1] * 100.0
        zmin = self.transform._machine_dims[1][0] * 100.0
        zmax = self.transform._machine_dims[1][1] * 100.0
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

        # Convert grid axes back to meters for equilibrium interpolation calls.
        _R = grid["r2d"][:, 0] * 1.0e-2# ie. take any column and all values in that column
        _z = grid["z2d"][0, :] * 1.0e-2# The same thing. Just that R is transposed!
        R = xr.DataArray(_R, coords={"R": _R})
        z = xr.DataArray(_z, coords={"z": _z})
    
        #2D map. We build the 2d grid from equilibrium
        rhop_2d = equilibrium.rhop.interp(t=self.t).interp(R=R, z=z)
        rhot_2d, _ = self.transform.equilibrium.convert_flux_coords(rhop_2d, t=self.t)
        br_2d, bz_2d, bt_2d, _ = self.transform.equilibrium.Bfield(R, z, t=self.t, full_Rz=True)

        # Mask where plasma profiles are available
        max_rhop_profiles = np.max(self.Te.rhop)
        mask = xr.full_like(rhop_2d, 1)
        mask = xr.where(rhop_2d <= max_rhop_profiles, mask, 0)


        map_masked = (
            lambda profile, scale=1.0: xr.where(
                mask > 0,
                profile.interp(rhop=rhop_2d) * scale,
                0,
            )
            .T.data
        )
        
        plasma = {
            "data_source": "Indica",
            "time": self.t,
            "zeff": map_masked(self.Zeff),
            "ti": map_masked(self.Ti, 1.0e-03),
            "te": map_masked(self.Te, 1.0e-03),
            "denn": map_masked(self.Nn, 1.0e-06),
            "dene": map_masked(self.Ne, 1.0e-06),
            "vr": np.zeros_like(rhop_2d).T,
            "vz": np.zeros_like(rhop_2d).T,
            "vt": map_masked(self.Vtor, 100.0),
            "mask": np.int_(mask.data.T),
            "plasma_ion_amu": self.target_element_info["A"],
            "grid": grid,
            "flux": rhop_2d,
            "bgrid": bgrid,
        }

        # Add midplane profiles to plasma dictionary
        zmag = xr.full_like(R, self.transform.equilibrium.zmag.interp(t=self.t).data)
        rhop_midplane,_,_ = self.transform.equilibrium.flux_coords(R, zmag, t=self.t)
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
            "br": br_2d.data.T,
            "bz": bz_2d.data.T,
            "bt": bt_2d.data.T,
            "er": np.zeros_like(br_2d.data.T),
            "ez": np.zeros_like(br_2d.data.T),
            "et": np.zeros_like(br_2d.data.T),
            "mask": np.int32(mask.data.T),
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
#            "impurity_charge": int(np.mean(self.MeanZ)),
            "impurity_charge": 5,
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
        print("Running beam code")

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
        # Step 1: normalize to a float numpy array so downstream math is stable.
        coord = np.asarray(coord_1d, dtype=float)
        # Step 2: single-point axis has no neighbors, so use unit width fallback.
        if coord.size == 1:
            return np.ones(1, dtype=float)

        # Step 3: compute local widths from center-to-center spacing.
        widths = np.empty_like(coord)
        widths[1:-1] = 0.5 * (coord[2:] - coord[:-2])
        # Step 4: edge cells use one-sided neighbor distance.
        widths[0] = coord[1] - coord[0]
        widths[-1] = coord[-1] - coord[-2]
        # Step 5: enforce positive widths in case axis direction is reversed.
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
        # Step 1: load target rhop bin centers.
        rhop = np.asarray(rhop_centers, dtype=float)
        if rhop.size == 1:
            # Step 2a: degenerate one-bin case -> one weighted average over valid points.
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

        # Step 2b: reconstruct bin edges from center points.
        mids = 0.5 * (rhop[1:] + rhop[:-1])
        edges = np.empty(rhop.size + 1, dtype=float)
        edges[1:-1] = mids
        edges[0] = rhop[0] - (mids[0] - rhop[0])
        edges[-1] = rhop[-1] + (rhop[-1] - mids[-1])

        # Step 3: assign each sample to a rhop bin.
        idx = np.digitize(rhop_values, edges) - 1
        # Step 4: keep only finite, positive-weight, in-range samples.
        mask = (
            np.isfinite(values)
            & np.isfinite(rhop_values)
            & np.isfinite(weights)
            & (weights > 0.0)
            & (idx >= 0)
            & (idx < rhop.size)
        )

        # Step 5: accumulate weighted numerator and denominator per bin.
        num = np.bincount(
            idx[mask], weights=values[mask] * weights[mask], minlength=rhop.size
        )
        den = np.bincount(idx[mask], weights=weights[mask], minlength=rhop.size)

        # Step 6: finalize per-bin averages (NaN when no contributing volume).
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
        # Step 1: cache the target rhop grid and time-sliced equilibrium rhop map.
        rhop = np.asarray(self.Nn.coords["rhop"].data, dtype=float)
        self._rhop_refactor = rhop

        eq_rhop = self.transform.equilibrium.rhop.interp(t=self.t)

        # Step 2: build beam-grid cell-center coordinates in FIDASIM beam frame (cm).
        # Beam-grid (uvw in cm) -> machine (R,z in m) for neutral density maps.
        x_cm = np.linspace(float(bgrid["xmin"]), float(bgrid["xmax"]), int(bgrid["nx"]))
        y_cm = np.linspace(float(bgrid["ymin"]), float(bgrid["ymax"]), int(bgrid["ny"]))
        z_cm = np.linspace(float(bgrid["zmin"]), float(bgrid["zmax"]), int(bgrid["nz"]))
        z3, y3, x3 = np.meshgrid(z_cm, y_cm, x_cm, indexing="ij")
        uvw = np.vstack((x3.ravel(), y3.ravel(), z3.ravel()))

        # Step 3: transform beam-frame points to machine coordinates, then map to rhop.
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

        # Step 4: precompute neutral-cell volume weights (cm^3 -> m^3) and flatten.
        dx_cm = self._cell_widths(x_cm)
        dy_cm = self._cell_widths(y_cm)
        dz_cm = self._cell_widths(z_cm)
        self._neutral_weights_flat = (
            dz_cm[:, None, None] * dy_cm[None, :, None] * dx_cm[None, None, :]
        ).ravel() * 1.0e-6

        # Step 5: map interpolation-grid (R,z) points to rhop for denf reduction.
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

        # Step 6: compute poloidal cell areas while handling grid orientation.
        if r2d.shape == (z_axis.size, r_axis.size):
            area = dz_m[:, None] * dr_m[None, :]
        elif r2d.shape == (r_axis.size, z_axis.size):
            area = dr_m[:, None] * dz_m[None, :]
        else:
            # Fallback for unexpected orientation; use mean cell sizes.
            area = np.full_like(r2d, float(np.mean(dr_m) * np.mean(dz_m)))

        # Step 7: convert area to cylindrical volume weight using Jacobian ~ R.
        self._fast_weights_flat = (np.abs(r2d) * 1.0e-2 * area).ravel()

    def _read_neutral_components_m3(self) -> dict[str, np.ndarray]:
        """Read FIDASIM neutral components and convert cm^-3 -> m^-3."""
        with h5.File(self.neut_file, "r") as h5f:
            return {
                "fdens": np.asarray(h5f["fdens"][...], dtype=float) * 1.0e6,
                "hdens": np.asarray(h5f["hdens"][...], dtype=float) * 1.0e6,
                "tdens": np.asarray(h5f["tdens"][...], dtype=float) * 1.0e6,
            }

    @staticmethod
    def _ground_state_component(component_4d: np.ndarray) -> np.ndarray:
        """Select ground-state contribution (excitation index 0)."""
        return np.asarray(component_4d[..., 0], dtype=float)

    def _total_ground_state_neutral_density_m3(self) -> np.ndarray:
        """Build total neutral density map from full/half/third components."""
        components = self._read_neutral_components_m3()
        fdens_gs = self._ground_state_component(components["fdens"])
        hdens_gs = self._ground_state_component(components["hdens"])
        tdens_gs = self._ground_state_component(components["tdens"])
        return fdens_gs + hdens_gs + tdens_gs

    def _map_neutral_grid_to_rhop(self, neutral_m3: np.ndarray) -> np.ndarray:
        """Reduce beam-grid neutral map to rhop profile."""
        return self._rhop_bin_average(
            np.asarray(neutral_m3, dtype=float).ravel(),
            np.asarray(self._neutral_rhop_flat, dtype=float),
            np.asarray(self._neutral_weights_flat, dtype=float),
            np.asarray(self._rhop_refactor, dtype=float),
        )

    def _neutral_density_profile(self) -> np.ndarray:
        """Map FIDASIM neutral densities to rhop using TriWaSp-like staging."""
        # Step 1: read component maps (fdens/hdens/tdens) in SI units.
        # Step 2: select ground-state and sum full/half/third contributions.
        neutral_m3 = self._total_ground_state_neutral_density_m3()
        # Step 3: collapse the 3D beam map to a 1D rhop profile.
        return self._map_neutral_grid_to_rhop(neutral_m3)

    def _read_fast_ion_density_map_m3(self) -> np.ndarray | None:
        """Read FIDASIM denf map and convert cm^-3 -> m^-3, if available."""
        if not os.path.exists(self.distribution_file):
            return None

        with h5.File(self.distribution_file, "r") as h5f:
            if "denf" not in h5f:
                return None
            return np.asarray(h5f["denf"][...], dtype=float) * 1.0e6

    def _map_fast_ion_grid_to_rhop(self, denf_m3: np.ndarray) -> np.ndarray:
        """Reduce denf map to rhop profile via cached interpolation-grid mapping."""
        return self._rhop_bin_average(
            np.asarray(denf_m3, dtype=float).ravel(),
            np.asarray(self._fast_rhop_flat, dtype=float),
            np.asarray(self._fast_weights_flat, dtype=float),
            np.asarray(self._rhop_refactor, dtype=float),
        )

    def _fast_ion_density_profile(self) -> np.ndarray:
        """Map FIDASIM denf to rhop using precomputed interpolation-grid mapping."""
        # Step 1: read denf map from distribution output.
        denf_m3 = self._read_fast_ion_density_map_m3()
        # Step 2: preserve output contract when denf is unavailable.
        if denf_m3 is None:
            return np.full(self._rhop_refactor.size, np.nan, dtype=float)
        # Step 3: collapse denf map to the target 1D rhop profile.
        return self._map_fast_ion_grid_to_rhop(denf_m3)

    def _build_refactor_profiles(self) -> dict[str, np.ndarray]:
        """Compute all radial profiles before DataArray packaging."""
        rhop_size = int(np.asarray(self._rhop_refactor, dtype=float).size)
        pressure_placeholder = np.full(rhop_size, np.nan, dtype=float)
        return {
            "neutral_density": self._neutral_density_profile(),
            "fast_ion_density": self._fast_ion_density_profile(),
            "parallel_fast_ion_pressure": pressure_placeholder.copy(),
            "perpendicular_fast_ion_pressure": pressure_placeholder.copy(),
        }

    def refactor_output(self) -> dict:
        """
        Return Indica-native NBI results following the abstract operator contract.

        Implementation notes
        --------------------
        - Contract source: abstract_nbioperator.NbiOperator.refactor_output docs.
        - Geometry mappings are precomputed in prepare(); this stage only reads
          output densities and applies cached reductions to rhop profiles.
        """
        print("refactoring")
        # Step 1: define output coordinates (single-time slice over cached rhop grid).
        rhop = np.asarray(self._rhop_refactor, dtype=float)
        t_coord = np.array([float(self.t)], dtype=float)

        # Step 2: compute radial profiles with explicit staged helper workflow.
        profiles = self._build_refactor_profiles()

        print(self.neut_file)
        def _to_da(name: str, profile: np.ndarray, status: str) -> xr.DataArray:
            # Local helper: wrap each profile into the contract-compliant DataArray.
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

        # Step 3: assemble the abstract-operator output dictionary.
        result = {
            "neutral_density": _to_da(
                "neutral_density",
                profiles["neutral_density"],
                "mapped_from_fidasim_neutrals_h5",
            ),
            "fast_ion_density": _to_da(
                "fast_ion_density",
                profiles["fast_ion_density"],
                "mapped_from_fidasim_distribution_denf",
            ),
            "parallel_fast_ion_pressure": _to_da(
                "parallel_fast_ion_pressure",
                profiles["parallel_fast_ion_pressure"],
                "placeholder_distribution_moment_not_implemented",
            ),
            "perpendicular_fast_ion_pressure": _to_da(
                "perpendicular_fast_ion_pressure",
                profiles["perpendicular_fast_ion_pressure"],
                "placeholder_distribution_moment_not_implemented",
            ),
        }
        # Step 4: cache last result for plotting convenience and return.
        self._last_result = result
        return result

    def _result_for_plot(self, result: dict | None) -> dict:
        """Resolve plot input result, reusing cached output when available."""
        if result is not None:
            return result
        if hasattr(self, "_last_result"):
            return self._last_result
        return self.refactor_output()

    @staticmethod
    def _resolve_z_plane_index(z_m: np.ndarray, z_index: int | None) -> int:
        """Pick z-plane index; default is plane closest to z=0."""
        if z_index is None:
            return int(np.argmin(np.abs(z_m)))
        return int(np.clip(z_index, 0, len(z_m) - 1))

    def _read_neutral_plot_data_m3(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Read neutral components and beam-grid axes for plotting."""
        with h5.File(self.neut_file, "r") as h5f:
            components = {
                "fdens": np.asarray(h5f["fdens"][...], dtype=float) * 1.0e6,
                "hdens": np.asarray(h5f["hdens"][...], dtype=float) * 1.0e6,
                "tdens": np.asarray(h5f["tdens"][...], dtype=float) * 1.0e6,
            }
            x_m = np.asarray(h5f["grid"]["x"][...], dtype=float) * 1.0e-2
            y_m = np.asarray(h5f["grid"]["y"][...], dtype=float) * 1.0e-2
            z_m = np.asarray(h5f["grid"]["z"][...], dtype=float) * 1.0e-2
        return components, x_m, y_m, z_m

    def _ground_state_neutral_components_for_plot(
        self, components: dict[str, np.ndarray]
    ) -> list[tuple[np.ndarray, str]]:
        """Prepare ground-state full/half/third component maps for plotting."""
        return [
            (self._ground_state_component(components["fdens"]), "fdens (full)"),
            (self._ground_state_component(components["hdens"]), "hdens (half)"),
            (self._ground_state_component(components["tdens"]), "tdens (third)"),
        ]

    def _plot_profile_panel(self, plt, result: dict):
        """Plot refactor_output radial profiles in a 2x2 panel."""
        fig_profiles, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        profile_keys = [
            "neutral_density",
            "fast_ion_density",
            "parallel_fast_ion_pressure",
            "perpendicular_fast_ion_pressure",
        ]
        for ax, key in zip(axs.flat, profile_keys):
            da = result[key]
            ax.plot(da.rhop.values, da.isel(t=0).values)
            ax.set_title(key)
            ax.set_xlabel("rhop")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
        fig_profiles.tight_layout()
        return fig_profiles

    def _plot_neutral_plane_panel(self, plt, z_index: int | None):
        """Plot ground-state neutral components on one selected z-plane."""
        if not os.path.exists(self.neut_file):
            return None

        # Step 1: read neutral component maps + beam-grid axes.
        components, x_m, y_m, z_m = self._read_neutral_plot_data_m3()
        # Step 2: choose plotting plane (defaults to nearest z=0).
        plane_index = self._resolve_z_plane_index(z_m, z_index)
        # Step 3: select full/half/third ground-state component maps.
        components_ground = self._ground_state_neutral_components_for_plot(components)

        # Step 4: render the three neutral components on the selected plane.
        fig_neutrals, axs2 = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
        for ax, (arr, title) in zip(axs2, components_ground):
            im = ax.pcolormesh(x_m, y_m, arr[plane_index, :, :], shading="auto")
            ax.set_title(f"{title} @ z={z_m[plane_index]:.3f} m")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            fig_neutrals.colorbar(im, ax=ax)
        fig_neutrals.tight_layout()
        return fig_neutrals

    def plot(
        self,
        result: dict | None = None,
        z_index: int | None = None,
        show: bool = True,
    ) -> dict:
        """
        Quick-look plotting for NBI outputs.

        This mirrors the legacy TriWaSp workflow style at a practical level:
        - plot refactor_output radial profiles,
        - plot full/half/third neutral components on a beam-grid plane.
        """
        import matplotlib.pyplot as plt

        # Step 1: resolve data to plot.
        result_plot = self._result_for_plot(result)
        # Step 2: render radial profile panel.
        fig_profiles = self._plot_profile_panel(plt, result_plot)
        # Step 3: render neutral component plane panel (if neutrals output exists).
        fig_neutrals = self._plot_neutral_plane_panel(plt, z_index)

        if show:
            plt.show()

        return {
            "profiles": fig_profiles,
            "neutrals": fig_neutrals,
        }

#MARCO: This class had a bunch of references to self.equilibrium, should be self.transform.equilbirium.
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
        "name":"testbeam",
        "shape": shape,
        "src": 100 * np.array(transform.origin[0]),
        "axis": axis,
        "widy": 100 * transform.spot_width,
        "widz": 100 * transform.spot_height,

        "divy": transform.div_width,
        "divz": transform.div_height,
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
    rstart = transform._machine_dims[0][1]*100.0
    bgrid = beam_grid(
        beam_cfg,
        rstart,
        length=rstart * 2.5,
        width=rstart * 2.5,
        height=rstart / 2.0,
        dv=2.0,
    )

    return bgrid, beam_cfg
