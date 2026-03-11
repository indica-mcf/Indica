import copy
import json
import os
import shutil
import subprocess

import fidasim
from fidasim.utils import beam_grid
from fidasim.utils import rz_grid
import h5py as h5
import numpy as np
import xarray as xr

from ..nbi_configs import build_general_settings
from ..nbi_configs import build_nbi_settings
from ..nbi_configs import build_plasma_settings
from ..nbi_configs import FIDASIM_BASE_DIR
from ..nbi_configs import FIDASIM_BIN_PATH
from ..nbi_configs import FIDASIM_OUTPUT_DIR
from ..nbi_configs import get_hnbi_geo
from ..nbi_configs import get_rfx_geo
from ..nbi_configs import MC_SETTINGS_COARSE
from ..nbi_configs import MC_SETTINGS_FINE
from ..nbi_configs import PLASMA_INTERP_GRID_SETTINGS
from ..nbi_configs import SIMULATION_SWITCHES
from ..nbi_configs import FIDASIM_FI_DIST_FILE
from ..nbi_configs import WAVELENGTH_GRID_SETTINGS
from ..nbi_configs import WEIGHT_FUNCTION_SETTINGS

# from cxspec import CxsSpec
# import plot
# from batch import submit_fidasim_batch_job

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"


def convert_to_list(resdict):
    """Recursively search nested dict for arrays and convert to list."""

    for key, value in resdict.items():
        if isinstance(value, dict):
            convert_to_list(value)
        elif isinstance(value, np.ndarray):
            resdict[key] = value.tolist()


def create_st40_beam_grid(beam, plot_bgrid=False, ax=None, delta_src=0.0, delta_ang=0.0):
    """Fidasim beam grid creation for ST-40 beams."""

    rfx = get_rfx_geo()
    hnbi = get_hnbi_geo()

    # Modify RFX source position
    delta = delta_src
    norm_angle = np.arctan2(rfx["axis"][1], rfx["axis"][0]) + np.pi / 2
    dx = delta * np.cos(norm_angle)
    dy = delta * np.sin(norm_angle)
    rfx["src"][0] = rfx["src"][0] + dx * 100
    rfx["src"][1] = rfx["src"][1] + dy * 100

    # Modify RFX angle
    rfx_angle = np.arctan2(rfx["axis"][1], rfx["axis"][0])
    rfx_angle_new = rfx_angle + delta_ang
    axis_new = np.array([np.cos(rfx_angle_new), np.sin(rfx_angle_new), 0.0])
    rfx["axis"] = axis_new

    nbi_list = [rfx, hnbi]

    if ax:
        for beam_cfg in nbi_list:
            ax.scatter(
                beam_cfg["src"][0],
                beam_cfg["src"][1],
                beam_cfg["src"][2],
                marker="x",
                color="k",
            )
            pini_len = 1000
            pinix = beam_cfg["src"][0] + beam_cfg["axis"][0] * pini_len
            piniy = beam_cfg["src"][1] + beam_cfg["axis"][1] * pini_len
            piniz = beam_cfg["src"][2] + beam_cfg["axis"][2] * pini_len
            ax.plot3D(
                [beam_cfg["src"][0], pinix],
                [beam_cfg["src"][1], piniy],
                zs=[beam_cfg["src"][2], piniz],
                color="r",
            )

    rstart = 100  # [cm]

    if beam.upper() == "RFX":
        bgrid = beam_grid(
            rfx,
            rstart,
            length=250.0,
            width=250.0,
            height=50.0,
            dv=2.0,
        )
    else:
        bgrid = beam_grid(
            hnbi,
            rstart,
            length=250.0,
            width=250.0,
            height=50.0,
            dv=2.0,
        )

    return bgrid, nbi_list


def parse_input_file(input_dict_file):
    """Parses and checks jet-fidasim input dictionary.

    Parameters
    ----------

    """

    if not os.path.isfile(input_dict_file):
        raise FileNotFoundError(f"{input_dict_file} not found.")

    # Strip comments and read input dictionary
    with open(input_dict_file, mode="r", encoding="utf-8") as f:
        with open("temp.json", "w") as wf:
            for line in f.readlines():
                if line[0:2] == "//" or line[0:1] == "#":
                    continue
                wf.write(line)

    with open("temp.json", "r") as f:
        input_dict = json.load(f)

    os.remove("temp.json")

    # Check if write permissions to save directory
    if not os.access(input_dict["save_dir"], os.R_OK):
        raise PermissionError(
            "You do not have read permissions in the specified save directory "
            f"({input_dict['save_dir']})"
        )

    # Check is transp files exist
    for file_key, file_path in input_dict["input_files"].items():
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_key} not found ({file_path})")

    if "cxs_spec" in input_dict:
        if "chord_IDs" not in input_dict["cxs_spec"]:
            raise ValueError("chords not specified for cxs spec.")

    return input_dict


def _fidasim_out_to_xarray_dataset(h5_path: str) -> xr.Dataset:
    data_vars = {}
    with h5.File(h5_path, "r") as h5f:
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
    return xr.Dataset(data_vars=data_vars, attrs=root_attrs)


def _time_to_ms(time: float) -> int:
    return int(round(float(time) * 1.0e3))


def _build_run_prefix(file_name: str, time: float, nbi_name: str) -> str:
    base_name = str(file_name).strip()
    if not base_name:
        raise ValueError("file_name cannot be empty")
    return f"{base_name}_{_time_to_ms(time)}_ms_{nbi_name}"


def _run_fidasim(operator, ctx: dict) -> dict:
    # From this point on, everything is FIDASIM specific.
    plasmaconfig = {
        "R": ctx["R_2d"],
        "z": ctx["z_2d"],
        "rho_1d": ctx["rho_1d"],
        "rho": ctx["rho"],
        "rho_t": ctx["rho_tor"],
        "br": ctx["br"],
        "bz": ctx["bz"],
        "bt": ctx["bt"],
        "ti": ctx["ion_temperature"],
        "te": ctx["electron_temperature"],
        "nn": ctx["neutral_density"],
        "ne": ctx["electron_density"],
        "omegator": ctx["toroidal_rotation"],
        "zeff": ctx["zeffective"],
        "plasma_ion_amu": operator.plasma_ion_amu,
    }

    # Run TE-fidasim
    run_fidasim_flag = True

    file_name = ctx["file_name"]
    time = ctx["time"]

    beam = operator.name
    run_prefix = _build_run_prefix(file_name, time, beam)

    # File paths
    save_dir = FIDASIM_OUTPUT_DIR
    run_dir = os.path.join(save_dir, run_prefix)
    beam_save_dir = os.path.join(run_dir, beam)
    num_cores = 3
    fidasim_out = os.path.join(beam_save_dir, f"{run_prefix}_inputs.dat")

    # Remove the existing folder if re-running fidasim
    if run_fidasim_flag:
        try:
            shutil.rmtree(run_dir)
        except FileNotFoundError:
            pass

    # Run pre-processing code
    # This takes in filename/time context, the nbi configuration, and plasma.
    prepare_fidasim(
        file_name,
        time,
        {
            "name": operator.name,
            "einj": operator.einj,
            "pinj": operator.pinj,
            "current_fractions": operator.current_fractions,
            "ab": operator.ab,
        },
        plasmaconfig,
        save_dir=save_dir,
        plot_geo=False,
        fine_MC_res=True,
    )

    if run_fidasim_flag:
        subprocess.run(
            [
                FIDASIM_BIN_PATH,
                fidasim_out,
                f"{num_cores}",
            ]
        )

    neut_file = os.path.join(beam_save_dir, f"{run_prefix}_neutrals.h5")

    if not os.path.exists(neut_file):
        raise FileNotFoundError(f"Neutrals file not found: {neut_file}")

    neutrals_by_time = {
        float(time): {
            "path": neut_file,
            "data": _fidasim_out_to_xarray_dataset(neut_file),
        }
    }
    return neutrals_by_time


def prepare_fidasim(
    file_name: str,
    time: float,
    nbiconfig: dict,
    plasmaconfig: dict,
    fi_dist_file: str = FIDASIM_FI_DIST_FILE,
    save_dir: str = FIDASIM_OUTPUT_DIR,
    fida_dir: str = FIDASIM_BASE_DIR,
    fine_MC_res: bool = False,
    imp_charge: int = 6,
    plot_geo: bool = True,
):
    """Process jet-fidasim input into the required format for launching fidasim.
    Prepares and submits batch jobs in LoadLeveler for each pini.

    force_no_plasma_rot: Turns off rotation, even if OMEGA is available in TRANSP output
    """
    # Output dictionary for storing jet-fidasim relevant outputs.
    out_dict = {}

    # INPUT DICT - OK
    # REQUIRED FILES - OK
    # NOW DO FIDASIM PREPROCESSING

    time = time
    # geqdsk_file = input_dict['input_files']['geqdsk_file']
    st40_beams = nbiconfig
    # beam_amu = st40_beams["ab"]
    beam_name = st40_beams["name"]
    run_prefix = _build_run_prefix(file_name, time, beam_name)
    plasma_ion_amu = plasmaconfig["plasma_ion_amu"]
    # vtor_peak_kms = input_dict['vtor_peak_kms']

    # Preprocessing for each participating pini
    # Note: preprocessing.py modifies input dictionaries, so recreate the same
    # inputs for every pini.
    beam_id = st40_beams["name"]

    # Define plasma interpolation grid bounds
    rmin = PLASMA_INTERP_GRID_SETTINGS["rmin"]
    rmax = PLASMA_INTERP_GRID_SETTINGS["rmax"]
    zmin = PLASMA_INTERP_GRID_SETTINGS["zmin"]
    zmax = PLASMA_INTERP_GRID_SETTINGS["zmax"]
    nr = PLASMA_INTERP_GRID_SETTINGS["nr"]
    nz = PLASMA_INTERP_GRID_SETTINGS["nz"]
    grid = rz_grid(rmin, rmax, nr, zmin, zmax, nz)

    # Create the beam grid oriented on the RFX axis
    bgrid, nbis = create_st40_beam_grid(beam_name)

    # Geometry plotting removed (see fidasim_utils_plotting_legacy.py).

    # fields, rhogrid, btipsign = read_geqdsk(geqdsk_file, grid, poloidal=True)
    equil = dict()
    equil["time"] = time
    equil["br"] = plasmaconfig["br"]
    equil["bt"] = plasmaconfig["bt"]
    equil["bz"] = plasmaconfig["bz"]
    equil["er"] = plasmaconfig["br"] * 0.0
    equil["et"] = plasmaconfig["br"] * 0.0
    equil["ez"] = plasmaconfig["br"] * 0.0

    # Interpolate data according to fast particle grid
    for key in equil.keys():
        if key != "time":
            r_plasma = plasmaconfig["R"][0, :]
            z_plasma = plasmaconfig["z"][:, 0]
            # data_obj = RegularGridInterpolator(
            #     (r_plasma, z_plasma), equil[key], bounds_error=False,
            #     fill_value=0.0
            # )

            from scipy.interpolate import interp2d

            data_obj = interp2d(
                r_plasma, z_plasma, equil[key], bounds_error=False, fill_value=0.0
            )

            data_interp = np.zeros((nr, nz))
            for i_z in range(nz):
                for i_r in range(nr):
                    data_interp[i_r, i_z] = (
                        data_obj(grid["r2d"][i_r, 0] * 1e-2, grid["z2d"][0, i_z] * 1e-2)
                        / 2
                        * np.pi
                    )

            # data_interp = data_obj((grid['r2d']*1e-2, grid['z2d']*1e-2))
            equil[key] = data_interp

    equil["data_source"] = "Indica"
    equil["mask"] = np.ones_like(equil["br"], dtype=np.int32)

    # Read the dummy fast-ion distribution
    _fi_dist = h5.File(fi_dist_file, "r")
    fi_dist = dict()
    fi_dist["type"] = int(_fi_dist["type"][()])
    fi_dist["time"] = _fi_dist["time"][()]
    fi_dist["nenergy"] = int(_fi_dist["nenergy"][()])
    fi_dist["energy"] = _fi_dist["energy"][()]
    fi_dist["npitch"] = int(_fi_dist["npitch"][()])
    fi_dist["pitch"] = _fi_dist["pitch"][()]
    fbm_grid = np.zeros((fi_dist["nenergy"], fi_dist["npitch"], nr, nz))
    # fi_dist['f'] = np.asarray(_fi_dist['f'][()].T.tolist())
    fi_dist["f"] = fbm_grid
    fi_dist["denf"] = _fi_dist["denf"][()]
    fi_dist["data_source"] = str(_fi_dist["data_source"][()])

    # Fast particle positions
    r_fi = _fi_dist["r"][()]
    z_fi = _fi_dist["z"][()]
    r_fi, z_fi = np.meshgrid(r_fi, z_fi)

    # Interpolate rho grid
    rhogrid = plasmaconfig["rho"]
    r_plasma = plasmaconfig["R"][0, :]
    z_plasma = plasmaconfig["z"][:, 0]
    from scipy.interpolate import interp2d

    data_obj = interp2d(
        r_plasma, z_plasma, rhogrid, bounds_error=False, fill_value=10.0
    )
    data_interp = np.zeros((nr, nz))
    for i_z in range(nz):
        for i_r in range(nr):
            data_interp[i_r, i_z] = data_obj(
                grid["r2d"][i_r, 0] * 1e-2, grid["z2d"][0, i_z] * 1e-2
            )
    rhogrid = data_interp

    # Interpolate kinetic data
    from scipy.interpolate import interp1d

    # dims = rhogrid.shape
    f_zeff = interp1d(
        plasmaconfig["rho_1d"], plasmaconfig["zeff"], fill_value="extrapolate"
    )
    zeff = f_zeff(rhogrid)
    zeff = np.where(zeff > 1, zeff, 1.0).astype("float64")

    f_te = interp1d(
        plasmaconfig["rho_1d"], plasmaconfig["te"], fill_value="extrapolate"
    )
    te = f_te(rhogrid)
    te = np.where(te > 0.0, te, 0.0).astype("float64")

    f_ti = interp1d(
        plasmaconfig["rho_1d"], plasmaconfig["ti"], fill_value="extrapolate"
    )
    ti = f_ti(rhogrid)
    ti = np.where(ti > 0.0, ti, 0.0).astype("float64")

    f_nn = interp1d(
        plasmaconfig["rho_1d"], plasmaconfig["nn"], fill_value="extrapolate"
    )
    nn = f_nn(rhogrid)
    nn = np.where(nn > 0.0, nn, 0.0).astype("float64")

    f_ne = interp1d(
        plasmaconfig["rho_1d"], plasmaconfig["ne"], fill_value="extrapolate"
    )
    ne = f_ne(rhogrid)
    ne = np.where(ne > 0.0, ne, 0.0).astype("float64")

    f_omega = interp1d(
        plasmaconfig["rho_1d"], plasmaconfig["omegator"], fill_value="extrapolate"
    )
    omega = f_omega(rhogrid)
    omega = np.where(omega > 0.0, omega, 0.0).astype("float64")
    vt = grid["r2d"] * omega  # cm/s

    # TODO: double check units
    plasma = dict()
    plasma["time"] = time
    plasma["zeff"] = zeff
    plasma["te"] = 1.0e-03 * te  # fidasim expects keV
    plasma["ti"] = 1.0e-03 * ti  # fidasim expects keV
    plasma["denn"] = 1.0e-06 * nn  # fidasim expects cm^-3
    plasma["dene"] = 1.0e-06 * ne  # fidasim expects cm^-3
    plasma["vr"] = np.zeros_like(plasma["ti"])
    plasma["vz"] = np.zeros_like(plasma["ti"])
    plasma["vt"] = vt
    plasma["data_source"] = "Indica"
    max_rho = np.nanmax(np.abs(plasmaconfig["rho_1d"]))
    mask = np.zeros_like(plasma["ti"], dtype="int")
    w = np.where(rhogrid <= max_rho)  # where we have profiles
    mask[w] = 1
    plasma["mask"] = mask

    # Add grid and flux to plasma dict
    plasma["grid"] = grid
    plasma["flux"] = rhogrid
    plasma["bgrid"] = bgrid

    # extract omp profiles from 2D plasma grid
    # Assume z grid is up-down symmetric
    plasma["profiles"] = {}
    i_z = int(nz / 2)
    plasma["profiles"]["ti"] = plasma["ti"][:, i_z]
    plasma["profiles"]["te"] = plasma["te"][:, i_z]
    plasma["profiles"]["dene"] = plasma["dene"][:, i_z]
    plasma["profiles"]["denn"] = plasma["denn"][:, i_z]
    plasma["profiles"]["rho"] = plasma["flux"][:, i_z]
    plasma["profiles"]["r_omp"] = plasma["grid"]["r"]

    # manual v_tor profile
    plasma["profiles"]["vt"] = plasma["vt"][:, i_z]

    # Create results directory
    case_save_dir = os.path.join(save_dir, run_prefix)
    if not os.path.exists(case_save_dir):
        os.makedirs(case_save_dir)

    _case_save_dir = case_save_dir
    if not os.path.exists(_case_save_dir):
        os.makedirs(_case_save_dir)

    out_dict["plasma"] = copy.deepcopy(plasma)
    out_dict["plasma"]["time"] = str(out_dict["plasma"]["time"])
    out_dict["flux"] = copy.deepcopy(rhogrid)
    out_dict["grid"] = copy.deepcopy(grid)
    out_dict["bgrid"] = copy.deepcopy(bgrid)
    convert_to_list(out_dict)
    # Write plasma dictionary in JSON format and save to run directory
    save_plasma_file = os.path.join(_case_save_dir, f"{run_prefix}_plasma.json")
    with open(save_plasma_file, mode="w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    # Create results directory for each beam
    beam_save_dir = os.path.join(_case_save_dir, beam_id)
    if not os.path.exists(beam_save_dir):
        os.makedirs(beam_save_dir)

    general_settings = build_general_settings(
        0, time, run_prefix, beam_save_dir, fida_dir
    )
    simulation_switches = SIMULATION_SWITCHES

    if fine_MC_res:
        mc_settings = MC_SETTINGS_FINE
    else:
        mc_settings = MC_SETTINGS_COARSE

    nbi_settings = build_nbi_settings(st40_beams)
    plasma_settings = build_plasma_settings(plasma_ion_amu, imp_charge)
    wavelength_grid_settings = WAVELENGTH_GRID_SETTINGS
    weight_function_settings = WEIGHT_FUNCTION_SETTINGS

    inputs = dict(general_settings)
    inputs.update(simulation_switches)
    inputs.update(mc_settings)
    inputs.update(nbi_settings)
    inputs.update(plasma_settings)
    inputs.update(wavelength_grid_settings)
    inputs.update(weight_function_settings)
    inputs.update(bgrid)

    for beam in nbis:
        if beam_id == beam["name"]:
            fidasim.prefida(inputs, grid, beam, plasma, equil, fi_dist)

    # If here then preprocessing was successful for this beam. Launch batch job.
    # submit_fidasim_batch_job(beam_save_dir)


# Might be used later.
"""
def postproc_fidasim(
    shot: int,
    time: float,
    nbiconfig: dict,
    specconfig: dict,
    plasmaconfig: dict,
    save_dir: str = FIDASIM_OUTPUT_DIR,
    process_spec=True,
    block=False,
    debug=False,
    los_type="center",
):

    Collect fidasim hdf5 results from each pini.
    Optionally fit CXS spectra and save processed output to a JSON dictionary.

    Parameters
    ----------
    process_spec : bool
        Flag for collecting and fitting CXS spectra for each pini, as well as
        the total of all pinis.



    out_dict = {}  # Ouptut dictionary containing combined pini results
    time = time
    st40_beams = nbiconfig
    beam_amu = st40_beams["ab"]
    beam_name = st40_beams["name"]
    st40_spec = specconfig
    runid = pwd.getpwuid(os.getuid())[0]
    spec_name = st40_spec["name"]
    cross_section_corr = False
    if "cross_section_corr" in st40_spec:
        cross_section_corr = st40_spec["cross_section_corr"]
    plasma_ion_amu = plasmaconfig["plasma_ion_amu"]

    out_dict["amu"] = plasma_ion_amu

    # Configure spec dictionary compatible with fidasim format.
    spec = None
    if spec_name in st40_spec["name"]:
        pi_spec = CxsSpec(
            shot,
            chord_IDs=st40_spec["chord_IDs"],
            amu=plasma_ion_amu,
            beam_amu=beam_amu,
            beam_name=beam_name,
            spec_name=spec_name,
            cross_section_corr=cross_section_corr,
            custom_geo_dict=st40_spec["geom_dict"],
        )
        nchan = len(st40_spec["chord_IDs"])

        ids = []
        for id in st40_spec["chord_IDs"]:
            ids.append(id.encode(encoding="utf_8"))

        ids = []
        radius = []
        lens = []
        axis = []
        _spot_radius = 1.25  # TODO: estimate spot radius on Princeton foreoptic
        spot_size = []
        _sigma_pi_ratio = 1.0  # default sigma/pi ratio
        sigma_pi = []

        ## import LOS data from local pickle file (J Wood 29/07/22)
        # import pickle
        # los_data = pickle.load(open('PI_LOS_geometry_processed.p', 'rb'))
        # los_data = los_data['3POINT_AV']

        for index, chord in enumerate(pi_spec.chords):

            ids.append(chord.id.encode(encoding="utf_8"))
            radius.append(chord.tang_rad)
            lens.append(chord.origin)
            axis.append(chord.diruvec)
            spot_size.append(_spot_radius)
            sigma_pi.append(_sigma_pi_ratio)

    # run directory
    time_str = "t_{:8.6f}".format(time)
    run_dir = save_dir + "/" + str(shot) + "/" + time_str
    plasma_file = run_dir + "/TE-fidasim_plasma.json"

    # Collect fidasim results for each beam and store in output dictionary
    # icnt = 0
    # for beam_id, beam_detail in st40_beams.items():

    beam_save_dir = run_dir + "/" + beam_name

    if not os.path.exists(beam_save_dir):
        raise FileNotFoundError(f"Results directory path not found: {beam_save_dir}")

    if spec_name in st40_spec["name"] and process_spec:
        spec_file = beam_save_dir + "/" + runid + "_spectra.h5"
        geo_file = beam_save_dir + "/" + runid + "_geometry.h5"
        # dcx_file = beam_save_dir + '/' + runid + '_dcx.h5'
        neut_file = beam_save_dir + "/" + runid + "_neutrals.h5"

        try:
            open(spec_file, "rb")
        except FileNotFoundError:
            raise FileNotFoundError(f"Results spectra file not found: {spec_file}")

        # Collect results from fidasim
        pi_spec.collect_pini_spectra(beam_name, spec_file, geo_file, neut_file)

        # Using fidasim DCX and halo density, manually perform line-integration
        # as a sanity check against fidasim.
        pi_spec.los_integrate_pini_brightness(
            beam_name, beam_save_dir, plasma_file, neut_file
        )

        # Using fidasim full-energy neutral beam density, manually perform CVI
        # line-integration.
        # Assume constant C_6+ concetration
        pi_spec.los_integrate_CVI_brightness(
            beam_name, beam_save_dir, plasma_file, neut_file, block=block
        )

    export_dict = dict()
    if spec_name in st40_spec["name"] and process_spec:
        # Fit fidasim spectra from individual pini and sum of pinis for Ti, v_tor
        pi_spec.fit_spectra(block=block)

        # Calculate Doppler shifts for full, half, and third-energy components
        # of each pini.
        pi_spec.calc_bes_dopp_shifts()

        # Also fit manually line-integrated spectra from each beam and the
        # beam-summed spectra for Ti, v_tor.
        # Spectra are generated using fidasim 3D density plots and 2D poloidal
        # plasma Ti contours.
        pi_spec.fit_spectra(fit_manual_los_integral=True, block=block)
        pi_spec.fit_spectra(fit_manual_cvi_integral=True, block=block, run_dir=run_dir)

        # Save results to JSON dictionary and append to main output dictionary
        out_dict[spec_name] = pi_spec.serialize()
        # Extract fit data, export as dictionary
        Ti = np.zeros(len(out_dict[spec_name].keys()))
        Ti_err = np.zeros(len(out_dict[spec_name].keys()))
        cwl = np.zeros(len(out_dict[spec_name].keys()))
        cwl_err = np.zeros(len(out_dict[spec_name].keys()))
        vtor = np.zeros(len(out_dict[spec_name].keys()))
        vtor_err = np.zeros(len(out_dict[spec_name].keys()))
        for i_chord, id in enumerate(out_dict[spec_name].keys()):
            Ti[i_chord] = out_dict[spec_name][id]["res"][beam_name]["man_los_integral"][
                "fit_cvi"
            ]["Ti"]
            Ti_err[i_chord] = out_dict[spec_name][id]["res"][beam_name][
                "man_los_integral"
            ]["fit_cvi"]["Ti_err"]
            cwl[i_chord] = out_dict[spec_name][id]["res"][beam_name][
                "man_los_integral"
            ]["fit_cvi"]["cwl"]
            cwl_err[i_chord] = out_dict[spec_name][id]["res"][beam_name][
                "man_los_integral"
            ]["fit_cvi"]["cwl_err"]

            # Convert Doppler shift to toroidal rotation
            vtor[i_chord] = get_v_tor_v_pol(
                out_dict[spec_name][id]["origin"],
                np.array(out_dict[spec_name][id]["beam_intersect_pos"][beam_name]),
                529.059 - cwl[i_chord],
                529.059,
            )
            vtor_err[i_chord] = get_v_tor_v_pol(
                out_dict[spec_name][id]["origin"],
                np.array(out_dict[spec_name][id]["beam_intersect_pos"][beam_name]),
                cwl_err[i_chord],
                529.059,
            )

        export_dict["chord_id"] = list(out_dict[spec_name].keys())
        export_dict["Ti"] = Ti
        export_dict["Ti_err"] = Ti_err
        export_dict["cwl"] = cwl
        export_dict["cwl_err"] = cwl_err
        export_dict["vtor"] = vtor
        export_dict["vtor_err"] = vtor_err

    # Write output dictionary in JSON format and save to run directory
    savefile = run_dir + "/TE-fidasim_output.json"
    with open(savefile, mode="w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    # Plotting removed (see fidasim_utils_plotting_legacy.py).

    # Export temperature and velocity results from simulated data
    return export_dict
    """
