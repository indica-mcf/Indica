from scipy.interpolate import RegularGridInterpolator
import json, os, sys, pwd, copy
import numpy as np
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import shutil

import h5py as h5
from .nbi_configs import FIDASIM_BASE_DIR
from .nbi_configs import FIDASIM_OUTPUT_DIR
from .nbi_configs import TE_FIDASIM_FI_DIST_FILE
from .nbi_configs import TE_FIDASIM_INPUT_REWRITE_FROM
from .nbi_configs import FIDASIM_INPUT_REWRITE_TO
from .nbi_configs import build_general_settings
from .nbi_configs import build_nbi_settings
from .nbi_configs import build_plasma_settings
from .nbi_configs import MC_SETTINGS_COARSE
from .nbi_configs import MC_SETTINGS_FINE
from .nbi_configs import PLASMA_INTERP_GRID_SETTINGS
from .nbi_configs import SIMULATION_SWITCHES
from .nbi_configs import WAVELENGTH_GRID_SETTINGS
from .nbi_configs import WEIGHT_FUNCTION_SETTINGS

os.environ["HDF5_DISABLE_VERSION_CHECK"] = '1'

from fidasim.utils import rz_grid, read_geqdsk, beam_grid
import fidasim
#from st40_utils import extract_hda_plasma, create_st40_beam_grid, convert_to_list, get_v_tor_v_pol
#from cxspec import CxsSpec
#import plot
# from batch import submit_fidasim_batch_job

os.environ["HDF5_DISABLE_VERSION_CHECK"] = '1'

def parse_input_file(input_dict_file):
    """Parses and checks jet-fidasim input dictionary.

    Parameters
    ----------

    """

    if os.path.isfile(input_dict_file):
        print('Found input dictionary: ', input_dict_file)
    else:
        sys.exit('\033[91m' + input_dict_file + ' not found.' + '\033[0m')

    # Strip comments and read input dictionary
    with open(input_dict_file, mode='r', encoding='utf-8') as f:
        with open("temp.json", 'w') as wf:
            for line in f.readlines():
                if line[0:2] == '//' or line[0:1] == '#':
                    continue
                wf.write(line)

    with open("temp.json", 'r') as f:
        input_dict = json.load(f)

    os.remove('temp.json')

    # Check input dict
    input_dict['input_files']['geqdsk_file'] = input_dict['input_files']['geqdsk_file'].replace(
        TE_FIDASIM_INPUT_REWRITE_FROM,
        FIDASIM_INPUT_REWRITE_TO
    )
    input_dict['input_files']['fi_dist_file'] = input_dict['input_files']['fi_dist_file'].replace(
        TE_FIDASIM_INPUT_REWRITE_FROM,
        FIDASIM_INPUT_REWRITE_TO
    )
    print(input_dict['save_dir'])
    print(input_dict['input_files']['geqdsk_file'])
    print(input_dict['input_files']['fi_dist_file'])

    # Check if write permissions to save directory
    if not os.access(input_dict['save_dir'], os.R_OK):
        print('\033[91m' + 'ERROR: You do not have read permissions in the specified save directory (' +
              input_dict['save_dir'] + ')' + '\033[0m')
        exit()

    # Check is transp files exist
    for file_key, file_path in input_dict['input_files'].items():
        if os.path.isfile(file_path):
            print('Found ' + file_key + ' (' + file_path + ')')
        else:
            sys.exit('\033[91m' + 'ERROR: ' + file_key + ' not found (' + file_path + ')' + '\033[0m')

    if 'cxs_spec' in input_dict:
        if not 'chord_IDs' in input_dict['cxs_spec']:
            print('\033[91m' + 'ERROR: chords not specified for cxs spec.' + '\033[0m')
            exit()
            
    return input_dict

def prepare_fidasim(
        shot: int,
        time: float,
        nbiconfig: dict,
        specconfig: dict,
        plasmaconfig: dict,
        fi_dist_file: str = TE_FIDASIM_FI_DIST_FILE,
        save_dir: str = FIDASIM_OUTPUT_DIR,
        fida_dir: str = FIDASIM_BASE_DIR,
        fine_MC_res: bool = False,
        imp_charge: int = 6,
        plot_geo: bool = True,
):
    """Processes the jet-fidasim input dictionary into the specific input requirements for launching fidasim.
        Prepares and submits batch jobs in LoadLeveler for each pini.

        force_no_plasma_rot: Turns off rotation, even if OMEGA is available in TRANSP output
    """
    #TODO: spec. what do we do with spec?
 
    # Output dictionary for storing jet-fidasim relevant outputs.
    out_dict = {}

    # INPUT DICT - OK
    # REQUIRED FILES - OK
    # NOW DO FIDASIM PREPROCESSING

    ax = None
    if plot_geo:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=90, azim=-90)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)
        Rmaj = plt.Circle((0,0), 40, color='k', fill=False)
        Rsep = plt.Circle((0,0), 40+26, color='darkgrey', fill=False) 
        ax.add_patch(Rmaj)
        ax.add_patch(Rsep)
        art3d.pathpatch_2d_to_3d(Rmaj, z=0, zdir="z")
        art3d.pathpatch_2d_to_3d(Rsep, z=0, zdir="z")

    time = time
    #geqdsk_file = input_dict['input_files']['geqdsk_file']
    st40_beams = nbiconfig
    beam_amu = st40_beams['ab']
    beam_name = st40_beams['name']
    st40_spec = specconfig
    #run = input_dict['run']
    runid = pwd.getpwuid(os.getuid())[0]
    spec_name = st40_spec['name']
    cross_section_corr = False
    if 'cross_section_corr' in st40_spec:
        cross_section_corr = st40_spec['cross_section_corr']
    plasma_ion_amu = plasmaconfig['plasma_ion_amu']
    #vtor_peak_kms = input_dict['vtor_peak_kms']

    # Configure spec dictionary compatible with fidasim format.
    spec = None
    if spec_name in st40_spec['name']:
        #pi_spec = CxsSpec(shot, chord_IDs=input_dict['cxs_spec']['chord_IDs'],
        #          amu=plasma_ion_amu, plot_chords=plot_geo, ax=ax,
        #          spec_name=spec_name, cross_section_corr=cross_section_corr)
        pi_spec = CxsSpec(shot, chord_IDs=st40_spec['chord_IDs'],
                  amu=plasma_ion_amu, plot_chords=plot_geo, ax=ax,
                  spec_name=spec_name, beam_amu=beam_amu, beam_name=beam_name, cross_section_corr=cross_section_corr,
                  custom_geo_dict=st40_spec["geom_dict"])
        nchan = len(st40_spec['chord_IDs'])

        ids = []
        for id in st40_spec['chord_IDs']:
            ids.append(id.encode(encoding='utf_8'))

        ids = []
        radius = []
        lens = []
        axis = []
        _spot_radius = 1.25  # TODO: estimate spot radius on Princeton foreoptic
        spot_size = []
        _sigma_pi_ratio = 1.  # default sigma/pi ratio
        sigma_pi = []
        for chord in pi_spec.chords:
            ids.append(chord.id.encode(encoding='utf_8'))
            radius.append(chord.tang_rad)
            lens.append(chord.origin)
            axis.append(chord.diruvec)
            spot_size.append(_spot_radius)
            sigma_pi.append(_sigma_pi_ratio)

    # Preprocessing for each participating pini
    # Note, since input dictionaries are modified in preprocessing.py, recreate the same inputs for every pini.
    beam_id = st40_beams["name"]

    spec = {'nchan': nchan,
            'system': spec_name,
            'data_source': 'MDSplus',
            'id': np.asarray(ids),
            'radius': np.asarray(radius),
            'lens': np.asarray(lens).T,
            'axis': np.asarray(axis).T,
            'spot_size': np.asarray(spot_size),
            'sigma_pi': np.asarray(sigma_pi),
            }

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

    # Geometry plot for inspection
    if plot_geo:
        plt.show()

    #fields, rhogrid, btipsign = read_geqdsk(geqdsk_file, grid, poloidal=True)
    equil = dict()
    equil["time"] = time
    equil["br"] = plasmaconfig['br']
    equil["bt"] = plasmaconfig['bt']
    equil["bz"] = plasmaconfig['bz']
    equil["er"] = plasmaconfig['br'] * 0.0
    equil["et"] = plasmaconfig['br'] * 0.0
    equil["ez"] = plasmaconfig['br'] * 0.0

    print(np.shape(plasmaconfig['br']))

    # Interpolate data according to fast particle grid
    for key in equil.keys():
        if key != 'time':
            r_plasma = plasmaconfig['R'][0, :]
            z_plasma = plasmaconfig['z'][:, 0]
            #data_obj = RegularGridInterpolator((r_plasma, z_plasma), equil[key], bounds_error=False, fill_value=0.0)

            from scipy.interpolate import interp2d
            data_obj = interp2d(r_plasma, z_plasma, equil[key], bounds_error=False, fill_value=0.0)

            data_interp = np.zeros((nr, nz))
            for i_z in range(nz):
                for i_r in range(nr):
                    data_interp[i_r, i_z] = data_obj(grid['r2d'][i_r, 0]*1e-2, grid['z2d'][0, i_z]*1e-2) / 2*np.pi

            #data_interp = data_obj((grid['r2d']*1e-2, grid['z2d']*1e-2))
            equil[key] = data_interp

    equil["data_source"] = 'Indica'
    equil["mask"] = np.ones_like(equil['br'], dtype=np.int32)

    # Read the dummy fast-ion distribution
    _fi_dist = h5.File(fi_dist_file, 'r')
    fi_dist = dict()
    fi_dist['type'] = int(_fi_dist['type'][()])
    fi_dist['time'] = _fi_dist['time'][()]
    fi_dist['nenergy'] = int(_fi_dist['nenergy'][()])
    fi_dist['energy'] = _fi_dist['energy'][()]
    fi_dist['npitch'] = int(_fi_dist['npitch'][()])
    fi_dist['pitch'] = _fi_dist['pitch'][()]
    fbm_grid = np.zeros((fi_dist['nenergy'],fi_dist['npitch'],nr,nz))
    # fi_dist['f'] = np.asarray(_fi_dist['f'][()].T.tolist())
    fi_dist['f'] = fbm_grid
    fi_dist['denf'] = _fi_dist['denf'][()]
    fi_dist['data_source'] = str(_fi_dist['data_source'][()])

    # Fast particle positions
    r_fi = _fi_dist['r'][()]
    z_fi = _fi_dist['z'][()]
    r_fi, z_fi = np.meshgrid(r_fi, z_fi)

 

    # Interpolate rho grid
    rhogrid = plasmaconfig['rho']
    print(np.shape(rhogrid))
    r_plasma = plasmaconfig['R'][0, :]
    z_plasma = plasmaconfig['z'][:, 0]
    from scipy.interpolate import interp2d
    data_obj = interp2d(r_plasma, z_plasma, rhogrid, bounds_error=False, fill_value=10.0)
    data_interp = np.zeros((nr, nz))
    for i_z in range(nz):
        for i_r in range(nr):
            data_interp[i_r, i_z] = data_obj(grid['r2d'][i_r, 0] * 1e-2, grid['z2d'][0, i_z] * 1e-2)
    rhogrid = data_interp

    # Interpolate kinetic data
    from scipy.interpolate import interp1d
    dims = rhogrid.shape
    f_zeff = interp1d(plasmaconfig['rho_1d'], plasmaconfig['zeff'], fill_value='extrapolate')
    zeff = f_zeff(rhogrid)
    zeff = np.where(zeff > 1, zeff, 1.0).astype('float64')

    f_te = interp1d(plasmaconfig['rho_1d'], plasmaconfig['te'], fill_value='extrapolate')
    te = f_te(rhogrid)
    te = np.where(te > 0.0, te, 0.0).astype('float64')

    f_ti = interp1d(plasmaconfig['rho_1d'], plasmaconfig['ti'], fill_value='extrapolate')
    ti = f_ti(rhogrid)
    ti = np.where(ti > 0.0, ti, 0.0).astype('float64')

    f_nn = interp1d(plasmaconfig['rho_1d'], plasmaconfig['nn'], fill_value='extrapolate')
    nn = f_nn(rhogrid)
    nn = np.where(nn > 0.0, nn, 0.0).astype('float64')

    f_ne = interp1d(plasmaconfig['rho_1d'], plasmaconfig['ne'], fill_value='extrapolate')
    ne = f_ne(rhogrid)
    ne = np.where(ne > 0.0, ne, 0.0).astype('float64')

    f_omega = interp1d(plasmaconfig['rho_1d'], plasmaconfig['omegator'], fill_value='extrapolate')
    omega = f_omega(rhogrid)
    omega = np.where(omega > 0.0, omega, 0.0).astype('float64')
    vt = grid['r2d'] * omega # cm/s


    #TODO: double check units
    plasma = dict()
    plasma['time'] = time
    plasma['zeff'] = zeff
    plasma['te'] = 1.e-03 * te  # fidasim expects keV
    plasma['ti'] = 1.e-03 * ti  # fidasim expects keV
    plasma['denn'] = 1.e-06 * nn  # fidasim expects cm^-3
    plasma['dene'] = 1.e-06 * ne  # fidasim expects cm^-3
    plasma['vr'] = np.zeros_like(plasma['ti'])
    plasma['vz'] = np.zeros_like(plasma['ti'])
    plasma['vt'] = vt
    plasma['data_source'] = 'Indica'
    max_rho = np.nanmax(np.abs(plasmaconfig['rho_1d']))
    mask = np.zeros_like(plasma['ti'], dtype='int')
    w = np.where(rhogrid <= max_rho) #where we have profiles
    mask[w] = 1
    plasma['mask'] = mask

    debugging_shape = False
    if debugging_shape:
        from matplotlib import pyplot as plt

        plt.figure()
        plt.imshow(mask)
        plt.show()

        plt.figure()
        plt.subplot(121)
        plt.contourf(
            grid["z2d"][0, :],
            grid["r2d"][:, 0],
            plasma['ti'],
        )
        plt.contour(
            grid["z2d"][0, :],
            grid["r2d"][:, 0],
            plasma['ti'],
            [1.0*1e-3],
            colors='k',
        )
        plt.ylim([rmin, rmax])
        plt.xlim([zmin, zmax])
        plt.subplot(122)
        plt.contourf(
            #plasmaconfig['R'][0, :]*1e2,
            #plasmaconfig['z'][:, 0]*1e2,
            grid["z2d"][0, :],
            grid["r2d"][:, 0],
            rhogrid,
        )
        plt.contour(
            #plasmaconfig['R'][0, :]*1e2,
            #plasmaconfig['z'][:, 0]*1e2,
            grid["z2d"][0, :],
            grid["r2d"][:, 0],
            rhogrid,
            [1.0],
            colors='k',
        )
        plt.ylim([rmin, rmax])
        plt.xlim([zmin, zmax])

        plt.figure()
        plt.subplot(131)
        plt.contourf(
            grid['r2d'][:, 0],
            grid['z2d'][0, :],
            equil['br'],
        )
        plt.colorbar()
        plt.subplot(132)
        plt.contourf(
            grid['r2d'][:, 0],
            grid['z2d'][0, :],
            equil['bz'],
        )
        plt.colorbar()
        plt.subplot(133)
        plt.contourf(
            grid['r2d'][:, 0],
            grid['z2d'][0, :],
            equil['bt'],
        )
        plt.colorbar()

        plt.show(block=True)

    ## Transpose magnetic data
    #equil["br"] = np.transpose(equil['br'], (1, 0))
    #equil["bt"] = np.transpose(equil['bt'], (1, 0))
    #equil["bz"] = np.transpose(equil['bz'], (1, 0))
    #equil["er"] = np.transpose(equil['er'], (1, 0))
    #equil["et"] = np.transpose(equil['et'], (1, 0))
    #equil["ez"] = np.transpose(equil['ez'], (1, 0))
    #equil["mask"] = np.transpose(equil['mask'], (1, 0))

    # Add grid and flux to plasma dict
    plasma['grid'] = grid
    plasma['flux'] = rhogrid
    plasma['bgrid'] = bgrid

    # extract omp profiles from 2D plasma grid
    # Assume z grid is up-down symmetric
    plasma['profiles'] = {}
    i_z = int(nz/2)
    plasma['profiles']['ti'] = plasma['ti'][:,i_z]
    plasma['profiles']['te'] = plasma['te'][:,i_z]
    plasma['profiles']['dene'] = plasma['dene'][:,i_z]
    plasma['profiles']['denn'] = plasma['denn'][:,i_z]
    plasma['profiles']['rho'] = plasma['flux'][:,i_z]
    plasma['profiles']['r_omp'] = plasma['grid']['r']

    # manual v_tor profile
    plasma['profiles']['vt'] = plasma['vt'][:,i_z]


    # Create results directory
    case_save_dir = save_dir + '/' + str(shot)
    print(f'save_dir = {save_dir}')
    print(f'case_save_dir = {case_save_dir}')
    if not os.path.exists(case_save_dir):
        os.makedirs(case_save_dir)

    time_str = 't_{:8.6f}'.format(time)
    _case_save_dir = case_save_dir + '/' + time_str
    if not os.path.exists(_case_save_dir):
        os.makedirs(_case_save_dir)

    out_dict['plasma'] = copy.deepcopy(plasma)
    out_dict['plasma']['time'] = str(out_dict['plasma']['time'])
    out_dict['flux'] = copy.deepcopy(rhogrid)
    out_dict['grid'] = copy.deepcopy(grid)
    out_dict['bgrid'] = copy.deepcopy(bgrid)
    convert_to_list(out_dict)
    # Write plasma dictionary in JSON format and save to run directory
    save_plasma_file = _case_save_dir + '/TE-fidasim_plasma.json'
    with open(save_plasma_file, mode='w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2)

    # Create results directory for each beam
    beam_save_dir = _case_save_dir + '/' + beam_id
    if not os.path.exists(beam_save_dir):
        os.makedirs(beam_save_dir)

    general_settings = build_general_settings(shot, time, runid, beam_save_dir, fida_dir)
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
        if beam_id == beam['name']:
            fidasim.prefida(inputs, grid, beam, plasma, equil, fi_dist, spec=spec)

    # If here then preprocessing was successful for this beam. Launch batch job.
    print('Pre-processing complete. Save dir: ', beam_save_dir)

    # submit_fidasim_batch_job(beam_save_dir)


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
        los_type='center'
):
    
    """Collects the fidasim hdf5 results from each pini. Optionally fits cxs spectra, and saves processed output
        to a JSON dictionary.

    Parameters
    ----------
    process_spec : bool
        Flag for collecting and fitting cxs spectra for each pini, as well as the total of all pinis.

    """

    out_dict = {} # Ouptut dictionary containing combined pini results
    time = time
    st40_beams = nbiconfig
    beam_amu = st40_beams['ab']
    beam_name = st40_beams['name']
    st40_spec = specconfig
    runid = pwd.getpwuid(os.getuid())[0]
    spec_name = st40_spec['name']
    cross_section_corr = False
    if 'cross_section_corr' in st40_spec:
        cross_section_corr = st40_spec['cross_section_corr']
    plasma_ion_amu = plasmaconfig['plasma_ion_amu']

    out_dict['amu'] = plasma_ion_amu

    # Configure spec dictionary compatible with fidasim format.
    spec = None
    if spec_name in st40_spec['name']:
        pi_spec = CxsSpec(shot, chord_IDs=st40_spec['chord_IDs'],
                  amu=plasma_ion_amu, beam_amu=beam_amu,  beam_name=beam_name, spec_name=spec_name,
                  cross_section_corr=cross_section_corr,
                  custom_geo_dict=st40_spec["geom_dict"])
        nchan = len(st40_spec['chord_IDs'])

        ids = []
        for id in st40_spec['chord_IDs']:
            ids.append(id.encode(encoding='utf_8'))

        ids = []
        radius = []
        lens = []
        axis = []
        _spot_radius = 1.25  # TODO: estimate spot radius on Princeton foreoptic
        spot_size = []
        _sigma_pi_ratio = 1.  # default sigma/pi ratio
        sigma_pi = []

        ## import LOS data from local pickle file (J Wood 29/07/22)
        #import pickle
        #los_data = pickle.load(open('PI_LOS_geometry_processed.p', 'rb'))
        #los_data = los_data['3POINT_AV']

        for index, chord in enumerate(pi_spec.chords):

            """
            if spec_name == "Princeton":
                if los_type == 'center':
                    origin_new = los_data['CENTER']['ORIGIN']
                    direction_new = los_data['CENTER']['DIRECTION']
                elif los_type == 'lhs':
                    origin_new = los_data['LHS']['ORIGIN']
                    direction_new = los_data['LHS']['DIRECTION']
                elif los_type == 'rhs':
                    origin_new = los_data['RHS']['ORIGIN']
                    direction_new = los_data['RHS']['DIRECTION']
                else:
                    raise ValueError

                print(index)
                print(chord.origin)
                print(chord.diruvec)

                if index > 0:
                    chord.origin = [origin_new[index-1, 0]*100, origin_new[index-1, 1]*100, 0.0]
                    chord.diruvec = [direction_new[index-1, 0], direction_new[index-1, 1], 0.0]
                else:
                    chord.origin = [origin_new[0, 0]*100, origin_new[0, 1]*100, 0.0]
                    chord.diruvec = [-0.90, -0.50, 0.0]
                pi_spec.chords[index] = chord

                print(chord.origin)
                print(chord.diruvec)
                print(' ')
            """

            ids.append(chord.id.encode(encoding='utf_8'))
            radius.append(chord.tang_rad)
            lens.append(chord.origin)
            axis.append(chord.diruvec)
            spot_size.append(_spot_radius)
            sigma_pi.append(_sigma_pi_ratio)

    # run directory        
    time_str = 't_{:8.6f}'.format(time)
    run_dir = save_dir + '/' + str(shot) + '/' + time_str
    plasma_file = run_dir + '/TE-fidasim_plasma.json'

    # Collect fidasim results for each beam and store in output dictionary
    #icnt = 0
    #for beam_id, beam_detail in st40_beams.items():

    beam_save_dir = run_dir + '/' +beam_name

    if not os.path.exists(beam_save_dir):
        print('\033[91m' + 'ERROR: results directory path not found: ' + beam_save_dir + '\033[0m')

    if spec_name in st40_spec['name'] and process_spec:
        spec_file = beam_save_dir + '/' + runid + '_spectra.h5'
        geo_file = beam_save_dir + '/' + runid + '_geometry.h5'
        # dcx_file = beam_save_dir + '/' + runid + '_dcx.h5'
        neut_file = beam_save_dir + '/' + runid + '_neutrals.h5'

        print(spec_file)
        print(geo_file)
        print(neut_file)

        try:
            open(spec_file, 'rb')
        except FileNotFoundError:
            print('\033[91m' + 'ERROR: results spectra file not found: ' + spec_file + '\033[0m')
            sys.exit()

        # Collect results from fidasim
        pi_spec.collect_pini_spectra(beam_name, spec_file, geo_file, neut_file)

        # Using fidasim dcx and halo density, manually perform line-integration as sanity check against fidasim
        pi_spec.los_integrate_pini_brightness(beam_name, beam_save_dir, plasma_file, neut_file)

        # Using fidasim full energy neutral beam density, manually perform CVI line-integration
        # Assume constant C_6+ concetration
        pi_spec.los_integrate_CVI_brightness(beam_name, beam_save_dir, plasma_file, neut_file, block=block)

    export_dict = dict()
    if spec_name in st40_spec['name'] and process_spec:
        # Fit fidasim spectra from individual pini and sum of pinis for Ti, v_tor
        pi_spec.fit_spectra(block=block)

        # Calculate doppler shifts for full, half and third energy components of each pini
        pi_spec.calc_bes_dopp_shifts()

        # Also fit manually line-integrated spectra from individual beam and sum of beams for Ti, v_tor
        # Spectra are generated using the fidasim 3d density plots and the 2D poloidal plasma Ti contours
        pi_spec.fit_spectra(fit_manual_los_integral=True, block=block)
        pi_spec.fit_spectra(fit_manual_cvi_integral=True, block=block, run_dir=run_dir)

        # Save results to JSON dictionary and append to main output dictionary
        out_dict[spec_name] = pi_spec.serialize()
        print()

        # Extract fit data, export as dictionary
        Ti = np.zeros(len(out_dict[spec_name].keys()))
        Ti_err = np.zeros(len(out_dict[spec_name].keys()))
        cwl = np.zeros(len(out_dict[spec_name].keys()))
        cwl_err = np.zeros(len(out_dict[spec_name].keys()))
        vtor = np.zeros(len(out_dict[spec_name].keys()))
        vtor_err = np.zeros(len(out_dict[spec_name].keys()))
        for i_chord, id in enumerate(out_dict[spec_name].keys()):
            Ti[i_chord] = out_dict[spec_name][id]['res'][beam_name]['man_los_integral']['fit_cvi']['Ti']
            Ti_err[i_chord] = out_dict[spec_name][id]['res'][beam_name]['man_los_integral']['fit_cvi']['Ti_err']
            cwl[i_chord] = out_dict[spec_name][id]['res'][beam_name]['man_los_integral']['fit_cvi']['cwl']
            cwl_err[i_chord] = out_dict[spec_name][id]['res'][beam_name]['man_los_integral']['fit_cvi']['cwl_err']

            print(out_dict[spec_name][id]['origin'])
            print(out_dict[spec_name][id]['beam_intersect_pos'])

            # Convert Doppler shift to toroidal rotation
            vtor[i_chord] = get_v_tor_v_pol(
                out_dict[spec_name][id]['origin'],
                np.array(out_dict[spec_name][id]['beam_intersect_pos'][beam_name]),
                529.059 - cwl[i_chord],
                529.059,
            )
            vtor_err[i_chord] = get_v_tor_v_pol(
                out_dict[spec_name][id]['origin'],
                np.array(out_dict[spec_name][id]['beam_intersect_pos'][beam_name]),
                cwl_err[i_chord],
                529.059,
            )

        export_dict['chord_id'] = list(out_dict[spec_name].keys())
        export_dict['Ti'] = Ti
        export_dict['Ti_err'] = Ti_err
        export_dict['cwl'] = cwl
        export_dict['cwl_err'] = cwl_err
        export_dict['vtor'] = vtor
        export_dict['vtor_err'] = vtor_err

    #print(out_dict.keys())
    #print(out_dict['Princeton'].keys())
    #print(out_dict['Princeton']['M5'].keys())
    #print(out_dict['Princeton']['M5']['res'].keys())
    #dict_keys(['res', 'id', 'diruvec', 'origin', 'beam_intersect_pos', 'tang_rad'])
    #print(out_dict['Princeton']['M5']['diruvec'])
    #print(out_dict['Princeton']['M5']['origin'])
    #print(out_dict['Princeton']['M5']['beam_intersect_pos'])
    #print(out_dict['Princeton']['M5']['tang_rad'])
    #print(out_dict['Princeton']['M5']['res']['total'].keys())
    #print(out_dict['Princeton']['M5']['res']['rfx'].keys())
    #print(out_dict['Princeton']['M5']['res']['rfx']['man_los_integral'].keys())
    ##print(out_dict['Princeton']['M5']['res']['rfx']['dopp'].keys())
    #print(out_dict['Princeton']['M5']['res']['rfx']['man_los_integral']['cvi'].keys())
    #print(out_dict['Princeton']['M5']['res']['rfx']['man_los_integral']['fit_cvi'].keys())
    #print(' ')
    #print(out_dict['Princeton']['M5']['res']['rfx']['man_los_integral']['fit_cvi'])
    #print(' ')
    #print(' ')
    #print(out_dict['Princeton']['M5']['res']['rfx']['man_los_integral']['cvi']['lambda'])
    #print(' ')
    #print('aa'**2)

    # Write output dictionary in JSON format and save to run directory
    savefile = run_dir + '/TE-fidasim_output.json'
    with open (savefile, mode='w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2)
    print('Saving post-processed fidasim output to:', savefile)

    if (not block) and debug:
        plt.show(block=True)
    else:
        plt.close('all')

    # Export temperature and velocity results from simulated data
    return export_dict
