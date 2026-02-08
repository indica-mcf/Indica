import os, copy, sys
import numpy as np
import matplotlib.pyplot as plt
from fidasim.utils import rz_grid, extract_transp_plasma, read_geqdsk, nubeam_geometry, beam_grid, read_nubeam
from MDSplus import *
import fidasim
import pandas as pd
from scipy.interpolate import interp1d, LinearNDInterpolator
from itertools import product
from input.beam_geo import get_rfx_geo, get_hnbi_geo

import sys
sys.path.append('/home/jonathan.wood/git_home/Indica')
#import indica.profiles_gauss as profiles
from indica.profilers.profiler_gauss import ProfilerGauss as profiles


PATH_TO_TE_FIDASIM = os.path.dirname(os.path.realpath(__file__))

# import ppf

# Constants
c = 299792458.0  # m/s
amu_kg = 1.67377e-27  # kg
J_to_eV = 6.242e18

# Isotope amu
H_amu = 1.00794
D_amu = 2.0141078
T_amu = 3.0160492


def extract_hda_plasma(
        shot, run, intime, grid, rhogrid, vtor_peak_kms=400,
        ti0=None, wcenter=None, peaking=None,
        vt0=None, vt_wcenter=None, vt_peaking=None
):

    print(f'shot = {shot}')
    print(f'run = {run}')

    tree=Tree('HDA',shot)

    hda_rho=tree.getNode('RUN' + str(run) + '.PROFILES.PSI_NORM.RHOP').data()
    hda_te=tree.getNode('RUN' + str(run) + '.PROFILES.PSI_NORM.TE').data()
    hda_ti=tree.getNode('RUN' + str(run) + '.PROFILES.PSI_NORM.TI').data()
    hda_ne=tree.getNode('RUN' + str(run) + '.PROFILES.PSI_NORM.NE').data()
    hda_ni=tree.getNode('RUN' + str(run) + '.PROFILES.PSI_NORM.NI').data()
    hda_zeff=tree.getNode('RUN' + str(run) + '.PROFILES.PSI_NORM.ZEFF').data()
    hda_t=tree.getNode('RUN' + str(run) + '.TIME').data()

    it = np.argmin(np.abs(hda_t-intime))
    time=hda_t[it]

    # Interpolate onto r-z grid
    dims = rhogrid.shape
    f_dene = interp1d(hda_rho,hda_ne[it,:],fill_value='extrapolate')
    dene = f_dene(rhogrid)
    dene = np.where(dene > 0.0, dene, 0.0).astype('float64')

#    f_denn = interp1d(rho,np.log(transp_nn),fill_value=np.nan,bounds_error=False)
#    log_denn = f_denn(rhogrid)
#    denn = np.where(~np.isnan(log_denn), np.exp(log_denn), 0.0).astype('float64')
    denn = 0*dene

    f_te = interp1d(hda_rho,hda_te[it,:],fill_value='extrapolate')
    te = f_te(rhogrid)
    te = np.where(te > 0, te, 0.0).astype('float64')

    # Use custom Ti profile
    ti_edge = hda_ti[it, -1]
    #plim = (1.5, 5.0)
    #Ti_profiles_list = profiles.scan_profile_peaking(y0=ti0, wcenter=wcenter, plim=plim)
    #Ti_profile = Ti_profiles_list[iprofile]

    #plim = (peaking, peaking)
    #Ti_profiles_list = profiles.scan_profile_peaking(y0=ti0, wcenter=wcenter, plim=plim)
    #Ti_profile = Ti_profiles_list[0]
    #rho = Ti_profile.xspl
    #Ti = Ti_profile.yspl.data
    #Ti_mod = Ti - Ti[-1]  # remove baseline
    #Ti_mod = Ti_mod * (ti0 - ti_edge) / np.max(Ti_mod)
    #Ti_mod = Ti_mod + ti_edge
    ## f_ti = interp1d(hda_rho,hda_ti[it,:],fill_value='extrapolate')
    #f_ti = interp1d(rho, Ti_mod, fill_value='extrapolate')

    print(f'ti0 = {ti0}')
    print(f'vt0 = {vt0}')

    if isinstance(ti0, float):
        plim = (peaking, peaking)
        Ti_profiles_list = profiles.scan_profile_peaking(y0=ti0, wcenter=wcenter, plim=plim)
        Ti_profile = Ti_profiles_list[0]
        rho = Ti_profile.xspl
        Ti = Ti_profile.yspl.data
        Ti_mod = Ti - Ti[-1]  # remove baseline
        Ti_mod = Ti_mod * (ti0 - ti_edge) / np.max(Ti_mod)
        Ti_mod = Ti_mod + ti_edge
        # f_ti = interp1d(hda_rho,hda_ti[it,:],fill_value='extrapolate')
        f_ti = interp1d(rho, Ti_mod, fill_value='extrapolate')
    else:
        f_ti = interp1d(hda_rho, hda_ti[it, :], fill_value='extrapolate')

    ti = f_ti(rhogrid)
    ti = np.where(ti > 0, ti, 0.0).astype('float64')

    f_zeff = interp1d(hda_rho,hda_zeff, fill_value=1.0, bounds_error=False)
    zeff = f_zeff(rhogrid)
    zeff = np.where(zeff > 1, zeff, 1.0).astype('float64')


    # Omega profile from Ti profile
    if isinstance(vt0, float):
        vtor_profiles_list = profiles.scan_profile_peaking(
            y0=vt0*1000, wcenter=vt_wcenter, plim=(vt_peaking, vt_peaking)
        )
        vtor_rho = vtor_profiles_list[0].xspl
        vtor_prof = vtor_profiles_list[0].yspl.data

        vtor_prof_mod = vtor_prof - vtor_prof[-1]
        vtor_prof = np.max(vtor_prof) * vtor_prof_mod / np.max(vtor_prof_mod)
        R_core = 0.5
        hda_omega = vtor_prof / R_core

        # print(f'vtor_rho = {vtor_rho}')
        # print(f'hda_omega = {hda_omega}')
        # print('aa'**2)

        f_omega = interp1d(vtor_rho, hda_omega, fill_value='extrapolate')
        omega = f_omega(rhogrid)
    else:
        vtor_peak_kms = 240.0
        vtor_core = vtor_peak_kms *1000 # m/s
        R_core = 0.5 #m
        omega_core = vtor_core / R_core

        hda_omega = omega_core * hda_ti[it,:] / np.max(hda_ti[it,:])

        factor = omega_core / np.max(hda_ti[it,:])
        #print(f'omega_core = {omega_core}')
        #print(f'np.max(hda_ti[it,:]) = {np.max(hda_ti[it,:])}')
        #print(f'factor = {factor}')
        #print('aa'**2)

        flag = False
        if flag:
            ind = np.linspace(1.0, 0.0, len(hda_ti[it,:]), dtype=float)
            hda_omega = hda_omega * ind

        f_omega = interp1d(hda_rho,hda_omega,fill_value='extrapolate')
        omega = f_omega(rhogrid)

    omega = np.where(omega > 0, omega, 0.0).astype('float64')
    vt = grid['r2d']*omega # cm/s


    # Use Ti profile shape for Vtor
    # vtor_scal = 400.0*1000*100 # cm/s
    # vt = vtor_scal*ti/np.max(ti)
    # Interpolate onto r-z grid
    # f_vtor = interp1d(vtors_prof_ms_rho,1.e02*np.asarray(vtor_prof_ms_vt),
    #                   fill_value='extrapolate') # cm/s
    # vt = f_vtor(rhogrid)
    # vt = np.where(vt > 0.0, vt, 0.0).astype('float64')
    # Convert vt to rad/s with grid['r2d'] and back to cm/s 
    # vt = grid['r2d']*(vt/grid['r2d']) # cm/s

    # Debugging
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # cf = ax.imshow(vt.T, aspect='auto')
    # # cbar = fig.colorbar(cf, ax=ax)
    # # plt.plot(grid['r2d'][:,50], vt[:,50])
    # ax.plot(rhogrid[:,50],omega[:,50])
    # plt.show()


#    f_omega = interp1d(rho,transp_omega,fill_value='extrapolate')
    # vt = grid['r2d']*f_omega(rhogrid).astype('float64') # rad/s
    # vt = np.zeros(dims,dtype='float64')
    vr = np.zeros(dims,dtype='float64')
    vz = np.zeros(dims,dtype='float64')

    max_rho = max(abs(hda_rho))

    mask = np.zeros(dims,dtype='int')
    w = np.where(rhogrid <= max_rho) #where we have profiles
    mask[w] = 1

    # SAVE IN PROFILES STRUCTURE
    plasma={"data_source":"HDA " + str(shot) + " RUN" + str(run),"time":time,
            "mask":mask,"dene":dene,"denn":denn,"te":te,"ti":ti,
            "vr":vr,"vt":vt,"vz":vz,"zeff":zeff}

    return plasma


def convert_to_list(resdict):
    """Recursively search nested dict for arrays and convert to list.

    """

    for key, value in resdict.items():
        if isinstance(value, dict):
            convert_to_list(value)
        else:
            if isinstance(value, np.ndarray):
                resdict[key]=value.tolist()

    return

def Tait_Bryan_rotate(point_xyz, alpha, beta, gamma):
    """
        https://en.wikipedia.org/wiki/Davenport_chained_rotations
        Section: Tait-Bryan chained rotations
    """

    alpha_yaw = np.array((
        np.array((np.cos(alpha),-1.*np.sin(alpha),0)),
        np.array((np.sin(alpha), np.cos(alpha), 0)),
        np.array((0, 0, 1)),
    ))
    beta_pitch = np.array((
        np.array((np.cos(beta),0,np.sin(beta))),
        np.array((0, 1, 0)),
        np.array((-1.*np.sin(beta), 0, np.cos(beta))),
    ))
    gamma_roll = np.array((
        np.array((1,0,0)),
        np.array((0, np.cos(gamma), -1.*np.sin(gamma))),
        np.array((0, np.sin(gamma), np.cos(gamma))),
    ))

    M = np.matmul(gamma_roll,np.matmul(beta_pitch, alpha_yaw))
    rot_point = np.matmul(M,point_xyz)

    return rot_point
    # return np.dot(gamma_roll, np.dot(beta_pitch, np.dot(alpha_yaw, point_xyz)))


def gaussian_params_to_Ti(cwl, std, amu):

    """Get Ti from gaussian parameters

    Parameters
    ----------
    cwl :
        Central wevelength (nm).
    std :
        Gaussian width (nm)
    amu :
        Atomic mass units of ion.

    Returns
    -------
    Ti_eV :
        Ion temperature in eV

    """

    # c = 299792458.0 # m/s
    # amu_kg = 1.67377e-27 # kg

    # Calculate Ti using Maxwellian
    Ti_J = 0.5 * amu * amu_kg * c * c * (std/cwl)**2
    Ti_eV = J_to_eV * Ti_J

    return Ti_eV


def Ti_to_gaussian_params_to_Ti(Ti_eV, amu, cwl):

    """Get Ti from gaussian parameters

    Parameters
    ----------
    Ti_eV :
        Ion temperature in eV
    cwl :
        Central wevelength (nm).
    amu :
        Atomic mass units of ion.

    Returns
    -------

    std :
        Gaussian width (nm)

    """

    # Calculate Ti using Maxwellian
    std = np.sqrt(Ti_eV/ ( J_to_eV * 0.5 * amu * amu_kg * c * c)) * cwl

    return std


def gaussian(x, offset, amp, std, cwl):

    return offset + amp/(std*np.sqrt(np.pi)) * np.exp( -1*(x-cwl)**2 / (std*std) )


def get_H_isotope_cwl(amu):
    """ Return hydrogen isotope central wavelength in nm. """

    epsilon = 1.e-03

    if (H_amu - amu) / amu < epsilon:  # Hydrogen
        cwl = 656.283
    elif (D_amu - amu) / amu < epsilon:  # Deuterium
        cwl = 656.093
    elif (T_amu - amu) / amu < epsilon:  # Tritium
        cwl = 656.039
    else:
        cwl = 656.093  # Deuterium by default

    return cwl


# intersection function
# https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: define the line
    p_co, p_no: define the plane:
        p_co is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction;
             (does not need to be normalized).

    return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # the factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)
    else:
        # The segment is parallel to plane
        return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
        )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
        )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
        )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
        )


def get_v_tor_v_pol(los_origin, los_pini_intersect, del_lambda, cwl):
    """Converts the observed doppler shift in the spectal line central wavelength
        to a toroidal velocity
    """

    # machine axis vector pointing down
    mach_axis = np.array((0,0,1))

    # tangency vector relative to machine center at pini intersection point
    tan_vec = np.cross(mach_axis, los_pini_intersect)
    # tan_vec = np.cross(los_pini_intersect, mach_axis)

    # line-of-sight vector
    los_vec = los_origin - los_pini_intersect

    # From dot product of two vectors
    gamma = np.arccos(np.dot(los_vec,tan_vec)/(np.sqrt(np.dot(los_vec,los_vec))*np.sqrt(np.dot(tan_vec,tan_vec))))

    # print('gamma', gamma*180./np.pi)

    # Doppler shift (km/s)
    try:
        v_tor = 1.0e-03*c*(del_lambda/cwl)/np.cos(gamma)
    except TypeError:
        v_tor = np.nan

    return v_tor


def get_del_lambda(los_origin, los_pini_intersect, vtor_kms, cwl):
    """Converts the toroidal velocity to a delta lambda offset from the spectal line 
        central wavelength
    """

    # machine axis vector pointing down
    mach_axis = np.array((0,0,1))

    los_origin = np.asarray(los_origin)
    los_pini_intersect = np.asarray(los_pini_intersect)

    # tangency vector relative to machine center at pini intersection point
    tan_vec = np.cross(mach_axis, los_pini_intersect)
    # tan_vec = np.cross(los_pini_intersect, mach_axis)

    # line-of-sight vector
    los_vec = los_origin - los_pini_intersect

    # From dot product of two vectors
    gamma = np.arccos(np.dot(los_vec,tan_vec)/(np.sqrt(np.dot(los_vec,los_vec))*np.sqrt(np.dot(tan_vec,tan_vec))))


    # Delta lambda (nm)

    del_lambda = 1.0e03 * cwl * vtor_kms * np.cos(gamma) / c

    return del_lambda



def read_cxs_spec_geometry(shot, origin=[107.7264, -36.442098, 0],
                            spec_name='Princeton'):
    """Read cxs spec geometry and store in dictionary.
       
    Parameters
    ----------
    shot : int
       
    Returns
    -------
    geo_dict : dict
        Dictionary for storing geometry. Output units in cm!
    berr : bool
        True if error encountered, False otherwise.
       
    """

    path = PATH_TO_TE_FIDASIM + '/input/spec_geom/' + spec_name + '/'

    folder = os.fsencode(path)

    shot_valid = False
    geo_found = False
    _filename = None

    for file in os.listdir(folder):
        filename = os.fsencode(file)
        if not os.path.isdir(folder+filename):
            dum1, start_shot, end_shot, dum2 = filename.split(b'.')
            #if shot >= int(start_shot) and shot <= int(end_shot):
            shot_valid = True
            geo_found = True
            _filename = filename

    if not shot_valid:
        print('Princeton geo files not found for shot: ', shot)
        return None, True


    geo_dict = {}


    if geo_found and _filename:
        filepath = path + '/' + _filename.decode('utf-8')

        # Read in csv
        csv_data = np.genfromtxt(filepath, delimiter=',', dtype=None)

        origin = []
        for i in range(len(csv_data)):

            # get origin from first line
            if i == 0:
                origin = [csv_data[i][1],csv_data[i][2], csv_data[i][3]]
            elif i == 4 and spec_name == 'Chers_mod':
                origin = [csv_data[i][1],csv_data[i][2], csv_data[i][3]]
            else:
                los_name = csv_data[i][0].decode('utf-8-sig') # https://stackoverflow.com/questions/17912307/u-ufeff-in-python-string

                if los_name not in geo_dict:

                    geo_dict[los_name] = {'origin': origin, 'diruvec': [
                        csv_data[i][1],csv_data[i][2], csv_data[i][3]]
                        }

            if spec_name == 'Chers_new':
                if (i == 0) or (i == 2) or (i == 4):
                    origin = [csv_data[i][1],csv_data[i][2], csv_data[i][3]]
                else:
                    los_name = csv_data[i][0].decode(
                        'utf-8-sig')  # https://stackoverflow.com/questions/17912307/u-ufeff-in-python-string

                    if los_name not in geo_dict:

                        geo_dict[los_name] = {'origin': origin, 'diruvec': [
                            csv_data[i][1],csv_data[i][2], csv_data[i][3]]
                            }

    return geo_dict, False




def interp_adf12_qef(ein, dion, tion, zeff, bmag, transition='8-7'):
    """Interp adf12 QEF blocks .
    
     Parameters
     ----------
    
     Returns
     -------
    
     """
    # eion   20000 # eV/amu
    # dion   = 1.0e13 # cm^-3
    # tion   = 5.0e3 # eV
    # zeff   = 2.5
    # bmag   = 3.0 #T

    # read data from n=1 and n=2 adf12 files
    file_n1 = '/adf12/qef93#h_c6.dat'
    qef, berr = read_adf12_qef(file_n1, transition='8-7', level='n1')
    qef_n1 = qef['8-7']['n1']
    qef_n2 = qef['8-7']['n2']
    qefref_n1 = qef_n1['qefref']
    qefref_n2 = qef_n2['qefref']

    # interpolate

    f = interp1d(qef_n1['ener'], qef_n1['qener'], fill_value='extrapolate')
    _qener_n1 = float(f(ein))
    f = interp1d(qef_n1['tiev'], qef_n1['qtiev'], fill_value='extrapolate')
    _qtiev_n1 = float(f(tion))
    f = interp1d(qef_n1['densi'], qef_n1['qdensi'], fill_value='extrapolate')
    _qdensi_n1 = float(f(dion))
    f = interp1d(qef_n1['zeff'], qef_n1['qzeff'], fill_value='extrapolate')
    _qzeff_n1 = float(f(zeff))
    _qbmag_n1 = qef_n1['qbmag']

    qef_n1 = qefref_n1 * _qener_n1/qefref_n1  *   _qtiev_n1/qefref_n1  *  _qdensi_n1/qefref_n1  * _qzeff_n1/qefref_n1 * _qbmag_n1/qefref_n1

    f = interp1d(qef_n2['ener'], qef_n2['qener'], fill_value='extrapolate')
    _qener_n2 = float(f(ein))
    f = interp1d(qef_n2['tiev'], qef_n2['qtiev'], fill_value='extrapolate')
    _qtiev_n2 = float(f(tion))
    f = interp1d(qef_n2['densi'], qef_n2['qdensi'], fill_value='extrapolate')
    _qdensi_n2 = float(f(dion))
    f = interp1d(qef_n2['zeff'], qef_n2['qzeff'], fill_value='extrapolate')
    _qzeff_n2 = float(f(zeff))
    _qbmag_n2 = qef_n2['qbmag']

    qef_n2 = qefref_n2 * _qener_n2/qefref_n2  *   _qtiev_n2/qefref_n2  *  _qdensi_n2/qefref_n2  * _qzeff_n2/qefref_n2 * _qbmag_n2/qefref_n2


    return qef_n1, qef_n2  # cm^3 s^-1


def read_adf12_qef(adf12file, transition='8-7', level='n1'):
    """Read adf12 QEF blocks and store in dict.

     Parameters
     ----------
     file : adf12 data file
     transition: atomic transition corresponding to CX spectral line

     Returns
     -------
     qef_dict : dict
         Dictionary for storing oct1 and oct7 geometry. Output units in cm!
     berr : bool
         True if error encountered, False otherwise.

     """

    berr = False

    # Find all catia geo files and check that given pulse is valid
    path = 'adf12/'

    # filepath = path + adf12file

    # f = open(filepath, 'r')
    # lines = f.readlines()

    qef_dict = {}
    qef_dict[transition]= {}

    qef_dict[transition]['n1'] = {'qefref':9.91e-09,
                            'parmref': np.array((4.00e+04, 5.00e+03,  2.50e+13,  2.00e+00,  3.00e+00)),
        'ener': np.array((1.00e+03, 1.50e+03, 2.00e+03, 3.00e+03, 5.00e+03, 7.00e+03,
                            1.00e+04, 1.50e+04, 2.00e+04, 3.00e+04, 4.00e+04, 5.00e+04,
                            6.00e+04, 7.00e+04, 8.00e+04, 1.00e+05, 1.50e+05, 2.00e+05,
                            3.00e+05)),
        'qener': np.array((1.19e-11, 1.90e-11, 2.54e-11, 3.52e-11, 6.93e-11, 1.38e-10,
                            2.84e-10, 6.77e-10, 1.50e-09, 4.81e-09, 9.91e-09, 1.24e-08,
                            1.23e-08, 1.16e-08, 1.03e-08, 7.26e-09, 2.53e-09, 1.05e-09,
                            2.35e-10)),
        'tiev': np.array((1.00e+03, 2.00e+03, 3.00e+03, 5.00e+03, 7.00e+03, 1.00e+04,
                              1.30e+04, 1.60e+04, 1.90e+04, 2.20e+04, 2.50e+04, 3.00e+04)),
        'qtiev': np.array((8.73e-09, 9.21e-09, 9.51e-09, 9.91e-09, 1.02e-08, 1.05e-08,
                              1.07e-08, 1.09e-08, 1.10e-08, 1.11e-08, 1.12e-08, 1.14e-08)),
        'densi': np.array((1.00e+11, 2.00e+11, 3.00e+11, 5.00e+11, 7.00e+11, 1.00e+12,
                            2.00e+12, 3.00e+12, 5.00e+12, 7.00e+12, 1.00e+13, 2.00e+13,
                            2.50e+13, 3.00e+13, 5.00e+13, 7.00e+13, 1.00e+14)),
        'qdensi': np.array((1.39e-08, 1.40e-08, 1.41e-08, 1.42e-08, 1.42e-08, 1.42e-08,
                            1.40e-08, 1.37e-08, 1.31e-08, 1.25e-08, 1.19e-08, 1.04e-08,
                            9.91e-09, 9.51e-09, 8.43e-09, 7.80e-09, 7.22e-09)),
        'zeff': np.array((1.00e+00, 1.20e+00, 1.50e+00, 1.70e+00, 2.00e+00, 2.50e+00,
                          3.00e+00, 3.50e+00, 4.00e+00, 4.50e+00, 5.00e+00, 6.00e+00)),
        'qzeff': np.array((1.26e-08, 1.19e-08, 1.10e-08, 1.04e-08, 9.91e-09, 8.93e-09,
                               8.31e-09, 7.75e-09, 7.38e-09, 7.04e-09, 6.82e-09, 6.45e-09)),
        'bmap': 3.0, 'qbmag': 9.91e-09
            }

    qef_dict[transition]['n2'] = {'qefref':3.55e-08,
                'parmref': np.array((4.00e+04, 5.00e+03, 2.50e+13, 2.00e+00, 3.00e+00)),
                'ener': np.array((1.00e+03, 1.50e+03, 2.00e+03, 3.00e+03, 5.00e+03, 7.00e+03,
                        1.00e+04, 1.50e+04, 2.00e+04, 3.00e+04, 4.00e+04, 5.00e+04,
                        6.00e+04, 7.00e+04, 8.00e+04, 9.00e+04, 1.00e+05)),
                'qener': np.array((3.56e-07, 4.66e-07, 5.74e-07, 6.90e-07, 7.78e-07, 7.56e-07,
                        6.49e-07, 4.92e-07, 2.43e-07, 8.59e-08, 3.55e-08, 2.17e-08,
                        1.09e-08, 6.68e-09, 3.90e-09, 2.64e-09, 1.33e-09)),
                'tiev': np.array((1.00e+03, 2.00e+03, 3.00e+03, 5.00e+03, 7.00e+03, 1.00e+04,
                                  1.30e+04, 1.60e+04, 1.90e+04, 2.20e+04, 2.50e+04, 3.00e+04)),
                'qtiev': np.array((2.96e-08, 3.19e-08, 3.35e-08, 3.55e-08, 3.70e-08, 3.87e-08,
                                   3.99e-08, 4.09e-08, 4.18e-08, 4.25e-08, 4.32e-08, 4.41e-08)),
                'densi': np.array((1.00e+11, 2.00e+11, 3.00e+11, 5.00e+11, 7.00e+11, 1.00e+12,
                        2.00e+12, 3.00e+12, 5.00e+12, 7.00e+12, 1.00e+13, 2.00e+13,
                        2.50e+13, 3.00e+13, 5.00e+13, 7.00e+13, 1.00e+14)),
                'qdensi': np.array((7.70e-08, 7.64e-08, 7.57e-08, 7.44e-08, 7.31e-08, 7.12e-08,
                        6.61e-08, 6.21e-08, 5.61e-08, 5.18e-08, 4.71e-08, 3.82e-08,
                        3.55e-08, 3.35e-08, 2.83e-08, 2.55e-08, 2.30e-08)),
                'zeff': np.array((1.00e+00, 1.20e+00, 1.50e+00, 1.70e+00, 2.00e+00, 2.50e+00,
                                  3.00e+00, 3.50e+00, 4.00e+00, 4.50e+00, 5.00e+00, 6.00e+00)),
                'qzeff': np.array((5.22e-08, 4.74e-08, 4.15e-08, 3.84e-08, 3.55e-08, 3.07e-08,
                                   2.78e-08, 2.53e-08, 2.37e-08, 2.22e-08, 2.13e-08, 1.98e-08)),
                'bmap': 3.0, 'qbmag': 3.55e-08

                    }

    return qef_dict, berr



def multidim_interpolator(coords, data, point):
    dims = len(point)
    indices = []
    sub_coords = []
    for j in range(dims) :
        idx = np.digitize([point[j]], coords[j])[0]
        indices += tuple([[idx - 1, idx]])
        sub_coords += [coords[j][indices[-1]]]
    indices = tuple(np.array([j for j in product(*indices)]))
    sub_coords = np.array([j for j in product(*sub_coords)])
    sub_data = data[list(np.swapaxes(tuple(indices), 0, 1))]
    li = LinearNDInterpolator(sub_coords, sub_data)
    return li([point])[0]



def create_st40_beam_grid(beam, plot_bgrid=False, ax=None, delta_src=0.0, delta_ang=0.0):
    """Fidasim beam grid creation for ST-40 beams.

    Beam grid is static regardless of the beams used for a given pulse. This is necessary for superposition
    of multi-beam runs.


    Parameters
    ----------


    Returns
    -------

    """

    rfx = get_rfx_geo()
    hnbi = get_hnbi_geo()

    #print(rfx["src"])
    #print(rfx["axis"])
    #print('aa'**2)

    # Modify RFX source position
    delta = delta_src
    norm_angle = np.arctan2(rfx["axis"][1], rfx["axis"][0]) + np.pi/2
    dx = delta * np.cos(norm_angle)
    dy = delta * np.sin(norm_angle)
    rfx["src"][0] = rfx["src"][0] + dx*100
    rfx["src"][1] = rfx["src"][1] + dy*100

    # Modify RFX angle
    rfx_angle = np.arctan2(rfx["axis"][1], rfx["axis"][0])
    delta_ang = delta_ang
    rfx_angle_new = rfx_angle + delta_ang
    axis_new = np.array([np.cos(rfx_angle_new), np.sin(rfx_angle_new), 0.0])
    rfx["axis"] = axis_new



    # rfx = dict()
    # hnbi = dict()

    # rfx["name"] = "rfx"
    # rfx["shape"] = 2
    # rfx["data_source"] = "RFX DNBI & HNBI - 18062019 - VER. 2.pdf"
    # rfx["src"] = 100*np.array([-2.0199,-2.6323,0.0])
    # tangency = 100*np.array([0.2985,-0.2955,0.0])
    # rfx["axis"] = (tangency-rfx["src"]) / np.linalg.norm(tangency-rfx["src"])
    # rfx["widy"] = 17.2
    # rfx["widz"] = 17.2
    # # rfx["widy"] = 1. # narrow beams
    # # rfx["widz"] = 1. # narrow beams
    # rfx["divy"] = np.array([0.014,0.014,0.014])
    # rfx["divz"] = np.array([0.014,0.014,0.014])
    # # rfx["divy"] = 1.5*np.array([0.014,0.014,0.014])
    # # rfx["divz"] = 1.5*np.array([0.014,0.014,0.014])
    # # rfx["divy"] = np.array([0.0014,0.0014,0.0014]) # narrow beams
    # # rfx["divz"] = np.array([0.0014,0.0014,0.0014]) # narrow beams
    # rfx["focy"] = 160.0
    # rfx["focz"] = 160.0
    # # rfx["focy"] = 1.5*140.0
    # # rfx["focz"] = 1.5*140.0
    # # rfx["focy"] = 300.0 # narrow beams
    # # rfx["focz"] = 300.0 # narrow beams
    # rfx["naperture"] = 0
    # # inputs["pinj"] = 0.5
    # # inputs["einj"] = 23.0
    # # inputs["current_fractions"] = np.array([0.37,0.29,0.34])

    # hnbi["name"] = "hnbi"
    # hnbi["shape"] = 2
    # hnbi["data_source"] = "RFX DNBI & HNBI - 18062019 - VER. 2.pdf"
    # hnbi["src"] = 100*np.array([3.322,3.945,0.0])
    # tangency = 100*np.array([-0.2985,0.2955,0.0])
    # hnbi["axis"] = (tangency-hnbi["src"]) / np.linalg.norm(tangency-hnbi["src"])
    # hnbi["widy"] = 25.0
    # hnbi["widz"] = 25.0
    # hnbi["divy"] = np.array([0.014,0.014,0.014])
    # hnbi["divz"] = np.array([0.014,0.014,0.014])
    # hnbi["focy"] = 420.0
    # hnbi["focz"] = 420.0
    # hnbi["naperture"] = 0
    # # inputs["pinj"] = 0.6
    # # inputs["einj"] = 55.0
    # # inputs["current_fractions"] = np.array([0.64,0.25,0.11])

    nbi_list = [rfx, hnbi]

    if ax:
        for beam in nbi_list:
            ax.scatter(beam['src'][0], beam['src'][1], beam['src'][2], marker='x', color='k')
            pini_len = 1000
            pinix = beam['src'][0] + beam['axis'][0] * pini_len
            piniy = beam['src'][1] + beam['axis'][1] * pini_len
            piniz = beam['src'][2] + beam['axis'][2] * pini_len
            # ax.scatter(pinix, piniy, piniz, marker='o', color='k')
            ax.plot3D([beam['src'][0], pinix], [beam['src'][1], piniy], zs=[beam['src'][2], piniz], color='r')

    rstart = 100  # [cm]

    if beam.upper() == 'RFX':
        bgrid = beam_grid(rfx, rstart,
                          length = 250.0,
                          width = 250.0,
                          height = 50.0,
                          dv=2.0,) # default 8)
                          #plot_grid=True, ax=ax, plot_color='k')
    else:
        bgrid = beam_grid(hnbi, rstart,
                          length = 250.0,
                          width = 250.0,
                          height = 50.0,
                          dv=2.0,) # default 8)
                          #plot_grid=True, ax=ax, plot_color='k')

    return bgrid, nbi_list



def plot_cxs_spec_chords(geo_dict, chord_IDs, ax, plot2d=False):
    los_len = 400

    for chord in chord_IDs:
        for key, item in geo_dict.items():
            if chord == key:
                if plot2d:
                    ax.plot([item['origin'][0], item['origin'][0] + item['diruvec'][0] * los_len],
                              [item['origin'][1], item['origin'][1] + item['diruvec'][1] * los_len],
                             color='r')
                else:
                    ax.plot3D([item['origin'][0], item['origin'][0] + item['diruvec'][0] * los_len],
                              [item['origin'][1], item['origin'][1] + item['diruvec'][1] * los_len],
                              zs=[item['origin'][2], item['origin'][2] + item['diruvec'][2] * los_len],
                              color='r')




if __name__=='__main__':
    print()
