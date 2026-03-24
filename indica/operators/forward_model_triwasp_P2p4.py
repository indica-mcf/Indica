import h5py
import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline
from matplotlib import pyplot as plt
#from matplotlib import cm
import sys
import matplotlib.cm as cm
import pickle
import scipy as sp
import scipy.constants as constants

# Import adf reader
sys.path.append("/home/jwood/git_home/read_adf/")
#import atomic_data as atomic


# Indica imports
from indica.converters import LineOfSightTransform
import indica.physics as ph

# Import local functions
import utility_functions as utils


# Inputs ----------------------------------------------------------------------
pulse = 13475
time = 0.09
user = "jwood"
beam_name = "hnbi"
beam_energy = 52.0 * 1e3
plasma_file = "plasma_pkl_files/13475_indica_plasma_C_Ar_Li3_He2.pkl"
path_to_read_adf = "/home/jwood/git_home/read_adf"
which_impurity = "helium"  # e.g. "helium", "lithium", "carbon", "neon", "boron"
# concentration = 0.005

# fidasim_run = "13475_TWS_P2p4_version_1"
# geometry_file = "geometry_pkl_files/TriWaSp_geometry_6los_50-75_sector1_centre.pkl"
# focal_length = -10.0  # meter
# spot_width = 0.010  # meter
# spot_height = 0.010  # meter

fidasim_run = "13475_TWS_P2p4_version_2"
geometry_file = "geometry_pkl_files/TriWaSp_geometry_7los_50-77_sector1.pkl"
focal_length = -0.03995269  # meter
spot_width = 1.1 * 1e-3  # meter
spot_height = 1.1 * 1e-3  # meter

machine_dims = ((0.15, 1.00), (-0.75, 0.75))
spot_shape = "round"
beamlets_method = "adaptive"
n_beamlets = 9
plot_beamlets = False
# -----------------------------------------------------------------------------

# Read Neutral beam data ------------------------------------------------------
path_to_fidasim_results = f"/home/{user}/fidasim_output"
file_to_read = f"{path_to_fidasim_results}/{fidasim_run}/t_{time:0.6f}/{beam_name}/"
h5 = h5py.File(file_to_read + f"{user}_neutrals.h5")
geom = h5py.File(file_to_read + f"{user}_geometry.h5")

# Get Neutral beam densities for full, half and third energy components
fdens = np.array(h5["fdens"]) * 1e6  # Full energy neutral density (z, y, x, excitation level), m-3
hdens = np.array(h5["hdens"]) * 1e6  # 1/2 energy neutral density (z, y, x, excitation level), m-3
tdens = np.array(h5["tdens"]) * 1e6  # 1/3 energy neutral density (z, y, x, excitation level), m-3

# Get neutral density grid
grid = h5["grid"]
xgrid = np.array(grid["x"]) * 1e-2  # (m)
ygrid = np.array(grid["y"]) * 1e-2  # (m)
zgrid = np.array(grid["z"]) * 1e-2  # (m)

# Use the ground state neutral density, index=0
fdens_summed = fdens[:, :, :, 0]
hdens_summed = hdens[:, :, :, 0]
tdens_summed = tdens[:, :, :, 0]

# Rotate to ST40 axis
with open(file_to_read + f"{user}_inputs.dat", "r") as f:
    lines = f.readlines()
origin_x_str = lines[78]
origin_y_str = lines[79]
origin_z_str = lines[80]

origin_x_str = origin_x_str.replace(" ", "")
origin_x_lst = origin_x_str.split("=")
origin_x = float(origin_x_lst[1].split('!')[0]) * 1e-2

origin_y_str = origin_y_str.replace(" ", "")
origin_y_lst = origin_y_str.split("=")
origin_y = float(origin_y_lst[1].split('!')[0]) * 1e-2

origin_z_str = origin_z_str.replace(" ", "")
origin_z_lst = origin_z_str.split("=")
origin_z = float(origin_z_lst[1].split('!')[0]) * 1e-2

src = [origin_x, origin_y, origin_z]
axis = list(geom['nbi']['axis'])

# Beam angle
beam_angle = np.arctan2(axis[1], axis[0])

# Beam centre line
xb = np.array([geom['nbi']['src'][0] * 1e-2, geom['nbi']['src'][0] * 1e-2 + 5.0 * axis[0]])
yb = np.array([geom['nbi']['src'][1] * 1e-2, geom['nbi']['src'][1] * 1e-2 + 5.0 * axis[1]])

# Rotate grid
xgrid2d, ygrid2d = np.meshgrid(xgrid, ygrid)
xgrid_rot = np.zeros_like(xgrid2d)
ygrid_rot = np.zeros_like(ygrid2d)
for i_x in range(len(xgrid)):
    for i_y in range(len(ygrid)):
        vec = utils.rotate(xgrid[i_x], ygrid[i_y], 0.0, 0.0, beam_angle)
        x_new = vec[0] + origin_x
        y_new = vec[1] + origin_y
        xgrid_rot[i_y, i_x] = x_new
        ygrid_rot[i_y, i_x] = y_new

# Interpolate for the neutral beam density
i_z = np.argmin(np.abs(zgrid - 0.0))
fdens_obj = interp2d(xgrid, ygrid, fdens_summed[i_z, :, :], bounds_error=False, fill_value=0.0)
hdens_obj = interp2d(xgrid, ygrid, hdens_summed[i_z, :, :], bounds_error=False, fill_value=0.0)
tdens_obj = interp2d(xgrid, ygrid, tdens_summed[i_z, :, :], bounds_error=False, fill_value=0.0)
# fdens_obj = RectBivariateSpline(xgrid, ygrid, fdens_summed[i_z, :, :])
# hdens_obj = RectBivariateSpline(xgrid, ygrid, hdens_summed[i_z, :, :])
# tdens_obj = RectBivariateSpline(xgrid, ygrid, tdens_summed[i_z, :, :])
# -----------------------------------------------------------------------------

# Read LOS --------------------------------------------------------------------
cx_geom = pickle.load(open(geometry_file, "rb"))
origin = cx_geom["origin"]
direction = cx_geom["direction"]
x_pos = cx_geom["x_pos"]
y_pos = cx_geom["y_pos"]
r_pos = np.sqrt(x_pos**2 + y_pos**2)

los_transform = LineOfSightTransform(
    origin[:, 0],
    origin[:, 1],
    origin[:, 2],
    direction[:, 0],
    direction[:, 1],
    direction[:, 2],
    name="TriWaSp_P2p4",
    dl=0.01,
    spot_width=spot_width,
    spot_height=spot_height,
    spot_shape=spot_shape,
    beamlets_method=beamlets_method,
    n_beamlets=n_beamlets,
    focal_length=focal_length,
    machine_dimensions=machine_dims,
    passes=1,
    plot_beamlets=plot_beamlets,
)
# -----------------------------------------------------------------------------

# Read Plasma -----------------------------------------------------------------
plasma = pickle.load(open(plasma_file, "rb"))
plasma_impurities = plasma.impurities
impurity_concentration = plasma.impurity_concentration

los_transform.set_equilibrium(plasma.equilibrium)
# -----------------------------------------------------------------------------

# Read Atomic Data ------------------------------------------------------------
d0 = atomic.Atom(element='H', mass_number=2, charge_state=0)
d1 = atomic.Atom(element='H', mass_number=2, charge_state=1)
d0_n2 = atomic.Atom(element='H', mass_number=2, charge_state=0, n_level=2)
h0_n3_2 = atomic.Transition(atom=atomic.Atom(element='H', charge_state=0), n_upper=3, n_lower=2)

c6 = atomic.Atom(element='C', charge_state=6)
b5 = atomic.Atom(element='B', charge_state=5)
ne10 = atomic.Atom(element='Ne', charge_state=10, mass_number=20)
li3 = atomic.Atom(element='Li', charge_state=3)
he2 = atomic.Atom(element='He', charge_state=2)
ar18 = atomic.Atom(element='Ar', charge_state=18)
b4_n7_6 = atomic.Transition(atom=atomic.Atom(element='B', charge_state=4), n_upper=7, n_lower=6)
c5_n8_7 = atomic.Transition(atom=atomic.Atom(element='C', charge_state=5), n_upper=8, n_lower=7)
ne10_n11_10 = atomic.Transition(atom=atomic.Atom(element='Ne', charge_state=9, mass_number=20), n_upper=11,
                                n_lower=10)
li2_n5_4 = atomic.Transition(atom=atomic.Atom(element='Li', charge_state=2), n_upper=5, n_lower=4)  # N =  4 -  5    4498.9
li2_n7_5 = atomic.Transition(atom=atomic.Atom(element='Li', charge_state=2), n_upper=7, n_lower=5)  # N =  5 -  7    5166.7
he2_n4_3 = atomic.Transition(atom=atomic.Atom(element='He', charge_state=1), n_upper=4, n_lower=3)

qcx = atomic.Charge_Exchange_Emission_Rate()
bmp = atomic.Excited_Beam_Population()

if which_impurity == "carbon":

    qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef93#h/qef93#h_c6.dat",
                    transition=c5_n8_7, beam_atom=d0)
    qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef97#h/qef97#h_en2_kvi#c6.dat",
                    transition=c5_n8_7, beam_atom=d0_n2)

    bmp.set_dataset(filename=f"{path_to_read_adf}/adf22/bmp97#h/bmp97#h_2_c6.dat", plasma_atom=c6,
                    beam_atom=d0_n2)

    element_label = "c"
    cx_atom = c6
    cx_transition = c5_n8_7
    cx_cwl = 529.059  # [nm], C-VI 8->7
    cx_mass = 12.0
    wl_range = [527.00, 531.00]
    wavelen = np.linspace(wl_range[0], wl_range[1], 2000)
    frac_abu = plasma.fz['c'].sel(ion_charge=6, t=time)

elif which_impurity == "lithium":

    qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef07#h/qef07#h_arf#li3.dat",
                    transition=li2_n7_5, beam_atom=d0)
    qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef97#h/qef97#h_en2_kvi#li3.dat",
                    transition=li2_n7_5, beam_atom=d0_n2)

    # qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef07#h/qef07#h_arf#li3.dat",
    #                 transition=li2_n5_4, beam_atom=d0)
    # qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef97#h/qef97#h_en2_kvi#li3.dat",
    #                 transition=li2_n5_4, beam_atom=d0_n2)

    bmp.set_dataset(filename=f"{path_to_read_adf}/adf22/bmp97#h/bmp97#h_2_li3.dat", plasma_atom=li3,
                    beam_atom=d0_n2)

    element_label = "li"
    cx_atom = li3
    cx_transition = li2_n7_5
    cx_cwl = 516.69  # [nm], Li-III 7->5
    wl_range = [514, 519]
    # cx_cwl = 449.89  # [nm], Li-III 7->5
    # wl_range = [447, 451]
    cx_mass = 6.941
    wavelen = np.linspace(wl_range[0], wl_range[1], 2000)
    frac_abu = plasma.fz['li'].sel(ion_charge=3, t=time)

elif which_impurity == "helium":

    qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef93#h/qef93#h_he2.dat",
                    transition=he2_n4_3, beam_atom=d0)
    qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef97#h/qef97#h_en2_kvi#he2.dat",
                    transition=he2_n4_3, beam_atom=d0_n2)

    bmp.set_dataset(filename=f"{path_to_read_adf}/adf22/bmp97#h/bmp97#h_2_he2.dat", plasma_atom=he2,
                    beam_atom=d0_n2)

    element_label = "he"
    cx_atom = he2
    cx_transition = he2_n4_3
    cx_cwl = 468.571  # [nm], He-II 4->3
    wl_range = [466, 470]
    cx_mass = 4.0
    wavelen = np.linspace(wl_range[0], wl_range[1], 2000)
    frac_abu = plasma.fz['he'].sel(ion_charge=2, t=time)

elif which_impurity == "boron":

    qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef93#h/qef93#h_b5.dat",
                    transition=he2_n4_3, beam_atom=d0)
    qcx.set_dataset(filename=f"{path_to_read_adf}/adf12/qef97#h/qef97#h_en2_kvi#b5.dat",
                    transition=he2_n4_3, beam_atom=d0_n2)

    bmp.set_dataset(filename=f"{path_to_read_adf}/adf22/bmp97#h/bmp97#h_2_b5.dat", plasma_atom=he2,
                    beam_atom=d0_n2)

    element_label = "b"
    cx_atom = b5
    cx_transition = b4_n7_6
    cx_cwl = 494.497  # [nm], B-V 7>6
    wl_range = [492, 496]
    cx_mass = 10.81
    wavelen = np.linspace(wl_range[0], wl_range[1], 2000)
    frac_abu = plasma.fz['b'].sel(ion_charge=5, t=time)

else:
    raise ValueError(f"{which_impurity} not available")

bmp.set_dataset(filename=f"{path_to_read_adf}/adf22/bmp97#h/bmp97#h_2_h1.dat", plasma_atom=d1,
                beam_atom=d0_n2)

qcx.read_datasets()  # read file only once
qcx.define_grid()
bmp.read_datasets()  # read file only once
bmp.define_grid()
# -----------------------------------------------------------------------------

# Interpolate along LOS -------------------------------------------------------
# Map profiles to LOS
Te_mapped = los_transform.map_profile_to_los(
    plasma.electron_temperature.sel(t=time),
    t=time
)
ne_mapped = los_transform.map_profile_to_los(
    plasma.electron_density.sel(t=time),
    t=time
)
Ti_mapped = los_transform.map_profile_to_los(
    plasma.ion_temperature.sel(t=time),
    t=time
)
ni_mapped = los_transform.map_profile_to_los(
    plasma.impurity_density.sel(t=time, element=element_label),
    t=time
)
nn_mapped = los_transform.map_profile_to_los(
    plasma.neutral_density.sel(t=time),
    t=time
)
omega_mapped = los_transform.map_profile_to_los(
    plasma.toroidal_rotation.sel(t=time),
    t=time
)
fz_mapped = los_transform.map_profile_to_los(
    frac_abu,
    t=time
)
nz_mapped = ni_mapped * fz_mapped
zeff_mapped = los_transform.map_profile_to_los(
    plasma.zeff.sum("element").sel(t=time),
    t=time
)

# Looping
cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(los_transform.x1), dtype=float))

fig1 = plt.figure()
ax1 = plt.axes()

fig2 = plt.figure()
fig2.set_figheight(7)
fig2.set_figwidth(10)
ax2a = plt.subplot(321)
ax2b = plt.subplot(322, sharex=ax2a)
ax2c = plt.subplot(323, sharex=ax2a)
ax2d = plt.subplot(324, sharex=ax2a)
ax2e = plt.subplot(325, sharex=ax2a)
ax2f = plt.subplot(326, sharex=ax2a)

fig3 = plt.figure()
ax3a = plt.subplot(311)
ax3b = plt.subplot(312)
ax3c = plt.subplot(313)

fig4 = plt.figure()
ax4 = plt.axes()

Ti_fit = np.zeros(len(los_transform.x1))
vtor_fit = np.zeros(len(los_transform.x1))
for channel in los_transform.x1:
    total_spectrum = np.zeros_like(wavelen)
    for beamlet in range(los_transform.beamlets):
        # LOS
        x = los_transform.x.sel(channel=channel, beamlet=beamlet).values
        y = los_transform.y.sel(channel=channel, beamlet=beamlet).values
        dist = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)

        # Get Neutral Beam Density along LOS
        x_rot = np.zeros(len(x))
        y_rot = np.zeros(len(y))
        for k in range(len(x)):
            vec = utils.rotate(
                x[k] - origin_x,
                y[k] - origin_y,
                0.0,
                0.0,
                -beam_angle
            )
            x_rot[k] = vec[0]
            y_rot[k] = vec[1]

        fdens_los = np.zeros_like(x)
        hdens_los = np.zeros_like(x)
        tdens_los = np.zeros_like(x)
        for k in range(len(x)):
            fdens_los[k] = fdens_obj(x_rot[k], y_rot[k])
            hdens_los[k] = hdens_obj(x_rot[k], y_rot[k])
            tdens_los[k] = tdens_obj(x_rot[k], y_rot[k])
        fdens_los[np.isnan(fdens_los)] = 0.0
        hdens_los[np.isnan(hdens_los)] = 0.0
        tdens_los[np.isnan(tdens_los)] = 0.0
        ax1.plot(dist, fdens_los, c=cols[channel])

        # Get plasma parameters along LOS
        R = los_transform.R.sel(channel=channel, beamlet=beamlet)
        z = los_transform.z.sel(channel=channel, beamlet=beamlet)
        Te_along_los = Te_mapped.sel(channel=channel, beamlet=beamlet).values
        ne_along_los = ne_mapped.sel(channel=channel, beamlet=beamlet).values
        Ti_along_los = Ti_mapped.sel(channel=channel, beamlet=beamlet).values
        ni_along_los = ni_mapped.sel(channel=channel, beamlet=beamlet).values
        nn_along_los = nn_mapped.sel(channel=channel, beamlet=beamlet).values
        omega_along_los = omega_mapped.sel(channel=channel, beamlet=beamlet).values
        nz_along_los = ni_mapped.sel(channel=channel, beamlet=beamlet).values
        zeff_along_los = zeff_mapped.sel(channel=channel, beamlet=beamlet).values

        ax2a.plot(dist, Ti_along_los, c=cols[channel])
        ax2b.plot(dist, Te_along_los, c=cols[channel])
        ax2c.plot(dist, omega_along_los, c=cols[channel])
        ax2d.plot(dist, nn_along_los, c=cols[channel])
        ax2e.plot(dist, ni_along_los, c=cols[channel])
        ax2f.plot(dist, ne_along_los, c=cols[channel])

        # Get adf cross-sections
        bmag_along_los = np.zeros_like(ne_along_los)
        full_e_along_los = np.ones_like(ne_along_los) * beam_energy
        half_e_along_los = np.ones_like(ne_along_los) * beam_energy / 2
        third_e_along_los = np.ones_like(ne_along_los) * beam_energy / 3

        # Calculate charge exchange rates and beam populations along LOS
        cz = []
        charge = []
        for i_impurity, impurity in enumerate(plasma_impurities):
            if impurity == 'c':
                atom = c6
            elif impurity == 'ar':
                atom = ar18
            elif impurity == 'he':
                atom = he2
            elif impurity == 'li':
                atom = li3
            elif impurity == 'ne':
                atom = ne10
            else:
                raise ValueError('Invalid atom')
            cz.append({
                'atom': atom,
                'data': np.ones_like(ne_along_los) * impurity_concentration[i_impurity]
            })
            charge.append(atom.get_charge_state())

        cz.append({
            'atom': d1,
            'data': np.ones_like(ne_along_los) * (1 - np.sum(
                np.array(charge) * np.array(impurity_concentration)
            ))
        })
        zeff_along_los = atomic.Zeff(cz)

        popn2_full = bmp.get_data(
            full_e_along_los, ne_along_los, Te_along_los, cz, beam_atom=d0_n2)  # n=2 population
        popn2_half = bmp.get_data(
            half_e_along_los, ne_along_los, Te_along_los, cz, beam_atom=d0_n2)  # n=2 population
        popn2_third = bmp.get_data(
            third_e_along_los, ne_along_los, Te_along_los, cz, beam_atom=d0_n2)  # n=2 population

        out_full = qcx.get_data(
            full_e_along_los, ne_along_los, Te_along_los, zeff_along_los, bmag_along_los, cx_transition,
            [{'atom': d0, 'pop': np.ones_like(ne_along_los) - popn2_full}, {'atom': d0_n2, 'pop': popn2_full}]
        )
        out_half = qcx.get_data(
            half_e_along_los, ne_along_los, Te_along_los, zeff_along_los, bmag_along_los, cx_transition,
            [{'atom': d0, 'pop': np.ones_like(ne_along_los) - popn2_full}, {'atom': d0_n2, 'pop': popn2_half}]
        )
        out_third = qcx.get_data(
            third_e_along_los, ne_along_los, Te_along_los, zeff_along_los, bmag_along_los, cx_transition,
            [{'atom': d0, 'pop': np.ones_like(ne_along_los) - popn2_full}, {'atom': d0_n2, 'pop': popn2_third}]
        )
        out_full[np.isnan(out_full)] = 0.0
        out_half[np.isnan(out_half)] = 0.0
        out_third[np.isnan(out_third)] = 0.0
        ni_along_los[np.isnan(ni_along_los)] = 0.0

        ax3a.plot(dist, out_full, c=cols[channel])
        ax3b.plot(dist, out_half, c=cols[channel])
        ax3c.plot(dist, out_third, c=cols[channel])

        # Simulate spectra
        photon_energy = (constants.h * constants.c) / (cx_cwl * 1.0e-9)  # J
        line_intensities = [
            nz_along_los * fdens_los * out_full,
            nz_along_los * hdens_los * out_half,
            nz_along_los * tdens_los * out_third,
        ]

        # # Bremsstrahlung
        # brem_spectra = np.zeros((len(Te_along_los), len(wavelen)))
        # for i_wave in range(len(wavelen)):
        #     brem_spectra[:, i_wave] = ph.zeff_bremsstrahlung(
        #         Te_mapped.sel(channel=channel, beamlet=beamlet),
        #         ne_mapped.sel(channel=channel, beamlet=beamlet),
        #         wavelen[i_wave],
        #         zeff=zeff_mapped.sel(channel=channel, beamlet=beamlet),
        #     ).values  # W / m^3 / nm

        spectrum = np.zeros_like(wavelen)
        for i_los in range(len(R)):
            for i_energy in range(len(line_intensities)):
                # Calculate angle between LOS and velocity
                angle = utils.calculate_angle_between_los_and_beam(
                    direction[channel, 0:2],
                    np.array([x[i_los], y[i_los]])
                )

                vtor_here = omega_along_los[i_los] * R.values[i_los]  # m/s
                shift = vtor_here * cx_cwl * np.cos(angle) / constants.c

                # width = np.sqrt(Ti_along_los[i_los] / (cx_mass * 1.7e8)) * cx_cwl / (2*np.sqrt(2*np.log(2)))
                # amp = line_intensities[i_energy][i_los] / np.sqrt(2*np.pi*width**2)

                if not np.isnan(Ti_along_los[i_los]):
                    intens = ph.doppler_broaden(
                        wavelen, line_intensities[i_energy][i_los], cx_cwl + shift, cx_mass, Ti_along_los[i_los]
                    )
                    spectrum += intens * (dist[1]-dist[0])
                    # spectrum += brem_spectra[i_los, :] * (dist[1]-dist[0]) / photon_energy

        spectrum = spectrum * (1/(4*np.pi)) * photon_energy  # W / m^2 / str
        total_spectrum += spectrum
        #ax4.plot(wavelen, spectrum, c=cols[channel])
    ax4.plot(wavelen, total_spectrum, c=cols[channel])

    # Check fitting, to compare with Bart's TE-fidasim
    fit_p = [0.0, np.max(total_spectrum), cx_cwl, 0.5]
    Gfit_par, Gfit_cov = sp.optimize.curve_fit(utils.f_1G, wavelen, total_spectrum, p0=fit_p)

    # Calculate angle between LOS and velocity
    angle = utils.calculate_angle_between_los_and_beam(
        direction[channel, 0:2],
        np.array([x_pos[channel], y_pos[channel]])
    )

    Ti_fit[channel] = 1.7e8 * cx_mass * (Gfit_par[3] * 2 * np.sqrt(2 * np.log(2)) / cx_cwl) ** 2  # eV
    vtor_fit[channel] = - constants.c * ((cx_cwl - Gfit_par[2]) / cx_cwl) / np.cos(angle)  # m/s

    # L_cx = np.sum([
        #     np.trapz(ni_along_los * fdens_los * out_full, dx=dist[1]-dist[0]),
        #     np.trapz(ni_along_los * hdens_los * out_half, dx=dist[1]-dist[0]),
        #     np.trapz(ni_along_los * tdens_los * out_third, dx=dist[1]-dist[0]),
        # ]) * (1/(4*np.pi)) * photon_energy  # W / m^2 / str
        # print(f"L_cx = {L_cx} [W / m^2 / str]")

print(f"Ti_fit = {Ti_fit}")
print(f"vtor_fit = {vtor_fit}")
# -----------------------------------------------------------------------------

# Plots -----------------------------------------------------------------------
rho_2d = plasma.equilibrium.rhop.interp(t=time, method="nearest")
Ti_2d = plasma.ion_temperature.sel(t=time).interp(rhop=rho_2d)
Ti_mp = Ti_2d.interp(z=0.0)
omega_2d = plasma.toroidal_rotation.sel(t=time).interp(rhop=rho_2d)
vtor_mp = omega_2d.interp(z=0.0) * omega_2d.R

fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(6)
ax = plt.subplot(211)
plt.plot(Ti_mp.R.values, Ti_mp.values, c='darkgrey')
plt.plot(r_pos, Ti_fit, c=cols[-1], ls="none", marker="o")
plt.ylabel("Ti (eV)")

plt.subplot(212, sharex=ax)
plt.plot(vtor_mp.R.values, vtor_mp.values, c='darkgrey')
plt.plot(r_pos, vtor_fit, c=cols[-1], ls="none", marker="o")
plt.ylabel("Vtor (m/s)")
plt.xlabel("R (m)")
plt.tight_layout()

ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Neutral Density (m-3)")

ax2a.set_ylabel("Ti")
ax2b.set_ylabel("Te")
ax2c.set_ylabel("omega")
ax2d.set_ylabel("nn")
ax2e.set_xlabel("dist")
ax2e.set_ylabel("ni")
ax2f.set_xlabel("dist")
ax2f.set_ylabel("ne")
fig2.set_layout_engine(layout="tight")

ax3a.set_ylabel('full')
ax3b.set_ylabel('half')
ax3c.set_ylabel('third')
ax3c.set_xlabel('dist (m)')
fig3.set_layout_engine(layout="tight")

ax4.set_xlabel("Wavelength (nm)")
ax4.set_ylabel("Spectral Radiance (W/m^2/str/nm)")

x_st40 = np.linspace(-1.0, 1.0, 499)
y_st40 = np.linspace(-1.0, 1.0, 500)
X_st40, Y_st40 = np.meshgrid(x_st40, y_st40, indexing="ij")
fdens_st40 = np.zeros((len(y_st40), len(x_st40)))
for i_y in range(len(y_st40)):
    for i_x in range(len(x_st40)):
        vec = utils.rotate(
            x_st40[i_x] - origin_x,
            y_st40[i_y] - origin_y,
            0.0,
            0.0,
            -beam_angle
        )
        fdens_st40[i_y, i_x] = fdens_obj(vec[0], vec[1])

print(f"src = {src}")

#plt.figure()
#plt.contourf(xgrid, ygrid, fdens_summed[i_z, :, :])

plt.figure()
plt.contourf(x_st40, y_st40, fdens_st40, cmap='Greens')
plt.plot(src[0], src[1], 'rx')

for x1 in los_transform.x1:
    for beamlet in range(los_transform.beamlets):
        x = los_transform.x.sel(channel=x1, beamlet=beamlet)
        y = los_transform.y.sel(channel=x1, beamlet=beamlet)
        plt.plot(x, y, c=cols[x1])

plt.show()
# -----------------------------------------------------------------------------
