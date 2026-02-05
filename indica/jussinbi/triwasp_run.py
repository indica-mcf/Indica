from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import sys
import pickle
import xarray as xr
from xarray import DataArray
from MDSplus import *

# Indica imports
from indica.converters.transect import TransectCoordinates
from indica.converters import LineOfSightTransform
from indica.readers import ST40Reader
from indica import Plasma
from indica import Equilibrium
from indica.readers import ReaderProcessor
from indica.defaults.load_defaults import load_default_objects


# Rotation function
def rotate(x, y, xo, yo, theta):
    xr = np.cos(theta) * (x - xo) - np.sin(theta) * (y - yo) + xo
    yr = np.sin(theta) * (x - xo) + np.cos(theta) * (y - yo) + yo
    return np.array([xr, yr])


# Function for calculating angle between 2 lines
def calculate_angle_of_intersection(ds, dv):
    return np.arccos(np.dot(ds, dv) / (np.sqrt(np.dot(ds, ds)) * np.sqrt(np.dot(dv, dv))))


# Function for calculating angle between los and beam
def calculate_angle_between_los_and_beam(
        los_dir: np.ndarray,
        intersect: np.ndarray
):
    # spectrometer los direction
    ds = np.array([los_dir[0], los_dir[1]])

    # find the direction of the velocity vector at intercept point
    theta0 = np.arctan2(0-intersect[1], 0-intersect[0])
    if theta0 < 0.0:
        theta0 = theta0 + 2*np.pi
    theta_v = theta0 - np.pi/2
    dv = np.array([np.cos(theta_v), np.sin(theta_v)])

    # Calculate angle of intersection between the 2 vectors
    return calculate_angle_of_intersection(ds, dv)


# Gaussian function
def f_1G(x, y0, A, xc, sig):
    return y0 + A*(1/(sig*(np.sqrt(2*np.pi))))*(np.exp((-1/2)*(((x - xc)/sig)**2)))


# Function to read ASTRA data from MDS+
def read_astra(pulse, run):

    # Initialise dictionary
    data = {}

    # Connect with Tree
    t = Tree("ASTRA", pulse)

    # Read profiles
    data['time'] = t.getNode(f"{run}.TIME").data()
    data['psi_n'] = t.getNode(f"{run}.PROFILES.ASTRA.PSIN").data()
    data['Ti'] = t.getNode(f"{run}.PROFILES.ASTRA.TI").data() * 1e3
    data['Te'] = t.getNode(f"{run}.PROFILES.ASTRA.TE").data() * 1e3
    data['ni'] = t.getNode(f"{run}.PROFILES.ASTRA.NI").data() * 1e19
    data['ne'] = t.getNode(f"{run}.PROFILES.ASTRA.NE").data() * 1e19
    data['nn'] = t.getNode(f"{run}.PROFILES.ASTRA.NN").data() * 1e19

    # Disconnect from Tree
    t.close()

    return data



# Read ST40 data and create profiles. This should just be done by other InDiCa parts.
tstart = 0.09
tend = 0.1
dt = 0.01
pulse = 13475
st40 = ST40Reader(pulse=pulse, tstart=tstart, tend=tend, dt=dt)
st40(instruments=["efit"])
equilibrium_data = st40.get("", "efit", 0)
raw_data = {"efit": equilibrium_data}

processor = ReaderProcessor()
binned_raw_data = processor(raw_data, tstart, tend, dt)
binned_raw_data["efit"]["psin"] = equilibrium_data["psin"]
binned_raw_data["efit"]["index"] = equilibrium_data["index"]
binned_raw_data["efit"]["R"] = equilibrium_data["R"]
binned_raw_data["efit"]["z"] = equilibrium_data["z"]
equilibrium = Equilibrium(binned_raw_data["efit"])

impurities = ("c", "ar", "li", "he")
impurity_concentration = (0.01, 0.001, 0.02, 0.02)
plasma = Plasma(tstart=tstart, tend=tend, dt=dt,
                impurities=impurities,
                impurity_concentration=impurity_concentration)
plasma.build_atomic_data()
plasma.set_equilibrium(equilibrium)

# Profiles
omegator_core = 100 * 1e3 / 0.5
pulse_astra = 13013475
run_astra = "RUN606"
astra_data = read_astra(pulse_astra, run_astra)

psin_plasma = plasma.ion_temperature.rhop.values ** 2
for i_time, t_ in enumerate(plasma.t.values):
    i_min = np.argmin(np.abs(astra_data['time'] - t_))
    Ti_astra_now = astra_data["Ti"][i_min, :]
    Te_astra_now = astra_data["Te"][i_min, :]
    ne_astra_now = astra_data["ne"][i_min, :]
    nn_astra_now = astra_data["nn"][i_min, :]
    ni_astra_now = astra_data["ni"][i_min, :]
    omega_astra_now = astra_data["Ti"][i_min, :] * omegator_core / np.max(astra_data["Ti"][i_min, :])
    psi_n_now = astra_data["psi_n"][i_min, :]

    plasma.ion_temperature[i_time, :] = np.interp(
        psin_plasma, psi_n_now, Ti_astra_now,
        left=Ti_astra_now[0], right=Ti_astra_now[-1],
    )
    plasma.electron_temperature[i_time, :] = np.interp(
        psin_plasma, psi_n_now, Te_astra_now,
        left=Te_astra_now[0], right=Te_astra_now[-1],
    )
    plasma.electron_density[i_time, :] = np.interp(
        psin_plasma, psi_n_now, ne_astra_now,
        left=ne_astra_now[0], right=ne_astra_now[-1],
    )
    plasma.neutral_density[i_time, :] = np.interp(
        psin_plasma, psi_n_now, nn_astra_now,
        left=nn_astra_now[0], right=nn_astra_now[-1],
    )
    plasma.toroidal_rotation[i_time, :] = np.interp(
        psin_plasma, psi_n_now, omega_astra_now,
        left=omega_astra_now[0], right=omega_astra_now[-1],
    )
    for i_atom in range(len(impurities)):
        plasma.impurity_density[i_atom, i_time, :] = np.interp(
            psin_plasma, psi_n_now, ni_astra_now * plasma.impurity_concentration[i_atom],
            left=ni_astra_now[0] * plasma.impurity_concentration[i_atom],
            right=ni_astra_now[-1] * plasma.impurity_concentration[i_atom],
        )



# Export pkl
# pickle.dump(plasma, open(f"{pulse}_indica_plasma.pkl", "wb"))

# Plasma ion mass
plasma_ion_amu = 2.014

# Load TriWaSp geometry
# tws_geom = pickle.load(
#     open('geometry_pkl_files/TriWaSp_geometry_6los_50-75_sector1_centre.pkl', 'rb')
# )
# focal_length = -10.0  # meter
# spot_width = 0.010  # meter
# spot_height = 0.010  # meter
tws_geom = pickle.load(
    open('geometry_pkl_files/TriWaSp_geometry_7los_50-77_sector1.pkl', 'rb')
)
focal_length = -0.03995269  # meter
spot_width = 1.1 * 1e-3  # meter
spot_height = 1.1 * 1e-3  # meter
origin = tws_geom['origin']
direction = tws_geom['direction']
x_pos = tws_geom['x_pos']
y_pos = tws_geom['y_pos']
z_pos = tws_geom['z_pos']

# make LineOfSightTransform
machine_dims = ((0.15, 1.00), (-0.75, 0.75))
spot_shape = "round"
beamlets_method = "adaptive"
n_beamlets = 1
plot_beamlets = False

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

# Make Transect
transect = TransectCoordinates(
    x_pos,
    y_pos,
    z_pos,
    "TriWaSp_P2p4",
    machine_dimensions=machine_dims,
)

plot_geometry = False
if plot_geometry:
    # Plotting...
    cols = cm.gnuplot2(np.linspace(0.3, 0.75, len(los_transform.x1), dtype=float))

    plt.figure()
    th = np.linspace(0, 2 * np.pi, 1000)
    x_ivc = machine_dims[0][1] * np.cos(th)
    y_ivc = machine_dims[0][1] * np.sin(th)
    x_cc = machine_dims[0][0] * np.cos(th)
    y_cc = machine_dims[0][0] * np.sin(th)

    plt.plot(x_cc, y_cc, c="k", lw=2.0)
    plt.plot(x_ivc, y_ivc, c="k", lw=2.0)
    for x1 in los_transform.x1:
        for beamlet in los_transform.beamlets:
            x = los_transform.x.sel(channel=x1, beamlet=beamlet)
            y = los_transform.y.sel(channel=x1, beamlet=beamlet)

            plt.plot(x, y, c=cols[x1])

    plt.tight_layout()
    plt.axis('equal')
    plt.figure()

    plt.plot(
        [machine_dims[0][1], machine_dims[0][1]],
        [machine_dims[1][0], machine_dims[1][1]],
        c="k",
        lw=2.0,
    )

    plt.plot(
        [machine_dims[0][0], machine_dims[0][0]],
        [machine_dims[1][0], machine_dims[1][1]],
        c="k",
        lw=2.0,
    )

    for x1 in los_transform.x1:
        for beamlet in los_transform.beamlets:
            R = los_transform.R.sel(channel=x1, beamlet=beamlet)
            z = los_transform.z.sel(channel=x1, beamlet=beamlet)

            plt.plot(R, z, c=cols[x1])

    plt.tight_layout()
    plt.axis('equal')
    plt.show(block=True)

# Set-up FIDASIM run
# Build beam configuration
nbiconfig = {
    "name": "hnbi",
    "einj": 52.0,  # keV
    "pinj": 0.5,   # MW
    "current_fractions": [
        0.5,
        0.35,
        0.15
    ],
    "ab": 2.014
}

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

# Loop over time
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
    path_to_code = "/home/jussi.hakosalo/te-fidasim"
    sys.path.append(path_to_code)
    import fidasim_ST40_indica

    results = fidasim_ST40_indica.main(
        pulse,
        time,
        nbiconfig,
        specconfig,
        plasmaconfig,
        num_cores=3,
        user="jussi.hakosalo",
        force_run_fidasim=run_fidasim,
        save_dir="/home/jussi.hakosalo/fidasim_output"
    )



