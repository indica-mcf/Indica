from hda.models.spectrometer import XRCSpectrometer
from hda.models.interferometer import Interferometer
from hda.manage_data import bin_data_in_time, map_on_equilibrium, initialize_bckc
from hda.models.plasma import Plasma
import hda.profiles as profiles
from hda.read_st40 import ST40data
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates, LinesOfSightTransform
from indica.provenance import get_prov_attribute

from hda.optimizations.interferometer import match_interferometer_los_int

import matplotlib.pylab as plt
from copy import deepcopy
import numpy as np
from pytest import approx

plt.ion()

"""
Workflow designed to replicate the results of the pre-refactored HDA as implemented
in hdatests.test_hda
"""

pulse = 9780
tstart = 0.025
tend = 0.14
dt = 0.015
diagn_ne = "smmh1"
diagn_te = "xrcs"
quant_ne = "ne"
quant_te = "te_kw"
quant_ti = "ti_w"
quant_ar = "int_w"
main_ion = "h"
impurities = ("c", "ar", "he")
imp_conc = (0.03, 0.001, 0.01)
equilibrium_diagnostic = "efit"
extrapolate = None
marchuk = True
plot = True

# Read raw data
raw = ST40data(pulse, tstart - dt / 2, tend + dt / 2)
raw_data = raw.get_all()

# Initialize equilibrium and flux transform objects
equilibrium_data = raw_data[equilibrium_diagnostic]
equilibrium = Equilibrium(equilibrium_data)
flux_transform = FluxSurfaceCoordinates("poloidal")
flux_transform.set_equilibrium(equilibrium)

# Initialize plasma class and assign equilibrium related objects
pl = Plasma(tstart=tstart, tend=tend, dt=dt, impurities=impurities, imp_conc=imp_conc)
pl.set_equilibrium(equilibrium)
pl.set_flux_transform(flux_transform)
pl.calculate_geometry()

revision = get_prov_attribute(
    equilibrium_data[list(equilibrium_data)[0]].provenance, "revision"
)
pl.optimisation["equil"] = f"{equilibrium_diagnostic}:{revision}"

# Assign default profile values and objects to plasma class
pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
profs = profiles.profile_scans(rho=pl.rho)
pl.Ne_prof = profs["Ne"]["peaked"]
pl.Te_prof = profs["Te"]["peaked"]
pl.Ti_prof = profs["Ti"]["peaked"]
pl.Nimp_prof = profs["Nimp"]["peaked"]
pl.Vrot_prof = profs["Vrot"]["peaked"]

# Bin data as required to match plasma class, assign equlibrium objects to
data = {}
for kinstr in raw_data.keys():
    data[kinstr] = bin_data_in_time(raw_data[kinstr], pl.tstart, pl.tend, pl.dt,)
    map_on_equilibrium(data[kinstr], flux_transform=pl.flux_transform)

# Initialize back-calculated (bckc) diactionary and forward model objects
# Any transform can be associated to the model, also of non-existent systems
bckc = initialize_bckc(data)
forward_models = {}
forward_models["xrcs"] = XRCSpectrometer(marchuk=marchuk, extrapolate=extrapolate)
interferometers = ["smmh1", "nirh1"]
for diag in interferometers:
    forward_models[diag] = Interferometer(name=diag)
    forward_models[diag].set_los_transform(data[diag]["ne"].attrs["transform"])

# Optimize electron density to match interferometer
bckc[diagn_ne][quant_ne], Ne = match_interferometer_los_int(
    forward_models[diagn_ne], data[diagn_ne][quant_ne], pl.Ne_prof
)
pl.el_dens.values = Ne.values

# Calculate the LOS-integral of all the interferometers
for diag in interferometers:
    los_integral, _ = forward_models[diag].line_integrated_density(pl.el_dens)
    bckc[diag][quant_ne].values = los_integral.values

# -------------------------------------------------------------------
# Add invented interferometer with different LOS 20 cm above the SMMH1
diag = "smmh2"
forward_models[diag] = Interferometer(name=diag)
_trans = data["smmh1"]["ne"].attrs["transform"]
los_transform = LinesOfSightTransform(
    x_start=_trans.x_start.values,
    x_end=_trans.x_end.values,
    y_start=_trans.y_start.values,
    y_end=_trans.y_end.values,
    z_start=_trans.z_start.values + 0.15,
    z_end=_trans.z_end.values + 0.15,
    name=diag,
    machine_dimensions=_trans._machine_dims,
)
los_transform.set_equilibrium(flux_transform.equilibrium)
los_transform.set_flux_transform(flux_transform)
_ = los_transform.remap_los(t=data["smmh1"]["ne"].t)
forward_models[diag].set_los_transform(los_transform)
bckc["smmh2"] = {}
los_integral, _ = forward_models["smmh2"].line_integrated_density(pl.el_dens)
bckc["smmh2"][quant_ne] = los_integral

# if plot:
# Plot comparison of raw data, binned data and back-calculated values
plt.figure()
colors = {"nirh1": "blue", "smmh1": "purple"}
for diag in interferometers:
    raw_data[diag][quant_ne].plot(color=colors[diag], label=diag)
    data[diag][quant_ne].plot(color=colors[diag], marker="o")
    bckc[diag][quant_ne].plot(color=colors[diag], marker="x")

bckc["smmh2"][quant_ne].plot(color="red", marker="D", label="smmh2")
plt.legend()

# Plot resulting density profiles
plt.figure()
plt.plot(pl.el_dens.sel(t=slice(0.02, 0.12)).transpose())

# Plot lines of sights and equilibrium on (R, z) plane
plt.figure()
t = 0.05
levels = [0.1, 0.3, 0.5, 0.7, 0.95]
equilibrium.rho.sel(t=t, method="nearest").plot.contour(levels=levels)
for diag in interferometers:
    plt.plot(
        forward_models[diag].los_transform.R,
        forward_models[diag].los_transform.z,
        color=colors[diag],
        label=diag,
    )

plt.plot(
    forward_models["smmh2"].los_transform.R,
    forward_models["smmh2"].los_transform.z,
    color="red",
    label="smmh2",
)
plt.axis("scaled")
plt.xlim(0.1, 0.8)
plt.ylim(-0.6, 0.6)
plt.legend()

# Plot rho along line of sight
plt.figure()
for diag in interferometers:
    plt.plot(
        forward_models[diag].los_transform.rho.transpose(),
        color=colors[diag],
        label=diag,
    )

plt.plot(
    forward_models["smmh2"].los_transform.rho.transpose(), color="red", label="smmh2"
)
plt.legend()
