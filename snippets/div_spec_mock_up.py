from matplotlib import pyplot as plt, cm

import indica
import numpy as np

from indica import Plasma, PlasmaProfiler
from indica.models.passive_spectrometer import PassiveSpectrometer, read_adf15s, format_pecs
from indica.defaults.load_defaults import load_default_objects
from indica.profilers.profiler_gauss import initialise_gauss_profilers
from indica.readers import SOLPSReader, ST40Reader

config = {
    "c": {str(charge): dict(file_type="pju", year="96") for charge in range(0, 6)},
    "h": {str(charge): dict(file_type="pju", year="12") for charge in range(0, 1)},
    # "he": {
    #     "1": dict(
    #         file_type="bnd",
    #         year="96",
    #     ),
    # },
    # "mo": {str(charge): dict(file_type="pju", year="96") for charge in range(0, 3)},
    # "b": {str(charge): dict(file_type="pju", year="96") for charge in range(0, 3)},
    # "li": {str(charge): dict(file_type="pju", year="96") for charge in range(0, 3)},

    # "ar": {
    #     str(charge): dict(file_type="llu", year="transport") for charge in range(16, 18)
    # },
}

pulse = 11890
t = 0.105

solps_reader = SOLPSReader(pulse, t)
solps_data = solps_reader.get()

st40_reader = ST40Reader(pulse, tstart=0.01, tend=0.12)
equil_data = st40_reader.get("","efit",)
equilibrium = indica.Equilibrium(equil_data)

transforms = load_default_objects("st40", "geometry")

# plasma = Plasma(tstart=0.06,
#                 tend=0.07,
#                 dt=0.01,
#                 machine="st40",
#                 impurities=("c", "li", "he")
#                 )
# plasma.build_atomic_data()
# profilers = initialise_gauss_profilers(
#         plasma.rhop,
#         profile_names=[
#             "electron_density",
#             "ion_temperature",
#             "electron_temperature",
#             "impurity_density:li",
#             "impurity_density:c",
#             "impurity_density:he",
#         ],
#     )
# profilers["impurity_density:li"].y0 = 2e19
# profilers["impurity_density:li"].y1 = 5e18
# plasma_profiler = PlasmaProfiler(
#     plasma=plasma,
#     profilers=profilers,
# )
# plasma_profiler()
# plasma.set_equilibrium(equilibrium=equilibrium)

transform = transforms["blom_dv1"]
transform.set_equilibrium(equilibrium=equilibrium)

adf15 = read_adf15s(elements=config.keys(), config=config)
window = np.linspace(350, 750, 500)

pecs = format_pecs(adf15, wavelength_bounds=slice(window.min(), window.max()),
                   electron_density_bounds=slice(1e18, 5e20),
                   electron_temperature_bounds=slice(10, 5000), )

divspec = PassiveSpectrometer(name="test", pecs=pecs, window=window)

# divspec.set_plasma(plasma)
divspec.set_transform(transform)



Nh = solps_data["nion"].sel(element="h") * solps_data["fz"]["h"].sel(ion_charge=0)
Nh = Nh.drop_vars(("element", "ion_charge", ))

bckc = divspec(Te=solps_data["te"], Ne=solps_data["ne"], Ni=solps_data["nion"],
            Fz=solps_data["fz"], Nh=Nh, Ti=500, t=[t], )


emissivity = divspec.intensity["h"].sum("wavelength").sum("t") + divspec.intensity["c"].sum("wavelength").sum("t")
# nd0 = (solps_data["nion"].sel(element="h") * solps_data["fz"]["h"].sel(ion_charge=0)).sel(t=t)
extent = [emissivity.R.min(), emissivity.R.max(), emissivity.z.min(), emissivity.z.max()]


# plt.figure()
# plt.title("neutral deuterium")
# plt.imshow(nd0.values, extent=extent, norm=LogNorm(vmin=1e13, vmax=1e19))
# plt.axis("equal")
# plt.grid()
# plt.colorbar()

plt.figure()
plt.title("emissivity (photons/m^3)")
plt.imshow(emissivity.values, extent=extent )
plt.colorbar()
plt.axis("equal")
plt.grid()

plt.figure()
plt.grid()
divspec.transform.plot(t, figure=False, orientation="Rz", )
plt.imshow(emissivity.values, extent=extent )


plt.figure()
spectra = divspec.bckc["spectra"]

cols_chan = cm.gnuplot2(np.linspace(0.1, 0.75, len(spectra.channel), dtype=float))
for idx, chan_num in enumerate(spectra.channel.values):
    plt.plot(
        spectra.wavelength,
        spectra.sel(t=t, channel=chan_num),
        label=f"channel={chan_num}",
        color=cols_chan[idx],
        alpha=0.8,
    )
plt.ylabel("Emissivity (photon/m^2/nm)")
plt.xlabel("Wavelength (nm)")
plt.legend()

plt.show(block=True)

