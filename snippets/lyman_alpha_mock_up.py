from matplotlib import cm

import indica
import numpy as np

from indica.converters.abstractconverter import plot_geometry
from indica.models.passive_spectrometer import PassiveSpectrometer, read_adf15s, format_pecs
from indica.defaults.load_defaults import load_default_objects
from indica.readers import SOLPSReader, ST40Reader
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from indica.utilities import set_axis_sci

config = {
    "h": {str(charge): dict(file_type="pju", year="12") for charge in range(0, 1)},
}
window = np.linspace(121.4, 121.8, 500)
adf15 = read_adf15s(elements=["h"], config=config)
pecs = format_pecs(adf15, wavelength_bounds=slice(window.min(), window.max()))

pulse = 11890
t = 0.105
st40_reader = ST40Reader(pulse, tstart=0.01, tend=0.12)
equil_data = st40_reader.get("","efit",)
equilibrium = indica.Equilibrium(equil_data)

instr = "sxrc_rz1"
transforms = load_default_objects("st40", "geometry")
transform = transforms[instr]
transform.set_equilibrium(equilibrium=equilibrium)


solps_reader = SOLPSReader(pulse, t)
solps_data = solps_reader.get()

#
lyman_alpha = PassiveSpectrometer(name=instr, pecs=pecs, window=window)
# lyman_alpha.set_plasma(plasma)
lyman_alpha.set_transform(transform)
#
#
bckc = lyman_alpha(Te=solps_data["te"], Ne=solps_data["ne"], Ni=solps_data["nion"],
            Fz=solps_data["fz"], Nh=solps_data["nion"].sel(element="h") * solps_data["fz"]["h"].sel(ion_charge=0), Ti=10, t=[t], )

emissivity = lyman_alpha.intensity["h"].sum("wavelength").sum("t")
nd0 = (solps_data["nion"].sel(element="h") * solps_data["fz"]["h"].sel(ion_charge=0)).sel(t=t)
extent = [emissivity.R.min(), emissivity.R.max(), emissivity.z.min(), emissivity.z.max()]


plt.figure()
plt.title("neutral deuterium")
plt.imshow(nd0.values, extent=extent, norm=LogNorm(vmin=1e13, vmax=1e19))
plt.axis("equal")
plt.grid()
plt.colorbar()

plt.figure()
plt.title("emissivity (W/m^3)")
plt.imshow(emissivity.values, extent=extent )
plt.colorbar()
plt.axis("equal")
plt.grid()

plt.figure()
plt.grid()
lyman_alpha.transform.plot(t, figure=False, orientation="Rz")
plt.imshow(emissivity.values, extent=extent )


plt.figure()
spectra = lyman_alpha.bckc["spectra"]

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

print()
