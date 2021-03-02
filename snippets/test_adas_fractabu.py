import matplotlib.pylab as plt
import numpy as np
import snippets.fac_profiles as fac
from xarray import DataArray

from indica.readers import ADASReader

reader = ADASReader()
scd_W = reader.get_adf11("scd", "W", "89")
acd_W = reader.get_adf11("acd", "W", "89")
plt_W = reader.get_adf11("plt", "W", "50")
prb_W = reader.get_adf11("prb", "W", "89")
nz = scd_W.coords["ion_charges"].size

# Make fake electron temperature/density profiles

profs = fac.main_plasma_profs()
rho = profs.te.coords["rho_poloidal"]
ntemp = profs.te.size

# Interpolate ionization and recombination rates on electron
# temperature (electron density is irrelevant in the case of LTE
# equilibrium)
# Density from ADAS files has already been converted to m**-3
scd_W = scd_W.interp(log10_electron_temperature=np.log10(profs.te)).interp(
    log10_electron_density=np.mean(np.log10(profs.ne))
)
acd_W = acd_W.interp(log10_electron_temperature=np.log10(profs.te)).interp(
    log10_electron_density=np.mean(np.log10(profs.ne))
)
lin_rad = plt_W.interp(log10_electron_temperature=np.log10(profs.te)).interp(
    log10_electron_density=np.mean(np.log10(profs.ne))
)
rec_rad = prb_W.interp(log10_electron_temperature=np.log10(profs.te)).interp(
    log10_electron_density=np.mean(np.log10(profs.ne))
)

dim1, dim2 = scd_W.dims
coords = [
    (dim1, np.arange(scd_W.coords[dim1].max() + 2)),
    (dim2, scd_W.coords[dim2]),
]
nz = scd_W.shape[0]
nother = scd_W.shape[1]
fz = np.ones((nz + 1, nother))
fz = DataArray(
    fz,
    name="fractional_abundance",
    coords=coords,
)

fz[0, :] = fz[0, :] * 10 ** (-scd_W[0, :] + acd_W[0, :])
for i in range(1, nz):
    fz[i, :] = fz[i - 1, :] * 10 ** (scd_W[i - 1, :] - acd_W[i - 1, :])
fz[i + 1, :] = fz[i, :] * 10 ** (scd_W[i, :] - acd_W[i, :])
for j in range(nother):
    norm = np.nansum(fz[:, j], axis=0)
    fz[:, j] /= norm

fz[i + 1, :] = fz[i, :] * 10 ** (scd_W[i, :] - acd_W[i, :])
for i in range(nz - 1, 0, -1):
    fz[i, :] = fz[i - 1, :] * 10 ** (scd_W[i - 1, :] - acd_W[i - 1, :])
fz[0, :] = fz[0, :] * 10 ** (-scd_W[0, :] + acd_W[0, :])
for j in range(nother):
    norm = np.nansum(fz[:, j], axis=0)
    fz[:, j] /= norm

# Calculate total radiation loss parameter
tot_rad_loss = ((10 ** lin_rad + 10 ** rec_rad) * fz).sum(axis=0)

# Plot results
plt.figure()
profs.te.plot()
plt.show()

plt.figure()
plt.plot(rho, fz.transpose())
plt.xlabel("Rhop")
plt.ylabel("Fractional abundance")
plt.show()

plt.figure()
plt.plot(rho, tot_rad_loss.transpose())
plt.xlabel("Rhop")
plt.ylabel("Total cooling factor")
plt.show()
