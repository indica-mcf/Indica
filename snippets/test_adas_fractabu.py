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
ion = scd_W.interp(log_electron_temperature=np.log10(profs.te)).interp(
    log_electron_density=np.mean(np.log10(profs.ne))
)
rec = acd_W.interp(log_electron_temperature=np.log10(profs.te)).interp(
    log_electron_density=np.mean(np.log10(profs.ne))
)
lin_rad = plt_W.interp(log_electron_temperature=np.log10(profs.te)).interp(
    log_electron_density=np.mean(np.log10(profs.ne))
)
rec_rad = prb_W.interp(log_electron_temperature=np.log10(profs.te)).interp(
    log_electron_density=np.mean(np.log10(profs.ne))
)

drop = ["log_electron_temperature", "log_electron_density"]
rec = rec.drop_vars(drop)
ion = ion.drop_vars(drop)
lin_rad = lin_rad.drop_vars(drop)
rec_rad = rec_rad.drop_vars(drop)

# Calculate the fractional abundance of all ionization stages
fz = np.empty((nz, ntemp))
fz.fill(1.0)
fz = DataArray(
    fz,
    name="fractional_abundance",
    coords=[("ion_charges", scd_W.coords["ion_charges"]), ("rho_poloidal", rho)],
)
for iz in range(1, nz):
    fz[iz, :] = fz[iz - 1, :] * 10 ** (ion[iz - 1, :] - rec[iz - 1, :])
    norm = np.nansum(fz[:iz, :].values, axis=0)
    for j in range(iz + 1):
        fz[j, :] = fz[j, :] / norm

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
