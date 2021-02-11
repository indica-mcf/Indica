from indica.readers import ADASReader
import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray
import snippets.fac_profiles as fac

#plt.ion()

reader = ADASReader()
scd_W = reader.get_adf11("scd", "W", "89")
acd_W = reader.get_adf11("acd", "W", "89")
plt_W = reader.get_adf11("plt", "W", "50")
prb_W = reader.get_adf11("prb", "W", "89")
nz = scd_W.coords["ion_charges"].size

# Make fake electron temperature/density profiles
rhop, temp, dens = fac.ne_te(te0=4.e3, ne0=6.e19)
ntemp = np.size(temp)

# Interpolate ionization and recombination rates on electron temperature (electron density is irrelevant
# in the case of LTE equilibrium)
# Density from ADAS files has already been converted to m**-3
ion = scd_W.interp(log_electron_temperature=np.log10(temp)).interp(log_electron_density=np.mean(np.log10(dens)))
rec = acd_W.interp(log_electron_temperature=np.log10(temp)).interp(log_electron_density=np.mean(np.log10(dens)))
lin_rad = plt_W.interp(log_electron_temperature=np.log10(temp)).interp(log_electron_density=np.mean(np.log10(dens)))
rec_rad = prb_W.interp(log_electron_temperature=np.log10(temp)).interp(log_electron_density=np.mean(np.log10(dens)))

# Calculate the fractional abundance of all ionization stages
fz = np.empty((nz, ntemp))
fz.fill(1.)
fz = DataArray(fz, coords=[("ion_charges", scd_W.coords["ion_charges"]),
                           ("log_electron_temperature", np.log10(temp))])
for iz in range(1, nz):
    fz[iz, :] = fz[iz-1, :] * 10**(ion[iz-1, :] - rec[iz-1, :])
    norm = fz[:iz+1, :].sum(axis=0, skipna=True)
    for j in range(iz+1):
        fz[j, :] = fz[j, :]/norm

# Calculate total radiation loss parameter
tot_rad_loss = ((10**lin_rad + 10**rec_rad) * fz).sum(axis=0, skipna=True)

plt.figure()
plt.plot(rhop, fz.transpose())
plt.xlabel("Rhop")
plt.ylabel("Fractional abundance")
plt.show()

plt.figure()
plt.plot(rhop, tot_rad_loss.transpose())
plt.xlabel("Rhop")
plt.ylabel("Total cooling factor")
plt.show()

plt.figure()
plt.plot(rhop, temp)
plt.xlabel("Rhop")
plt.ylabel("Temperature")
plt.show()
