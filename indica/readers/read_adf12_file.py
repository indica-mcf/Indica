import sys
sys.path.insert(0, "../../")
from indica.readers import ADASReader
from matplotlib import pyplot as plt
import numpy as np

# ADAS reader
adas_reader = ADASReader()


## Read ADF11 file
#ADF11 = {"ar": {"scd": "89", "acd": "89", "ccd": "89"}}
#scd = adas_reader.get_adf11("scd", 'ar', ADF11['ar']["scd"])
#print(scd)

# Get ADF12 file
qener, qtiev, qdensi, qzeff, qefref = adas_reader.get_adf12("H", "0", "C", "6", "cx", "99")

print(qener)
print(qtiev)
print(qdensi)
print(qzeff)
print(qefref)

# Find cross-sections
Ebeam  = 25 * 1e3           # eV/amu
ne_cm3 = (3*1e19) * 1e-6    # cm^-3
Ti_eV  = 5000.              # eV
Zeff   = 2.5                # dimensionless


# Compute
lqefref = np.log(qefref)

lnq = np.zeros(np.broadcast(Ebeam, ne_cm3, Ti_eV, Zeff).shape)
lnq += lqefref * (1 - 4)
lnq += np.interp( np.log(Ti_eV), np.log(qtiev[qtiev.dims[0]].data), np.log(qtiev.data))
lnq += np.interp( np.log(ne_cm3), np.log(qdensi[qdensi.dims[0]].data), np.log(qdensi.data))
lnq += np.interp( np.log(Ebeam), np.log(qener[qener.dims[0]].data), np.log(qener.data))
lnq += np.interp( np.log(Zeff), np.log(qzeff[qzeff.dims[0]].data), np.log(qzeff.data))
lnq = np.exp(lnq) * 1e-6  # m^3 / s

print(' ')
print('Beam energy    = {:0.2e} eV/amu'.format(Ebeam))
print('Ne             = {:0.2e} cm^-3'.format(ne_cm3))
print('Ti             = {:0.2e} eV'.format(Ti_eV))
print('Zeff           = {:0.2f}'.format(Zeff))
print('CX coeeficient = {:0.6e} m^3/s'.format(lnq))


# Density and temperatures
ne_cm3_vec = np.linspace(5.0e12, 2.0e14, 100, dtype=float)
Ti_eV_vec  = np.linspace(200.0, 12000.0, 50, dtype=float)

lnq = np.zeros( (len(ne_cm3_vec), len(Ti_eV_vec)), dtype=float)
lnq += lqefref * (1 - 4)
lnq += np.interp( np.log(Ebeam), np.log(qener[qener.dims[0]].data), np.log(qener.data))
lnq += np.interp( np.log(Zeff), np.log(qzeff[qzeff.dims[0]].data), np.log(qzeff.data))
for i in range(len(ne_cm3_vec)):
    for j in range(len(Ti_eV_vec)):
        lnq[i, j] += np.interp( np.log(Ti_eV_vec[j]), np.log(qtiev[qtiev.dims[0]].data), np.log(qtiev.data))
        lnq[i, j] += np.interp( np.log(ne_cm3_vec[i]), np.log(qdensi[qdensi.dims[0]].data), np.log(qdensi.data))
lnq = np.exp(lnq) * 1e-6  # m^3 / s

print(np.shape(lnq))

plt.figure()
plt.contour(Ti_eV_vec, ne_cm3_vec*1e6, lnq, 100)
plt.colorbar()
plt.xlabel('Ti (eV)')
plt.ylabel('Ne (m^-3)')


# Plots of coefficient data

plt.figure()
qener.plot()
plt.show()

# qener.plot()
plt.figure()
plt.plot(qener[qener.dims[0]].data, qener.data, '.-')  # qener.beam_energy
plt.title('Beam energy')
plt.xlabel('Beam energy (eV/amu)')
plt.ylabel('Coeff. (cm^3/s)')

plt.figure()
plt.plot(qtiev[qtiev.dims[0]].data, qtiev.data, '.-')
plt.title('Ion temperature')
plt.xlabel('Ion temperature (eV)')
plt.ylabel('Coeff. (cm^3/s)')

plt.figure()
plt.plot(qdensi[qdensi.dims[0]].data, qdensi.data, '.-')
plt.title('Electron Denstiy')
plt.xlabel('Electron density (cm^-3)')
plt.ylabel('Coeff. (cm^3/s)')

plt.figure()
plt.plot(qzeff[qzeff.dims[0]].data, qzeff.data, '.-')
plt.title('Effective Charge')
plt.xlabel('Zeff (dimensionless)')
plt.ylabel('Coeff. (cm^3/s)')


plt.show()

