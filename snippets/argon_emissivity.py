from matplotlib import pyplot as plt

import indica
import numpy as np
import xarray as xr
from indica.models.passive_spectrometer import read_adf15s, format_pecs
from indica.configs.readers.adasconf import ADF15


adf15 = read_adf15s(elements=["ar", ], )
pecs = format_pecs(adf15, wavelength_bounds=None)

ar_pec = pecs["ar"].sel(electron_density = 1e19, method="nearest").sel(type="excit/ispb").sum(("ion_charge", ))



plt.figure()
plt.title("emissivity (photon/m^3/s)")
plt.imshow(ar_pec.values)
# plt.pcolormesh(ar_pec.electron_temperature.values, ar_pec.wavelength.values, ar_pec.values, norm="log")
# plt.colorbar()
# plt.axis("equal")
# plt.grid()
plt.show(block=True)
print()