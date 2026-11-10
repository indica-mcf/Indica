from indica.defaults.load_defaults import load_default_objects
from indica.operators.adas_nbioperator import NbiADAS

import numpy as np
from indica import Equilibrium, Plasma
from indica.converters import LineOfSightTransform
from indica.operators import FractionalAbundanceAdas
from indica.readers import ST40Reader, SALUtils
from indica.utilities import build_dataarrays

reader = ST40Reader(11089, 0.02, 0.15)
data = reader(["efit", "ppts"])
equil = Equilibrium(data["efit"])


te = data["ppts"]["te_rhop"]
ne = data["ppts"]["ne_rhop"]

plasma = Plasma(0.02, 0.15, 0.01, machine="st40", impurities=("c",))
plasma.electron_density.loc[:] = ne.interp(rhop=plasma.rhop, t=plasma.t)
plasma.electron_temperature.loc[:] = te.interp(rhop=plasma.rhop, t=plasma.t)
# plasma.ion_temperature.loc[:] = ti.interp(rhop=plasma.rhop)
# plasma.impurity_density.loc[:] = ni.interp(rhop=plasma.rhop)
plasma.fract_abu = {
    elem: FractionalAbundanceAdas(elem, full_run=True) for elem in plasma.elements
}

default_transforms = load_default_objects("st40", "geometry")

nbi = default_transforms["xrcs"]
# origin_x = np.array([3.322], dtype=float)  # DB: LOCATION[..., x] [m]
# origin_y = np.array([3.945], dtype=float)  # DB: LOCATION[..., y] [m]
# origin_z = np.array([0.0], dtype=float)  # DB: LOCATION[..., z] [m]
# direction_x = np.array([-0.704], dtype=float)  # DB: DIRECTION[..., xhat]
# direction_y = np.array([-0.7099], dtype=float)  # DB: DIRECTION[..., yhat]
# direction_z = np.array([0.0], dtype=float) # DB: DIRECTION[..., zhat]
#
# nbi = LineOfSightTransform(
#     origin_x,
#     origin_y,
#     origin_z,
#     direction_x,
#     direction_y,
#     direction_z,
#     name="hnbi",
# )


nbi.set_equilibrium(equil)


# Example placeholders only:
beam_energy = 52e3  # eV
beam_power = 0.5e6  # W
# beam_energy *= 1e-3  # Convert to keV
# beam_power *= 1e-6  # Convert to MW
power_fractions = (0.5, 0.35, 0.15)
power_fractions_identifiers = np.array((1.0, 1/2.0, 1/3.0),)


beam_model = NbiADAS(
    name=nbi.name,
    energy=beam_energy,
    power=beam_power,
    nbi_element="d",
    power_fractions=power_fractions,
    power_fractions_identifiers=power_fractions_identifiers,
)
beam_model.set_transform(nbi)

plasma.impurity_density += 2e18
plasma.toroidal_rotation += 1000
plasma.ion_temperature += 1000
results = beam_model(
    Ti=plasma.ion_temperature,
    Te=plasma.electron_temperature,
    Ne=plasma.electron_density,
    Ni=plasma.ion_density,
    Vtor=plasma.toroidal_rotation,
    t=0.05,
)
zeta = results["zeta"]  # Currently only result is ZETA factor

import matplotlib.pyplot as plt
# plt.figure()
# plasma.ion_temperature.sel(t=0.05, method="nearest").plot()
# plt.figure()
# plasma.electron_density.sel(t=0.05, method="nearest").plot()
# plt.figure()
# plasma.electron_temperature.sel(t=0.05, method="nearest").plot()
# plt.figure()
# plasma.impurity_density.sel(t=0.05, method="nearest").plot()
# plt.figure()
# plasma.ion_density.sel(element="h").sel(t=0.05, method="nearest").plot()
# plt.figure()
# plasma.toroidal_rotation.sel(t=0.05, method="nearest").plot()
# plt.figure()
print(zeta)
plt.plot(zeta.los_position.values, zeta[0, 0, 0, :].values, label=f"power fraction: {power_fractions_identifiers[0]}", marker="o")
plt.plot(zeta.los_position.values, zeta[1, 0, 0, :].values, label=f"power fraction: {power_fractions_identifiers[1]}", marker="x")
plt.plot(zeta.los_position.values, zeta[2, 0, 0, :].values, label=f"power fraction: {power_fractions_identifiers[2]}", linestyle="-.")
plt.legend()
plt.show(block=True)