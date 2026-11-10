import numpy as np

from indica import Equilibrium, Plasma
from indica.converters import LineOfSightTransform
from indica.operators import NbiADAS
from indica.readers import ST40Reader

# ---------------------------------------------------------------------
# 1. Read ST40 shot and equilibria
# ---------------------------------------------------------------------
pulse = 12345
t = 0.05  # s
reader = ST40Reader(pulse, t - 0.02, t + 0.02)

# ST40 'efit' is used in the examples
efit_data = reader.get("", "efit", 0)
equilibrium = Equilibrium(efit_data)

# ---------------------------------------------------------------------
# 2. Load ST40 plasma profiles
# ---------------------------------------------------------------------
# These are placeholders. In a real script, you would replace
# these with the actual ST40 diagnostics for your shot and time.
#
# Typical pattern:
# - use Thomson scattering for Ne and Te
# - use charge exchange or profile fits for Ti
# - use impurity density data if available
#
# The exact data source depends on the ST40 tree and what was saved.
#
# Example pattern (same idea as the JET script):
# ne = ...
# te = ...
# ti = ...
# ni = ...
#
# Then convert to a uniform rhop grid:
# ne = ne.assign_coords(rhop=...).swap_dims(...).interp(rhop=rhop)...
# te = te.assign_coords(rhop=...).swap_dims(...).interp(rhop=rhop)...
# ti = ti.assign_coords(rhop=...).swap_dims(...).interp(rhop=rhop)...
# ni = ni.assign_coords(rhop=...).swap_dims(...).interp(rhop=rhop)...

# For a minimal template, assume we already have profile DataArrays
# with rhop as the radial coordinate.
# Example placeholders:
# ne = ...
# te = ...
# ti = ...
# ni = ...

# If you do not have impurity profiles, you can still construct a Plasma
# object and fill only the main profiles as needed.

plasma = Plasma(
    tstart=t,
    tend=t + 1e-3,
    dt=1e-3,
    machine="st40",
    impurities=("ne",),  # replace if needed
)

# Example only: these lines are placeholders and should be replaced.
# plasma.electron_density.loc[:] = ne.interp(rhop=plasma.rhop)
# plasma.electron_temperature.loc[:] = te.interp(rhop=plasma.rhop)
# plasma.ion_temperature.loc[:] = ti.interp(rhop=plasma.rhop)
# plasma.impurity_density.loc[:] = ni.interp(rhop=plasma.rhop)

# ---------------------------------------------------------------------
# 3. Define the ST40 NBI line-of-sight
# ---------------------------------------------------------------------
# This must come from the actual ST40 NBI geometry for the relevant beam.
# Replace with real beam origin/direction from your ST40 data.
dist_centre = 3.0  # m, placeholder
ang_beamline = -20.0  # degrees, placeholder

origin = np.asarray([dist_centre, 0.0, 0.0])
direction = np.asarray(
    [
        -np.cos(np.deg2rad(ang_beamline)),
        np.sin(np.deg2rad(ang_beamline)),
        0.0,
    ],
)
direction /= np.linalg.norm(direction)

tr = LineOfSightTransform(
    np.asarray([origin[0]]),
    np.asarray([origin[1]]),
    np.asarray([origin[2]]),
    np.asarray([direction[0]]),
    np.asarray([direction[1]]),
    np.asarray([direction[2]]),
    name="ST40_NBI",
)
tr.set_equilibrium(equilibrium)

# ---------------------------------------------------------------------
# 4. Read NBI power and energy data
# ---------------------------------------------------------------------
# In a real ST40 script, this would come from the NBI instrument
# in the ST40 tree. The ST40 reader already supports an "_nbi" method
# in indica/readers/st40reader.py, so this is the intended route.
#
# You would normally get:
# - beam energy
# - beam power
# - power fractions / current fractions
# from the ST40 NBI diagnostic in your database.
#
# Example placeholders only:
beam_energy = 50e3  # eV
beam_power = 1.0e6  # W
power_fractions = (0.5, 0.3, 0.2)
power_fractions_identifiers = (1.0, 0.5, 1.0 / 3.0)

# ---------------------------------------------------------------------
# 5. Construct the NBI attenuation model
# ---------------------------------------------------------------------
pini = NbiADAS(
    name=tr.name,
    energy=beam_energy,
    power=beam_power,
    nbi_element="d",  # change to "h" if your ST40 beam is hydrogen
    power_fractions=power_fractions,
    power_fractions_identifiers=power_fractions_identifiers,
)
pini.set_transform(tr)

# ---------------------------------------------------------------------
# 6. Run the model
# ---------------------------------------------------------------------
# The exact plasma arrays are placeholders. In a real run, these are the
# actual ST40 electron density, temperature, ion temperature, impurity density.
results = pini(
    Ti=plasma.ion_temperature,
    Te=plasma.electron_temperature,
    Ne=plasma.electron_density,
    Ni=plasma.ion_density,
    Vtor=plasma.toroidal_rotation,
    t=t,
)

zeta = results["zeta"]
print(zeta)