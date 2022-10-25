
from hda.models.interferometer import Interferometer
from hda.models.plasma import Plasma
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates
from indica.converters.line_of_sight import LineOfSightTransform

import matplotlib.pylab as plt
import numpy as np

from indica.readers import ST40Reader

plt.ion()

"""
Methods to build diagnostic models and los remapping
"""

MAIN_ION = "h"
IMPURITIES = ("c", "ar", "he")
IMPURITY_CONCENTRATION = (0.03, 0.001, 0.01)
FULL_RUN = False
# Create default plasma object
TSTART = 0.02
TEND = 0.1
DT = 0.01
PLASMA = Plasma(
    tstart=TSTART,
    tend=TEND,
    dt=DT,
    main_ion=MAIN_ION,
    impurities=IMPURITIES,
    impurity_concentration=IMPURITY_CONCENTRATION,
    full_run=FULL_RUN,
)



def fake_interferometer():

def interferometer(tstart:float=0.02, tend:float=0.1, dt:float=0.01, pulse:int = 10009):

    # Create plasma object
    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=MAIN_ION,
        impurities=IMPURITIES,
        impurity_concentration=IMPURITY_CONCENTRATION,
        pulse=pulse,
        full_run=FULL_RUN,
    )

    # Read equilibrium data and initialize Equilibrium and Flux-surface transform objects
    reader = ST40Reader(pulse, tstart-dt, tend+dt)

    uid, instrument, revision = ("", "efit", 0) # (sub-structure, diagnostic name, run identifier)
    equilibrium_data = reader.get(uid, instrument, revision)
    equilibrium = Equilibrium(equilibrium_data)
    flux_transform = FluxSurfaceCoordinates("poloidal")
    flux_transform.set_equilibrium(equilibrium)

    # Assign transforms to plasma object
    plasma.set_equilibrium(equilibrium)
    plasma.set_flux_transform(flux_transform)

    # create new interferometer and assign transforms for remapping
    diagnostic_name = "smmh2"
    smmh2 = Interferometer(name=diagnostic_name)
    los_start = np.array([0.8, 0, 0])
    los_end = np.array([0.17, 0, 0])
    origin = los_start
    direction = los_end - los_start
    los_transform = LineOfSightTransform(
        origin_x=origin[0],
        origin_y=origin[1],
        origin_z=origin[2],
        direction_x=direction[0],
        direction_y=direction[1],
        direction_z=direction[2],
        name=diagnostic_name,
        dl=0.006,
        machine_dimensions=plasma.machine_dimensions,
    )

    los_transform.set_flux_transform(plasma.flux_transform)
    _ = los_transform.convert_to_rho(t=plasma.t)
    smmh2.set_los_transform(los_transform)

    return plasma, smmh2
