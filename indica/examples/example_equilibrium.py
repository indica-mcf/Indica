from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

from indica.defaults.load_defaults import load_default_objects


def flux_coords(machine: str = "st40"):
    """Convert from (R, z) to  (rhop, theta)"""
    equilibrium = load_default_objects(machine, "equilibrium")

    # Coordinates to convert
    # R & z common coord "channel" tells DataArray to return 1D array
    _R = np.array([0.7, 0.6, 0.5, 0.4, 0.3])
    _z = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    R = xr.DataArray(_R, coords={"channel": np.arange(len(_R))})
    z = xr.DataArray(_z, coords={"channel": np.arange(len(_z))})

    # Convert (R, z) to (rhop, theta) at time = t
    t = 0.05
    rhop_mid, theta, t = equilibrium.flux_coords(R, z, t=t)

    # Shift one channel in z and see what happens
    _z[3] += 0.3
    z = xr.DataArray(_z, coords={"channel": np.arange(len(_z))})
    rhop_shift, theta_shift, t = equilibrium.flux_coords(R, z, t=t)

    plt.figure()
    rhop_mid.plot(marker="o", label="Nominal")
    rhop_shift.plot(marker="x", label="Chan. 2 with z += 0.3")
    plt.ylabel("R (m)")
    plt.legend()
