
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates
from indica.readers import ST40Reader
from indica.converters.time import convert_in_time_dt


def remap(diag_data, flux_transform, npts=100):
    """
    Calculate mapping on equilibrium for specified diagnostic
    """
    new_attrs = {}
    trans = diag_data.transform
    x2 = DataArray(np.linspace(0, 1, npts), dims=trans.x2_name)
    dl = trans.distance(trans.x2_name, DataArray(0), x2[0:2], 0)[1]
    R, z = trans.convert_to_Rz(diag_data.coords[trans.x1_name], x2, 0)

    dt_equil = flux_transform.equilibrium.rho.t[1] - flux_transform.equilibrium.rho.t[0]
    dt_data = diag_data.t[1] - diag_data.t[0]
    if dt_data > dt_equil:
        t = diag_data.t
    else:
        t = None
    rho_equil, _ = flux_transform.convert_from_Rz(R, z, t=t)

    rho = convert_in_time_dt(diag_data.t[0].values, diag_data.t[-1].values, dt_data.values, rho_equil)
    rho = xr.where(rho >= 0, rho, 0.0)
    rho.coords[trans.x2_name] = x2

    new_attrs["x2"] = x2
    new_attrs["dl"] = dl
    new_attrs["R"], new_attrs["z"] = R, z
    new_attrs["rho"] = rho

    return new_attrs

def remap_xrcs(pulse, tstart, tend, plot=True):
    st40reader = ST40Reader(pulse, tstart, tend)

    xrcs_revision = 1
    efit_revision = 1
    xrcs = st40reader.get("sxr", "xrcs", xrcs_revision)
    efit = st40reader.get("", "efit", efit_revision)

    equilibrium = Equilibrium(efit)
    flux_coords = FluxSurfaceCoordinates("poloidal")
    flux_coords.set_equilibrium(equilibrium)

    for kquant in xrcs.keys():
        xrcs[kquant].transform.set_equilibrium(equilibrium, force=True)
        geom_attrs = remap(xrcs[kquant], flux_coords)
        for kattrs in geom_attrs:
            xrcs[kquant].attrs[kattrs] = geom_attrs[kattrs]

    if plot:
        tmid = np.mean([tstart, tend])

        plt.figure()
        equilibrium.rho.sel(t=tmid, method="nearest").plot.contour(levels=15, label="Equilibrium")
        plt.axis("scaled")
        plt.plot(xrcs[kquant].R, xrcs[kquant].z, color="red", label="XRCS LOS")
        plt.legend()

        plt.figure()
        xrcs[kquant].rho.sel(t=tmid, method="nearest").plot(marker="o", label="Rho along LOS")
        plt.xlabel("rho-poloidal")
        plt.legend()



    return xrcs