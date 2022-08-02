import xarray as xr
from xarray import DataArray

def interferometer(rho_los, dl, el_dens, passes=2):
    ne_los_int = calc_los_int(rho_los, dl, el_dens, passes=passes)
    return ne_los_int * passes


def calc_los_int(rho_los, dl, el_dens:DataArray):
    el_dens_rho = el_dens.interp(rho_poloidal=rho_los)
    el_dens_rho = xr.where(rho_los <= 1, el_dens_rho, 0,).values
    return el_dens_rho.sum() * dl
