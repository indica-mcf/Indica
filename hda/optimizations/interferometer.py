import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray
from indica.provenance import get_prov_attribute
from hda.manage_data import initialize_bckc

plt.ion()
def interferometer(self, data, bckc={}, diagnostic=None, quantity=None):
    """
    Calculate expected diagnostic measurement given plasma profile
    """

    if diagnostic is None:
        diagnostic = ["nirh1", "nirh1_bin", "smmh1"]
    if quantity is None:
        quantity = ["ne"]
    diagnostic = list(diagnostic)
    quantity = list(quantity)

    for diag in diagnostic:
        if diag not in data.keys():
            continue

        for quant in quantity:
            if quant not in data[diag].keys():
                continue

            bckc = initialize_bckc(diag, quant, data, bckc=bckc)

            bckc[diag][quant].values = self.calc_ne_los_int(
                data[diag][quant]
            ).values

    return bckc

def match_interferometer(
    self,
    data,
    bckc={},
    diagnostic: str = "nirh1",
    quantity: str = "ne_bin",
    error=False,
    niter=3,
    time=None,
):
    """
    Rescale density profiles to match the interferometer measurements

    Parameters
    ----------
    data
        diagnostic data as returned by build_data()
    bckc
        back calculated data dictionary
    interf
        Name of interferometer to be used

    Returns
    -------

    """
    print_like(
        f"Re-calculating density profiles to match {diagnostic}.{quantity} values"
    )

    if time is None:
        time = self.t

    # TODO: make more elegant optimisation

    bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)

    Ne_prof = self.Ne_prof
    for t in time:
        const = 1.0
        for j in range(niter):
            ne0 = Ne_prof.yspl.sel(rho_poloidal=0) * const
            ne0 = xr.where((ne0 <= 0) or (not np.isfinite(ne0)), 5.0e19, ne0)
            Ne_prof.y0 = ne0.values
            Ne_prof.build_profile()
            self.el_dens.loc[dict(t=t)] = Ne_prof.yspl.values
            ne_bckc = self.calc_ne_los_int(data[diagnostic][quantity], t=t)
            const = (data[diagnostic][quantity].sel(t=t) / ne_bckc).values

        bckc[diagnostic][quantity].loc[dict(t=t)] = ne_bckc.values

    revision = get_prov_attribute(data[diagnostic][quantity].provenance, "revision")
    self.optimisation["el_dens"] = f"{diagnostic}.{quantity}:{revision}"
    self.optimisation["stored_en"] = ""

    return bckc

def recover_density(
    self, data, diagnostic: str = "efit", quantity: str = "wp", niter=3,
):
    """
    Match stored energy by adapting electron density

    Parameters
    ----------
    data
    diagnostic
    quantity
    niter

    Returns
    -------

    """
    print("\n Re-calculating density to match plasma energy \n")

    Ne_prof = self.Ne_prof
    const = DataArray([1.0] * len(self.t), coords=[("t", self.t)])
    ne0 = self.el_dens.sel(rho_poloidal=0)
    data_tmp = data[diagnostic][quantity]
    for j in range(niter):
        for t in self.t:
            if np.isfinite(const.sel(t=t)):
                Ne_prof.y0 = (ne0 * const).sel(t=t).values
                Ne_prof.build_profile()
                self.el_dens.loc[dict(t=t)] = Ne_prof.yspl.values
        self.calc_imp_dens()
        bckc_tmp = self.wp.sel()
        const = 1 + (data_tmp - bckc_tmp) / bckc_tmp

    revision = get_prov_attribute(data[diagnostic][quantity].provenance, "revision")
    self.optimisation["stored_en"] = f"{diagnostic}.{quantity}:{revision}"

def calc_ne_los_int(self, data, passes=2, t=None):
    """
    Calculate line of sight integral for a specified number of
    passes through the plasma

    Returns
    -------
    los_int
        Integral along the line of sight

    """
    dl = data.attrs["dl"]
    rho = data.attrs["rho"]
    transform = data.attrs["transform"]

    x2_name = transform.x2_name

    el_dens = self.el_dens.interp(rho_poloidal=rho)
    if t is not None:
        el_dens = el_dens.sel(t=t, method="nearest")
        rho = rho.sel(t=t, method="nearest")
    el_dens = xr.where(rho <= 1, el_dens, 0,)
    el_dens_int = passes * el_dens.sum(x2_name) * dl

    return el_dens_int

