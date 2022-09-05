from hda.profiles import Profiles
from hda.utils import print_like
from hda.manage_data import initialize_bckc_dataarray
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import least_squares
import xarray as xr
from xarray import DataArray
from xarray import Dataset

from indica.converters import FluxSurfaceCoordinates
from indica.converters.time import bin_in_time_dt
from indica.converters.time import get_tlabels_dt
from indica.datatypes import ELEMENTS
from indica.numpy_typing import LabeledArray
from indica.equilibrium import Equilibrium
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.provenance import get_prov_attribute
from indica.readers import ADASReader
from hda.models.xray_crystal_spectrometer import XRCSpectrometer

plt.ion()


def match_line_ratios(
    model: XRCSpectrometer,
    Te_prof: Profiles,
    data: dict,
    t: float,
    Ne: DataArray,
    Nimp: DataArray = None,
    Nh: DataArray = None,
    tau: DataArray = None,
    quantities: list = ["int_k/int_w", "int_n3/int_w", "int_n3/int_tot"],
    te0: float = 1.0e3,
    bckc:dict=None,
    bounds=(100.0, 10.0e3),
):

    """
    Optimize electron temperature profile to match XRCS line ratios.

    Parameters
    ----------
    model
        Forward model of the XRCS diagnostic
    Te_prof
        Profile object to build electron temperature profile for optimization
    data
        Dictionary of Dataarrays of XRCS data as returned by ST40reader
    t
        Time for which optimisation must be performed
    Ne
        Electron densit
    Nh
        Neutral (thermal) hydrogen density
    Nimp
        Total impurity densities as defined in plasma.py
    tau
        Residence time for the calculation of the ionisation balance
    quantities
        Measurement identifiers to be optimised
    te0
        Initial guess of central electron temperature (eV)
    bckc
        Dictionary where back-calculated values are to be saved (same structure as data)
    """

    def residuals(te0):
        Te_prof.y0 = te0
        Te_prof.build_profile()

        model.calculate_emission(Te_prof.yspl, Ne, Nimp=Nimp, Nh=Nh, tau=tau)
        bckc_value, _ = model.integrate_on_los(t=t)

        resid = []
        for quantity in quantities:
            resid.append(data_value[quantity] - bckc_value[quantity])

        return np.array(resid).sum()

    # Initialize variables
    if Te_prof.y0 < 1.0e3:
        Te_prof.y0 = 1.0e3
        Te_prof.build_profile()

    data_value = {}
    for quantity in quantities:
        data_value[quantity] = data[quantity].sel(t=t)

    least_squares(residuals, te0, bounds=bounds, method="dogbox")

    bckc_tmp = {}
    for quantity in quantities:
        bckc_tmp[quantity] = model.los_integral[quantity]

    if bckc is None:
        bckc = bckc_tmp
    else:
        for quantity in quantities:
            bckc[quantity].loc[dict(t=t)] = bckc_tmp[quantity]

    return bckc, Te_prof


def match_ion_temperature(
    model: XRCSpectrometer,
    Ti_prof: Profiles,
    data: dict,
    t: float,
    Te: DataArray = None,
    Ne: DataArray = None,
    Nimp: DataArray = None,
    Nh: DataArray = None,
    tau: DataArray = None,
    quantities: list = ["ti_w", "ti_z"],
    lines: list = ["w", "z"],
    bckc:dict=None,
    ti0: float = 1.0e3,
    te0_ref = None,
    wcenter_exp=0.05,
    method="moment",
    bounds=(100.0, 10.0e3),
):

    """
    Optimize ion temperature profile to match XRCS measured line widths.

    Parameters
    ----------
    model
        Forward model of the XRCS diagnostic
    Ti_prof
        Profile object to build ion temperature profile for optimization
    data
        Dictionary of Dataarrays of XRCS data as returned by ST40reader
    t
        Time for which optimisation must be performed
    Te
        Electron temperature
    Ne
        Electron densit
    Nh
        Neutral (thermal) hydrogen density
    Nimp
        Total impurity densities as defined in plasma.py
    tau
        Residence time for the calculation of the ionisation balance
    quantities
        Measurement identifiers to be optimised
    lines
        Spectral lines used to calculate the experimental ion temperature
    bckc
        Dictionary where back-calculated values are to be saved (same structure as data)
    ti0
        Initial guess of central ion temperature (eV)
    te0_ref
        Increase peaking if Ti(0) > Te(0) as explained in Profiles
    wcenter_exp
        Exponent for central peaking when te0_ref is not None
    method
        Method for the calculation ("moment" = use moment_analysis, "gaussians" = sum gaussians, ...)
    """

    def residuals_moment(ti0):
        Ti_prof.y0 = ti0
        Ti_prof.build_profile(y0_ref=te0_ref, wcenter_exp=wcenter_exp)

        resid = []
        for quantity, line in zip(quantities, lines):
            bckc_value = model.moment_analysis(Ti_prof.yspl, t, line=line)
            resid.append(data_value[quantity] - bckc_value)

        return np.array(resid).sum()

    # Initialize variables
    if Ti_prof.y0 < 1.0e3:
        Ti_prof.y0 = 1.0e3
        Ti_prof.build_profile(y0_ref=te0_ref, wcenter_exp=wcenter_exp)

    # Calculate emission if inputs are different or not present in model
    if (
        not hasattr(model, "Te")
        or not hasattr(model, "Ne")
        or not hasattr(model, "Nh")
        or not hasattr(model, "Nimp")
        or not hasattr(model, "tau")
    ):
        model.calculate_emission(Te, Ne, Nimp=Nimp, Nh=Nh, tau=tau)
    else:
        if (Te is not None) and (Ne is not None) and (Nh is not None) and (Nimp is not None):
            if (
                np.array_equal(Te, model.Te)
                or np.array_equal(Ne, model.Ne)
                or np.array_equal(Nh, model.Nh)
                or np.array_equal(Nimp, model.Nimp)
                or np.array_equal(tau, model.tau)
            ):
                model.calculate_emission(Te, Ne, Nimp=Nimp, Nh=Nh, tau=tau)

    model.map_to_los(t=t)

    data_value = {}
    for quantity in quantities:
        data_value[quantity] = data[quantity].sel(t=t)

    if method == "moment":
        least_squares(residuals_moment, ti0, bounds=bounds, method="dogbox")
    else:
        raise Exception("No other optimisation method currently supported...")

    bckc_tmp = {}
    for quantity, line in zip(quantities, lines):
        bckc_tmp[quantity] = model.moment_analysis(Ti_prof.yspl, t, line=line)

    if bckc is None:
        bckc = bckc_tmp
    else:
        for quantity in quantities:
            bckc[quantity].loc[dict(t=t)] = bckc_tmp[quantity]

    return bckc, Ti_prof


def match_xrcs_intensity(
    self,
    data,
    bckc={},
    diagnostic: str = "xrcs",
    quantity: str = "int_w",
    elem="ar",
    cal=2.0e3,
    niter=2,
    time=None,
    scale=True,
):
    """
    TODO: separate calculation of line intensity from optimisation
    TODO: tau currently not included in calculation
    Compute Ar density to match the XRCS spectrometer measurements

    Parameters
    ----------
    data
        diagnostic data as returned by build_data()
    bckc
        back-calculated data
    diagnostic
        diagnostic name corresponding to xrcs
    quantity_int
        Measurement to be used for determining the impurity concentration
        from line intensity
    cal
        Calibration factor for measurement
        Default value calculated to match Zeff from LINES.BREMS_MP for pulse 9408
    elem
        Element responsible for measured spectra
    niter
        Number of iterations

    Returns
    -------

    """

    if diagnostic not in data.keys():
        print_like(f"No {diagnostic.upper()} data available")
        return

    if time is None:
        time = self.t

    if diagnostic not in bckc:
        bckc[diagnostic] = {}
    if quantity not in bckc[diagnostic].keys():
        bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)
    line = quantity.split("_")[1]

    # Initialize back calculated values of diagnostic quantities
    forward_model = self.forward_models[diagnostic]
    dl = data[diagnostic][quantity].attrs["dl"]
    for t in time:
        print(t)

        int_data = data[diagnostic][quantity].sel(t=t)
        Te = self.el_temp.sel(t=t)
        if np.isnan(Te).any():
            continue

        Ne = self.el_dens.sel(t=t)
        # tau = self.tau.sel(t=t)
        Nh = self.neutral_dens.sel(t=t)
        if np.isnan(Te).any():
            continue
        rho_los = data[diagnostic][quantity].attrs["rho"].sel(t=t)

        const = 1.0
        for j in range(niter):
            Nimp = {elem: self.imp_dens.sel(element=elem, t=t) * const}
            _ = forward_model(Te, Ne, Nimp=Nimp, Nh=Nh, rho_los=rho_los, dl=dl,)
            int_bckc = forward_model.intensity[line] * cal
            const = (int_data / int_bckc).values

            if (np.abs(1 - const) < 1.0e-4) or not (scale):
                break

        self.imp_dens.loc[dict(element=elem, t=t)] = Nimp[elem].values
        bckc[diagnostic][quantity].loc[dict(t=t)] = int_bckc.values

    bckc[diagnostic][quantity].attrs["calibration"] = cal

    revision = get_prov_attribute(data[diagnostic][quantity].provenance, "revision")
    self.optimisation["imp_dens"] = f"{diagnostic}.{quantity}:{revision}"

    return bckc
