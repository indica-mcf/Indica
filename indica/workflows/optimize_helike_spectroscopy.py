from indica.profiles import Profiles
from indica.utilities import print_like
from indica.readers.manage_data import initialize_bckc_dataarray
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
from indica.models.helike_spectroscopy import Helike_spectroscopy
from indica.models.plasma import Plasma

plt.ion()


def check_model_inputs(model, Te, Ne, Nh, Nimp, tau):
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
        if (
            (Te is not None)
            and (Ne is not None)
            and (Nh is not None)
            and (Nimp is not None)
        ):
            if (
                np.array_equal(Te, model.Te)
                or np.array_equal(Ne, model.Ne)
                or np.array_equal(Nh, model.Nh)
                or np.array_equal(Nimp, model.Nimp)
                or np.array_equal(tau, model.tau)
            ):
                model.calculate_emission(Te, Ne, Nimp=Nimp, Nh=Nh, tau=tau)

    return model.Te, model.Ne, model.Nh, model.Nimp, model.tau


def match_line_ratios(
    models: dict,
    plasma:Plasma,
    data: dict,
    t: float,
    instruments: list = ["xrcs"],
    quantities: list = ["int_k/int_w", "int_n3/int_w", "int_n3/int_tot"],
    te0: float = 1.0e3,
    bounds=(100.0, 20.0e3),
):
    model = models["xrcs"]
    list_data = []
    list_model = []
    list_quantity = []
    for instrument in instruments:
        if instrument in models:
            for quantity in quantities:
                list_model.append(models[instrument])
                list_data.append(data[instrument])
                list_quantity.append(quantity)

    """
    Optimize electron temperature profile to match XRCS line ratios.
    """


    def residuals(te0):
        Te_prof.y0 = te0
        Te_prof.build_profile()
        Te = Te_prof.yspl

        bckc = model(Te, Ne, Nimp=Nimp, Nh=Nh, tau=tau)

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

    bckc = {}
    for quantity in quantities:
        bckc[quantity] = model.los_integral[quantity]

    return bckc, Te_prof


def match_ion_temperature(
    model: Helike_spectroscopy,
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
    ti0: float = 1.0e3,
    te0_ref=None,
    wcenter_exp=0.05,
    method="moment",
    bounds=(100.0, 20.0e3),
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

    check_model_inputs(model, Te, Ne, Nh, Nimp, tau)

    data_value = {}
    for quantity in quantities:
        data_value[quantity] = data[quantity].sel(t=t)

    if method == "moment":
        least_squares(residuals_moment, ti0, bounds=bounds, method="dogbox")
    else:
        raise Exception("No other optimisation method currently supported...")

    bckc = {}
    for quantity, line in zip(quantities, lines):
        bckc[quantity] = model.moment_analysis(Ti_prof.yspl, t, line=line)

    return bckc, Ti_prof


def match_intensity(
    model: Helike_spectroscopy,
    Nimp_prof: Profiles,
    data: dict,
    t: float,
    Te: DataArray = None,
    Ne: DataArray = None,
    Nimp: DataArray = None,
    Nh: DataArray = None,
    tau: DataArray = None,
    quantities: list = ["int_w"],
    lines: list = ["w"],
    nimp0: float = 1.0e15,
    bounds=(1.0e12, 1.0e21),
):

    """
    Optimize line intensity to match XRCS measurements

    Parameters
    ----------
    model
        Forward model of the XRCS diagnostic
    Nimp_prof
        Profile object to build impurity temperture profile for optimization
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
    tau
        Residence time for the calculation of the ionisation balance
    quantities
        Measurement identifiers to be optimised
    lines
        Spectral lines used to calculate the experimental ion temperature
    nimp0
        Initial guess of central impurity density
    """

    def scale_impurity_density(nimp0, niter=3, bounds:tuple=(1.0e12, 1.0e21)):
        # Scale whole profile
        resid = []

        element = model.adf15[lines[0]]["element"]

        mult = nimp0 / Nimp_prof.y0
        for i in range(niter):
            if Nimp_prof.y0 * mult < bounds[0]:
                mult = bounds[0]/Nimp_prof.y0
            if Nimp_prof.y0 * mult > bounds[1]:
                mult = bounds[1]/Nimp_prof.y0
            Nimp_prof.y0 *= mult
            Nimp_prof.y1 *= mult
            Nimp_prof.yend *= mult
            Nimp_prof.build_profile()

            Nimp.loc[dict(element=element)] = Nimp_prof.yspl.interp(
                rho_poloidal=Nimp.sel(element=element).rho_poloidal
            )

            model.calculate_emission(Te, Ne, Nimp=Nimp, Nh=Nh, tau=tau)
            model.integrate_on_los(t=t)

            _mult = []
            for quantity, line in zip(quantities, lines):
                bckc_value = model.los_integral[line]
                _mult.append((data_value[quantity]/bckc_value).values)

            mult = np.array(_mult).mean()

        return np.array(resid).sum()

    Te, Ne, Nh, Nimp, tau = check_model_inputs(model, Te, Ne, Nh, Nimp, tau)
    model.integrate_on_los(t=t)

    data_value = {}
    for quantity in quantities:
        data_value[quantity] = data[quantity].sel(t=t)

    scale_impurity_density(nimp0, bounds=bounds)

    bckc = {}
    for quantity, line in zip(quantities, lines):
        bckc[quantity] = model.los_integral[line]

    return bckc, Nimp_prof
