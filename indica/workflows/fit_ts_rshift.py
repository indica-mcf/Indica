from xarray import DataArray
import xarray as xr
from indica.operators.spline_fit_Rshift import fit_profile_and_Rshift


def fit_ts(
    te_data: DataArray,
    te_err: DataArray,
    ne_data: DataArray,
    ne_err: DataArray,
    fit_Rshift: bool = True,
    verbose: bool = False,
):
    """
    Fit TS data including (or not) an ad-hoc Rshift of scattering volume positions
    (that can also be interpreted as a shift in the equilibrium)

    N.B. Rshift calculated only for Te and then applied to Ne!!!!

    Parameters
    ----------
    te_data - electron temperature data
    te_err - electron temperature error
    ne_data - electron density data
    ne_err - electron density data
    fit_Rshift - True if Rshift is to be fitted

    Returns
    -------
    Te and Ne fits, and corresponding Rshift

    """
    if not fit_Rshift:
        Rshift = xr.full_like(te_data.t, 0.0)
    else:
        Rshift = None

    ts_R = te_data.R
    ts_z = te_data.z
    equilibrium = te_data.transform.equilibrium
    print("  Te")
    te_fit, te_Rshift = fit_profile_and_Rshift(
        ts_R,
        ts_z,
        te_data,
        te_err,
        xknots=[0, 0.4, 0.6, 0.8, 1.1],
        equilibrium=equilibrium,
        Rshift=Rshift,
        verbose=verbose,
    )
    print("  Ne")
    ne_fit, ne_Rshift = fit_profile_and_Rshift(
        ts_R,
        ts_z,
        ne_data,
        ne_err,
        xknots=[0, 0.4, 0.8, 0.95, 1.1],
        equilibrium=equilibrium,
        Rshift=te_Rshift,
        verbose=verbose,
    )

    return te_fit, ne_fit, te_Rshift, ne_Rshift
