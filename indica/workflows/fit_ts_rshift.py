import xarray as xr
from xarray import DataArray

from indica.operators.spline_fit_R_shift import fit_profile_and_R_shift


def fit_ts(
    te_data: DataArray,
    te_err: DataArray,
    ne_data: DataArray,
    ne_err: DataArray,
    fit_R_shift: bool = True,
    verbose: bool = False,
):
    """
    Fit TS data including (or not) an ad-hoc R_shift of scattering volume positions
    (that can also be interpreted as a shift in the equilibrium)

    R_shift calculated only for Te and then applied to Ne!!!!

    Parameters
    ----------
    te_data - electron temperature data
    te_err - electron temperature error
    ne_data - electron density data
    ne_err - electron density data
    fit_R_shift - True if R_shift is to be fitted

    Returns
    -------
    Te and Ne fits, and corresponding R_shift

    """
    if not fit_R_shift:
        R_shift = xr.full_like(te_data.t, 0.0)
    else:
        R_shift = None

    ts_R = te_data.transform.R
    ts_z = te_data.transform.z
    equilibrium = te_data.transform.equilibrium
    if verbose:
        print("  Te")
    te_fit, te_R_shift, te_rho = fit_profile_and_R_shift(
        ts_R,
        ts_z,
        te_data,
        te_err,
        xknots=[0, 0.4, 0.85, 0.9, 0.98, 1.1],
        equilibrium=equilibrium,
        R_shift=R_shift,
        verbose=verbose,
    )
    if verbose:
        print("  Ne")
    ne_fit, ne_R_shift, ne_rho = fit_profile_and_R_shift(
        ts_R,
        ts_z,
        ne_data,
        ne_err,
        xknots=[0, 0.4, 0.85, 0.9, 0.98, 1.1],
        equilibrium=equilibrium,
        R_shift=te_R_shift,
        verbose=verbose,
    )

    te_fit.attrs["R_shift"] = te_R_shift
    ne_fit.attrs["R_shift"] = ne_R_shift
    te_data.attrs["rhop"] = te_rho
    ne_data.attrs["rhop"] = ne_rho

    return te_fit, ne_fit
