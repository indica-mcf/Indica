"""Various miscellaneous functions for handling of atomic data."""

import numpy as np
from xarray import DataArray
from indica.numpy_typing import ArrayLike
from typing import Dict


def atomdat_files(element: str) -> Dict:
    file_info = None
    if element.lower() == "ar":
        file_info = {
            "scd": "89",
            "acd": "89",
            "plt": "89",
            "prb": "89",
            "pec": ("llu", "89"),
        }
    elif element.lower() == "c":
        file_info = {
            "scd": "89",
            "acd": "89",
            "plt": "89",
            "prb": "89",
            "pec": ("bnd", "96"),
        }
    else:
        print("\n Element not implemented! \n")

    return file_info


def get_atomdat(reader, element, charge, transition=None, wavelength=None):
    files = atomdat_files(element)
    atomdat = {}
    for k in files.keys():
        if k == "pec":
            atomdat[k] = reader.get_adf15(
                element, charge, files[k][0], year=files[k][1]
            )
        elif k in ["scd", "acd", "plt", "prb"]:
            atomdat[k] = reader.get_adf11(k, element, files[k])
        else:
            "\n File format not supported... \n"

    if transition:
        atomdat["pec"] = (
            atomdat["pec"]
            .swap_dims({"index": "transition"})
            .sel(transition=transition, drop=True)
        )
        if "transition" in atomdat["pec"].dims:
            atomdat["pec"] = atomdat["pec"].swap_dims({"transition": "index"})
    if wavelength:
        if len(np.unique(atomdat["pec"].coords["wavelength"].values)) > 1:
            atomdat["pec"] = (
                atomdat["pec"]
                .swap_dims({"index": "wavelength"})
                .sel(wavelength=wavelength, method="nearest", drop=True)
            )

    return files, atomdat


def fractional_abundance(
    scd: ArrayLike,
    acd: ArrayLike,
    ne_tau=1.0,
    gen_type="fractional_abundance",
    element="",
) -> ArrayLike:
    """Returns the equilibrium fractional abundance given ionization and recombination
    rates as read from ADAS adf11 files with original file dimensions. Rate variables
    must be 1d (ion_charges) or 2d (ion_charges, electron, temperature)

    Still to be included:
        * provenance
        * charge-exchange rates
        * impurity residence time ne_tau for transient equilibrium [L. Casali EPJ 79, 01007 (2014),
        A. Kallenbach PPCF 55 (2013) 124041]. Current version is not correct.

    Parameters
    ----------
    scd
        Effective ionization coefficients.
    acd
        Effective recombination coefficients.
    ne_tau
        Residence time in units of (m**-3 s)
    gen_type
        General datatype
    element
        Element (specific datatype for name and attribute)

    Returns
    -------
    fz
        The fractional abundance of the ion.

    """

    assert scd.shape == acd.shape
    dim1, dim2 = scd.dims
    coords = [
        (dim1, np.arange(scd.coords[dim1].max() + 2)),
        (dim2, scd.coords[dim2]),
    ]

    nz = scd.shape[0]
    nother = scd.shape[1]
    fz = np.ones((nz + 1, nother))
    fz = DataArray(
        fz,
        coords=coords,
    )

    fz[0, :] = fz[0, :] * 10 ** (-scd[0, :] + acd[0, :]) * ne_tau
    for i in range(1, nz):
        fz[i, :] = fz[i - 1, :] * 10 ** (scd[i - 1, :] - acd[i - 1, :]) * ne_tau
    fz[i + 1, :] = fz[i, :] * 10 ** (scd[i, :] - acd[i, :]) * ne_tau
    for j in range(nother):
        norm = np.nansum(fz[:, j], axis=0)
        fz[:, j] /= norm

    fz[i + 1, :] = fz[i, :] * 10 ** (scd[i, :] - acd[i, :]) * ne_tau
    for i in range(nz - 1, 0, -1):
        fz[i, :] = fz[i - 1, :] * 10 ** (scd[i - 1, :] - acd[i - 1, :]) * ne_tau
    fz[0, :] = fz[0, :] * 10 ** (-scd[0, :] + acd[0, :]) * ne_tau
    for j in range(nother):
        norm = np.nansum(fz[:, j], axis=0)
        fz[:, j] /= norm

    if gen_type and element:
        attrs = {
            "datatype": (gen_type, element),
            "provenance": "",
        }
        name = f"{element}_{gen_type}"
        fz.attrs = attrs
        fz.name = name

    return fz


def radiated_power(
    plt: ArrayLike,
    prb: ArrayLike,
    fz: ArrayLike,
    gen_type="radiated_power",
    element="",
) -> ArrayLike:
    """Returns the radiated power coefficients given the fractional abundance, line and recombination rates
    read from ADAS adf11 files with original file dimensions. Coefficients must be 1d (ion_charges) or
    2d (ion_charges, electron, temperature)

    Still to be included:
        * provenance

    Parameters
    ----------
    plt
        Line power driven by excitation of dominant ions
    prb
        Continuum and line power driven by recombination and Bremsstrahlung of dominant ions
    fz
        Fractional abundance of element ionization stages

    Returns
    -------
    rad_pow
        The radiated power coefficients (W m**3)

    """

    assert plt.shape == prb.shape
    dim1, dim2 = plt.dims
    coords = [
        (dim1, np.arange(plt.coords[dim1].max() + 2)),
        (dim2, plt.coords[dim2]),
    ]

    nz = plt.shape[0]
    nother = plt.shape[1]
    rad_pow = np.ones((nz + 1, nother))
    rad_pow = DataArray(
        rad_pow,
        coords=coords,
    )

    rad_pow[0, :] = fz[0, :] * 10 ** plt[0, :]
    for i in range(1, nz):
        rad_pow[i, :] = fz[i, :] * (10 ** plt[i, :] + 10 ** prb[i - 1, :])
    rad_pow[i + 1, :] = fz[i + 1, :] * 10 ** prb[i, :]

    if gen_type and element:
        attrs = {
            "datatype": (gen_type, element),
            "provenance": "",
        }
        name = f"{element}_{gen_type}"
        rad_pow.attrs = attrs
        rad_pow.name = name

    return rad_pow
