"""Various miscellaneous functions for handling of atomic data."""

import numpy as np
from xarray import DataArray


def atomdat_files(element: str):
    """
    For a specified element, return file parameters for reading ADAS atomic data

    Parameters
    ----------
    element
        Name of element (e.g. Ar, Ne, W, etc.)

    Returns
    -------
    file_info
        Dictionary with information to retrieve atomic ADAS type data-files

    """
    all_files = {
        "h": {"scd": "96", "acd": "96", "ccd": "96", "plt": "96", "prb": "96", "prc": "96"},
        "he": {"scd": "96", "acd": "96", "ccd": "96", "plt": "96", "prb": "96", "prc": "96"},
        "li": {"scd": "96", "acd": "96", "ccd": "89", "plt": "96", "prb": "96", "prc": "96"},
        "b": {"scd": "89", "acd": "89", "ccd": "89", "plt": "89", "prb": "89", "prc": "89"},
        "c": {
            "scd": "96",
            "acd": "96",
            "ccd": "96",
            "plt": "96",
            "prb": "96",
            "prc": "96",
            "pec": {"5": ("bnd", "96")},  # charge, file_type, year_identifier
        },
        "n": {"scd": "96", "acd": "96", "ccd": "96", "plt": "96", "prb": "96", "prc": "96"},
        "o": {
            "scd": "96",
            "acd": "96",
            "ccd": "89",
            "plt": "96",
            "prb": "96",
            "prc": "96",
            "pec": {"4": ("pju", "93")},
        },  # charge, file_type, year_identifier},
        "f": {"scd": "89", "acd": "89", "ccd": "89", "plt": "89", "prb": "89"},
        "ne": {"scd": "96", "acd": "96", "ccd": "89", "plt": "96", "prb": "96"},
        "na": {"scd": "85", "acd": "85"},
        "al": {"scd": "89", "acd": "89", "ccd": "89", "plt": "89", "prb": "89"},
        "cl": {"scd": "89", "acd": "89", "ccd": "89", "plt": "89", "prb": "89"},
        "cr": {"scd": "89", "acd": "89", "ccd": "89", "plt": "89", "prb": "89"},
        "fe": {"scd": "89", "acd": "89", "ccd": "89", "plt": "89", "prb": "89"},
        "ni": {"scd": "89", "acd": "89", "ccd": "89", "plt": "89", "prb": "89"},
        "ar": {
            "scd": "89",
            "acd": "89",
            "ccd": "89",
            "plt": "89",
            "prb": "89",
            "pec": {"16": ("llu", "transport")},
        },
        "mo": {"scd": "89", "acd": "89", "ccd": "89", "plt": "89", "prb": "89"},
    }

    key = element.strip().lower()
    if key in all_files.keys():
        return all_files[key]
    else:
        print("\n Element not implemented, returning all-files dictionary.. \n")
        return all_files


def get_atomdat(
    reader, element: str, charge: str, transition=None, wavelength=None, files=None
):
    """
    Read atomic data for specified element.

    Parameters
    ----------
    reader
        Class for reading atomic-data files
    element
        Element name (e.g. Ar, Ne, W, etc.)
    charge
        Charge state of element (not Roman numerals, e.g. "6" for fully stripped C6+)
        May also include string characters due to file path details
    transition
        Transition details for PEC data (e.g. "(1)1(1.0)-(1)0(0.0)" or "n=8-n=7")
    wavelength
        Wavelength of transition (Angstroms)
    files
        Dictionary with information to retrieve atomic ADAS type data-files
        (see atomdat_files function)

    Returns
    -------
        Dictionary with atomic data as read from ADAS files, using the same keys
        as input files

    """

    if files is None:
        files = atomdat_files(element)

    charge = charge.strip()
    atomdat = {}
    for k in files.keys():
        if k == "pec" and len(charge) > 0:
            if charge in files[k].keys():
                atomdat[k] = reader.get_adf15(
                    element,
                    charge,
                    files[k][charge][0],
                    year=files[k][charge][1],
                )
        elif k in ["scd", "acd", "ccd", "plt", "prb", "prc"]:
            atomdat[k] = reader.get_adf11(k, element, files[k])
        else:
            "\n File format not supported... \n"

    if "pec" in atomdat.keys() and transition is not None:
        atomdat["pec"] = (
            atomdat["pec"]
            .swap_dims({"index": "transition"})
            .sel(transition=transition, drop=True)
        )
        if "transition" in atomdat["pec"].dims:
            atomdat["pec"] = atomdat["pec"].swap_dims({"transition": "index"})
    if wavelength is not None:
        if len(np.unique(atomdat["pec"].coords["wavelength"].values)) > 1:
            atomdat["pec"] = (
                atomdat["pec"]
                .swap_dims({"index": "wavelength"})
                .sel(wavelength=wavelength, method="nearest", drop=True)
            )

    return files, atomdat


def fractional_abundance(
    scd,
    acd,
    ccd=[None],
    el_dens=[None],
    h_dens=[None],
    gen_type="fractional_abundance",
    element="",
) :
    """Returns the equilibrium fractional abundance given ionization, recombination
    and charge exchange rates from ADAS adf11 files. Rate variables must be 1d (ion_charges) or
    2d (ion_charges, electron_temperature)

    Still to be included:
        * provenance
        * transport or residence time
        [L. Casali EPJ 79, 01007 (2014), A. Kallenbach PPCF 55 (2013) 124041].
        Current version is not correct.

    Parameters
    ----------
    scd
        Effective ionization coefficients.
    acd
        Effective recombination coefficients.
    ccd
        Effective charge exchange recombination coefficients.
    el_dens
        electron density
    h_dens
        neutral hydrogen density
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

    nz = scd.shape[0] + 1
    nother = scd.shape[1]
    fz = np.ones((nz, nother))
    fz = DataArray(
        fz,
        coords=coords,
    )

    if el_dens[0] is None:
        el_dens = np.ones(nother)
    if h_dens[0] is None:
        h_dens = np.full(nother, 0)
    if ccd[0] is None:
        ccd = np.ones(scd.shape)

    i = 0
    fz[i:] = (
        fz[i, :] * (el_dens * acd[i, :] + h_dens * ccd[i, :]) / (el_dens * scd[i, :])
    )
    if nz > 1:
        for i in range(1, nz):
            fz[i, :] = (
                fz[i - 1, :]
                * (el_dens * scd[i - 1, :])
                / (el_dens * acd[i - 1, :] + h_dens * ccd[i - 1, :])
            )
    for j in range(nother):
        norm = np.nansum(fz[:, j], axis=0)
        fz[:, j] /= norm

    if nz > 1:
        for i in range(nz - 1, 0, -1):
            fz[i, :] = (
                fz[i - 1, :]
                * (el_dens * scd[i - 1, :])
                / (el_dens * acd[i - 1, :] + h_dens * ccd[i - 1, :])
            )
    fz[0, :] = (
        fz[0, :] * (el_dens * acd[0, :] + h_dens * ccd[0, :]) / (el_dens * scd[0, :])
    )
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
    plt,
    prb,
    fz,
    gen_type="radiated_power",
    element="",
) :
    """Returns the radiated power (W m**-3) of all ionization stages, given the
    fractional abundance, line and recombination rates read from ADAS adf11
    files. Input coefficients must be either 1d (ion_charges) or
    2d (ion_charges, electron, temperature)

    Still to be included:
        * provenance

    Parameters
    ----------
    plt
        Line power driven by excitation of dominant ions
    prb
        Continuum and line power driven by recombination and Bremsstrahlung of
        dominant ions
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

    i = 0
    rad_pow[i, :] = fz[i, :] * plt[i, :]
    if nz > 1:
        for i in range(1, nz):
            rad_pow[i, :] = fz[i, :] * (plt[i, :] + prb[i - 1, :])
    rad_pow[i + 1, :] = fz[i + 1, :] * prb[i, :]

    if gen_type and element:
        attrs = {
            "datatype": (gen_type, element),
            "provenance": "",
        }
        name = f"{element}_{gen_type}"
        rad_pow.attrs = attrs
        rad_pow.name = name

    return rad_pow
