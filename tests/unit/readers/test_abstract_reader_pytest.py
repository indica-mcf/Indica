"""Test methods present on the base class DataReader."""

from numbers import Number
from typing import Collection
from typing import Iterable
from typing import Set

import numpy as np
from xarray import DataArray

from indica.numpy_typing import RevisionLike
from indica.readers import PPFReader
from indica.readers.available_quantities import AVAILABLE_QUANTITIES

# from .st40reader import ST40Reader


# TODO these values should come from the machine dimensions variable of the reader

TSTART = 0
TEND = 10


def _test_get_methods(
    uid="jetppf",
    instrument="hrts",
    method="thomson_scattering",
    nsamples=10,
    machine: str = "JET",
):
    """Test the get_thomson_scattering method correctly combines and processes
    raw data."""

    def selector(
        data: DataArray,
        channel_dim: str,
        bad_channels: Collection[Number],
        unselected_channels: Iterable[Number] = [],
    ):
        return bad_channels

    def _get(
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        empty=False,
    ):
        """Strategy to produce a dictionary of DataArrays of the type that
        could be returned by a read operation.
        """

        if empty:
            return {}
        else:
            return database_results

    get_method = f"get_{method}"
    _get_method = f"_get_{method}"

    quantities = set(AVAILABLE_QUANTITIES[get_method])
    for i in range(nsamples):
        # Initialize reader
        if machine.upper() == "JET":
            reader = PPFReader(
                0,
                TSTART,
                TEND,
                selector=selector,
            )
        # elif machine.upper() == "ST40":
        #     reader = ST40Reader(
        #         0,
        #         database_results["times"].min(),
        #         database_results["times"].max(),
        #         open_conn=False,
        #     )
        else:
            print(f"Machine {machine} still to be  implemented in testing")
            raise ValueError

        # Generate fake results from database
        database_results = globals()[f"set_{method}"](reader, quantities)

        # Bypass reader _get method to test abstract reader only
        setattr(reader, _get_method, _get)

        # Run abstract_reader get method for desired diagnostic
        results = getattr(reader, get_method)(
            uid, instrument, database_results["revision"], quantities
        )

        # Check whether data is as expected
        for q, actual, expected in [
            (q, results[q], database_results[q]) for q in quantities
        ]:
            assert np.all(actual.values == expected)


def set_thomson_scattering(reader, quantities: Set):

    Rmin, Rmax = reader.MACHINE_DIMS[0][0], reader.MACHINE_DIMS[0][1]
    zmin, zmax = reader.MACHINE_DIMS[1][0], reader.MACHINE_DIMS[1][1]

    database_results = {}
    database_results["dt"] = np.random.uniform(0.001, 1.0)
    database_results["times"] = np.arange(TSTART, TEND, database_results["dt"])
    nt = len(database_results["times"])
    database_results["revision"] = np.random.randint(0, 10)
    database_results["length"] = np.random.randint(4, 20)
    database_results["z"] = np.random.uniform(zmin, zmax, (database_results["length"],))
    database_results["R"] = np.random.uniform(Rmin, Rmax, (database_results["length"],))
    database_results["te"] = np.random.uniform(
        10, 10.0e3, (nt, database_results["length"])
    )
    database_results["ne"] = np.random.uniform(
        1.0e16, 1.0e21, (nt, database_results["length"])
    )
    database_results["te_error"] = np.sqrt(database_results["te"])
    database_results["ne_error"] = np.sqrt(database_results["ne"])
    database_results["bad_channels"] = []

    for quantity in quantities:
        database_results[f"{quantity}_records"] = [
            f"{quantity}_Rz_path",
            f"{quantity}_value_path",
            f"{quantity}_error_path",
        ]

    return database_results


def set_charge_exchange(reader, quantities: Set):
    Rmin, Rmax = reader.MACHINE_DIMS[0][0], reader.MACHINE_DIMS[0][1]
    zmin, zmax = reader.MACHINE_DIMS[1][0], reader.MACHINE_DIMS[1][1]

    database_results = {}
    database_results["length"] = np.random.randint(4, 20)
    database_results["dt"] = np.random.uniform(0.001, 1.0)
    database_results["times"] = np.arange(TSTART, TEND, database_results["dt"])
    database_results["texp"] = np.full_like(
        database_results["times"], database_results["dt"]
    )
    nt = len(database_results["times"])
    database_results["element"] = "element"
    database_results["revision"] = np.random.randint(0, 10)
    database_results["R"] = np.random.uniform(Rmin, Rmax, (database_results["length"],))
    database_results["z"] = np.random.uniform(zmin, zmax, (database_results["length"],))
    database_results["ti"] = np.random.uniform(
        10, 10.0e3, (nt, database_results["length"])
    )
    database_results["ti_error"] = np.sqrt(database_results["ti"])
    database_results["angf"] = np.random.uniform(
        1.0e2, 1.0e6, (nt, database_results["length"])
    )
    database_results["angf_error"] = np.sqrt(database_results["angf"])
    database_results["conc"] = np.random.uniform(
        1.0e-6, 1.0e-1, (nt, database_results["length"])
    )
    database_results["conc_error"] = np.sqrt(database_results["conc"])
    database_results["bad_channels"] = []

    for quantity in quantities:
        database_results[f"{quantity}_records"] = [
            f"{quantity}_R_path",
            f"{quantity}_z_path",
            f"{quantity}_element_path",
            f"{quantity}_time_path",
            f"{quantity}_value_path",
            f"{quantity}_error_path",
        ]

    return database_results


def set_cyclotron_emissions(reader, quantities: Set):
    zmin, zmax = reader.MACHINE_DIMS[1][0], reader.MACHINE_DIMS[1][1]

    database_results = {}

    database_results["machine_dims"] = reader.MACHINE_DIMS
    database_results["length"] = np.random.randint(4, 20)
    database_results["dt"] = np.random.uniform(0.001, 1.0)
    database_results["times"] = np.arange(TSTART, TEND, database_results["dt"])
    nt = len(database_results["times"])

    database_results["revision"] = np.random.randint(0, 10)
    database_results["z"] = np.random.uniform(zmin, zmax)

    database_results["te"] = np.random.uniform(
        10, 10.0e3, (nt, database_results["length"])
    )
    database_results["te_error"] = np.sqrt(database_results["te"])
    database_results["Btot"] = np.random.uniform(0.1, 5, (database_results["length"]))
    database_results["bad_channels"] = []

    for quantity in quantities:
        database_results[f"{quantity}_records"] = []
        for i in range(database_results["length"]):
            database_results[f"{quantity}_records"].append(f"chan_{i}_path")

    return database_results


def set_equilibrium(reader, quantities: Set):
    Rmin, Rmax = reader.MACHINE_DIMS[0][0], reader.MACHINE_DIMS[0][1]
    zmin, zmax = reader.MACHINE_DIMS[1][0], reader.MACHINE_DIMS[1][1]

    database_results = {}

    database_results["dt"] = np.random.uniform(0.001, 1.0)
    database_results["times"] = np.arange(TSTART, TEND, database_results["dt"])
    nt = len(database_results["times"])
    nrho = np.random.randint(20, 40)

    database_results["element"] = "element"

    database_results["R"] = np.random.uniform(Rmin, Rmax, (nrho,))
    database_results["z"] = np.random.uniform(zmin, zmax, (nrho,))

    database_results["rgeo"] = np.random.uniform(Rmin, Rmax, (nt,))
    database_results["rmag"] = np.random.uniform(Rmin, Rmax, (nt,))
    database_results["zmag"] = np.random.uniform(zmin, zmax, (nt,))
    database_results["ipla"] = np.random.uniform(1.0e4, 1.0e6, (nt,))
    database_results["wp"] = np.random.uniform(1.0e3, 1.0e5, (nt,))
    database_results["df"] = np.random.uniform(0, 1, (nt,))
    database_results["faxs"] = np.random.uniform(1.0e-6, 0.1, (nt,))
    database_results["fbnd"] = np.random.uniform(-1, 1, (nt,))

    database_results["psin"] = np.random.uniform(0, 1, (nrho,))
    database_results["psi_r"] = np.random.uniform(Rmin, Rmax, (nrho,))
    database_results["psi_z"] = np.random.uniform(zmin, zmax, (nrho,))

    database_results["f"] = np.random.uniform(1.0e-6, 0.1, (nt, nrho))
    database_results["ftor"] = np.random.uniform(1.0e-4, 1.0e-2, (nt, nrho))
    database_results["vjac"] = np.random.uniform(1.0e-3, 2.0, (nt, nrho))
    database_results["rmji"] = np.random.uniform(Rmin, Rmax, (nt, nrho))
    database_results["rmjo"] = np.random.uniform(Rmin, Rmax, (nt, nrho))
    database_results["rbnd"] = np.random.uniform(Rmin, Rmax, (nt, nrho))
    database_results["zbnd"] = np.random.uniform(zmin, zmax, (nt, nrho))
    database_results["psi"] = np.random.uniform(-1, 1, (nt, nrho, nrho))

    database_results["revision"] = np.random.randint(0, 10)

    for quantity in quantities:
        database_results[f"{quantity}_records"] = [f"{quantity}_records"]
    database_results["psi_records"] = ["value_records", "R_records", "z_records"]

    return database_results


def set_radiation(reader, quantities: Set):
    _, Rmax = reader.MACHINE_DIMS[0][0], reader.MACHINE_DIMS[0][1]
    zmin, zmax = reader.MACHINE_DIMS[1][0], reader.MACHINE_DIMS[1][1]

    database_results = {}

    database_results["revision"] = np.random.randint(0, 10)
    database_results["machine_dims"] = reader.MACHINE_DIMS
    length = np.random.randint(4, 20)
    database_results["dt"] = np.random.uniform(0.001, 1.0)
    database_results["times"] = np.arange(TSTART, TEND, database_results["dt"])
    nt = len(database_results["times"])

    database_results["v_times"] = database_results["times"]
    database_results["v"] = np.random.uniform(0, 1.0e6, (nt, length))
    database_results["v_error"] = np.sqrt(database_results["v"])
    database_results["v_xstart"] = np.random.uniform(-Rmax, Rmax, (length,))
    database_results["v_ystart"] = np.random.uniform(-Rmax, Rmax, (length,))
    database_results["v_zstart"] = np.random.uniform(zmin, zmax, (length,))
    database_results["v_xstop"] = np.random.uniform(-Rmax, Rmax, (length,))
    database_results["v_ystop"] = np.random.uniform(-Rmax, Rmax, (length,))
    database_results["v_zstop"] = np.random.uniform(zmin, zmax, (length,))

    database_results["h_times"] = database_results["times"]
    database_results["h"] = np.random.uniform(0, 1.0e6, (nt, length))
    database_results["h_error"] = np.sqrt(database_results["h"])
    database_results["h_xstart"] = np.random.uniform(-Rmax, Rmax, (length,))
    database_results["h_ystart"] = np.random.uniform(-Rmax, Rmax, (length,))
    database_results["h_zstart"] = np.random.uniform(zmin, zmax, (length,))
    database_results["h_xstop"] = np.random.uniform(-Rmax, Rmax, (length,))
    database_results["h_ystop"] = np.random.uniform(-Rmax, Rmax, (length,))
    database_results["h_zstop"] = np.random.uniform(zmin, zmax, (length,))

    database_results["length"] = {}
    database_results["length"]["v"] = length
    database_results["length"]["h"] = length

    for quantity in quantities:
        database_results[f"{quantity}_records"] = [f"{quantity}_records"] * length

    return database_results


def set_bremsstrahlung_spectroscopy(reader, quantities: Set):
    Rmin, Rmax = reader.MACHINE_DIMS[0][0], reader.MACHINE_DIMS[0][1]
    zmin, zmax = reader.MACHINE_DIMS[1][0], reader.MACHINE_DIMS[1][1]

    database_results = {}

    database_results["revision"] = np.random.randint(0, 10)
    database_results["machine_dims"] = reader.MACHINE_DIMS
    database_results["dt"] = np.random.uniform(0.001, 1.0)
    database_results["times"] = np.arange(TSTART, TEND, database_results["dt"])
    nt = len(database_results["times"])

    length = 1
    signals = ["zefv", "zefh"]
    database_results["length"] = {}
    for k in signals:
        database_results[k] = np.random.uniform(0, 1.0e6, (nt,))
        database_results[f"{k}_error"] = np.sqrt(database_results[k])
        database_results[f"{k}_xstart"] = np.array([np.random.uniform(Rmin, Rmax)])
        database_results[f"{k}_ystart"] = np.array([np.random.uniform(Rmin, Rmax)])
        database_results[f"{k}_zstart"] = np.array([np.random.uniform(zmin, zmax)])
        database_results[f"{k}_xstop"] = np.array([np.random.uniform(Rmin, Rmax)])
        database_results[f"{k}_ystop"] = np.array([np.random.uniform(Rmin, Rmax)])
        database_results[f"{k}_zstop"] = np.array([np.random.uniform(zmin, zmax)])
        database_results["length"][k] = length

    for quantity in quantities:
        database_results[f"{quantity}_records"] = [
            f"{quantity}_path_records",
            f"{quantity}_los_records",
        ]

    return database_results


def set_helike_spectroscopy(reader, quantities: Set):
    Rmin, Rmax = reader.MACHINE_DIMS[0][0], reader.MACHINE_DIMS[0][1]
    zmin, zmax = reader.MACHINE_DIMS[1][0], reader.MACHINE_DIMS[1][1]

    database_results = {}

    nwavelength = np.random.randint(256, 1024)
    wavelength_start, wavelength_end = 3.8, 4.0

    database_results["revision"] = np.random.randint(0, 10)
    database_results["machine_dims"] = reader.MACHINE_DIMS
    database_results["dt"] = np.random.uniform(0.001, 1.0)
    database_results["times"] = np.arange(TSTART, TEND, database_results["dt"])
    nt = len(database_results["times"])

    database_results["wavelength"] = np.linspace(
        wavelength_start, wavelength_end, nwavelength
    )

    database_results["length"] = 1
    database_results["xstart"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["ystart"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["zstart"] = np.array([np.random.uniform(zmin, zmax)])
    database_results["xstop"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["ystop"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["zstop"] = np.array([np.random.uniform(zmin, zmax)])
    for quantity in quantities:
        if quantity == "spectra":
            database_results[quantity] = np.random.uniform(0, 1.0e6, (nt, nwavelength))
        else:
            database_results[quantity] = np.random.uniform(0, 1.0e4, (nt,))
        database_results[f"{quantity}_error"] = np.sqrt(database_results[quantity])

        database_results[f"{quantity}_records"] = [
            f"{quantity}_path_records",
        ]

    return database_results


def set_filters(reader, quantities: Set):
    Rmin, Rmax = reader.MACHINE_DIMS[0][0], reader.MACHINE_DIMS[0][1]
    zmin, zmax = reader.MACHINE_DIMS[1][0], reader.MACHINE_DIMS[1][1]

    database_results = {}

    database_results["revision"] = np.random.randint(0, 10)
    database_results["machine_dims"] = reader.MACHINE_DIMS
    database_results["dt"] = np.random.uniform(0.001, 1.0)
    database_results["times"] = np.arange(TSTART, TEND, database_results["dt"])
    nt = len(database_results["times"])

    database_results["length"] = 1
    database_results["xstart"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["ystart"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["zstart"] = np.array([np.random.uniform(zmin, zmax)])
    database_results["xstop"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["ystop"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["zstop"] = np.array([np.random.uniform(zmin, zmax)])
    for quantity in quantities:
        database_results[quantity] = np.random.uniform(0, 1.0e6, (nt,))
        database_results[f"{quantity}_error"] = np.sqrt(database_results[quantity])

        database_results[f"{quantity}_records"] = [
            f"{quantity}_path_records",
        ]
        database_results[f"{quantity}_error_records"] = [
            f"{quantity}_error_path_records",
        ]

    return database_results


def set_interferometry(reader, quantities: Set):
    Rmin, Rmax = reader.MACHINE_DIMS[0][0], reader.MACHINE_DIMS[0][1]
    zmin, zmax = reader.MACHINE_DIMS[1][0], reader.MACHINE_DIMS[1][1]

    database_results = {}

    database_results["revision"] = np.random.randint(0, 10)
    database_results["machine_dims"] = reader.MACHINE_DIMS
    database_results["dt"] = np.random.uniform(0.001, 1.0)
    database_results["times"] = np.arange(TSTART, TEND, database_results["dt"])
    nt = len(database_results["times"])

    database_results["length"] = 1
    database_results["xstart"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["ystart"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["zstart"] = np.array([np.random.uniform(zmin, zmax)])
    database_results["xstop"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["ystop"] = np.array([np.random.uniform(Rmin, Rmax)])
    database_results["zstop"] = np.array([np.random.uniform(zmin, zmax)])
    for quantity in quantities:
        database_results[quantity] = np.random.uniform(0, 1.0e6, (nt,))
        database_results[f"{quantity}_error"] = np.sqrt(database_results[quantity])

        database_results[f"{quantity}_records"] = [
            f"{quantity}_path_records",
        ]
        database_results[f"{quantity}_error_records"] = [
            f"{quantity}_error_path_records",
        ]

    return database_results


def test_get_thomson_scattering():
    _test_get_methods(
        uid="jetppf", instrument="hrts", method="thomson_scattering", nsamples=10
    )


def test_get_charge_exchange():
    _test_get_methods(
        uid="jetppf", instrument="cxg6", method="charge_exchange", nsamples=10
    )


def test_get_cyclotron_emissions():
    _test_get_methods(
        uid="jetppf", instrument="kk3", method="cyclotron_emissions", nsamples=10
    )


def test_get_equilibrium():
    _test_get_methods(
        uid="jetppf", instrument="efit", method="equilibrium", nsamples=10
    )


def test_get_radiation():
    _test_get_methods(uid="jetppf", instrument="sxr", method="radiation", nsamples=10)


def test_get_bremsstrahlung_spectroscopy():
    _test_get_methods(
        uid="jetppf",
        instrument="ks3",
        method="bremsstrahlung_spectroscopy",
        nsamples=10,
    )


#
# def test_get_helike_spectroscopy():
#     _test_get_methods(
#         uid="spectrom",
#         instrument="xrcs",
#         method="helike_spectroscopy",
#         nsamples=10,
#         machine="ST40",
#     )
#
#
# def test_get_filters():
#     _test_get_methods(
#         uid="", instrument="lines", method="filters",
#         nsamples=10, machine="ST40"
#     )
#
#
# def test_get_interferometry():
#     _test_get_methods(
#         uid="", instrument="nirh1", method="interferometry",
#         nsamples=10, machine="ST40"
#     )
