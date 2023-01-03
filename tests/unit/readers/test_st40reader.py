import matplotlib.pylab as plt
import numpy as np

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.readers import ST40Reader

PULSE = 9229
TSTART = 0.01
TEND = 0.1

READER = ST40Reader(PULSE, TSTART, TEND)
EQUILIBRIUM_DATA = READER.get("", "efit", 0)
EQUILIBRIUM = Equilibrium(EQUILIBRIUM_DATA)
FLUX_TRANSFORM = FluxSurfaceCoordinates("poloidal")
FLUX_TRANSFORM.set_equilibrium(EQUILIBRIUM)

INSTRUMENT_INFO = {
    "xrcs": ("sxr", "xrcs", 0, set()),
    "brems": ("spectrom", "brems", -1, set()),
    "halpha": ("spectrom", "halpha", -1, set()),
    "sxr_diode_4": ("sxr", "sxr_diode_4", 0, set()),
    "sxr_camera_4": ("sxr", "sxr_camera_4", 0, set()),
    "smmh1": ("interferom", "smmh1", 0, set()),
    "nirh1": ("interferom", "nirh1", 0, set()),
    "efit": ("", "efit", 0, set()),
}

def run_reader_get_methods(
    instrument_name: str,
    mds_only=False,
):
    """
    General test script to read data from MDS+ and calculate LOS information
    including Cartesian-flux surface mapping

    TODO: currently only runs, but no assertions to check what data it returns

    Parameters
    ----------
    instrument_name
        Key from INSTRUMENT_INFO to give all the necessary inputs
    mds_only
        Returns only ST40Reader database dictionary. Otherwise returns also
        data structure crunched by the abstractreader

    Returns
    -------

    """
    if instrument_name not in INSTRUMENT_INFO:
        raise ValueError(
            f"Instrument not available. Possible choices: {list(INSTRUMENT_INFO)}"
        )

    uid, instrument, revision, quantities = INSTRUMENT_INFO[instrument_name]

    print(f"Reading {uid}, {instrument}, {revision}")
    instrument_method = READER.INSTRUMENT_METHODS[instrument]
    database_quantities = READER.available_quantities(instrument)
    if quantities:
        database_quantities = quantities

    database_results = getattr(READER, f"_{instrument_method}")(
        uid,
        instrument,
        revision,
        database_quantities,
    )

    if mds_only:
        return database_results

    data = READER.get(uid, instrument, revision, set(quantities))

    quantities = list(data)
    trans = data[quantities[0]].transform
    if hasattr(trans, "set_flux_transform"):
        trans.set_flux_transform(FLUX_TRANSFORM)
        trans._convert_to_rho(t=np.array([0.02, 0.03, 0.04]))

    return data, database_results

def check_transforms(instrument_name:str, diagnostic_data:dict):
    """
    Check transforms associated to data read

    Parameters
    ----------
    instrument_name
        instrument string identifier
    diagnostic_data
        data dictionary as returned by abstractreader
    """
    for quant, data in diagnostic_data.items():
        if hasattr(data, "transform"):
            if "LineOfSightTransform" in str(data.transform):
                if "line_of_sight" not in str(data.transform):
                    raise ValueError(
                        f"{instrument_name}:{quant} using"
                        f" \n {str(data.transform)}"
                    )

def test_xrcs(instrument_name:str = "xrcs"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)

def test_brems(instrument_name:str = "brems"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)

def test_halpha(instrument_name:str = "halpha"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)

def test_sxr_diode_4(instrument_name:str = "sxr_diode_4"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)

def test_sxr_camera_4(instrument_name:str = "sxr_camera_4"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)

def test_smmh1(instrument_name:str = "smmh1"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)

def test_nirh1(instrument_name:str = "nirh1"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)

def test_efit(instrument_name:str = "efit"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)

