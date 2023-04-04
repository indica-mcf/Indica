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

INSTRUMENT_INFO: dict = {
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

    return database_results


def test_xrcs(instrument_name: str = "xrcs"):
    _ = run_reader_get_methods(instrument_name)


def test_brems(instrument_name: str = "brems"):
    _ = run_reader_get_methods(instrument_name)


def test_halpha(instrument_name: str = "halpha"):
    _ = run_reader_get_methods(instrument_name)


def test_sxr_diode_4(instrument_name: str = "sxr_diode_4"):
    _ = run_reader_get_methods(instrument_name)


def test_sxr_camera_4(instrument_name: str = "sxr_camera_4"):
    _ = run_reader_get_methods(instrument_name)


def test_smmh1(instrument_name: str = "smmh1"):
    _ = run_reader_get_methods(instrument_name)


def test_nirh1(instrument_name: str = "nirh1"):
    _ = run_reader_get_methods(instrument_name)


def test_efit(instrument_name: str = "efit"):
    _ = run_reader_get_methods(instrument_name)
