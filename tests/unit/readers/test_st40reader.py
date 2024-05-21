from indica.readers import ST40Reader

PULSE = 10605
TSTART = 0.01
TEND = 0.1

READER = ST40Reader(PULSE, TSTART, TEND)
INSTRUMENT_INFO: dict = {
    "xrcs": ("sxr", "xrcs", 0, set()),
    "brems": ("spectrom", "brems", -1, set()),
    "halpha": ("spectrom", "halpha", -1, set()),
    "sxr_diode_4": ("sxr", "sxr_diode_4", 0, set()),
    "sxr_camera_4": ("sxr", "sxr_camera_4", 0, set()),
    "smmh1": ("interferom", "smmh1", 0, set()),
    "nirh1": ("interferom", "nirh1", 0, set()),
    "efit": ("", "efit", 0, set()),
    "cxff_pi": ("", "cxff_pi", 0, set()),
    "cxff_tws_c": ("", "cxff_tws_c", 0, set()),
    "pi": ("", "pi", 0, set()),
    "tws_c": ("", "tws_c", 0, set()),
    "ts": ("", "ts", 0, set()),
    "ppts": ("", "ppts", 0, set()),
}


def run_reader_get_methods(
    instrument_name: str,
    mds_only=False,
):
    """
    General test script to read data from MDS+ and calculate LOS information
    including Cartesian-flux surface mapping

    TODO: Not testing MDS+ reading as tests currently using mock reader!!!

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

    return data, database_results


def check_transforms(instrument_name: str, diagnostic_data: dict):
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
                        f"{instrument_name}:{quant} using" f" \n {str(data.transform)}"
                    )


def test_xrcs(instrument_name: str = "xrcs"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_brems(instrument_name: str = "brems"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_halpha(instrument_name: str = "halpha"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_sxr_diode_4(instrument_name: str = "sxr_diode_4"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_sxr_camera_4(instrument_name: str = "sxr_camera_4"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_smmh1(instrument_name: str = "smmh1"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_nirh1(instrument_name: str = "nirh1"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_efit(instrument_name: str = "efit"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_cxff_pi(instrument_name: str = "cxff_pi"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_cxff_tws_c(instrument_name: str = "cxff_tws_c"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_pi(instrument_name: str = "pi"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_tws_c(instrument_name: str = "tws_c"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)


def test_ts(instrument_name: str = "ts"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)

def test_ppts(instrument_name: str = "ppts"):
    data, database_results = run_reader_get_methods(instrument_name)
    check_transforms(instrument_name, data)