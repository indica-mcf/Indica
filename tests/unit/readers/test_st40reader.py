from indica.readers import ST40Reader

PULSE = 11419
TSTART = 0.01
TEND = 0.1

INSTRUMENTS: list = [
    "xrcs",
    "lines",
    "smmh",
    "efit",
    "cxff_pi",
    "pi",
    "ts",
    "ppts",
    "zeff_brems",
]


def run_reader_get_methods(
    reader: ST40Reader,
    instrument: str,
):
    print(instrument)
    data = reader.get("", instrument, 0)
    return data


def test_reader_get_methods(return_dataarrays=True, verbose=True):
    reader = ST40Reader(
        PULSE, TSTART, TEND, verbose=verbose, return_dataarrays=return_dataarrays
    )
    for instrument in INSTRUMENTS:
        data = run_reader_get_methods(reader, instrument)
        assert type(data) == dict
        assert len(data) > 0
