from indica.readers import st40reader

PULSE = 40000042
TSTART = 0.01
TEND = 0.1
REVISION="J10"

INSTRUMENTS: list = [
    "transp"
]


def run_reader_get_methods(
    reader: st40reader,
    instrument: str,
):
    print(instrument)
    data = reader.get("", instrument, revision=REVISION)
    return data


def test_reader_get_methods(return_dataarrays=True, verbose=True):
    reader = st40reader(
        PULSE, TSTART, TEND, verbose=verbose, return_dataarrays=return_dataarrays
    )
    for instrument in INSTRUMENTS:
        data = run_reader_get_methods(reader, instrument)
        assert type(data) == dict
        assert len(data) > 0

test_reader_get_methods()