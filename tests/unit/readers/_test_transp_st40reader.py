from indica.readers import ST40Reader

PULSE = 40000042
TSTART = 0.01
TEND = 0.1
REVISION = "J10"
TREE = "TRANSP_TEST"

INSTRUMENTS: list = ["transp"]


def run_reader_get_methods(
    reader: ST40Reader,
    instrument: str,
):
    print(instrument)
    data = reader.get("", instrument, revision=REVISION)


def test_reader_get_methods(return_dataarrays=True, verbose=True):
    reader = ST40Reader(
        PULSE,
        TSTART,
        TEND,
        verbose=verbose,
        return_dataarrays=return_dataarrays,
        tree=TREE,
    )
    for instrument in INSTRUMENTS:
        data = run_reader_get_methods(reader, instrument)
        assert type(data) == dict
        assert len(data) > 0


test_reader_get_methods()
