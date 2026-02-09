from indica.readers import ST40Reader

PULSE = 11418
TSTART = 0
TEND = 20

# ENTRIES NOT SAVED TO 5-DIGIT PULSE & NOT HAVING RUN/BEST STRUCTURE
INSTRUMENTS = {
    "astra": {"pulse": 13013666, "revision": "RUN602"},
    "transp_test": {"pulse": 40013565, "revision": "X01"},
    "metis": {"pulse": 40011890, "revision": "RUN01"},
}

# ALL OTHER DATABASE ENTRIES
for instr in [
    "xrcs",
    "lines",
    "smmh",
    "efit",
    "cxff_pi",
    "pi",
    "ts",
    "ppts",
    "zeff_brems",
]:
    INSTRUMENTS[instr] = {"pulse": PULSE, "revision": 0}


def test_reader_get_methods(return_dataarrays=True, verbose=False):
    for instrument in INSTRUMENTS.keys():
        print(f"\n Reading: {instrument.upper()}")
        _pulse = INSTRUMENTS[instrument]["pulse"]
        _revision = INSTRUMENTS[instrument]["revision"]
        _tree = instrument
        reader = ST40Reader(
            _pulse,
            TSTART,
            TEND,
            tree=_tree,
            verbose=verbose,
            return_dataarrays=return_dataarrays,
        )
        data = reader.get("", instrument, _revision)
        assert type(data) == dict
        assert len(data) > 0

    return data


if __name__ == "__main__":
    test_reader_get_methods()
