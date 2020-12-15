"""Check common functionality for reading ADAS data."""

from contextlib import contextmanager
import datetime
from io import StringIO
import os.path
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock
from unittest.mock import patch

from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import characters
from hypothesis.strategies import composite
from hypothesis.strategies import from_regex
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies import times
import prov.model as prov
from xarray.testing import assert_allclose

from indica.datatypes import ORDERED_ELEMENTS
from indica.readers import ADASReader
import indica.readers.adas as adas
from indica.session import hash_vals
from ..data_strategies import ADAS_GENERAL_DATATYPES
from ..data_strategies import adf11_data

_characters = characters(
    blacklist_categories=("Cs",), blacklist_characters=["\n", "\r"]
)

pathstrings = from_regex("[a-zA-Z0-9/]+", fullmatch=True)
optional_pathstrings = from_regex("[a-zA-Z0-9/]*", fullmatch=True)
paths = one_of(optional_pathstrings, optional_pathstrings.map(Path))


@composite
def adas_readers(draw):
    return ADASReader(draw(paths), MagicMock())


@given(adas_readers())
def test_adas_needs_authentication(reader):
    assert not reader.requires_authentication


@given(adas_readers(), text(), text())
def test_authenticate_anyway(reader, username, password):
    assert reader.authenticate(username, password)


@given(paths)
def test_context_manager(path):
    with ADASReader(path, MagicMock()) as reader:
        print(reader.requires_authentication)


@given(paths, times())
def test_reader_provenance(path, creation_time):
    with patch("datetime.datetime", MagicMock()):
        datetime.datetime.now.return_value = creation_time
        reader = ADASReader(path, MagicMock())
    path = Path(path)
    openadas = path == Path("")
    if openadas:
        path = Path.home() / adas.CACHE_DIR / "adas"
    prov_id = hash_vals(path=path)
    reader.session.prov.agent.assert_called_once_with(prov_id)
    assert reader.agent is reader.session.prov.agent.return_value
    reader.session.prov.entity.assert_called_once_with(
        prov_id, {"path": str(path.resolve())}
    )
    assert reader.entity is reader.session.prov.entity.return_value
    reader.session.prov.delegation.assert_called_once_with(
        reader.session.agent, reader.agent
    )
    reader.session.prov.generation.assert_called_once_with(
        reader.entity, reader.session.session, time=creation_time
    )
    reader.session.prov.attribution.assert_called_once_with(
        reader.entity, reader.session.agent
    )


@given(adas_readers(), text(min_size=1), times(), times())
def test_data_provenance(reader, filename, starttime, endtime):
    with patch("datetime.datetime", MagicMock()):
        datetime.datetime.now.return_value = endtime
        entity = reader.create_provenance(filename, starttime)
    assert entity is reader.session.prov.entity.return_value
    file_id = f"{reader.namespace}:{filename}"
    entity_id = f"{hash_vals(filename=filename, start_time=starttime)}"
    activity_id = f"{hash_vals(agent=reader.prov_id, date=starttime)}"
    reader.session.prov.entity.assert_called_with(entity_id)
    reader.session.prov.activity.assert_called_once_with(
        activity_id, starttime, endtime, {prov.PROV_TYPE: "ReadData"}
    )
    activity = reader.session.prov.activity.return_value
    reader.session.prov.association.assert_any_call(activity, reader.agent)
    reader.session.prov.association.assert_any_call(activity, reader.session.agent)
    reader.session.prov.communication.assert_called_once_with(
        activity, reader.session.session
    )
    reader.session.prov.generation.assert_called_with(entity, activity, endtime)
    reader.session.prov.attribution.assert_any_call(entity, reader.agent)
    reader.session.prov.attribution.assert_any_call(entity, reader.session.agent)
    reader.session.prov.derivation.assert_called_once_with(entity, file_id, activity)


@contextmanager
def cachedir(*args):
    """Set up a fake cache directory for storing downloaded OpenADAS data.

    """

    old_cache = adas.CACHE_DIR
    userdir = os.path.expanduser("~")
    with TemporaryDirectory(dir=userdir) as new_cache:
        adas.CACHE_DIR = os.path.relpath(new_cache, userdir)
        try:
            yield adas.CACHE_DIR
        finally:
            adas.CACHE_DIR = old_cache


def test_cache_openadas():
    # TODO: Compile different options so can test with larger range of inputs
    adas_class = "adf11"
    adas_file = Path("scd89") / "scd12_h.dat"
    with cachedir() as cache, patch("indica.readers.adas.urlretrieve") as urlretrieve:
        reader = ADASReader("", MagicMock())
        cachepath = Path().home() / cache
        cache_file = cachepath / "adas" / "adf11" / adas_file
        urlretrieve.side_effect = lambda x, y: y.touch()
        filestream = reader._get_file(adas_class, adas_file)
        assert filestream.name == str(cache_file)
        urlretrieve.assert_called_once_with(
            f"https://open.adas.ac.uk/download/{adas_class}/{adas_file}", cache_file,
        )
        urlretrieve.reset_mock()
        filestream2 = reader._get_file(adas_class, adas_file)
        assert filestream2.name == str(cache_file)
        urlretrieve.assert_not_called()


@given(pathstrings)
# TODO: test with other bits of ADAS data
def test_localadas(path):
    adas_class = "adf11"
    adas_file = Path("scd89") / "scd12_h.dat"
    with TemporaryDirectory() as path:
        reader = ADASReader(path, MagicMock())
        filepath = Path(path) / adas_class / adas_file
        filepath.parent.mkdir(parents=True)
        filepath.touch()
        filestream = reader._get_file(adas_class, adas_file)
        assert filestream.name == str(filepath)


def adf11_array_to_str(
    data, include_metastable_indices=False, date_divider="/", indent=0
):
    """Convert a DataArray object containing ADF11 data into a string
    representing how it is stored in ADAS files. Currently only supports
    unresolved data.

    Parameters
    ----------
    data
        The ADF11 data to be converted.
    include_metastable_indices
        Whether to print the indices of the metastable states in section
        headers.
    date_divider
        What to use to separate day, month, and year in the date for the data.
    indent
        How many spaces to indent the text from the beginning of the line.

    """

    newline = " " * indent + "\n"

    def rows_of_eight_1d(flat_array):
        result = ""
        count = 0
        for element in flat_array:
            count += 1
            if count % 8 == 0:
                result += newline
            result += f"{float(element):10.6f}"
        result += newline
        return result

    def rows_of_eight(array):
        if array.ndim == 1:
            return rows_of_eight_1d(array)
        elif array.ndim == 2:
            return "".join(
                [rows_of_eight_1d(array[i, :]) for i in range(array.shape[0])]
            )
        else:
            raise ValueError("Only accepts 1d and 2d arrays.")

    element = data.attrs["datatype"][1]
    z = ORDERED_ELEMENTS.index(element)
    nd = len(data.log_electron_density)
    nt = len(data.log_electron_temperature)
    zmin = int(data.ion_charges[0]) + 1
    zmax = int(data.ion_charges[-1]) + 1
    result = (
        " " * indent
        + f"{z:4}{nd:5}{nt:5}{zmin:5}{zmax:5}     /{element.upper():15}/{{}}"
        + newline
    )
    result += "-" * 80 + newline
    result += rows_of_eight(data.log_electron_density - 6)
    result += rows_of_eight(data.log_electron_temperature)
    d = date_divider
    for charge in data.ion_charges + 1:
        if include_metastable_indices:
            result += "-" * 20 + "/ IPRT= 1  / IGRD= 1  /--------/"
        else:
            result += "-" * 51 + "/"
        result += (
            f" Z1={int(charge):<5}/ DATE= {data.attrs['date']:%d{d}%m{d}%y}" + newline
        )
        result += rows_of_eight(data.sel(ion_charges=charge) + 6)
    return result


@composite
def adas_comments(draw):
    """Returns a block of comments of the sort you might find at the
    bottom of an ADAS file.

    """
    csymbol = draw(sampled_from(["C", "c"]))
    comment_text = "\n".join(draw(lists(text(_characters))))
    return re.sub(r"(?m)^(?=\w*\W)", csymbol, comment_text)


@composite
def adf11_data_and_file(draw):
    """Creates ADF11 data and a file-like object imitating how that data
    would be saved to the disk.

    """
    data = draw(adf11_data())
    indent = draw(integers(0, 5))
    divider = draw(sampled_from(["/", ".", "-"]))
    top_comment = draw(text(_characters))
    string_data = adf11_array_to_str(data, draw(booleans()), divider, indent).format(
        top_comment
    )
    comment = draw(adas_comments())
    fileobj = StringIO(string_data + comment)
    return data, fileobj


@given(
    adas_readers(),
    adf11_data_and_file(),
    from_regex("[a-zA-Z]+", fullmatch=True),
    integers(1921, 2020),
)
def test_read_adf11(reader, data_fileobj, element, year):
    data, fileobj = data_fileobj
    reader._get_file = MagicMock(return_value=fileobj)
    reader.create_provenance = MagicMock()
    general_type = data.attrs["datatype"][0]
    for q, dt in ADAS_GENERAL_DATATYPES.items():
        if dt == general_type:
            quantity = q
            break
    else:
        raise ValueError(f"Unrecognised ADAS datatype '{general_type}'")
    adas_base = f"{quantity}{year % 100:02}"
    expected_file = Path(adas_base) / f"{adas_base}_{element}.dat"
    now = datetime.datetime.now()
    result = reader.get_adf11(quantity, element, year)
    reader._get_file.assert_called_once_with("adf11", expected_file)
    reader.create_provenance.assert_called_once()
    args, kwargs = reader.create_provenance.call_args
    assert args[0] == expected_file
    assert args[1] >= now
    assert_allclose(data, result)
    assert data.attrs["datatype"] == result.attrs["datatype"]
    assert result.attrs["provenance"] == reader.create_provenance.return_value
