"""Check common functionality for reading ADAS data."""

from contextlib import contextmanager
import datetime
import os.path
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from tempfile import TemporaryFile
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
import numpy as np
from numpy.testing._private.utils import assert_raises
import prov.model as prov

from indica.datatypes import ADF11_GENERAL_DATATYPES
from indica.datatypes import ELEMENTS
from indica.readers import OpenADASReader
import indica.readers.openadas as adas
from indica.session import hash_vals
from ..data_strategies import adf11_data

_characters = characters(
    blacklist_categories=("Cs",), blacklist_characters=["\n", "\r"]
)

pathstrings = from_regex("[a-zA-Z0-9/]+", fullmatch=True)
optional_pathstrings = from_regex("[a-zA-Z0-9/]*", fullmatch=True)
paths = one_of(optional_pathstrings, optional_pathstrings.map(Path))


@composite
def adas_readers(draw):
    return OpenADASReader(draw(paths), MagicMock())


@given(adas_readers())
def test_adas_needs_authentication(reader):
    assert not reader.requires_authentication


@given(adas_readers(), text(), text())
def test_authenticate_anyway(reader, username, password):
    assert reader.authenticate(username, password)


@given(paths)
def test_context_manager(path):
    with OpenADASReader(path, MagicMock()) as reader:
        print(reader.requires_authentication)


@given(paths, times())
def test_reader_provenance(path, creation_time):
    with patch("datetime.datetime", MagicMock()):
        datetime.datetime.now.return_value = creation_time
        reader = OpenADASReader(path, MagicMock())
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
    """Set up a fake cache directory for storing downloaded OpenADAS data."""

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
        reader = OpenADASReader("", MagicMock())
        cachepath = Path().home() / cache
        cache_file = cachepath / "adas" / "adf11" / adas_file
        urlretrieve.side_effect = lambda x, y: y.touch()
        filestream = reader._get_file(adas_class, adas_file)
        assert filestream.name == str(cache_file)
        urlretrieve.assert_called_once_with(
            f"https://open.adas.ac.uk/download/{adas_class}/{adas_file}",
            cache_file,
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
        reader = OpenADASReader(path, MagicMock())
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
            result += f"{float(element):10.5f}"
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
    z = [value[0] for value in ELEMENTS.values() if value[2] == element][0]
    nd = len(data.electron_density)
    nt = len(data.electron_temperature)
    zmin = int(data.ion_charges[0]) + 1
    zmax = int(data.ion_charges[-1]) + 1
    result = (
        " " * indent
        + f"{z:4}{nd:5}{nt:5}{zmin:5}{zmax:5}     /{element.upper():15}/{{}}"
        + newline
    )
    result += "-" * 80 + newline
    result += rows_of_eight(np.log10(data.electron_density) - 6)
    result += rows_of_eight(np.log10(data.electron_temperature))
    d = date_divider
    for charge in data.ion_charges:
        if include_metastable_indices:
            result += "-" * 20 + "/ IPRT= 1  / IGRD= 1  /--------/"
        else:
            result += "-" * 51 + "/"
        result += (
            f" Z1={int(charge)+1:<5}/ DATE= {data.attrs['date']:%d{d}%m{d}%y}" + newline
        )
        result += rows_of_eight(np.log10(data.sel(ion_charges=charge)) + 6)
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
    """Creates ADF11 data and a string portraying how that data
    would be saved to the disk.

    """
    data = draw(adf11_data(max_z=10))
    indent = draw(integers(0, 5))
    divider = draw(sampled_from(["/", ".", "-"]))
    top_comment = draw(text(_characters))
    string_data = adf11_array_to_str(data, draw(booleans()), divider, indent).format(
        top_comment
    )
    comment = draw(adas_comments())
    return data, string_data + comment


@given(
    adas_readers(),
    adf11_data_and_file(),
    from_regex("[a-zA-Z]+", fullmatch=True),
    text(),
)
def test_read_adf11(reader, data_file, element, year):
    data, file_contents = data_file
    general_type = data.attrs["datatype"][0]
    for q, dt in ADF11_GENERAL_DATATYPES.items():
        if dt == general_type:
            quantity = q
            break
    else:
        raise ValueError(f"Unrecognised ADAS datatype '{general_type}'")
    adas_base = f"{quantity}{year}"
    expected_file = Path(adas_base) / f"{adas_base}_{element.lower()}.dat"
    now = datetime.datetime.now()
    reader.create_provenance = MagicMock()
    with TemporaryFile(mode="w+") as adf11_file:
        adf11_file.write(file_contents)
        adf11_file.seek(0)
        reader._get_file = MagicMock(return_value=adf11_file)
        result = reader.get_adf11(quantity, element, year)
    reader._get_file.assert_called_once_with("adf11", expected_file)
    reader.create_provenance.assert_called_once()
    args, kwargs = reader.create_provenance.call_args

    assert args[0] == expected_file
    assert args[1] >= now
    np.testing.assert_allclose(data, result, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(data.ion_charges, result.ion_charges, atol=1e-5)
    np.testing.assert_allclose(
        data.electron_temperature, result.electron_temperature, atol=1e-5, rtol=1e-4
    )
    np.testing.assert_allclose(
        data.electron_density, result.electron_density, atol=1e-5, rtol=1e-4
    )
    assert data.name == result.name
    assert data.attrs["datatype"] == result.attrs["datatype"]
    assert result.attrs["provenance"] == reader.create_provenance.return_value


def test_read_invalid_adf11():
    reader = OpenADASReader("", MagicMock())

    quantity = "scd"

    # The following check is not strictly necessary but is included in case
    # ADF11_GENERAL_DATATYPES is changed in the future to exclude "scd" for some reason.
    try:
        _ = ADF11_GENERAL_DATATYPES[quantity]
    except Exception as e:
        raise e

    with assert_raises(AssertionError):
        invalid_file_name = Path("tests/unit/readers/invalid_adf11_file.dat")
        with open(invalid_file_name, "r") as invalid_file:
            reader._get_file = MagicMock(return_value=invalid_file)
            _ = reader.get_adf11(quantity, "he", "89")
