from datetime import datetime
from typing import Tuple
from typing import Union
from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import builds
from hypothesis.strategies import composite
from hypothesis.strategies import dictionaries
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import sets
from hypothesis.strategies import shared
from hypothesis.strategies import text
import prov.model as prov
from pytest import mark
from pytest import raises
from xarray import DataArray
from xarray import Dataset

from indica.datatypes import DataType
from indica.operators import Operator
from indica.operators import OperatorError
from ..converters.test_abstract_transform import coordinate_transforms_and_axes
from ..data_strategies import array_datatypes
from ..data_strategies import dataset_datatypes
from ..data_strategies import general_datatypes
from ..data_strategies import incompatible_array_types
from ..data_strategies import incompatible_dataset_types

small_coords = coordinate_transforms_and_axes(
    ((1.83, 3.9), (-1.75, 2.0), (50.0, 120.0)),
    2,
    2,
)


@composite
def small_data_arrays_from_type(draw, datatype):
    return draw(
        builds(
            DataArray,
            just(0.0),
            attrs=fixed_dictionaries(
                {"datatype": just(datatype)},
                optional={"provenance": builds(MagicMock())},
            ),
        )
    )


small_data_arrays = array_datatypes(allow_none=False).flatmap(
    small_data_arrays_from_type
)


@composite
def small_datasets_from_type(draw, datatype, allow_extra=False):
    contents = {
        key: DataArray(0.0, attrs={"datatype": (general_type, datatype[0])})
        for key, general_type in datatype[1].items()
    }
    if allow_extra:
        extras = draw(
            dictionaries(
                text().filter(lambda x: x not in contents),
                general_datatypes(datatype[0]).flatmap(
                    lambda x: small_data_arrays_from_type((x, datatype[0]))
                ),
                max_size=5,
            )
        )
        contents.update(extras)
        datatype = datatype[0], {
            **datatype[1],
            **{k: data.attrs["datatype"][0] for k, data in extras.items()},
        }
    attrs = {"datatype": datatype}
    if draw(booleans()):
        attrs["provenance"] = MagicMock()
    return Dataset(contents, attrs=attrs)


small_datasets = dataset_datatypes(min_size=1, allow_none=False).flatmap(
    small_datasets_from_type
)
data = one_of(small_data_arrays, small_datasets)
arguments = lists(data, min_size=0, max_size=3)
one_or_more_arguments = lists(data, min_size=1, max_size=3)


@composite
def compatible_datatypes(draw, arg):
    """Returns a datatype which this argument would be compatible with."""
    if isinstance(arg, DataArray):
        datatype = arg.attrs["datatype"]
        return draw(sampled_from([datatype[0], None])), draw(
            sampled_from([datatype[1], None])
        )
    elif isinstance(arg, Dataset):
        datatype = arg.attrs["datatype"]
        components = draw(
            sets(sampled_from(sorted(datatype[1].keys()))).map(
                lambda keys: {k: datatype[1][k] for k in keys}
            )
        )
        return draw(sampled_from([datatype[0], None])), components
    else:
        raise ValueError("Argument must be a DataArray or Dataset")


@composite
def variadic_arguments(draw, args):
    """Creates additional arguments of the same type as the final argument
    This can be used to test operators taking variadic arguments."""
    if isinstance(args[-1], DataArray):
        return args + draw(
            lists(
                small_data_arrays.map(
                    lambda x: x.assign_attrs(datatype=args[-1].attrs["datatype"])
                ),
                max_size=10,
            )
        )
    elif isinstance(args[-1], Dataset):
        return args + draw(
            lists(
                small_datasets_from_type(args[-1].attrs["datatype"], allow_extra=True),
                max_size=10,
            )
        )
    else:
        raise ValueError("Arguments must be data arrays or datasets.")


@composite
def incompatible_datatypes(draw, arg):
    if isinstance(arg, DataArray):
        return draw(incompatible_array_types(arg.attrs["datatype"]))
    elif isinstance(arg, Dataset):
        return draw(incompatible_dataset_types(arg.attrs["datatype"]))
    else:
        raise ValueError("Argument must be a DataArray or Dataset")


@composite
def compatible_datatype_lists(draw, args):
    """Returns a list of datatypes which the list of arguments would be
    compatible with."""
    return [draw(compatible_datatypes(arg)) for arg in args]


@composite
def compatible_variadic_datatype_lists(draw, args):
    """Returns a list of datatypes which the list of arguments would be
    compatible with, but with the final element being an Ellipsis object."""
    return draw(compatible_datatype_lists(args)) + [Ellipsis]


class FakeOperator(Operator):
    """Concrete version of the Operator class. It doesn't actually
    implement an operator but will allow it to be instantiated."""

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        """Indicates the datatypes of the results when calling the operator
        with arguments of the given types.
        """
        raise NotImplementedError(
            "{} does not implement a "
            "'return_types' method.".format(self.__class__.__name__)
        )

    def __call__(self, *args: DataArray) -> Union[DataArray, Dataset]:
        """The invocation of the operator."""
        raise NotImplementedError(
            "{} does not implement a "
            "'__call__' method.".format(self.__class__.__name__)
        )


@composite
def compatible_operators(draw, args):
    operator = FakeOperator(MagicMock())
    operator.ARGUMENT_TYPES = draw(compatible_datatype_lists(args))
    return operator


@composite
def compatible_variadic_operators(draw, args):
    operator = draw(compatible_operators(args))
    operator.ARGUMENT_TYPES.append(Ellipsis)
    return operator


@given(
    shared(arguments, key="args"),
    shared(arguments, key="args").flatmap(compatible_operators),
)
def test_compatible_datatypes(args, operator):
    before = datetime.now()
    operator.validate_arguments(*args)
    after = datetime.now()
    assert before < operator._start_time < after
    for arg, input_prov in zip(
        filter(lambda x: "provenance" in x.attrs, args), operator._input_provenance
    ):
        assert arg.attrs["provenance"] is input_prov


@given(
    shared(one_or_more_arguments, key="args").flatmap(variadic_arguments),
    shared(one_or_more_arguments, key="args").flatmap(compatible_variadic_operators),
)
def test_compatible_variadic_datatypes(args, operator):
    operator.validate_arguments(*args)


@given(
    shared(data, key="args"), shared(data, key="args").flatmap(incompatible_datatypes)
)
def test_incompatible_datatypes(arg, bad_type):
    operator = FakeOperator(MagicMock())
    operator.ARGUMENT_TYPES = [bad_type]
    with raises(OperatorError):
        operator.validate_arguments(arg)


@given(
    shared(one_or_more_arguments, key="args"),
    shared(one_or_more_arguments, key="args").flatmap(compatible_operators),
)
def test_too_many_args(args, operator):
    operator.ARGUMENT_TYPES.pop()
    with raises(OperatorError):
        operator.validate_arguments(*args)


@given(
    shared(one_or_more_arguments, key="args").map(lambda x: x[:-1]),
    shared(one_or_more_arguments, key="args").flatmap(compatible_operators),
)
def test_too_few_args(args, operator):
    with raises(OperatorError):
        operator.validate_arguments(*args)


@composite
def compatible_generic_datatypes(draw, data):

    if isinstance(data, DataArray):
        pass
    elif isinstance(data, Dataset):
        pass
    else:
        raise ValueError("Expected DataArray or Dataset")


def array_with_type(general, specific):
    return DataArray(0.0, attrs={"datatype": (general, specific)})


def dataset_with_type(specific, general):
    return Dataset(attrs={"datatype": (specific, general)})


@mark.parametrize(
    "arg_type, arg1, arg2",
    [
        (
            (None, "beryllium"),
            array_with_type("temperature", "beryllium"),
            array_with_type("concentration", "beryllium"),
        ),
        (
            ("temperature", None),
            array_with_type("temperature", "lithium"),
            array_with_type("temperature", "beryllium"),
        ),
        (
            (None, None),
            array_with_type("temperature", "lithium"),
            array_with_type("concentration", "beryllium"),
        ),
        (
            (None, {"T": "temperature", "N": "number_density"}),
            dataset_with_type("beryllium", {"T": "temperature", "N": "number_density"}),
            dataset_with_type("lithium", {"T": "temperature", "N": "number_density"}),
        ),
    ],
)
def test_mismatched_variadic_args(arg_type, arg1, arg2):
    operator = FakeOperator(MagicMock())
    operator.ARGUMENT_TYPES = [arg_type, Ellipsis]
    with raises(OperatorError):
        operator.validate_arguments(arg1, arg2)


@given(
    shared(arguments, key="args"),
    shared(arguments, key="args").flatmap(compatible_operators),
    small_data_arrays,
)
def test_assign_array_provenance(args, operator, array):
    doc = prov.ProvDocument()
    doc.set_default_namespace("https://ccfe.ukaea.uk/")
    session = MagicMock(
        prov=doc,
        agent=doc.agent("session_agent"),
        session=doc.activity("session_activity"),
    )
    operator._session = session
    # before = datetime.now()
    operator.validate_arguments(*args)
    # middle = datetime.now()
    operator.assign_provenance(array)
    # after = datetime.now()
    # Check generated by activity with correct start/end times
    # Check activity associated with correct agents
    # Check attributed to correct agents
    # Check activity used args
    # Check result derived from args


@given(
    shared(arguments, key="args"),
    shared(arguments, key="args").flatmap(compatible_operators),
    small_datasets,
)
def test_assign_dataset_provenance(args, operator, dataset):
    doc = prov.ProvDocument()
    doc.set_default_namespace("https://ccfe.ukaea.uk/")
    session = MagicMock(
        prov=doc,
        agent=doc.agent("session_agent"),
        session=doc.activity("session_activity"),
    )
    operator._session = session
    # before = datetime.now()
    operator.validate_arguments(*args)
    # middle = datetime.now()
    operator.assign_provenance(dataset)
    # after = datetime.now()
    # Check generated by activity with correct start/end times
    # Check activity associated with correct agents
    # Check attributed to correct agents
    # Check activity used args
    # Check result derived from args
    # Check provenance generated for contents which don't already have it
    # Check provenance of dataset is collection of provenance of contents


@given(
    shared(arguments, key="args"),
    shared(arguments, key="args").flatmap(compatible_operators),
    data,
    data,
)
def test_assign_multiple_provenance(args, operator, result1, result2):
    # Check successive entities are distinct and have different hashes
    # Check they were generated by the same activity entity
    assert False
