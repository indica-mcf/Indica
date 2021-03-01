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
from hypothesis.strategies import from_regex
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import sets
from hypothesis.strategies import shared
import prov.model as prov
from pytest import mark
from pytest import raises
from xarray import DataArray
from xarray import Dataset

from indica.datatypes import DataType
from indica.operators import Operator
from indica.operators import OperatorError
from ..converters.test_trivial import trivial_transforms
from ..data_strategies import array_datatypes
from ..data_strategies import dataset_datatypes
from ..data_strategies import general_datatypes
from ..data_strategies import incompatible_array_types
from ..data_strategies import incompatible_dataset_types


@composite
def small_data_arrays_from_type(draw, datatype):
    return draw(
        builds(
            DataArray,
            just(0.0),
            attrs=fixed_dictionaries(
                {
                    "datatype": just(datatype),
                    "transform": trivial_transforms(),
                },
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
        key: draw(small_data_arrays_from_type((general_type, datatype[0])))
        for key, general_type in datatype[1].items()
    }
    if allow_extra:
        extras = draw(
            dictionaries(
                from_regex("[a-z0-9]+", fullmatch=True).filter(
                    lambda x: x not in contents
                ),
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
                max_size=3,
            )
        )
    elif isinstance(args[-1], Dataset):
        return args + draw(
            lists(
                small_datasets_from_type(args[-1].attrs["datatype"], allow_extra=True),
                max_size=3,
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
    doc = prov.ProvDocument()
    doc.set_default_namespace("https://ccfe.ukaea.uk/")
    session = MagicMock(
        prov=doc,
        agent=doc.agent("session_agent"),
        session=doc.activity("session_activity"),
    )
    operator = FakeOperator(session)
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
    session = operator._session
    doc = session.prov
    for i, arg in enumerate(args):
        if "provenance" in arg.attrs:
            arg.attrs["provenance"] = doc.entity(f"arg{i}")
    operator.validate_arguments(*args)
    before = datetime.now()
    operator.assign_provenance(array)
    after = datetime.now()
    entity = array.attrs["partial_provenance"]
    assert entity.get_attribute(prov.PROV_TYPE) == {"DataArray"}
    assert entity.get_attribute(prov.PROV_VALUE) == {",".join(array.attrs["datatype"])}
    generated_candidates = list(
        filter(
            lambda x: {entity.identifier} == x.get_attribute("prov:entity"),
            doc.get_records(prov.ProvGeneration),
        )
    )
    assert len(generated_candidates) == 1
    generated = generated_candidates[0]
    assert {entity.identifier} == generated.get_attribute("prov:entity")
    activity_id = next(iter(generated.get_attribute("prov:activity")))
    end_time = next(iter(generated.get_attribute("prov:time")))
    assert before < end_time < after
    comms = list(doc.get_records(prov.ProvCommunication))
    assert len(comms) == 1
    informed = comms[0]
    assert {activity_id} == informed.get_attribute("prov:informed")
    assert {session.session.identifier} == informed.get_attribute("prov:informant")
    expected_agents = [session.agent.identifier, operator.agent.identifier]
    for a in doc.get_records(prov.ProvAssociation):
        assert {activity_id} == a.get_attribute("prov:activity")
        agent_id = next(iter(a.get_attribute("prov:agent")))
        assert agent_id in expected_agents
        expected_agents.remove(agent_id)
    assert len(expected_agents) == 0
    expected_agents = [session.agent.identifier, operator.agent.identifier]
    for a in filter(
        lambda x: {entity.identifier} == x.get_attribute("prov:entity"),
        doc.get_records(prov.ProvAttribution),
    ):
        agent_id = next(iter(a.get_attribute("prov:agent")))
        assert agent_id in expected_agents
        expected_agents.remove(agent_id)
    assert len(expected_agents) == 0
    data = [
        arg.attrs["provenance"].identifier for arg in args if "provenance" in arg.attrs
    ]
    data2 = list(data)
    for d in doc.get_records(prov.ProvDerivation):
        assert {entity.identifier} == d.get_attribute("prov:generatedEntity")
        used_id = next(iter(d.get_attribute("prov:usedEntity")))
        assert used_id in data
        data.remove(used_id)
    assert len(data) == 0
    data = data2
    for u in doc.get_records(prov.ProvUsage):
        assert {activity_id} == u.get_attribute("prov:activity")
        entity_id = next(iter(u.get_attribute("prov:entity")))
        assert entity_id in data
        data.remove(entity_id)
    assert len(data) == 0


@given(
    shared(arguments, key="args"),
    shared(arguments, key="args").flatmap(compatible_operators),
    shared(small_datasets, key="result"),
    shared(small_datasets, key="result").flatmap(
        lambda x: fixed_dictionaries({k: booleans() for k in x.data_vars})
    ),
)
def test_assign_dataset_provenance(args, operator, dataset, replace_equilib):
    doc = prov.ProvDocument()
    doc.set_default_namespace("https://ccfe.ukaea.uk/")
    session = MagicMock(
        prov=doc,
        agent=doc.agent("session_agent"),
        session=doc.activity("session_activity"),
    )
    operator._session = session
    for i, arg in enumerate(args):
        if "provenance" in arg.attrs:
            arg.attrs["provenance"] = doc.entity(f"arg{i}")
    contents_prov = {}
    equilib = MagicMock(provenance=doc.entity("equilibrium"), _session=session)
    for key, var in dataset.data_vars.items():
        if "provenance" in var.attrs:
            var.attrs["provenance"] = var.attrs["partial_provenance"] = doc.entity(
                f"variable_{key}"
            )
        del var.indica.equilibrium
        if replace_equilib[key]:
            var.indica.equilibrium = equilib
        if "provenance" in var.attrs:
            contents_prov[key] = var.attrs["provenance"]
    operator.validate_arguments(*args)
    before = datetime.now()
    operator.assign_provenance(dataset)
    after = datetime.now()
    entity = dataset.attrs["provenance"]
    assert "Dataset" in entity.get_attribute(prov.PROV_TYPE)
    generated_candidates = list(
        filter(
            lambda x: {entity.identifier} == x.get_attribute("prov:entity"),
            doc.get_records(prov.ProvGeneration),
        )
    )
    assert len(generated_candidates) == 1
    generated = generated_candidates[0]
    assert {entity.identifier} == generated.get_attribute("prov:entity")
    activity_id = next(iter(generated.get_attribute("prov:activity")))
    end_time = next(iter(generated.get_attribute("prov:time")))
    assert before < end_time < after
    comms = list(
        filter(
            lambda x: x.get_attribute("prov:informed") == {activity_id},
            doc.get_records(prov.ProvCommunication),
        )
    )
    assert len(comms) == 1
    informed = comms[0]
    assert {session.session.identifier} == informed.get_attribute("prov:informant")
    expected_agents = [session.agent.identifier, operator.agent.identifier]
    for a in filter(
        lambda a: a.get_attribute("prov:activity") == {activity_id},
        doc.get_records(prov.ProvAssociation),
    ):
        agent_id = next(iter(a.get_attribute("prov:agent")))
        assert agent_id in expected_agents
        expected_agents.remove(agent_id)
    assert len(expected_agents) == 0
    expected_agents = [session.agent.identifier, operator.agent.identifier]
    for a in filter(
        lambda x: {entity.identifier} == x.get_attribute("prov:entity"),
        doc.get_records(prov.ProvAttribution),
    ):
        agent_id = next(iter(a.get_attribute("prov:agent")))
        assert agent_id in expected_agents
        expected_agents.remove(agent_id)
    assert len(expected_agents) == 0
    data = [
        arg.attrs["provenance"].identifier for arg in args if "provenance" in arg.attrs
    ]
    data2 = list(data)
    for d in filter(
        lambda d: {entity.identifier} == d.get_attribute("prov:generatedEntity"),
        doc.get_records(prov.ProvDerivation),
    ):
        used_id = next(iter(d.get_attribute("prov:usedEntity")))
        assert used_id in data
        data.remove(used_id)
    assert len(data) == 0
    data = data2
    for u in doc.get_records(prov.ProvUsage):
        assert {activity_id} == u.get_attribute("prov:activity")
        entity_id = next(iter(u.get_attribute("prov:entity")))
        assert entity_id in data
        data.remove(entity_id)
    assert len(data) == 0
    for key, var in dataset.data_vars.items():
        if key in contents_prov:
            assert contents_prov[key] == var.attrs["provenance"]
        else:
            assert "provenance" in var.attrs
            contents_prov[key] = var.attrs["provenance"]
    # Check provenance of dataset is collection of provenance of contents
    contents = [c.identifier for c in contents_prov.values()]
    for e in filter(
        lambda e: {entity.identifier} == e.get_attribute("prov:collection"),
        doc.get_records(prov.ProvMembership),
    ):
        contents_id = next(iter(e.get_attribute("prov:entity")))
        assert contents_id in contents
        contents.remove(contents_id)
    assert len(contents) == 0


@given(
    shared(arguments, key="args"),
    shared(arguments, key="args").flatmap(compatible_operators),
    shared(small_data_arrays, key="result"),
    shared(small_data_arrays, key="result"),
)
def test_assign_multiple_provenance(args, operator, result1, result2):
    session = operator._session
    doc = session.prov
    for i, arg in enumerate(args):
        if "provenance" in arg.attrs:
            arg.attrs["provenance"] = doc.entity(f"arg{i}")
    operator.validate_arguments(*args)
    operator.assign_provenance(result1)
    entity1 = result1.attrs["partial_provenance"]
    operator.assign_provenance(result2)
    entity2 = result2.attrs["partial_provenance"]
    assert entity1 is not entity2
    assert entity1.identifier != entity2.identifier
    generated_candidates = list(doc.get_records(prov.ProvGeneration))
    generated1_candidates = list(
        filter(
            lambda x: {entity1.identifier} == x.get_attribute("prov:entity"),
            generated_candidates,
        )
    )
    assert len(generated1_candidates) == 1
    generated1 = generated1_candidates[0]
    activity1 = next(iter(generated1.get_attribute("prov:activity")))
    generated2_candidates = list(
        filter(
            lambda x: {entity2.identifier} == x.get_attribute("prov:entity"),
            generated_candidates,
        )
    )
    assert len(generated2_candidates) == 1
    generated2 = generated2_candidates[0]
    activity2 = next(iter(generated2.get_attribute("prov:activity")))
    assert activity1 == activity2
