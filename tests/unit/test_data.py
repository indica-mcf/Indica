"""Tests of methods on the custom accessors used with xarray objects."""

from copy import copy
from unittest.mock import MagicMock

from hypothesis import assume
from hypothesis import given
from hypothesis.strategies import dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies import tuples
import numpy as np
from pytest import approx
from pytest import raises

from indica.data import aggregate
from .converters.test_abstract_transform import coordinate_transforms
from .data_strategies import compatible_dataset_types
from .data_strategies import data_arrays
from .data_strategies import datasets
from .data_strategies import general_datatypes
from .data_strategies import incompatible_dataset_types
from .data_strategies import specific_datatypes


def dropped_dimension(array):
    """Helper routine to return name of dimension from which data has been
    dropped. If not data has been dropped, return ``None``.

    """
    if "dropped" not in array.attrs:
        return None
    was_dropped = [
        array.coords[d] == array.attrs["dropped"].coords[d] for d in array.dims
    ]
    assert was_dropped.count(True) == 1
    return array.dims[was_dropped.index(True)]


# DataArray tests:
# ----------------


@given(data_arrays(), data_arrays())
def test_remap_metadata(original, template):
    """Test remapped data has appropriate metadata, provenance, and
    coordinates."""
    remapped = original.indica.remap_like(template)
    assert remapped.attrs["datatype"] == original.attrs["datatype"]
    assert remapped.attrs["transform"] is template.attrs["transform"]
    assert remapped.coords == template.coords
    template.equilibrium._session.prov.entity.assert_called()
    remapped.attrs["provenance"].hadMember.assert_any_call(
        template.attrs["transform"].provenance
    )
    remapped.attrs["provenance"].hadMember.assert_any_call(
        remapped.attrs["partial_provenance"]
    )
    remapped.attrs["partial_provenance"].alternateOf.assert_called_with(
        original.attrs["partial_provenance"]
    )


@given(data_arrays(), data_arrays())
def test_remap_inverts(original, template):
    """Check result of remapping and then remapping back to original is
    approximately teh same as original data."""
    remapped = original.remap_like(template).remap_like(original)
    assert np.all(remapped == approx(original, rel=1e-4))


@given(lists(data_arrays(), min_size=3, max_size=10))
def test_remap_invariant(arrays):
    expected = arrays[0].indica.remap_like(arrays[1])
    actual = arrays[0]
    for template in arrays[1:]:
        actual = actual.indica.remap_like(template)
    assert np.all(actual == approx(expected, rel=1e-4))


@given(data_arrays(rel_sigma=0.0, abs_sigma=0.0), data_arrays())
def test_remap_values(original, template):
    """Check results of remapping are sensible."""
    # TODO: Rewrite so checking against function used to create the fake data
    remapped = original.indica.remap_like(template)
    assert np.all(original.min() <= remapped <= original.max())


@given(data_arrays())
def test_restoring_dropped_data(array):
    """Check reconstruction of array with ignored data is correct."""
    all_data = array.with_ignored_data
    assert np.all(np.logical_or(all_data == array.with_ignored_data, np.isnan(array)))
    if "dropped" in array.attrs:
        assert all_data.loc[array.attrs["dropped"].coords].equals(
            array.attrs["dropped"]
        )


@given(
    data_arrays()
    .flatmap(
        lambda d: tuples(
            just(d),
            just(dropped_dimension(d))
            if dropped_dimension(d)
            else sampled_from(d.dims),
        )
    )
    .flatmap(
        lambda x: tuples(
            just(x[0]), just(x[1]), lists(sampled_from(x[0].coords[x[1]]), unique=True),
        )
    )
)
def test_dropping_data(arguments):
    """Check dropping new or additional data works as expected."""
    array, drop_dim, to_drop = arguments
    result = array.indica.ignore_data(to_drop, drop_dim)
    assert np.all(np.isnan(result.loc[{drop_dim: to_drop}]))
    assert result.attrs["dropped"].equals(
        array.indica.with_ignored_data.loc[
            {drop_dim: result.attrs["dropped"].coords[drop_dim]}
        ]
    )
    if "dropped" in array.attrs:
        assert np.all(
            np.isnan(result.loc[{drop_dim: array.attrs["dropped"].coords[drop_dim]}])
        )
    if "dropped" in array.attrs:
        dropped_coords = np.unique(
            np.concat([to_drop, array.attrs["dropped"].coords[drop_dim]])
        )
    else:
        dropped_coords = np.unique(to_drop)
    assert np.all(dropped_coords == result.attrs["dropped"].coords[drop_dim])
    assert np.all(np.isnan(result.loc[{drop_dim: dropped_coords}]))
    assert result.drop_sel(**{drop_dim: dropped_coords}).equals(
        array.drop_sel(**{drop_dim: dropped_coords})
    )


@given(
    data_arrays()
    .filter(lambda d: "dropped" in d.attrs)
    .flatmap(
        lambda d: tuples(just(d), sampled_from(set(d.dims) - {dropped_dimension(d)}))
    )
    .flatmap(
        lambda x: tuples(
            just(x[0]), just(x[1]), lists(sampled_from(x[0].coords[x[1]]), unique=True),
        )
    )
)
@given(data_arrays())
def test_dropping_invalid_data(arguments):
    """Test fails when trying to drop data in different dimension than
    existing dropped channels"""
    array, drop_dim, to_drop = arguments
    with raises(ValueError):
        array.indica.ignore_data(to_drop, drop_dim)


@given(data_arrays(), lists(one_of(integers(), floats())), text())
def test_dropping_invalid_dim(array, to_drop, drop_dim):
    """Test fails when trying to drop data in a nonexistent dimension."""
    assume(drop_dim not in array.dims)
    with raises(ValueError):
        array.indica.ignore_data(to_drop, drop_dim)


@given(data_arrays())
def test_get_equilibrium(array):
    """Check equilibrium is same one associated with this array's transform
    object.

    """
    assert array.indica.equilibrium is array.attrs["transform"].equilibrium


@given(data_arrays())
def test_set_equilibrium(array):
    """Checking setting of equilibrium is consistent and updates PROV
    accordingly.

    """
    new_equilib = MagicMock()
    assert array.indica.equilibrium
    old_partial_prov = array.attrs["partial_provenance"]
    old_prov = array.attrs["provenance"]
    array.indica.equilibrium = new_equilib
    assert array.attrs["transform"].equilibrium is new_equilib
    assert array.attrs["partial_provenance"] is old_partial_prov
    assert array.attrs["provenance"] is not old_prov
    new_equilib._session.prov.entity.assert_called()
    array.attrs["provenance"].hadMember.assert_any_call(new_equilib.provenance)
    array.attrs["provenance"].hadMember.assert_any_call(array.attrs["provenance"])


@given(data_arrays)
def test_set_same_equilibrium(array):
    """Check setting the equilibrium to its existing value does nothing."""
    equilib = array.indica.equilibrium
    old_partial_prov = array.attrs["partial_provenance"]
    old_prov = array.attrs["provenance"]
    array.indica.equilibrium = equilib
    assert array.attrs["transform"].equilibrium is equilib
    assert array.attrs["partial_provenance"] is old_partial_prov
    assert array.attrs["provenance"] is old_prov


@given(data_arrays())
def test_compatible_data_array_type(array):
    """Test checking datatype for compatible types."""
    dt = array.attrs["datatype"]
    assert array.indica.check_datatype(dt)
    assert array.indica.check_datatype((None, dt[1]))
    assert array.indica.check_datatype((dt[0], None))


@given(
    data_arrays().flatmap(
        lambda d: tuples(
            just(d),
            general_datatypes().filter(lambda x: x != d.attrs["datatype"][0]),
            specific_datatypes().filter(lambda x: x != d.attrs["datatype"][1]),
        )
    )
)
def test_incompatible_data_array_type(arguments):
    """Test checking datatype for incompatible types."""
    array, general, specific = arguments
    dt = array.attrs["datatype"]
    assert not array.indica.check_datatype((general, dt[1]))
    assert not array.indica.check_datatype((dt[0], specific))
    assert not array.indica.check_datatype((general, specific))


# Dataset tests:
# --------------


@given(datasets(), text(), specific_datatypes(), general_datatypes())
def test_attach_invalid_type(dataset, key, specific, general):
    """Check attach fails when different specific datatype is used."""
    assume(dataset.attrs["datatype"][0] != specific)
    new_data = next(dataset.values()).assign_attrs(datatype=(general, specific))
    with raises(ValueError):
        dataset.indica.attach(key, new_data, overwrite=True)


@given(datasets(), text(), data_arrays())
def test_attach_invalid_dims(dataset, key, data_array):
    """Check attach method fails with incompatible dimensions."""
    sample = next(dataset.values())
    assume(sample.attrs["transform"] != data_array.attrs["transform"])
    data_array.attrs["datatype"] = sample.attrs["datatype"]
    with raises(ValueError):
        dataset.indica.attach(key, data_array, overwrite=True)


@given(datasets(), text())
def test_attach_valid(dataset, key):
    """Check attach method works correctly when given valid data"""
    assume(key not in dataset)
    new_data = next(dataset.values()).copy()
    dataset.indica.attach(key, new_data)
    assert dataset[key] is new_data
    assert dataset.indica.datatype[1][key] == new_data.attrs["datatype"]
    dataset.attrs["provenance"].hadMember.assert_called_with(
        new_data.attrs["provenance"]
    )


@given(datasets())
def test_attach_fail_overwrite(dataset):
    """Test attach method fails to overwite existing key by default."""
    key, value = next(dataset.items())
    with raises(ValueError):
        dataset.indica.attach(key, value.copy())


@given(datasets())
def test_attach_overwrite(dataset):
    """Test attach method succeeds in overwriting existing key when
    explicitly told it may do so.

    """
    key, value = next(dataset.items())
    value = value.copy()
    assert dataset[key] is not value
    dataset.indica.attach(key, value, overwrite=True)
    assert dataset[key] is value


@given(datasets())
def test_dataset_datatype(dataset):
    """Check datatype matches contents of Dataset"""
    dtype = dataset.indica.datatype
    assert len(dtype[1]) == len(dataset.data_vars)
    for key, general_type in dtype[1].items():
        var_type = dataset[key].attrs["datatype"]
        assert var_type[0] == general_type
        assert var_type[1] == dtype[0]


@given(
    datasets().flatmap(
        lambda d: tuples(just(d), compatible_dataset_types(d.indica.datatype))
    )
)
def test_compatible_dataset_type(arguments):
    """Test that checking datatype works for compatible ones: fewer
    variables and where a variable has unconstrained general_datatype."""
    dataset, compatible_type = arguments
    assert dataset.indica.compatible_dataset_types(compatible_type)


@given(
    datasets().flatmap(
        lambda d: tuples(just(d), incompatible_dataset_types(d.indica.datatype))
    )
)
def test_incompatible_dataset_type(arguments):
    """Test that checking datatype works for incompatible ones where a
    variable is required that is not present, there are inconsistent
    specific datatypes, or the general datatypes of a variable are
    inconsistent.

    """
    dataset, incompatible_type = arguments
    assert not dataset.indica.compatible_dataset_types(incompatible_type)


@given(datasets())
def test_aggregate(dataset):
    """Check aggregate function combines all compatible input and produces
    correct datatype/PROV.

    """
    new_dataset = aggregate(**dataset.data_vars)
    assert new_dataset.equals(dataset)
    for var in new_dataset.data_vars.values():
        new_dataset.attrs["provenance"].hadMember.assert_any_call(
            var.attrs["provenance"]
        )


@given(
    datasets().flatmap(
        lambda d: tuples(
            just(d),
            dictionaries(
                sampled_from(d.data_vars),
                specific_datatypes().filter(lambda t: t != d.attrs["datatype"][0]),
                min_size=1,
                max_size=len(d) - 1,
            ),
        )
    )
)
def test_aggregate_incompatible_types(arguments):
    """Test aggregate function fails when one or more data array have
    incompatible specific types."""
    dataset, invalid_keys = arguments
    for key, new_type in invalid_keys:
        var = dataset.data_vars[key]
        var.attrs["datatype"] = (new_type, var.attrs["datatype"][1])
    with raises(ValueError):
        aggregate(**dataset.data_vars)


@given(
    datasets().flatmap(
        lambda d: tuples(
            just(d),
            dictionaries(
                sampled_from(d.data_vars),
                coordinate_transforms(),
                min_size=1,
                max_size=len(d) - 1,
            ),
        )
    )
)
def test_aggregate_incompatible_transforms(arguments):
    """Test aggregate function fails when one or more data array have
    incompatible coordinate transforms."""
    dataset, invalid_keys = arguments
    for key, new_transform in invalid_keys:
        dataset.data_vars[key].attrs["transform"] = new_transform
    with raises(ValueError):
        aggregate(**dataset.data_vars)


@given(
    datasets().flatmap(
        lambda d: tuples(
            just(d),
            lists(sampled_from(d.data_vars), min_size=1, max_size=len(d) - 1).flatmap(
                lambda keys: {
                    key: data_arrays(d[key].attrs["datatype"]) for key in keys
                }
            ),
        )
    )
)
def test_aggregate_incompatible_grids(arguments):
    """Test aggregate function fails when one or more data array have
    different grids."""
    dataset, new_data = arguments
    transform = next(iter(dataset.data_vars)).attrs["transform"]
    arrays = copy(dataset.data_vars)
    for key, val in new_data:
        val.attrs["transform"] = transform
        arrays[key] = val.swap_dims(
            {old: new for old, new in zip(arrays[key].dims, val.dims)}
        )
    with raises(ValueError):
        aggregate(**dataset.data_vars)
