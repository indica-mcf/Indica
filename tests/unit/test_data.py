"""Tests of methods on the custom accessors used with xarray objects."""

from unittest.mock import MagicMock
from unittest.mock import patch

from hypothesis import assume
from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
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
from pytest import mark
from pytest import raises
from xarray.testing import assert_allclose

from indica.data import aggregate
from .converters.test_abstract_transform import coordinate_transforms
from .converters.test_abstract_transform import coordinate_transforms_and_axes
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
        array.coords[d].equals(array.attrs["dropped"].coords[d]) for d in array.dims
    ]
    assert was_dropped.count(False) == 1
    return array.dims[was_dropped.index(False)]


# DataArray tests:
# ----------------

# TODO: Write tests for inversion routines, interp2d, with_Rz_coords


@settings(deadline=300)
@given(
    data_arrays(
        coordinates_and_axes=coordinate_transforms_and_axes(min_side=2, max_side=3)
    ),
    coordinate_transforms(),
)
def test_convert_coords(array, transform):
    actual = array.indica.convert_coords(transform)
    expected = array.attrs["transform"].convert_to(
        transform,
        array.coords[array.attrs["transform"].x1_name],
        array.coords[array.attrs["transform"].x2_name],
        array.coords["t"],
    )
    for a, e in zip(actual, expected):
        a.equals(e)


@given(
    data_arrays(
        coordinates_and_axes=coordinate_transforms_and_axes(min_side=2, max_side=5)
    ),
    coordinate_transforms(),
)
def test_convert_coords_cache(array, transform):
    assume(transform.x1_name not in array.coords)
    assume(transform.x2_name not in array.coords)
    with patch.object(
        array.attrs["transform"], "get_converter", MagicMock(return_value=None)
    ), patch.object(transform, "convert_from_Rz", wraps=transform.convert_from_Rz):
        result = array.indica.convert_coords(
            transform,
        )
        result2 = array.indica.convert_coords(transform)
        transform.convert_from_Rz.assert_called_once()
    for r1, r2 in zip(result, result2):
        assert r1.identical(r2)


@given(data_arrays())
def test_get_coords(array):
    x1 = MagicMock()
    x2 = MagicMock()
    convert = MagicMock(return_value=(x1, x2))
    with patch.object(array.indica, "convert_coords", convert):
        result = array.indica.get_coords(MagicMock())
    assert result[0] is x1
    assert result[1] is x2
    assert result[2].identical(array.coords["t"])


@given(data_arrays())
def test_get_coords_default(array):
    result = array.indica.get_coords()
    assert result[0].identical(array.coords[array.attrs["transform"].x1_name])
    assert result[1].identical(array.coords[array.attrs["transform"].x2_name])
    assert result[2].identical(array.coords["t"])


# TODO: Ensure the template data all fall within domain of the original (how?)
@mark.skip(reason="Test not yet written to reliably work on all data.")
@given(
    data_arrays(
        coordinates_and_axes=coordinate_transforms_and_axes(min_side=2, max_side=5)
    ),
    data_arrays(
        coordinates_and_axes=coordinate_transforms_and_axes(min_side=2, max_side=5)
    ),
)
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


@mark.skip(reason="Test not yet written to reliably work on all data.")
@given(
    data_arrays(
        coordinates_and_axes=coordinate_transforms_and_axes(min_side=2, max_side=5)
    ),
    data_arrays(
        coordinates_and_axes=coordinate_transforms_and_axes(min_side=2, max_side=5)
    ),
)
def test_remap_inverts(original, template):
    """Check result of remapping and then remapping back to original is
    approximately teh same as original data."""
    remapped = original.indica.remap_like(template).indica.remap_like(original)
    assert assert_allclose(remapped, original, rtol=1e-4)


@mark.skip(reason="Test not yet written to reliably work on all data.")
@given(
    lists(
        data_arrays(
            coordinates_and_axes=coordinate_transforms_and_axes(min_side=2, max_side=5)
        ),
        min_size=3,
        max_size=10,
    )
)
def test_remap_invariant(arrays):
    expected = arrays[0].indica.remap_like(arrays[1])
    actual = arrays[0]
    for template in arrays[1:]:
        actual = actual.indica.remap_like(template)
    assert assert_allclose(actual, expected, rtol=1e-4)


@mark.skip(reason="Test not yet written to reliably work on all data.")
@given(
    data_arrays(rel_sigma=0.0, abs_sigma=0.0),
    data_arrays(
        coordinates_and_axes=coordinate_transforms_and_axes(min_side=5, max_side=5)
    ),
)
def test_remap_values(original, template):
    """Check results of remapping are sensible."""
    # TODO: Rewrite so checking against function used to create the fake data
    remapped = original.indica.remap_like(template)
    minval = original.min()
    maxval = original.max()
    assert np.all(
        np.logical_and(
            minval - 0.05 * np.abs(minval) <= remapped,
            remapped <= maxval + 0.05 * np.abs(maxval),
        )
    )


@given(data_arrays())
def test_restoring_dropped_data(array):
    """Check reconstruction of array with ignored data is correct."""
    all_data = array.indica.with_ignored_data
    assert np.all(np.logical_or(all_data == array, np.isnan(array)))
    if "dropped" in array.attrs:
        assert all_data is not array
        assert all_data.loc[array.attrs["dropped"].coords].equals(
            array.attrs["dropped"]
        )
    if "error" in array.attrs:
        assert np.all(
            np.logical_or(
                all_data.attrs["error"] == array.attrs["error"],
                np.isnan(array.attrs["error"]),
            )
        )
        if "dropped" in array.attrs:
            assert all_data.attrs["error"] is not array.attrs["error"]
            assert (
                all_data.attrs["error"]
                .loc[array.attrs["dropped"].coords]
                .equals(array.attrs["dropped"].attrs["error"])
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
            just(x[0]),
            just(x[1]),
            lists(sampled_from(x[0].coords[x[1]].values), unique=True),
        )
    )
)
def test_dropping_data(arguments):
    """Check dropping new or additional data works as expected."""
    array, drop_dim, to_drop = arguments
    original_attrs = array.attrs
    result = array.indica.ignore_data(to_drop, drop_dim)
    assert original_attrs == array.attrs
    if "dropped" in array.attrs:
        assert np.all(
            np.isnan(result.loc[{drop_dim: array.attrs["dropped"].coords[drop_dim]}])
        )
        if "error" in array.attrs:
            assert np.all(
                np.isnan(
                    result.attrs["error"].loc[
                        {drop_dim: array.attrs["dropped"].coords[drop_dim]}
                    ]
                )
            )
        dropped_coords = np.unique(
            np.concatenate([to_drop, array.attrs["dropped"].coords[drop_dim]])
        )
    else:
        dropped_coords = np.unique(to_drop)
    if len(to_drop) > 0:
        assert np.all(np.isnan(result.loc[{drop_dim: to_drop}]))
        if "dropped" in array.attrs and np.any(
            np.isin(to_drop, array.attrs["dropped"].coords[drop_dim], invert=True)
        ):
            assert result.attrs["dropped"] is not array.attrs["dropped"]
        assert result.attrs["dropped"].equals(
            array.indica.with_ignored_data.loc[
                {drop_dim: result.attrs["dropped"].coords[drop_dim]}
            ]
        )
        assert np.all(
            dropped_coords == np.sort(result.attrs["dropped"].coords[drop_dim])
        )
        assert np.all(np.logical_not(np.isnan(result.attrs["dropped"])))
        if "error" in array.attrs:
            if "dropped" in array.attrs:
                if np.any(
                    np.isin(
                        to_drop, array.attrs["dropped"].coords[drop_dim], invert=True
                    )
                ):
                    assert (
                        result.attrs["dropped"].attrs["error"]
                        is not array.attrs["dropped"].attrs["error"]
                    )
                    assert result.attrs["error"] is not array.attrs["error"]
            else:
                assert result.attrs["error"] is not array.attrs["error"]
            assert np.all(np.isnan(result.attrs["error"].loc[{drop_dim: to_drop}]))
            assert (
                result.attrs["dropped"]
                .attrs["error"]
                .equals(
                    array.indica.with_ignored_data.attrs["error"].loc[
                        {drop_dim: result.attrs["dropped"].coords[drop_dim]}
                    ]
                )
            )
            assert np.all(
                dropped_coords
                == np.sort(result.attrs["dropped"].attrs["error"].coords[drop_dim])
            )
            assert np.all(
                np.logical_not(np.isnan(result.attrs["dropped"].attrs["error"]))
            )
    if len(dropped_coords) > 0:
        assert np.all(np.isnan(result.loc[{drop_dim: dropped_coords}]))
        assert result.drop_sel(**{drop_dim: dropped_coords}).equals(
            array.drop_sel(**{drop_dim: dropped_coords})
        )
        if "error" in array.attrs:
            assert np.all(
                np.isnan(result.attrs["error"].loc[{drop_dim: dropped_coords}])
            )
            assert (
                result.attrs["error"]
                .drop_sel(**{drop_dim: dropped_coords})
                .equals(array.attrs["error"].drop_sel(**{drop_dim: dropped_coords}))
            )


@mark.skip(reason="Struggling to generate good data to test on")
@given(
    data_arrays()
    .filter(lambda d: "dropped" in d.attrs and len(d.dims) > 1)
    .flatmap(
        lambda d: tuples(
            just(d), sampled_from(sorted(set(d.dims) - {dropped_dimension(d)}))
        )
    )
    .flatmap(
        lambda x: tuples(
            just(x[0]),
            just(x[1]),
            lists(sampled_from(x[0].coords[x[1]].values), unique=True),
        )
    )
)
def test_dropping_invalid_data(arguments):
    """Test fails when trying to drop data in different dimension than
    existing dropped channels"""
    array, drop_dim, to_drop = arguments
    with raises(ValueError):
        array.indica.ignore_data(to_drop, drop_dim)


@given(data_arrays(), lists(one_of(integers(), floats()), min_size=1), text())
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
def test_no_equilibrium(array):
    """Check getting equilibrium from array without one returns None."""
    del array.attrs["transform"].equilibrium
    assert array.indica.equilibrium is None


@given(data_arrays())
def test_set_equilibrium(array):
    """Checking setting of equilibrium is consistent and updates PROV
    accordingly.

    """
    new_equilib = MagicMock()
    assert array.indica.equilibrium
    del array.attrs["transform"].equilibrium
    old_partial_prov = array.attrs["partial_provenance"]
    old_prov = array.attrs["provenance"]
    array.indica.equilibrium = new_equilib
    assert array.attrs["transform"].equilibrium is new_equilib
    assert array.attrs["partial_provenance"] is old_partial_prov
    assert array.attrs["provenance"] is not old_prov
    new_equilib._session.prov.collection.assert_called()
    array.attrs["provenance"].hadMember.assert_any_call(new_equilib.provenance)
    array.attrs["provenance"].hadMember.assert_any_call(
        array.attrs["partial_provenance"]
    )


@given(data_arrays())
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
def test_del_equilibrium(array):
    assert array.attrs["provenance"] != array.attrs["partial_provenance"]
    del array.indica.equilibrium
    assert array.attrs["provenance"] is array.attrs["partial_provenance"]


@given(data_arrays())
def test_del_no_equilibrium(array):
    del array.attrs["transform"].equilibrium
    del array.indica.equilibrium
    # No action should have been taken to explicitly reset provenance
    assert array.attrs["provenance"] != array.attrs["partial_provenance"]


@mark.skip(reason="Method not implemented yet.")
@given(data_arrays())
def test_compatible_data_array_type(array):
    """Test checking datatype for compatible types."""
    dt = array.attrs["datatype"]
    assert array.indica.check_datatype(dt)
    assert array.indica.check_datatype((None, dt[1]))
    assert array.indica.check_datatype((dt[0], None))


@mark.skip(reason="Method not implemented yet.")
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
    new_data = next(iter(dataset.values())).assign_attrs(datatype=(general, specific))
    with raises(ValueError):
        dataset.indica.attach(key, new_data, True, MagicMock())


@given(datasets(), text(), data_arrays())
def test_attach_invalid_dims(dataset, key, data_array):
    """Check attach method fails with incompatible dimensions."""
    sample = next(iter(dataset.values()))
    assume(sample.attrs["transform"] != data_array.attrs["transform"])
    data_array.attrs["datatype"] = sample.attrs["datatype"]
    with raises(ValueError):
        dataset.indica.attach(key, data_array, True, MagicMock())


@given(datasets(), text())
def test_attach_valid(dataset, key):
    """Check attach method works correctly when given valid data"""
    assume(key not in dataset)
    new_data = next(iter(dataset.values())).copy()
    dataset.indica.attach(key, new_data, sess=MagicMock())
    assert dataset[key].equals(new_data)
    assert dataset.indica.datatype[1][key] == new_data.attrs["datatype"][0]
    dataset.attrs["provenance"].hadMember.assert_called_with(
        new_data.attrs["provenance"]
    )


@given(datasets())
def test_attach_fail_overwrite(dataset):
    """Test attach method fails to overwite existing key by default."""
    key, value = next(iter(dataset.items()))
    with raises(ValueError):
        dataset.indica.attach(key, value.copy(), sess=MagicMock())


@given(datasets())
def test_attach_overwrite(dataset):
    """Test attach method succeeds in overwriting existing key when
    explicitly told it may do so.

    """
    key, original_value = next(iter(dataset.items()))
    value = (original_value + 1).assign_attrs(original_value.attrs)
    assert dataset[key].equals(original_value)
    dataset.indica.attach(key, value, True, MagicMock())
    assert dataset[key].equals(value)


@given(datasets())
def test_dataset_datatype(dataset):
    """Check datatype matches contents of Dataset"""
    dtype = dataset.indica.datatype
    assert len(dtype[1]) == len(dataset.data_vars)
    for key, general_type in dtype[1].items():
        var_type = dataset[key].attrs["datatype"]
        assert var_type[0] == general_type
        assert var_type[1] == dtype[0]


@mark.skip(reason="Method not implemented")
@given(
    datasets().flatmap(
        lambda d: tuples(just(d), compatible_dataset_types(d.indica.datatype))
    )
)
def test_compatible_dataset_type(arguments):
    """Test that checking datatype works for compatible ones: fewer
    variables and where a variable has unconstrained general_datatype."""
    dataset, compatible_type = arguments
    assert dataset.indica.check_datatype(compatible_type)


@mark.skip(reason="Method not implemented")
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
    assert not dataset.indica.check_datatype(incompatible_type)


@given(datasets())
def test_aggregate(dataset):
    """Check aggregate function combines all compatible input and produces
    correct datatype/PROV.

    """
    new_dataset = aggregate(sess=MagicMock(), **dataset.data_vars)
    assert new_dataset.equals(dataset)
    for var in new_dataset.data_vars.values():
        new_dataset.attrs["provenance"].hadMember.assert_any_call(
            var.attrs["provenance"]
        )


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    datasets()
    .filter(lambda x: len(x.data_vars) > 1)
    .flatmap(
        lambda d: tuples(
            just(d),
            dictionaries(
                sampled_from(sorted(d.data_vars)),
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
    for key, new_type in invalid_keys.items():
        var = dataset.data_vars[key]
        var.attrs["datatype"] = (var.attrs["datatype"][0], new_type)
    with raises(ValueError):
        aggregate(sess=MagicMock(), **dataset.data_vars)


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    datasets()
    .filter(lambda x: len(x.data_vars) > 1)
    .flatmap(
        lambda d: tuples(
            just(d),
            dictionaries(
                sampled_from(sorted(d.data_vars)),
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
    for key, new_transform in invalid_keys.items():
        dataset.data_vars[key].attrs["transform"] = new_transform
    with raises(ValueError):
        aggregate(sess=MagicMock(), **dataset.data_vars)
