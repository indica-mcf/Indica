"""Test the contents of the utilities module."""

import re

from hypothesis import assume, example, given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import dictionaries, from_regex, none, sampled_from
import numpy as np

from src import utilities

VALID_FILENAME = re.compile(r"^[a-zA-Z0-9_\-().]+$")


def test_positional_parameters1():
    def example(a, b, c=None, d=5):
        pass

    params, varpos = utilities.positional_parameters(example)
    assert params == ["a", "b", "c", "d"]
    assert varpos is None


def test_positional_parameters2():
    def example():
        pass

    params, varpos = utilities.positional_parameters(example)
    assert params == []
    assert varpos is None


def test_positional_parameters3():
    def example(a, /, b, c, *, d):
        pass

    params, varpos = utilities.positional_parameters(example)
    assert params == ["a", "b", "c"]
    assert varpos is None


def test_positional_parameters4():
    def example(a, b, c, *args, **kwargs):
        pass

    params, varpos = utilities.positional_parameters(example)
    assert params == ["a", "b", "c"]
    assert varpos == "args"


@given(arrays(np.float, array_shapes(), elements=sampled_from([1.0, -1.0])))
def test_sum_squares_ones(a):
    """Test summing arrays made up only of +/- 1."""
    print(a)
    print(a.shape)
    for i, l in enumerate(a.shape):
        assert np.all(utilities.sum_squares(a, i) == l)


@given(dictionaries(from_regex("[_a-zA-Z0-9]+", fullmatch=True), none()))
def test_sum_squares_known(kwargs):
    assume("x" not in kwargs)
    assume("axis" not in kwargs)
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert utilities.sum_squares(a, 0, **kwargs) == 55


@given(from_regex(VALID_FILENAME.pattern[1:-1]))
@example("this/ is \\ the 'n@stiest' èxample_I-can think of")
def test_to_filename(name_in):
    # Check there are some valid characthers in the filename to start with
    assume(VALID_FILENAME.match(name_in))
    name_out = utilities.to_filename(name_in)
    print(name_in, name_out, VALID_FILENAME.match(name_out))
    assert VALID_FILENAME.match(name_out) is not None


def test_to_filename_known_result():
    assert utilities.to_filename("a/b/C\\d-e(f, g)") == "a-b-C-d-e(f_g)"
