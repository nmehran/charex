import numpy as np
import pytest


def copy_arg(arg):
    if isinstance(arg, np.ndarray):
        return arg.copy()
    return arg


def copy_args(args):
    return tuple(copy_arg(arg) for arg in args)


def assert_arrays_unchanged(args, before):
    for arg, expected in zip(args, before):
        if isinstance(arg, np.ndarray):
            if np.array_equal(arg, expected):
                continue
            try:
                unchanged = np.array_equal(arg, expected, equal_nan=True)
            except TypeError:
                unchanged = False
            if not unchanged:
                np.testing.assert_array_equal(arg, expected)


def assert_same(implementation, baseline, *args):
    impl_args = copy_args(args)
    base_args = copy_args(args)
    before = copy_args(impl_args)

    expected = baseline(*base_args)
    actual = implementation(*impl_args)

    assert isinstance(actual, np.ndarray) is isinstance(expected, np.ndarray)
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    assert_arrays_unchanged(impl_args, before)


def assert_same_exception(implementation, baseline, *args):
    impl_args = copy_args(args)
    base_args = copy_args(args)
    before = copy_args(impl_args)

    with pytest.raises(Exception) as expected:
        baseline(*base_args)
    with pytest.raises(type(expected.value)):
        implementation(*impl_args)
    assert_arrays_unchanged(impl_args, before)


def assert_same_view(implementation, baseline, *args):
    before = copy_args(args)

    expected = baseline(*args)
    actual = implementation(*args)

    assert isinstance(actual, np.ndarray) is isinstance(expected, np.ndarray)
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    assert_arrays_unchanged(args, before)


def assert_same_view_exception(implementation, baseline, *args):
    before = copy_args(args)

    with pytest.raises(Exception) as expected:
        baseline(*args)
    with pytest.raises(type(expected.value)):
        implementation(*args)
    assert_arrays_unchanged(args, before)


def assert_same_view_outcome(implementation, baseline, *args):
    try:
        baseline(*args)
    except Exception:
        assert_same_view_exception(implementation, baseline, *args)
    else:
        assert_same_view(implementation, baseline, *args)
