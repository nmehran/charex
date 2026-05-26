"""Baseline tests for future NumPy StringDType support."""

import numpy as np
import pytest
from numba import njit, typeof

from charex.tests.definitions import (
    StringsComparisonOperators, StringsInformation,
)
from charex.tests.support import assert_same


STRINGS = getattr(np, 'strings', None)
STRING_DTYPE = getattr(getattr(np, 'dtypes', None), 'StringDType', None)
pytestmark = pytest.mark.skipif(
    STRINGS is None or STRING_DTYPE is None,
    reason='StringDType requires NumPy 2.x',
)


def stringdtype_array(values):
    return np.array(values, dtype=STRING_DTYPE())


def test_charex_registers_stringdtype_array_type():
    values = stringdtype_array(['a', 'é', '🙂'])

    value_type = typeof(values)

    assert value_type.ndim == 1
    assert value_type.layout == 'C'
    assert value_type.dtype.name == 'StringDTypePacket'


def test_stringdtype_shape_metadata_compiles():
    values = stringdtype_array(['a', 'é', '🙂'])

    @njit
    def shape_info(value):
        return value.shape[0], value.itemsize, value.strides[0]

    assert shape_info(values) == (3, 16, 16)


def test_numpy_stringdtype_strlen_counts_codepoints():
    values = stringdtype_array([
        'a', 'é', '🙂', '', 'a\x00b', 'a\x00', 'a\x00\x00',
        '\x00b', '\x00\x00',
    ])

    np.testing.assert_array_equal(
        STRINGS.str_len(values),
        np.array([1, 1, 1, 0, 3, 1, 1, 2, 0]),
    )


def test_stringdtype_str_len_target_behavior():
    strings = StringsInformation()
    values = stringdtype_array([
        'a', 'é', '🙂', '', 'a\x00b', 'a\x00', 'a\x00\x00',
        '\x00b', '\x00\x00',
    ])

    assert_same(strings.strings_str_len, STRINGS.str_len, values)


def test_direct_numba_stringdtype_target_behavior():
    values = stringdtype_array(['a', 'é', '🙂'])

    @njit
    def strlen(x):
        return np.strings.str_len(x)

    np.testing.assert_array_equal(strlen(values), STRINGS.str_len(values))


def test_stringdtype_str_len_rejects_null_strings():
    dtype = STRING_DTYPE(na_object=None)
    values = np.array(['a', None, 'bb'], dtype=dtype)
    strings = StringsInformation()

    with pytest.raises(ValueError, match='length of a null string'):
        STRINGS.str_len(values)
    with pytest.raises(ValueError, match='length of a null string'):
        strings.strings_str_len(values)


def test_stringdtype_array_equal_matches_numpy():
    strings = StringsComparisonOperators()
    left = stringdtype_array([
        'a', 'é', '🙂', '', 'a\x00b', 'a\x00', 'a\x00\x00',
        '\x00b', '\x00\x00', 'a\x00b\x00',
    ])
    right = stringdtype_array([
        'a', 'e', '🙂', '', 'a\x00c', 'a', 'a\x00b',
        '\x00c', '\x00c', 'a\x00c\x00',
    ])

    assert_same(strings.strings_equal, STRINGS.equal, left, right)
    assert_same(strings.strings_not_equal, STRINGS.not_equal, left, right)
