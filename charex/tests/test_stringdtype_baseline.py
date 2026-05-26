"""Baseline tests for future NumPy StringDType support."""

import numpy as np
import pytest
from numba import njit, typeof
from numba.core.errors import NumbaValueError, TypingError

from charex.tests.definitions import StringsInformation
from charex.tests.support import assert_same


STRINGS = getattr(np, 'strings', None)
STRING_DTYPE = getattr(getattr(np, 'dtypes', None), 'StringDType', None)
pytestmark = pytest.mark.skipif(
    STRINGS is None or STRING_DTYPE is None,
    reason='StringDType requires NumPy 2.x',
)


def stringdtype_array(values):
    return np.array(values, dtype=STRING_DTYPE())


def test_numba_rejects_stringdtype_arrays_before_charex_dispatch():
    values = stringdtype_array(['a', 'é', '🙂'])

    with pytest.raises(NumbaValueError, match='Unsupported array dtype'):
        typeof(values)


def test_numpy_stringdtype_strlen_counts_codepoints():
    values = stringdtype_array(['a', 'é', '🙂', '', 'a\x00b'])

    np.testing.assert_array_equal(
        STRINGS.str_len(values),
        np.array([1, 1, 1, 0, 3]),
    )


@pytest.mark.xfail(
    raises=TypingError,
    strict=True,
    reason='Numba 0.65.1 rejects StringDType arrays before charex overloads',
)
def test_stringdtype_str_len_target_behavior():
    strings = StringsInformation()
    values = stringdtype_array(['a', 'é', '🙂', '', 'a\x00b'])

    assert_same(strings.strings_str_len, STRINGS.str_len, values)


@pytest.mark.xfail(
    raises=TypingError,
    strict=True,
    reason='Numba 0.65.1 rejects StringDType arrays before charex overloads',
)
def test_direct_numba_stringdtype_target_behavior():
    values = stringdtype_array(['a', 'é', '🙂'])

    @njit
    def strlen(x):
        return np.strings.str_len(x)

    np.testing.assert_array_equal(strlen(values), STRINGS.str_len(values))
