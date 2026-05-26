"""Baseline tests for future NumPy StringDType support."""

import numpy as np
import pytest
import subprocess
import sys
import textwrap
from numba import njit, typeof
from numba.core.errors import NumbaValueError, TypingError

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


@pytest.mark.parametrize('impl_name, binary', [
    ('strings_str_len', False),
    ('strings_equal', True),
    ('strings_not_equal', True),
    ('strings_startswith', True),
    ('strings_endswith', True),
    ('strings_find', True),
    ('strings_rfind', True),
    ('strings_count', True),
])
def test_stringdtype_zero_dimensional_arrays_are_rejected(impl_name, binary):
    info = StringsInformation()
    comparisons = StringsComparisonOperators()
    owner = comparisons if impl_name in {'strings_equal',
                                         'strings_not_equal'} else info
    values = np.array('abc', dtype=STRING_DTYPE())
    call_args = (values, values) if binary else (values,)

    with pytest.raises(TypingError, match='one-dimensional arrays'):
        getattr(owner, impl_name)(*call_args)


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


def test_stringdtype_str_len_long_trailing_nuls():
    strings = StringsInformation()
    values = stringdtype_array([
        'abc' + '\x00' * 128,
        'é' * 8 + '\x00' * 64,
        '🙂' * 4 + '\x00' * 64,
        '\x00' * 128,
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
    with pytest.raises(TypingError, match='without na_object'):
        strings.strings_str_len(values)


@pytest.mark.parametrize('na_object', [None, np.nan, 'MISSING'])
def test_stringdtype_na_object_variants_are_rejected(na_object):
    values = np.array(['a', na_object, 'bb'],
                      dtype=STRING_DTYPE(na_object=na_object))

    with pytest.raises(NumbaValueError, match='without na_object'):
        typeof(values)


def test_stringdtype_requires_native_helper():
    script = r'''
import sys


class BlockStringDTypeHelper:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'charex._stringdtype':
            raise ImportError('blocked native helper')
        return None


sys.meta_path.insert(0, BlockStringDTypeHelper())

import numpy as np
from numba import typeof
import charex

values = np.array(['a'], dtype=np.dtypes.StringDType())
try:
    typeof(values)
except Exception as exc:
    message = str(exc)
    if 'compiled charex._stringdtype helper' in message:
        raise SystemExit(0)
    print(type(exc).__name__, message)
    raise SystemExit(2)
raise SystemExit(1)
'''
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


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


def test_stringdtype_array_equal_embedded_nul_symmetry():
    strings = StringsComparisonOperators()
    left = stringdtype_array([
        'ab\x00', 'a\x00x', 'abc\x00', 'abc\x00x', '\x00ab',
        'abc', 'abc\x00', 'a\x00bc', 'ab\x00c',
    ])
    right = stringdtype_array([
        'a\x00x', 'ab\x00', 'abc\x00y', 'abc\x00z', '\x00cd',
        'ab\x00', 'abcx', 'a\x00zz', 'ab\x00z',
    ])

    assert_same(strings.strings_equal, STRINGS.equal, left, right)
    assert_same(strings.strings_not_equal, STRINGS.not_equal, left, right)
    assert_same(strings.strings_equal, STRINGS.equal, right, left)
    assert_same(strings.strings_not_equal, STRINGS.not_equal, right, left)


def test_stringdtype_array_equal_same_array_matches_numpy():
    strings = StringsComparisonOperators()
    values = stringdtype_array(['a', 'é', '🙂', 'a\x00\x00'])

    assert_same(strings.strings_equal, STRINGS.equal, values, values)
    assert_same(strings.strings_not_equal, STRINGS.not_equal, values, values)


def test_stringdtype_array_equal_empty_arrays_match_numpy():
    strings = StringsComparisonOperators()
    left = stringdtype_array([])
    right = stringdtype_array([])

    assert_same(strings.strings_equal, STRINGS.equal, left, right)
    assert_same(strings.strings_not_equal, STRINGS.not_equal, left, right)


def test_stringdtype_array_equal_readonly_arrays_match_numpy():
    strings = StringsComparisonOperators()
    left = stringdtype_array(['a', 'é', '🙂'])
    right = stringdtype_array(['a', 'e', '🙂'])
    left.flags.writeable = False
    right.flags.writeable = False

    assert_same(strings.strings_equal, STRINGS.equal, left, right)
    assert_same(strings.strings_not_equal, STRINGS.not_equal, left, right)


def test_stringdtype_array_equal_shape_mismatch():
    strings = StringsComparisonOperators()
    left = stringdtype_array(['a', 'b', 'c'])
    right = stringdtype_array(['a', 'b'])

    with pytest.raises(ValueError, match='shape mismatch'):
        strings.strings_equal(left, right)
    with pytest.raises(ValueError, match='shape mismatch'):
        strings.strings_not_equal(left, right)


def test_stringdtype_array_equal_rejects_noncontiguous_arrays():
    strings = StringsComparisonOperators()
    values = stringdtype_array(['a', 'b', 'c', 'd'])

    with pytest.raises(TypingError, match='C-contiguous'):
        strings.strings_equal(values[::2], values[::2])


def test_stringdtype_array_equal_rejects_multidimensional_arrays():
    strings = StringsComparisonOperators()
    values = stringdtype_array(['a', 'b', 'c', 'd']).reshape(2, 2)

    with pytest.raises(TypingError, match='one-dimensional arrays'):
        strings.strings_equal(values, values)


@pytest.mark.parametrize('scalar_left', [False, True])
def test_stringdtype_array_equal_rejects_mixed_stringdtype_inputs(scalar_left):
    strings = StringsComparisonOperators()
    values = stringdtype_array(['a', 'b'])
    args = ('a', values) if scalar_left else (values, 'a')

    with pytest.raises(TypingError, match='two StringDType arrays'):
        strings.strings_equal(*args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
])
@pytest.mark.parametrize('args', [
    (),
    (0, None),
    (0, 0),
    (1, None),
    (-3, None),
    (0, -1),
    (-4, -1),
    (10, None),
])
def test_stringdtype_array_affix_matches_numpy(impl_name, baseline, args):
    strings = StringsInformation()
    values = stringdtype_array([
        'abc', 'abc\x00', 'abc\x00x', 'a\x00bc', '\x00abc',
        '', '🙂abc', 'a🙂c',
    ])
    patterns = stringdtype_array([
        'abc', 'abc', 'abc\x00', 'a\x00', '\x00a',
        '', '🙂', '🙂c',
    ])

    assert_same(getattr(strings, impl_name), baseline,
                values, patterns, *args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
])
def test_stringdtype_array_affix_empty_arrays_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array([])
    patterns = stringdtype_array([])

    assert_same(getattr(strings, impl_name), baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
])
def test_stringdtype_array_affix_readonly_arrays_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array(['abc', 'éfg', '🙂abc'])
    patterns = stringdtype_array(['ab', 'éf', '🙂'])
    values.flags.writeable = False
    patterns.flags.writeable = False

    assert_same(getattr(strings, impl_name), baseline, values, patterns)


@pytest.mark.parametrize('impl_name', [
    'strings_startswith',
    'strings_endswith',
])
def test_stringdtype_array_affix_shape_mismatch(impl_name):
    strings = StringsInformation()
    values = stringdtype_array(['abc', 'def', 'ghi'])
    patterns = stringdtype_array(['a', 'd'])

    with pytest.raises(ValueError, match='shape mismatch'):
        getattr(strings, impl_name)(values, patterns)


@pytest.mark.parametrize('impl_name', [
    'strings_startswith',
    'strings_endswith',
])
def test_stringdtype_array_affix_rejects_noncontiguous_arrays(impl_name):
    strings = StringsInformation()
    values = stringdtype_array(['a', 'b', 'c', 'd'])

    with pytest.raises(TypingError, match='C-contiguous'):
        getattr(strings, impl_name)(values[::2], values[::2])


@pytest.mark.parametrize('impl_name', [
    'strings_startswith',
    'strings_endswith',
])
def test_stringdtype_array_affix_rejects_multidimensional_arrays(impl_name):
    strings = StringsInformation()
    values = stringdtype_array(['a', 'b', 'c', 'd']).reshape(2, 2)

    with pytest.raises(TypingError, match='one-dimensional arrays'):
        getattr(strings, impl_name)(values, values)


@pytest.mark.parametrize('impl_name', [
    'strings_startswith',
    'strings_endswith',
])
@pytest.mark.parametrize('scalar_left', [False, True])
def test_stringdtype_array_affix_rejects_mixed_stringdtype_inputs(
        impl_name, scalar_left):
    strings = StringsInformation()
    values = stringdtype_array(['a', 'b'])
    args = ('a', values) if scalar_left else (values, 'a')

    with pytest.raises(TypingError, match='two StringDType arrays'):
        getattr(strings, impl_name)(*args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
])
def test_stringdtype_array_affix_none_start_rejected_by_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array(['abc'])
    patterns = stringdtype_array(['a'])

    with pytest.raises(Exception):
        baseline(values, patterns, None, None)
    with pytest.raises(Exception):
        getattr(strings, impl_name)(values, patterns, None, None)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
])
@pytest.mark.parametrize('args', [
    (),
    (0, None),
    (0, 0),
    (1, None),
    (-4, None),
    (0, -1),
    (-5, -1),
    (10, None),
])
def test_stringdtype_array_search_matches_numpy(impl_name, baseline, args):
    strings = StringsInformation()
    values = stringdtype_array([
        'abcabc', 'éfgé', '🙂a🙂', 'a\x00bc\x00bc', '\x00abc',
        '', 'aaaa', 'a🙂a🙂',
    ])
    patterns = stringdtype_array([
        'bc', 'é', '🙂', '\x00bc', '\x00a',
        '', 'aa', '🙂a',
    ])

    assert_same(getattr(strings, impl_name), baseline,
                values, patterns, *args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
])
def test_stringdtype_array_search_trailing_nul_patterns_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array([
        'a', 'ab', 'ab\x00x', 'abc', 'abc\x00x',
        'é', 'é\x00x', 'xé\x00y',
        '🙂', '🙂\x00x', 'x🙂\x00y',
        'a\x00b', 'a\x00bc', 'x\x00b\x00y', '\x00b',
    ])
    patterns = stringdtype_array([
        'a\x00', 'ab\x00', 'ab\x00', 'c\x00', 'bc\x00',
        'é\x00', 'é\x00', 'é\x00',
        '🙂\x00', '🙂\x00', '🙂\x00',
        'a\x00b\x00', 'a\x00b\x00', '\x00b\x00', '\x00b\x00',
    ])

    assert_same(getattr(strings, impl_name), baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
])
def test_stringdtype_array_search_empty_arrays_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array([])
    patterns = stringdtype_array([])

    assert_same(getattr(strings, impl_name), baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
])
def test_stringdtype_array_search_readonly_arrays_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array(['abcabc', 'éfgé', '🙂a🙂'])
    patterns = stringdtype_array(['bc', 'é', '🙂'])
    values.flags.writeable = False
    patterns.flags.writeable = False

    assert_same(getattr(strings, impl_name), baseline, values, patterns)


@pytest.mark.parametrize('impl_name', [
    'strings_find',
    'strings_rfind',
    'strings_count',
])
def test_stringdtype_array_search_shape_mismatch(impl_name):
    strings = StringsInformation()
    values = stringdtype_array(['abc', 'def', 'ghi'])
    patterns = stringdtype_array(['a', 'd'])

    with pytest.raises(ValueError, match='shape mismatch'):
        getattr(strings, impl_name)(values, patterns)


@pytest.mark.parametrize('impl_name', [
    'strings_find',
    'strings_rfind',
    'strings_count',
])
def test_stringdtype_array_search_rejects_noncontiguous_arrays(impl_name):
    strings = StringsInformation()
    values = stringdtype_array(['a', 'b', 'c', 'd'])

    with pytest.raises(TypingError, match='C-contiguous'):
        getattr(strings, impl_name)(values[::2], values[::2])


@pytest.mark.parametrize('impl_name', [
    'strings_find',
    'strings_rfind',
    'strings_count',
])
def test_stringdtype_array_search_rejects_multidimensional_arrays(impl_name):
    strings = StringsInformation()
    values = stringdtype_array(['a', 'b', 'c', 'd']).reshape(2, 2)

    with pytest.raises(TypingError, match='one-dimensional arrays'):
        getattr(strings, impl_name)(values, values)


@pytest.mark.parametrize('impl_name', [
    'strings_find',
    'strings_rfind',
    'strings_count',
])
@pytest.mark.parametrize('scalar_left', [False, True])
def test_stringdtype_array_search_rejects_mixed_stringdtype_inputs(
        impl_name, scalar_left):
    strings = StringsInformation()
    values = stringdtype_array(['a', 'b'])
    args = ('a', values) if scalar_left else (values, 'a')

    with pytest.raises(TypingError, match='two StringDType arrays'):
        getattr(strings, impl_name)(*args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
])
def test_stringdtype_array_search_none_start_rejected_by_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array(['abc'])
    patterns = stringdtype_array(['a'])

    with pytest.raises(Exception):
        baseline(values, patterns, None, None)
    with pytest.raises(Exception):
        getattr(strings, impl_name)(values, patterns, None, None)
