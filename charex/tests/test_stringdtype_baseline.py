"""Baseline tests for future NumPy StringDType support."""

import numpy as np
import pytest
import subprocess
import sys
import textwrap
from numba import njit, typeof
from numba.core.errors import TypingError

from charex.tests.definitions import (
    StringsComparisonOperators, StringsInformation,
)
from charex.tests.support import (
    assert_arrays_unchanged, assert_same, assert_same_exception, copy_args,
)


STRINGS = getattr(np, 'strings', None)
STRING_DTYPE = getattr(getattr(np, 'dtypes', None), 'StringDType', None)
pytestmark = pytest.mark.skipif(
    STRINGS is None or STRING_DTYPE is None,
    reason='StringDType requires NumPy 2.x',
)


def stringdtype_array(values):
    return np.array(values, dtype=STRING_DTYPE())


NA_OBJECT_VARIANTS = [
    None, np.nan, np.float32(np.nan), complex(np.nan, 0),
    'MISSING', '', 'é', '🙂', 0, False,
]


def na_object_is_nan_like(na_object):
    try:
        return bool(np.isnan(na_object))
    except TypeError:
        return False


def na_object_is_string_like(na_object):
    return isinstance(na_object, str)


STRINGDTYPE_PREDICATES = [
    ('strings_isalpha', STRINGS.isalpha),
    ('strings_isalnum', STRINGS.isalnum),
    ('strings_isdecimal', STRINGS.isdecimal),
    ('strings_isdigit', STRINGS.isdigit),
    ('strings_islower', STRINGS.islower),
    ('strings_isnumeric', STRINGS.isnumeric),
    ('strings_isspace', STRINGS.isspace),
    ('strings_istitle', STRINGS.istitle),
    ('strings_isupper', STRINGS.isupper),
]


STRINGDTYPE_ORDER_COMPARISONS = [
    ('strings_greater', STRINGS.greater),
    ('strings_greater_equal', STRINGS.greater_equal),
    ('strings_less', STRINGS.less),
    ('strings_less_equal', STRINGS.less_equal),
]


STRINGDTYPE_AFFIX_METHODS = [
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
]


STRINGDTYPE_SEARCH_METHODS = [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
]


STRINGDTYPE_AFFIX_SEARCH_METHODS = [
    *STRINGDTYPE_AFFIX_METHODS,
    *STRINGDTYPE_SEARCH_METHODS,
]


STRINGDTYPE_COMPARISON_METHODS = {
    'strings_equal', 'strings_not_equal', 'strings_greater',
    'strings_greater_equal', 'strings_less', 'strings_less_equal',
}


def strings_impl(impl_name):
    if impl_name in STRINGDTYPE_COMPARISON_METHODS:
        return getattr(StringsComparisonOperators(), impl_name)
    return getattr(StringsInformation(), impl_name)


def assert_same_outcome(implementation, baseline, *args):
    try:
        baseline(*copy_args(args))
    except Exception:
        assert_same_exception(implementation, baseline, *args)
    else:
        assert_same(implementation, baseline, *args)


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


def test_charex_registers_stringdtype_array_type():
    values = stringdtype_array(['a', 'é', '🙂'])

    value_type = typeof(values)

    assert value_type.ndim == 1
    assert value_type.layout == 'C'
    assert value_type.dtype.name == 'StringDTypePacket'


def test_charex_registers_stringdtype_strided_array_type():
    values = stringdtype_array(['a', 'x', 'é', 'y', '🙂'])[::2]

    value_type = typeof(values)

    assert value_type.ndim == 1
    assert value_type.layout == 'A'
    assert value_type.dtype.name == 'StringDTypePacket'


def test_charex_registers_stringdtype_zero_stride_array_type():
    values = np.broadcast_to(stringdtype_array(['abc']), (3,))

    value_type = typeof(values)

    assert value_type.ndim == 1
    assert value_type.layout == 'A'
    assert value_type.dtype.name == 'StringDTypePacket'


def test_stringdtype_shape_metadata_compiles():
    values = stringdtype_array(['a', 'é', '🙂'])

    @njit
    def shape_info(value):
        return value.shape[0], value.itemsize, value.strides[0]

    assert shape_info(values) == (3, 16, 16)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_str_len', STRINGS.str_len),
    *STRINGDTYPE_PREDICATES,
])
def test_stringdtype_zero_dimensional_unary_matches_numpy(
        impl_name, baseline):
    value = np.array('abc', dtype=STRING_DTYPE())

    assert_same(strings_impl(impl_name), baseline, value)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_str_len', STRINGS.str_len),
    *STRINGDTYPE_PREDICATES,
])
@pytest.mark.parametrize('value', [
    'a', 'é', '🙂', 'a\x00', 'a\x00x', '', '\ud800',
])
def test_strings_python_str_scalar_unary_matches_numpy(
        impl_name, baseline, value):
    assert_same(strings_impl(impl_name), baseline, value)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
def test_stringdtype_zero_dimensional_binary_matches_numpy(
        impl_name, baseline):
    value = np.array('abcabc', dtype=STRING_DTYPE())
    pattern = np.array('a', dtype=STRING_DTYPE())

    assert_same(strings_impl(impl_name), baseline, value, pattern)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
def test_stringdtype_zero_dimensional_broadcast_matches_numpy(
        impl_name, baseline):
    values = stringdtype_array(['abcabc', 'a', 'ba', 'a🙂'])
    patterns = stringdtype_array(['a', 'bc', 'abc', ''])
    value = np.array('abcabc', dtype=STRING_DTYPE())
    pattern = np.array('a', dtype=STRING_DTYPE())

    assert_same(strings_impl(impl_name), baseline, values, pattern)
    assert_same(strings_impl(impl_name), baseline, value, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
@pytest.mark.parametrize('scalar_factory', [
    pytest.param(np.str_, id='numpy-str-scalar'),
    pytest.param(lambda value: np.array(value, dtype='U32'),
                 id='zero-dimensional-unicode-array'),
])
def test_stringdtype_mixed_unicode_scalar_variants_match_numpy(
        impl_name, baseline, scalar_factory):
    values = stringdtype_array(['abcabc', 'a', 'ba', 'a🙂'])
    patterns = stringdtype_array(['a', 'bc', 'abc', ''])
    value = scalar_factory('abcabcé🙂')
    pattern = scalar_factory('a')

    assert_same(strings_impl(impl_name), baseline, values, pattern)
    assert_same(strings_impl(impl_name), baseline, value, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
])
def test_stringdtype_mixed_zero_dimensional_unicode_nul_matches_numpy(
        impl_name, baseline):
    values = stringdtype_array(['a\x00x', 'a', '\x00x', ''])
    patterns = stringdtype_array(['a\x00x', 'a', '\x00x', ''])
    value = np.array('a\x00x', dtype='U8')
    pattern = np.array('a\x00x', dtype='U8')

    assert_same(strings_impl(impl_name), baseline, values, pattern)
    assert_same(strings_impl(impl_name), baseline, value, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
def test_stringdtype_mixed_unicode_arrays_match_numpy(
        impl_name, baseline):
    values = stringdtype_array(['abcabc', 'a', 'ba', 'a🙂', 'éa'])
    patterns = stringdtype_array(['a', '', 'b', '🙂', 'é'])
    unicode_values = np.array(['abcabc', 'a', 'ba', 'a🙂', 'éa'],
                              dtype='U16')
    unicode_patterns = np.array(['a', '', 'b', '🙂', 'é'],
                                dtype='U16')
    value = np.array('abcabcé🙂', dtype=STRING_DTYPE())
    pattern = np.array('a', dtype=STRING_DTYPE())

    assert_same(strings_impl(impl_name), baseline,
                values, unicode_patterns)
    assert_same(strings_impl(impl_name), baseline,
                unicode_values, patterns)
    assert_same(strings_impl(impl_name), baseline,
                value, unicode_patterns)
    assert_same(strings_impl(impl_name), baseline,
                unicode_values, pattern)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
])
def test_stringdtype_mixed_unicode_array_nul_matches_numpy(
        impl_name, baseline):
    values = stringdtype_array(['a\x00x', 'a', '\x00x', '', 'ab\x00c'])
    patterns = stringdtype_array(['a\x00x', 'a', '\x00x', '', 'ab\x00'])
    unicode_values = np.array(['a\x00x', 'a', '\x00x', '', 'ab\x00c'],
                              dtype='U8')
    unicode_patterns = np.array(['a\x00x', 'a', '\x00x', '', 'ab\x00'],
                                dtype='U8')

    assert_same(strings_impl(impl_name), baseline,
                values, unicode_patterns)
    assert_same(strings_impl(impl_name), baseline,
                unicode_values, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
])
def test_stringdtype_mixed_noncontiguous_unicode_array_matches_numpy(
        impl_name, baseline):
    values = stringdtype_array(['a', 'b', 'a\x00x'])
    patterns = stringdtype_array(['a', 'b', 'a\x00x'])
    unicode_values = np.array(['a', 'x', 'b', 'y', 'a\x00x', 'z'],
                              dtype='U8')[::2]
    unicode_patterns = np.array(['a', 'x', 'b', 'y', 'a\x00x', 'z'],
                                dtype='U8')[::2]

    assert_same(strings_impl(impl_name), baseline,
                values, unicode_patterns)
    assert_same(strings_impl(impl_name), baseline,
                unicode_values, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
def test_stringdtype_mixed_unicode_array_shape_mismatch_matches_numpy(
        impl_name, baseline):
    values = stringdtype_array(['abc', 'def', 'ghi'])
    patterns = stringdtype_array(['a', 'd', 'g'])
    unicode_values = np.array(['abc', 'def'], dtype='U8')
    unicode_patterns = np.array(['a', 'd'], dtype='U8')

    assert_same_exception(strings_impl(impl_name), baseline,
                          values, unicode_patterns)
    assert_same_exception(strings_impl(impl_name), baseline,
                          unicode_values, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
])
def test_stringdtype_mixed_none_equality_matches_numpy(
        impl_name, baseline):
    values = stringdtype_array(['a', '', 'a\x00x'])
    value = np.array('a', dtype=STRING_DTYPE())

    assert_same(strings_impl(impl_name), baseline, values, None)
    assert_same(strings_impl(impl_name), baseline, None, values)
    assert_same(strings_impl(impl_name), baseline, value, None)
    assert_same(strings_impl(impl_name), baseline, None, value)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_greater', STRINGS.greater),
    ('strings_startswith', STRINGS.startswith),
    ('strings_find', STRINGS.find),
])
@pytest.mark.parametrize('operand', [
    b'a',
    np.bytes_(b'a'),
    np.array(b'a', dtype='S8'),
    np.array([b'a', b'b'], dtype='S8'),
])
def test_stringdtype_mixed_bytes_operands_are_rejected(
        impl_name, baseline, operand):
    values = stringdtype_array(['a', 'b'])

    with pytest.raises(Exception):
        baseline(values, operand)
    with pytest.raises(Exception):
        strings_impl(impl_name)(values, operand)
    with pytest.raises(Exception):
        baseline(operand, values)
    with pytest.raises(Exception):
        strings_impl(impl_name)(operand, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_greater', STRINGS.greater),
])
@pytest.mark.parametrize('operand', [
    np.array('a', dtype=object),
    np.array(['a', 'b'], dtype=object),
])
def test_stringdtype_mixed_object_comparison_is_nopython_gap(
        impl_name, baseline, operand):
    values = stringdtype_array(['a', 'b'])

    baseline(values, operand)
    with pytest.raises(TypingError):
        strings_impl(impl_name)(values, operand)

    baseline(operand, values)
    with pytest.raises(TypingError):
        strings_impl(impl_name)(operand, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_startswith', STRINGS.startswith),
    ('strings_find', STRINGS.find),
])
@pytest.mark.parametrize('operand', [
    np.array('a', dtype=object),
    np.array(['a', 'b'], dtype=object),
])
def test_stringdtype_mixed_object_information_operands_are_rejected(
        impl_name, baseline, operand):
    values = stringdtype_array(['a', 'b'])

    with pytest.raises(Exception):
        baseline(values, operand)
    with pytest.raises(TypingError):
        strings_impl(impl_name)(values, operand)
    with pytest.raises(Exception):
        baseline(operand, values)
    with pytest.raises(TypingError):
        strings_impl(impl_name)(operand, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_greater', STRINGS.greater),
    ('strings_startswith', STRINGS.startswith),
    ('strings_find', STRINGS.find),
])
def test_stringdtype_mixed_unicode_array_invalid_unicode_matches_numpy(
        impl_name, baseline):
    values = stringdtype_array(['a', 'b'])
    unicode_values = np.array(['a', '\ud800'], dtype='U8')

    assert_same_exception(strings_impl(impl_name), baseline,
                          values, unicode_values)
    assert_same_exception(strings_impl(impl_name), baseline,
                          unicode_values, values)


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


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_PREDICATES)
def test_stringdtype_array_predicates_match_numpy(impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array([
        'alpha', 'α', '١', 'Ⅷ', '一', 'A', 'a', 'ǅuro', 'ǆuro',
        '中A', ' ', '\t', '\x1c', '\x1f', '', 'ab\x00cd',
        'abc\x00', '\x00abc', 'A1', '１２', '²', '⅕',
        'Title Case', 'Title case', 'UPPER', 'lower', '🙂',
        '\x00\x00', '🙂A', 'A🙂',
    ])

    assert_same(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_PREDICATES)
def test_stringdtype_array_predicates_empty_arrays_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array([])

    assert_same(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_PREDICATES)
def test_stringdtype_array_predicates_readonly_arrays_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array(['alpha', 'α', '١', 'Title Case'])
    values.flags.writeable = False

    assert_same(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_PREDICATES)
def test_stringdtype_array_predicates_embedded_nul_matches_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array([
        'ab\x00cd', 'ABC\x00DEF', 'Title\x00Case', '\x00abc',
        '\x00ABC', '\x00Title Case', 'abc\x00', 'ABC\x00',
    ])

    assert_same(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('impl_name', [
    name for name, _ in STRINGDTYPE_PREDICATES
])
def test_stringdtype_array_predicates_noncontiguous_arrays_match_numpy(
        impl_name):
    strings = StringsInformation()
    values = stringdtype_array([
        'x', '🙂', 'z', '', 'y', 'é', 'w', 'alpha',
    ])[::-2]
    baseline = dict(STRINGDTYPE_PREDICATES)[impl_name]

    assert_same_view(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_str_len', STRINGS.str_len),
    ('strings_isalpha', STRINGS.isalpha),
])
def test_stringdtype_unary_readonly_zero_stride_views_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = np.broadcast_to(stringdtype_array(['alpha']), (4,))

    assert not values.flags.writeable
    assert values.strides == (0,)
    assert_same_view(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('impl_name', [
    name for name, _ in STRINGDTYPE_PREDICATES
])
def test_stringdtype_array_predicates_reject_multidimensional_arrays(
        impl_name):
    strings = StringsInformation()
    values = stringdtype_array(['a', 'b', 'c', 'd']).reshape(2, 2)

    with pytest.raises(TypingError, match='one-dimensional arrays'):
        getattr(strings, impl_name)(values)


def test_direct_numba_stringdtype_target_behavior():
    values = stringdtype_array(['a', 'é', '🙂'])

    @njit
    def strlen(x):
        return np.strings.str_len(x)

    np.testing.assert_array_equal(strlen(values), STRINGS.str_len(values))


def test_stringdtype_str_len_rejects_non_string_nulls():
    dtype = STRING_DTYPE(na_object=None)
    values = np.array(['a', None, 'bb'], dtype=dtype)
    strings = StringsInformation()

    with pytest.raises(ValueError, match='length of a null string'):
        STRINGS.str_len(values)
    with pytest.raises(ValueError, match='length of a null string'):
        strings.strings_str_len(values)


@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_na_object_variants_are_typed(na_object):
    values = np.array(['a', na_object, 'bb'],
                      dtype=STRING_DTYPE(na_object=na_object))

    value_type = typeof(values)

    assert value_type.dtype.name.startswith('StringDTypePacket')


@pytest.mark.parametrize('left,right', [
    ('', 'MISSING'),
    ('MISSING', 'NA'),
    ('é', '🙂'),
])
def test_stringdtype_string_na_object_type_includes_sentinel_text(
        left, right):
    missing = typeof(np.array(['a', left],
                              dtype=STRING_DTYPE(na_object=left)))
    na = typeof(np.array(['a', right], dtype=STRING_DTYPE(na_object=right)))

    assert missing.dtype.na_name == left.encode('utf-8')
    assert na.dtype.na_name == right.encode('utf-8')
    assert missing.dtype.name != na.dtype.name


@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_str_len_na_object_variants_match_numpy(na_object):
    values = np.array(['a', na_object, '', 'MISSING', 'aa'],
                      dtype=STRING_DTYPE(na_object=na_object))
    strings = StringsInformation()

    if na_object_is_string_like(na_object):
        assert_same(strings.strings_str_len, STRINGS.str_len, values)
    else:
        assert_same_exception(strings.strings_str_len, STRINGS.str_len, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_str_len', STRINGS.str_len),
    ('strings_isalpha', STRINGS.isalpha),
    ('strings_isupper', STRINGS.isupper),
])
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_zero_dimensional_na_object_variants_match_numpy(
        impl_name, baseline, na_object):
    value = np.array(na_object, dtype=STRING_DTYPE(na_object=na_object))
    strings = StringsInformation()
    implementation = getattr(strings, impl_name)

    if impl_name == 'strings_str_len':
        if na_object_is_string_like(na_object):
            assert_same(implementation, baseline, value)
        else:
            assert_same_exception(implementation, baseline, value)
    elif na_object is None or (
            not na_object_is_string_like(na_object)
            and not na_object_is_nan_like(na_object)):
        assert_same_exception(implementation, baseline, value)
    else:
        assert_same(implementation, baseline, value)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_PREDICATES)
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_predicate_na_object_variants_match_numpy(
        impl_name, baseline, na_object):
    values = np.array(['a', na_object, '', 'MISSING', 'aa'],
                      dtype=STRING_DTYPE(na_object=na_object))
    strings = StringsInformation()
    implementation = getattr(strings, impl_name)

    if na_object_is_string_like(na_object) \
            or na_object_is_nan_like(na_object):
        assert_same(implementation, baseline, values)
    else:
        assert_same_exception(implementation, baseline, values)


@pytest.mark.parametrize('impl_name, baseline',
                         STRINGDTYPE_AFFIX_SEARCH_METHODS)
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_na_object_affix_search_arrays_match_numpy(
        impl_name, baseline, na_object):
    values = np.array(['a', na_object, '', 'MISSING', 'aa'],
                      dtype=STRING_DTYPE(na_object=na_object))
    patterns = np.array(['a', 'a', na_object, '', 'z'],
                        dtype=STRING_DTYPE(na_object=na_object))
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, values, values)
    assert_same_outcome(implementation, baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline',
                         STRINGDTYPE_AFFIX_SEARCH_METHODS)
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_zero_dimensional_na_object_affix_search_match_numpy(
        impl_name, baseline, na_object):
    dtype = STRING_DTYPE(na_object=na_object)
    value_null = np.array(na_object, dtype=dtype)
    pattern_null = np.array(na_object, dtype=dtype)
    value_text = np.array('abcabc', dtype=dtype)
    pattern_text = np.array('a', dtype=dtype)
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, value_null, pattern_text)
    assert_same_outcome(implementation, baseline, value_text, pattern_null)
    assert_same_outcome(implementation, baseline, value_null, pattern_null)


@pytest.mark.parametrize('impl_name, baseline',
                         STRINGDTYPE_AFFIX_SEARCH_METHODS)
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_na_object_mixed_default_array_affix_search_match_numpy(
        impl_name, baseline, na_object):
    default = stringdtype_array(['a', 'x', '', 'MISSING', 'aa'])
    sentinel = np.array(['a', na_object, '', 'MISSING', 'aa'],
                        dtype=STRING_DTYPE(na_object=na_object))
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, default, sentinel)
    assert_same_outcome(implementation, baseline, sentinel, default)


@pytest.mark.parametrize('impl_name, baseline',
                         STRINGDTYPE_AFFIX_SEARCH_METHODS)
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_na_object_mixed_unicode_array_affix_search_match_numpy(
        impl_name, baseline, na_object):
    sentinel = np.array(['a', na_object, '', 'MISSING', 'aa'],
                        dtype=STRING_DTYPE(na_object=na_object))
    unicode = np.array(['a', 'a', '', 'MISSING', 'z'], dtype='U16')
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, sentinel, unicode)
    assert_same_outcome(implementation, baseline, unicode, sentinel)


@pytest.mark.parametrize('impl_name, baseline',
                         STRINGDTYPE_AFFIX_SEARCH_METHODS)
@pytest.mark.parametrize('left_na, right_na', [
    (None, np.nan),
    (np.nan, np.float32(np.nan)),
    (np.nan, complex(np.nan, 0)),
    ('', 'MISSING'),
    ('é', 'MISSING'),
    (None, 0),
])
def test_stringdtype_incompatible_na_object_affix_search_match_numpy(
        impl_name, baseline, left_na, right_na):
    left = np.array(['a', left_na], dtype=STRING_DTYPE(na_object=left_na))
    right = np.array(['a', right_na], dtype=STRING_DTYPE(na_object=right_na))

    assert_same_exception(strings_impl(impl_name), baseline, left, right)


@pytest.mark.parametrize('impl_name, baseline',
                         STRINGDTYPE_AFFIX_SEARCH_METHODS)
@pytest.mark.parametrize('left_na, right_na', [
    (0, False),
    (0, np.int64(0)),
    (False, np.bool_(False)),
    (1, np.int64(1)),
    (1, 1.0),
    (1, complex(1, 0)),
    (np.inf, np.float64(np.inf)),
    (complex(np.inf, 0), np.complex128(complex(np.inf, 0))),
    (np.nan, np.float64(np.nan)),
])
def test_stringdtype_compatible_numeric_na_object_affix_search_match_numpy(
        impl_name, baseline, left_na, right_na):
    left = np.array(['a', left_na, ''], dtype=STRING_DTYPE(na_object=left_na))
    right = np.array(['a', right_na, ''],
                     dtype=STRING_DTYPE(na_object=right_na))

    assert_same_outcome(strings_impl(impl_name), baseline, left, right)


@pytest.mark.parametrize('impl_name, baseline',
                         STRINGDTYPE_AFFIX_SEARCH_METHODS)
@pytest.mark.parametrize('na_object', [
    None, np.nan, 'MISSING', 0,
])
@pytest.mark.parametrize('scalar', ['', 'a'])
@pytest.mark.parametrize('scalar_factory', [
    pytest.param(lambda value: value, id='python-str'),
    pytest.param(np.str_, id='numpy-str-scalar'),
    pytest.param(lambda value: np.array(value, dtype='U16'),
                 id='zero-dimensional-unicode-array'),
])
def test_stringdtype_na_object_unicode_scalar_affix_search_match_numpy(
        impl_name, baseline, na_object, scalar, scalar_factory):
    values = np.array(['a', na_object, '', 'MISSING', 'aa'],
                      dtype=STRING_DTYPE(na_object=na_object))
    scalar = scalar_factory(scalar)
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, values, scalar)
    assert_same_outcome(implementation, baseline, scalar, values)


@pytest.mark.parametrize('impl_name, baseline',
                         STRINGDTYPE_AFFIX_SEARCH_METHODS)
@pytest.mark.parametrize('na_object', [
    None, np.nan, 'MISSING', 0,
])
def test_stringdtype_na_object_affix_search_slices_match_numpy(
        impl_name, baseline, na_object):
    values = np.array(['abcabc', na_object, '', 'MISSING', 'aa'],
                      dtype=STRING_DTYPE(na_object=na_object))
    patterns = np.array(['b', '', na_object, 'S', 'a'],
                        dtype=STRING_DTYPE(na_object=na_object))
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, values, patterns, 1, 4)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_str_len', STRINGS.str_len),
    ('strings_isalpha', STRINGS.isalpha),
])
@pytest.mark.parametrize('na_object', [None, np.nan, 'MISSING', 0])
def test_stringdtype_na_object_strided_unary_matches_numpy(
        impl_name, baseline, na_object):
    values = np.array(
        ['a', 'x', na_object, 'y', 'MISSING', 'z', 'aa'],
        dtype=STRING_DTYPE(na_object=na_object),
    )[::-2]
    implementation = strings_impl(impl_name)

    assert_same_view_outcome(implementation, baseline, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    *STRINGDTYPE_AFFIX_SEARCH_METHODS,
])
@pytest.mark.parametrize('na_object', [None, np.nan, 'MISSING', 0])
def test_stringdtype_na_object_strided_binary_matches_numpy(
        impl_name, baseline, na_object):
    values = np.array(
        ['a', 'x', na_object, 'y', 'MISSING', 'z', 'aa'],
        dtype=STRING_DTYPE(na_object=na_object),
    )[::2]
    patterns = np.array(
        ['aa', 'z', '', 'y', na_object, 'x', 'a'],
        dtype=STRING_DTYPE(na_object=na_object),
    )[::-2]
    implementation = strings_impl(impl_name)

    assert_same_view_outcome(implementation, baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_na_object_array_comparisons_match_numpy(
        impl_name, baseline, na_object):
    values = np.array(['a', na_object, '', 'MISSING', 'aa'],
                      dtype=STRING_DTYPE(na_object=na_object))
    patterns = np.array(['a', 'a', na_object, '', 'z'],
                        dtype=STRING_DTYPE(na_object=na_object))
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, values, values)
    assert_same_outcome(implementation, baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_na_object_mixed_default_array_comparisons_match_numpy(
        impl_name, baseline, na_object):
    default = stringdtype_array(['a', 'x', '', 'MISSING', 'aa'])
    sentinel = np.array(['a', na_object, '', 'MISSING', 'aa'],
                        dtype=STRING_DTYPE(na_object=na_object))
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, default, sentinel)
    assert_same_outcome(implementation, baseline, sentinel, default)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
@pytest.mark.parametrize('left_na, right_na', [
    (None, np.nan),
    (np.nan, np.float32(np.nan)),
    (np.nan, complex(np.nan, 0)),
    ('', 'MISSING'),
    ('é', 'MISSING'),
    (None, 0),
])
def test_stringdtype_incompatible_na_object_comparisons_match_numpy(
        impl_name, baseline, left_na, right_na):
    left = np.array(['a', left_na], dtype=STRING_DTYPE(na_object=left_na))
    right = np.array(['a', right_na], dtype=STRING_DTYPE(na_object=right_na))

    assert_same_exception(strings_impl(impl_name), baseline, left, right)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
@pytest.mark.parametrize('left_na, right_na', [
    (0, False),
    (0, np.int64(0)),
    (False, np.bool_(False)),
    (1, np.int64(1)),
    (1, 1.0),
    (1, complex(1, 0)),
    (np.inf, np.float64(np.inf)),
    (complex(np.inf, 0), np.complex128(complex(np.inf, 0))),
    (np.nan, np.float64(np.nan)),
])
def test_stringdtype_compatible_numeric_na_object_comparisons_match_numpy(
        impl_name, baseline, left_na, right_na):
    left = np.array(['a', left_na, ''], dtype=STRING_DTYPE(na_object=left_na))
    right = np.array(['a', right_na, ''], dtype=STRING_DTYPE(na_object=right_na))

    assert_same_outcome(strings_impl(impl_name), baseline, left, right)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
def test_stringdtype_na_object_mixed_unicode_array_comparisons_match_numpy(
        impl_name, baseline, na_object):
    sentinel = np.array(['a', na_object, '', 'MISSING', 'aa'],
                        dtype=STRING_DTYPE(na_object=na_object))
    unicode = np.array(['a', 'a', '', 'MISSING', 'z'], dtype='U16')
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, sentinel, unicode)
    assert_same_outcome(implementation, baseline, unicode, sentinel)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
@pytest.mark.parametrize('na_object', NA_OBJECT_VARIANTS)
@pytest.mark.parametrize('scalar', ['', 'a', 'MISSING', 'é'])
@pytest.mark.parametrize('scalar_factory', [
    pytest.param(lambda value: value, id='python-str'),
    pytest.param(np.str_, id='numpy-str-scalar'),
    pytest.param(lambda value: np.array(value, dtype='U16'),
                 id='zero-dimensional-unicode-array'),
])
def test_stringdtype_na_object_unicode_scalar_comparisons_match_numpy(
        impl_name, baseline, na_object, scalar, scalar_factory):
    values = np.array(['a', na_object, '', 'MISSING', 'aa'],
                      dtype=STRING_DTYPE(na_object=na_object))
    scalar = scalar_factory(scalar)
    implementation = strings_impl(impl_name)

    assert_same_outcome(implementation, baseline, values, scalar)
    assert_same_outcome(implementation, baseline, scalar, values)


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


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_ORDER_COMPARISONS)
def test_stringdtype_array_order_matches_numpy(impl_name, baseline):
    strings = StringsComparisonOperators()
    left = stringdtype_array([
        'a', 'b', 'a', 'aa', '', 'é', 'α', '🙂', '一',
        'a\x00', 'a\x00x', 'a\x00y', 'a\x00x', 'ab\x00',
        'ab\x00x', '\x00a', '\x00', '\x00x', 'é\x00x',
        'a\x01x',
    ])
    right = stringdtype_array([
        'a', 'a', 'aa', 'a', 'a', 'e', 'β', '🙃', '二',
        'a', 'a\x00y', 'a\x00x', 'a\x00', 'ab\x00x',
        'ab\x00y', '\x00b', '', '\x01x', 'é',
        'a\x00x',
    ])

    assert_same(getattr(strings, impl_name), baseline, left, right)
    assert_same(getattr(strings, impl_name), baseline, right, left)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_ORDER_COMPARISONS)
def test_stringdtype_array_order_same_array_matches_numpy(
        impl_name, baseline):
    strings = StringsComparisonOperators()
    values = stringdtype_array(['a', 'é', '🙂', 'a\x00x', '\x00abc'])

    assert_same(getattr(strings, impl_name), baseline, values, values)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_ORDER_COMPARISONS)
def test_stringdtype_array_order_empty_arrays_match_numpy(
        impl_name, baseline):
    strings = StringsComparisonOperators()
    left = stringdtype_array([])
    right = stringdtype_array([])

    assert_same(getattr(strings, impl_name), baseline, left, right)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_ORDER_COMPARISONS)
def test_stringdtype_array_order_readonly_arrays_match_numpy(
        impl_name, baseline):
    strings = StringsComparisonOperators()
    left = stringdtype_array(['a', 'é', '🙂', 'a\x00'])
    right = stringdtype_array(['b', 'e', '🙃', 'a'])
    left.flags.writeable = False
    right.flags.writeable = False

    assert_same(getattr(strings, impl_name), baseline, left, right)


def test_stringdtype_array_equal_shape_mismatch():
    strings = StringsComparisonOperators()
    left = stringdtype_array(['a', 'b', 'c'])
    right = stringdtype_array(['a', 'b'])

    with pytest.raises(ValueError, match='shape mismatch'):
        strings.strings_equal(left, right)
    with pytest.raises(ValueError, match='shape mismatch'):
        strings.strings_not_equal(left, right)


@pytest.mark.parametrize('impl_name', [
    name for name, _ in STRINGDTYPE_ORDER_COMPARISONS
])
def test_stringdtype_array_order_shape_mismatch(impl_name):
    strings = StringsComparisonOperators()
    left = stringdtype_array(['a', 'b', 'c'])
    right = stringdtype_array(['a', 'b'])

    with pytest.raises(ValueError, match='shape mismatch'):
        getattr(strings, impl_name)(left, right)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
])
def test_stringdtype_array_equal_noncontiguous_arrays_match_numpy(
        impl_name, baseline):
    strings = StringsComparisonOperators()
    left = stringdtype_array(['a', 'x', 'é', 'y', '🙂', 'z'])[::2]
    right = stringdtype_array(['z', '🙂', 'y', 'e', 'x', 'a'])[::-2]

    assert_same_view(getattr(strings, impl_name), baseline, left, right)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
def test_stringdtype_zero_stride_binary_views_match_numpy(
        impl_name, baseline):
    implementation = strings_impl(impl_name)
    values = np.broadcast_to(stringdtype_array(['abcabc']), (4,))
    patterns = np.broadcast_to(stringdtype_array(['a']), (4,))

    assert values.strides == (0,)
    assert patterns.strides == (0,)
    assert_same_view_outcome(implementation, baseline, values, patterns)


@pytest.mark.parametrize('impl_name', [
    name for name, _ in STRINGDTYPE_ORDER_COMPARISONS
])
def test_stringdtype_array_order_noncontiguous_arrays_match_numpy(impl_name):
    strings = StringsComparisonOperators()
    left = stringdtype_array(['a', 'x', 'é', 'y', '🙂', 'z'])[::2]
    right = stringdtype_array(['z', '🙂', 'y', 'e', 'x', 'b'])[::-2]
    baseline = dict(STRINGDTYPE_ORDER_COMPARISONS)[impl_name]

    assert_same_view(getattr(strings, impl_name), baseline, left, right)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
@pytest.mark.parametrize('scalar', ['a', '', 'é', '🙂'])
def test_stringdtype_noncontiguous_comparison_mixed_python_str_matches_numpy(
        impl_name, baseline, scalar):
    strings = StringsComparisonOperators()
    values = stringdtype_array([
        'x', '🙂', 'z', 'é', 'y', '', 'w', 'a',
    ])[::-2]

    assert_same_view(getattr(strings, impl_name), baseline, values, scalar)
    assert_same_view(getattr(strings, impl_name), baseline, scalar, values)


def test_stringdtype_array_equal_rejects_multidimensional_arrays():
    strings = StringsComparisonOperators()
    values = stringdtype_array(['a', 'b', 'c', 'd']).reshape(2, 2)

    with pytest.raises(TypingError, match='one-dimensional arrays'):
        strings.strings_equal(values, values)


@pytest.mark.parametrize('impl_name', [
    name for name, _ in STRINGDTYPE_ORDER_COMPARISONS
])
def test_stringdtype_array_order_rejects_multidimensional_arrays(impl_name):
    strings = StringsComparisonOperators()
    values = stringdtype_array(['a', 'b', 'c', 'd']).reshape(2, 2)

    with pytest.raises(TypingError, match='one-dimensional arrays'):
        getattr(strings, impl_name)(values, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
])
@pytest.mark.parametrize('scalar', [
    'a', 'a\x00', 'a\x00x', '\x00', '', 'é', 'é\x00x', '🙂',
])
def test_stringdtype_array_equal_mixed_python_str_matches_numpy(
        impl_name, baseline, scalar):
    strings = StringsComparisonOperators()
    values = stringdtype_array([
        'a', 'a\x00', 'a\x00\x00', 'a\x00x', '\x00', '\x00\x00',
        '', 'é', 'é\x00x', '🙂',
    ])

    assert_same(getattr(strings, impl_name), baseline, values, scalar)
    assert_same(getattr(strings, impl_name), baseline, scalar, values)


@pytest.mark.parametrize('impl_name', [
    name for name, _ in STRINGDTYPE_ORDER_COMPARISONS
])
@pytest.mark.parametrize('scalar', [
    'a', 'a\x00', 'a\x00x', '\x00', '', 'é', 'é\x00x', '🙂',
])
def test_stringdtype_array_order_mixed_python_str_matches_numpy(
        impl_name, scalar):
    strings = StringsComparisonOperators()
    baseline = dict(STRINGDTYPE_ORDER_COMPARISONS)[impl_name]
    values = stringdtype_array([
        'a', 'b', 'aa', '', 'a\x00', 'a\x00x', 'a\x00y',
        '\x00', '\x00x', 'é', 'é\x00x', '🙂',
    ])

    assert_same(getattr(strings, impl_name), baseline, values, scalar)
    assert_same(getattr(strings, impl_name), baseline, scalar, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
def test_stringdtype_array_comparison_long_mixed_python_str_matches_numpy(
        impl_name, baseline):
    strings = StringsComparisonOperators()
    values = stringdtype_array([
        'a' * 256,
        'b' + 'a' * 255,
        'a' * 7 + 'b' + 'a' * 248,
        'a' * 255 + 'b',
        'a' * 128 + '\x00x',
        'é' * 128,
        'é' * 127 + 'x',
        '🙂' * 64,
        '🙂' * 63 + 'x',
    ])
    scalars = [
        'a' * 256,
        'a' * 7 + 'm' + 'a' * 248,
        'a' * 128 + '\x00z',
        'é' * 128,
        '🙂' * 64,
    ]

    for scalar in scalars:
        assert_same(getattr(strings, impl_name), baseline, values, scalar)
        assert_same(getattr(strings, impl_name), baseline, scalar, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
def test_stringdtype_zero_dimensional_comparison_long_mixed_python_str_matches_numpy(
        impl_name, baseline):
    strings = StringsComparisonOperators()
    value = np.array('a' * 256, dtype=STRING_DTYPE())
    scalars = [
        'a' * 256,
        'a' * 255 + 'b',
        'a' * 128 + '\x00z',
        'é' * 128,
        '🙂' * 64,
    ]

    for scalar in scalars:
        assert_same(getattr(strings, impl_name), baseline, value, scalar)
        assert_same(getattr(strings, impl_name), baseline, scalar, value)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
    *STRINGDTYPE_ORDER_COMPARISONS,
])
@pytest.mark.parametrize('scalar', ['\ud800', 'a\ud800', 'a\x00\ud800'])
def test_stringdtype_array_comparison_mixed_python_str_invalid_unicode_matches_numpy(
        impl_name, baseline, scalar):
    strings = StringsComparisonOperators()
    values = stringdtype_array(['a', 'b'])

    assert_same_exception(getattr(strings, impl_name), baseline, values, scalar)
    assert_same_exception(getattr(strings, impl_name), baseline, scalar, values)


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
def test_stringdtype_array_affix_noncontiguous_arrays_match_numpy(impl_name):
    strings = StringsInformation()
    values = stringdtype_array([
        'abc', 'x', 'éfg', 'y', '🙂abc', 'z',
    ])[::2]
    patterns = stringdtype_array(['z', '🙂', 'y', 'é', 'x', 'a'])[::-2]
    baseline = STRINGS.startswith if impl_name == 'strings_startswith' \
        else STRINGS.endswith

    assert_same_view(getattr(strings, impl_name), baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_AFFIX_METHODS)
def test_stringdtype_array_affix_empty_strided_views_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array(['abc', 'x'])[:0:2]
    patterns = stringdtype_array(['a', 'x'])[:0:2]

    assert values.size == 0
    assert patterns.size == 0
    assert_same_view(getattr(strings, impl_name), baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_AFFIX_METHODS)
@pytest.mark.parametrize('scalar', ['a', '', 'é', '🙂'])
def test_stringdtype_noncontiguous_affix_mixed_python_str_matches_numpy(
        impl_name, baseline, scalar):
    strings = StringsInformation()
    values = stringdtype_array([
        'x', '🙂abc', 'z', 'éfg', 'y', '', 'w', 'abc',
    ])[::-2]

    assert_same_view(getattr(strings, impl_name), baseline, values, scalar)
    assert_same_view(getattr(strings, impl_name), baseline, scalar, values)


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
@pytest.mark.parametrize('scalar', [
    'a', 'a\x00', 'a\x00x', '\x00', '\x00x', '', 'é', '🙂',
])
@pytest.mark.parametrize('args', [(), (0, None), (1, None), (0, -1)])
def test_stringdtype_array_affix_mixed_python_str_matches_numpy(
        impl_name, scalar, args):
    strings = StringsInformation()
    baseline = STRINGS.startswith if impl_name == 'strings_startswith' \
        else STRINGS.endswith
    values = stringdtype_array([
        'a', 'a\x00', 'a\x00x', 'a\x00y', '\x00', '\x00x',
        '', 'éabc', '🙂abc',
    ])

    assert_same(getattr(strings, impl_name), baseline, values, scalar, *args)
    assert_same(getattr(strings, impl_name), baseline, scalar, values, *args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
])
@pytest.mark.parametrize('values, scalar, args', [
    (['a' * 256, 'x' + 'a' * 255, 'a' * 255 + 'x'], 'a' * 128, ()),
    (['é' * 128, 'ê' + 'é' * 127, 'é' * 127 + 'ê'], 'é' * 64, ()),
    (['🙂' * 64, '🙃' + '🙂' * 63, '🙂' * 63 + '🙃'], '🙂' * 32, ()),
    (['a' * 128 + '\x00x', 'a' * 128 + '\x00y'],
     'a' * 128 + '\x00z', ()),
    (['a' * 64 + '\x00', 'a' * 64, 'a' * 64 + '\x00x'],
     'a' * 64 + '\x00\x00', ()),
    (['x' + 'a' * 128 + 'x', 'x' + 'é' * 64 + 'x'],
     'a' * 32, (1, -1)),
])
def test_stringdtype_array_affix_long_mixed_python_str_matches_numpy(
        impl_name, baseline, values, scalar, args):
    strings = StringsInformation()
    values = stringdtype_array(values)

    assert_same(getattr(strings, impl_name), baseline,
                values, scalar, *args)
    assert_same(getattr(strings, impl_name), baseline,
                scalar, values, *args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_startswith', STRINGS.startswith),
    ('strings_endswith', STRINGS.endswith),
])
@pytest.mark.parametrize('scalar', ['\ud800', 'a\ud800', 'a\x00\ud800'])
def test_stringdtype_array_affix_mixed_python_str_invalid_unicode_matches_numpy(
        impl_name, baseline, scalar):
    strings = StringsInformation()
    values = stringdtype_array(['a', 'b'])

    assert_same_exception(getattr(strings, impl_name), baseline, values, scalar)
    assert_same_exception(getattr(strings, impl_name), baseline, scalar, values)


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
@pytest.mark.parametrize('args', [
    (),
    (1, -1),
    (10, None),
])
def test_stringdtype_array_search_all_nul_patterns_match_numpy(
        impl_name, baseline, args):
    strings = StringsInformation()
    values = stringdtype_array([
        'abc', '', '\x00', '\x00\x00', 'a\x00b', '🙂',
    ])
    patterns = stringdtype_array([
        '\x00', '\x00', '\x00', '\x00\x00', '\x00\x00', '\x00',
    ])

    assert_same(getattr(strings, impl_name), baseline,
                values, patterns, *args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
@pytest.mark.parametrize('args, values, patterns', [
    (
        (),
        [
            'abcabc', 'éfgé', '🙂a🙂', 'a\x00bc\x00bc', '\x00abc',
            '', 'aaaa', 'a🙂a🙂',
        ],
        [
            'bc', 'é', '🙂', '\x00bc', '\x00a',
            '', 'aa', '🙂a',
        ],
    ),
    (
        (1, None),
        ['xabc', 'xéfg', 'x🙂a', 'xa\x00bc', 'x\x00abc',
         'xaaaa', 'xa🙂a'],
        ['ab', 'é', '🙂', 'a\x00', '\x00a', 'aa', '🙂a'],
    ),
    (
        (0, -1),
        ['abcx', 'éfgx', '🙂ax', 'a\x00bcx', '\x00abcx',
         'aaaax', 'a🙂ax'],
        ['bc', 'é', '🙂', 'a\x00', '\x00a', 'aa', '🙂a'],
    ),
])
def test_stringdtype_array_index_matches_numpy(
        impl_name, baseline, args, values, patterns):
    strings = StringsInformation()

    assert_same(getattr(strings, impl_name), baseline,
                stringdtype_array(values), stringdtype_array(patterns), *args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
@pytest.mark.parametrize('values, patterns', [
    (
        ['abc', 'def', '🙂a', 'a\x00b'],
        ['a', 'z', '🙂', '\x00b\x00'],
    ),
    (
        ['xxx', 'abc', 'def', '🙂a'],
        ['z', 'a', 'd', '🙂'],
    ),
])
def test_stringdtype_array_index_not_found_matches_numpy(
        impl_name, baseline, values, patterns):
    strings = StringsInformation()
    values = stringdtype_array(values)
    patterns = stringdtype_array(patterns)

    with pytest.raises(ValueError, match='substring not found'):
        baseline(values, patterns)
    with pytest.raises(ValueError, match='substring not found'):
        getattr(strings, impl_name)(values, patterns)

    assert_same(strings.strings_find, STRINGS.find, values, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
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
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
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
    'strings_index',
    'strings_rindex',
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
    'strings_index',
    'strings_rindex',
])
def test_stringdtype_array_search_noncontiguous_arrays_match_numpy(impl_name):
    strings = StringsInformation()
    values = stringdtype_array([
        'abcabc', 'x', 'éfgé', 'y', '🙂a🙂', 'z',
    ])[::2]
    patterns = stringdtype_array(['z', '🙂', 'y', 'é', 'x', 'bc'])[::-2]
    baseline = dict(STRINGDTYPE_SEARCH_METHODS)[impl_name]

    assert_same_view(getattr(strings, impl_name), baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_SEARCH_METHODS)
def test_stringdtype_array_search_empty_strided_views_match_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array(['abc', 'x'])[:0:2]
    patterns = stringdtype_array(['a', 'x'])[:0:2]

    assert values.size == 0
    assert patterns.size == 0
    assert_same_view(getattr(strings, impl_name), baseline, values, patterns)


@pytest.mark.parametrize('impl_name, baseline', STRINGDTYPE_SEARCH_METHODS)
@pytest.mark.parametrize('scalar', ['a', '', 'é', '🙂'])
def test_stringdtype_noncontiguous_search_mixed_python_str_matches_numpy(
        impl_name, baseline, scalar):
    strings = StringsInformation()
    values = stringdtype_array([
        'x', '🙂a🙂', 'z', 'éfgé', 'y', '', 'w', 'abcabc',
    ])[::-2]

    assert_same_view_outcome(
        getattr(strings, impl_name), baseline, values, scalar)
    assert_same_view_outcome(
        getattr(strings, impl_name), baseline, scalar, values)


@pytest.mark.parametrize('impl_name', [
    'strings_find',
    'strings_rfind',
    'strings_count',
    'strings_index',
    'strings_rindex',
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
@pytest.mark.parametrize('scalar', [
    'a', 'a\x00', 'a\x00x', '\x00', '\x00x', '', 'é', '🙂',
])
@pytest.mark.parametrize('args', [(), (0, None), (1, None), (0, -1)])
def test_stringdtype_array_search_mixed_python_str_matches_numpy(
        impl_name, scalar, args):
    strings = StringsInformation()
    baseline = {
        'strings_find': STRINGS.find,
        'strings_rfind': STRINGS.rfind,
        'strings_count': STRINGS.count,
    }[impl_name]
    values = stringdtype_array([
        'abcabc', 'a\x00bc', 'a\x00x', 'a\x00y', '\x00', '\x00x',
        '', 'éfgé', '🙂a🙂',
    ])

    assert_same(getattr(strings, impl_name), baseline, values, scalar, *args)
    assert_same(getattr(strings, impl_name), baseline, scalar, values, *args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
])
@pytest.mark.parametrize('values, scalar, args', [
    (['a' * 128 + 'z' * 128, 'x' + 'a' * 127 + 'z' * 128],
     'a' * 64, ()),
    (['z' * 128 + 'a' * 128, 'z' * 128 + 'a' * 127 + 'x'],
     'a' * 64, ()),
    (['z' * 256, 'y' * 256], 'a' * 64, ()),
    (['é' * 128, 'ê' + 'é' * 127], 'é' * 32, ()),
    (['a' * 128 + '\x00x', 'a' * 128 + '\x00y'],
     'a' * 128 + '\x00z', ()),
    (['x' + 'a' * 128 + 'x', 'x' + 'é' * 64 + 'x'],
     'a' * 32, (1, -1)),
])
def test_stringdtype_array_search_long_mixed_python_str_matches_numpy(
        impl_name, baseline, values, scalar, args):
    strings = StringsInformation()
    values = stringdtype_array(values)

    assert_same(getattr(strings, impl_name), baseline,
                values, scalar, *args)
    assert_same(getattr(strings, impl_name), baseline,
                scalar, values, *args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
@pytest.mark.parametrize('args', [(), (0, None)])
def test_stringdtype_array_index_mixed_python_str_matches_numpy(
        impl_name, baseline, args):
    strings = StringsInformation()
    values = stringdtype_array(['abcabc', 'a', 'ba', 'a🙂', 'éa'])
    patterns = stringdtype_array(['a', 'bc', 'abc', '', 'é'])
    value = 'abcabcé🙂'
    pattern = 'a'

    assert_same(getattr(strings, impl_name), baseline, values, pattern, *args)
    assert_same(getattr(strings, impl_name), baseline, value, patterns, *args)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
def test_stringdtype_array_index_long_mixed_python_str_matches_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array([
        'a' * 128 + 'z', 'z' + 'a' * 128, 'x' + 'a' * 64 + 'y',
    ])
    patterns = stringdtype_array(['a' * 64, 'é' * 32, '🙂' * 16, ''])
    value = 'x' + 'a' * 128 + 'é' * 64 + '🙂' * 32

    assert_same(getattr(strings, impl_name), baseline, values, 'a' * 32)
    assert_same(getattr(strings, impl_name), baseline, value, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
def test_stringdtype_array_index_mixed_python_str_not_found_matches_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array(['abc', 'def'])
    patterns = stringdtype_array(['a', 'z'])

    assert_same_exception(getattr(strings, impl_name), baseline, values, 'z')
    assert_same_exception(getattr(strings, impl_name), baseline, 'abc',
                          patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
def test_stringdtype_array_index_long_mixed_python_str_not_found_matches_numpy(
        impl_name, baseline):
    strings = StringsInformation()
    values = stringdtype_array(['a' * 128, 'b' * 128])
    patterns = stringdtype_array(['a' * 64, 'z' * 64])

    assert_same_exception(getattr(strings, impl_name), baseline,
                          values, 'z' * 64)
    assert_same_exception(getattr(strings, impl_name), baseline,
                          'a' * 128, patterns)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
])
@pytest.mark.parametrize('scalar', ['\ud800', 'a\ud800', 'a\x00\ud800'])
def test_stringdtype_array_search_mixed_python_str_invalid_unicode_matches_numpy(
        impl_name, baseline, scalar):
    strings = StringsInformation()
    values = stringdtype_array(['a', 'b'])

    assert_same_exception(getattr(strings, impl_name), baseline, values, scalar)
    assert_same_exception(getattr(strings, impl_name), baseline, scalar, values)


@pytest.mark.parametrize('impl_name, baseline', [
    ('strings_find', STRINGS.find),
    ('strings_rfind', STRINGS.rfind),
    ('strings_count', STRINGS.count),
    ('strings_index', STRINGS.index),
    ('strings_rindex', STRINGS.rindex),
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
