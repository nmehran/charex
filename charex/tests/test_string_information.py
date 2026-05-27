"""Tests for numpy character information overloads."""

import numpy as np
import pytest
from numba import njit

from charex.tests.definitions import StringInformation
from charex.tests.support import (
    assert_same, assert_same_exception, assert_same_view,
    assert_same_view_outcome,
)


NUMPY_MAJOR = int(np.__version__.split('.')[0])


OCCURRENCE_FUNCS = [
    ('count', 'char_count', np.char.count),
    ('find', 'char_find', np.char.find),
    ('rfind', 'char_rfind', np.char.rfind),
    ('startswith', 'char_startswith', np.char.startswith),
    ('endswith', 'char_endswith', np.char.endswith),
]

INDEX_FUNCS = [
    ('index', 'char_index', np.char.index),
    ('rindex', 'char_rindex', np.char.rindex),
]

PROPERTY_FUNCS = [
    ('str_len', 'char_str_len', np.char.str_len),
    ('isalpha', 'char_isalpha', np.char.isalpha),
    ('isalnum', 'char_isalnum', np.char.isalnum),
    ('isdigit', 'char_isdigit', np.char.isdigit),
    ('islower', 'char_islower', np.char.islower),
    ('isspace', 'char_isspace', np.char.isspace),
    ('istitle', 'char_istitle', np.char.istitle),
    ('isupper', 'char_isupper', np.char.isupper),
]

UNICODE_ONLY_PROPERTY_FUNCS = [
    ('isdecimal', 'char_isdecimal', np.char.isdecimal),
    ('isnumeric', 'char_isnumeric', np.char.isnumeric),
]


@njit(nogil=True, cache=False)
def default_count(a, sub):
    return np.char.count(a, sub)


@njit(nogil=True, cache=False)
def default_find(a, sub):
    return np.char.find(a, sub)


@njit(nogil=True, cache=False)
def default_rfind(a, sub):
    return np.char.rfind(a, sub)


@njit(nogil=True, cache=False)
def default_startswith(a, sub):
    return np.char.startswith(a, sub)


@njit(nogil=True, cache=False)
def default_endswith(a, sub):
    return np.char.endswith(a, sub)


DEFAULT_OCCURRENCE_FUNCS = [
    ('count', default_count, np.char.count),
    ('find', default_find, np.char.find),
    ('rfind', default_rfind, np.char.rfind),
    ('startswith', default_startswith, np.char.startswith),
    ('endswith', default_endswith, np.char.endswith),
]

OCCURRENCE_CASES = [
    (np.array(['abcabc', 'bcxxbc'], dtype='U6'), 'bc'),
    (np.array(['ab\x00cd', 'xx\x00cd'], dtype='U6'), 'cd'),
    (np.array(['abcabc', 'bcxxbc'], dtype='U6'), 'bc', 2, None),
    (np.array(['abcabc', 'bcxxbc'], dtype='U6'), 'bc', -4, -1),
    ('abcabc', 'bc'),
    (b'abcabc', b'bc'),
    (np.bytes_(b'abcabc'), np.bytes_(b'bc')),
    (np.array('abcabc', dtype='U6'), np.array('bc', dtype='U2')),
    (np.array([b'abcabc', b'bcxxbc'], dtype='S6'), b'bc'),
    (np.array([b'ab\x00cd', b'xx\x00cd'], dtype='S6'), b'cd'),
    (np.array(['abc', ''], dtype='U3'), ''),
]

INDEX_CASES = [
    (np.array(['abcabc', 'bcxxbc'], dtype='U6'), 'bc'),
    (np.array(['ab\x00cd', 'xx\x00cd'], dtype='U6'), 'cd'),
    (np.array(['abcabc', 'bcxxbc'], dtype='U6'), 'bc', 2, None),
    ('abcabc', 'bc'),
    (b'abcabc', b'bc'),
    (np.bytes_(b'abcabc'), np.bytes_(b'bc')),
    (np.array('abcabc', dtype='U6'), np.array('bc', dtype='U2')),
    (np.array([b'abcabc', b'bcxxbc'], dtype='S6'), b'bc'),
    (np.array([b'ab\x00cd', b'xx\x00cd'], dtype='S6'), b'cd'),
    (np.array(['abc', ''], dtype='U3'), ''),
]

PROPERTY_UNICODE = np.array(
    [
        'alpha', 'α', '١', 'Ⅷ', '一', 'A', 'a', 'ǅuro', '中A',
        ' ', '\x1c', '\x1f', ''
    ],
    dtype='U5',
)

PROPERTY_BYTES = np.array(
    [b'A', b'z', bytes([0xc0]), bytes([0xaa]), b'1', b' ', b''],
    dtype='S1',
)

SCALAR_PROPERTIES = [
    'alpha', 'α', '١', 'Ⅷ', '一', 'A', 'a', 'ǅuro', '中A',
    ' ', '\x1c', '\x1f', '',
    b'A', b'z', b'1', b' ', b'',
    np.str_('alpha'), np.bytes_(b'A'),
    np.array('alpha', dtype='U5'), np.array(b'A', dtype='S1'),
]

SCALAR_UNICODE_PROPERTIES = [
    'alpha', 'α', '١', 'Ⅷ', '一', 'A', 'a', 'ǅuro', '中A',
    ' ', '\x1c', '\x1f', '',
    np.str_('alpha'), np.array('alpha', dtype='U5'),
]

NONE_START_CASES = [
    (np.array(['abcabc', 'bcxxbc'], dtype='U6'), 'bc', None, None),
    (np.array(['abcabc', 'bcxxbc'], dtype='U6'), 'bc', None, 3),
    (np.array([b'abcabc', b'bcxxbc'], dtype='S6'), b'bc', None, None),
]


STRIDED_OCCURRENCE_CASES = [
    (
        np.array(['abcabc', 'skip', 'xabc', 'skip', 'abcx'], dtype='U6')[::2],
        np.array(['abc', 'skip', 'x', 'skip', 'x'], dtype='U3')[::2],
    ),
    (
        np.array([b'abcabc', b'skip', b'xabc', b'skip', b'abcx'],
                 dtype='S6')[::2],
        np.array([b'abc', b'skip', b'x', b'skip', b'x'],
                 dtype='S3')[::2],
    ),
    (
        np.array(['abcabc', 'skip', 'xabc', 'skip', 'abcx'],
                 dtype='U6')[::-2],
        np.array(['abc', 'skip', 'x', 'skip', 'x'], dtype='U3')[::-2],
    ),
    (
        np.array([b'abcabc', b'skip', b'xabc', b'skip', b'abcx'],
                 dtype='S6')[::-2],
        np.array([b'abc', b'skip', b'x', b'skip', b'x'],
                 dtype='S3')[::-2],
    ),
    (
        np.broadcast_to(np.array(['abcabc'], dtype='U6'), (3,)),
        np.broadcast_to(np.array(['abc'], dtype='U3'), (3,)),
    ),
    (
        np.broadcast_to(np.array([b'abcabc'], dtype='S6'), (3,)),
        np.broadcast_to(np.array([b'abc'], dtype='S3'), (3,)),
    ),
    (
        np.array(['abcabc', 'skip', 'xabc', 'skip', 'abcx'], dtype='U6')[::2],
        'abc',
    ),
    (
        np.array([b'abcabc', b'skip', b'xabc', b'skip', b'abcx'],
                 dtype='S6')[::2],
        b'abc',
    ),
]


STRIDED_PROPERTY_CASES = [
    np.array(['Alpha', 'skip', 'abc\x00x', 'skip', ''], dtype='U6')[::2],
    np.array([b'Alpha', b'skip', b'abc\x00x', b'skip', b''],
             dtype='S6')[::2],
    np.array(['Alpha', 'skip', 'abc\x00x', 'skip', ''], dtype='U6')[::-2],
    np.array([b'Alpha', b'skip', b'abc\x00x', b'skip', b''],
             dtype='S6')[::-2],
    np.broadcast_to(np.array(['Alpha'], dtype='U6'), (3,)),
    np.broadcast_to(np.array([b'Alpha'], dtype='S6'), (3,)),
    np.array(['Alpha', 'skip'], dtype='U6')[:0:2],
    np.array([b'Alpha', b'skip'], dtype='S6')[:0:2],
]


STRIDED_UNICODE_PROPERTY_CASES = [
    np.array(['Alpha', 'skip', '١٢٣', 'skip', 'Ⅷ'], dtype='U6')[::2],
    np.array(['Alpha', 'skip', '١٢٣', 'skip', 'Ⅷ'], dtype='U6')[::-2],
    np.broadcast_to(np.array(['١٢٣'], dtype='U6'), (3,)),
    np.array(['Alpha', 'skip'], dtype='U6')[:0:2],
]


@pytest.mark.parametrize('_, impl_name, baseline', OCCURRENCE_FUNCS)
@pytest.mark.parametrize('args', OCCURRENCE_CASES)
def test_occurrence_matches_numpy(_, impl_name, baseline, args):
    ch = StringInformation()
    assert_same(getattr(ch, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline',
                         OCCURRENCE_FUNCS + INDEX_FUNCS)
@pytest.mark.parametrize('args', STRIDED_OCCURRENCE_CASES)
def test_occurrence_strided_arrays_match_numpy(_, impl_name, baseline, args):
    ch = StringInformation()
    assert_same_view_outcome(getattr(ch, impl_name), baseline, *args)


@pytest.mark.parametrize('_, implementation, baseline', DEFAULT_OCCURRENCE_FUNCS)
def test_occurrence_default_arguments_match_numpy(_, implementation, baseline):
    values = np.array(['abcabc', 'bcxxbc'], dtype='U6')
    assert_same(implementation, baseline, values, 'bc')


@pytest.mark.parametrize('_, impl_name, baseline',
                         OCCURRENCE_FUNCS + INDEX_FUNCS)
@pytest.mark.parametrize('args', NONE_START_CASES)
def test_none_start_matches_numpy_1x(_, impl_name, baseline, args):
    if NUMPY_MAJOR >= 2:
        pytest.skip('NumPy 2 rejects start=None for np.char occurrence APIs')
    ch = StringInformation()
    assert_same(getattr(ch, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline',
                         OCCURRENCE_FUNCS + INDEX_FUNCS)
@pytest.mark.parametrize('args', NONE_START_CASES)
def test_none_start_rejected_by_numpy_2x(_, impl_name, baseline, args):
    if NUMPY_MAJOR < 2:
        pytest.skip('NumPy 1 accepts start=None for np.char occurrence APIs')
    ch = StringInformation()
    with pytest.raises(Exception):
        baseline(*args)
    with pytest.raises(Exception):
        getattr(ch, impl_name)(*args)


@pytest.mark.parametrize('_, impl_name, baseline', INDEX_FUNCS)
@pytest.mark.parametrize('args', INDEX_CASES)
def test_index_matches_numpy(_, impl_name, baseline, args):
    ch = StringInformation()
    assert_same(getattr(ch, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline', INDEX_FUNCS)
@pytest.mark.parametrize(
    'args',
    [
        (np.array(['abc', 'xx'], dtype='U3'), 'zz'),
        (np.array([b'abc', b'xx'], dtype='S3'), b'zz'),
        ('abc', 'zz'),
    ],
)
def test_index_not_found_matches_numpy(_, impl_name, baseline, args):
    ch = StringInformation()
    assert_same_exception(getattr(ch, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline',
                         OCCURRENCE_FUNCS + INDEX_FUNCS)
@pytest.mark.parametrize('args', [
    (np.array(['abc', 'def', 'ghi'], dtype='U3'),
     np.array(['a', 'd'], dtype='U1')),
    (np.array([b'abc', b'def', b'ghi'], dtype='S3'),
     np.array([b'a', b'd'], dtype='S1')),
])
def test_occurrence_shape_mismatch_matches_numpy(_, impl_name, baseline, args):
    ch = StringInformation()
    assert_same_exception(getattr(ch, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline', PROPERTY_FUNCS)
@pytest.mark.parametrize('values', [PROPERTY_UNICODE, PROPERTY_BYTES])
def test_properties_match_numpy(_, impl_name, baseline, values):
    ch = StringInformation()
    assert_same(getattr(ch, impl_name), baseline, values)


@pytest.mark.parametrize('_, impl_name, baseline', PROPERTY_FUNCS)
@pytest.mark.parametrize('values', STRIDED_PROPERTY_CASES)
def test_properties_strided_arrays_match_numpy(_, impl_name, baseline, values):
    ch = StringInformation()
    assert_same_view(getattr(ch, impl_name), baseline, values)


@pytest.mark.parametrize('_, impl_name, baseline', PROPERTY_FUNCS)
@pytest.mark.parametrize('value', SCALAR_PROPERTIES)
def test_scalar_properties_match_numpy(_, impl_name, baseline, value):
    ch = StringInformation()
    assert_same(getattr(ch, impl_name), baseline, value)


@pytest.mark.parametrize('_, impl_name, baseline', PROPERTY_FUNCS)
@pytest.mark.parametrize(
    'values',
    [
        np.array(['ab\x00cd', 'abc\x00', '\x00abc'], dtype='U5'),
        np.array([b'ab\x00cd', b'abc\x00', b'\x00abc'], dtype='S5'),
    ],
)
def test_properties_with_embedded_nulls_match_numpy(_, impl_name, baseline,
                                                   values):
    ch = StringInformation()
    assert_same(getattr(ch, impl_name), baseline, values)


@pytest.mark.parametrize('_, impl_name, baseline', UNICODE_ONLY_PROPERTY_FUNCS)
def test_unicode_only_properties_match_numpy(_, impl_name, baseline):
    ch = StringInformation()
    assert_same(getattr(ch, impl_name), baseline, PROPERTY_UNICODE)


@pytest.mark.parametrize('_, impl_name, baseline', UNICODE_ONLY_PROPERTY_FUNCS)
@pytest.mark.parametrize('values', STRIDED_UNICODE_PROPERTY_CASES)
def test_unicode_only_properties_strided_arrays_match_numpy(
        _, impl_name, baseline, values):
    ch = StringInformation()
    assert_same_view(getattr(ch, impl_name), baseline, values)


@pytest.mark.parametrize('_, impl_name, baseline', UNICODE_ONLY_PROPERTY_FUNCS)
@pytest.mark.parametrize('value', SCALAR_UNICODE_PROPERTIES)
def test_scalar_unicode_only_properties_match_numpy(
        _, impl_name, baseline, value):
    ch = StringInformation()
    assert_same(getattr(ch, impl_name), baseline, value)


@pytest.mark.parametrize('_, impl_name, baseline', UNICODE_ONLY_PROPERTY_FUNCS)
def test_unicode_only_properties_with_embedded_nulls_match_numpy(
        _, impl_name, baseline):
    ch = StringInformation()
    values = np.array(['ab\x00cd', 'abc\x00', '\x00abc'], dtype='U5')
    assert_same(getattr(ch, impl_name), baseline, values)
