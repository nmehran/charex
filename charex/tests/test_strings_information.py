"""Tests for NumPy strings information overloads."""

import numpy as np
import pytest

from charex.tests.definitions import StringsInformation
from charex.tests.support import (
    assert_same, assert_same_exception, assert_same_view,
    assert_same_view_outcome,
)


STRINGS = getattr(np, 'strings', None)
pytestmark = pytest.mark.skipif(
    STRINGS is None,
    reason='np.strings is only available on NumPy 2.x',
)

OCCURRENCE_FUNCS = [] if STRINGS is None else [
    ('count', 'strings_count', STRINGS.count),
    ('find', 'strings_find', STRINGS.find),
    ('rfind', 'strings_rfind', STRINGS.rfind),
    ('startswith', 'strings_startswith', STRINGS.startswith),
    ('endswith', 'strings_endswith', STRINGS.endswith),
]

INDEX_FUNCS = [] if STRINGS is None else [
    ('index', 'strings_index', STRINGS.index),
    ('rindex', 'strings_rindex', STRINGS.rindex),
]

PROPERTY_FUNCS = [] if STRINGS is None else [
    ('str_len', 'strings_str_len', STRINGS.str_len),
    ('isalpha', 'strings_isalpha', STRINGS.isalpha),
    ('isalnum', 'strings_isalnum', STRINGS.isalnum),
    ('isdigit', 'strings_isdigit', STRINGS.isdigit),
    ('islower', 'strings_islower', STRINGS.islower),
    ('isspace', 'strings_isspace', STRINGS.isspace),
    ('istitle', 'strings_istitle', STRINGS.istitle),
    ('isupper', 'strings_isupper', STRINGS.isupper),
]

UNICODE_ONLY_PROPERTY_FUNCS = [] if STRINGS is None else [
    ('isdecimal', 'strings_isdecimal', STRINGS.isdecimal),
    ('isnumeric', 'strings_isnumeric', STRINGS.isnumeric),
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
def test_strings_occurrence_matches_numpy(_, impl_name, baseline, args):
    strings = StringsInformation()
    assert_same(getattr(strings, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline',
                         OCCURRENCE_FUNCS + INDEX_FUNCS)
@pytest.mark.parametrize('args', STRIDED_OCCURRENCE_CASES)
def test_strings_occurrence_strided_arrays_match_numpy(
        _, impl_name, baseline, args):
    strings = StringsInformation()
    assert_same_view_outcome(getattr(strings, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline',
                         OCCURRENCE_FUNCS + INDEX_FUNCS)
@pytest.mark.parametrize('args', NONE_START_CASES)
def test_strings_none_start_rejected_by_numpy(_, impl_name, baseline, args):
    strings = StringsInformation()
    with pytest.raises(Exception):
        baseline(*args)
    with pytest.raises(Exception):
        getattr(strings, impl_name)(*args)


@pytest.mark.parametrize('_, impl_name, baseline', INDEX_FUNCS)
@pytest.mark.parametrize('args', INDEX_CASES)
def test_strings_index_matches_numpy(_, impl_name, baseline, args):
    strings = StringsInformation()
    assert_same(getattr(strings, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline', INDEX_FUNCS)
@pytest.mark.parametrize(
    'args',
    [
        (np.array(['abc', 'xx'], dtype='U3'), 'zz'),
        (np.array([b'abc', b'xx'], dtype='S3'), b'zz'),
        ('abc', 'zz'),
    ],
)
def test_strings_index_not_found_matches_numpy(_, impl_name, baseline, args):
    strings = StringsInformation()
    assert_same_exception(getattr(strings, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline',
                         OCCURRENCE_FUNCS + INDEX_FUNCS)
@pytest.mark.parametrize('args', [
    (np.array(['abc', 'def', 'ghi'], dtype='U3'),
     np.array(['a', 'd'], dtype='U1')),
    (np.array([b'abc', b'def', b'ghi'], dtype='S3'),
     np.array([b'a', b'd'], dtype='S1')),
])
def test_strings_occurrence_shape_mismatch_matches_numpy(
        _, impl_name, baseline, args):
    strings = StringsInformation()
    assert_same_exception(getattr(strings, impl_name), baseline, *args)


@pytest.mark.parametrize('_, impl_name, baseline', PROPERTY_FUNCS)
@pytest.mark.parametrize('values', [PROPERTY_UNICODE, PROPERTY_BYTES])
def test_strings_properties_match_numpy(_, impl_name, baseline, values):
    strings = StringsInformation()
    assert_same(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('_, impl_name, baseline', PROPERTY_FUNCS)
@pytest.mark.parametrize('values', STRIDED_PROPERTY_CASES)
def test_strings_properties_strided_arrays_match_numpy(
        _, impl_name, baseline, values):
    strings = StringsInformation()
    assert_same_view(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('_, impl_name, baseline', PROPERTY_FUNCS)
@pytest.mark.parametrize('value', SCALAR_PROPERTIES)
def test_strings_scalar_properties_match_numpy(_, impl_name, baseline, value):
    strings = StringsInformation()
    assert_same(getattr(strings, impl_name), baseline, value)


@pytest.mark.parametrize('_, impl_name, baseline', PROPERTY_FUNCS)
@pytest.mark.parametrize(
    'values',
    [
        np.array(['ab\x00cd', 'abc\x00', '\x00abc'], dtype='U5'),
        np.array([b'ab\x00cd', b'abc\x00', b'\x00abc'], dtype='S5'),
    ],
)
def test_strings_properties_with_embedded_nulls_match_numpy(
        _, impl_name, baseline, values):
    strings = StringsInformation()
    assert_same(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('_, impl_name, baseline',
                         UNICODE_ONLY_PROPERTY_FUNCS)
def test_strings_unicode_only_properties_match_numpy(
        _, impl_name, baseline):
    strings = StringsInformation()
    assert_same(getattr(strings, impl_name), baseline, PROPERTY_UNICODE)


@pytest.mark.parametrize('_, impl_name, baseline',
                         UNICODE_ONLY_PROPERTY_FUNCS)
@pytest.mark.parametrize('values', STRIDED_UNICODE_PROPERTY_CASES)
def test_strings_unicode_only_properties_strided_arrays_match_numpy(
        _, impl_name, baseline, values):
    strings = StringsInformation()
    assert_same_view(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('_, impl_name, baseline',
                         UNICODE_ONLY_PROPERTY_FUNCS)
@pytest.mark.parametrize('value', SCALAR_UNICODE_PROPERTIES)
def test_strings_scalar_unicode_only_properties_match_numpy(
        _, impl_name, baseline, value):
    strings = StringsInformation()
    assert_same(getattr(strings, impl_name), baseline, value)


@pytest.mark.parametrize('_, impl_name, baseline',
                         UNICODE_ONLY_PROPERTY_FUNCS)
def test_strings_unicode_only_properties_with_embedded_nulls_match_numpy(
        _, impl_name, baseline):
    strings = StringsInformation()
    values = np.array(['ab\x00cd', 'abc\x00', '\x00abc'], dtype='U5')
    assert_same(getattr(strings, impl_name), baseline, values)


@pytest.mark.parametrize('_, impl_name, baseline',
                         UNICODE_ONLY_PROPERTY_FUNCS)
@pytest.mark.parametrize(
    'value',
    [
        PROPERTY_BYTES,
        b'123',
        np.bytes_(b'123'),
        np.array(b'123', dtype='S3'),
    ],
)
def test_strings_unicode_only_properties_reject_bytes(
        _, impl_name, baseline, value):
    strings = StringsInformation()
    with pytest.raises(Exception):
        baseline(value)
    with pytest.raises(Exception):
        getattr(strings, impl_name)(value)
