"""Tests for NumPy strings comparison overloads."""

import numpy as np
import pytest

from charex.tests.definitions import StringsComparisonOperators
from charex.tests.support import assert_same


STRINGS = getattr(np, 'strings', None)
pytestmark = pytest.mark.skipif(
    STRINGS is None,
    reason='np.strings is only available on NumPy 2.x',
)

COMPARE_FUNCS = [] if STRINGS is None else [
    ('equal', 'strings_equal', STRINGS.equal),
    ('not_equal', 'strings_not_equal', STRINGS.not_equal),
    ('greater_equal', 'strings_greater_equal', STRINGS.greater_equal),
    ('greater', 'strings_greater', STRINGS.greater),
    ('less', 'strings_less', STRINGS.less),
    ('less_equal', 'strings_less_equal', STRINGS.less_equal),
]

EQUAL_FUNCS = [] if STRINGS is None else [
    ('strings_equal', STRINGS.equal),
    ('strings_not_equal', STRINGS.not_equal),
]

COMPARISON_CASES = [
    ('abc', 'abc'),
    ('ab ', 'ab'),
    (b'abc', b'abc'),
    (b'ab ', b'ab'),
    ('ab', np.array(['abc', 'ab', 'a'], dtype='U3')),
    ('abc', np.array(['ab', 'abc', 'abcd'], dtype='U4')),
    (np.array('ab', dtype='U2'), np.array(['abc', 'ab', 'a'],
                                          dtype='U3')),
    (np.array(['abc', 'ab', 'a'], dtype='U3'), 'ab'),
    (np.array(['abc', 'ab', 'a'], dtype='U3'), np.array('ab', dtype='U2')),
    (b'ab', np.array([b'abc', b'ab', b'a'], dtype='S3')),
    (np.array(b'ab', dtype='S2'), np.array([b'abc', b'ab', b'a'],
                                           dtype='S3')),
    (np.array([b'abc', b'ab', b'a'], dtype='S3'), b'ab'),
    (np.array([b'abc', b'ab', b'a'], dtype='S3'), np.array(b'ab',
                                                           dtype='S2')),
    (np.array(['abc ', 'abc', 'abc\x00', 'abc\x00x', 'abcx'],
              dtype='U5'), 'abc'),
    (np.array([b'abc ', b'abc', b'abc\x00', b'abc\x00x', b'abcx'],
              dtype='S5'), b'abc'),
]


@pytest.mark.parametrize('_, impl_name, baseline', COMPARE_FUNCS)
@pytest.mark.parametrize('left, right', COMPARISON_CASES)
def test_strings_comparison_matches_numpy(_, impl_name, baseline,
                                          left, right):
    strings = StringsComparisonOperators()
    assert_same(getattr(strings, impl_name), baseline, left, right)


@pytest.mark.parametrize(
    'left, right, expected',
    [
        (np.array(['abc ', 'abc', 'abc\x00', 'abcx'], dtype='U4'),
         'abc', [False, True, True, False]),
        (np.array([b'abc ', b'abc', b'abc\x00', b'abcx'], dtype='S4'),
         b'abc', [False, True, True, False]),
    ],
)
def test_strings_equal_does_not_rstrip_like_char(left, right, expected):
    strings = StringsComparisonOperators()
    assert_same(strings.strings_equal, STRINGS.equal, left, right)
    np.testing.assert_array_equal(strings.strings_equal(left, right),
                                  np.array(expected))


@pytest.mark.parametrize('impl_name, baseline', EQUAL_FUNCS)
@pytest.mark.parametrize('kind, width', [('U', 5), ('U', 12), ('U', 31),
                                         ('S', 5), ('S', 12), ('S', 31)])
def test_sub32_same_width_equality_matches_numpy(impl_name, baseline,
                                                 kind, width):
    strings = StringsComparisonOperators()
    dtype = f'{kind}{width}'
    encode = (lambda value: value) if kind == 'U' \
        else (lambda value: value.encode())
    prefix = 'a' * (width - 2)
    last_prefix = 'b' * (width - 1)
    left_values = [
        'a' * width,
        'x' + 'a' * (width - 1),
        last_prefix + 'x',
        prefix + ' \t',
        prefix + '\x00',
        'a\x00x',
    ]
    right_values = [
        'a' * width,
        'y' + 'a' * (width - 1),
        last_prefix + 'y',
        prefix,
        prefix,
        'a',
    ]
    left = np.array([encode(value) for value in left_values], dtype=dtype)
    right = np.array([encode(value) for value in right_values], dtype=dtype)

    assert_same(getattr(strings, impl_name), baseline, left, right)


def test_strings_overload_does_not_hijack_numeric_equal():
    strings = StringsComparisonOperators()
    left = np.arange(6)
    right = np.array([0, 2, 2, 4, 4, 6])

    assert_same(strings.numpy_equal, np.equal, left, right)


@pytest.mark.parametrize('_, impl_name, baseline', COMPARE_FUNCS)
def test_strings_numeric_fallback_matches_numpy(_, impl_name, baseline):
    strings = StringsComparisonOperators()
    left = np.arange(6)
    right = np.array([0, 2, 2, 4, 4, 6])

    assert_same(getattr(strings, impl_name), baseline, left, right)


@pytest.mark.parametrize('_, impl_name, baseline', COMPARE_FUNCS)
@pytest.mark.parametrize(
    'left, right',
    [
        (np.array(['a'], dtype='U1'), np.array([b'a'], dtype='S1')),
        (np.array([b'a'], dtype='S1'), 'a'),
        (np.array(['a'], dtype='U1'), b'a'),
    ],
)
def test_strings_mixed_bytes_unicode_rejected(_, impl_name, baseline,
                                              left, right):
    strings = StringsComparisonOperators()

    with pytest.raises(Exception):
        baseline(left, right)
    with pytest.raises(Exception):
        getattr(strings, impl_name)(left, right)
