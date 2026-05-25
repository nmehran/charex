"""Tests for numpy character comparison overloads."""

import numpy as np
import pytest

from charex.tests.definitions import ComparisonOperators
from charex.tests.support import assert_same


COMPARE_FUNCS = [
    ('equal', 'char_equal', np.char.equal),
    ('not_equal', 'char_not_equal', np.char.not_equal),
    ('greater_equal', 'char_greater_equal', np.char.greater_equal),
    ('greater', 'char_greater', np.char.greater),
    ('less', 'char_less', np.char.less),
    ('less_equal', 'char_less_equal', np.char.less_equal),
]

COMPARISON_CASES = [
    ('abc', 'abc'),
    (b'abc', b'abc'),
    ('ab', np.array(['abc', 'ab', 'a'], dtype='U3')),
    ('abc', np.array(['ab', 'abc', 'abcd'], dtype='U4')),
    (np.array('ab', dtype='U2'), np.array(['abc', 'ab', 'a'], dtype='U3')),
    (np.array(['abc', 'ab', 'a'], dtype='U3'), 'ab'),
    (np.array(['abc', 'ab', 'a'], dtype='U3'), np.array('ab', dtype='U2')),
    (b'ab', np.array([b'abc', b'ab', b'a'], dtype='S3')),
    (np.array(b'ab', dtype='S2'), np.array([b'abc', b'ab', b'a'], dtype='S3')),
    (np.array([b'abc', b'ab', b'a'], dtype='S3'), b'ab'),
    (np.array([b'abc', b'ab', b'a'], dtype='S3'), np.array(b'ab', dtype='S2')),
    (np.array(['abc ', 'abc', 'abd'], dtype='U4'), 'abc'),
]


@pytest.mark.parametrize('_, impl_name, baseline', COMPARE_FUNCS)
@pytest.mark.parametrize('left, right', COMPARISON_CASES)
def test_comparison_matches_numpy(_, impl_name, baseline, left, right):
    ch = ComparisonOperators()
    assert_same(getattr(ch, impl_name), baseline, left, right)


@pytest.mark.parametrize('_, impl_name, baseline', COMPARE_FUNCS)
@pytest.mark.parametrize(
    'left, right',
    [
        (np.array(['abc ', 'abc\t', 'abc\n', 'abc\r',
                   'abc\f', 'abc\v', 'abc\x00', 'abc\x00x', 'abcx'],
                  dtype='U8'),
         'abc'),
        (np.array([b'abc ', b'abc\t', b'abc\n', b'abc\r',
                   b'abc\f', b'abc\v', b'abc\x00', b'abc\x00x', b'abcx'],
                  dtype='S8'),
         b'abc'),
    ],
)
def test_comparison_rstrips_like_numpy_char(_, impl_name, baseline,
                                            left, right):
    ch = ComparisonOperators()
    assert_same(getattr(ch, impl_name), baseline, left, right)


@pytest.mark.parametrize(
    'impl_name, baseline',
    [
        ('char_equal', np.char.equal),
        ('char_not_equal', np.char.not_equal),
    ],
)
@pytest.mark.parametrize('kind, width', [('U', 5), ('U', 12), ('U', 31),
                                         ('S', 5), ('S', 12), ('S', 31)])
def test_sub32_same_width_equality_matches_numpy(impl_name, baseline,
                                                 kind, width):
    ch = ComparisonOperators()
    dtype = f'{kind}{width}'
    enc = (lambda value: value) if kind == 'U' \
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
    left = np.array([enc(value) for value in left_values], dtype=dtype)
    right = np.array([enc(value) for value in right_values], dtype=dtype)

    assert_same(getattr(ch, impl_name), baseline, left, right)


@pytest.mark.parametrize('cmp', ['==', '!=', '>=', '>', '<', '<='])
@pytest.mark.parametrize('rstrip', [True, False])
@pytest.mark.parametrize(
    'left, right',
    [
        ('ab', np.array(['abc', 'ab', 'a'], dtype='U3')),
        (np.array('ab', dtype='U2'),
         np.array(['abc', 'ab', 'a'], dtype='U3')),
        (b'ab', np.array([b'abc', b'ab', b'a'], dtype='S3')),
        (np.array(b'ab', dtype='S2'),
         np.array([b'abc', b'ab', b'a'], dtype='S3')),
        (np.array(['abc ', 'abc', 'abd'], dtype='U4'), 'abc'),
    ],
)
def test_compare_chararrays_matches_numpy(left, right, cmp, rstrip):
    ch = ComparisonOperators()
    assert_same(ch.char_compare_chararrays,
                np.char.compare_chararrays,
                left, right, cmp, rstrip)


@pytest.mark.parametrize('_, impl_name, baseline', COMPARE_FUNCS)
def test_unicode_comparisons_do_not_mutate_inputs(_, impl_name, baseline):
    ch = ComparisonOperators()
    values = np.array(['abc   ', 'def\t  ', 'ghi'], dtype='U6')
    assert_same(getattr(ch, impl_name), baseline, values, 'abc')


def test_compare_chararrays_does_not_mutate_inputs():
    ch = ComparisonOperators()
    values = np.array(['abc   ', 'def\t  ', 'ghi'], dtype='U6')
    assert_same(ch.char_compare_chararrays,
                np.char.compare_chararrays,
                values, 'abc', '==', True)
