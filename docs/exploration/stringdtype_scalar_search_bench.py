"""Explore Python str scalar spans for mixed StringDType search."""

from pathlib import Path
import argparse
import random
import statistics
import sys
import time

from numba import njit
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import charex  # noqa: F401
from charex.numpy.stringdtype import (
    stringdtype_acquire_allocator,
    stringdtype_count_utf8_data,
    stringdtype_count_unicode_data,
    stringdtype_data_ptr,
    stringdtype_find_utf8_data,
    stringdtype_find_unicode_data,
    stringdtype_free_utf8_span,
    stringdtype_release_allocator,
    stringdtype_rfind_utf8_data,
    stringdtype_rfind_unicode_data,
    stringdtype_unicode_parts,
    stringdtype_unicode_utf8_span,
    stringdtype_unicode_valid,
    stringdtype_utf8_search_slice,
    utf8_count_stringdtype_sliced_data,
    utf8_find_stringdtype_sliced_data,
    utf8_rfind_stringdtype_sliced_data,
)
from charex.tests.definitions import StringsInformation


@njit(nogil=True, cache=False)
def find_unicode_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    result = np.empty(values.size, np.int64)
    if values.size == 0:
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_find_unicode_data(
            data, i, allocator, pattern, parts[0], parts[1], start, end)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def rfind_unicode_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    result = np.empty(values.size, np.int64)
    if values.size == 0:
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_rfind_unicode_data(
            data, i, allocator, pattern, parts[0], parts[1], start, end)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def count_unicode_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    result = np.empty(values.size, np.int64)
    if values.size == 0:
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_count_unicode_data(
            data, i, allocator, pattern, parts[0], parts[1], start, end)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def find_utf8_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    span = stringdtype_unicode_utf8_span(pattern, parts[0], parts[1])
    result = np.empty(values.size, np.int64)
    if values.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_find_utf8_data(
            data, i, allocator, span[0], span[1], start, end)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def rfind_utf8_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    span = stringdtype_unicode_utf8_span(pattern, parts[0], parts[1])
    result = np.empty(values.size, np.int64)
    if values.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_rfind_utf8_data(
            data, i, allocator, span[0], span[1], start, end)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def count_utf8_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    span = stringdtype_unicode_utf8_span(pattern, parts[0], parts[1])
    result = np.empty(values.size, np.int64)
    if values.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_count_utf8_data(
            data, i, allocator, span[0], span[1], start, end)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def find_utf8_value(value, patterns, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(value):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(value)
    span = stringdtype_unicode_utf8_span(value, parts[0], parts[1])
    slice_parts = stringdtype_utf8_search_slice(
        span[0], span[1], start, end)
    result = np.empty(patterns.size, np.int64)
    if patterns.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(patterns)
    data = stringdtype_data_ptr(patterns)
    for i in range(patterns.size):
        result[i] = utf8_find_stringdtype_sliced_data(
            span[0], slice_parts[0], slice_parts[1], slice_parts[2],
            slice_parts[3], slice_parts[4], data, i, allocator)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def rfind_utf8_value(value, patterns, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(value):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(value)
    span = stringdtype_unicode_utf8_span(value, parts[0], parts[1])
    slice_parts = stringdtype_utf8_search_slice(
        span[0], span[1], start, end)
    result = np.empty(patterns.size, np.int64)
    if patterns.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(patterns)
    data = stringdtype_data_ptr(patterns)
    for i in range(patterns.size):
        result[i] = utf8_rfind_stringdtype_sliced_data(
            span[0], slice_parts[0], slice_parts[1], slice_parts[2],
            slice_parts[3], slice_parts[4], data, i, allocator)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def count_utf8_value(value, patterns, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(value):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(value)
    span = stringdtype_unicode_utf8_span(value, parts[0], parts[1])
    slice_parts = stringdtype_utf8_search_slice(
        span[0], span[1], start, end)
    result = np.empty(patterns.size, np.int64)
    if patterns.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(patterns)
    data = stringdtype_data_ptr(patterns)
    for i in range(patterns.size):
        result[i] = utf8_count_stringdtype_sliced_data(
            span[0], slice_parts[0], slice_parts[1], slice_parts[2],
            slice_parts[3], slice_parts[4], data, i, allocator)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


def make_array(pattern, n):
    repeats = (n + len(pattern) - 1) // len(pattern)
    return np.array((pattern * repeats)[:n], dtype=np.dtypes.StringDType())


def time_candidates(candidates, repeat):
    timings = {name: [] for name in candidates}
    order = list(candidates) * repeat
    random.Random(7331).shuffle(order)
    for name in order:
        begin = time.perf_counter_ns()
        candidates[name]()
        timings[name].append((time.perf_counter_ns() - begin) / 1e6)
    return timings


def report(label, candidates, expected, repeat):
    for name, call in candidates.items():
        np.testing.assert_array_equal(call(), expected, err_msg=name)
    timings = time_candidates(candidates, repeat)
    print(f'\n{label}')
    base = statistics.median(timings['numpy'])
    current = statistics.median(timings['current'])
    for name in candidates:
        median = statistics.median(timings[name])
        print(f'{name:<8} min {min(timings[name]):8.3f} ms  '
              f'median {median:8.3f} ms  '
              f'vs numpy {base / median:7.3f}x  '
              f'vs current {current / median:7.3f}x')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, action='append')
    parser.add_argument('--repeat', type=int, default=11)
    parser.add_argument('--case-filter', default='')
    args = parser.parse_args(argv)

    strings = StringsInformation()
    max_end = 9223372036854775807
    cases = [
        ('short-first', ['abcxxx', 'xabcxx', 'xxabcx', ''], 'abc', 0,
         max_end),
        ('short-last', ['xxxabc', 'xxabcx', 'xabcxx', ''], 'abc', 0,
         max_end),
        ('short-slice', ['xabcx', 'xéfgx', 'x🙂abcx', 'xa\x00bcx'],
         'abc', 1, -1),
        ('empty-pattern', ['abc', '', 'é', '🙂'], '', 0, max_end),
        ('embedded-nul', ['a\x00bc', 'xxa\x00yy', '\x00abc', 'abc\x00x'],
         'a\x00', 0, max_end),
        ('trailing-nul-pattern', ['abc\x00x', 'abc\x00y', 'abc'],
         'abc\x00z', 0, max_end),
        ('unicode', ['éabcé', 'xxéyy', '🙂é', '漢字é'], 'é', 0, max_end),
        ('long-first', ['a' * 128 + 'z' * 128,
                        'x' + 'a' * 127 + 'z' * 128],
         'a' * 64, 0, max_end),
        ('long-last', ['z' * 128 + 'a' * 128,
                       'z' * 128 + 'a' * 127 + 'x'],
         'a' * 64, 0, max_end),
        ('long-none', ['z' * 256, 'y' * 256], 'a' * 64, 0, max_end),
        ('long-repeated', ['ab' * 128, 'ab' * 127 + 'xx'], 'ab' * 16, 0,
         max_end),
        ('long-unicode', ['é' * 128, 'ê' + 'é' * 127], 'é' * 32, 0,
         max_end),
        ('long-nul', ['a' * 128 + '\x00x', 'a' * 128 + '\x00y'],
         'a' * 128 + '\x00z', 0, max_end),
        ('long-trailing-nul', ['a' * 64 + '\x00', 'a' * 64,
                               'a' * 64 + '\x00x'],
         'a' * 64 + '\x00\x00', 0, max_end),
    ]
    n_values = args.n or [1000, 100000]

    ops = [
        ('find', np.strings.find, strings.strings_find, find_unicode_pattern,
         find_utf8_pattern, find_utf8_value),
        ('rfind', np.strings.rfind, strings.strings_rfind,
         rfind_unicode_pattern, rfind_utf8_pattern, rfind_utf8_value),
        ('count', np.strings.count, strings.strings_count,
         count_unicode_pattern, count_utf8_pattern, count_utf8_value),
    ]

    for n in n_values:
        for label, value_pattern, scalar, start, end in cases:
            if args.case_filter and args.case_filter not in label:
                continue
            values = make_array(value_pattern, n)
            for op_name, numpy_op, current_op, unicode_pattern, utf8_pattern, \
                    utf8_value in ops:
                candidates = {
                    'numpy': lambda op=numpy_op, v=values, s=scalar,
                    st=start, en=end: op(v, s, st, en),
                    'current': lambda op=current_op, v=values, s=scalar,
                    st=start, en=end: op(v, s, st, en),
                    'unicode': lambda fn=unicode_pattern, v=values, s=scalar,
                    st=start, en=end: fn(v, s, st, en),
                    'utf8': lambda fn=utf8_pattern, v=values, s=scalar,
                    st=start, en=end: fn(v, s, st, en),
                }
                expected = numpy_op(values, scalar, start, end)
                report(f'n={n} value-array {op_name} {label}',
                       candidates, expected, args.repeat)

                patterns = make_array(value_pattern, n)
                scalar_value = scalar
                candidates = {
                    'numpy': lambda op=numpy_op, v=scalar_value, p=patterns,
                    st=start, en=end: op(v, p, st, en),
                    'current': lambda op=current_op, v=scalar_value,
                    p=patterns, st=start, en=end: op(v, p, st, en),
                    'utf8': lambda fn=utf8_value, v=scalar_value, p=patterns,
                    st=start, en=end: fn(v, p, st, en),
                }
                expected = numpy_op(scalar_value, patterns, start, end)
                report(f'n={n} scalar-value {op_name} {label}',
                       candidates, expected, args.repeat)


if __name__ == '__main__':
    main()
