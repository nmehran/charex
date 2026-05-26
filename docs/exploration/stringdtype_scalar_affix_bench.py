"""Explore Python str scalar spans for mixed StringDType affix checks."""

from pathlib import Path
import argparse
import random
import statistics
import sys
import time

from llvmlite import ir
from numba import njit
from numba.core import cgutils, types
from numba.core.typing import signature
from numba.extending import intrinsic
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import charex  # noqa: F401
from charex.numpy.stringdtype import (
    _normalise_slice,
    _packed_string_ptr_from_data,
    _trimmed_size,
    _load_string,
    stringdtype_acquire_allocator,
    stringdtype_data_ptr,
    stringdtype_free_utf8_span,
    stringdtype_release_allocator,
    stringdtype_startswith_unicode_data,
    stringdtype_endswith_unicode_data,
    stringdtype_unicode_parts,
    stringdtype_unicode_utf8_span,
    stringdtype_unicode_valid,
)
from charex.tests.definitions import StringsInformation


_UTF8_SLICE_TYPE = types.Tuple((types.intp, types.intp, types.boolean))


def _memcmp_equal(builder, left, right, size, int8, int32):
    memcmp_type = ir.FunctionType(
        int32, [int8.as_pointer(), int8.as_pointer(), size.type])
    memcmp = cgutils.get_or_insert_function(
        builder.module, memcmp_type, 'memcmp')
    cmp_result = builder.call(memcmp, [left, right, size])
    return builder.icmp_signed('==', cmp_result, ir.Constant(int32, 0))


@intrinsic
def utf8_normalise_slice(typingctx, value_data, value_size, start, end):
    if value_data != types.voidptr \
            or not isinstance(value_size, types.Integer) \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(_UTF8_SLICE_TYPE, value_data, types.intp, types.intp,
                    types.intp)

    def codegen(context, builder, signature, args):
        value_data, value_size, start, end = args
        int8 = ir.IntType(8)
        intp = context.get_value_type(types.intp)
        _, _, start_offset, end_offset, slice_valid = _normalise_slice(
            builder, value_size, value_data, start, end, intp, int8)
        return context.make_tuple(
            builder, signature.return_type,
            [start_offset, end_offset, slice_valid],
        )

    return sig, codegen


def _utf8_stringdtype_affix_template(typingctx, value_data, value_size,
                                     pattern_data, pattern_index,
                                     pattern_allocator, start, end, suffix):
    if value_data != types.voidptr \
            or not isinstance(value_size, types.Integer) \
            or pattern_data != types.voidptr \
            or not isinstance(pattern_index, types.Integer) \
            or pattern_allocator != types.voidptr \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(types.boolean, value_data, types.intp, pattern_data,
                    types.intp, pattern_allocator, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value_data, value_size, pattern_data, pattern_index, \
            pattern_allocator, start, end = args
        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        packed = _packed_string_ptr_from_data(
            builder, pattern_data, pattern_index, intp)
        status, pattern_size, pattern_buffer = _load_string(
            builder, pattern_allocator, packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))

        with builder.if_then(valid):
            _, _, start_offset, end_offset, slice_valid = _normalise_slice(
                builder, value_size, value_data, start, end, intp, int8)
            pattern_effective_size = _trimmed_size(
                builder, pattern_size, pattern_buffer, intp, int8)
            slice_size = builder.sub(end_offset, start_offset)
            empty_pattern = builder.icmp_unsigned(
                '==', pattern_effective_size, ir.Constant(intp, 0))
            builder.store(builder.and_(slice_valid, empty_pattern), result)

            nonempty_pattern = builder.not_(empty_pattern)
            fits = builder.icmp_unsigned('<=', pattern_effective_size,
                                         slice_size)
            with builder.if_then(builder.and_(slice_valid,
                                              builder.and_(nonempty_pattern,
                                                           fits))):
                if suffix:
                    compare_offset = builder.sub(end_offset,
                                                 pattern_effective_size)
                else:
                    compare_offset = start_offset
                left = builder.gep(value_data, [compare_offset])
                builder.store(
                    _memcmp_equal(builder, left, pattern_buffer,
                                  pattern_effective_size, int8, int32),
                    result,
                )

        return builder.load(result)

    return sig, codegen


def _utf8_stringdtype_affix_sliced_template(
        typingctx, value_data, start_offset, end_offset, slice_valid,
        pattern_data, pattern_index, pattern_allocator, suffix):
    if value_data != types.voidptr \
            or not isinstance(start_offset, types.Integer) \
            or not isinstance(end_offset, types.Integer) \
            or not isinstance(slice_valid, types.Boolean) \
            or pattern_data != types.voidptr \
            or not isinstance(pattern_index, types.Integer) \
            or pattern_allocator != types.voidptr:
        return None

    sig = signature(types.boolean, value_data, types.intp, types.intp,
                    types.boolean, pattern_data, types.intp,
                    pattern_allocator)

    def codegen(context, builder, signature, args):
        value_data, start_offset, end_offset, slice_valid, pattern_data, \
            pattern_index, pattern_allocator = args
        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        packed = _packed_string_ptr_from_data(
            builder, pattern_data, pattern_index, intp)
        status, pattern_size, pattern_buffer = _load_string(
            builder, pattern_allocator, packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))

        with builder.if_then(valid):
            pattern_effective_size = _trimmed_size(
                builder, pattern_size, pattern_buffer, intp, int8)
            slice_size = builder.sub(end_offset, start_offset)
            empty_pattern = builder.icmp_unsigned(
                '==', pattern_effective_size, ir.Constant(intp, 0))
            builder.store(builder.and_(slice_valid, empty_pattern), result)

            nonempty_pattern = builder.not_(empty_pattern)
            fits = builder.icmp_unsigned('<=', pattern_effective_size,
                                         slice_size)
            with builder.if_then(builder.and_(slice_valid,
                                              builder.and_(nonempty_pattern,
                                                           fits))):
                if suffix:
                    compare_offset = builder.sub(end_offset,
                                                 pattern_effective_size)
                else:
                    compare_offset = start_offset
                left = builder.gep(value_data, [compare_offset])
                builder.store(
                    _memcmp_equal(builder, left, pattern_buffer,
                                  pattern_effective_size, int8, int32),
                    result,
                )

        return builder.load(result)

    return sig, codegen


def _stringdtype_utf8_affix_template(typingctx, value_data, value_index,
                                     value_allocator, pattern_data,
                                     pattern_size, start, end, suffix):
    if value_data != types.voidptr \
            or not isinstance(value_index, types.Integer) \
            or value_allocator != types.voidptr \
            or pattern_data != types.voidptr \
            or not isinstance(pattern_size, types.Integer) \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(types.boolean, value_data, types.intp, value_allocator,
                    pattern_data, types.intp, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value_data, value_index, value_allocator, pattern_data, pattern_size, \
            start, end = args
        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        packed = _packed_string_ptr_from_data(
            builder, value_data, value_index, intp)
        status, value_size, value_buffer = _load_string(
            builder, value_allocator, packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))

        with builder.if_then(valid):
            _, _, start_offset, end_offset, slice_valid = _normalise_slice(
                builder, value_size, value_buffer, start, end, intp, int8)
            slice_size = builder.sub(end_offset, start_offset)
            empty_pattern = builder.icmp_unsigned(
                '==', pattern_size, ir.Constant(intp, 0))
            builder.store(builder.and_(slice_valid, empty_pattern), result)

            nonempty_pattern = builder.not_(empty_pattern)
            fits = builder.icmp_unsigned('<=', pattern_size, slice_size)
            with builder.if_then(builder.and_(slice_valid,
                                              builder.and_(nonempty_pattern,
                                                           fits))):
                if suffix:
                    compare_offset = builder.sub(end_offset, pattern_size)
                else:
                    compare_offset = start_offset
                left = builder.gep(value_buffer, [compare_offset])
                builder.store(
                    _memcmp_equal(builder, left, pattern_data, pattern_size,
                                  int8, int32),
                    result,
                )

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_startswith_utf8_data(typingctx, value_data, value_index,
                                     value_allocator, pattern_data,
                                     pattern_size, start, end):
    return _stringdtype_utf8_affix_template(
        typingctx, value_data, value_index, value_allocator, pattern_data,
        pattern_size, start, end, False)


@intrinsic
def stringdtype_endswith_utf8_data(typingctx, value_data, value_index,
                                   value_allocator, pattern_data,
                                   pattern_size, start, end):
    return _stringdtype_utf8_affix_template(
        typingctx, value_data, value_index, value_allocator, pattern_data,
        pattern_size, start, end, True)


@intrinsic
def utf8_startswith_stringdtype_data(typingctx, value_data, value_size,
                                     pattern_data, pattern_index,
                                     pattern_allocator, start, end):
    return _utf8_stringdtype_affix_template(
        typingctx, value_data, value_size, pattern_data, pattern_index,
        pattern_allocator, start, end, False)


@intrinsic
def utf8_endswith_stringdtype_data(typingctx, value_data, value_size,
                                   pattern_data, pattern_index,
                                   pattern_allocator, start, end):
    return _utf8_stringdtype_affix_template(
        typingctx, value_data, value_size, pattern_data, pattern_index,
        pattern_allocator, start, end, True)


@intrinsic
def utf8_startswith_stringdtype_sliced_data(
        typingctx, value_data, start_offset, end_offset, slice_valid,
        pattern_data, pattern_index, pattern_allocator):
    return _utf8_stringdtype_affix_sliced_template(
        typingctx, value_data, start_offset, end_offset, slice_valid,
        pattern_data, pattern_index, pattern_allocator, False)


@intrinsic
def utf8_endswith_stringdtype_sliced_data(
        typingctx, value_data, start_offset, end_offset, slice_valid,
        pattern_data, pattern_index, pattern_allocator):
    return _utf8_stringdtype_affix_sliced_template(
        typingctx, value_data, start_offset, end_offset, slice_valid,
        pattern_data, pattern_index, pattern_allocator, True)


@njit(nogil=True, cache=False)
def startswith_current_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    result = np.empty(values.size, np.bool_)
    if values.size == 0:
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_startswith_unicode_data(
            data, i, allocator, pattern, parts[0], parts[1], start, end)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def endswith_current_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    result = np.empty(values.size, np.bool_)
    if values.size == 0:
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_endswith_unicode_data(
            data, i, allocator, pattern, parts[0], parts[1], start, end)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def startswith_utf8_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    span = stringdtype_unicode_utf8_span(pattern, parts[0], parts[1])
    result = np.empty(values.size, np.bool_)
    if values.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_startswith_utf8_data(
            data, i, allocator, span[0], span[1], start, end)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def startswith_hybrid_pattern(values, pattern, start=0,
                              end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    if parts[1] <= 16:
        return startswith_current_pattern(values, pattern, start, end)
    return startswith_utf8_pattern(values, pattern, start, end)


@njit(nogil=True, cache=False)
def endswith_utf8_pattern(values, pattern, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    span = stringdtype_unicode_utf8_span(pattern, parts[0], parts[1])
    result = np.empty(values.size, np.bool_)
    if values.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_endswith_utf8_data(
            data, i, allocator, span[0], span[1], start, end)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def endswith_hybrid_pattern(values, pattern, start=0,
                            end=9223372036854775807):
    if not stringdtype_unicode_valid(pattern):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(pattern)
    if parts[1] <= 16:
        return endswith_current_pattern(values, pattern, start, end)
    return endswith_utf8_pattern(values, pattern, start, end)


@njit(nogil=True, cache=False)
def startswith_utf8_value(value, patterns, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(value):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(value)
    span = stringdtype_unicode_utf8_span(value, parts[0], parts[1])
    result = np.empty(patterns.size, np.bool_)
    if patterns.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(patterns)
    data = stringdtype_data_ptr(patterns)
    for i in range(patterns.size):
        result[i] = utf8_startswith_stringdtype_data(
            span[0], span[1], data, i, allocator, start, end)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def endswith_utf8_value(value, patterns, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(value):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(value)
    span = stringdtype_unicode_utf8_span(value, parts[0], parts[1])
    result = np.empty(patterns.size, np.bool_)
    if patterns.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(patterns)
    data = stringdtype_data_ptr(patterns)
    for i in range(patterns.size):
        result[i] = utf8_endswith_stringdtype_data(
            span[0], span[1], data, i, allocator, start, end)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def startswith_utf8_value_sliced(
        value, patterns, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(value):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(value)
    span = stringdtype_unicode_utf8_span(value, parts[0], parts[1])
    slice_parts = utf8_normalise_slice(span[0], span[1], start, end)
    result = np.empty(patterns.size, np.bool_)
    if patterns.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(patterns)
    data = stringdtype_data_ptr(patterns)
    for i in range(patterns.size):
        result[i] = utf8_startswith_stringdtype_sliced_data(
            span[0], slice_parts[0], slice_parts[1], slice_parts[2],
            data, i, allocator)
    stringdtype_release_allocator(allocator)
    stringdtype_free_utf8_span(span[0], span[2])
    return result


@njit(nogil=True, cache=False)
def endswith_utf8_value_sliced(
        value, patterns, start=0, end=9223372036854775807):
    if not stringdtype_unicode_valid(value):
        raise TypeError('Invalid unicode code point found')
    parts = stringdtype_unicode_parts(value)
    span = stringdtype_unicode_utf8_span(value, parts[0], parts[1])
    slice_parts = utf8_normalise_slice(span[0], span[1], start, end)
    result = np.empty(patterns.size, np.bool_)
    if patterns.size == 0:
        stringdtype_free_utf8_span(span[0], span[2])
        return result
    allocator = stringdtype_acquire_allocator(patterns)
    data = stringdtype_data_ptr(patterns)
    for i in range(patterns.size):
        result[i] = utf8_endswith_stringdtype_sliced_data(
            span[0], slice_parts[0], slice_parts[1], slice_parts[2],
            data, i, allocator)
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
        ('short-ascii', ['abc', 'abcdef', 'xabc', '', 'abc\x00'], 'abc', 0,
         max_end),
        ('short-slice', ['xabcx', 'xéfgx', 'x🙂abcx', 'xa\x00bcx'],
         'abc', 1, -1),
        ('embedded-nul', ['a\x00bc', 'ab\x00c', '\x00abc', 'abc\x00x'],
         'a\x00', 0, max_end),
        ('unicode', ['éclair', 'κόσμε', '🙂abc', '漢字abc'], 'é', 0,
         max_end),
        ('long-prefix', ['a' * 256, 'x' + 'a' * 255], 'a' * 128, 0,
         max_end),
        ('long-suffix', ['a' * 256, 'a' * 255 + 'x'], 'a' * 128, 0,
         max_end),
        ('long-unicode', ['é' * 128, 'ê' + 'é' * 127], 'é' * 64, 0,
         max_end),
        ('long-emoji', ['🙂' * 64, '🙃' + '🙂' * 63], '🙂' * 32, 0,
         max_end),
        ('long-nul', ['a' * 128 + '\x00x', 'a' * 128 + '\x00y'],
         'a' * 128 + '\x00z', 0, max_end),
    ]
    n_values = args.n or [1000, 100000]

    for n in n_values:
        for label, value_pattern, scalar, start, end in cases:
            if args.case_filter and args.case_filter not in label:
                continue
            values = make_array(value_pattern, n)

            candidates = {
                'numpy': lambda v=values, s=scalar, st=start, en=end:
                    np.strings.startswith(v, s, st, en),
                'current': lambda v=values, s=scalar, st=start, en=end:
                    strings.strings_startswith(v, s, st, en),
                'unicode': lambda v=values, s=scalar, st=start, en=end:
                    startswith_current_pattern(v, s, st, en),
                'utf8': lambda v=values, s=scalar, st=start, en=end:
                    startswith_utf8_pattern(v, s, st, en),
                'hybrid': lambda v=values, s=scalar, st=start, en=end:
                    startswith_hybrid_pattern(v, s, st, en),
            }
            expected = np.strings.startswith(values, scalar, start, end)
            report(f'n={n} value-array startswith {label}',
                   candidates, expected, args.repeat)

            candidates = {
                'numpy': lambda v=values, s=scalar, st=start, en=end:
                    np.strings.endswith(v, s, st, en),
                'current': lambda v=values, s=scalar, st=start, en=end:
                    strings.strings_endswith(v, s, st, en),
                'unicode': lambda v=values, s=scalar, st=start, en=end:
                    endswith_current_pattern(v, s, st, en),
                'utf8': lambda v=values, s=scalar, st=start, en=end:
                    endswith_utf8_pattern(v, s, st, en),
                'hybrid': lambda v=values, s=scalar, st=start, en=end:
                    endswith_hybrid_pattern(v, s, st, en),
            }
            expected = np.strings.endswith(values, scalar, start, end)
            report(f'n={n} value-array endswith {label}',
                   candidates, expected, args.repeat)

            patterns = make_array(value_pattern, n)
            scalar_value = scalar
            candidates = {
                'numpy': lambda v=scalar_value, p=patterns, st=start, en=end:
                    np.strings.startswith(v, p, st, en),
                'current': lambda v=scalar_value, p=patterns, st=start, en=end:
                    strings.strings_startswith(v, p, st, en),
                'utf8': lambda v=scalar_value, p=patterns, st=start, en=end:
                    startswith_utf8_value(v, p, st, en),
                'utf8s': lambda v=scalar_value, p=patterns, st=start, en=end:
                    startswith_utf8_value_sliced(v, p, st, en),
            }
            expected = np.strings.startswith(scalar_value, patterns,
                                             start, end)
            report(f'n={n} scalar-value startswith {label}',
                   candidates, expected, args.repeat)

            candidates = {
                'numpy': lambda v=scalar_value, p=patterns, st=start, en=end:
                    np.strings.endswith(v, p, st, en),
                'current': lambda v=scalar_value, p=patterns, st=start, en=end:
                    strings.strings_endswith(v, p, st, en),
                'utf8': lambda v=scalar_value, p=patterns, st=start, en=end:
                    endswith_utf8_value(v, p, st, en),
                'utf8s': lambda v=scalar_value, p=patterns, st=start, en=end:
                    endswith_utf8_value_sliced(v, p, st, en),
            }
            expected = np.strings.endswith(scalar_value, patterns, start, end)
            report(f'n={n} scalar-value endswith {label}',
                   candidates, expected, args.repeat)


if __name__ == '__main__':
    main()
