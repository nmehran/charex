"""Explore Python str scalar spans for mixed StringDType search."""

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
    _codepoint_count,
    _load_string,
    _normalise_slice,
    _packed_string_ptr_from_data,
    _trimmed_size,
    stringdtype_acquire_allocator,
    stringdtype_count_unicode_data,
    stringdtype_data_ptr,
    stringdtype_find_unicode_data,
    stringdtype_free_utf8_span,
    stringdtype_release_allocator,
    stringdtype_rfind_unicode_data,
    stringdtype_unicode_parts,
    stringdtype_unicode_utf8_span,
    stringdtype_unicode_valid,
)
from charex.tests.definitions import StringsInformation


_UTF8_SEARCH_SLICE_TYPE = types.Tuple((
    types.intp, types.intp, types.intp, types.intp, types.boolean))


def _memcmp_equal(builder, left, right, size, int8, int32):
    memcmp_type = ir.FunctionType(
        int32, [int8.as_pointer(), int8.as_pointer(), size.type])
    memcmp = cgutils.get_or_insert_function(
        builder.module, memcmp_type, 'memcmp')
    cmp_result = builder.call(memcmp, [left, right, size])
    return builder.icmp_signed('==', cmp_result, ir.Constant(int32, 0))


def _byte_search_sliced(builder, value_buffer, start_index, end_index,
                        start_offset, end_offset, slice_valid,
                        pattern_effective_size, pattern_match_size,
                        pattern_buffer, mode, intp, int8, int32):
    zero = ir.Constant(intp, 0)
    one = ir.Constant(intp, 1)
    result = cgutils.alloca_once(builder, intp)
    if mode == 'count':
        builder.store(zero, result)
    else:
        builder.store(ir.Constant(intp, -1), result)

    empty_pattern = builder.icmp_unsigned('==', pattern_effective_size, zero)
    with builder.if_then(builder.and_(slice_valid, empty_pattern)):
        if mode == 'find':
            builder.store(start_index, result)
        elif mode == 'rfind':
            builder.store(end_index, result)
        else:
            builder.store(builder.add(builder.sub(end_index, start_index),
                                      one), result)

    slice_size = builder.sub(end_offset, start_offset)
    nonempty_pattern = builder.not_(empty_pattern)
    fits = builder.icmp_unsigned('<=', pattern_match_size, slice_size)
    with builder.if_then(builder.and_(slice_valid,
                                      builder.and_(nonempty_pattern, fits))):
        first_pattern = builder.load(pattern_buffer)
        found = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, found)

        if mode == 'rfind':
            pos = cgutils.alloca_once(builder, intp)
            last = builder.sub(end_offset, pattern_match_size)
            builder.store(last, pos)
            cond = builder.append_basic_block('stringdtype.utf8.rfind.cond')
            body = builder.append_basic_block('stringdtype.utf8.rfind.body')
            decrement = builder.append_basic_block(
                'stringdtype.utf8.rfind.decrement')
            after = builder.append_basic_block('stringdtype.utf8.rfind.after')
            builder.branch(cond)

            builder.position_at_end(cond)
            in_range = builder.icmp_signed('>=', builder.load(pos),
                                           start_offset)
            builder.cbranch(builder.and_(in_range,
                                         builder.not_(builder.load(found))),
                            body, after)

            builder.position_at_end(body)
            value_byte = builder.load(
                builder.gep(value_buffer, [builder.load(pos)]))
            first_matches = builder.icmp_unsigned(
                '==', value_byte, first_pattern)
            with builder.if_then(first_matches):
                matched = _memcmp_equal(
                    builder, builder.gep(value_buffer, [builder.load(pos)]),
                    pattern_buffer, pattern_match_size, int8, int32)
                with builder.if_then(matched):
                    builder.store(
                        _codepoint_count(builder, builder.load(pos),
                                         value_buffer, intp, int8),
                        result,
                    )
                    builder.store(cgutils.true_bit, found)
            builder.branch(decrement)

            builder.position_at_end(decrement)
            builder.store(builder.sub(builder.load(pos), one), pos)
            builder.branch(cond)

            builder.position_at_end(after)
        else:
            pos = cgutils.alloca_once(builder, intp)
            last = builder.sub(end_offset, pattern_match_size)
            builder.store(start_offset, pos)
            cond = builder.append_basic_block('stringdtype.utf8.find.cond')
            body = builder.append_basic_block('stringdtype.utf8.find.body')
            advance = builder.append_basic_block(
                'stringdtype.utf8.find.advance')
            after = builder.append_basic_block('stringdtype.utf8.find.after')
            builder.branch(cond)

            builder.position_at_end(cond)
            in_range = builder.icmp_signed('<=', builder.load(pos), last)
            continue_search = in_range
            if mode == 'find':
                continue_search = builder.and_(
                    in_range, builder.not_(builder.load(found)))
            builder.cbranch(continue_search, body, after)

            builder.position_at_end(body)
            value_byte = builder.load(
                builder.gep(value_buffer, [builder.load(pos)]))
            first_matches = builder.icmp_unsigned(
                '==', value_byte, first_pattern)
            matched = cgutils.alloca_once(builder, ir.IntType(1))
            builder.store(cgutils.false_bit, matched)
            with builder.if_then(first_matches):
                builder.store(
                    _memcmp_equal(
                        builder,
                        builder.gep(value_buffer, [builder.load(pos)]),
                        pattern_buffer, pattern_match_size, int8, int32),
                    matched,
                )
            with builder.if_then(builder.load(matched)):
                if mode == 'find':
                    builder.store(
                        _codepoint_count(builder, builder.load(pos),
                                         value_buffer, intp, int8),
                        result,
                    )
                    builder.store(cgutils.true_bit, found)
                else:
                    builder.store(builder.add(builder.load(result), one),
                                  result)
            builder.branch(advance)

            builder.position_at_end(advance)
            if mode == 'count':
                next_match = builder.add(builder.load(pos),
                                         pattern_match_size)
                next_scan = builder.add(builder.load(pos), one)
                builder.store(
                    builder.select(builder.load(matched), next_match,
                                   next_scan),
                    pos,
                )
            else:
                builder.store(builder.add(builder.load(pos), one), pos)
            builder.branch(cond)

            builder.position_at_end(after)

    return builder.load(result)


def _byte_search(builder, value_size, value_buffer, pattern_effective_size,
                 pattern_match_size, pattern_buffer, start, end, mode, intp,
                 int8, int32):
    start_index, end_index, start_offset, end_offset, slice_valid = \
        _normalise_slice(builder, value_size, value_buffer, start, end, intp,
                         int8)
    return _byte_search_sliced(
        builder, value_buffer, start_index, end_index, start_offset,
        end_offset, slice_valid, pattern_effective_size, pattern_match_size,
        pattern_buffer, mode, intp, int8, int32)


@intrinsic
def utf8_search_slice(typingctx, value_data, value_size, start, end):
    if value_data != types.voidptr \
            or not isinstance(value_size, types.Integer) \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(_UTF8_SEARCH_SLICE_TYPE, value_data, types.intp,
                    types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value_data, value_size, start, end = args
        int8 = ir.IntType(8)
        intp = context.get_value_type(types.intp)
        start_index, end_index, start_offset, end_offset, slice_valid = \
            _normalise_slice(builder, value_size, value_data, start, end, intp,
                             int8)
        return context.make_tuple(
            builder, signature.return_type,
            [start_index, end_index, start_offset, end_offset, slice_valid],
        )

    return sig, codegen


def _stringdtype_utf8_search_template(typingctx, value_data, value_index,
                                      value_allocator, pattern_data,
                                      pattern_size, start, end, mode):
    if value_data != types.voidptr \
            or not isinstance(value_index, types.Integer) \
            or value_allocator != types.voidptr \
            or pattern_data != types.voidptr \
            or not isinstance(pattern_size, types.Integer) \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(types.intp, value_data, types.intp, value_allocator,
                    pattern_data, types.intp, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value_data, value_index_value, value_allocator, pattern_data, \
            pattern_size, start, end = args
        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        value_packed = _packed_string_ptr_from_data(
            builder, value_data, value_index_value, intp)
        value_status, value_size, value_buffer = _load_string(
            builder, value_allocator, value_packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        if mode == 'count':
            builder.store(ir.Constant(intp, 0), result)
        else:
            builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', value_status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            builder.store(
                _byte_search(
                    builder, value_size, value_buffer, pattern_size,
                    pattern_size, pattern_data, start, end, mode, intp, int8,
                    int32),
                result,
            )

        return builder.load(result)

    return sig, codegen


def _utf8_stringdtype_search_sliced_template(
        typingctx, value_data, start_index, end_index, start_offset,
        end_offset, slice_valid, pattern_data, pattern_index,
        pattern_allocator, mode):
    if value_data != types.voidptr \
            or not isinstance(start_index, types.Integer) \
            or not isinstance(end_index, types.Integer) \
            or not isinstance(start_offset, types.Integer) \
            or not isinstance(end_offset, types.Integer) \
            or not isinstance(slice_valid, types.Boolean) \
            or pattern_data != types.voidptr \
            or not isinstance(pattern_index, types.Integer) \
            or pattern_allocator != types.voidptr:
        return None

    sig = signature(types.intp, value_data, types.intp, types.intp,
                    types.intp, types.intp, types.boolean, pattern_data,
                    types.intp, pattern_allocator)

    def codegen(context, builder, signature, args):
        value_data, start_index, end_index, start_offset, end_offset, \
            slice_valid, pattern_data, pattern_index_value, \
            pattern_allocator = args
        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        one = ir.Constant(intp, 1)
        pattern_packed = _packed_string_ptr_from_data(
            builder, pattern_data, pattern_index_value, intp)
        pattern_status, pattern_size, pattern_buffer = _load_string(
            builder, pattern_allocator, pattern_packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        if mode == 'count':
            builder.store(ir.Constant(intp, 0), result)
        else:
            builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', pattern_status,
                                    ir.Constant(int32, 0))
        with builder.if_then(valid):
            pattern_effective_size = _trimmed_size(
                builder, pattern_size, pattern_buffer, intp, int8)
            empty_pattern = builder.icmp_unsigned(
                '==', pattern_effective_size, ir.Constant(intp, 0))
            if mode == 'count':
                pattern_match_size = builder.select(
                    empty_pattern, ir.Constant(intp, 0), pattern_size)
            else:
                short_pattern = builder.icmp_unsigned(
                    '<=', pattern_effective_size, one)
                pattern_match_size = builder.select(
                    short_pattern, pattern_effective_size, pattern_size)
            builder.store(
                _byte_search_sliced(
                    builder, value_data, start_index, end_index, start_offset,
                    end_offset, slice_valid, pattern_effective_size,
                    pattern_match_size, pattern_buffer, mode, intp, int8,
                    int32),
                result,
            )

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_find_utf8_data(typingctx, value_data, value_index,
                               value_allocator, pattern_data, pattern_size,
                               start, end):
    return _stringdtype_utf8_search_template(
        typingctx, value_data, value_index, value_allocator, pattern_data,
        pattern_size, start, end, 'find')


@intrinsic
def stringdtype_rfind_utf8_data(typingctx, value_data, value_index,
                                value_allocator, pattern_data, pattern_size,
                                start, end):
    return _stringdtype_utf8_search_template(
        typingctx, value_data, value_index, value_allocator, pattern_data,
        pattern_size, start, end, 'rfind')


@intrinsic
def stringdtype_count_utf8_data(typingctx, value_data, value_index,
                                value_allocator, pattern_data, pattern_size,
                                start, end):
    return _stringdtype_utf8_search_template(
        typingctx, value_data, value_index, value_allocator, pattern_data,
        pattern_size, start, end, 'count')


@intrinsic
def utf8_find_stringdtype_sliced_data(
        typingctx, value_data, start_index, end_index, start_offset,
        end_offset, slice_valid, pattern_data, pattern_index,
        pattern_allocator):
    return _utf8_stringdtype_search_sliced_template(
        typingctx, value_data, start_index, end_index, start_offset,
        end_offset, slice_valid, pattern_data, pattern_index,
        pattern_allocator, 'find')


@intrinsic
def utf8_rfind_stringdtype_sliced_data(
        typingctx, value_data, start_index, end_index, start_offset,
        end_offset, slice_valid, pattern_data, pattern_index,
        pattern_allocator):
    return _utf8_stringdtype_search_sliced_template(
        typingctx, value_data, start_index, end_index, start_offset,
        end_offset, slice_valid, pattern_data, pattern_index,
        pattern_allocator, 'rfind')


@intrinsic
def utf8_count_stringdtype_sliced_data(
        typingctx, value_data, start_index, end_index, start_offset,
        end_offset, slice_valid, pattern_data, pattern_index,
        pattern_allocator):
    return _utf8_stringdtype_search_sliced_template(
        typingctx, value_data, start_index, end_index, start_offset,
        end_offset, slice_valid, pattern_data, pattern_index,
        pattern_allocator, 'count')


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
    slice_parts = utf8_search_slice(span[0], span[1], start, end)
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
    slice_parts = utf8_search_slice(span[0], span[1], start, end)
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
    slice_parts = utf8_search_slice(span[0], span[1], start, end)
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
