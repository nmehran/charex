"""Benchmark active StringDType equality path."""

from pathlib import Path
import random
import statistics
import sys
import time

from numba import njit
from numba.core import cgutils, types
from numba.core.typing import signature
from numba.extending import intrinsic
import numpy as np
from llvmlite import ir

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import charex  # noqa: F401
from charex.numpy.stringdtype import (
    stringdtype_acquire_allocators,
    stringdtype_data_ptr,
    stringdtype_equal_data,
    stringdtype_release_allocators,
)


_PACKED_STRING_SIZE = 16


def load_string(builder, allocator, packed, intp, byte_ptr):
    int32 = ir.IntType(32)
    static_type = ir.LiteralStructType([intp, byte_ptr])
    load_type = ir.FunctionType(
        int32, [byte_ptr, byte_ptr, static_type.as_pointer()],
    )
    load = cgutils.get_or_insert_function(
        builder.module, load_type, 'charex_NpyString_load',
    )
    static = cgutils.alloca_once(builder, static_type)
    status = builder.call(load, [allocator, packed, static])
    unpacked = builder.load(static)
    return status, builder.extract_value(unpacked, 0), \
        builder.extract_value(unpacked, 1)


@intrinsic
def stringdtype_equal_memcmp_data(typingctx, left_data, left_index,
                                  left_allocator, right_data, right_index,
                                  right_allocator):
    if left_data != types.voidptr \
            or not isinstance(left_index, types.Integer) \
            or left_allocator != types.voidptr \
            or right_data != types.voidptr \
            or not isinstance(right_index, types.Integer) \
            or right_allocator != types.voidptr:
        return None

    sig = signature(types.boolean, left_data, types.intp, left_allocator,
                    right_data, types.intp, right_allocator)

    def codegen(context, builder, signature, args):
        left_data, left_index_value, left_allocator, \
            right_data, right_index_value, right_allocator = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        left_offset = builder.mul(left_index_value,
                                  ir.Constant(intp, _PACKED_STRING_SIZE))
        right_offset = builder.mul(right_index_value,
                                   ir.Constant(intp, _PACKED_STRING_SIZE))
        left_packed = builder.gep(builder.bitcast(left_data, byte_ptr),
                                  [left_offset])
        right_packed = builder.gep(builder.bitcast(right_data, byte_ptr),
                                   [right_offset])
        left_status, left_size, left_buffer = load_string(
            builder, left_allocator, left_packed, intp, byte_ptr)
        right_status, right_size, right_buffer = load_string(
            builder, right_allocator, right_packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)

        left_valid = builder.icmp_signed(
            '==', left_status, ir.Constant(int32, 0))
        right_valid = builder.icmp_signed(
            '==', right_status, ir.Constant(int32, 0))
        both_valid = builder.and_(left_valid, right_valid)
        same_size = builder.icmp_unsigned('==', left_size, right_size)

        with builder.if_then(builder.and_(both_valid, same_size)):
            memchr_type = ir.FunctionType(byte_ptr, [byte_ptr, int32, intp])
            memcmp_type = ir.FunctionType(int32, [byte_ptr, byte_ptr, intp])
            memchr = cgutils.get_or_insert_function(
                builder.module, memchr_type, 'memchr',
            )
            memcmp = cgutils.get_or_insert_function(
                builder.module, memcmp_type, 'memcmp',
            )
            nul_ptr = builder.call(
                memchr, [left_buffer, ir.Constant(int32, 0), left_size],
            )
            found_nul = builder.icmp_unsigned(
                '!=', nul_ptr, ir.Constant(byte_ptr, None),
            )
            left_addr = builder.ptrtoint(left_buffer, intp)
            nul_addr = builder.ptrtoint(nul_ptr, intp)
            nul_size = builder.add(builder.sub(nul_addr, left_addr),
                                   ir.Constant(intp, 1))
            compare_size = builder.select(found_nul, nul_size, left_size)
            cmp_result = builder.call(
                memcmp, [left_buffer, right_buffer, compare_size],
            )
            builder.store(
                builder.icmp_signed('==', cmp_result, ir.Constant(int32, 0)),
                result,
            )

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_equal_hybrid_data(typingctx, left_data, left_index,
                                  left_allocator, right_data, right_index,
                                  right_allocator):
    if left_data != types.voidptr \
            or not isinstance(left_index, types.Integer) \
            or left_allocator != types.voidptr \
            or right_data != types.voidptr \
            or not isinstance(right_index, types.Integer) \
            or right_allocator != types.voidptr:
        return None

    sig = signature(types.boolean, left_data, types.intp, left_allocator,
                    right_data, types.intp, right_allocator)

    def codegen(context, builder, signature, args):
        left_data, left_index_value, left_allocator, \
            right_data, right_index_value, right_allocator = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        int1 = ir.IntType(1)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        left_offset = builder.mul(left_index_value,
                                  ir.Constant(intp, _PACKED_STRING_SIZE))
        right_offset = builder.mul(right_index_value,
                                   ir.Constant(intp, _PACKED_STRING_SIZE))
        left_packed = builder.gep(builder.bitcast(left_data, byte_ptr),
                                  [left_offset])
        right_packed = builder.gep(builder.bitcast(right_data, byte_ptr),
                                   [right_offset])
        left_status, left_size, left_buffer = load_string(
            builder, left_allocator, left_packed, intp, byte_ptr)
        right_status, right_size, right_buffer = load_string(
            builder, right_allocator, right_packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, int1)
        builder.store(cgutils.false_bit, result)

        left_valid = builder.icmp_signed(
            '==', left_status, ir.Constant(int32, 0))
        right_valid = builder.icmp_signed(
            '==', right_status, ir.Constant(int32, 0))
        both_valid = builder.and_(left_valid, right_valid)
        same_size = builder.icmp_unsigned('==', left_size, right_size)

        with builder.if_then(builder.and_(both_valid, same_size)):
            is_large = builder.icmp_unsigned(
                '>', left_size, ir.Constant(intp, _PACKED_STRING_SIZE))
            with builder.if_else(is_large) as (large, small):
                with large:
                    memchr_type = ir.FunctionType(
                        byte_ptr, [byte_ptr, int32, intp])
                    memcmp_type = ir.FunctionType(
                        int32, [byte_ptr, byte_ptr, intp])
                    memchr = cgutils.get_or_insert_function(
                        builder.module, memchr_type, 'memchr',
                    )
                    memcmp = cgutils.get_or_insert_function(
                        builder.module, memcmp_type, 'memcmp',
                    )
                    nul_ptr = builder.call(
                        memchr, [left_buffer, ir.Constant(int32, 0),
                                 left_size],
                    )
                    found_nul = builder.icmp_unsigned(
                        '!=', nul_ptr, ir.Constant(byte_ptr, None),
                    )
                    left_addr = builder.ptrtoint(left_buffer, intp)
                    nul_addr = builder.ptrtoint(nul_ptr, intp)
                    nul_size = builder.add(builder.sub(nul_addr, left_addr),
                                           ir.Constant(intp, 1))
                    compare_size = builder.select(
                        found_nul, nul_size, left_size)
                    cmp_result = builder.call(
                        memcmp, [left_buffer, right_buffer, compare_size],
                    )
                    builder.store(
                        builder.icmp_signed(
                            '==', cmp_result, ir.Constant(int32, 0)),
                        result,
                    )
                with small:
                    active = cgutils.alloca_once(builder, int1)
                    builder.store(cgutils.true_bit, result)
                    builder.store(cgutils.true_bit, active)
                    with cgutils.for_range(builder, left_size,
                                           intp=intp) as loop:
                        left_char = builder.load(
                            builder.gep(left_buffer, [loop.index]))
                        right_char = builder.load(
                            builder.gep(right_buffer, [loop.index]))
                        same_char = builder.icmp_unsigned(
                            '==', left_char, right_char)
                        still_active = builder.load(active)
                        builder.store(
                            builder.select(
                                still_active,
                                builder.and_(builder.load(result),
                                             same_char),
                                builder.load(result)),
                            result,
                        )
                        nonzero = builder.icmp_unsigned(
                            '!=', left_char, ir.Constant(int8, 0))
                        builder.store(
                            builder.and_(
                                still_active,
                                builder.and_(same_char, nonzero)),
                            active,
                        )

        return builder.load(result)

    return sig, codegen


@njit(nogil=True, cache=False)
def current_equal(left, right):
    return np.strings.equal(left, right)


@njit(nogil=True, cache=False)
def current_not_equal(left, right):
    return np.strings.not_equal(left, right)


@njit(nogil=True, cache=False)
def direct_equal(left, right):
    if left.size != right.size:
        raise ValueError('shape mismatch')
    result = np.empty(left.size, np.bool_)
    allocators = stringdtype_acquire_allocators(left, right)
    left_allocator = allocators[0]
    right_allocator = allocators[1]
    left_data = stringdtype_data_ptr(left)
    right_data = stringdtype_data_ptr(right)
    for i in range(left.size):
        result[i] = stringdtype_equal_data(
            left_data, i, left_allocator,
            right_data, i, right_allocator,
        )
    stringdtype_release_allocators(allocators)
    return result


@njit(nogil=True, cache=False)
def memcmp_equal(left, right):
    if left.size != right.size:
        raise ValueError('shape mismatch')
    result = np.empty(left.size, np.bool_)
    allocators = stringdtype_acquire_allocators(left, right)
    left_allocator = allocators[0]
    right_allocator = allocators[1]
    left_data = stringdtype_data_ptr(left)
    right_data = stringdtype_data_ptr(right)
    for i in range(left.size):
        result[i] = stringdtype_equal_memcmp_data(
            left_data, i, left_allocator,
            right_data, i, right_allocator,
        )
    stringdtype_release_allocators(allocators)
    return result


@njit(nogil=True, cache=False)
def hybrid_equal(left, right):
    if left.size != right.size:
        raise ValueError('shape mismatch')
    result = np.empty(left.size, np.bool_)
    allocators = stringdtype_acquire_allocators(left, right)
    left_allocator = allocators[0]
    right_allocator = allocators[1]
    left_data = stringdtype_data_ptr(left)
    right_data = stringdtype_data_ptr(right)
    for i in range(left.size):
        result[i] = stringdtype_equal_hybrid_data(
            left_data, i, left_allocator,
            right_data, i, right_allocator,
        )
    stringdtype_release_allocators(allocators)
    return result


def make_values(left_pattern, right_pattern, n):
    left_repeats = (n + len(left_pattern) - 1) // len(left_pattern)
    right_repeats = (n + len(right_pattern) - 1) // len(right_pattern)
    dtype = np.dtypes.StringDType()
    return (
        np.array((left_pattern * left_repeats)[:n], dtype=dtype),
        np.array((right_pattern * right_repeats)[:n], dtype=dtype),
    )


def bench_methods(methods, expected, left, right, repeat=15):
    valid_methods = []
    for name, method in methods:
        result = method(left, right)
        np.testing.assert_array_equal(result, expected)
        valid_methods.append((name, method))

    timings = {name: [] for name, _ in valid_methods}
    order = valid_methods * repeat
    random.Random(1337).shuffle(order)
    for name, method in order:
        start = time.perf_counter_ns()
        method(left, right)
        timings[name].append((time.perf_counter_ns() - start) / 1e6)

    medians = {}
    for name, samples in timings.items():
        medians[name] = statistics.median(samples)
        print(f'{name:<18} min {min(samples):8.3f} ms  '
              f'median {medians[name]:8.3f} ms')

    numpy_median = medians.get('numpy')
    if numpy_median:
        for name, median in medians.items():
            if name == 'numpy':
                continue
            print(f'{name:<18} median speedup {numpy_median / median:8.3f}x')


def main():
    cases = [
        ('equal-short',
         ['a', 'abc', 'é', '🙂', '', 'a\x00\x00'],
         ['a', 'abc', 'é', '🙂', '', 'a\x00xx']),
        ('first-mismatch',
         ['abc', 'éfg', '🙂x', 'short'],
         ['xbc', 'zfg', 'x🙂', 'xhort']),
        ('late-mismatch',
         ['a' * 31 + 'x', 'é' * 15 + 'x', '🙂' * 7 + 'x'],
         ['a' * 31 + 'y', 'é' * 15 + 'y', '🙂' * 7 + 'y']),
        ('unequal-bytes',
         ['a', 'abcd', 'é', '🙂'],
         ['aa', 'abc', 'ée', '🙂🙂']),
        ('embedded-nul',
         ['a\x00b', 'a\x00\x00', '\x00x', 'abc\x00tail'],
         ['a\x00c', 'a\x00xx', '\x00y', 'abc\x00zzzz']),
        ('unicode-equal',
         ['éé', 'κόσμε', '漢字', '🙂🙂', 'aé🙂'],
         ['éé', 'κόσμε', '漢字', '🙂🙂', 'aé🙂']),
        ('long-equal',
         ['a' * 256, 'é' * 128, '🙂' * 64],
         ['a' * 256, 'é' * 128, '🙂' * 64]),
        ('long-late-mismatch',
         ['a' * 255 + 'x', 'é' * 127 + 'x', '🙂' * 63 + 'x'],
         ['a' * 255 + 'y', 'é' * 127 + 'y', '🙂' * 63 + 'y']),
        ('long-first-mismatch',
         ['x' + 'a' * 255, 'x' + 'é' * 127, 'x' + '🙂' * 63],
         ['y' + 'a' * 255, 'y' + 'é' * 127, 'y' + '🙂' * 63]),
    ]

    for case_name, left_pattern, right_pattern in cases:
        for n in [1000, 100000]:
            left, right = make_values(left_pattern, right_pattern, n)
            print(f'\ncase={case_name} n={n}')
            bench_methods(
                [
                    ('numpy', np.strings.equal),
                    ('current', current_equal),
                    ('direct', direct_equal),
                    ('memcmp', memcmp_equal),
                    ('hybrid', hybrid_equal),
                ],
                np.strings.equal(left, right),
                left,
                right,
            )
            bench_methods(
                [
                    ('numpy not_equal', np.strings.not_equal),
                    ('current not_equal', current_not_equal),
                ],
                np.strings.not_equal(left, right),
                left,
                right,
            )


if __name__ == '__main__':
    main()
