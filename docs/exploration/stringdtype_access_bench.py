"""Benchmark active StringDType str_len access candidates.

Rejected Tranche 1 experiments are documented in
docs/exploration/stringdtype_access_rejected.md. This file intentionally keeps
only the current runtime path and candidates still worth comparing.
"""

from pathlib import Path
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
    is_stringdtype_array_type,
    stringdtype_acquire_allocator,
    stringdtype_codepoint_len_data,
    stringdtype_data_ptr,
    stringdtype_release_allocator,
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


def byte_prefix_codepoint_count(builder, size, buffer, intp, int8):
    count = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), count)

    with cgutils.for_range(builder, size, intp=intp) as loop:
        char = builder.load(builder.gep(buffer, [loop.index]))
        tag = builder.and_(char, ir.Constant(int8, 0xc0))
        continuation = builder.icmp_unsigned(
            '==', tag, ir.Constant(int8, 0x80),
        )
        increment = builder.select(
            continuation, ir.Constant(intp, 0), ir.Constant(intp, 1),
        )
        builder.store(builder.add(builder.load(count), increment), count)

    return builder.load(count)


def backward_trim_size(builder, size, buffer, intp, int8):
    effective_size = cgutils.alloca_once(builder, intp)
    builder.store(size, effective_size)

    cond = builder.append_basic_block('trim_back.cond')
    check = builder.append_basic_block('trim_back.check')
    body = builder.append_basic_block('trim_back.body')
    after = builder.append_basic_block('trim_back.after')

    builder.branch(cond)
    builder.position_at_end(cond)
    current_size = builder.load(effective_size)
    has_remaining = builder.icmp_unsigned('>', current_size,
                                          ir.Constant(intp, 0))
    builder.cbranch(has_remaining, check, after)

    builder.position_at_end(check)
    previous = builder.sub(builder.load(effective_size),
                           ir.Constant(intp, 1))
    char = builder.load(builder.gep(buffer, [previous]))
    is_zero = builder.icmp_unsigned('==', char, ir.Constant(int8, 0))
    builder.cbranch(is_zero, body, after)

    builder.position_at_end(body)
    builder.store(previous, effective_size)
    builder.branch(cond)

    builder.position_at_end(after)
    return builder.load(effective_size)


def backward_trim_size_wordctlz(builder, size, buffer, intp, int8):
    int1 = ir.IntType(1)
    int64 = ir.IntType(64)
    int64_ptr = int64.as_pointer()
    effective_size = cgutils.alloca_once(builder, intp)
    builder.store(size, effective_size)

    word_size = ir.Constant(intp, 8)
    high_mask = ir.Constant(int64, 0x8080808080808080)
    one_mask = ir.Constant(int64, 0x0101010101010101)
    ctlz_type = ir.FunctionType(int64, [int64, int1])
    ctlz = cgutils.get_or_insert_function(
        builder.module, ctlz_type, 'llvm.ctlz.i64',
    )

    cond = builder.append_basic_block('trim_word.cond')
    check = builder.append_basic_block('trim_word.check')
    zero_word = builder.append_basic_block('trim_word.zero_word')
    partial_word = builder.append_basic_block('trim_word.partial_word')
    byte_trim = builder.append_basic_block('trim_word.byte_trim')
    after = builder.append_basic_block('trim_word.after')

    builder.branch(cond)
    builder.position_at_end(cond)
    has_word = builder.icmp_unsigned('>=', builder.load(effective_size),
                                     word_size)
    builder.cbranch(has_word, check, byte_trim)

    builder.position_at_end(check)
    base = builder.sub(builder.load(effective_size), word_size)
    word_ptr = builder.gep(buffer, [base])
    word_load = builder.load(builder.bitcast(word_ptr, int64_ptr))
    word_load.align = 1
    zero_bits = builder.and_(
        builder.and_(builder.sub(word_load, one_mask),
                     builder.not_(word_load)),
        high_mask,
    )
    nonzero_bits = builder.and_(builder.not_(zero_bits), high_mask)
    any_nonzero = builder.icmp_unsigned(
        '!=', nonzero_bits, ir.Constant(int64, 0),
    )
    builder.cbranch(any_nonzero, partial_word, zero_word)

    builder.position_at_end(zero_word)
    builder.store(base, effective_size)
    builder.branch(cond)

    builder.position_at_end(partial_word)
    leading = builder.call(ctlz, [nonzero_bits, ir.Constant(int1, 0)])
    highest = builder.sub(ir.Constant(int64, 63), leading)
    byte_offset64 = builder.lshr(highest, ir.Constant(int64, 3))
    if intp.width != 64:
        byte_offset = builder.trunc(byte_offset64, intp)
    else:
        byte_offset = byte_offset64
    builder.store(builder.add(builder.add(base, byte_offset),
                              ir.Constant(intp, 1)),
                  effective_size)
    builder.branch(after)

    builder.position_at_end(byte_trim)
    byte_cond = builder.append_basic_block('trim_word.byte_cond')
    byte_check = builder.append_basic_block('trim_word.byte_check')
    byte_body = builder.append_basic_block('trim_word.byte_body')
    builder.branch(byte_cond)

    builder.position_at_end(byte_cond)
    has_remaining = builder.icmp_unsigned('>', builder.load(effective_size),
                                          ir.Constant(intp, 0))
    builder.cbranch(has_remaining, byte_check, after)

    builder.position_at_end(byte_check)
    previous = builder.sub(builder.load(effective_size),
                           ir.Constant(intp, 1))
    char = builder.load(builder.gep(buffer, [previous]))
    is_zero = builder.icmp_unsigned('==', char, ir.Constant(int8, 0))
    builder.cbranch(is_zero, byte_body, after)

    builder.position_at_end(byte_body)
    builder.store(previous, effective_size)
    builder.branch(byte_cond)

    builder.position_at_end(after)
    return builder.load(effective_size)


def continuation_count_word(builder, size, buffer, intp, int8):
    int64 = ir.IntType(64)
    int64_ptr = int64.as_pointer()
    count = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), count)

    word_size = ir.Constant(intp, 8)
    word_count = builder.udiv(size, word_size)
    byte_count = builder.mul(word_count, word_size)
    ctpop_type = ir.FunctionType(int64, [int64])
    ctpop = cgutils.get_or_insert_function(
        builder.module, ctpop_type, 'llvm.ctpop.i64',
    )
    high_mask = ir.Constant(int64, 0x8080808080808080)
    second_mask = ir.Constant(int64, 0x4040404040404040)

    with cgutils.for_range(builder, word_count, intp=intp) as loop:
        byte_index = builder.mul(loop.index, word_size)
        byte_ptr = builder.gep(buffer, [byte_index])
        word_load = builder.load(builder.bitcast(byte_ptr, int64_ptr))
        word_load.align = 1
        high_bits = builder.and_(word_load, high_mask)
        second_as_high = builder.shl(
            builder.and_(word_load, second_mask), ir.Constant(int64, 1),
        )
        continuation_bits = builder.and_(high_bits,
                                         builder.not_(second_as_high))
        pop = builder.call(ctpop, [continuation_bits])
        if intp.width != 64:
            pop = builder.trunc(pop, intp)
        builder.store(builder.add(builder.load(count), pop), count)

    tail_size = builder.sub(size, byte_count)
    with cgutils.for_range(builder, tail_size, intp=intp) as loop:
        byte_index = builder.add(byte_count, loop.index)
        char = builder.load(builder.gep(buffer, [byte_index]))
        tag = builder.and_(char, ir.Constant(int8, 0xc0))
        continuation = builder.icmp_unsigned(
            '==', tag, ir.Constant(int8, 0x80),
        )
        increment = builder.select(
            continuation, ir.Constant(intp, 1), ir.Constant(intp, 0),
        )
        builder.store(builder.add(builder.load(count), increment), count)

    return builder.load(count)


def backward_effective_count(builder, size, buffer, intp, int8):
    effective_size = backward_trim_size(builder, size, buffer, intp, int8)
    return byte_prefix_codepoint_count(builder, effective_size, buffer,
                                       intp, int8)


def backward_word_effective_count(builder, size, buffer, intp, int8):
    effective_size = backward_trim_size(builder, size, buffer, intp, int8)
    continuations = continuation_count_word(builder, effective_size, buffer,
                                           intp, int8)
    return builder.sub(effective_size, continuations)


def wordtrim_word_effective_count(builder, size, buffer, intp, int8):
    effective_size = backward_trim_size_wordctlz(builder, size, buffer,
                                                intp, int8)
    continuations = continuation_count_word(builder, effective_size, buffer,
                                           intp, int8)
    return builder.sub(effective_size, continuations)


def peeled16_effective_count(builder, size, buffer, intp, int8):
    count = cgutils.alloca_once(builder, intp)
    effective_count = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), count)
    builder.store(ir.Constant(intp, 0), effective_count)

    for offset in range(_PACKED_STRING_SIZE):
        lane = ir.Constant(intp, offset)
        in_range = builder.icmp_unsigned('>', size, lane)
        with builder.if_then(in_range):
            char = builder.load(builder.gep(buffer, [lane]))
            tag = builder.and_(char, ir.Constant(int8, 0xc0))
            continuation = builder.icmp_unsigned(
                '==', tag, ir.Constant(int8, 0x80),
            )
            increment = builder.select(
                continuation, ir.Constant(intp, 0), ir.Constant(intp, 1),
            )
            next_count = builder.add(builder.load(count), increment)
            builder.store(next_count, count)
            nonzero = builder.icmp_unsigned('!=', char, ir.Constant(int8, 0))
            builder.store(
                builder.select(nonzero, next_count,
                               builder.load(effective_count)),
                effective_count,
            )

    return builder.load(effective_count)


def peeled16_word_hybrid_effective_count(builder, size, buffer, intp, int8):
    result = cgutils.alloca_once(builder, intp)
    small = builder.icmp_unsigned(
        '<=', size, ir.Constant(intp, _PACKED_STRING_SIZE),
    )
    with builder.if_else(small) as (then, otherwise):
        with then:
            builder.store(
                peeled16_effective_count(builder, size, buffer, intp, int8),
                result,
            )
        with otherwise:
            builder.store(
                backward_word_effective_count(builder, size, buffer,
                                              intp, int8),
                result,
            )
    return builder.load(result)


def make_data_len_intrinsic(effective_count):
    @intrinsic
    def impl(typingctx, data, index, allocator):
        if data != types.voidptr \
                or not isinstance(index, types.Integer) \
                or allocator != types.voidptr:
            return None

        sig = signature(types.intp, data, types.intp, allocator)

        def codegen(context, builder, signature, args):
            data, index_value, allocator = args
            int8 = ir.IntType(8)
            int32 = ir.IntType(32)
            intp = context.get_value_type(types.intp)
            byte_ptr = int8.as_pointer()
            offset = builder.mul(index_value,
                                 ir.Constant(intp, _PACKED_STRING_SIZE))
            packed = builder.gep(builder.bitcast(data, byte_ptr), [offset])
            status, size, buffer = load_string(
                builder, allocator, packed, intp, byte_ptr,
            )

            result = cgutils.alloca_once(builder, intp)
            builder.store(ir.Constant(intp, -1), result)
            valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
            with builder.if_then(valid):
                builder.store(
                    effective_count(builder, size, buffer, intp, int8),
                    result,
                )
            return builder.load(result)

        return sig, codegen

    return impl


codepoint_len_backward_data = make_data_len_intrinsic(
    backward_effective_count,
)
codepoint_len_backward_word_data = make_data_len_intrinsic(
    backward_word_effective_count,
)
codepoint_len_wordtrim_word_data = make_data_len_intrinsic(
    wordtrim_word_effective_count,
)
codepoint_len_peeled16_word_hybrid_data = make_data_len_intrinsic(
    peeled16_word_hybrid_effective_count,
)


@njit(nogil=True, cache=False)
def current_strlen(values):
    return np.strings.str_len(values)


@njit(nogil=True, cache=False)
def backward_data_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = codepoint_len_backward_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def backward_word_data_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = codepoint_len_backward_word_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def wordtrim_word_data_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = codepoint_len_wordtrim_word_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def peeled16_word_hybrid_data_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = codepoint_len_peeled16_word_hybrid_data(
            data, i, allocator,
        )
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def runtime_intrinsic_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = stringdtype_acquire_allocator(values)
    data = stringdtype_data_ptr(values)
    for i in range(values.size):
        result[i] = stringdtype_codepoint_len_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


def bench_methods(methods, values, repeat=15):
    expected = np.strings.str_len(values)
    valid_methods = []
    for name, method in methods:
        try:
            result = method(values)
            np.testing.assert_array_equal(result, expected)
        except Exception as exc:  # pragma: no cover - exploration output
            print(f'{name:<28} correctness failed: {type(exc).__name__}')
            continue
        valid_methods.append((name, method))

    timings = {name: [] for name, _ in valid_methods}
    order = valid_methods * repeat
    random.Random(1337).shuffle(order)
    for name, method in order:
        start = time.perf_counter_ns()
        method(values)
        timings[name].append((time.perf_counter_ns() - start) / 1e6)

    medians = {}
    for name, samples in timings.items():
        medians[name] = statistics.median(samples)
        print(f'{name:<28} min {min(samples):8.3f} ms  '
              f'median {medians[name]:8.3f} ms')

    numpy_median = medians.get('numpy')
    if numpy_median:
        for name, median in medians.items():
            if name == 'numpy':
                continue
            print(f'{name:<28} median speedup {numpy_median / median:8.3f}x')


def make_values(pattern, n):
    repeats = (n + len(pattern) - 1) // len(pattern)
    return np.array((pattern * repeats)[:n], dtype=np.dtypes.StringDType())


def main():
    cases = [
        ('mixed-short', ['a', 'abc', 'é', '🙂', '', 'abc\x00',
                         'a\x00b', '漢']),
        ('ascii-short', ['a', 'abc', 'abcdef', '', 'abc\x00', 'xyz']),
        ('nul-heavy', ['\x00', 'a\x00\x00', 'abc\x00\x00',
                       '\x00\x00', 'é\x00']),
        ('unicode-short', ['é', '🙂', '漢字', 'κόσμε', 'aé🙂', '']),
        ('long-mixed', ['a' * 64, 'é' * 32, '🙂' * 16,
                        'abc\x00' * 16, '漢字' * 16]),
        ('long-ascii', ['a' * 64, 'b' * 128, 'c' * 256,
                        'alphabet' * 32, 'xyz' * 96]),
        ('long-nul-tail', ['a' * 16 + '\x00' * 48,
                           'é' * 16 + '\x00' * 32,
                           '🙂' * 8 + '\x00' * 32,
                           '\x00' * 64,
                           'abc' * 16 + '\x00' * 16]),
    ]
    methods = [
        ('numpy', np.strings.str_len),
        ('current runtime', current_strlen),
        ('runtime intrinsic', runtime_intrinsic_strlen),
        ('backward data', backward_data_strlen),
        ('backward word', backward_word_data_strlen),
        ('wordtrim word', wordtrim_word_data_strlen),
        ('peeled16 word-hybrid', peeled16_word_hybrid_data_strlen),
    ]

    for case_name, pattern in cases:
        for n in [1000, 100000]:
            values = make_values(pattern, n)
            print(f'\ncase={case_name} n={n}')
            bench_methods(methods, values)


if __name__ == '__main__':
    main()
