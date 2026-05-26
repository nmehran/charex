"""Compare StringDType descriptor/allocator access strategies.

This is exploratory code for Tranche 1. It intentionally tests approaches that
may be rejected from the final implementation.
"""

import ctypes
from pathlib import Path
import random
import statistics
import subprocess
import sys
import sysconfig
import tempfile
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
    stringdtype_codepoint_len,
    stringdtype_release_allocator,
)


_SLOTS = charex.numpy.stringdtype._API_SLOTS
_NpyString_acquire_allocator = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p,
)(_SLOTS[316])
_DEFAULT_DTYPE = np.dtype('T')
_DEFAULT_DTYPE_ADDR = id(_DEFAULT_DTYPE)
_C_HELPER_ACQUIRE_ADDR = 0
_C_HELPER_LIBRARY = None
_PACKED_STRING_SIZE = 16


_ACQUIRE_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)


@_ACQUIRE_CALLBACK
def _callback_acquire(parent_addr):
    parent = ctypes.cast(parent_addr, ctypes.py_object).value
    return _NpyString_acquire_allocator(id(parent.dtype))


_CALLBACK_ACQUIRE_ADDR = ctypes.cast(
    _callback_acquire, ctypes.c_void_p,
).value


@intrinsic
def callback_acquire_allocator(typingctx, array):
    if not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        array_struct = context.make_array(signature.args[0])(
            context, builder, args[0],
        )
        uintp = context.get_value_type(types.uintp)
        callback_type = ir.FunctionType(ir.IntType(8).as_pointer(), [uintp])
        callback_addr = context.get_constant(
            types.uintp, _CALLBACK_ACQUIRE_ADDR,
        )
        callback = builder.inttoptr(
            callback_addr, callback_type.as_pointer(),
        )
        parent_addr = builder.ptrtoint(array_struct.parent, uintp)
        return builder.call(callback, [parent_addr])

    return sig, codegen


@intrinsic
def default_descriptor_acquire_allocator(typingctx, array):
    if not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        byte_ptr = ir.IntType(8).as_pointer()
        acquire_type = ir.FunctionType(byte_ptr, [byte_ptr])
        acquire = builder.module.globals.get(
            'charex_NpyString_acquire_allocator',
        )
        if acquire is None:
            acquire = ir.Function(
                builder.module,
                acquire_type,
                'charex_NpyString_acquire_allocator',
            )
        descriptor_addr = context.get_constant(
            types.uintp, _DEFAULT_DTYPE_ADDR,
        )
        descriptor = builder.inttoptr(descriptor_addr, byte_ptr)
        return builder.call(acquire, [descriptor])

    return sig, codegen


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


def onepass_effective_count(builder, size, buffer, intp, int8):
    count = cgutils.alloca_once(builder, intp)
    effective_count = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), count)
    builder.store(ir.Constant(intp, 0), effective_count)

    with cgutils.for_range(builder, size, intp=intp) as loop:
        char = builder.load(builder.gep(buffer, [loop.index]))
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


def onepass_effective_size(builder, size, buffer, intp, int8):
    continuation_count = cgutils.alloca_once(builder, intp)
    effective_continuations = cgutils.alloca_once(builder, intp)
    effective_size = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), continuation_count)
    builder.store(ir.Constant(intp, 0), effective_continuations)
    builder.store(ir.Constant(intp, 0), effective_size)

    with cgutils.for_range(builder, size, intp=intp) as loop:
        char = builder.load(builder.gep(buffer, [loop.index]))
        tag = builder.and_(char, ir.Constant(int8, 0xc0))
        continuation = builder.icmp_unsigned(
            '==', tag, ir.Constant(int8, 0x80),
        )
        increment = builder.select(
            continuation, ir.Constant(intp, 1), ir.Constant(intp, 0),
        )
        next_continuations = builder.add(
            builder.load(continuation_count), increment,
        )
        builder.store(next_continuations, continuation_count)

        nonzero = builder.icmp_unsigned('!=', char, ir.Constant(int8, 0))
        next_size = builder.add(loop.index, ir.Constant(intp, 1))
        builder.store(
            builder.select(nonzero, next_size,
                           builder.load(effective_size)),
            effective_size,
        )
        builder.store(
            builder.select(nonzero, next_continuations,
                           builder.load(effective_continuations)),
            effective_continuations,
        )

    return builder.sub(
        builder.load(effective_size), builder.load(effective_continuations),
    )


def twopass_effective_count(builder, size, buffer, intp, int8):
    effective_size = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), effective_size)
    with cgutils.for_range(builder, size, intp=intp) as loop:
        char = builder.load(builder.gep(buffer, [loop.index]))
        is_nonzero = builder.icmp_unsigned(
            '!=', char, ir.Constant(int8, 0),
        )
        next_size = builder.add(loop.index, ir.Constant(intp, 1))
        builder.store(
            builder.select(is_nonzero, next_size,
                           builder.load(effective_size)),
            effective_size,
        )

    count = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), count)
    with cgutils.for_range(builder, builder.load(effective_size),
                           intp=intp) as loop:
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


def backward_trim_effective_count(builder, size, buffer, intp, int8):
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
    count = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), count)
    with cgutils.for_range(builder, builder.load(effective_size),
                           intp=intp) as loop:
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


@intrinsic
def stringdtype_data_ptr(typingctx, array):
    if not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        array_struct = context.make_array(signature.args[0])(
            context, builder, args[0],
        )
        return builder.bitcast(array_struct.data, ir.IntType(8).as_pointer())

    return sig, codegen


@intrinsic
def codepoint_len_twopass_data(typingctx, data, index, allocator):
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
        status, size, buffer = load_string(builder, allocator, packed, intp,
                                           byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            builder.store(
                twopass_effective_count(builder, size, buffer, intp, int8),
                result,
            )
        return builder.load(result)

    return sig, codegen


@intrinsic
def codepoint_len_backward_data(typingctx, data, index, allocator):
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
        status, size, buffer = load_string(builder, allocator, packed, intp,
                                           byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            builder.store(
                backward_trim_effective_count(builder, size, buffer,
                                              intp, int8),
                result,
            )
        return builder.load(result)

    return sig, codegen


@intrinsic
def codepoint_len_peeled16_hybrid_data(typingctx, data, index, allocator):
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
        status, size, buffer = load_string(builder, allocator, packed, intp,
                                           byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            is_small = builder.icmp_unsigned(
                '<=', size, ir.Constant(intp, _PACKED_STRING_SIZE),
            )
            with builder.if_else(is_small) as (then, otherwise):
                with then:
                    builder.store(
                        peeled16_effective_count(builder, size, buffer,
                                                 intp, int8),
                        result,
                    )
                with otherwise:
                    builder.store(
                        twopass_effective_count(builder, size, buffer,
                                                intp, int8),
                        result,
                    )
        return builder.load(result)

    return sig, codegen


@intrinsic
def codepoint_len_peeled16_backward_hybrid_data(typingctx, data, index,
                                                allocator):
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
        status, size, buffer = load_string(builder, allocator, packed, intp,
                                           byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            is_small = builder.icmp_unsigned(
                '<=', size, ir.Constant(intp, _PACKED_STRING_SIZE),
            )
            with builder.if_else(is_small) as (then, otherwise):
                with then:
                    builder.store(
                        peeled16_effective_count(builder, size, buffer,
                                                 intp, int8),
                        result,
                    )
                with otherwise:
                    builder.store(
                        backward_trim_effective_count(builder, size, buffer,
                                                      intp, int8),
                        result,
                    )
        return builder.load(result)

    return sig, codegen


@intrinsic
def codepoint_len_onepass_array(typingctx, array, index, allocator):
    if not is_stringdtype_array_type(array) \
            or not isinstance(index, types.Integer) \
            or allocator != types.voidptr:
        return None

    sig = signature(types.intp, array, types.intp, allocator)

    def codegen(context, builder, signature, args):
        array_type = signature.args[0]
        array_value, index_value, allocator = args
        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        array_struct = context.make_array(array_type)(
            context, builder, array_value,
        )
        data = builder.bitcast(array_struct.data, byte_ptr)
        stride = builder.extract_value(array_struct.strides, 0)
        packed = builder.gep(data, [builder.mul(index_value, stride)])
        status, size, buffer = load_string(builder, allocator, packed, intp,
                                           byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            builder.store(
                onepass_effective_count(builder, size, buffer, intp, int8),
                result,
            )
        return builder.load(result)

    return sig, codegen


@intrinsic
def codepoint_len_onepass_data(typingctx, data, index, allocator):
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
        status, size, buffer = load_string(builder, allocator, packed, intp,
                                           byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            builder.store(
                onepass_effective_count(builder, size, buffer, intp, int8),
                result,
            )
        return builder.load(result)

    return sig, codegen


@intrinsic
def codepoint_len_hybrid16_data(typingctx, data, index, allocator):
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
        status, size, buffer = load_string(builder, allocator, packed, intp,
                                           byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            is_small = builder.icmp_unsigned(
                '<=', size, ir.Constant(intp, _PACKED_STRING_SIZE),
            )
            with builder.if_else(is_small) as (then, otherwise):
                with then:
                    builder.store(
                        onepass_effective_count(builder, size, buffer,
                                                intp, int8),
                        result,
                    )
                with otherwise:
                    builder.store(
                        twopass_effective_count(builder, size, buffer,
                                                intp, int8),
                        result,
                    )
        return builder.load(result)

    return sig, codegen


@intrinsic
def codepoint_len_hybrid32_data(typingctx, data, index, allocator):
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
        status, size, buffer = load_string(builder, allocator, packed, intp,
                                           byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            is_small = builder.icmp_unsigned(
                '<=', size, ir.Constant(intp, _PACKED_STRING_SIZE * 2),
            )
            with builder.if_else(is_small) as (then, otherwise):
                with then:
                    builder.store(
                        onepass_effective_count(builder, size, buffer,
                                                intp, int8),
                        result,
                    )
                with otherwise:
                    builder.store(
                        twopass_effective_count(builder, size, buffer,
                                                intp, int8),
                        result,
                    )
        return builder.load(result)

    return sig, codegen


@intrinsic
def codepoint_len_onepass_size_data(typingctx, data, index, allocator):
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
        status, size, buffer = load_string(builder, allocator, packed, intp,
                                           byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            builder.store(
                onepass_effective_size(builder, size, buffer, intp, int8),
                result,
            )
        return builder.load(result)

    return sig, codegen


@intrinsic
def codepoint_len_onepass_unchecked_data(typingctx, data, index, allocator):
    if data != types.voidptr \
            or not isinstance(index, types.Integer) \
            or allocator != types.voidptr:
        return None

    sig = signature(types.intp, data, types.intp, allocator)

    def codegen(context, builder, signature, args):
        data, index_value, allocator = args
        int8 = ir.IntType(8)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        offset = builder.mul(index_value,
                             ir.Constant(intp, _PACKED_STRING_SIZE))
        packed = builder.gep(builder.bitcast(data, byte_ptr), [offset])
        _status, size, buffer = load_string(builder, allocator, packed, intp,
                                            byte_ptr)
        return onepass_effective_count(builder, size, buffer, intp, int8)

    return sig, codegen


def build_c_helper():
    source = r'''
#define PY_SSIZE_T_CLEAN
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include <Python.h>
#include <stdint.h>
#include <numpy/arrayobject.h>

int charex_bench_import_array(void) {
    import_array1(-1);
    return 0;
}

void *charex_bench_acquire_allocator(PyObject *array) {
    PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *)array);
    return NpyString_acquire_allocator((PyArray_StringDTypeObject *)descr);
}

static int64_t codepoint_len_twopass(const npy_static_string *string) {
    size_t effective_size = 0;
    for (size_t i = 0; i < string->size; i++) {
        if ((unsigned char)string->buf[i] != 0) {
            effective_size = i + 1;
        }
    }

    int64_t count = 0;
    for (size_t i = 0; i < effective_size; i++) {
        if (((unsigned char)string->buf[i] & 0xc0) != 0x80) {
            count++;
        }
    }
    return count;
}

static int64_t codepoint_len_onepass_size(const npy_static_string *string) {
    size_t effective_size = 0;
    int64_t continuations = 0;
    int64_t effective_continuations = 0;
    for (size_t i = 0; i < string->size; i++) {
        unsigned char c = (unsigned char)string->buf[i];
        continuations += ((c & 0xc0) == 0x80);
        if (c != 0) {
            effective_size = i + 1;
            effective_continuations = continuations;
        }
    }
    return (int64_t)effective_size - effective_continuations;
}

int charex_bench_strlen_twopass(PyObject *array, int64_t *out) {
    PyArrayObject *arr = (PyArrayObject *)array;
    PyArray_Descr *descr = PyArray_DESCR(arr);
    npy_string_allocator *allocator = NpyString_acquire_allocator(
        (PyArray_StringDTypeObject *)descr
    );
    char *data = PyArray_BYTES(arr);
    npy_intp stride = PyArray_STRIDES(arr)[0];
    npy_intp size = PyArray_SIZE(arr);
    for (npy_intp i = 0; i < size; i++) {
        npy_static_string string = {0, NULL};
        int status = NpyString_load(
            allocator,
            (npy_packed_static_string *)(data + i * stride),
            &string
        );
        if (status != 0) {
            NpyString_release_allocator(allocator);
            return status;
        }
        out[i] = codepoint_len_twopass(&string);
    }
    NpyString_release_allocator(allocator);
    return 0;
}

int charex_bench_strlen_onepass_size(PyObject *array, int64_t *out) {
    PyArrayObject *arr = (PyArrayObject *)array;
    PyArray_Descr *descr = PyArray_DESCR(arr);
    npy_string_allocator *allocator = NpyString_acquire_allocator(
        (PyArray_StringDTypeObject *)descr
    );
    char *data = PyArray_BYTES(arr);
    npy_intp stride = PyArray_STRIDES(arr)[0];
    npy_intp size = PyArray_SIZE(arr);
    for (npy_intp i = 0; i < size; i++) {
        npy_static_string string = {0, NULL};
        int status = NpyString_load(
            allocator,
            (npy_packed_static_string *)(data + i * stride),
            &string
        );
        if (status != 0) {
            NpyString_release_allocator(allocator);
            return status;
        }
        out[i] = codepoint_len_onepass_size(&string);
    }
    NpyString_release_allocator(allocator);
    return 0;
}
'''
    build_dir = Path(tempfile.mkdtemp(prefix='charex-stringdtype-c-'))
    source_path = build_dir / 'helper.c'
    output_path = build_dir / 'helper.so'
    source_path.write_text(source)
    includes = [
        sysconfig.get_path('include'),
        sysconfig.get_path('platinclude'),
        np.get_include(),
    ]
    command = [
        'gcc', '-shared', '-fPIC', '-O3',
        *(f'-I{path}' for path in includes if path),
        str(source_path), '-o', str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    library = ctypes.PyDLL(str(output_path))
    library.charex_bench_import_array.argtypes = []
    library.charex_bench_import_array.restype = ctypes.c_int
    if library.charex_bench_import_array() != 0:
        raise RuntimeError('failed to initialize NumPy C API')
    library.charex_bench_acquire_allocator.argtypes = [ctypes.py_object]
    library.charex_bench_acquire_allocator.restype = ctypes.c_void_p
    library.charex_bench_strlen_twopass.argtypes = [
        ctypes.py_object, ctypes.POINTER(ctypes.c_int64),
    ]
    library.charex_bench_strlen_twopass.restype = ctypes.c_int
    library.charex_bench_strlen_onepass_size.argtypes = [
        ctypes.py_object, ctypes.POINTER(ctypes.c_int64),
    ]
    library.charex_bench_strlen_onepass_size.restype = ctypes.c_int
    address = ctypes.cast(
        library.charex_bench_acquire_allocator,
        ctypes.c_void_p,
    ).value
    strlen_twopass = ctypes.cast(
        library.charex_bench_strlen_twopass,
        ctypes.c_void_p,
    ).value
    strlen_onepass_size = ctypes.cast(
        library.charex_bench_strlen_onepass_size,
        ctypes.c_void_p,
    ).value
    return library, address, strlen_twopass, strlen_onepass_size


try:
    _C_HELPER_LIBRARY, _C_HELPER_ACQUIRE_ADDR, \
        _C_HELPER_STRLEN_TWOPASS_ADDR, \
        _C_HELPER_STRLEN_ONEPASS_SIZE_ADDR = build_c_helper()
except Exception as exc:  # pragma: no cover - exploration-only fallback
    print(f'C helper unavailable: {type(exc).__name__}: {exc}')
    _C_HELPER_STRLEN_TWOPASS_ADDR = 0
    _C_HELPER_STRLEN_ONEPASS_SIZE_ADDR = 0


@intrinsic
def c_helper_acquire_allocator(typingctx, array):
    if not _C_HELPER_ACQUIRE_ADDR or not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        array_struct = context.make_array(signature.args[0])(
            context, builder, args[0],
        )
        uintp = context.get_value_type(types.uintp)
        callback_type = ir.FunctionType(ir.IntType(8).as_pointer(), [uintp])
        callback_addr = context.get_constant(
            types.uintp, _C_HELPER_ACQUIRE_ADDR,
        )
        callback = builder.inttoptr(
            callback_addr, callback_type.as_pointer(),
        )
        parent_addr = builder.ptrtoint(array_struct.parent, uintp)
        return builder.call(callback, [parent_addr])

    return sig, codegen


def c_helper_batch_intrinsic(address):
    @intrinsic
    def impl(typingctx, array, out):
        if not address \
                or not is_stringdtype_array_type(array) \
                or not isinstance(out, types.Array) \
                or out.dtype != types.int64 \
                or out.ndim != 1 \
                or out.layout != 'C':
            return None

        sig = signature(types.int32, array, out)

        def codegen(context, builder, signature, args):
            array_struct = context.make_array(signature.args[0])(
                context, builder, args[0],
            )
            out_struct = context.make_array(signature.args[1])(
                context, builder, args[1],
            )
            int32 = ir.IntType(32)
            byte_ptr = ir.IntType(8).as_pointer()
            int64_ptr = ir.IntType(64).as_pointer()
            helper_type = ir.FunctionType(int32, [byte_ptr, int64_ptr])
            helper_addr = context.get_constant(types.uintp, address)
            helper = builder.inttoptr(helper_addr, helper_type.as_pointer())
            parent = builder.bitcast(array_struct.parent, byte_ptr)
            out_data = builder.bitcast(out_struct.data, int64_ptr)
            return builder.call(helper, [parent, out_data])

        return sig, codegen

    return impl


c_helper_batch_strlen_twopass = c_helper_batch_intrinsic(
    _C_HELPER_STRLEN_TWOPASS_ADDR,
)
c_helper_batch_strlen_onepass_size = c_helper_batch_intrinsic(
    _C_HELPER_STRLEN_ONEPASS_SIZE_ADDR,
)


@njit(nogil=True, cache=False)
def current_strlen(values):
    return np.strings.str_len(values)


@njit(nogil=True, cache=False)
def current_per_element_strlen(values):
    result = np.empty(values.size, np.int64)
    for i in range(values.size):
        allocator = stringdtype_acquire_allocator(values)
        result[i] = stringdtype_codepoint_len(values, i, allocator)
        stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def callback_acquire_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = callback_acquire_allocator(values)
    for i in range(values.size):
        result[i] = stringdtype_codepoint_len(values, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def default_descriptor_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = default_descriptor_acquire_allocator(values)
    for i in range(values.size):
        result[i] = stringdtype_codepoint_len(values, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def onepass_array_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_onepass_array(values, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def twopass_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_twopass_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def backward_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_backward_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def peeled16_hybrid_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_peeled16_hybrid_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def peeled16_backward_hybrid_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_peeled16_backward_hybrid_data(
            data, i, allocator,
        )
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def onepass_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_onepass_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def hybrid16_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_hybrid16_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def hybrid32_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_hybrid32_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def onepass_size_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_onepass_size_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def onepass_unchecked_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = stringdtype_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_onepass_unchecked_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def c_helper_onepass_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = c_helper_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_onepass_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def c_helper_hybrid16_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = c_helper_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_hybrid16_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def c_helper_peeled16_hybrid_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = c_helper_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_peeled16_hybrid_data(data, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def c_helper_peeled16_backward_hybrid_data_strlen(values):
    result = np.empty(values.size, np.int64)
    data = stringdtype_data_ptr(values)
    allocator = c_helper_acquire_allocator(values)
    for i in range(values.size):
        result[i] = codepoint_len_peeled16_backward_hybrid_data(
            data, i, allocator,
        )
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def c_helper_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = c_helper_acquire_allocator(values)
    for i in range(values.size):
        result[i] = stringdtype_codepoint_len(values, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def c_helper_batch_twopass_strlen(values):
    result = np.empty(values.size, np.int64)
    status = c_helper_batch_strlen_twopass(values, result)
    if status != 0:
        raise ValueError('StringDType load failed')
    return result


@njit(nogil=True, cache=False)
def c_helper_batch_onepass_size_strlen(values):
    result = np.empty(values.size, np.int64)
    status = c_helper_batch_strlen_onepass_size(values, result)
    if status != 0:
        raise ValueError('StringDType load failed')
    return result


def bench_methods(methods, values, repeat=15):
    number = max(1, min(1000, 100000 // max(values.size, 1)))
    valid_methods = []

    for label, fn in methods:
        try:
            np.testing.assert_array_equal(fn(values), np.strings.str_len(values))
        except Exception as exc:
            print(f'{label:28} correctness failed: {type(exc).__name__}')
        else:
            valid_methods.append((label, fn))

    samples = {label: [] for label, _ in valid_methods}
    order = list(valid_methods)
    rng = random.Random(1009 + values.size)

    for _ in range(repeat):
        rng.shuffle(order)
        for label, fn in order:
            start = time.perf_counter()
            for _ in range(number):
                fn(values)
            samples[label].append((time.perf_counter() - start) / number)

    results = {}
    for label in samples:
        result = {
            'min': min(samples[label]) * 1000,
            'median': statistics.median(samples[label]) * 1000,
        }
        results[label] = result
        print(
            f'{label:28} min {result["min"]:8.3f} ms  '
            f'median {result["median"]:8.3f} ms'
        )
    return results


def make_values(pattern, n):
    repeats = (n + len(pattern) - 1) // len(pattern)
    return np.array((pattern * repeats)[:n], dtype='T')


def main():
    cases = [
        ('mixed-short', ['alpha', 'é', '🙂', '', 'a\x00b']),
        ('ascii-short', ['alpha', 'beta', 'gamma', 'delta', 'epsilon']),
        ('nul-heavy', ['', '\x00', '\x00\x00', 'a\x00', 'ab\x00\x00']),
        ('unicode-short', ['ééé', '🙂🙂', 'αβγδ', '漢字仮名', 'a🙂é']),
        ('long-mixed', ['a' * 64, 'é' * 32, '🙂' * 16,
                        'abc\x00' * 16, '漢字' * 16]),
    ]
    for case_name, pattern in cases:
        for n in [1000, 100000]:
            values = make_values(pattern, n)
            print(f'\ncase={case_name} n={n}')
            methods = [
                ('numpy', np.strings.str_len),
                ('current parent-offset', current_strlen),
                ('twopass data', twopass_data_strlen),
                ('backward data', backward_data_strlen),
                ('peeled16 hybrid', peeled16_hybrid_data_strlen),
                ('peeled16 back-hybrid',
                 peeled16_backward_hybrid_data_strlen),
                ('onepass array', onepass_array_strlen),
                ('onepass data', onepass_data_strlen),
                ('onepass size-data', onepass_size_data_strlen),
                ('hybrid16 data', hybrid16_data_strlen),
                ('hybrid32 data', hybrid32_data_strlen),
                ('onepass unchecked', onepass_unchecked_data_strlen),
                ('acquire per element', current_per_element_strlen),
                ('callback acquire', callback_acquire_strlen),
                ('default descriptor', default_descriptor_strlen),
            ]
            if _C_HELPER_ACQUIRE_ADDR:
                methods.append(('C helper acquire', c_helper_strlen))
                methods.append(('C helper onepass',
                                c_helper_onepass_data_strlen))
                methods.append(('C helper hybrid16',
                                c_helper_hybrid16_data_strlen))
                methods.append(('C helper peeled16',
                                c_helper_peeled16_hybrid_data_strlen))
                methods.append(('C helper peeled16-back',
                                c_helper_peeled16_backward_hybrid_data_strlen))
                if _C_HELPER_STRLEN_TWOPASS_ADDR:
                    methods.append(('C batch twopass',
                                    c_helper_batch_twopass_strlen))
                if _C_HELPER_STRLEN_ONEPASS_SIZE_ADDR:
                    methods.append(('C batch onepass-size',
                                    c_helper_batch_onepass_size_strlen))
            results = bench_methods(methods, values)
            baseline = results['numpy']['median']
            for label in results:
                if label != 'numpy':
                    print(
                        f'{label:28} median speedup '
                        f'{baseline / results[label]["median"]:8.3f}x'
                    )


if __name__ == '__main__':
    main()
