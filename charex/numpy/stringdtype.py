"""Experimental NumPy StringDType typing support."""

import ctypes

from llvmlite import binding as llvm
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel import models, register_default
from numba.core.errors import NumbaValueError
from numba.core.typing import signature
from numba.core.typing.typeof import typeof_impl
from numba.extending import intrinsic
from numba.np import numpy_support
import numpy as np


_STRING_DTYPE = getattr(getattr(np, 'dtypes', None), 'StringDType', None)


class StringDTypePacket(types.Type):
    """Numba type for StringDType's 16-byte packed element record."""

    def __init__(self):
        super().__init__('StringDTypePacket')


stringdtype_packet = StringDTypePacket()


def _numpy_api_slots():
    capsule = np._core.multiarray._ARRAY_API
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    api = get_pointer(capsule, None)
    return (ctypes.c_void_p * 400).from_address(api)


class _StaticString(ctypes.Structure):
    _fields_ = [('size', ctypes.c_size_t), ('buf', ctypes.c_void_p)]


if _STRING_DTYPE is not None:
    _API_SLOTS = _numpy_api_slots()
    llvm.add_symbol('charex_NpyString_load', _API_SLOTS[313])
    llvm.add_symbol('charex_NpyString_acquire_allocator', _API_SLOTS[316])
    llvm.add_symbol('charex_NpyString_release_allocator', _API_SLOTS[318])
else:
    _API_SLOTS = None


def _array_descr_offset():
    if _STRING_DTYPE is None:
        return 0
    array = np.array([''], dtype=_STRING_DTYPE())
    target = id(array.dtype)
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    for offset in range(0, 256, pointer_size):
        value = ctypes.c_void_p.from_address(id(array) + offset).value
        if value == target:
            return offset
    raise RuntimeError('could not locate NumPy array descriptor offset')


_ARRAY_DESCR_OFFSET = _array_descr_offset()


@register_default(StringDTypePacket)
class StringDTypePacketModel(models.DataModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type)
        self._be_type = ir.ArrayType(ir.IntType(64), 2)

    def get_value_type(self):
        return self._be_type

    def get_data_type(self):
        return self._be_type

    def as_data(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value


def is_stringdtype(dtype):
    return _STRING_DTYPE is not None and getattr(dtype, 'char', None) == 'T'


def has_stringdtype_na_object(dtype):
    if not is_stringdtype(dtype):
        return False
    try:
        dtype.na_object
    except AttributeError:
        return False
    return True


def is_stringdtype_array_type(value):
    return isinstance(value, types.Array) \
        and isinstance(value.dtype, StringDTypePacket)


def _array_descriptor(context, builder, array_type, array_value):
    intp = context.get_value_type(types.intp)
    byte_ptr = ir.IntType(8).as_pointer()
    array_struct = context.make_array(array_type)(
        context, builder, array_value,
    )
    parent = builder.bitcast(array_struct.parent, byte_ptr)
    descr_addr = builder.gep(
        parent, [ir.Constant(intp, _ARRAY_DESCR_OFFSET)],
    )
    return builder.load(builder.bitcast(descr_addr, byte_ptr.as_pointer()))


def _packed_string_ptr(context, builder, array_type, array_value, index_value):
    int8 = ir.IntType(8)
    byte_ptr = int8.as_pointer()
    array_struct = context.make_array(array_type)(
        context, builder, array_value,
    )
    data = builder.bitcast(array_struct.data, byte_ptr)
    stride = builder.extract_value(array_struct.strides, 0)
    offset = builder.mul(index_value, stride)
    return builder.gep(data, [offset])


def _load_string(builder, allocator, packed, intp, byte_ptr):
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
def stringdtype_acquire_allocator(typingctx, array):
    if not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        byte_ptr = ir.IntType(8).as_pointer()
        descriptor = _array_descriptor(context, builder, signature.args[0],
                                       args[0])
        acquire_type = ir.FunctionType(byte_ptr, [byte_ptr])
        acquire = cgutils.get_or_insert_function(
            builder.module, acquire_type, 'charex_NpyString_acquire_allocator',
        )
        return builder.call(acquire, [descriptor])

    return sig, codegen


@intrinsic
def stringdtype_release_allocator(typingctx, allocator):
    if allocator != types.voidptr:
        return None

    sig = signature(types.void, allocator)

    def codegen(context, builder, signature, args):
        byte_ptr = ir.IntType(8).as_pointer()
        release_type = ir.FunctionType(ir.VoidType(), [byte_ptr])
        release = cgutils.get_or_insert_function(
            builder.module, release_type, 'charex_NpyString_release_allocator',
        )
        builder.call(release, [args[0]])

    return sig, codegen


@intrinsic
def stringdtype_codepoint_len(typingctx, array, index, allocator):
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
        packed = _packed_string_ptr(context, builder, array_type, array_value,
                                    index_value)

        status, size, buffer = _load_string(builder, allocator, packed, intp,
                                            byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))

        with builder.if_then(valid):
            effective_size = cgutils.alloca_once(builder, intp)
            builder.store(ir.Constant(intp, 0), effective_size)
            with cgutils.for_range(builder, size, intp=intp) as loop:
                char = builder.load(builder.gep(buffer, [loop.index]))
                is_nonzero = builder.icmp_unsigned(
                    '!=', char, ir.Constant(int8, 0))
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
                builder.store(builder.add(builder.load(count), increment),
                              count)

            builder.store(builder.load(count), result)

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_equal(typingctx, left, left_index, left_allocator,
                      right, right_index, right_allocator):
    if not is_stringdtype_array_type(left) \
            or not isinstance(left_index, types.Integer) \
            or left_allocator != types.voidptr \
            or not is_stringdtype_array_type(right) \
            or not isinstance(right_index, types.Integer) \
            or right_allocator != types.voidptr:
        return None

    sig = signature(types.boolean, left, types.intp, left_allocator,
                    right, types.intp, right_allocator)

    def codegen(context, builder, signature, args):
        left_type = signature.args[0]
        right_type = signature.args[3]
        left_value, left_index_value, left_allocator, \
            right_value, right_index_value, right_allocator = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        left_packed = _packed_string_ptr(context, builder, left_type,
                                         left_value, left_index_value)
        right_packed = _packed_string_ptr(context, builder, right_type,
                                          right_value, right_index_value)
        left_status, left_size, left_buffer = _load_string(
            builder, left_allocator, left_packed, intp, byte_ptr)
        right_status, right_size, right_buffer = _load_string(
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
            active = cgutils.alloca_once(builder, ir.IntType(1))
            builder.store(cgutils.true_bit, result)
            builder.store(cgutils.true_bit, active)
            with cgutils.for_range(builder, left_size, intp=intp) as loop:
                left_char = builder.load(
                    builder.gep(left_buffer, [loop.index]))
                right_char = builder.load(
                    builder.gep(right_buffer, [loop.index]))
                same_char = builder.icmp_unsigned('==', left_char, right_char)
                still_active = builder.load(active)
                builder.store(
                    builder.select(still_active,
                                   builder.and_(builder.load(result),
                                                same_char),
                                   builder.load(result)),
                    result,
                )
                nonzero = builder.icmp_unsigned(
                    '!=', left_char, ir.Constant(int8, 0))
                builder.store(
                    builder.and_(still_active,
                                 builder.and_(same_char, nonzero)),
                    active,
                )

        return builder.load(result)

    return sig, codegen


def _install_typeof():
    if _STRING_DTYPE is None:
        return
    if getattr(typeof_impl, '_charex_stringdtype_installed', False):
        return

    old_ndarray_typeof = typeof_impl.registry[np.ndarray]
    typeof_impl._charex_stringdtype_old_ndarray_typeof = old_ndarray_typeof

    @typeof_impl.register(np.ndarray)
    def _stringdtype_ndarray_typeof(value, context):
        if is_stringdtype(value.dtype):
            if has_stringdtype_na_object(value.dtype):
                raise NumbaValueError(
                    'charex StringDType support currently requires default '
                    'StringDType without na_object',
                )
            layout = numpy_support.map_layout(value)
            readonly = not value.flags.writeable
            return types.Array(
                stringdtype_packet,
                value.ndim,
                layout,
                readonly=readonly,
            )
        return old_ndarray_typeof(value, context)

    typeof_impl._charex_stringdtype_installed = True
    typeof_impl._clear_cache()


_install_typeof()
