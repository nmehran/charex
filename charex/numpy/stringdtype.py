"""Experimental NumPy StringDType typing support."""

import ctypes
import importlib

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
_PACKED_STRING_SIZE = 16


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


def _native_stringdtype_helper():
    if _STRING_DTYPE is None:
        return None, 0, 0, 0
    try:
        native = importlib.import_module('charex._stringdtype')
        if not native.has_stringdtype_api():
            return None, 0, 0, 0
        library = ctypes.CDLL(native.__file__)
        acquire = library.charex_stringdtype_acquire_allocator
        acquire.argtypes = [ctypes.py_object]
        acquire.restype = ctypes.c_void_p
        acquire_two = library.charex_stringdtype_acquire_two_allocators
        acquire_two.argtypes = [
            ctypes.py_object, ctypes.py_object,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        acquire_two.restype = None
        release_two = library.charex_stringdtype_release_two_allocators
        release_two.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        release_two.restype = None
        return (
            library,
            ctypes.cast(acquire, ctypes.c_void_p).value,
            ctypes.cast(acquire_two, ctypes.c_void_p).value,
            ctypes.cast(release_two, ctypes.c_void_p).value,
        )
    except (AttributeError, ImportError, OSError):
        return None, 0, 0, 0


# Keep the CDLL object alive for the function address embedded in generated IR.
_NATIVE_LIBRARY, _NATIVE_ACQUIRE_ADDR, _NATIVE_ACQUIRE_TWO_ADDR, \
    _NATIVE_RELEASE_TWO_ADDR = _native_stringdtype_helper()


if _STRING_DTYPE is not None:
    _API_SLOTS = _numpy_api_slots()
    llvm.add_symbol('charex_NpyString_load', _API_SLOTS[313])
    llvm.add_symbol('charex_NpyString_release_allocator', _API_SLOTS[318])
else:
    _API_SLOTS = None


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


def _trimmed_size(builder, size, buffer, intp, int8):
    int64 = ir.IntType(64)
    int64_ptr = int64.as_pointer()
    word_size = ir.Constant(intp, 8)
    effective_size = cgutils.alloca_once(builder, intp)
    builder.store(size, effective_size)

    word_cond = builder.append_basic_block('stringdtype.trim.word_cond')
    word_check = builder.append_basic_block('stringdtype.trim.word_check')
    word_body = builder.append_basic_block('stringdtype.trim.word_body')
    byte_cond = builder.append_basic_block('stringdtype.trim.byte_cond')
    byte_check = builder.append_basic_block('stringdtype.trim.byte_check')
    byte_body = builder.append_basic_block('stringdtype.trim.byte_body')
    after = builder.append_basic_block('stringdtype.trim.after')

    builder.branch(word_cond)

    builder.position_at_end(word_cond)
    current_size = builder.load(effective_size)
    has_word = builder.icmp_unsigned('>=', current_size, word_size)
    builder.cbranch(has_word, word_check, byte_cond)

    builder.position_at_end(word_check)
    base = builder.sub(builder.load(effective_size), word_size)
    word_ptr = builder.gep(buffer, [base])
    word = builder.load(builder.bitcast(word_ptr, int64_ptr))
    word.align = 1
    is_zero_word = builder.icmp_unsigned('==', word,
                                         ir.Constant(int64, 0))
    builder.cbranch(is_zero_word, word_body, byte_cond)

    builder.position_at_end(word_body)
    builder.store(base, effective_size)
    builder.branch(word_cond)

    builder.position_at_end(byte_cond)
    has_remaining = builder.icmp_unsigned(
        '>', builder.load(effective_size), ir.Constant(intp, 0),
    )
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


def _codepoint_count(builder, size, buffer, intp, int8):
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


def _string_codepoint_len(builder, size, buffer, intp, int8):
    effective_size = _trimmed_size(builder, size, buffer, intp, int8)
    return _codepoint_count(builder, effective_size, buffer, intp, int8)


def _packed_string_ptr_from_data(builder, data, index_value, intp):
    int8 = ir.IntType(8)
    byte_ptr = int8.as_pointer()
    offset = builder.mul(index_value,
                         ir.Constant(intp, _PACKED_STRING_SIZE))
    return builder.gep(builder.bitcast(data, byte_ptr), [offset])


@intrinsic
def stringdtype_acquire_allocator(typingctx, array):
    if not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        byte_ptr = ir.IntType(8).as_pointer()
        array_struct = context.make_array(signature.args[0])(
            context, builder, args[0],
        )
        acquire_type = ir.FunctionType(byte_ptr, [byte_ptr])
        acquire_addr = context.get_constant(types.uintp, _NATIVE_ACQUIRE_ADDR)
        acquire = builder.inttoptr(
            acquire_addr, acquire_type.as_pointer(),
        )
        return builder.call(
            acquire, [builder.bitcast(array_struct.parent, byte_ptr)],
        )

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
def stringdtype_acquire_allocators(typingctx, left, right):
    if not is_stringdtype_array_type(left) \
            or not is_stringdtype_array_type(right):
        return None

    allocators_type = types.UniTuple(types.voidptr, 2)
    sig = signature(allocators_type, left, right)

    def codegen(context, builder, signature, args):
        intp = context.get_value_type(types.intp)
        byte_ptr = ir.IntType(8).as_pointer()
        left_struct = context.make_array(signature.args[0])(
            context, builder, args[0],
        )
        right_struct = context.make_array(signature.args[1])(
            context, builder, args[1],
        )
        allocators = cgutils.alloca_once(builder, byte_ptr, size=2)
        acquire_type = ir.FunctionType(
            ir.VoidType(), [byte_ptr, byte_ptr, byte_ptr.as_pointer()],
        )
        acquire_addr = context.get_constant(types.uintp,
                                            _NATIVE_ACQUIRE_TWO_ADDR)
        acquire = builder.inttoptr(
            acquire_addr, acquire_type.as_pointer(),
        )
        builder.call(
            acquire,
            [
                builder.bitcast(left_struct.parent, byte_ptr),
                builder.bitcast(right_struct.parent, byte_ptr),
                allocators,
            ],
        )
        return context.make_tuple(
            builder,
            signature.return_type,
            [
                builder.load(builder.gep(allocators,
                                         [ir.Constant(intp, 0)])),
                builder.load(builder.gep(allocators,
                                         [ir.Constant(intp, 1)])),
            ],
        )

    return sig, codegen


@intrinsic
def stringdtype_release_allocators(typingctx, allocators):
    allocators_type = types.UniTuple(types.voidptr, 2)
    if allocators != allocators_type:
        return None

    sig = signature(types.void, allocators)

    def codegen(context, builder, signature, args):
        intp = context.get_value_type(types.intp)
        byte_ptr = ir.IntType(8).as_pointer()
        allocator_values = cgutils.unpack_tuple(builder, args[0], 2)
        allocator_array = cgutils.alloca_once(builder, byte_ptr, size=2)
        builder.store(
            allocator_values[0],
            builder.gep(allocator_array, [ir.Constant(intp, 0)]),
        )
        builder.store(
            allocator_values[1],
            builder.gep(allocator_array, [ir.Constant(intp, 1)]),
        )
        release_type = ir.FunctionType(
            ir.VoidType(), [byte_ptr.as_pointer()],
        )
        release_addr = context.get_constant(types.uintp,
                                            _NATIVE_RELEASE_TWO_ADDR)
        release = builder.inttoptr(
            release_addr, release_type.as_pointer(),
        )
        builder.call(release, [allocator_array])

    return sig, codegen


@intrinsic
def stringdtype_data_ptr(typingctx, array):
    if not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        array_struct = context.make_array(signature.args[0])(
            context, builder, args[0],
        )
        return builder.bitcast(array_struct.data,
                               ir.IntType(8).as_pointer())

    return sig, codegen


@intrinsic
def stringdtype_codepoint_len_data(typingctx, data, index, allocator):
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
        packed = _packed_string_ptr_from_data(builder, data, index_value, intp)
        status, size, buffer = _load_string(builder, allocator, packed, intp,
                                            byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        builder.store(ir.Constant(intp, -1), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))

        with builder.if_then(valid):
            builder.store(
                _string_codepoint_len(builder, size, buffer, intp, int8),
                result,
            )

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_equal_data(typingctx, left_data, left_index, left_allocator,
                           right_data, right_index, right_allocator):
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
        left_packed = _packed_string_ptr_from_data(
            builder, left_data, left_index_value, intp)
        right_packed = _packed_string_ptr_from_data(
            builder, right_data, right_index_value, intp)
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
            memchr_type = ir.FunctionType(byte_ptr, [byte_ptr, int32, intp])
            memcmp_type = ir.FunctionType(int32, [byte_ptr, byte_ptr, intp])
            memchr = cgutils.get_or_insert_function(
                builder.module, memchr_type, 'memchr',
            )
            memcmp = cgutils.get_or_insert_function(
                builder.module, memcmp_type, 'memcmp',
            )
            full_cmp = builder.call(
                memcmp, [left_buffer, right_buffer, left_size],
            )
            full_equal = builder.icmp_signed(
                '==', full_cmp, ir.Constant(int32, 0))
            builder.store(full_equal, result)
            with builder.if_then(builder.not_(full_equal)):
                nul_ptr = builder.call(
                    memchr, [left_buffer, ir.Constant(int32, 0), left_size],
                )
                found_nul = builder.icmp_unsigned(
                    '!=', nul_ptr, ir.Constant(byte_ptr, None),
                )
                with builder.if_then(found_nul):
                    left_addr = builder.ptrtoint(left_buffer, intp)
                    nul_addr = builder.ptrtoint(nul_ptr, intp)
                    compare_size = builder.add(
                        builder.sub(nul_addr, left_addr),
                        ir.Constant(intp, 1),
                    )
                    nul_cmp = builder.call(
                        memcmp, [left_buffer, right_buffer, compare_size],
                    )
                    builder.store(
                        builder.icmp_signed('==', nul_cmp,
                                            ir.Constant(int32, 0)),
                        result,
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
            if not _NATIVE_ACQUIRE_ADDR \
                    or not _NATIVE_ACQUIRE_TWO_ADDR \
                    or not _NATIVE_RELEASE_TWO_ADDR:
                raise NumbaValueError(
                    'StringDType support requires the compiled '
                    'charex._stringdtype helper; reinstall charex or run '
                    'build_ext --inplace',
                )
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
