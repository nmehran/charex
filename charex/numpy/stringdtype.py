"""Experimental NumPy StringDType typing support."""

import ctypes
import importlib

from llvmlite import binding as llvm
from llvmlite import ir
from charex.core import JIT_OPTIONS
from numba.core import cgutils, types
from numba.core.datamodel import models, register_default
from numba.core.errors import NumbaValueError
from numba.core.typing import signature
from numba.core.typing.typeof import typeof_impl
from numba.cpython.unicode_support import (
    _PyUnicode_IsAlpha, _PyUnicode_IsDecimalDigit, _PyUnicode_IsDigit,
    _PyUnicode_IsLowercase, _PyUnicode_IsNumeric, _PyUnicode_IsSpace,
    _PyUnicode_IsTitlecase, _PyUnicode_IsUppercase,
)
from numba.extending import intrinsic, register_jitable
from numba.np import numpy_support
import numpy as np


_STRING_DTYPE = getattr(getattr(np, 'dtypes', None), 'StringDType', None)
_PACKED_STRING_SIZE = 16


class StringDTypePacket(types.Type):
    """Numba type for StringDType's 16-byte packed element record."""

    def __init__(self):
        super().__init__('StringDTypePacket')


stringdtype_packet = StringDTypePacket()
_UNICODE_PARTS_TYPE = types.UniTuple(types.intp, 2)
_UTF8_SPAN_TYPE = types.Tuple((types.voidptr, types.intp, types.boolean))
_UTF8_SLICE_TYPE = types.Tuple((types.intp, types.intp, types.boolean))


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


@register_jitable(**JIT_OPTIONS)
def _stringdtype_isalpha_ord(chr_ord):
    if chr_ord < 128:
        return 65 <= chr_ord <= 90 or 97 <= chr_ord <= 122
    return bool(_PyUnicode_IsAlpha(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _stringdtype_isalnum_ord(chr_ord):
    if chr_ord < 128:
        return 65 <= chr_ord <= 90 or 97 <= chr_ord <= 122 \
            or 48 <= chr_ord <= 57
    return bool(_PyUnicode_IsAlpha(chr_ord)) \
        or bool(_PyUnicode_IsNumeric(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _stringdtype_isdecimal_ord(chr_ord):
    if chr_ord < 128:
        return 48 <= chr_ord <= 57
    return bool(_PyUnicode_IsDecimalDigit(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _stringdtype_isdigit_ord(chr_ord):
    if chr_ord < 128:
        return 48 <= chr_ord <= 57
    return bool(_PyUnicode_IsDigit(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _stringdtype_isnumeric_ord(chr_ord):
    if chr_ord < 128:
        return 48 <= chr_ord <= 57
    return bool(_PyUnicode_IsNumeric(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _stringdtype_isspace_ord(chr_ord):
    if chr_ord < 128:
        return 9 <= chr_ord <= 13 or 28 <= chr_ord <= 32
    return bool(_PyUnicode_IsSpace(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _stringdtype_islower_ord(chr_ord):
    if chr_ord < 128:
        return 97 <= chr_ord <= 122
    return bool(_PyUnicode_IsLowercase(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _stringdtype_isupper_ord(chr_ord):
    if chr_ord < 128:
        return 65 <= chr_ord <= 90
    return bool(_PyUnicode_IsUppercase(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _stringdtype_istitle_ord(chr_ord):
    if chr_ord < 128:
        return False
    return bool(_PyUnicode_IsTitlecase(chr_ord))


_SIMPLE_PROPERTY_HELPERS = {
    'isalpha': _stringdtype_isalpha_ord,
    'isalnum': _stringdtype_isalnum_ord,
    'isdecimal': _stringdtype_isdecimal_ord,
    'isdigit': _stringdtype_isdigit_ord,
    'isnumeric': _stringdtype_isnumeric_ord,
    'isspace': _stringdtype_isspace_ord,
}


def _call_property(context, builder, helper, codepoint):
    sig = signature(types.boolean, types.int32)
    return context.compile_internal(builder, helper, sig, [codepoint])


def _load_utf8_byte(builder, buffer, offset, delta, intp, int32):
    byte_offset = builder.add(offset, ir.Constant(intp, delta))
    return builder.zext(builder.load(builder.gep(buffer, [byte_offset])),
                        int32)


def _decode_utf8_codepoint(builder, buffer, offset, intp, int8, int32):
    codepoint = cgutils.alloca_once(builder, int32)
    next_offset = cgutils.alloca_once(builder, intp)

    first = builder.zext(builder.load(builder.gep(buffer, [offset])), int32)
    one_byte = builder.icmp_unsigned('<', first, ir.Constant(int32, 0x80))
    with builder.if_else(one_byte) as (single, multi):
        with single:
            builder.store(first, codepoint)
            builder.store(builder.add(offset, ir.Constant(intp, 1)),
                          next_offset)
        with multi:
            two_byte = builder.icmp_unsigned(
                '<', first, ir.Constant(int32, 0xe0))
            with builder.if_else(two_byte) as (two, longer):
                with two:
                    second = _load_utf8_byte(
                        builder, buffer, offset, 1, intp, int32)
                    value = builder.or_(
                        builder.shl(builder.and_(
                            first, ir.Constant(int32, 0x1f)),
                            ir.Constant(int32, 6)),
                        builder.and_(second, ir.Constant(int32, 0x3f)),
                    )
                    builder.store(value, codepoint)
                    builder.store(builder.add(offset, ir.Constant(intp, 2)),
                                  next_offset)
                with longer:
                    three_byte = builder.icmp_unsigned(
                        '<', first, ir.Constant(int32, 0xf0))
                    with builder.if_else(three_byte) as (three, four):
                        with three:
                            second = _load_utf8_byte(
                                builder, buffer, offset, 1, intp, int32)
                            third = _load_utf8_byte(
                                builder, buffer, offset, 2, intp, int32)
                            value = builder.or_(
                                builder.or_(
                                    builder.shl(builder.and_(
                                        first, ir.Constant(int32, 0x0f)),
                                        ir.Constant(int32, 12)),
                                    builder.shl(builder.and_(
                                        second, ir.Constant(int32, 0x3f)),
                                        ir.Constant(int32, 6)),
                                ),
                                builder.and_(third,
                                             ir.Constant(int32, 0x3f)),
                            )
                            builder.store(value, codepoint)
                            builder.store(
                                builder.add(offset, ir.Constant(intp, 3)),
                                next_offset,
                            )
                        with four:
                            second = _load_utf8_byte(
                                builder, buffer, offset, 1, intp, int32)
                            third = _load_utf8_byte(
                                builder, buffer, offset, 2, intp, int32)
                            fourth = _load_utf8_byte(
                                builder, buffer, offset, 3, intp, int32)
                            value = builder.or_(
                                builder.or_(
                                    builder.shl(builder.and_(
                                        first, ir.Constant(int32, 0x07)),
                                        ir.Constant(int32, 18)),
                                    builder.shl(builder.and_(
                                        second, ir.Constant(int32, 0x3f)),
                                        ir.Constant(int32, 12)),
                                ),
                                builder.or_(
                                    builder.shl(builder.and_(
                                        third, ir.Constant(int32, 0x3f)),
                                        ir.Constant(int32, 6)),
                                    builder.and_(
                                        fourth, ir.Constant(int32, 0x3f)),
                                ),
                            )
                            builder.store(value, codepoint)
                            builder.store(
                                builder.add(offset, ir.Constant(intp, 4)),
                                next_offset,
                            )

    return builder.load(codepoint), builder.load(next_offset)


def _emit_stringdtype_predicate(context, builder, mode, size, buffer,
                                intp, int8, int32):
    pos = cgutils.alloca_once(builder, intp)
    valid = cgutils.alloca_once(builder, ir.IntType(1))
    seen = cgutils.alloca_once(builder, ir.IntType(1))
    cased_state = None
    builder.store(ir.Constant(intp, 0), pos)
    builder.store(cgutils.true_bit, valid)
    builder.store(cgutils.false_bit, seen)
    if mode == 'istitle':
        cased_state = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, cased_state)

    cond = builder.append_basic_block(f'stringdtype.{mode}.cond')
    body = builder.append_basic_block(f'stringdtype.{mode}.body')
    after = builder.append_basic_block(f'stringdtype.{mode}.after')
    builder.branch(cond)

    builder.position_at_end(cond)
    within_size = builder.icmp_unsigned('<', builder.load(pos), size)
    keep_going = builder.and_(within_size, builder.load(valid))
    builder.cbranch(keep_going, body, after)

    builder.position_at_end(body)
    codepoint, next_pos = _decode_utf8_codepoint(
        builder, buffer, builder.load(pos), intp, int8, int32)

    if mode in {'isalpha', 'isalnum', 'isdecimal', 'isdigit',
                'isnumeric', 'isspace'}:
        property_result = _call_property(
            context, builder, _SIMPLE_PROPERTY_HELPERS[mode], codepoint)
        with builder.if_else(property_result) as (passed, failed):
            with passed:
                builder.store(cgutils.true_bit, seen)
            with failed:
                builder.store(cgutils.false_bit, valid)
    else:
        is_lower = _call_property(
            context, builder, _stringdtype_islower_ord, codepoint)
        is_upper = _call_property(
            context, builder, _stringdtype_isupper_ord, codepoint)
        is_title = _call_property(
            context, builder, _stringdtype_istitle_ord, codepoint)
        is_start = builder.or_(is_upper, is_title)

        if mode == 'islower':
            invalid = builder.or_(is_upper, is_title)
            with builder.if_then(invalid):
                builder.store(cgutils.false_bit, valid)
            with builder.if_then(is_lower):
                builder.store(cgutils.true_bit, seen)
        elif mode == 'isupper':
            invalid = builder.or_(is_lower, is_title)
            with builder.if_then(invalid):
                builder.store(cgutils.false_bit, valid)
            with builder.if_then(is_upper):
                builder.store(cgutils.true_bit, seen)
        else:
            with builder.if_else(builder.load(cased_state)) as (
                    in_cased, in_uncased):
                with in_cased:
                    with builder.if_then(is_start):
                        builder.store(cgutils.false_bit, valid)
                    builder.store(is_lower, cased_state)
                with in_uncased:
                    with builder.if_then(is_lower):
                        builder.store(cgutils.false_bit, valid)
                    builder.store(is_start, cased_state)
                    with builder.if_then(is_start):
                        builder.store(cgutils.true_bit, seen)

    builder.store(next_pos, pos)
    builder.branch(cond)

    builder.position_at_end(after)
    return builder.and_(builder.load(valid), builder.load(seen))


def _stringdtype_size_compare(builder, left_size, right_size, int32):
    return builder.select(
        builder.icmp_unsigned('>', left_size, right_size),
        ir.Constant(int32, 1),
        builder.select(
            builder.icmp_unsigned('<', left_size, right_size),
            ir.Constant(int32, -1),
            ir.Constant(int32, 0),
        ),
    )


def _stringdtype_byte_compare(builder, left_size, left_buffer,
                              right_size, right_buffer, intp, int8, int32):
    compare_size = builder.select(
        builder.icmp_unsigned('<', left_size, right_size),
        left_size,
        right_size,
    )
    strncmp_type = ir.FunctionType(
        int32, [int8.as_pointer(), int8.as_pointer(), intp])
    strncmp = cgutils.get_or_insert_function(
        builder.module, strncmp_type, 'strncmp')
    text_cmp = builder.call(strncmp, [left_buffer, right_buffer,
                                      compare_size])
    return builder.select(
        builder.icmp_signed('==', text_cmp, ir.Constant(int32, 0)),
        _stringdtype_size_compare(builder, left_size, right_size, int32),
        builder.select(
            builder.icmp_signed('>', text_cmp, ir.Constant(int32, 0)),
            ir.Constant(int32, 1),
            ir.Constant(int32, -1),
        ),
    )


def _memcmp_equal(builder, left, right, size, int8, int32):
    memcmp_type = ir.FunctionType(
        int32, [int8.as_pointer(), int8.as_pointer(), size.type])
    memcmp = cgutils.get_or_insert_function(
        builder.module, memcmp_type, 'memcmp')
    return builder.icmp_signed(
        '==', builder.call(memcmp, [left, right, size]),
        ir.Constant(int32, 0),
    )


def _store_byte_affix_result(builder, result, value_buffer, start_offset,
                             end_offset, slice_valid, pattern_buffer,
                             pattern_size, suffix, intp, int8, int32):
    slice_size = builder.sub(end_offset, start_offset)
    empty_pattern = builder.icmp_unsigned(
        '==', pattern_size, ir.Constant(intp, 0))
    builder.store(builder.and_(slice_valid, empty_pattern), result)

    nonempty_pattern = builder.not_(empty_pattern)
    fits = builder.icmp_unsigned('<=', pattern_size, slice_size)
    with builder.if_then(builder.and_(slice_valid,
                                      builder.and_(nonempty_pattern, fits))):
        if suffix:
            compare_offset = builder.sub(end_offset, pattern_size)
        else:
            compare_offset = start_offset
        builder.store(
            _memcmp_equal(builder, builder.gep(value_buffer,
                                               [compare_offset]),
                          pattern_buffer, pattern_size, int8, int32),
            result,
        )


def _utf8_strncmp_equal(builder, string_size, string_buffer, scalar_data,
                        scalar_size, intp, int8, int32):
    result = cgutils.alloca_once(builder, ir.IntType(1))
    builder.store(cgutils.false_bit, result)
    same_size = builder.icmp_unsigned('==', string_size, scalar_size)

    with builder.if_then(same_size):
        strncmp_type = ir.FunctionType(
            int32, [int8.as_pointer(), int8.as_pointer(), intp])
        strncmp = cgutils.get_or_insert_function(
            builder.module, strncmp_type, 'strncmp')
        cmp_result = builder.call(strncmp, [string_buffer, scalar_data,
                                            string_size])
        builder.store(
            builder.icmp_signed('==', cmp_result, ir.Constant(int32, 0)),
            result,
        )

    return builder.load(result)


def _load_word64(builder, buffer, int64):
    word = builder.load(builder.bitcast(buffer, int64.as_pointer()))
    word.align = 1
    return word


def _byte_zero_highbits(builder, word, int64):
    low_bits = ir.Constant(int64, 0x0101010101010101)
    high_bits = ir.Constant(int64, 0x8080808080808080)
    return builder.and_(
        builder.and_(builder.sub(word, low_bits), builder.not_(word)),
        high_bits,
    )


def _byte_nonzero_highbits(builder, word, int64):
    high_bits = ir.Constant(int64, 0x8080808080808080)
    return builder.and_(builder.not_(_byte_zero_highbits(builder, word, int64)),
                        high_bits)


def _utf8_word8_equal(builder, string_size, string_buffer, scalar_data,
                      scalar_size, intp, int8, int32):
    int1 = ir.IntType(1)
    int64 = ir.IntType(64)
    result = cgutils.alloca_once(builder, int1)
    builder.store(cgutils.false_bit, result)
    same_size = builder.icmp_unsigned('==', string_size, scalar_size)

    with builder.if_then(same_size):
        enough = builder.icmp_unsigned('>=', string_size,
                                       ir.Constant(intp, 8))
        with builder.if_else(enough) as (word_path, fallback):
            with word_path:
                left = _load_word64(builder, string_buffer, int64)
                right = _load_word64(builder, scalar_data, int64)
                diff = _byte_nonzero_highbits(
                    builder, builder.xor(left, right), int64)
                nul = _byte_zero_highbits(builder, left, int64)
                has_decision = builder.or_(
                    builder.icmp_unsigned('!=', diff, ir.Constant(int64, 0)),
                    builder.icmp_unsigned('!=', nul, ir.Constant(int64, 0)),
                )
                with builder.if_else(has_decision) as (decide, unknown):
                    with decide:
                        first_diff = builder.cttz(diff,
                                                  ir.Constant(int1, 0))
                        first_nul = builder.cttz(nul, ir.Constant(int1, 0))
                        builder.store(
                            builder.icmp_unsigned('<', first_nul,
                                                  first_diff),
                            result,
                        )
                    with unknown:
                        exactly_word = builder.icmp_unsigned(
                            '==', string_size, ir.Constant(intp, 8))
                        with builder.if_else(exactly_word) as (exact, longer):
                            with exact:
                                builder.store(cgutils.true_bit, result)
                            with longer:
                                builder.store(
                                    _utf8_strncmp_equal(
                                        builder, string_size, string_buffer,
                                        scalar_data, scalar_size, intp, int8,
                                        int32),
                                    result,
                                )
            with fallback:
                builder.store(
                    _utf8_strncmp_equal(
                        builder, string_size, string_buffer, scalar_data,
                        scalar_size, intp, int8, int32),
                    result,
                )

    return builder.load(result)


def _utf8_prefilter_compare(builder, string_size, string_buffer, scalar_data,
                            scalar_size, intp, int8, int32):
    result = cgutils.alloca_once(builder, int32)
    builder.store(ir.Constant(int32, 0), result)
    compare_size = builder.select(
        builder.icmp_unsigned('<', string_size, scalar_size),
        string_size,
        scalar_size,
    )
    empty_prefix = builder.icmp_unsigned('==', compare_size,
                                         ir.Constant(intp, 0))

    with builder.if_else(empty_prefix) as (empty, nonempty):
        with empty:
            builder.store(
                _stringdtype_size_compare(builder, string_size, scalar_size,
                                          int32),
                result,
            )
        with nonempty:
            first_left = builder.load(builder.gep(string_buffer, [
                ir.Constant(intp, 0)]))
            first_right = builder.load(builder.gep(scalar_data, [
                ir.Constant(intp, 0)]))
            first_equal = builder.icmp_unsigned('==', first_left, first_right)
            with builder.if_else(first_equal) as (same, different):
                with same:
                    builder.store(
                        _stringdtype_byte_compare(
                            builder, string_size, string_buffer, scalar_size,
                            scalar_data, intp, int8, int32),
                        result,
                    )
                with different:
                    builder.store(
                        builder.select(
                            builder.icmp_unsigned('>', first_left,
                                                  first_right),
                            ir.Constant(int32, 1),
                            ir.Constant(int32, -1),
                        ),
                        result,
                    )

    return builder.load(result)


def _unicode_codepoint(builder, data, kind, index, int32):
    int8 = ir.IntType(8)
    int16 = ir.IntType(16)
    result = cgutils.alloca_once(builder, int32)

    one_byte = builder.icmp_signed('==', kind, ir.Constant(int32, 1))
    with builder.if_else(one_byte) as (one, wider):
        with one:
            ptr = builder.bitcast(data, int8.as_pointer())
            value = builder.load(builder.gep(ptr, [index]))
            builder.store(builder.zext(value, int32), result)
        with wider:
            two_byte = builder.icmp_signed('==', kind, ir.Constant(int32, 2))
            with builder.if_else(two_byte) as (two, four):
                with two:
                    ptr = builder.bitcast(data, int16.as_pointer())
                    value = builder.load(builder.gep(ptr, [index]))
                    builder.store(builder.zext(value, int32), result)
                with four:
                    ptr = builder.bitcast(data, int32.as_pointer())
                    builder.store(builder.load(builder.gep(ptr, [index])),
                                  result)

    return builder.load(result)


def _unicode_trimmed_length(builder, data, length, kind, intp, int32):
    trimmed = cgutils.alloca_once(builder, intp)
    builder.store(length, trimmed)

    cond = builder.append_basic_block('stringdtype.unicode.trim.cond')
    body = builder.append_basic_block('stringdtype.unicode.trim.body')
    decrement = builder.append_basic_block('stringdtype.unicode.trim.decrement')
    after = builder.append_basic_block('stringdtype.unicode.trim.after')
    builder.branch(cond)

    builder.position_at_end(cond)
    has_char = builder.icmp_unsigned('>', builder.load(trimmed),
                                     ir.Constant(intp, 0))
    builder.cbranch(has_char, body, after)

    builder.position_at_end(body)
    previous = builder.sub(builder.load(trimmed), ir.Constant(intp, 1))
    codepoint = _unicode_codepoint(builder, data, kind, previous, int32)
    is_zero = builder.icmp_unsigned('==', codepoint, ir.Constant(int32, 0))
    builder.cbranch(is_zero, decrement, after)

    builder.position_at_end(decrement)
    builder.store(previous, trimmed)
    builder.branch(cond)

    builder.position_at_end(after)
    return builder.load(trimmed)


def _unicode_utf8_size(builder, data, length, kind, intp, int32):
    size = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), size)

    with cgutils.for_range(builder, length, intp=intp) as loop:
        codepoint = _unicode_codepoint(builder, data, kind, loop.index, int32)
        one_byte = builder.icmp_unsigned('<', codepoint,
                                         ir.Constant(int32, 0x80))
        two_byte = builder.icmp_unsigned('<', codepoint,
                                         ir.Constant(int32, 0x800))
        three_byte = builder.icmp_unsigned('<', codepoint,
                                           ir.Constant(int32, 0x10000))
        width = builder.select(
            one_byte, ir.Constant(intp, 1),
            builder.select(
                two_byte, ir.Constant(intp, 2),
                builder.select(three_byte, ir.Constant(intp, 3),
                               ir.Constant(intp, 4)),
            ),
        )
        builder.store(builder.add(builder.load(size), width), size)

    return builder.load(size)


def _store_utf8_byte(builder, buffer, pos, value, int8):
    builder.store(builder.trunc(value, int8), builder.gep(buffer, [pos]))


def _store_utf8_codepoint(builder, buffer, pos, codepoint, intp, int8, int32):
    result = cgutils.alloca_once(builder, intp)
    builder.store(pos, result)

    one_byte = builder.icmp_unsigned('<', codepoint, ir.Constant(int32, 0x80))
    two_byte = builder.icmp_unsigned('<', codepoint, ir.Constant(int32, 0x800))
    three_byte = builder.icmp_unsigned('<', codepoint,
                                       ir.Constant(int32, 0x10000))

    with builder.if_else(one_byte) as (is_one, not_one):
        with is_one:
            _store_utf8_byte(builder, buffer, pos, codepoint, int8)
            builder.store(builder.add(pos, ir.Constant(intp, 1)), result)
        with not_one:
            with builder.if_else(two_byte) as (is_two, not_two):
                with is_two:
                    first = builder.or_(
                        builder.lshr(codepoint, ir.Constant(int32, 6)),
                        ir.Constant(int32, 0xc0),
                    )
                    second = builder.or_(
                        builder.and_(codepoint, ir.Constant(int32, 0x3f)),
                        ir.Constant(int32, 0x80),
                    )
                    _store_utf8_byte(builder, buffer, pos, first, int8)
                    _store_utf8_byte(
                        builder, buffer, builder.add(pos, ir.Constant(intp, 1)),
                        second, int8,
                    )
                    builder.store(builder.add(pos, ir.Constant(intp, 2)),
                                  result)
                with not_two:
                    with builder.if_else(three_byte) as (is_three, is_four):
                        with is_three:
                            first = builder.or_(
                                builder.lshr(codepoint,
                                             ir.Constant(int32, 12)),
                                ir.Constant(int32, 0xe0),
                            )
                            second = builder.or_(
                                builder.and_(
                                    builder.lshr(codepoint,
                                                 ir.Constant(int32, 6)),
                                    ir.Constant(int32, 0x3f),
                                ),
                                ir.Constant(int32, 0x80),
                            )
                            third = builder.or_(
                                builder.and_(codepoint,
                                             ir.Constant(int32, 0x3f)),
                                ir.Constant(int32, 0x80),
                            )
                            _store_utf8_byte(builder, buffer, pos, first, int8)
                            _store_utf8_byte(
                                builder, buffer,
                                builder.add(pos, ir.Constant(intp, 1)),
                                second, int8,
                            )
                            _store_utf8_byte(
                                builder, buffer,
                                builder.add(pos, ir.Constant(intp, 2)),
                                third, int8,
                            )
                            builder.store(
                                builder.add(pos, ir.Constant(intp, 3)),
                                result,
                            )
                        with is_four:
                            first = builder.or_(
                                builder.lshr(codepoint,
                                             ir.Constant(int32, 18)),
                                ir.Constant(int32, 0xf0),
                            )
                            second = builder.or_(
                                builder.and_(
                                    builder.lshr(codepoint,
                                                 ir.Constant(int32, 12)),
                                    ir.Constant(int32, 0x3f),
                                ),
                                ir.Constant(int32, 0x80),
                            )
                            third = builder.or_(
                                builder.and_(
                                    builder.lshr(codepoint,
                                                 ir.Constant(int32, 6)),
                                    ir.Constant(int32, 0x3f),
                                ),
                                ir.Constant(int32, 0x80),
                            )
                            fourth = builder.or_(
                                builder.and_(codepoint,
                                             ir.Constant(int32, 0x3f)),
                                ir.Constant(int32, 0x80),
                            )
                            _store_utf8_byte(builder, buffer, pos, first, int8)
                            _store_utf8_byte(
                                builder, buffer,
                                builder.add(pos, ir.Constant(intp, 1)),
                                second, int8,
                            )
                            _store_utf8_byte(
                                builder, buffer,
                                builder.add(pos, ir.Constant(intp, 2)),
                                third, int8,
                            )
                            _store_utf8_byte(
                                builder, buffer,
                                builder.add(pos, ir.Constant(intp, 3)),
                                fourth, int8,
                            )
                            builder.store(
                                builder.add(pos, ir.Constant(intp, 4)),
                                result,
                            )

    return builder.load(result)


def _encode_utf8_buffer(builder, unicode_struct, value_length, buffer, intp,
                        int8, int32):
    pos = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), pos)
    with cgutils.for_range(builder, value_length, intp=intp) as loop:
        codepoint = _unicode_codepoint(
            builder, unicode_struct.data, unicode_struct.kind, loop.index,
            int32,
        )
        builder.store(
            _store_utf8_codepoint(builder, buffer, builder.load(pos),
                                  codepoint, intp, int8, int32),
            pos,
        )


def _unicode_parts(context, builder, unicode_value, intp, int32,
                   length=None, size=None):
    unicode_struct = context.make_helper(builder, types.unicode_type,
                                         value=unicode_value)
    if length is None:
        length = _unicode_trimmed_length(
            builder, unicode_struct.data, unicode_struct.length,
            unicode_struct.kind, intp, int32,
        )
    if size is None:
        size = _unicode_utf8_size(
            builder, unicode_struct.data, length, unicode_struct.kind, intp,
            int32,
        )
    return unicode_struct, length, size


def _unicode_is_valid(builder, data, length, kind, intp, int32):
    valid = cgutils.alloca_once(builder, ir.IntType(1))
    builder.store(cgutils.true_bit, valid)

    with cgutils.for_range(builder, length, intp=intp) as loop:
        codepoint = _unicode_codepoint(builder, data, kind, loop.index, int32)
        surrogate_start = builder.icmp_unsigned(
            '>=', codepoint, ir.Constant(int32, 0xd800))
        surrogate_end = builder.icmp_unsigned(
            '<=', codepoint, ir.Constant(int32, 0xdfff))
        with builder.if_then(builder.and_(surrogate_start, surrogate_end)):
            builder.store(cgutils.false_bit, valid)

    return builder.load(valid)


def _stringdtype_unicode_compare(builder, string_size, string_buffer,
                                 unicode_value, unicode_length, unicode_size,
                                 context, intp, int8, int32):
    unicode_struct, unicode_length, unicode_size = _unicode_parts(
        context, builder, unicode_value, intp, int32, unicode_length,
        unicode_size)

    result = cgutils.alloca_once(builder, int32)
    done = cgutils.alloca_once(builder, ir.IntType(1))
    byte_pos = cgutils.alloca_once(builder, intp)
    unicode_pos = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(int32, 0), result)
    builder.store(cgutils.false_bit, done)
    builder.store(ir.Constant(intp, 0), byte_pos)
    builder.store(ir.Constant(intp, 0), unicode_pos)

    cond = builder.append_basic_block('stringdtype.unicode.cmp.cond')
    body = builder.append_basic_block('stringdtype.unicode.cmp.body')
    after = builder.append_basic_block('stringdtype.unicode.cmp.after')
    builder.branch(cond)

    builder.position_at_end(cond)
    has_string = builder.icmp_unsigned('<', builder.load(byte_pos),
                                       string_size)
    has_unicode = builder.icmp_unsigned('<', builder.load(unicode_pos),
                                        unicode_length)
    is_equal = builder.icmp_signed('==', builder.load(result),
                                   ir.Constant(int32, 0))
    builder.cbranch(builder.and_(builder.and_(has_string, has_unicode),
                                 builder.and_(is_equal,
                                              builder.not_(builder.load(done)))),
                    body, after)

    builder.position_at_end(body)
    left_codepoint, next_byte_pos = _decode_utf8_codepoint(
        builder, string_buffer, builder.load(byte_pos), intp, int8, int32)
    right_codepoint = _unicode_codepoint(
        builder, unicode_struct.data, unicode_struct.kind,
        builder.load(unicode_pos), int32,
    )
    mismatch = builder.icmp_unsigned('!=', left_codepoint, right_codepoint)
    with builder.if_else(mismatch) as (different, same):
        with different:
            smaller = builder.icmp_unsigned('<', left_codepoint,
                                            right_codepoint)
            builder.store(
                builder.select(smaller, ir.Constant(int32, -1),
                               ir.Constant(int32, 1)),
                result,
            )
        with same:
            is_nul = builder.icmp_unsigned('==', left_codepoint,
                                           ir.Constant(int32, 0))
            with builder.if_else(is_nul) as (nul, non_nul):
                with nul:
                    builder.store(cgutils.true_bit, done)
                with non_nul:
                    builder.store(next_byte_pos, byte_pos)
                    builder.store(builder.add(builder.load(unicode_pos),
                                              ir.Constant(intp, 1)),
                                  unicode_pos)
    builder.branch(cond)

    builder.position_at_end(after)
    with builder.if_then(builder.icmp_signed('==', builder.load(result),
                                             ir.Constant(int32, 0))):
        sizes_equal = builder.icmp_unsigned('==', string_size, unicode_size)
        with builder.if_then(builder.not_(sizes_equal)):
            builder.store(
                _stringdtype_size_compare(builder, string_size, unicode_size,
                                          int32),
                result,
            )

    return builder.load(result)


def _stringdtype_unicode_equal(builder, string_size, string_buffer,
                               unicode_value, unicode_length, unicode_size,
                               context, intp, int8, int32):
    unicode_struct, unicode_length, unicode_size = _unicode_parts(
        context, builder, unicode_value, intp, int32, unicode_length,
        unicode_size)

    result = cgutils.alloca_once(builder, ir.IntType(1))
    done = cgutils.alloca_once(builder, ir.IntType(1))
    byte_pos = cgutils.alloca_once(builder, intp)
    unicode_pos = cgutils.alloca_once(builder, intp)
    builder.store(cgutils.false_bit, result)
    builder.store(cgutils.false_bit, done)
    builder.store(ir.Constant(intp, 0), byte_pos)
    builder.store(ir.Constant(intp, 0), unicode_pos)

    same_size = builder.icmp_unsigned('==', string_size, unicode_size)
    with builder.if_then(same_size):
        cond = builder.append_basic_block('stringdtype.unicode.eq.cond')
        body = builder.append_basic_block('stringdtype.unicode.eq.body')
        after = builder.append_basic_block('stringdtype.unicode.eq.after')
        builder.branch(cond)

        builder.position_at_end(cond)
        in_range = builder.icmp_unsigned('<', builder.load(byte_pos),
                                         string_size)
        builder.cbranch(builder.and_(in_range,
                                     builder.not_(builder.load(done))),
                        body, after)

        builder.position_at_end(body)
        left_codepoint, next_byte_pos = _decode_utf8_codepoint(
            builder, string_buffer, builder.load(byte_pos), intp, int8, int32)
        right_codepoint = _unicode_codepoint(
            builder, unicode_struct.data, unicode_struct.kind,
            builder.load(unicode_pos), int32,
        )
        mismatch = builder.icmp_unsigned('!=', left_codepoint, right_codepoint)
        with builder.if_else(mismatch) as (different, same):
            with different:
                builder.store(cgutils.true_bit, done)
            with same:
                is_nul = builder.icmp_unsigned('==', left_codepoint,
                                               ir.Constant(int32, 0))
                with builder.if_else(is_nul) as (nul, non_nul):
                    with nul:
                        builder.store(cgutils.true_bit, result)
                        builder.store(cgutils.true_bit, done)
                    with non_nul:
                        builder.store(next_byte_pos, byte_pos)
                        builder.store(builder.add(builder.load(unicode_pos),
                                                  ir.Constant(intp, 1)),
                                      unicode_pos)
        builder.branch(cond)

        builder.position_at_end(after)
        with builder.if_then(builder.not_(builder.load(done))):
            builder.store(cgutils.true_bit, result)

    return builder.load(result)


def _stringdtype_unicode_region_equal(builder, string_buffer, string_offset,
                                      unicode_struct, unicode_start,
                                      unicode_length, intp, int8, int32):
    result = cgutils.alloca_once(builder, ir.IntType(1))
    byte_pos = cgutils.alloca_once(builder, intp)
    unicode_pos = cgutils.alloca_once(builder, intp)
    remaining = cgutils.alloca_once(builder, intp)
    builder.store(cgutils.true_bit, result)
    builder.store(string_offset, byte_pos)
    builder.store(unicode_start, unicode_pos)
    builder.store(unicode_length, remaining)

    cond = builder.append_basic_block('stringdtype.unicode.region.cond')
    body = builder.append_basic_block('stringdtype.unicode.region.body')
    after = builder.append_basic_block('stringdtype.unicode.region.after')
    builder.branch(cond)

    builder.position_at_end(cond)
    keep_going = builder.and_(
        builder.icmp_unsigned('>', builder.load(remaining),
                              ir.Constant(intp, 0)),
        builder.load(result),
    )
    builder.cbranch(keep_going, body, after)

    builder.position_at_end(body)
    left_codepoint, next_byte_pos = _decode_utf8_codepoint(
        builder, string_buffer, builder.load(byte_pos), intp, int8, int32)
    right_codepoint = _unicode_codepoint(
        builder, unicode_struct.data, unicode_struct.kind,
        builder.load(unicode_pos), int32,
    )
    with builder.if_else(builder.icmp_unsigned('==', left_codepoint,
                                               right_codepoint)) as (same,
                                                                    different):
        with same:
            builder.store(next_byte_pos, byte_pos)
            builder.store(builder.add(builder.load(unicode_pos),
                                      ir.Constant(intp, 1)), unicode_pos)
            builder.store(builder.sub(builder.load(remaining),
                                      ir.Constant(intp, 1)), remaining)
        with different:
            builder.store(cgutils.false_bit, result)
    builder.branch(cond)

    builder.position_at_end(after)
    return builder.load(result)


def _unicode_stringdtype_region_equal(builder, unicode_struct, unicode_start,
                                      string_buffer, string_size,
                                      intp, int8, int32):
    result = cgutils.alloca_once(builder, ir.IntType(1))
    byte_pos = cgutils.alloca_once(builder, intp)
    unicode_pos = cgutils.alloca_once(builder, intp)
    builder.store(cgutils.true_bit, result)
    builder.store(ir.Constant(intp, 0), byte_pos)
    builder.store(unicode_start, unicode_pos)

    cond = builder.append_basic_block('stringdtype.unicode.pattern.cond')
    body = builder.append_basic_block('stringdtype.unicode.pattern.body')
    after = builder.append_basic_block('stringdtype.unicode.pattern.after')
    builder.branch(cond)

    builder.position_at_end(cond)
    keep_going = builder.and_(
        builder.icmp_unsigned('<', builder.load(byte_pos), string_size),
        builder.load(result),
    )
    builder.cbranch(keep_going, body, after)

    builder.position_at_end(body)
    left_codepoint = _unicode_codepoint(
        builder, unicode_struct.data, unicode_struct.kind,
        builder.load(unicode_pos), int32,
    )
    right_codepoint, next_byte_pos = _decode_utf8_codepoint(
        builder, string_buffer, builder.load(byte_pos), intp, int8, int32)
    with builder.if_else(builder.icmp_unsigned('==', left_codepoint,
                                               right_codepoint)) as (same,
                                                                    different):
        with same:
            builder.store(next_byte_pos, byte_pos)
            builder.store(builder.add(builder.load(unicode_pos),
                                      ir.Constant(intp, 1)), unicode_pos)
        with different:
            builder.store(cgutils.false_bit, result)
    builder.branch(cond)

    builder.position_at_end(after)
    return builder.load(result)


def _normalise_unicode_slice(builder, length, start, end, intp):
    zero = ir.Constant(intp, 0)

    start_from_end = builder.add(length, start)
    start_negative = builder.icmp_signed('<', start, zero)
    start_from_end = builder.select(
        builder.icmp_signed('<', start_from_end, zero),
        zero,
        start_from_end,
    )
    start_index = builder.select(start_negative, start_from_end, start)
    start_valid = builder.icmp_signed('<=', start_index, length)
    start_index = builder.select(
        builder.icmp_signed('>', start_index, length),
        length,
        start_index,
    )

    end_from_end = builder.add(length, end)
    end_negative = builder.icmp_signed('<', end, zero)
    end_from_end = builder.select(
        builder.icmp_signed('<', end_from_end, zero),
        zero,
        end_from_end,
    )
    end_index = builder.select(end_negative, end_from_end, end)
    end_index = builder.select(
        builder.icmp_signed('>', end_index, length),
        length,
        end_index,
    )

    ordered = builder.icmp_signed('<=', start_index, end_index)
    valid = builder.and_(start_valid, ordered)
    return start_index, end_index, valid


def _stringdtype_unicode_search(builder, value_size, value_buffer, pattern,
                                pattern_length, pattern_size, context, start,
                                end, mode, intp, int8, int32):
    zero = ir.Constant(intp, 0)
    one = ir.Constant(intp, 1)
    result = cgutils.alloca_once(builder, intp)
    if mode == 'count':
        builder.store(zero, result)
    else:
        builder.store(ir.Constant(intp, -1), result)

    unicode_struct, pattern_length, pattern_size = _unicode_parts(
        context, builder, pattern, intp, int32, pattern_length, pattern_size)
    start_index, end_index, start_offset, end_offset, slice_valid = \
        _normalise_slice(builder, value_size, value_buffer, start, end, intp,
                         int8)
    empty_pattern = builder.icmp_unsigned('==', pattern_size, zero)
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
    fits = builder.icmp_unsigned('<=', pattern_size, slice_size)
    with builder.if_then(builder.and_(slice_valid,
                                      builder.and_(nonempty_pattern, fits))):
        found = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, found)

        if mode == 'rfind':
            pos = cgutils.alloca_once(builder, intp)
            last = builder.sub(end_offset, pattern_size)
            builder.store(last, pos)
            cond = builder.append_basic_block('stringdtype.unicode.rfind.cond')
            body = builder.append_basic_block('stringdtype.unicode.rfind.body')
            decrement = builder.append_basic_block(
                'stringdtype.unicode.rfind.decrement')
            after = builder.append_basic_block('stringdtype.unicode.rfind.after')
            builder.branch(cond)

            builder.position_at_end(cond)
            in_range = builder.icmp_signed('>=', builder.load(pos),
                                           start_offset)
            builder.cbranch(builder.and_(in_range,
                                         builder.not_(builder.load(found))),
                            body, after)

            builder.position_at_end(body)
            matched = _stringdtype_unicode_region_equal(
                builder, value_buffer, builder.load(pos), unicode_struct,
                zero, pattern_length, intp, int8, int32)
            with builder.if_then(matched):
                builder.store(
                    _codepoint_count(builder, builder.load(pos), value_buffer,
                                     intp, int8),
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
            last = builder.sub(end_offset, pattern_size)
            builder.store(start_offset, pos)
            cond = builder.append_basic_block('stringdtype.unicode.find.cond')
            body = builder.append_basic_block('stringdtype.unicode.find.body')
            advance = builder.append_basic_block(
                'stringdtype.unicode.find.advance')
            after = builder.append_basic_block('stringdtype.unicode.find.after')
            builder.branch(cond)

            builder.position_at_end(cond)
            in_range = builder.icmp_signed('<=', builder.load(pos), last)
            continue_search = in_range
            if mode == 'find':
                continue_search = builder.and_(
                    in_range, builder.not_(builder.load(found)))
            builder.cbranch(continue_search, body, after)

            builder.position_at_end(body)
            matched = _stringdtype_unicode_region_equal(
                builder, value_buffer, builder.load(pos), unicode_struct,
                zero, pattern_length, intp, int8, int32)
            with builder.if_then(matched):
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
                next_match = builder.add(builder.load(pos), pattern_size)
                next_scan = builder.add(builder.load(pos), one)
                builder.store(
                    builder.select(matched, next_match, next_scan),
                    pos,
                )
            else:
                builder.store(builder.add(builder.load(pos), one), pos)
            builder.branch(cond)

            builder.position_at_end(after)

    return builder.load(result)


def _unicode_stringdtype_search(builder, value, value_length, value_size,
                                pattern_size, pattern_buffer, context, start,
                                end, mode, intp, int8, int32):
    zero = ir.Constant(intp, 0)
    one = ir.Constant(intp, 1)
    result = cgutils.alloca_once(builder, intp)
    if mode == 'count':
        builder.store(zero, result)
    else:
        builder.store(ir.Constant(intp, -1), result)

    unicode_struct, value_length, _ = _unicode_parts(
        context, builder, value, intp, int32, value_length, value_size)
    start_index, end_index, slice_valid = _normalise_unicode_slice(
        builder, value_length, start, end, intp)
    pattern_effective_size = _trimmed_size(
        builder, pattern_size, pattern_buffer, intp, int8)
    empty_pattern = builder.icmp_unsigned('==', pattern_effective_size, zero)
    if mode == 'count':
        pattern_match_size = builder.select(empty_pattern, zero, pattern_size)
    else:
        short_pattern = builder.icmp_unsigned('<=', pattern_effective_size,
                                              one)
        pattern_match_size = builder.select(short_pattern,
                                            pattern_effective_size,
                                            pattern_size)
    pattern_match_length = _codepoint_count(
        builder, pattern_match_size, pattern_buffer, intp, int8)

    with builder.if_then(builder.and_(slice_valid, empty_pattern)):
        if mode == 'find':
            builder.store(start_index, result)
        elif mode == 'rfind':
            builder.store(end_index, result)
        else:
            builder.store(builder.add(builder.sub(end_index, start_index),
                                      one), result)

    slice_length = builder.sub(end_index, start_index)
    nonempty_pattern = builder.not_(empty_pattern)
    fits = builder.icmp_unsigned('<=', pattern_match_length, slice_length)
    with builder.if_then(builder.and_(slice_valid,
                                      builder.and_(nonempty_pattern, fits))):
        found = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, found)

        if mode == 'rfind':
            pos = cgutils.alloca_once(builder, intp)
            last = builder.sub(end_index, pattern_match_length)
            builder.store(last, pos)
            cond = builder.append_basic_block(
                'stringdtype.unicode.value.rfind.cond')
            body = builder.append_basic_block(
                'stringdtype.unicode.value.rfind.body')
            decrement = builder.append_basic_block(
                'stringdtype.unicode.value.rfind.decrement')
            after = builder.append_basic_block(
                'stringdtype.unicode.value.rfind.after')
            builder.branch(cond)

            builder.position_at_end(cond)
            in_range = builder.icmp_signed('>=', builder.load(pos),
                                           start_index)
            builder.cbranch(builder.and_(in_range,
                                         builder.not_(builder.load(found))),
                            body, after)

            builder.position_at_end(body)
            matched = _unicode_stringdtype_region_equal(
                builder, unicode_struct, builder.load(pos), pattern_buffer,
                pattern_match_size, intp, int8, int32)
            with builder.if_then(matched):
                builder.store(builder.load(pos), result)
                builder.store(cgutils.true_bit, found)
            builder.branch(decrement)

            builder.position_at_end(decrement)
            builder.store(builder.sub(builder.load(pos), one), pos)
            builder.branch(cond)

            builder.position_at_end(after)
        else:
            pos = cgutils.alloca_once(builder, intp)
            last = builder.sub(end_index, pattern_match_length)
            builder.store(start_index, pos)
            cond = builder.append_basic_block(
                'stringdtype.unicode.value.find.cond')
            body = builder.append_basic_block(
                'stringdtype.unicode.value.find.body')
            advance = builder.append_basic_block(
                'stringdtype.unicode.value.find.advance')
            after = builder.append_basic_block(
                'stringdtype.unicode.value.find.after')
            builder.branch(cond)

            builder.position_at_end(cond)
            in_range = builder.icmp_signed('<=', builder.load(pos), last)
            continue_search = in_range
            if mode == 'find':
                continue_search = builder.and_(
                    in_range, builder.not_(builder.load(found)))
            builder.cbranch(continue_search, body, after)

            builder.position_at_end(body)
            matched = _unicode_stringdtype_region_equal(
                builder, unicode_struct, builder.load(pos), pattern_buffer,
                pattern_match_size, intp, int8, int32)
            with builder.if_then(matched):
                if mode == 'find':
                    builder.store(builder.load(pos), result)
                    builder.store(cgutils.true_bit, found)
                else:
                    builder.store(builder.add(builder.load(result), one),
                                  result)
            builder.branch(advance)

            builder.position_at_end(advance)
            if mode == 'count':
                next_match = builder.add(builder.load(pos),
                                         pattern_match_length)
                next_scan = builder.add(builder.load(pos), one)
                builder.store(
                    builder.select(matched, next_match, next_scan),
                    pos,
                )
            else:
                builder.store(builder.add(builder.load(pos), one), pos)
            builder.branch(cond)

            builder.position_at_end(after)

    return builder.load(result)


def _codepoint_offset(builder, size, buffer, target, intp, int8):
    offset = cgutils.alloca_once(builder, intp)
    count = cgutils.alloca_once(builder, intp)
    builder.store(ir.Constant(intp, 0), offset)
    builder.store(ir.Constant(intp, 0), count)

    cond = builder.append_basic_block('stringdtype.offset.cond')
    check = builder.append_basic_block('stringdtype.offset.check')
    body = builder.append_basic_block('stringdtype.offset.body')
    after = builder.append_basic_block('stringdtype.offset.after')

    builder.branch(cond)

    builder.position_at_end(cond)
    within_size = builder.icmp_unsigned('<', builder.load(offset), size)
    builder.cbranch(within_size, check, after)

    builder.position_at_end(check)
    char = builder.load(builder.gep(buffer, [builder.load(offset)]))
    tag = builder.and_(char, ir.Constant(int8, 0xc0))
    continuation = builder.icmp_unsigned(
        '==', tag, ir.Constant(int8, 0x80),
    )
    at_target = builder.icmp_signed('==', builder.load(count), target)
    builder.cbranch(builder.and_(at_target, builder.not_(continuation)),
                    after, body)

    builder.position_at_end(body)
    increment = builder.select(
        continuation, ir.Constant(intp, 0), ir.Constant(intp, 1),
    )
    builder.store(builder.add(builder.load(count), increment), count)
    builder.store(builder.add(builder.load(offset), ir.Constant(intp, 1)),
                  offset)
    builder.branch(cond)

    builder.position_at_end(after)
    return builder.load(offset)


def _normalise_slice(builder, size, buffer, start, end, intp, int8):
    zero = ir.Constant(intp, 0)
    effective_size = _trimmed_size(builder, size, buffer, intp, int8)
    codepoints = _codepoint_count(builder, effective_size, buffer, intp, int8)

    start_from_end = builder.add(codepoints, start)
    start_negative = builder.icmp_signed('<', start, zero)
    start_from_end = builder.select(
        builder.icmp_signed('<', start_from_end, zero),
        zero,
        start_from_end,
    )
    start_index = builder.select(start_negative, start_from_end, start)
    start_valid = builder.icmp_signed('<=', start_index, codepoints)
    start_index = builder.select(
        builder.icmp_signed('>', start_index, codepoints),
        codepoints,
        start_index,
    )

    end_from_end = builder.add(codepoints, end)
    end_negative = builder.icmp_signed('<', end, zero)
    end_from_end = builder.select(
        builder.icmp_signed('<', end_from_end, zero),
        zero,
        end_from_end,
    )
    end_index = builder.select(end_negative, end_from_end, end)
    end_index = builder.select(
        builder.icmp_signed('>', end_index, codepoints),
        codepoints,
        end_index,
    )

    ordered = builder.icmp_signed('<=', start_index, end_index)
    valid = builder.and_(start_valid, ordered)
    start_offset = _codepoint_offset(builder, effective_size, buffer,
                                     start_index, intp, int8)
    end_offset = _codepoint_offset(builder, effective_size, buffer,
                                   end_index, intp, int8)
    return start_index, end_index, start_offset, end_offset, valid


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


def _stringdtype_predicate_data(typingctx, data, index, allocator, mode):
    if data != types.voidptr \
            or not isinstance(index, types.Integer) \
            or allocator != types.voidptr:
        return None

    sig = signature(types.boolean, data, types.intp, allocator)

    def codegen(context, builder, signature, args):
        data, index_value, allocator = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        packed = _packed_string_ptr_from_data(builder, data, index_value, intp)
        status, size, buffer = _load_string(builder, allocator, packed, intp,
                                            byte_ptr)

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))

        with builder.if_then(valid):
            effective_size = _trimmed_size(builder, size, buffer, intp, int8)
            nonempty = builder.icmp_unsigned(
                '>', effective_size, ir.Constant(intp, 0))
            with builder.if_then(nonempty):
                builder.store(
                    _emit_stringdtype_predicate(
                        context, builder, mode, effective_size, buffer,
                        intp, int8, int32,
                    ),
                    result,
                )

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_isalpha_data(typingctx, data, index, allocator):
    return _stringdtype_predicate_data(
        typingctx, data, index, allocator, 'isalpha')


@intrinsic
def stringdtype_isalnum_data(typingctx, data, index, allocator):
    return _stringdtype_predicate_data(
        typingctx, data, index, allocator, 'isalnum')


@intrinsic
def stringdtype_isdecimal_data(typingctx, data, index, allocator):
    return _stringdtype_predicate_data(
        typingctx, data, index, allocator, 'isdecimal')


@intrinsic
def stringdtype_isdigit_data(typingctx, data, index, allocator):
    return _stringdtype_predicate_data(
        typingctx, data, index, allocator, 'isdigit')


@intrinsic
def stringdtype_isnumeric_data(typingctx, data, index, allocator):
    return _stringdtype_predicate_data(
        typingctx, data, index, allocator, 'isnumeric')


@intrinsic
def stringdtype_isspace_data(typingctx, data, index, allocator):
    return _stringdtype_predicate_data(
        typingctx, data, index, allocator, 'isspace')


@intrinsic
def stringdtype_islower_data(typingctx, data, index, allocator):
    return _stringdtype_predicate_data(
        typingctx, data, index, allocator, 'islower')


@intrinsic
def stringdtype_isupper_data(typingctx, data, index, allocator):
    return _stringdtype_predicate_data(
        typingctx, data, index, allocator, 'isupper')


@intrinsic
def stringdtype_istitle_data(typingctx, data, index, allocator):
    return _stringdtype_predicate_data(
        typingctx, data, index, allocator, 'istitle')


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


@intrinsic
def stringdtype_compare_data(typingctx, left_data, left_index, left_allocator,
                             right_data, right_index, right_allocator):
    if left_data != types.voidptr \
            or not isinstance(left_index, types.Integer) \
            or left_allocator != types.voidptr \
            or right_data != types.voidptr \
            or not isinstance(right_index, types.Integer) \
            or right_allocator != types.voidptr:
        return None

    sig = signature(types.int32, left_data, types.intp, left_allocator,
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

        result = cgutils.alloca_once(builder, int32)
        builder.store(ir.Constant(int32, 0), result)
        left_valid = builder.icmp_signed(
            '==', left_status, ir.Constant(int32, 0))
        right_valid = builder.icmp_signed(
            '==', right_status, ir.Constant(int32, 0))

        with builder.if_then(builder.and_(left_valid, right_valid)):
            builder.store(
                _stringdtype_byte_compare(
                    builder, left_size, left_buffer, right_size,
                    right_buffer, intp, int8, int32,
                ),
                result,
            )

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_unicode_valid(typingctx, value):
    if not isinstance(value, types.UnicodeType):
        return None

    sig = signature(types.boolean, value)

    def codegen(context, builder, signature, args):
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        value, = args
        unicode_struct = context.make_helper(builder, types.unicode_type,
                                             value=value)
        return _unicode_is_valid(
            builder, unicode_struct.data, unicode_struct.length,
            unicode_struct.kind, intp, int32,
        )

    return sig, codegen


@intrinsic
def stringdtype_unicode_parts(typingctx, value):
    if not isinstance(value, types.UnicodeType):
        return None

    sig = signature(_UNICODE_PARTS_TYPE, value)

    def codegen(context, builder, signature, args):
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        value, = args
        unicode_struct, length, size = _unicode_parts(
            context, builder, value, intp, int32)
        return context.make_tuple(builder, signature.return_type,
                                  [length, size])

    return sig, codegen


@intrinsic
def stringdtype_unicode_utf8_span(typingctx, value, value_length, value_size):
    if not isinstance(value, types.UnicodeType) \
            or not isinstance(value_length, types.Integer) \
            or not isinstance(value_size, types.Integer):
        return None

    sig = signature(_UTF8_SPAN_TYPE, value, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value, value_length, value_size = args

        int1 = ir.IntType(1)
        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        unicode_struct, value_length, value_size = _unicode_parts(
            context, builder, value, intp, int32, value_length, value_size)

        direct = builder.bitcast(unicode_struct.data, byte_ptr)
        buffer = cgutils.alloca_once(builder, byte_ptr)
        allocated = cgutils.alloca_once(builder, int1)
        builder.store(direct, buffer)
        builder.store(cgutils.false_bit, allocated)

        is_ascii = builder.icmp_unsigned(
            '!=', unicode_struct.is_ascii, ir.Constant(int32, 0))
        use_direct = builder.and_(
            is_ascii,
            builder.icmp_signed('==', unicode_struct.kind,
                                ir.Constant(int32, 1)),
        )
        with builder.if_then(builder.not_(use_direct)):
            stack = builder.alloca(
                int8, size=ir.Constant(intp, _PACKED_STRING_SIZE))
            use_stack = builder.icmp_unsigned(
                '<=', value_size, ir.Constant(intp, _PACKED_STRING_SIZE))
            with builder.if_else(use_stack) as (stack_path, heap_path):
                with stack_path:
                    _encode_utf8_buffer(
                        builder, unicode_struct, value_length, stack, intp,
                        int8, int32,
                    )
                    builder.store(stack, buffer)
                with heap_path:
                    malloc_type = ir.FunctionType(byte_ptr, [intp])
                    malloc = cgutils.get_or_insert_function(
                        builder.module, malloc_type, 'malloc')
                    alloc_size = builder.select(
                        builder.icmp_unsigned('>', value_size,
                                              ir.Constant(intp, 0)),
                        value_size,
                        ir.Constant(intp, 1),
                    )
                    heap = builder.call(malloc, [alloc_size])
                    _encode_utf8_buffer(
                        builder, unicode_struct, value_length, heap, intp,
                        int8, int32,
                    )
                    builder.store(heap, buffer)
                    builder.store(cgutils.true_bit, allocated)

        return context.make_tuple(
            builder, signature.return_type,
            [builder.load(buffer), value_size, builder.load(allocated)],
        )

    return sig, codegen


@intrinsic
def stringdtype_free_utf8_span(typingctx, data, allocated):
    if data != types.voidptr or not isinstance(allocated, types.Boolean):
        return None

    sig = signature(types.void, data, allocated)

    def codegen(context, builder, signature, args):
        data, allocated = args
        free_type = ir.FunctionType(ir.VoidType(), [ir.IntType(8).as_pointer()])
        free = cgutils.get_or_insert_function(builder.module, free_type,
                                              'free')
        with builder.if_then(allocated):
            builder.call(free, [data])
        return context.get_dummy_value()

    return sig, codegen


@intrinsic
def stringdtype_equal_unicode_data(typingctx, data, index, allocator, value,
                                   value_length, value_size):
    if data != types.voidptr \
            or not isinstance(index, types.Integer) \
            or allocator != types.voidptr \
            or not isinstance(value, types.UnicodeType) \
            or not isinstance(value_length, types.Integer) \
            or not isinstance(value_size, types.Integer):
        return None

    sig = signature(types.boolean, data, types.intp, allocator, value,
                    types.intp, types.intp)

    def codegen(context, builder, signature, args):
        data, index_value, allocator, value, value_length, value_size = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        packed = _packed_string_ptr_from_data(builder, data, index_value, intp)
        status, size, buffer = _load_string(builder, allocator, packed, intp,
                                            byte_ptr)

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            builder.store(
                _stringdtype_unicode_equal(
                    builder, size, buffer, value, value_length, value_size,
                    context, intp, int8, int32),
                result,
            )

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_compare_unicode_data(typingctx, data, index, allocator, value,
                                     value_length, value_size):
    if data != types.voidptr \
            or not isinstance(index, types.Integer) \
            or allocator != types.voidptr \
            or not isinstance(value, types.UnicodeType) \
            or not isinstance(value_length, types.Integer) \
            or not isinstance(value_size, types.Integer):
        return None

    sig = signature(types.int32, data, types.intp, allocator, value,
                    types.intp, types.intp)

    def codegen(context, builder, signature, args):
        data, index_value, allocator, value, value_length, value_size = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        packed = _packed_string_ptr_from_data(builder, data, index_value, intp)
        status, size, buffer = _load_string(builder, allocator, packed, intp,
                                            byte_ptr)

        result = cgutils.alloca_once(builder, int32)
        builder.store(ir.Constant(int32, 0), result)
        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            builder.store(
                _stringdtype_unicode_compare(
                    builder, size, buffer, value, value_length, value_size,
                    context, intp, int8, int32),
                result,
            )

        return builder.load(result)

    return sig, codegen


def _stringdtype_utf8_data(typingctx, data, index, allocator, scalar_data,
                           scalar_size, compare):
    if data != types.voidptr \
            or not isinstance(index, types.Integer) \
            or allocator != types.voidptr \
            or scalar_data != types.voidptr \
            or not isinstance(scalar_size, types.Integer):
        return None

    return_type = types.int32 if compare else types.boolean
    sig = signature(return_type, data, types.intp, allocator, scalar_data,
                    types.intp)

    def codegen(context, builder, signature, args):
        data, index_value, allocator, scalar_data, scalar_size = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        packed = _packed_string_ptr_from_data(builder, data, index_value, intp)
        status, size, buffer = _load_string(builder, allocator, packed, intp,
                                            byte_ptr)

        if compare:
            result = cgutils.alloca_once(builder, int32)
            builder.store(ir.Constant(int32, 0), result)
        else:
            result = cgutils.alloca_once(builder, ir.IntType(1))
            builder.store(cgutils.false_bit, result)

        valid = builder.icmp_signed('==', status, ir.Constant(int32, 0))
        with builder.if_then(valid):
            if compare:
                builder.store(
                    _utf8_prefilter_compare(
                        builder, size, buffer, scalar_data, scalar_size, intp,
                        int8, int32,
                    ),
                    result,
                )
            else:
                builder.store(
                    _utf8_word8_equal(
                        builder, size, buffer, scalar_data, scalar_size, intp,
                        int8, int32,
                    ),
                    result,
                )

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_equal_utf8_data(typingctx, data, index, allocator, scalar_data,
                                scalar_size):
    return _stringdtype_utf8_data(
        typingctx, data, index, allocator, scalar_data, scalar_size, False)


@intrinsic
def stringdtype_compare_utf8_data(typingctx, data, index, allocator,
                                  scalar_data, scalar_size):
    return _stringdtype_utf8_data(
        typingctx, data, index, allocator, scalar_data, scalar_size, True)


def _stringdtype_affix_data(typingctx, value_data, value_index,
                            value_allocator, pattern_data, pattern_index,
                            pattern_allocator, start, end, suffix):
    if value_data != types.voidptr \
            or not isinstance(value_index, types.Integer) \
            or value_allocator != types.voidptr \
            or pattern_data != types.voidptr \
            or not isinstance(pattern_index, types.Integer) \
            or pattern_allocator != types.voidptr \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(types.boolean, value_data, types.intp, value_allocator,
                    pattern_data, types.intp, pattern_allocator,
                    types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value_data, value_index_value, value_allocator, \
            pattern_data, pattern_index_value, pattern_allocator, \
            start, end = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        value_packed = _packed_string_ptr_from_data(
            builder, value_data, value_index_value, intp)
        pattern_packed = _packed_string_ptr_from_data(
            builder, pattern_data, pattern_index_value, intp)
        value_status, value_size, value_buffer = _load_string(
            builder, value_allocator, value_packed, intp, byte_ptr)
        pattern_status, pattern_size, pattern_buffer = _load_string(
            builder, pattern_allocator, pattern_packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)

        value_valid = builder.icmp_signed(
            '==', value_status, ir.Constant(int32, 0))
        pattern_valid = builder.icmp_signed(
            '==', pattern_status, ir.Constant(int32, 0))

        with builder.if_then(builder.and_(value_valid, pattern_valid)):
            _, _, start_offset, end_offset, slice_valid = _normalise_slice(
                builder, value_size, value_buffer, start, end, intp, int8)
            pattern_effective_size = _trimmed_size(
                builder, pattern_size, pattern_buffer, intp, int8)
            _store_byte_affix_result(
                builder, result, value_buffer, start_offset, end_offset,
                slice_valid, pattern_buffer, pattern_effective_size, suffix,
                intp, int8, int32)

        return builder.load(result)

    return sig, codegen


def _stringdtype_utf8_affix_data(typingctx, value_data, value_index,
                                 value_allocator, pattern_data, pattern_size,
                                 start, end, suffix):
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

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)
        value_valid = builder.icmp_signed(
            '==', value_status, ir.Constant(int32, 0))

        with builder.if_then(value_valid):
            _, _, start_offset, end_offset, slice_valid = _normalise_slice(
                builder, value_size, value_buffer, start, end, intp, int8)
            _store_byte_affix_result(
                builder, result, value_buffer, start_offset, end_offset,
                slice_valid, pattern_data, pattern_size, suffix, intp, int8,
                int32)

        return builder.load(result)

    return sig, codegen


def _stringdtype_unicode_affix_data(typingctx, value_data, value_index,
                                    value_allocator, pattern, pattern_length,
                                    pattern_size, start, end, suffix):
    if value_data != types.voidptr \
            or not isinstance(value_index, types.Integer) \
            or value_allocator != types.voidptr \
            or not isinstance(pattern, types.UnicodeType) \
            or not isinstance(pattern_length, types.Integer) \
            or not isinstance(pattern_size, types.Integer) \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(types.boolean, value_data, types.intp, value_allocator,
                    pattern, types.intp, types.intp, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value_data, value_index_value, value_allocator, pattern, \
            pattern_length, pattern_size, start, end = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        value_packed = _packed_string_ptr_from_data(
            builder, value_data, value_index_value, intp)
        value_status, value_size, value_buffer = _load_string(
            builder, value_allocator, value_packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)
        value_valid = builder.icmp_signed(
            '==', value_status, ir.Constant(int32, 0))

        with builder.if_then(value_valid):
            unicode_struct, pattern_length, pattern_size = _unicode_parts(
                context, builder, pattern, intp, int32, pattern_length,
                pattern_size)
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
                builder.store(
                    _stringdtype_unicode_region_equal(
                        builder, value_buffer, compare_offset, unicode_struct,
                        ir.Constant(intp, 0), pattern_length, intp, int8,
                        int32,
                    ),
                    result,
                )

        return builder.load(result)

    return sig, codegen


def _utf8_stringdtype_sliced_affix_data(
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
            pattern_index_value, pattern_allocator = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        pattern_packed = _packed_string_ptr_from_data(
            builder, pattern_data, pattern_index_value, intp)
        pattern_status, pattern_size, pattern_buffer = _load_string(
            builder, pattern_allocator, pattern_packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, ir.IntType(1))
        builder.store(cgutils.false_bit, result)
        pattern_valid = builder.icmp_signed(
            '==', pattern_status, ir.Constant(int32, 0))

        with builder.if_then(pattern_valid):
            pattern_effective_size = _trimmed_size(
                builder, pattern_size, pattern_buffer, intp, int8)
            _store_byte_affix_result(
                builder, result, value_data, start_offset, end_offset,
                slice_valid, pattern_buffer, pattern_effective_size, suffix,
                intp, int8, int32)

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_utf8_slice(typingctx, value_data, value_size, start, end):
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


@intrinsic
def stringdtype_startswith_data(typingctx, value_data, value_index,
                                value_allocator, pattern_data, pattern_index,
                                pattern_allocator, start, end):
    return _stringdtype_affix_data(
        typingctx, value_data, value_index, value_allocator,
        pattern_data, pattern_index, pattern_allocator, start, end, False,
    )


@intrinsic
def stringdtype_endswith_data(typingctx, value_data, value_index,
                              value_allocator, pattern_data, pattern_index,
                              pattern_allocator, start, end):
    return _stringdtype_affix_data(
        typingctx, value_data, value_index, value_allocator,
        pattern_data, pattern_index, pattern_allocator, start, end, True,
    )


@intrinsic
def stringdtype_startswith_utf8_data(typingctx, value_data, value_index,
                                     value_allocator, pattern_data,
                                     pattern_size, start, end):
    return _stringdtype_utf8_affix_data(
        typingctx, value_data, value_index, value_allocator, pattern_data,
        pattern_size, start, end, False,
    )


@intrinsic
def stringdtype_endswith_utf8_data(typingctx, value_data, value_index,
                                   value_allocator, pattern_data,
                                   pattern_size, start, end):
    return _stringdtype_utf8_affix_data(
        typingctx, value_data, value_index, value_allocator, pattern_data,
        pattern_size, start, end, True,
    )


@intrinsic
def stringdtype_startswith_unicode_data(typingctx, value_data, value_index,
                                        value_allocator, pattern,
                                        pattern_length, pattern_size, start,
                                        end):
    return _stringdtype_unicode_affix_data(
        typingctx, value_data, value_index, value_allocator, pattern,
        pattern_length, pattern_size, start, end, False,
    )


@intrinsic
def stringdtype_endswith_unicode_data(typingctx, value_data, value_index,
                                      value_allocator, pattern,
                                      pattern_length, pattern_size, start,
                                      end):
    return _stringdtype_unicode_affix_data(
        typingctx, value_data, value_index, value_allocator, pattern,
        pattern_length, pattern_size, start, end, True,
    )


@intrinsic
def utf8_startswith_stringdtype_sliced_data(
        typingctx, value_data, start_offset, end_offset, slice_valid,
        pattern_data, pattern_index, pattern_allocator):
    return _utf8_stringdtype_sliced_affix_data(
        typingctx, value_data, start_offset, end_offset, slice_valid,
        pattern_data, pattern_index, pattern_allocator, False,
    )


@intrinsic
def utf8_endswith_stringdtype_sliced_data(
        typingctx, value_data, start_offset, end_offset, slice_valid,
        pattern_data, pattern_index, pattern_allocator):
    return _utf8_stringdtype_sliced_affix_data(
        typingctx, value_data, start_offset, end_offset, slice_valid,
        pattern_data, pattern_index, pattern_allocator, True,
    )


def _stringdtype_search_data(typingctx, value_data, value_index,
                             value_allocator, pattern_data, pattern_index,
                             pattern_allocator, start, end, mode):
    if value_data != types.voidptr \
            or not isinstance(value_index, types.Integer) \
            or value_allocator != types.voidptr \
            or pattern_data != types.voidptr \
            or not isinstance(pattern_index, types.Integer) \
            or pattern_allocator != types.voidptr \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(types.intp, value_data, types.intp, value_allocator,
                    pattern_data, types.intp, pattern_allocator,
                    types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value_data, value_index_value, value_allocator, \
            pattern_data, pattern_index_value, pattern_allocator, \
            start, end = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
        zero = ir.Constant(intp, 0)
        one = ir.Constant(intp, 1)

        value_packed = _packed_string_ptr_from_data(
            builder, value_data, value_index_value, intp)
        pattern_packed = _packed_string_ptr_from_data(
            builder, pattern_data, pattern_index_value, intp)
        value_status, value_size, value_buffer = _load_string(
            builder, value_allocator, value_packed, intp, byte_ptr)
        pattern_status, pattern_size, pattern_buffer = _load_string(
            builder, pattern_allocator, pattern_packed, intp, byte_ptr)

        result = cgutils.alloca_once(builder, intp)
        if mode == 'count':
            builder.store(zero, result)
        else:
            builder.store(ir.Constant(intp, -1), result)

        value_valid = builder.icmp_signed(
            '==', value_status, ir.Constant(int32, 0))
        pattern_valid = builder.icmp_signed(
            '==', pattern_status, ir.Constant(int32, 0))

        with builder.if_then(builder.and_(value_valid, pattern_valid)):
            start_index, end_index, start_offset, end_offset, slice_valid = \
                _normalise_slice(
                    builder, value_size, value_buffer, start, end, intp, int8)
            pattern_effective_size = _trimmed_size(
                builder, pattern_size, pattern_buffer, intp, int8)
            empty_pattern = builder.icmp_unsigned(
                '==', pattern_effective_size, zero)
            # NumPy's StringDType search trims the value span, but keeps
            # trailing NUL bytes in non-empty substrings except for the
            # one-byte find/rfind case. Count always uses the raw substring.
            if mode == 'count':
                pattern_match_size = builder.select(
                    empty_pattern, zero, pattern_size,
                )
            else:
                short_pattern = builder.icmp_unsigned(
                    '<=', pattern_effective_size, one,
                )
                pattern_match_size = builder.select(
                    short_pattern, pattern_effective_size, pattern_size,
                )

            with builder.if_then(builder.and_(slice_valid, empty_pattern)):
                if mode == 'find':
                    builder.store(start_index, result)
                elif mode == 'rfind':
                    builder.store(end_index, result)
                else:
                    builder.store(builder.add(
                        builder.sub(end_index, start_index), one), result)

            slice_size = builder.sub(end_offset, start_offset)
            nonempty_pattern = builder.not_(empty_pattern)
            fits = builder.icmp_unsigned('<=', pattern_match_size,
                                         slice_size)
            with builder.if_then(builder.and_(slice_valid,
                                              builder.and_(nonempty_pattern,
                                                           fits))):
                memcmp_type = ir.FunctionType(
                    int32, [byte_ptr, byte_ptr, intp])
                memcmp = cgutils.get_or_insert_function(
                    builder.module, memcmp_type, 'memcmp',
                )
                first_pattern = builder.load(pattern_buffer)
                found = cgutils.alloca_once(builder, ir.IntType(1))
                builder.store(cgutils.false_bit, found)

                if mode == 'rfind':
                    pos = cgutils.alloca_once(builder, intp)
                    last = builder.sub(end_offset, pattern_match_size)
                    builder.store(last, pos)
                    cond = builder.append_basic_block(
                        'stringdtype.rfind.cond')
                    body = builder.append_basic_block(
                        'stringdtype.rfind.body')
                    decrement = builder.append_basic_block(
                        'stringdtype.rfind.decrement')
                    after = builder.append_basic_block(
                        'stringdtype.rfind.after')
                    builder.branch(cond)

                    builder.position_at_end(cond)
                    in_range = builder.icmp_signed(
                        '>=', builder.load(pos), start_offset)
                    builder.cbranch(builder.and_(
                        in_range, builder.not_(builder.load(found))),
                        body, after)

                    builder.position_at_end(body)
                    value_byte = builder.load(
                        builder.gep(value_buffer, [builder.load(pos)]))
                    first_matches = builder.icmp_unsigned(
                        '==', value_byte, first_pattern)
                    with builder.if_then(first_matches):
                        cmp_buffer = builder.gep(value_buffer,
                                                 [builder.load(pos)])
                        cmp_result = builder.call(
                            memcmp, [cmp_buffer, pattern_buffer,
                                     pattern_match_size],
                        )
                        matched = builder.icmp_signed(
                            '==', cmp_result, ir.Constant(int32, 0))
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
                    cond = builder.append_basic_block(
                        'stringdtype.find.cond')
                    body = builder.append_basic_block(
                        'stringdtype.find.body')
                    advance = builder.append_basic_block(
                        'stringdtype.find.advance')
                    after = builder.append_basic_block(
                        'stringdtype.find.after')
                    builder.branch(cond)

                    builder.position_at_end(cond)
                    in_range = builder.icmp_signed(
                        '<=', builder.load(pos), last)
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
                        cmp_buffer = builder.gep(value_buffer,
                                                 [builder.load(pos)])
                        cmp_result = builder.call(
                            memcmp, [cmp_buffer, pattern_buffer,
                                     pattern_match_size],
                        )
                        builder.store(
                            builder.icmp_signed(
                                '==', cmp_result, ir.Constant(int32, 0)),
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
                            builder.store(builder.add(builder.load(result),
                                                      one), result)
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
                        builder.store(builder.add(builder.load(pos), one),
                                      pos)
                    builder.branch(cond)

                    builder.position_at_end(after)

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_find_data(typingctx, value_data, value_index, value_allocator,
                          pattern_data, pattern_index, pattern_allocator,
                          start, end):
    return _stringdtype_search_data(
        typingctx, value_data, value_index, value_allocator,
        pattern_data, pattern_index, pattern_allocator, start, end, 'find',
    )


@intrinsic
def stringdtype_rfind_data(typingctx, value_data, value_index, value_allocator,
                           pattern_data, pattern_index, pattern_allocator,
                           start, end):
    return _stringdtype_search_data(
        typingctx, value_data, value_index, value_allocator,
        pattern_data, pattern_index, pattern_allocator, start, end, 'rfind',
    )


@intrinsic
def stringdtype_count_data(typingctx, value_data, value_index, value_allocator,
                           pattern_data, pattern_index, pattern_allocator,
                           start, end):
    return _stringdtype_search_data(
        typingctx, value_data, value_index, value_allocator,
        pattern_data, pattern_index, pattern_allocator, start, end, 'count',
    )


def _stringdtype_unicode_search_data(typingctx, value_data, value_index,
                                     value_allocator, pattern, pattern_length,
                                     pattern_size, start, end, mode):
    if value_data != types.voidptr \
            or not isinstance(value_index, types.Integer) \
            or value_allocator != types.voidptr \
            or not isinstance(pattern, types.UnicodeType) \
            or not isinstance(pattern_length, types.Integer) \
            or not isinstance(pattern_size, types.Integer) \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(types.intp, value_data, types.intp, value_allocator,
                    pattern, types.intp, types.intp, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value_data, value_index_value, value_allocator, pattern, \
            pattern_length, pattern_size, start, end = args

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
                _stringdtype_unicode_search(
                    builder, value_size, value_buffer, pattern,
                    pattern_length, pattern_size, context, start, end, mode,
                    intp, int8, int32,
                ),
                result,
            )

        return builder.load(result)

    return sig, codegen


def _unicode_stringdtype_search_data(typingctx, value, pattern_data,
                                     value_length, value_size, pattern_index,
                                     pattern_allocator, start, end, mode):
    if not isinstance(value, types.UnicodeType) \
            or pattern_data != types.voidptr \
            or not isinstance(value_length, types.Integer) \
            or not isinstance(value_size, types.Integer) \
            or not isinstance(pattern_index, types.Integer) \
            or pattern_allocator != types.voidptr \
            or not isinstance(start, types.Integer) \
            or not isinstance(end, types.Integer):
        return None

    sig = signature(types.intp, value, pattern_data, types.intp, types.intp,
                    types.intp, pattern_allocator, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        value, pattern_data, value_length, value_size, pattern_index_value, \
            pattern_allocator, start, end = args

        int8 = ir.IntType(8)
        int32 = ir.IntType(32)
        intp = context.get_value_type(types.intp)
        byte_ptr = int8.as_pointer()
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
            builder.store(
                _unicode_stringdtype_search(
                    builder, value, value_length, value_size, pattern_size,
                    pattern_buffer, context, start, end, mode, intp, int8,
                    int32,
                ),
                result,
            )

        return builder.load(result)

    return sig, codegen


@intrinsic
def stringdtype_find_unicode_data(typingctx, value_data, value_index,
                                  value_allocator, pattern, pattern_length,
                                  pattern_size, start, end):
    return _stringdtype_unicode_search_data(
        typingctx, value_data, value_index, value_allocator, pattern,
        pattern_length, pattern_size, start, end, 'find',
    )


@intrinsic
def stringdtype_rfind_unicode_data(typingctx, value_data, value_index,
                                   value_allocator, pattern, pattern_length,
                                   pattern_size, start, end):
    return _stringdtype_unicode_search_data(
        typingctx, value_data, value_index, value_allocator, pattern,
        pattern_length, pattern_size, start, end, 'rfind',
    )


@intrinsic
def stringdtype_count_unicode_data(typingctx, value_data, value_index,
                                   value_allocator, pattern, pattern_length,
                                   pattern_size, start, end):
    return _stringdtype_unicode_search_data(
        typingctx, value_data, value_index, value_allocator, pattern,
        pattern_length, pattern_size, start, end, 'count',
    )


@intrinsic
def unicode_find_stringdtype_data(typingctx, value, pattern_data, value_length,
                                  value_size, pattern_index,
                                  pattern_allocator, start, end):
    return _unicode_stringdtype_search_data(
        typingctx, value, pattern_data, value_length, value_size,
        pattern_index, pattern_allocator, start, end, 'find',
    )


@intrinsic
def unicode_rfind_stringdtype_data(typingctx, value, pattern_data,
                                   value_length, value_size, pattern_index,
                                   pattern_allocator, start, end):
    return _unicode_stringdtype_search_data(
        typingctx, value, pattern_data, value_length, value_size,
        pattern_index, pattern_allocator, start, end, 'rfind',
    )


@intrinsic
def unicode_count_stringdtype_data(typingctx, value, pattern_data,
                                   value_length, value_size, pattern_index,
                                   pattern_allocator, start, end):
    return _unicode_stringdtype_search_data(
        typingctx, value, pattern_data, value_length, value_size,
        pattern_index, pattern_allocator, start, end, 'count',
    )


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
