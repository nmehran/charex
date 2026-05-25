"""
Numba overloads for numpy.character routines
Copyright (c) 2022-present, Nima Mehrani
"""

from charex.core import JIT_OPTIONS
from llvmlite import ir
from numba.core import cgutils
from numba.cpython.charseq import charseq_get_code, unicode_charseq_get_code
from numba.cpython.unicode_support import (
    _PyUnicode_IsAlpha, _PyUnicode_IsDecimalDigit, _PyUnicode_IsDigit,
    _PyUnicode_IsLowercase, _PyUnicode_IsNumeric, _PyUnicode_IsSpace,
    _PyUnicode_IsTitlecase, _PyUnicode_IsUppercase
)
from numba.extending import intrinsic, register_jitable
from numba import types
import numpy as np


COLD_JIT_OPTIONS = dict(JIT_OPTIONS, forceinline=False)


# ----------------------------------------------------------------------------------------------------------------------
# Comparison Operators


@register_jitable(**JIT_OPTIONS)
def _ensure_comparison_shape(len_chr, len_cmp):
    """Ensure operands can broadcast to the existing comparison shape."""
    if len_cmp > 1 and len_cmp != len_chr:
        raise ValueError('shape mismatch: objects cannot be broadcast to a '
                         'single shape.  Mismatch is between arg 0 and arg 1.')


@register_jitable(**JIT_OPTIONS)
def _rstrip_ord(chr_ord):
    return chr_ord == 0 or 9 <= chr_ord <= 13 or chr_ord == 32


@register_jitable(**JIT_OPTIONS)
def _trim_ord(chr_ord, rstrip):
    return chr_ord == 0 or (rstrip and (9 <= chr_ord <= 13 or chr_ord == 32))


@register_jitable(**JIT_OPTIONS)
def _trim_suffix(chr_array, start, end, rstrip):
    for p in range(start, end):
        if not _trim_ord(chr_array[p], rstrip):
            return False
    return True


@register_jitable(**JIT_OPTIONS)
def _record_len(chr_array, start, size_chr):
    end = start + size_chr - 1
    for p in range(end, start - 1, -1):
        if chr_array[p]:
            return p - start + 1
    return 0


@register_jitable(**JIT_OPTIONS)
def _rstrip_record_len(chr_array, start, size_chr):
    p = start + size_chr - 1
    while p >= start and _rstrip_ord(chr_array[p]):
        p -= 1
    return p - start + 1


@register_jitable(**JIT_OPTIONS)
def _comparison_record_len(chr_array, start, size_chr, rstrip):
    if rstrip:
        return _rstrip_record_len(chr_array, start, size_chr)
    return _record_len(chr_array, start, size_chr)


@register_jitable(**JIT_OPTIONS, locals={'cmp_ord': types.int32})
def _compare_records(chr_array, start, chr_len,
                     cmp_array, cmp_start, cmp_len):
    size_stride = min(chr_len, cmp_len)
    for j in range(size_stride):
        cmp_ord = chr_array[start + j] - cmp_array[cmp_start + j]
        if cmp_ord != 0:
            return cmp_ord
    return chr_len - cmp_len


@intrinsic
def _memcmp_array(typingctx, chr_array, start, cmp_array, cmp_start, nitems):
    """Compare contiguous slices of two same-width ordinal arrays."""
    if not isinstance(chr_array, types.Array) \
            or not isinstance(cmp_array, types.Array):
        raise TypeError('memcmp operands must be arrays')
    if chr_array.dtype.bitwidth != cmp_array.dtype.bitwidth:
        raise TypeError('memcmp operands must have same-width dtypes')

    def codegen(context, builder, signature, args):
        chr_type, _, cmp_type, _, _ = signature.args
        chr_value, start_value, cmp_value, cmp_start_value, nitems_value = args
        chr_struct = context.make_array(chr_type)(context, builder, chr_value)
        cmp_struct = context.make_array(cmp_type)(context, builder, cmp_value)

        itemsize = chr_type.dtype.bitwidth // 8
        start_bytes = builder.mul(start_value,
                                  ir.Constant(start_value.type, itemsize))
        cmp_start_bytes = builder.mul(
            cmp_start_value,
            ir.Constant(cmp_start_value.type, itemsize),
        )
        nbytes = builder.mul(nitems_value,
                             ir.Constant(nitems_value.type, itemsize))
        chr_ptr = cgutils.pointer_add(builder, chr_struct.data, start_bytes,
                                      cgutils.voidptr_t)
        cmp_ptr = cgutils.pointer_add(builder, cmp_struct.data,
                                      cmp_start_bytes, cgutils.voidptr_t)
        fnty = ir.FunctionType(ir.IntType(32), [
            cgutils.voidptr_t,
            cgutils.voidptr_t,
            nbytes.type,
        ])
        fn = cgutils.insert_pure_function(builder.module, fnty, 'memcmp')
        return builder.call(fn, [chr_ptr, cmp_ptr, nbytes])

    sig = types.int32(chr_array, start, cmp_array, cmp_start, nitems)
    return sig, codegen


@intrinsic
def _is_zero_chunk8(typingctx, chr_array, start):
    """Return True if eight contiguous ordinal values are all zero."""
    if not isinstance(chr_array, types.Array):
        raise TypeError('chunk operand must be an array')

    def codegen(context, builder, signature, args):
        chr_type, _ = signature.args
        chr_value, start_value = args
        chr_struct = context.make_array(chr_type)(context, builder, chr_value)

        bitwidth = chr_type.dtype.bitwidth
        itemsize = bitwidth // 8
        start_bytes = builder.mul(start_value,
                                  ir.Constant(start_value.type, itemsize))
        ptr = cgutils.pointer_add(builder, chr_struct.data, start_bytes,
                                  cgutils.voidptr_t)
        wide_type = ir.IntType(bitwidth * 8)
        wide_ptr = builder.bitcast(ptr, wide_type.as_pointer())
        value = builder.load(wide_ptr, align=1)
        return builder.icmp_unsigned('==', value,
                                     ir.Constant(wide_type, 0))

    sig = types.boolean(chr_array, start)
    return sig, codegen


@intrinsic
def _mismatch_chunk8(typingctx, chr_array, start, cmp_array, cmp_start):
    """Return the first mismatching offset in an 8-code-unit chunk, else 8."""
    if not isinstance(chr_array, types.Array) \
            or not isinstance(cmp_array, types.Array):
        raise TypeError('chunk operands must be arrays')
    if chr_array.dtype.bitwidth != cmp_array.dtype.bitwidth:
        raise TypeError('chunk operands must have same-width dtypes')

    def codegen(context, builder, signature, args):
        chr_type, _, cmp_type, _ = signature.args
        chr_value, start_value, cmp_value, cmp_start_value = args
        chr_struct = context.make_array(chr_type)(context, builder, chr_value)
        cmp_struct = context.make_array(cmp_type)(context, builder, cmp_value)

        bitwidth = chr_type.dtype.bitwidth
        itemsize = bitwidth // 8
        wide_type = ir.IntType(bitwidth * 8)

        start_bytes = builder.mul(start_value,
                                  ir.Constant(start_value.type, itemsize))
        cmp_start_bytes = builder.mul(
            cmp_start_value,
            ir.Constant(cmp_start_value.type, itemsize),
        )
        chr_ptr = cgutils.pointer_add(builder, chr_struct.data, start_bytes,
                                      cgutils.voidptr_t)
        cmp_ptr = cgutils.pointer_add(builder, cmp_struct.data,
                                      cmp_start_bytes, cgutils.voidptr_t)
        chr_wide = builder.load(
            builder.bitcast(chr_ptr, wide_type.as_pointer()),
            align=1,
        )
        cmp_wide = builder.load(
            builder.bitcast(cmp_ptr, wide_type.as_pointer()),
            align=1,
        )
        diff = builder.xor(chr_wide, cmp_wide)
        bit_index = builder.cttz(diff, ir.Constant(ir.IntType(1), 0))
        unit_index = builder.udiv(bit_index,
                                  ir.Constant(wide_type, bitwidth))

        return_type = context.get_value_type(types.intp)
        if wide_type.width > return_type.width:
            return builder.trunc(unit_index, return_type)
        if wide_type.width < return_type.width:
            return builder.zext(unit_index, return_type)
        return unit_index

    sig = types.intp(chr_array, start, cmp_array, cmp_start)
    return sig, codegen


@intrinsic
def _mismatch_chunk4(typingctx, chr_array, start, cmp_array, cmp_start):
    """Return the first mismatching offset in a 4-code-unit chunk, else 4."""
    if not isinstance(chr_array, types.Array) \
            or not isinstance(cmp_array, types.Array):
        raise TypeError('chunk operands must be arrays')
    if chr_array.dtype.bitwidth != cmp_array.dtype.bitwidth:
        raise TypeError('chunk operands must have same-width dtypes')

    def codegen(context, builder, signature, args):
        chr_type, _, cmp_type, _ = signature.args
        chr_value, start_value, cmp_value, cmp_start_value = args
        chr_struct = context.make_array(chr_type)(context, builder, chr_value)
        cmp_struct = context.make_array(cmp_type)(context, builder, cmp_value)

        bitwidth = chr_type.dtype.bitwidth
        itemsize = bitwidth // 8
        wide_type = ir.IntType(bitwidth * 4)

        start_bytes = builder.mul(start_value,
                                  ir.Constant(start_value.type, itemsize))
        cmp_start_bytes = builder.mul(
            cmp_start_value,
            ir.Constant(cmp_start_value.type, itemsize),
        )
        chr_ptr = cgutils.pointer_add(builder, chr_struct.data, start_bytes,
                                      cgutils.voidptr_t)
        cmp_ptr = cgutils.pointer_add(builder, cmp_struct.data,
                                      cmp_start_bytes, cgutils.voidptr_t)
        chr_wide = builder.load(
            builder.bitcast(chr_ptr, wide_type.as_pointer()),
            align=1,
        )
        cmp_wide = builder.load(
            builder.bitcast(cmp_ptr, wide_type.as_pointer()),
            align=1,
        )
        diff = builder.xor(chr_wide, cmp_wide)
        bit_index = builder.cttz(diff, ir.Constant(ir.IntType(1), 0))
        unit_index = builder.udiv(bit_index,
                                  ir.Constant(wide_type, bitwidth))

        return_type = context.get_value_type(types.intp)
        if wide_type.width > return_type.width:
            return builder.trunc(unit_index, return_type)
        if wide_type.width < return_type.width:
            return builder.zext(unit_index, return_type)
        return unit_index

    sig = types.intp(chr_array, start, cmp_array, cmp_start)
    return sig, codegen


@register_jitable(**JIT_OPTIONS)
def _trim_suffix_zero8(chr_array, start, end, rstrip):
    p = end
    while p - 8 >= start:
        if not _is_zero_chunk8(chr_array, p - 8):
            break
        p -= 8
    while p > start:
        if not _trim_ord(chr_array[p - 1], rstrip):
            return False
        p -= 1
    return True


@register_jitable(**JIT_OPTIONS)
def _equal_records_after_raw_mismatch(chr_array, start, size_chr,
                                      cmp_array, cmp_start, rstrip):
    for j in range(1, size_chr):
        chr_ord = chr_array[start + j]
        cmp_ord = cmp_array[cmp_start + j]
        if chr_ord != cmp_ord:
            if _trim_ord(chr_ord, rstrip) and _trim_ord(cmp_ord, rstrip):
                return _trim_suffix_zero8(chr_array, start + j + 1,
                                          start + size_chr, rstrip) \
                    and _trim_suffix_zero8(cmp_array, cmp_start + j + 1,
                                           cmp_start + size_chr, rstrip)
            return False
    return True


@register_jitable(**COLD_JIT_OPTIONS)
def _equal_wide_after_raw_mismatch(chr_array, start, size_chr,
                                   cmp_array, cmp_start, rstrip):
    j = 1
    while j + 8 <= size_chr:
        mismatch = _mismatch_chunk8(chr_array, start + j,
                                    cmp_array, cmp_start + j)
        if mismatch < 8:
            return _equal_fixed_mismatch(
                chr_array, start, cmp_array, cmp_start,
                j + mismatch, size_chr, rstrip,
            )
        j += 8

    while j < size_chr:
        if chr_array[start + j] != cmp_array[cmp_start + j]:
            return _equal_fixed_mismatch(
                chr_array, start, cmp_array, cmp_start,
                j, size_chr, rstrip,
            )
        j += 1
    return True


@register_jitable(**JIT_OPTIONS)
def _equal_records(chr_array, start, size_chr,
                   cmp_array, cmp_start, size_cmp, rstrip, cmp_len):
    size_stride = min(size_chr, size_cmp)
    if size_stride:
        chr_ord = chr_array[start]
        cmp_ord = cmp_array[cmp_start]
        if chr_ord != cmp_ord:
            if _trim_ord(chr_ord, rstrip) and _trim_ord(cmp_ord, rstrip):
                return _trim_suffix(chr_array, start + 1,
                                    start + size_chr, rstrip) \
                    and _trim_suffix(cmp_array, cmp_start + 1,
                                     cmp_start + size_cmp, rstrip)
            return False

    if size_chr == size_cmp and size_chr:
        if _memcmp_array(chr_array, start, cmp_array, cmp_start, size_chr) == 0:
            return True
        if size_chr >= 32:
            return _equal_wide_after_raw_mismatch(
                chr_array, start, size_chr,
                cmp_array, cmp_start, rstrip,
            )
        return _equal_records_after_raw_mismatch(
            chr_array, start, size_chr,
            cmp_array, cmp_start,
            rstrip,
        )

    chr_len = _comparison_record_len(chr_array, start, size_chr, rstrip)
    if cmp_len < 0:
        cmp_len = _comparison_record_len(cmp_array, cmp_start, size_cmp,
                                         rstrip)
    return chr_len == cmp_len and (
        chr_len == 0
        or _memcmp_array(chr_array, start, cmp_array, cmp_start, chr_len) == 0
    )


@register_jitable(**JIT_OPTIONS)
def _equal_fixed_mismatch(chr_array, start,
                          cmp_array, cmp_start, offset, size_chr, rstrip):
    chr_ord = chr_array[start + offset]
    cmp_ord = cmp_array[cmp_start + offset]
    if _trim_ord(chr_ord, rstrip) and _trim_ord(cmp_ord, rstrip):
        return _trim_suffix_zero8(chr_array, start + offset + 1,
                                  start + size_chr, rstrip) \
            and _trim_suffix_zero8(cmp_array, cmp_start + offset + 1,
                                   cmp_start + size_chr, rstrip)
    return False


@register_jitable(**JIT_OPTIONS)
def _equal_sub32_bytes_record(chr_array, start,
                              cmp_array, cmp_start, size_chr, rstrip):
    offset = 0
    if size_chr >= 8:
        mismatch = _mismatch_chunk8(chr_array, start, cmp_array, cmp_start)
        if mismatch < 8:
            return _equal_fixed_mismatch(
                chr_array, start, cmp_array, cmp_start,
                mismatch, size_chr, rstrip,
            )
        offset = 8
    if size_chr >= 16:
        mismatch = _mismatch_chunk8(chr_array, start + 8,
                                    cmp_array, cmp_start + 8)
        if mismatch < 8:
            return _equal_fixed_mismatch(
                chr_array, start, cmp_array, cmp_start,
                8 + mismatch, size_chr, rstrip,
            )
        offset = 16
    if size_chr >= 24:
        mismatch = _mismatch_chunk8(chr_array, start + 16,
                                    cmp_array, cmp_start + 16)
        if mismatch < 8:
            return _equal_fixed_mismatch(
                chr_array, start, cmp_array, cmp_start,
                16 + mismatch, size_chr, rstrip,
            )
        offset = 24
    if offset + 4 <= size_chr:
        mismatch = _mismatch_chunk4(chr_array, start + offset,
                                    cmp_array, cmp_start + offset)
        if mismatch < 4:
            return _equal_fixed_mismatch(
                chr_array, start, cmp_array, cmp_start,
                offset + mismatch, size_chr, rstrip,
            )
        offset += 4

    while offset < size_chr:
        if chr_array[start + offset] != cmp_array[cmp_start + offset]:
            return _equal_fixed_mismatch(
                chr_array, start, cmp_array, cmp_start,
                offset, size_chr, rstrip,
            )
        offset += 1
    return True


@register_jitable(**JIT_OPTIONS)
def _equal_sub32_unicode_record(chr_array, start,
                                cmp_array, cmp_start, size_chr, rstrip):
    offset = 0
    while offset + 4 <= size_chr:
        mismatch = _mismatch_chunk4(chr_array, start + offset,
                                    cmp_array, cmp_start + offset)
        if mismatch < 4:
            return _equal_fixed_mismatch(
                chr_array, start, cmp_array, cmp_start,
                offset + mismatch, size_chr, rstrip,
            )
        offset += 4

    while offset < size_chr:
        if chr_array[start + offset] != cmp_array[cmp_start + offset]:
            return _equal_fixed_mismatch(
                chr_array, start, cmp_array, cmp_start,
                offset, size_chr, rstrip,
            )
        offset += 1
    return True


@register_jitable(**JIT_OPTIONS)
def _equal_sub32(chr_array, len_chr, size_chr,
                 cmp_array, len_cmp, size_cmp, as_bytes, rstrip):
    """Native np.char.equal for same-width records below 32 code units."""
    if size_chr != size_cmp or size_chr >= 32:
        return equal(chr_array, len_chr, size_chr,
                     cmp_array, len_cmp, size_cmp, rstrip)

    _ensure_comparison_shape(len_chr, len_cmp)
    equal_to = np.empty(len_chr, 'bool')
    stride = stride_cmp = 0
    step_cmp = (len_cmp > 1 and size_chr) or 0
    for i in range(len_chr):
        if as_bytes:
            equal_to[i] = _equal_sub32_bytes_record(
                chr_array, stride, cmp_array, stride_cmp, size_chr, rstrip,
            )
        else:
            equal_to[i] = _equal_sub32_unicode_record(
                chr_array, stride, cmp_array, stride_cmp, size_chr, rstrip,
            )
        stride += size_chr
        stride_cmp += step_cmp
    return equal_to


@register_jitable(**JIT_OPTIONS)
def equal_sub32_bytes(chr_array, len_chr, size_chr,
                      cmp_array, len_cmp, size_cmp, rstrip=True):
    return _equal_sub32(chr_array, len_chr, size_chr,
                        cmp_array, len_cmp, size_cmp, True, rstrip)


@register_jitable(**JIT_OPTIONS)
def equal_sub32_unicode(chr_array, len_chr, size_chr,
                        cmp_array, len_cmp, size_cmp, rstrip=True):
    return _equal_sub32(chr_array, len_chr, size_chr,
                        cmp_array, len_cmp, size_cmp, False, rstrip)


@register_jitable(**JIT_OPTIONS, locals={'cmp_ord': types.int32})
def greater_equal(chr_array, len_chr, size_chr,
                  cmp_array, len_cmp, size_cmp, inv=False, rstrip=True):
    """Native Implementation of np.char.greater_equal"""
    if 1 == size_chr == size_cmp and not rstrip:
        return cmp_array >= chr_array if inv else chr_array >= cmp_array

    _ensure_comparison_shape(len_chr, len_cmp)
    greater_equal_than = np.empty(len_chr, 'bool')
    stride = stride_cmp = 0
    step_cmp = (len_cmp > 1 and size_cmp) or 0
    for i in range(len_chr):
        cmp_ord = _compare_records(
            chr_array, stride,
            _comparison_record_len(chr_array, stride, size_chr, rstrip),
            cmp_array, stride_cmp,
            _comparison_record_len(cmp_array, stride_cmp, size_cmp, rstrip),
        )
        if inv:
            cmp_ord = -cmp_ord
        greater_equal_than[i] = cmp_ord >= 0
        stride += size_chr
        stride_cmp += step_cmp
    return greater_equal_than


@register_jitable(**JIT_OPTIONS, locals={'cmp_ord': types.int32})
def greater(chr_array, len_chr, size_chr,
            cmp_array, len_cmp, size_cmp, inv=False, rstrip=True):
    """Native Implementation of np.char.greater"""
    if 1 == size_chr == size_cmp and not rstrip:
        return cmp_array > chr_array if inv else chr_array > cmp_array

    _ensure_comparison_shape(len_chr, len_cmp)
    greater_than = np.empty(len_chr, 'bool')
    stride = stride_cmp = 0
    step_cmp = (len_cmp > 1 and size_cmp) or 0
    for i in range(len_chr):
        cmp_ord = _compare_records(
            chr_array, stride,
            _comparison_record_len(chr_array, stride, size_chr, rstrip),
            cmp_array, stride_cmp,
            _comparison_record_len(cmp_array, stride_cmp, size_cmp, rstrip),
        )
        if inv:
            cmp_ord = -cmp_ord
        greater_than[i] = cmp_ord > 0
        stride += size_chr
        stride_cmp += step_cmp
    return greater_than


@register_jitable(**JIT_OPTIONS)
def equal(chr_array, len_chr, size_chr,
          cmp_array, len_cmp, size_cmp, rstrip=True):
    """Native Implementation of np.char.equal"""
    if 1 == size_chr == size_cmp and not rstrip:
        return chr_array == cmp_array

    _ensure_comparison_shape(len_chr, len_cmp)
    equal_to = np.empty(len_chr, 'bool')
    stride = stride_cmp = 0
    step_cmp = (len_cmp > 1 and size_cmp) or 0
    cmp_len = -1
    if len_cmp == 1:
        cmp_len = _comparison_record_len(cmp_array, 0, size_cmp, rstrip)
    for i in range(len_chr):
        equal_to[i] = _equal_records(chr_array, stride, size_chr,
                                     cmp_array, stride_cmp, size_cmp,
                                     rstrip, cmp_len)
        stride += size_chr
        stride_cmp += step_cmp
    return equal_to


@register_jitable(**JIT_OPTIONS)
def compare_chararrays(chr_array, len_chr, size_chr,
                       cmp_array, len_cmp, size_cmp, inv, cmp, rstrip):
    """Native Implementation of np.char.compare_chararrays"""
    # { “<”,    “<=”,     “==”,     “>=”,   “>”,     “!=”}
    # { (60,) (60, 61), (61, 61), (62, 61), (62,), (33, 61) }
    # The argument cmp can be passed as bytes or string, hence ordinal mapping.
    if len(cmp) == 1:
        cmp_ord = ord(cmp)
        if cmp_ord == 60:
            return ~greater_equal(chr_array, len_chr, size_chr,
                                  cmp_array, len_cmp, size_cmp, inv, rstrip)
        elif cmp_ord == 62:
            return greater(chr_array, len_chr, size_chr,
                           cmp_array, len_cmp, size_cmp, inv, rstrip)
    elif len(cmp) == 2 and ord(cmp[1]) == 61:
        cmp_ord = ord(cmp[0])
        if cmp_ord == 60:
            return ~greater(chr_array, len_chr, size_chr,
                            cmp_array, len_cmp, size_cmp, inv, rstrip)
        elif cmp_ord == 61:
            return equal(chr_array, len_chr, size_chr,
                         cmp_array, len_cmp, size_cmp, rstrip)
        elif cmp_ord == 62:
            return greater_equal(chr_array, len_chr, size_chr,
                                 cmp_array, len_cmp, size_cmp, inv, rstrip)
        elif cmp_ord == 33:
            return ~equal(chr_array, len_chr, size_chr,
                          cmp_array, len_cmp, size_cmp, rstrip)
    raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")


# ----------------------------------------------------------------------------------------------------------------------
# String Information


@register_jitable(**JIT_OPTIONS)
def _init_sub_indices(start, end, size_chr):
    """Initialize substring start and end indices"""
    if end is None:
        end = size_chr
    else:
        end = max(min(end, size_chr), -size_chr)
    if start < 0:
        start = max(start, -size_chr)
    return start, end


@register_jitable(**JIT_OPTIONS)
def _get_sub_indices(chr_lens, len_chr, sub_lens, len_sub, start, end, i):
    """Calculate substring start and end indices"""
    n_chr = chr_lens[(len_chr > 1 and i) or 0]
    n_sub = sub_lens[(len_sub > 1 and i) or 0]
    o = max(start < 0 and start + n_chr or start, 0)
    n = min(n_chr, max(end < 0 and end + n_chr or end, 0))
    return n_chr, n_sub, o, n


@register_jitable(**JIT_OPTIONS)
def count(chr_array, len_chr, size_chr,
          sub_array, len_sub, size_sub, start, end):
    """Native Implementation of np.char.count"""
    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return np.zeros(max(len_chr, len_sub), 'int64')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    count_sub = np.zeros(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_sub = (len_sub > 1 and size_sub) or 0
    stride = stride_sub = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride
            n += stride
            while o + n_sub <= n:
                for p in range(n_sub):
                    if chr_array[o + p] != sub_array[stride_sub + p]:
                        o += 1
                        break
                else:
                    count_sub[i] += 1
                    o += n_sub
        else:
            count_sub[i] = o <= n and max(1 + n - o, 1)
        stride += size_chr
        stride_sub += size_sub
    return count_sub


@register_jitable(**JIT_OPTIONS)
def endswith(chr_array, len_chr, size_chr,
             sub_array, len_sub, size_sub,
             start, end):
    """Native Implementation of np.char.endswith"""
    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return np.zeros(max(len_chr, len_sub), 'bool')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    endswith_sub = np.ones(len_cast, 'bool')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_sub = (len_sub > 1 and size_sub) or 0
    stride = stride_sub = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if o + n_sub <= n:
            n += stride - 1
            r = stride_sub + n_sub - 1
            for p in range(n_sub):
                if chr_array[n - p] != sub_array[r - p]:
                    endswith_sub[i] = False
                    break
        else:
            endswith_sub[i] = not n_sub and o <= n
        stride += size_chr
        stride_sub += size_sub
    return endswith_sub


@register_jitable(**JIT_OPTIONS)
def startswith(chr_array, len_chr, size_chr,
               sub_array, len_sub, size_sub,
               start, end):
    """Native Implementation of np.char.startswith"""
    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return np.zeros(max(len_chr, len_sub), 'bool')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    startswith_sub = np.ones(len_cast, 'bool')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_sub = (len_sub > 1 and size_sub) or 0
    stride = stride_sub = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if o + n_sub <= n:
            o = stride + o
            for p in range(n_sub):
                if chr_array[o + p] != sub_array[stride_sub + p]:
                    startswith_sub[i] = False
                    break
        else:
            startswith_sub[i] = not n_sub and o <= n
        stride += size_chr
        stride_sub += size_sub
    return startswith_sub


@register_jitable(**JIT_OPTIONS)
def find(chr_array, len_chr, size_chr,
         sub_array, len_sub, size_sub, start, end):
    """Native Implementation of np.char.find"""
    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return -np.ones(max(len_chr, len_sub), 'int64')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    find_sub = -np.ones(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_sub = (len_sub > 1 and size_sub) or 0
    stride = stride_sub = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride
            n += stride
            while o + n_sub <= n:
                for p in range(n_sub):
                    if chr_array[o + p] != sub_array[stride_sub + p]:
                        o += 1
                        break
                else:
                    find_sub[i] = o - stride
                    break
        else:
            find_sub[i] = (o <= n and o + 1) - 1
        stride += size_chr
        stride_sub += size_sub
    return find_sub


@register_jitable(**JIT_OPTIONS)
def rfind(chr_array, len_chr, size_chr,
          sub_array, len_sub, size_sub,
          start, end):
    """Native Implementation of np.char.rfind"""
    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        return -np.ones(max(len_chr, len_sub), 'int64')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    rfind_sub = -np.ones(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_sub = (len_sub > 1 and size_sub) or 0
    stride = stride_sub = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride - 1
            n += stride - 1
            r = stride_sub + n_sub - 1
            while n - n_sub >= o:
                for p in range(n_sub):
                    if chr_array[n - p] != sub_array[r - p]:
                        n -= 1
                        break
                else:
                    rfind_sub[i] = n - n_sub - stride + 1
                    break
        else:
            rfind_sub[i] = (o <= n and n + 1) - 1
        stride += size_chr
        stride_sub += size_sub
    return rfind_sub


@register_jitable(**JIT_OPTIONS)
def index(chr_array, len_chr, size_chr,
          sub_array, len_sub, size_sub,
          start, end):
    """Native Implementation of np.char.index"""
    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        raise ValueError('substring not found')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    index_sub = -np.ones(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_sub = (len_sub > 1 and size_sub) or 0
    stride = stride_sub = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride
            n += stride
            while o + n_sub <= n:
                for p in range(n_sub):
                    if chr_array[o + p] != sub_array[stride_sub + p]:
                        o += 1
                        break
                else:
                    index_sub[i] = o - stride
                    break
            else:
                raise ValueError('substring not found')
        else:
            if o > n:
                raise ValueError('substring not found')
            index_sub[i] = o
        stride += size_chr
        stride_sub += size_sub
    return index_sub


@register_jitable(**JIT_OPTIONS)
def rindex(chr_array, len_chr, size_chr,
           sub_array, len_sub, size_sub,
           start, end):
    """Native Implementation of np.char.rindex"""
    start, end = _init_sub_indices(start, end, size_chr)
    if start > size_chr or start > end + size_chr:
        raise ValueError('substring not found')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    sub_lens = str_len(sub_array, len_sub, size_sub)

    len_cast = max(len_chr, len_sub)
    rfind_sub = -np.ones(len_cast, 'int64')

    size_chr = (len_chr > 1 and size_chr) or 0
    size_sub = (len_sub > 1 and size_sub) or 0
    stride = stride_sub = 0
    for i in range(len_cast):
        n_chr, n_sub, o, n = _get_sub_indices(chr_lens, len_chr,
                                              sub_lens, len_sub,
                                              start, end, i)
        if n_sub:
            o += stride - 1
            n += stride - 1
            r = stride_sub + n_sub - 1
            while n - n_sub >= o:
                for p in range(n_sub):
                    if chr_array[n - p] != sub_array[r - p]:
                        n -= 1
                        break
                else:
                    rfind_sub[i] = n - n_sub - stride + 1
                    break
            else:
                raise ValueError('substring not found')
        else:
            if o > n:
                raise ValueError('substring not found')
            rfind_sub[i] = n
        stride += size_chr
        stride_sub += size_sub
    return rfind_sub


@register_jitable(**JIT_OPTIONS)
def str_len(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.str_len"""
    if not size_chr:
        return np.zeros(len_chr, 'int64')

    str_length = np.empty(len_chr, 'int64')
    j = 0
    for i in range(0, chr_array.size, size_chr):
        str_length[j] = _record_len(chr_array, i, size_chr)
        j += 1
    return str_length


@register_jitable(**JIT_OPTIONS)
def scalar_bytes_len(value):
    """Length of a fixed-width bytes scalar, excluding null padding."""
    for i in range(len(value) - 1, -1, -1):
        if charseq_get_code(value, i):
            return i + 1
    return 0


@register_jitable(**JIT_OPTIONS)
def scalar_strings_len(value):
    """Length of a fixed-width Unicode scalar, excluding null padding."""
    for i in range(len(value) - 1, -1, -1):
        if unicode_charseq_get_code(value, i):
            return i + 1
    return 0


@register_jitable(**JIT_OPTIONS)
def _scalar_string_ord(value, index):
    return unicode_charseq_get_code(value, index)


@register_jitable(**JIT_OPTIONS)
def _ascii_alpha(chr_ord):
    return 65 <= chr_ord <= 90 or 97 <= chr_ord <= 122


@register_jitable(**JIT_OPTIONS)
def _ascii_digit(chr_ord):
    return 48 <= chr_ord <= 57


@register_jitable(**JIT_OPTIONS)
def _ascii_lower(chr_ord):
    return 97 <= chr_ord <= 122


@register_jitable(**JIT_OPTIONS)
def _ascii_upper(chr_ord):
    return 65 <= chr_ord <= 90


@register_jitable(**JIT_OPTIONS)
def _ascii_space(chr_ord):
    return 9 <= chr_ord <= 13 or chr_ord == 32


@register_jitable(**JIT_OPTIONS)
def _unicode_ascii_space(chr_ord):
    return 9 <= chr_ord <= 13 or 28 <= chr_ord <= 32


@register_jitable(**JIT_OPTIONS)
def _isalpha_ord(chr_ord, as_bytes):
    if as_bytes or chr_ord < 128:
        return _ascii_alpha(chr_ord)
    return bool(_PyUnicode_IsAlpha(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _isalnum_ord(chr_ord, as_bytes):
    if as_bytes or chr_ord < 128:
        return _ascii_alpha(chr_ord) or _ascii_digit(chr_ord)
    return bool(_PyUnicode_IsAlpha(chr_ord)) \
        or bool(_PyUnicode_IsNumeric(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _isdecimal_ord(chr_ord):
    if chr_ord < 128:
        return _ascii_digit(chr_ord)
    return bool(_PyUnicode_IsDecimalDigit(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _isdigit_ord(chr_ord, as_bytes):
    if as_bytes or chr_ord < 128:
        return _ascii_digit(chr_ord)
    return bool(_PyUnicode_IsDigit(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _isnumeric_ord(chr_ord):
    if chr_ord < 128:
        return _ascii_digit(chr_ord)
    return bool(_PyUnicode_IsNumeric(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _isspace_ord(chr_ord, as_bytes):
    if as_bytes:
        return _ascii_space(chr_ord)
    if chr_ord < 128:
        return _unicode_ascii_space(chr_ord)
    return bool(_PyUnicode_IsSpace(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _islower_ord(chr_ord, as_bytes):
    if as_bytes or chr_ord < 128:
        return _ascii_lower(chr_ord)
    return bool(_PyUnicode_IsLowercase(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _isupper_ord(chr_ord, as_bytes):
    if as_bytes or chr_ord < 128:
        return _ascii_upper(chr_ord)
    return bool(_PyUnicode_IsUppercase(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _istitle_only_ord(chr_ord, as_bytes):
    if as_bytes or chr_ord < 128:
        return False
    return bool(_PyUnicode_IsTitlecase(chr_ord))


@register_jitable(**JIT_OPTIONS)
def _is_simple_property_ord(chr_ord, as_bytes, kind):
    if kind == 0:
        return _isalpha_ord(chr_ord, as_bytes)
    if kind == 1:
        return _isalnum_ord(chr_ord, as_bytes)
    if kind == 2:
        return _isdecimal_ord(chr_ord)
    if kind == 3:
        return _isdigit_ord(chr_ord, as_bytes)
    if kind == 4:
        return _isnumeric_ord(chr_ord)
    return _isspace_ord(chr_ord, as_bytes)


@register_jitable(**JIT_OPTIONS)
def _simple_property(chr_array, len_chr, size_chr, as_bytes, kind):
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    result = np.empty(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        seen = False
        valid = True
        pending_null = False
        for c in range(size_chr):
            chr_ord = chr_array[stride + c]
            if chr_ord == 0:
                pending_null = True
            else:
                if pending_null \
                        or not _is_simple_property_ord(chr_ord,
                                                       as_bytes, kind):
                    valid = False
                    break
                seen = True
        result[i] = valid and seen
        stride += size_chr
    return result


@register_jitable(**JIT_OPTIONS)
def _scalar_bytes_simple_property(value, kind):
    seen = False
    pending_null = False
    for i in range(len(value)):
        chr_ord = charseq_get_code(value, i)
        if chr_ord == 0:
            pending_null = True
        else:
            if pending_null or not _is_simple_property_ord(chr_ord,
                                                           True, kind):
                return False
            seen = True
    return seen


@register_jitable(**JIT_OPTIONS)
def _scalar_strings_simple_property(value, kind):
    seen = False
    pending_null = False
    for i in range(len(value)):
        chr_ord = _scalar_string_ord(value, i)
        if chr_ord == 0:
            pending_null = True
        else:
            if pending_null or not _is_simple_property_ord(chr_ord,
                                                           False, kind):
                return False
            seen = True
    return seen


@register_jitable(**JIT_OPTIONS)
def scalar_bytes_isalpha(value):
    return _scalar_bytes_simple_property(value, 0)


@register_jitable(**JIT_OPTIONS)
def scalar_strings_isalpha(value):
    return _scalar_strings_simple_property(value, 0)


@register_jitable(**JIT_OPTIONS)
def scalar_bytes_isalnum(value):
    return _scalar_bytes_simple_property(value, 1)


@register_jitable(**JIT_OPTIONS)
def scalar_strings_isalnum(value):
    return _scalar_strings_simple_property(value, 1)


@register_jitable(**JIT_OPTIONS)
def scalar_strings_isdecimal(value):
    return _scalar_strings_simple_property(value, 2)


@register_jitable(**JIT_OPTIONS)
def scalar_bytes_isdigit(value):
    return _scalar_bytes_simple_property(value, 3)


@register_jitable(**JIT_OPTIONS)
def scalar_strings_isdigit(value):
    return _scalar_strings_simple_property(value, 3)


@register_jitable(**JIT_OPTIONS)
def scalar_strings_isnumeric(value):
    return _scalar_strings_simple_property(value, 4)


@register_jitable(**JIT_OPTIONS)
def scalar_bytes_isspace(value):
    return _scalar_bytes_simple_property(value, 5)


@register_jitable(**JIT_OPTIONS)
def scalar_strings_isspace(value):
    return _scalar_strings_simple_property(value, 5)


@register_jitable(**JIT_OPTIONS)
def scalar_bytes_istitle(value):
    size_chr = scalar_bytes_len(value)
    is_title = False
    cased_state = False
    for i in range(size_chr):
        chr_ord = charseq_get_code(value, i)
        is_lower = _ascii_lower(chr_ord)
        is_start = _ascii_upper(chr_ord)
        if cased_state:
            if is_start:
                return False
            cased_state = is_lower
        else:
            if is_lower:
                return False
            cased_state = is_start
            is_title |= cased_state
    return is_title


@register_jitable(**JIT_OPTIONS)
def scalar_strings_istitle(value):
    size_chr = scalar_strings_len(value)
    is_title = False
    cased_state = False
    for i in range(size_chr):
        chr_ord = _scalar_string_ord(value, i)
        is_lower = _islower_ord(chr_ord, False)
        is_start = _isupper_ord(chr_ord, False) \
            or _istitle_only_ord(chr_ord, False)
        if cased_state:
            if is_start:
                return False
            cased_state = is_lower
        else:
            if is_lower:
                return False
            cased_state = is_start
            is_title |= cased_state
    return is_title


@register_jitable(**JIT_OPTIONS)
def scalar_bytes_isupper(value):
    size_chr = scalar_bytes_len(value)
    is_upper = False
    for i in range(size_chr):
        chr_ord = charseq_get_code(value, i)
        if _ascii_lower(chr_ord):
            return False
        is_upper |= _ascii_upper(chr_ord)
    return is_upper


@register_jitable(**JIT_OPTIONS)
def scalar_strings_isupper(value):
    size_chr = scalar_strings_len(value)
    is_upper = False
    for i in range(size_chr):
        chr_ord = _scalar_string_ord(value, i)
        if _islower_ord(chr_ord, False) \
                or _istitle_only_ord(chr_ord, False):
            return False
        is_upper |= _isupper_ord(chr_ord, False)
    return is_upper


@register_jitable(**JIT_OPTIONS)
def scalar_bytes_islower(value):
    size_chr = scalar_bytes_len(value)
    is_lower = False
    for i in range(size_chr):
        chr_ord = charseq_get_code(value, i)
        if _ascii_upper(chr_ord):
            return False
        is_lower |= _ascii_lower(chr_ord)
    return is_lower


@register_jitable(**JIT_OPTIONS)
def scalar_strings_islower(value):
    size_chr = scalar_strings_len(value)
    is_lower = False
    for i in range(size_chr):
        chr_ord = _scalar_string_ord(value, i)
        if _isupper_ord(chr_ord, False) \
                or _istitle_only_ord(chr_ord, False):
            return False
        is_lower |= _islower_ord(chr_ord, False)
    return is_lower


@register_jitable(**JIT_OPTIONS)
def isalpha(chr_array, len_chr, size_chr, as_bytes):
    """Native Implementation of np.char.isalpha"""
    return _simple_property(chr_array, len_chr, size_chr, as_bytes, 0)


@register_jitable(**JIT_OPTIONS)
def isalnum(chr_array, len_chr, size_chr, as_bytes):
    """Native Implementation of np.char.isalnum"""
    return _simple_property(chr_array, len_chr, size_chr, as_bytes, 1)


@register_jitable(**JIT_OPTIONS)
def isdecimal(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isdecimal"""
    return _simple_property(chr_array, len_chr, size_chr, False, 2)


@register_jitable(**JIT_OPTIONS)
def isdigit(chr_array, len_chr, size_chr, as_bytes):
    """Native Implementation of np.char.isdigit"""
    return _simple_property(chr_array, len_chr, size_chr, as_bytes, 3)


@register_jitable(**JIT_OPTIONS)
def isnumeric(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isnumeric"""
    return _simple_property(chr_array, len_chr, size_chr, False, 4)


@register_jitable(**JIT_OPTIONS)
def isspace(chr_array, len_chr, size_chr, as_bytes):
    """Native Implementation of np.char.isspace"""
    return _simple_property(chr_array, len_chr, size_chr, as_bytes, 5)


@register_jitable(**JIT_OPTIONS)
def istitle(chr_array, len_chr, size_chr, as_bytes):
    """Native Implementation of np.char.istitle"""
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_title = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        cased_state = False
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            is_lower = _islower_ord(chr_ord, as_bytes)
            is_start = _isupper_ord(chr_ord, as_bytes) \
                or _istitle_only_ord(chr_ord, as_bytes)
            if cased_state:
                if is_start:
                    is_title[i] = False
                    break
                cased_state = is_lower
            else:
                if is_lower:
                    is_title[i] = False
                    break
                cased_state = is_start
                is_title[i] |= cased_state
        stride += size_chr
    return is_title


@register_jitable(**JIT_OPTIONS)
def isupper(chr_array, len_chr, size_chr, as_bytes):
    """Native Implementation of np.char.isupper"""
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_upper = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if _islower_ord(chr_ord, as_bytes) \
                    or _istitle_only_ord(chr_ord, as_bytes):
                is_upper[i] = False
                break
            is_upper[i] |= _isupper_ord(chr_ord, as_bytes)
        stride += size_chr
    return is_upper


@register_jitable(**JIT_OPTIONS)
def islower(chr_array, len_chr, size_chr, as_bytes):
    """Native Implementation of np.char.islower"""
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_lower = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if _isupper_ord(chr_ord, as_bytes) \
                    or _istitle_only_ord(chr_ord, as_bytes):
                is_lower[i] = False
                break
            is_lower[i] |= _islower_ord(chr_ord, as_bytes)
        stride += size_chr
    return is_lower
