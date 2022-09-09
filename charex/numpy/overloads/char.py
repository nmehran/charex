"""
Numba overloads for numpy.character routines
Copyright (c) 2022, Nima Mehrani
"""

from charex.core import JIT_OPTIONS, OPTIONS
from charex.core._string_intrinsics import register_bytes, register_strings
from numba.extending import overload, register_jitable
from numba.core import types
import numpy as np


@register_jitable(**JIT_OPTIONS)
def register_types(x1, x2):
    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    x1_type = x1.dtype if isinstance(x1, types.Array) else x1
    x2_type = x2.dtype if isinstance(x2, types.Array) else x2

    register_type = cmp_type = None
    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_type = register_bytes
        cmp_type = bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_type = register_strings
        cmp_type = str
    return register_type, cmp_type


@register_jitable(**JIT_OPTIONS)
def set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv):
    if len_cmp > 1 and len_cmp != len_chr:
        msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
        raise ValueError(msg)
    if len_cmp == 1 and len_chr > 1:
        cmp_stride = np.zeros(size_chr, dtype='int8')
        cmp_stride[:size_cmp] = cmp_array[:size_chr]
        cmp = (chr_array.reshape(len_chr, size_chr) - cmp_stride).ravel()
        size_cmp = size_chr
    elif size_chr < size_cmp:
        cmp = chr_array - cmp_array.reshape(len_chr, size_cmp)[:, :size_chr].ravel()
    elif size_chr > size_cmp:
        cmp = chr_array.reshape(len_chr, size_chr)[:, :size_cmp].ravel() - cmp_array
    else:
        cmp = chr_array - cmp_array
    if inv:
        cmp = -cmp
    return cmp, size_cmp


@register_jitable(**JIT_OPTIONS)
def compare_any(x1: np.ndarray, x2: np.ndarray) -> int:
    for i in range(x1.size):
        difference = x1[i] - x2[i]
        if difference:
            return difference
    return 0


@register_jitable(**JIT_OPTIONS)
def compare_bool(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    cmp = np.empty(x1.size, dtype='bool')
    for i in range(x1.size):
        cmp[i] = x1[i] - x2[i]
    return cmp


@overload(np.char.equal, **OPTIONS)
def ov_nb_char_equal(x1, x2):
    """Native Implementation of np.char.equal"""

    accepted_types = (types.Array, types.Bytes, types.UnicodeType)
    if not isinstance(x1, accepted_types) or not isinstance(x2, accepted_types):
        raise TypeError('comparison of non-string arrays')

    if isinstance(x1, types.UnicodeType) and isinstance(x2, types.UnicodeType):
        def impl(x1, x2):
            return np.array(len(x1) == len(x2) and x1 == x2, dtype='bool')
        return impl

    if isinstance(x1, types.Bytes) and isinstance(x2, types.Bytes):
        def impl(x1, x2):
            return np.array(len(x1) == len(x2)
                            and not np.any(np.frombuffer(x1, dtype='int8') - np.frombuffer(x2, dtype='int8')),
                            dtype='bool')
        return impl

    @register_jitable(**JIT_OPTIONS)
    def equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp):
        ix = 0
        if len_cmp == 1:
            if size_chr < size_cmp:
                return np.zeros(len_chr, dtype='bool')
            elif size_chr > size_cmp:
                equal_to = np.empty(len_chr, dtype='bool')
                for i in range(len_chr):
                    equal_to[i] = (chr_array[ix + size_cmp] == 0
                                   and not compare_any(cmp_array, chr_array[ix:ix + size_cmp]))
                    ix += size_chr
            else:
                return ~np.array([compare_any(chr_array, cmp_array)], dtype='bool')
        elif len_chr == len_cmp:
            if size_chr == size_cmp:
                if size_chr == 1:
                    return chr_array == cmp_array
                return compare_bool(chr_array, cmp_array).reshape(len_chr, size_chr).sum(axis=1) == 0
            iy = 0
            equal_to = np.empty(len_chr, dtype='bool')
            if size_chr < size_cmp:
                for i in range(len_chr):
                    equal_to[i] = (cmp_array[iy + size_chr] == 0
                                   and not compare_any(chr_array[ix:ix + size_chr], cmp_array[iy:iy + size_chr]))
                    ix += size_chr
                    iy += size_cmp
            elif size_chr > size_cmp:
                for i in range(len_chr):
                    equal_to[i] = (chr_array[ix + size_cmp] == 0
                                   and not compare_any(chr_array[ix:ix + size_cmp], cmp_array[iy:iy + size_cmp]))
                    ix += size_chr
                    iy += size_cmp
        else:
            msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
            raise ValueError(msg)
        return equal_to

    register_type, cmp_type = register_types(x1, x2)
    if not register_type:
        def impl(x1, x2):
            raise NotImplementedError('NotImplemented')
        return impl

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return equal(*register_type(x2), *register_type(x1))
        return equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.not_equal, **OPTIONS)
def ov_nb_char_not_equal(x1, x2):
    """Native Implementation of np.char.not_equal"""

    accepted_types = (types.Array, types.Bytes, types.UnicodeType)
    if not isinstance(x1, accepted_types) or not isinstance(x2, accepted_types):
        raise TypeError('comparison of non-string arrays')

    if isinstance(x1, types.UnicodeType) and isinstance(x2, types.UnicodeType):
        def impl(x1, x2):
            return np.array(len(x1) != len(x2) or x1 != x2, dtype='bool')
        return impl

    if isinstance(x1, types.Bytes) and isinstance(x2, types.Bytes):
        def impl(x1, x2):
            return np.array(len(x1) != len(x2)
                            or np.any(np.frombuffer(x1, dtype='int8') - np.frombuffer(x2, dtype='int8')),
                            dtype='bool')
        return impl

    @register_jitable(**JIT_OPTIONS)
    def not_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp):
        ix = 0
        if len_cmp == 1:
            if size_chr < size_cmp:
                return np.ones(len_chr, dtype='bool')
            elif size_chr > size_cmp:
                not_equal_to = np.empty(len_chr, dtype='bool')
                for i in range(len_chr):
                    not_equal_to[i] = (chr_array[ix + size_cmp] != 0
                                       or compare_any(cmp_array, chr_array[ix:ix + size_cmp]))
                    ix += size_chr
            else:
                return np.array([compare_any(chr_array, cmp_array)], dtype='bool')
        elif len_chr == len_cmp:
            if size_chr == size_cmp:
                if size_chr == 1:
                    return chr_array != cmp_array
                return compare_bool(chr_array, cmp_array).reshape(len_chr, size_chr).sum(axis=1) != 0
            iy = 0
            not_equal_to = np.empty(len_chr, dtype='bool')
            if size_chr < size_cmp:
                for i in range(len_chr):
                    not_equal_to[i] = (cmp_array[iy + size_chr] != 0
                                       or compare_any(chr_array[ix:ix + size_chr], cmp_array[iy:iy + size_chr]))
                    ix += size_chr
                    iy += size_cmp
            elif size_chr > size_cmp:
                for i in range(len_chr):
                    not_equal_to[i] = (chr_array[ix + size_cmp] != 0
                                       or compare_any(chr_array[ix:ix + size_cmp], cmp_array[iy:iy + size_cmp]))
                    ix += size_chr
                    iy += size_cmp
        else:
            msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
            raise ValueError(msg)
        return not_equal_to

    register_type, cmp_type = register_types(x1, x2)
    if not register_type:
        def impl(x1, x2):
            raise NotImplementedError('NotImplemented')
        return impl

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return not_equal(*register_type(x2), *register_type(x1))
        return not_equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.greater, **OPTIONS)
def ov_nb_char_greater(x1, x2):
    """Native Implementation of np.char.greater"""

    accepted_types = (types.Array, types.Bytes, types.UnicodeType)
    if not isinstance(x1, accepted_types) or not isinstance(x2, accepted_types):
        raise TypeError('comparison of non-string arrays')

    @register_jitable(**JIT_OPTIONS)
    def greater(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
        cmp, size_cmp = set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        greater_than = np.zeros(len_chr, dtype='bool')
        size_stride = min(size_chr, size_cmp)
        stride = 0
        for i in range(len_chr):
            for j in range(size_stride):
                cmp_ord = cmp[stride + j]
                if cmp_ord == 0:
                    continue
                if cmp_ord < 0:
                    break
                greater_than[i] = 1
                break
            stride += size_stride
        if len_chr == 1 and size_chr > size_cmp and not greater_than and np.flatnonzero(cmp == 0).size == size_stride:
            return ~greater_than
        return greater_than

    register_type, cmp_type = register_types(x1, x2)
    if not register_type:
        def impl(x1, x2):
            raise NotImplementedError('NotImplemented')
        return impl

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return greater(*register_type(x2), *register_type(x1), True)
        return greater(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.greater_equal, **OPTIONS)
def ov_nb_char_greater_equal(x1, x2):
    """Native Implementation of np.char.greater_equal"""

    accepted_types = (types.Array, types.Bytes, types.UnicodeType)
    if not isinstance(x1, accepted_types) or not isinstance(x2, accepted_types):
        raise TypeError('comparison of non-string arrays')

    @register_jitable(**JIT_OPTIONS)
    def greater_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
        cmp, size_cmp = set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        greater_equal_than = np.ones(len_chr, dtype='bool')
        size_stride = min(size_chr, size_cmp)
        stride = 0
        for i in range(len_chr):
            for j in range(size_stride):
                cmp_ord = cmp[stride + j]
                if cmp_ord > 0:
                    break
                if cmp_ord < 0:
                    greater_equal_than[i] = 0
                    break
            stride += size_stride
        if len_chr == 1 and size_chr < size_cmp and greater_equal_than and np.flatnonzero(cmp == 0).size == size_stride:
            return ~greater_equal_than
        return greater_equal_than

    register_type, cmp_type = register_types(x1, x2)
    if not register_type:
        def impl(x1, x2):
            raise NotImplementedError('NotImplemented')
        return impl

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return greater_equal(*register_type(x2), *register_type(x1), True)
        return greater_equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.less, **OPTIONS)
def ov_nb_char_less(x1, x2):
    """Native Implementation of np.char.less"""

    accepted_types = (types.Array, types.Bytes, types.UnicodeType)
    if not isinstance(x1, accepted_types) or not isinstance(x2, accepted_types):
        raise TypeError('comparison of non-string arrays')

    @register_jitable(**JIT_OPTIONS)
    def less(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
        cmp, size_cmp = set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        less_than = np.zeros(len_chr, dtype='bool')
        size_stride = min(size_chr, size_cmp)
        stride = 0
        for i in range(len_chr):
            for j in range(size_stride):
                cmp_ord = cmp[stride + j]
                if cmp_ord == 0:
                    continue
                if cmp_ord > 0:
                    break
                less_than[i] = 1
                break
            stride += size_stride
        if len_chr == 1 and size_chr < size_cmp and not less_than and np.flatnonzero(cmp == 0).size == size_stride:
            return ~less_than
        return less_than

    register_type, cmp_type = register_types(x1, x2)
    if not register_type:
        def impl(x1, x2):
            raise NotImplementedError('NotImplemented')
        return impl

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return less(*register_type(x2), *register_type(x1), True)
        return less(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.less_equal, **OPTIONS)
def ov_nb_char_less_equal(x1, x2):
    """Native Implementation of np.char.less_equal"""

    accepted_types = (types.Array, types.Bytes, types.UnicodeType)
    if not isinstance(x1, accepted_types) or not isinstance(x2, accepted_types):
        raise TypeError('comparison of non-string arrays')

    @register_jitable(**JIT_OPTIONS)
    def less_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
        cmp, size_cmp = set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        less_equal_than = np.ones(len_chr, dtype='bool')
        size_stride = min(size_chr, size_cmp)
        stride = 0
        for i in range(len_chr):
            for j in range(size_stride):
                cmp_ord = cmp[stride + j]
                if cmp_ord < 0:
                    break
                if cmp_ord > 0:
                    less_equal_than[i] = 0
                    break
            stride += size_stride
        if len_chr == 1 and size_chr > size_cmp and less_equal_than and np.flatnonzero(cmp == 0).size == size_stride:
            return ~less_equal_than
        return less_equal_than

    register_type, cmp_type = register_types(x1, x2)
    if not register_type:
        def impl(x1, x2):
            raise NotImplementedError('NotImplemented')
        return impl

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return less_equal(*register_type(x2), *register_type(x1), True)
        return less_equal(*register_type(x1), *register_type(x2))
    return impl


# @overload(np.char.compare_chararrays, **OPTIONS)
# def ov_nb_char_compare_chararrays(a, b, cmp_op, rstrip):
#     """Native Implementation of np.char.compare_chararrays (rstrip is pending)"""
#     if not isinstance(cmp_op, (types.Bytes, types.UnicodeType)):
#         raise TypeError(f'a bytes-like object is required, not {cmp_op.name}')
#
#     @register_jitable(**JIT_OPTIONS)
#     def compare_chararrays(a, b, cmp_op, rstrip):
#         # {“<”, “<=”, “==”, “>=”, “>”, “!=”} || { (60,) (60, 61), (61, 61), (62, 61), (62,), (33, 61) }
#         if len(cmp_op) == 1:
#             cmp_ord = ord(cmp_op)
#             if cmp_ord == 60:
#                 impl_func = np.char.less
#             elif cmp_ord == 62:
#                 impl_func = np.char.greater
#             else:
#                 raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")
#         elif len(cmp_op) == 2 and ord(cmp_op[1]) == 61:
#             cmp_ord = ord(cmp_op[0])
#             if cmp_ord == 60:
#                 impl_func = np.char.less_equal
#             elif cmp_ord == 61:
#                 impl_func = np.char.equal
#             elif cmp_ord == 62:
#                 impl_func = np.char.greater_equal
#             elif cmp_ord == 33:
#                 impl_func = np.char.not_equal
#             else:
#                 raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")
#         else:
#             raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")
#         return impl_func(a, b)
#     return compare_chararrays
