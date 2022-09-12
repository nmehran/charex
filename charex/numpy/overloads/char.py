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
def register_pair(x1, x2):
    accepted_types = (types.Array, types.Bytes, types.UnicodeType)
    if not isinstance(x1, accepted_types) or not isinstance(x2, accepted_types):
        raise TypeError('comparison of non-string arrays')

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

    if not register_type:
        raise NotImplementedError('NotImplemented')
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
        if x1[i] != x2[i]:
            return True
    return False


@register_jitable(**JIT_OPTIONS)
def compare_bool(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    cmp = np.empty(x1.size, dtype='bool')
    for i in range(x1.size):
        cmp[i] = x1[i] != x2[i]
    return cmp


@overload(np.char.equal, **OPTIONS)
def ov_nb_char_equal(x1, x2):
    """Native Implementation of np.char.equal"""

    register_type, cmp_type = register_pair(x1, x2)

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

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return equal(*register_type(x2), *register_type(x1))
        return equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.not_equal, **OPTIONS)
def ov_nb_char_not_equal(x1, x2):
    """Native Implementation of np.char.not_equal"""

    register_type, cmp_type = register_pair(x1, x2)

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

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return not_equal(*register_type(x2), *register_type(x1))
        return not_equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.greater, **OPTIONS)
def ov_nb_char_greater(x1, x2):
    """Native Implementation of np.char.greater"""

    register_type, cmp_type = register_pair(x1, x2)

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

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return greater(*register_type(x2), *register_type(x1), True)
        return greater(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.greater_equal, **OPTIONS)
def ov_nb_char_greater_equal(x1, x2):
    """Native Implementation of np.char.greater_equal"""

    register_type, cmp_type = register_pair(x1, x2)

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

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return greater_equal(*register_type(x2), *register_type(x1), True)
        return greater_equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.less, **OPTIONS)
def ov_nb_char_less(x1, x2):
    """Native Implementation of np.char.less"""

    register_type, cmp_type = register_pair(x1, x2)

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

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return less(*register_type(x2), *register_type(x1), True)
        return less(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.less_equal, **OPTIONS)
def ov_nb_char_less_equal(x1, x2):
    """Native Implementation of np.char.less_equal"""

    register_type, cmp_type = register_pair(x1, x2)

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

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return less_equal(*register_type(x2), *register_type(x1), True)
        return less_equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.compare_chararrays)
def ov_nb_char_compare_chararrays(a1, a2, cmp, rstrip):
    """Native Implementation of np.char.compare_chararrays (rstrip is pending)"""
    if not isinstance(cmp, (types.Bytes, types.UnicodeType)):
        raise TypeError(f'a bytes-like object is required, not {cmp.name}')

    def compare_chararrays(a1, a2, cmp, rstrip):
        # The argument cmp can be passed as bytes or string.
        # {“<”, “<=”, “==”, “>=”, “>”, “!=”} || { (60,) (60, 61), (61, 61), (62, 61), (62,), (33, 61) }
        if len(cmp) == 1:
            cmp_ord = ord(cmp)
            if cmp_ord == 60:
                return np.char.less(a1, a2)
            elif cmp_ord == 62:
                return np.char.greater(a1, a2)
            else:
                raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")
        elif len(cmp) == 2 and ord(cmp[1]) == 61:
            cmp_ord = ord(cmp[0])
            if cmp_ord == 60:
                return np.char.less_equal(a1, a2)
            elif cmp_ord == 61:
                return np.char.equal(a1, a2)
            elif cmp_ord == 62:
                return np.char.greater_equal(a1, a2)
            elif cmp_ord == 33:
                return np.char.not_equal(a1, a2)
            else:
                raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")
        else:
            raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")
    return compare_chararrays
