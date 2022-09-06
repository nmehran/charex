"""
Numba overloads for numpy.character routines
Copyright (c) 2022, Nima Mehrani
"""

from charex.core import JIT_OPTIONS, OPTIONS
from charex.core._string_intrinsics import register_bytes, register_strings
from numba.extending import overload, register_jitable
from numba.core import types
import numpy as np


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
                            and (np.frombuffer(x1, dtype='int8') - np.frombuffer(x2, dtype='int8')).sum() == 0,
                            dtype='bool')
        return impl

    @register_jitable(**JIT_OPTIONS)
    def equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp):
        ix = 0
        if len_cmp == 1:
            if size_chr < size_cmp:
                return np.zeros(len_chr, dtype='bool')
            elif size_chr > size_cmp:
                if size_chr < 30:
                    cmp_stride = np.zeros(size_chr, dtype='int8')
                    cmp_stride[:size_cmp] = cmp_array
                    return (chr_array.reshape(len_chr, size_chr) - cmp_stride).sum(axis=1) == 0
                equal_to = np.empty(len_chr, dtype='bool')
                for i in range(len_chr):
                    equal_to[i] = chr_array[ix + size_cmp] == 0 and (cmp_array - chr_array[ix:ix + size_cmp]).sum() == 0
                    ix += size_chr
            else:
                return (chr_array.reshape(len_chr, size_chr) - cmp_array).sum(axis=1) == 0
        elif len_chr == len_cmp:
            if size_chr == size_cmp:
                if size_chr == 1:
                    return chr_array == cmp_array
                return (chr_array - cmp_array).reshape(-1, size_chr).sum(axis=1) == 0
            iy = 0
            equal_to = np.empty(len_chr, dtype='bool')
            if size_chr < size_cmp:
                for i in range(len_chr):
                    equal_to[i] = (cmp_array[iy + size_chr] == 0
                                   and (chr_array[ix:ix + size_chr] - cmp_array[iy:iy + size_chr]).sum() == 0)
                    ix += size_chr
                    iy += size_cmp
            elif size_chr > size_cmp:
                for i in range(len_chr):
                    equal_to[i] = (chr_array[ix + size_cmp] == 0
                                   and (chr_array[ix:ix + size_cmp] - cmp_array[iy:iy + size_cmp]).sum() == 0)
                    ix += size_chr
                    iy += size_cmp
        else:
            msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
            raise ValueError(msg)
        return equal_to

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    x1_type = x1.dtype if isinstance(x1, types.Array) else x1
    x2_type = x2.dtype if isinstance(x2, types.Array) else x2

    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_type = register_bytes
        cmp_type = bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_type = register_strings
        cmp_type = str
    else:
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
                            or (np.frombuffer(x1, dtype='int8') - np.frombuffer(x2, dtype='int8')).sum() != 0,
                            dtype='bool')
        return impl

    @register_jitable(**JIT_OPTIONS)
    def not_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp):
        ix = 0
        if len_cmp == 1:
            if size_chr < size_cmp:
                return np.ones(len_chr, dtype='bool')
            elif size_chr > size_cmp:
                if size_chr < 30:
                    cmp_stride = np.zeros(size_chr, dtype='int8')
                    cmp_stride[:size_cmp] = cmp_array
                    return (chr_array.reshape(len_chr, size_chr) - cmp_stride).sum(axis=1) != 0
                not_equal_to = np.empty(len_chr, dtype='bool')
                for i in range(len_chr):
                    not_equal_to[i] = chr_array[ix + size_cmp] != 0 or (cmp_array - chr_array[ix:ix + size_cmp]).sum() != 0
                    ix += size_chr
            else:
                return (chr_array.reshape(len_chr, size_chr) - cmp_array).sum(axis=1) != 0
        elif len_chr == len_cmp:
            if size_chr == size_cmp:
                if size_chr == 1:
                    return chr_array != cmp_array
                return (chr_array - cmp_array).reshape(-1, size_chr).sum(axis=1) != 0
            iy = 0
            not_equal_to = np.empty(len_chr, dtype='bool')
            if size_chr < size_cmp:
                for i in range(len_chr):
                    not_equal_to[i] = (cmp_array[iy + size_chr] != 0
                                       or (chr_array[ix:ix + size_chr] - cmp_array[iy:iy + size_chr]).sum() != 0)
                    ix += size_chr
                    iy += size_cmp
            elif size_chr > size_cmp:
                for i in range(len_chr):
                    not_equal_to[i] = (chr_array[ix + size_cmp] != 0
                                       or (chr_array[ix:ix + size_cmp] - cmp_array[iy:iy + size_cmp]).sum() != 0)
                    ix += size_chr
                    iy += size_cmp
        else:
            msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
            raise ValueError(msg)
        return not_equal_to

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    x1_type = x1.dtype if isinstance(x1, types.Array) else x1
    x2_type = x2.dtype if isinstance(x2, types.Array) else x2

    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_type = register_bytes
        cmp_type = bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_type = register_strings
        cmp_type = str
    else:
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
        if len_cmp > 1 and len_cmp != len_chr:
            msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
            raise ValueError(msg)

        if len_cmp == 1 and len_chr > 1:
            cmp_stride = np.zeros(size_chr, dtype='int8')
            cmp_stride[:size_cmp] = cmp_array
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

        greater_than = np.zeros(len_chr, dtype='bool')
        size_chr = min(size_chr, size_cmp)
        stride = 0
        for i in range(len_chr):
            for j in range(size_chr):
                cmp_ord = cmp[stride + j]
                if cmp_ord == 0:
                    continue
                if cmp_ord < 0:
                    break
                greater_than[i] = 1
                break
            stride += size_chr
        return greater_than

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    x1_type = x1.dtype if isinstance(x1, types.Array) else x1
    x2_type = x2.dtype if isinstance(x2, types.Array) else x2

    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_type = register_bytes
        cmp_type = bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_type = register_strings
        cmp_type = str
    else:
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
        if len_cmp > 1 and len_cmp != len_chr:
            msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
            raise ValueError(msg)

        if len_cmp == 1 and len_chr > 1:
            cmp_stride = np.zeros(size_chr, dtype='int8')
            cmp_stride[:size_cmp] = cmp_array
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

        greater_equal_than = np.ones(len_chr, dtype='bool')
        size_chr = min(size_chr, size_cmp)
        stride = 0
        for i in range(len_chr):
            for j in range(size_chr):
                cmp_ord = cmp[stride + j]
                if cmp_ord > 0:
                    break
                if cmp_ord < 0:
                    greater_equal_than[i] = 0
                    break
            stride += size_chr
        return greater_equal_than

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    x1_type = x1.dtype if isinstance(x1, types.Array) else x1
    x2_type = x2.dtype if isinstance(x2, types.Array) else x2

    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_type = register_bytes
        cmp_type = bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_type = register_strings
        cmp_type = str
    else:
        def impl(x1, x2):
            raise NotImplementedError('NotImplemented')
        return impl

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return greater_equal(*register_type(x2), *register_type(x1), True)
        return greater_equal(*register_type(x1), *register_type(x2))

    return impl