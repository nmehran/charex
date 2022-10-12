"""
Numba overloads for numpy.character routines
Copyright (c) 2022, Nima Mehrani
"""

from charex.core import JIT_OPTIONS
from charex.core.string_intrinsics import bisect_null
from numba.extending import register_jitable
from numba import types
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Comparison Operators


@register_jitable(**JIT_OPTIONS)
def _cast_comparison(size_chr, len_chr, size_cmp, len_cmp):
    """Determines the offsets used to align the comparison to the target."""
    if len_cmp > 1 and len_cmp != len_chr:
        raise ValueError('shape mismatch: objects cannot be broadcast to a single shape.  '
                         'Mismatch is between arg 0 and arg 1.')
    size_margin = size_chr - size_cmp
    if len_cmp == 1:
        size_stride = min(size_chr, size_cmp + (size_margin < 0))
        size_cmp = 0
    else:
        size_stride = min(size_chr, size_cmp)
    return size_cmp, size_stride, size_margin


@register_jitable(**JIT_OPTIONS)
def _compare_any(x1: np.ndarray, x2: np.ndarray) -> bool:
    for i in range(x1.size):
        if x1[i] != x2[i]:
            return True
    return False


@register_jitable(**JIT_OPTIONS, locals={'cmp_ord': types.int32})
def greater_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
    """Native Implementation of np.char.greater_equal"""
    if 1 == size_chr == size_cmp:
        return cmp_array >= chr_array if inv else chr_array >= cmp_array

    size_cmp, size_stride, size_margin = _cast_comparison(size_chr, len_chr, size_cmp, len_cmp)
    greater_equal_than = np.empty(len_chr, 'bool')
    stride = stride_cmp = 0
    for i in range(len_chr):
        for j in range(size_stride):
            cmp_ord = chr_array[stride + j] - cmp_array[stride_cmp + j]
            if cmp_ord != 0:
                greater_equal_than[i] = ((inv and -cmp_ord) or cmp_ord) >= 0
                break
        else:
            greater_equal_than[i] = size_margin >= 0 or not cmp_array[stride_cmp + size_stride]
        stride += size_chr
        stride_cmp += size_cmp
    return greater_equal_than


@register_jitable(**JIT_OPTIONS, locals={'cmp_ord': types.int32})
def greater(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
    """Native Implementation of np.char.greater"""
    if 1 == size_chr == size_cmp:
        return cmp_array > chr_array if inv else chr_array > cmp_array

    size_cmp, size_stride, size_margin = _cast_comparison(size_chr, len_chr, size_cmp, len_cmp)
    greater_than = np.empty(len_chr, 'bool')
    stride = stride_cmp = 0
    for i in range(len_chr):
        for j in range(size_stride):
            cmp_ord = chr_array[stride + j] - cmp_array[stride_cmp + j]
            if cmp_ord != 0:
                greater_than[i] = ((inv and -cmp_ord) or cmp_ord) >= 0
                break
        else:
            greater_than[i] = size_margin > 0 and chr_array[stride + size_stride]
        stride += size_chr
        stride_cmp += size_cmp
    return greater_than


@register_jitable(**JIT_OPTIONS)
def equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp):
    """Native Implementation of np.char.equal"""
    # This implementation is more verbose, but has higher performance than its non-inline version.
    if 1 == size_chr == size_cmp:
        return chr_array == cmp_array

    ix = 0
    if len_cmp == 1:
        if size_chr < size_cmp:
            if cmp_array[size_chr]:
                return np.zeros(len_chr, 'bool')
            return np.array([not _compare_any(chr_array, cmp_array[:size_chr])], 'bool')
        equal_to = np.empty(len_chr, 'bool')
        if size_chr > size_cmp:
            for i in range(len_chr):
                equal_to[i] = (chr_array[ix + size_cmp] == 0
                               and not _compare_any(cmp_array, chr_array[ix:ix + size_cmp]))
                ix += size_chr
        else:
            for i in range(len_chr):
                equal_to[i] = not _compare_any(cmp_array, chr_array[ix:ix + size_cmp])
                ix += size_chr
    elif len_chr == len_cmp:
        iy = 0
        equal_to = np.empty(len_chr, 'bool')
        if size_chr < size_cmp:
            for i in range(len_chr):
                equal_to[i] = (cmp_array[iy + size_chr] == 0
                               and not _compare_any(chr_array[ix:ix + size_chr], cmp_array[iy:iy + size_chr]))
                ix += size_chr
                iy += size_cmp
        elif size_chr > size_cmp:
            for i in range(len_chr):
                equal_to[i] = (chr_array[ix + size_cmp] == 0
                               and not _compare_any(chr_array[ix:ix + size_cmp], cmp_array[iy:iy + size_cmp]))
                ix += size_chr
                iy += size_cmp
        else:
            for i in range(len_chr):
                equal_to[i] = not _compare_any(chr_array[ix:ix + size_chr], cmp_array[ix:ix + size_chr])
                ix += size_chr
    else:
        msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
        raise ValueError(msg)
    return equal_to


@register_jitable(**JIT_OPTIONS)
def compare_chararrays(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv, cmp):
    """Native Implementation of np.char.compare_chararrays"""
    # {“<”, “<=”, “==”, “>=”, “>”, “!=”}
    # { (60,) (60, 61), (61, 61), (62, 61), (62,), (33, 61) }
    # The argument cmp can be passed as bytes or string, hence ordinal mapping.
    if len(cmp) == 1:
        cmp_ord = ord(cmp)
        if cmp_ord == 60:
            return ~greater_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        elif cmp_ord == 62:
            return greater(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
    elif len(cmp) == 2 and ord(cmp[1]) == 61:
        cmp_ord = ord(cmp[0])
        if cmp_ord == 60:
            return ~greater(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        elif cmp_ord == 61:
            return equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp)
        elif cmp_ord == 62:
            return greater_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        elif cmp_ord == 33:
            return ~equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp)
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
def count(chr_array, len_chr, size_chr, sub_array, len_sub, size_sub, start, end):
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
def endswith(chr_array, len_chr, size_chr, sub_array, len_sub, size_sub, start, end):
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
def startswith(chr_array, len_chr, size_chr, sub_array, len_sub, size_sub, start, end):
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
def find(chr_array, len_chr, size_chr, sub_array, len_sub, size_sub, start, end):
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
def index(chr_array, len_chr, size_chr, sub_array, len_sub, size_sub, start, end):
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
def rfind(chr_array, len_chr, size_chr, sub_array, len_sub, size_sub, start, end):
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
def rindex(chr_array, len_chr, size_chr, sub_array, len_sub, size_sub, start, end):
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
    stride = size_chr - 1
    j = 0
    for i in range(0, chr_array.size, size_chr):
        str_length[j] = (chr_array[i + stride] and size_chr) or bisect_null(chr_array, i, i + stride) - i
        j += 1
    return str_length


@register_jitable(**JIT_OPTIONS)
def isalpha(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isalpha"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 125,000.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    alpha = {
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98,
        99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        121, 122, 170, 181, 186, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
        209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
        232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_alpha = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in alpha:
                is_alpha[i] = False
                break
            is_alpha[i] |= chr_ord in alpha
        stride += size_chr
    return is_alpha


@register_jitable(**JIT_OPTIONS)
def isalnum(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isalnum"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 127,000.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    alnum = {
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
        83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
        113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 170, 178, 179, 181, 185, 186, 188, 189, 190, 192, 193, 194,
        195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217,
        218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
        240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_alnum = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in alnum:
                is_alnum[i] = False
                break
            is_alnum[i] |= chr_ord in alnum
        stride += size_chr
    return is_alnum


@register_jitable(**JIT_OPTIONS)
def isdecimal(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isdecimal"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 600.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    decimal = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57}

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_decimal = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in decimal:
                is_decimal[i] = False
                break
            is_decimal[i] |= chr_ord in decimal
        stride += size_chr
    return is_decimal


@register_jitable(**JIT_OPTIONS)
def isdigit(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isdigit"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 700.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    digit = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 178, 179, 185}

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_digit = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in digit:
                is_digit[i] = False
                break
            is_digit[i] |= chr_ord in digit
        stride += size_chr
    return is_digit


@register_jitable(**JIT_OPTIONS)
def isnumeric(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isnumeric"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 1800.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    numeric = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 178, 179, 185, 188, 189, 190}

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_numeric = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in numeric:
                is_numeric[i] = False
                break
            is_numeric[i] |= chr_ord in numeric
        stride += size_chr
    return is_numeric


@register_jitable(**JIT_OPTIONS)
def isspace(chr_array, len_chr, size_chr, as_bytes):
    """Native Implementation of np.char.isspace"""
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    if as_bytes:
        space = {9, 10, 11, 12, 13, 32}
    else:
        space = {9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160,
                 5760, 8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8232, 8233, 8239, 8287, 12288}

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_space = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord not in space:
                is_space[i] = False
                break
            is_space[i] |= chr_ord in space
        stride += size_chr
    return is_space


@register_jitable(**JIT_OPTIONS)
def istitle(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.istitle"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 2300.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    lower = {
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 170, 181, 186, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
        239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254
    }

    upper = {
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 192,
        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
        216, 217, 218, 219, 220, 221, 222
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_title = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        cased_state = False
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if (cased_state and chr_ord in upper) or (not cased_state and chr_ord in lower):
                is_title[i] = False
                break
            cased_state = chr_ord in upper
            is_title[i] |= cased_state
            cased_state |= chr_ord in lower
        stride += size_chr
    return is_title


@register_jitable(**JIT_OPTIONS)
def isupper(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.isupper"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 2300.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    lower = {
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 170, 181, 186, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
        239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254
    }

    upper = {
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 192,
        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
        216, 217, 218, 219, 220, 221, 222
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_upper = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord in lower:
                is_upper[i] = False
                break
            is_upper[i] |= chr_ord in upper
        stride += size_chr
    return is_upper


@register_jitable(**JIT_OPTIONS)
def islower(chr_array, len_chr, size_chr):
    """Native Implementation of np.char.islower"""
    # Restricted to extended ASCII range(0, 255).
    # Complete ordinal set requires length of N > 2300.
    if not size_chr:
        return np.zeros(len_chr, 'bool')

    lower = {
        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 170, 181, 186, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
        239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254
    }

    upper = {
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 192,
        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
        216, 217, 218, 219, 220, 221, 222
    }

    chr_lens = str_len(chr_array, len_chr, size_chr)
    is_lower = np.zeros(len_chr, 'bool')
    stride = 0
    for i in range(len_chr):
        for c in range(chr_lens[i]):
            chr_ord = chr_array[stride + c]
            if chr_ord in upper:
                is_lower[i] = False
                break
            is_lower[i] |= chr_ord in lower
        stride += size_chr
    return is_lower
