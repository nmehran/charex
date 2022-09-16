"""
Numba overloads for numpy.character routines
Copyright (c) 2022, Nima Mehrani
"""

from charex.core import JIT_OPTIONS
from numba.extending import register_jitable
import numpy as np


@register_jitable(**JIT_OPTIONS)
def _set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv):
    if len_cmp > 1 and len_cmp != len_chr:
        msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
        raise ValueError(msg)
    if len_cmp == 1:
        cmp = chr_array.reshape(len_chr, size_chr).copy()
        size_stride = min(size_chr, size_cmp)
        cmp[:, :size_stride] -= cmp_array[:size_stride]
        if inv:
            return -cmp.ravel(), size_chr
        return cmp.ravel(), size_chr
    if size_chr < size_cmp:
        cmp = cmp_array.reshape(len_chr, size_cmp).copy()
        cmp[:, :size_chr] -= chr_array.reshape(len_chr, size_chr)
        if inv:
            return cmp.ravel(), size_cmp
        return -cmp.ravel(), size_cmp
    if size_chr > size_cmp:
        cmp = chr_array.reshape(len_chr, size_chr).copy()
        cmp[:, :size_cmp] -= cmp_array.reshape(len_chr, size_cmp)
        if inv:
            return -cmp.ravel(), size_cmp
        return cmp.ravel(), size_cmp
    if inv:
        return cmp_array - chr_array, size_cmp
    return chr_array - cmp_array, size_cmp


@register_jitable(**JIT_OPTIONS)
def _compare_any(x1: np.ndarray, x2: np.ndarray) -> bool:
    for i in range(x1.size):
        if x1[i] != x2[i]:
            return True
    return False


@register_jitable(**JIT_OPTIONS)
def equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp):
    """Native Implementation of np.char.equal"""
    ix = 0
    if len_cmp == 1:
        if size_chr < size_cmp:
            return np.zeros(len_chr, 'bool')
        elif size_chr > size_cmp:
            equal_to = np.empty(len_chr, 'bool')
            for i in range(len_chr):
                equal_to[i] = (chr_array[ix + size_cmp] == 0
                               and not _compare_any(cmp_array, chr_array[ix:ix + size_cmp]))
                ix += size_chr
        else:
            if len_chr > 1:
                return (chr_array.reshape(len_chr, size_chr) != cmp_array).sum(axis=1) == 0
            return ~np.array([_compare_any(chr_array, cmp_array)], 'bool')
    elif len_chr == len_cmp:
        if size_chr == size_cmp:
            if size_chr == 1:
                return chr_array == cmp_array
            return (chr_array != cmp_array).reshape(len_chr, size_chr).sum(axis=1) == 0
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
        msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
        raise ValueError(msg)
    return equal_to


@register_jitable(**JIT_OPTIONS)
def not_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp):
    """Native Implementation of np.char.not_equal"""
    ix = 0
    if len_cmp == 1:
        if size_chr < size_cmp:
            return np.ones(len_chr, 'bool')
        elif size_chr > size_cmp:
            not_equal_to = np.empty(len_chr, 'bool')
            for i in range(len_chr):
                not_equal_to[i] = (chr_array[ix + size_cmp] != 0
                                   or _compare_any(cmp_array, chr_array[ix:ix + size_cmp]))
                ix += size_chr
        else:
            if len_chr > 1:
                return (chr_array.reshape(len_chr, size_chr) != cmp_array).sum(axis=1) != 0
            return np.array([_compare_any(chr_array, cmp_array)], 'bool')
    elif len_chr == len_cmp:
        if size_chr == size_cmp:
            if size_chr == 1:
                return chr_array != cmp_array
            return (chr_array != cmp_array).reshape(len_chr, size_chr).sum(axis=1) != 0
        iy = 0
        not_equal_to = np.empty(len_chr, 'bool')
        if size_chr < size_cmp:
            for i in range(len_chr):
                not_equal_to[i] = (cmp_array[iy + size_chr] != 0
                                   or _compare_any(chr_array[ix:ix + size_chr], cmp_array[iy:iy + size_chr]))
                ix += size_chr
                iy += size_cmp
        elif size_chr > size_cmp:
            for i in range(len_chr):
                not_equal_to[i] = (chr_array[ix + size_cmp] != 0
                                   or _compare_any(chr_array[ix:ix + size_cmp], cmp_array[iy:iy + size_cmp]))
                ix += size_chr
                iy += size_cmp
    else:
        msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
        raise ValueError(msg)
    return not_equal_to


@register_jitable(**JIT_OPTIONS)
def greater(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
    """Native Implementation of np.char.greater"""
    cmp, size_stride = _set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
    greater_than = np.zeros(len_chr, 'bool')
    size_stride = max(size_chr, size_stride)
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


@register_jitable(**JIT_OPTIONS)
def greater_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
    """Native Implementation of np.char.greater_equal"""
    cmp, size_stride = _set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
    greater_equal_than = np.ones(len_chr, 'bool')
    size_stride = max(size_chr, size_stride)
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


@register_jitable(**JIT_OPTIONS)
def less(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
    """Native Implementation of np.char.less"""
    cmp, size_stride = _set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
    less_than = np.zeros(len_chr, 'bool')
    size_stride = max(size_chr, size_stride)
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


@register_jitable(**JIT_OPTIONS)
def less_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv=False):
    """Native Implementation of np.char.less_equal"""
    cmp, size_stride = _set_comparison(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
    less_equal_than = np.ones(len_chr, 'bool')
    size_stride = max(size_chr, size_stride)
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


@register_jitable(**JIT_OPTIONS)
def compare_chararrays(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv, cmp):
    """Native Implementation of np.char.compare_chararrays"""
    # The argument cmp can be passed as bytes or string, hence the ordinal map.
    # {“<”, “<=”, “==”, “>=”, “>”, “!=”} | { (60,) (60, 61), (61, 61), (62, 61), (62,), (33, 61) }
    if len(cmp) == 1:
        cmp_ord = ord(cmp)
        if cmp_ord == 60:
            return less(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        elif cmp_ord == 62:
            return greater(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
    elif len(cmp) == 2 and ord(cmp[1]) == 61:
        cmp_ord = ord(cmp[0])
        if cmp_ord == 60:
            return less_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        elif cmp_ord == 61:
            return equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp)
        elif cmp_ord == 62:
            return greater_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp, inv)
        elif cmp_ord == 33:
            return not_equal(chr_array, len_chr, size_chr, cmp_array, len_cmp, size_cmp)
    raise ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")
