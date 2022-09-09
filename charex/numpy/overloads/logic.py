"""
Numba overloads for numpy logical routines
Copyright (c) 2022, Nima Mehrani
"""

from charex.core import JIT_OPTIONS, OPTIONS
from numba.extending import overload, register_jitable
from numba.core import types
from numba import objmode
import numpy as np


@overload(np.any, **OPTIONS)
def ov_nb_any(a, axis):
    """Native Implementation of np.any"""

    if not isinstance(a, types.Array) or not isinstance(axis, (types.Integer, types.NoneType)):
        raise TypeError('comparison of non-string arrays')

    @register_jitable(**JIT_OPTIONS)
    def any_1d(arr):
        for i in range(arr.size):
            if arr[i]:
                return True
        return False

    @register_jitable(**JIT_OPTIONS)
    def impl(a, axis):
        shape = a.shape
        item_size = a.itemsize
        with objmode(is_float_type='boolean'):
            is_float_type = a.dtype.num > 10
        item_offset = is_float_type and item_size - 1
        a = np.frombuffer(a, dtype='bool')[item_offset::item_size]
        if len(shape) == 1:
            if axis == 1:
                raise np.AxisError('axis 1 is out of bounds for array of dimension 1')
            # Cannot pass one-dimensional array with axis=0 in no_python mode, but compatible in CPython / Numpy.
            return np.array([any_1d(a)], dtype='bool')
        if axis is None:
            return np.array([any_1d(a)], dtype='bool')
        if axis == 1:
            any_ = np.empty(shape[0], dtype='bool')
            stride = 0
            size_stride = shape[1]
            for i in range(shape[0]):
                any_[i] = any_1d(a[stride:stride + size_stride])
                stride += size_stride
            return any_
        if axis == 0:
            size_stride = shape[1]
            any_ = np.zeros(size_stride, dtype='bool')
            stride = 0
            for i in range(shape[0]):
                any_ += a[stride:stride + size_stride]
                stride += size_stride
            return any_
    return impl
