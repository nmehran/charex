"""
Numba overloads for numpy.character routines
"""

from charex.core import OPTIONS
from charex.core.string_intrinsics import (
    register_array_bytes, register_scalar_bytes,
    register_array_strings, register_scalar_strings
)
from charex.numpy.overloads.definitions import greater_equal, greater, equal, compare_chararrays
from numba.core import types
from numba.extending import overload
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Comparison Operators


def _ensure_type(x):
    """Ensure argument is a character type with appropriate layout and shape."""
    ndim = -1
    if isinstance(x, types.Array):
        ndim = x.ndim
        if ndim > 1 or x.layout != 'C':
            msg = 'shape mismatch: objects cannot be broadcast to a single ' \
                  'shape.  Mismatch is between arg 0 and arg 1.'
            raise ValueError(msg)
        x = x.dtype
        if not isinstance(x, (types.CharSeq,
                              types.UnicodeCharSeq)) or not x.count:
            raise TypeError('comparison of non-string arrays')
    elif not isinstance(x, (types.Bytes, types.UnicodeType)):
        raise TypeError('comparison of non-string arrays')
    return x, ndim


def _get_register_type(x1, x2):
    """Determines the call function for the comparison pair, based on type."""
    (x1_type, x1_dim), (x2_type, x2_dim) = _ensure_type(x1), _ensure_type(x2)
    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_x1 = register_array_bytes if x1_dim >= 0 else register_scalar_bytes
        register_x2 = register_array_bytes if x2_dim >= 0 else register_scalar_bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_x1 = register_array_strings if x1_dim >= 0 else register_scalar_strings
        register_x2 = register_array_strings if x2_dim >= 0 else register_scalar_strings
    else:
        raise NotImplementedError('NotImplemented')
    return register_x1, register_x2, x1_dim, x2_dim


@overload(np.char.equal, **OPTIONS)
def ov_char_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return equal(*register_x2(x2), *register_x1(x1))
            return equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(equal(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.not_equal, **OPTIONS)
def ov_char_not_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return ~equal(*register_x2(x2), *register_x1(x1))
            return ~equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(~equal(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.greater_equal, **OPTIONS)
def ov_char_greater_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return greater_equal(*register_x2(x2), *register_x1(x1), True)
            return greater_equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(greater_equal(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.greater, **OPTIONS)
def ov_char_greater(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return greater(*register_x2(x2), *register_x1(x1), True)
            return greater(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(greater(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.less, **OPTIONS)
def ov_char_less(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return ~greater_equal(*register_x2(x2), *register_x1(x1), True)
            return ~greater_equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(~greater_equal(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.less_equal, **OPTIONS)
def ov_char_less_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _get_register_type(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return ~greater(*register_x2(x2), *register_x1(x1), True)
            return ~greater(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(~greater(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.compare_chararrays, **OPTIONS)
def ov_char_compare_chararrays(a1, a2, cmp, rstrip):
    if not isinstance(cmp, (types.Bytes, types.UnicodeType)):
        raise TypeError(f'a bytes-like object is required, not {cmp.name}')

    register_a1, register_a2, a1_dim, a2_dim = _get_register_type(a1, a2)

    if a1_dim > 0 or a2_dim > 0:
        def impl(a1, a2, cmp, rstrip):
            if a1_dim < 0 <= a2_dim:
                return compare_chararrays(*register_a2(a2, rstrip), *register_a1(a1, rstrip), True, cmp)
            return compare_chararrays(*register_a1(a1, rstrip), *register_a2(a2, rstrip), False, cmp)
    else:
        def impl(a1, a2, cmp, rstrip):
            return np.array(compare_chararrays(*register_a1(a1, rstrip), *register_a2(a2, rstrip), False, cmp)[0])
    return impl


# ----------------------------------------------------------------------------------------------------------------------
