"""
Numba overloads for numpy.character routines
"""

from charex.core import JIT_OPTIONS, OPTIONS
from charex.core._string_intrinsics import register_bytes, register_strings
from charex.numpy.overloads.definitions import greater_equal, greater, equal, compare_chararrays
from numba.extending import overload, register_jitable
from numba.core import types
import numpy as np


@register_jitable(**JIT_OPTIONS)
def get_register_type(x1, x2):
    accepted_types = (types.Array, types.Bytes, types.UnicodeType)
    if not isinstance(x1, accepted_types) or not isinstance(x2, accepted_types):
        raise TypeError('comparison of non-string arrays')

    x1_type = x1.dtype if isinstance(x1, types.Array) else x1
    x2_type = x2.dtype if isinstance(x2, types.Array) else x2

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

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


@overload(np.char.equal, **OPTIONS)
def ov_char_equal(x1, x2):
    """Native Overload of np.char.equal"""
    register_type, cmp_type = get_register_type(x1, x2)

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return equal(*register_type(x2), *register_type(x1))
        return equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.not_equal, **OPTIONS)
def ov_char_not_equal(x1, x2):
    """Native Overload of np.char.not_equal"""
    register_type, cmp_type = get_register_type(x1, x2)

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return ~equal(*register_type(x2), *register_type(x1))
        return ~equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.greater_equal, **OPTIONS)
def ov_char_greater_equal(x1, x2):
    """Native Overload of np.char.greater_equal"""
    register_type, cmp_type = get_register_type(x1, x2)

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return greater_equal(*register_type(x2), *register_type(x1), True)
        return greater_equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.greater, **OPTIONS)
def ov_char_greater(x1, x2):
    """Native Overload of np.char.greater"""
    register_type, cmp_type = get_register_type(x1, x2)

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return greater(*register_type(x2), *register_type(x1), True)
        return greater(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.less, **OPTIONS)
def ov_char_less(x1, x2):
    """Native Overload of np.char.less"""
    register_type, cmp_type = get_register_type(x1, x2)

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return ~greater_equal(*register_type(x2), *register_type(x1), True)
        return ~greater_equal(*register_type(x1), *register_type(x2))
    return impl


@overload(np.char.less_equal, **OPTIONS)
def ov_char_less_equal(x1, x2):
    """Native Overload of np.char.less_equal"""
    register_type, cmp_type = get_register_type(x1, x2)

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return ~greater(*register_type(x2), *register_type(x1), True)
        return ~greater(*register_type(x1), *register_type(x2))
    return impl


@register_jitable(**JIT_OPTIONS)
@overload(np.char.compare_chararrays)
def ov_char_compare_chararrays(a1, a2, cmp, rstrip):
    """Native Overload of np.char.compare_chararrays"""
    if not isinstance(cmp, (types.Bytes, types.UnicodeType)):
        raise TypeError(f'a bytes-like object is required, not {cmp.name}')

    register_type, cmp_type = get_register_type(a1, a2)

    def impl(a1, a2, cmp, rstrip):
        if isinstance(a1, cmp_type) and not isinstance(a2, cmp_type):
            return compare_chararrays(*register_type(a2, rstrip), *register_type(a1, rstrip), True, cmp)
        return compare_chararrays(*register_type(a1, rstrip), *register_type(a2, rstrip), False, cmp)
    return impl
