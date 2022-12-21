"""
Numba overloads for numpy.character routines
"""

from charex.core import OPTIONS
from charex.core.string_intrinsics import (
    register_array_bytes, register_scalar_bytes,
    register_array_strings, register_scalar_strings
)
from charex.numpy.overloads.definitions import (
    greater_equal, greater, equal, compare_chararrays,
    count, endswith, startswith, find, str_len,
    isalpha, isalnum, isdecimal, isdigit, islower, isnumeric, isspace, istitle, isupper

)
from numba.core import types
from numba.extending import overload
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Comparison Operators


def _ensure_slice(start, end):
    """Ensure start and end slice argument is an integer type."""
    slice_types = (types.Integer, types.NoneType)
    if not (isinstance(start, slice_types) and isinstance(end, slice_types)):
        raise TypeError("slice indices must be integers or None or have an __index__ method")
    return 0, np.iinfo(np.int64).max


def _ensure_type(x, exception: Exception = None):
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
            ndim = None
    elif not isinstance(x, (types.Bytes, types.UnicodeType)):
        ndim = None
    if isinstance(exception, Exception) and ndim is None:
        raise exception
    return x, ndim


def _str_type(x, as_np=True):
    """Infer string-type of an objects Numba instance."""
    if isinstance(x, types.Array):
        if isinstance(x.dtype, types.CharSeq):
            return 'numpy.bytes_' if as_np else 'bytes'
        if isinstance(x.dtype, types.UnicodeCharSeq):
            return 'numpy.str_' if as_np else 'str'
        return f'like {x.dtype.name}'

    if isinstance(x, types.Bytes):
        return 'numpy.bytes_' if as_np else 'bytes'
    if isinstance(x, types.UnicodeType):
        return 'numpy.str_' if as_np else 'str'
    return f'like {x.name}'


def _register_pair(x1, x2, exception: (Exception, int) = None):
    """Determines the call function for the comparison pair, based on type."""
    e = exception or TypeError("comparison of non-string arrays")
    x1_type, x1_dim = _ensure_type(x1, e)
    x2_type, x2_dim = _ensure_type(x2, e)

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_x1 = register_array_bytes if x1_dim >= 0 \
            else register_scalar_bytes
        register_x2 = register_array_bytes if x2_dim >= 0 \
            else register_scalar_bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_x1 = register_array_strings if x1_dim >= 0 \
            else register_scalar_strings
        register_x2 = register_array_strings if x2_dim >= 0 \
            else register_scalar_strings
    else:
        if exception == 1:
            as_type = _str_type(x1, as_np=False)
            if as_type in ('str', 'bytes'):
                e = TypeError(f"must be {as_type}, not {_str_type(x2)}")
            else:
                e = TypeError("string operation on non-string array")
        else:
            e = NotImplementedError('NotImplemented')
        raise e
    return register_x1, register_x2, x1_dim, x2_dim


def _register_single(x1, exception: Exception = None):
    """Determines the call function for the input, based on type."""
    e = exception or TypeError("string operation on non-string array")
    x1_type, x1_dim = _ensure_type(x1, e)

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    if isinstance(x1_type, byte_types):
        register_x1 = register_array_bytes if x1_dim >= 0 \
            else register_scalar_bytes
        as_bytes = True
    elif isinstance(x1_type, str_types):
        register_x1 = register_array_strings if x1_dim >= 0 \
            else register_scalar_strings
        as_bytes = False
    else:
        raise e
    return register_x1, x1_dim, as_bytes


@overload(np.char.equal, **OPTIONS)
def ov_char_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return equal(*register_x2(x2), *register_x1(x1))
            return equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(equal(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.not_equal)
def ov_char_not_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return ~equal(*register_x2(x2), *register_x1(x1))
            return ~equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(~equal(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.greater_equal)
def ov_char_greater_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return greater_equal(*register_x2(x2), *register_x1(x1), True)
            return greater_equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(greater_equal(*register_x1(x1),
                                          *register_x2(x2))[0])
    return impl


@overload(np.char.greater, **OPTIONS)
def ov_char_greater(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

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
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return ~greater_equal(*register_x2(x2), *register_x1(x1), True)
            return ~greater_equal(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(~greater_equal(*register_x1(x1),
                                           *register_x2(x2))[0])
    return impl


@overload(np.char.less_equal)
def ov_char_less_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if x1_dim < 0 <= x2_dim:
                return ~greater(*register_x2(x2), *register_x1(x1), True)
            return ~greater(*register_x1(x1), *register_x2(x2))
    else:
        def impl(x1, x2):
            return np.array(~greater(*register_x1(x1), *register_x2(x2))[0])
    return impl


@overload(np.char.compare_chararrays)
def ov_char_compare_chararrays(a1, a2, cmp, rstrip):
    if not isinstance(cmp, (types.Bytes, types.UnicodeType)):
        raise TypeError(f'a bytes-like object is required, not {cmp.name}')

    register_a1, register_a2, a1_dim, a2_dim = _register_pair(a1, a2)

    if a1_dim > 0 or a2_dim > 0:
        def impl(a1, a2, cmp, rstrip):
            if a1_dim < 0 <= a2_dim:
                return compare_chararrays(*register_a2(a2, rstrip),
                                          *register_a1(a1, rstrip), True, cmp)
            return compare_chararrays(*register_a1(a1, rstrip),
                                      *register_a2(a2, rstrip), False, cmp)
    else:
        def impl(a1, a2, cmp, rstrip):
            return np.array(compare_chararrays(*register_a1(a1, rstrip),
                                               *register_a2(a2, rstrip),
                                               False, cmp)[0])
    return impl


# ----------------------------------------------------------------------------------------------------------------------
# String Information


@overload(np.char.count, **OPTIONS)
def ov_char_count(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return count(*register_a(a, False),
                         *register_sub(sub, False),
                         start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(count(*register_a(a, False),
                                  *register_sub(sub, False),
                                  start, end)[0])
    return impl


@overload(np.char.endswith, **OPTIONS)
def ov_char_endswith(a, suffix, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, suffix, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, suffix, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return endswith(*register_a(a, False),
                            *register_sub(suffix, False),
                            start, end)
    else:
        def impl(a, suffix, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(endswith(*register_a(a, False),
                                     *register_sub(suffix, False),
                                     start, end)[0])
    return impl


@overload(np.char.startswith, **OPTIONS)
def ov_char_startswith(a, prefix, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, prefix, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, prefix, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return startswith(*register_a(a, False),
                              *register_sub(prefix, False),
                              start, end)
    else:
        def impl(a, prefix, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(startswith(*register_a(a, False),
                                       *register_sub(prefix, False),
                                       start, end)[0])
    return impl


@overload(np.char.find, **OPTIONS)
def ov_char_find(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return find(*register_a(a, False),
                        *register_sub(sub, False),
                        start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(find(*register_a(a, False),
                                 *register_sub(sub, False),
                                 start, end)[0])
    return impl


@overload(np.char.rfind, **OPTIONS)
def ov_char_rfind(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return find(*register_a(a, False),
                        *register_sub(sub, False),
                        start, end, True)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(find(*register_a(a, False),
                                 *register_sub(sub, False),
                                 start, end, True)[0])
    return impl


@overload(np.char.index, **OPTIONS)
def ov_char_index(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return find(*register_a(a, False),
                        *register_sub(sub, False),
                        start, end, False, True)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(find(*register_a(a, False),
                                 *register_sub(sub, False),
                                 start, end, False, True)[0])
    return impl


@overload(np.char.rindex, **OPTIONS)
def ov_char_rindex(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return find(*register_a(a, False),
                        *register_sub(sub, False),
                        start, end, True, True)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return np.array(find(*register_a(a, False),
                                 *register_sub(sub, False),
                                 start, end, True, True)[0])
    return impl


@overload(np.char.str_len)
def ov_char_str_len(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return str_len(*register_a(a, False))
    else:
        def impl(a):
            return np.array(str_len(*register_a(a, False))[0])
    return impl


@overload(np.char.isalpha, **OPTIONS)
def ov_char_isalpha(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isalpha(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isalpha(*register_a(a, False))[0])
    return impl


@overload(np.char.isalnum, **OPTIONS)
def ov_char_isalnum(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isalnum(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isalnum(*register_a(a, False))[0])
    return impl


@overload(np.char.isspace, **OPTIONS)
def ov_char_isspace(a):
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isspace(*register_a(a, False), as_bytes)
    else:
        def impl(a):
            return np.array(isspace(*register_a(a, False), as_bytes)[0])
    return impl


@overload(np.char.isdecimal, **OPTIONS)
def ov_char_isdecimal(a):
    catch_incompatible = TypeError("isnumeric is only available for "
                                   "Unicode strings and arrays")
    register_a, a_dim, as_bytes = _register_single(a, catch_incompatible)
    if as_bytes:
        raise catch_incompatible

    if a_dim > 0:
        def impl(a):
            return isdecimal(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isdecimal(*register_a(a, False))[0])
    return impl


@overload(np.char.isdigit, **OPTIONS)
def ov_char_isdigit(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isdigit(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isdigit(*register_a(a, False))[0])
    return impl


@overload(np.char.isnumeric, **OPTIONS)
def ov_char_isnumeric(a):
    catch_incompatible = TypeError("isnumeric is only available for "
                                   "Unicode strings and arrays")
    register_a, a_dim, as_bytes = _register_single(a, catch_incompatible)
    if as_bytes:
        raise catch_incompatible

    if a_dim > 0:
        def impl(a):
            return isnumeric(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isnumeric(*register_a(a, False))[0])
    return impl


@overload(np.char.istitle, **OPTIONS)
def ov_char_istitle(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return istitle(*register_a(a, False))
    else:
        def impl(a):
            return np.array(istitle(*register_a(a, False))[0])
    return impl


@overload(np.char.isupper, **OPTIONS)
def ov_char_isupper(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isupper(*register_a(a, False))
    else:
        def impl(a):
            return np.array(isupper(*register_a(a, False))[0])
    return impl


@overload(np.char.islower, **OPTIONS)
def ov_char_islower(a):
    register_a, a_dim, _ = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return islower(*register_a(a, False))
    else:
        def impl(a):
            return np.array(islower(*register_a(a, False))[0])
    return impl

# ----------------------------------------------------------------------------------------------------------------------
