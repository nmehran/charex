"""
Numba overloads for numpy.character routines
"""

from charex.core import OPTIONS
from charex.core.string_intrinsics import (
    register_array_bytes, register_scalar_bytes,
    register_array_strings, register_scalar_strings
)
from charex.numpy.overloads.definitions import (
    greater_equal, greater, equal,
    equal_sub32_bytes, equal_sub32_unicode,
    compare_chararrays,
    count, endswith, startswith, find, rfind, index, rindex, str_len,
    str_len_bytes, _str_len_loop,
    isalpha, isalnum, isdecimal, isdigit, islower, isnumeric, isspace,
    istitle, isupper, scalar_bytes_len, scalar_strings_len,
    scalar_bytes_isalpha, scalar_strings_isalpha,
    scalar_bytes_isalnum, scalar_strings_isalnum, scalar_strings_isdecimal,
    scalar_bytes_isdigit, scalar_strings_isdigit, scalar_strings_isnumeric,
    scalar_bytes_isspace, scalar_strings_isspace,
    scalar_bytes_istitle, scalar_strings_istitle,
    scalar_bytes_isupper, scalar_strings_isupper,
    scalar_bytes_islower, scalar_strings_islower
)
from numba.core import types
from numba.core.errors import (
    NumbaError, NumbaTypeError, NumbaNotImplementedError, NumbaValueError
)
from numba.core.typing.templates import AttributeTemplate
from numba.extending import infer_getattr, overload, register_jitable
from numba import literally
import numpy as np


_CHAR_INFO_SCALARS_AS_ARRAY = not isinstance(np.char.str_len, np.ufunc)
_NUMPY_MAJOR = int(np.__version__.split('.')[0])


def _char_count(a, sub, start=0, end=None):
    return np.char.count(a, sub, start, end)


def _char_endswith(a, suffix, start=0, end=None):
    return np.char.endswith(a, suffix, start, end)


def _char_startswith(a, prefix, start=0, end=None):
    return np.char.startswith(a, prefix, start, end)


def _char_find(a, sub, start=0, end=None):
    return np.char.find(a, sub, start, end)


def _char_rfind(a, sub, start=0, end=None):
    return np.char.rfind(a, sub, start, end)


def _char_index(a, sub, start=0, end=None):
    return np.char.index(a, sub, start, end)


def _char_rindex(a, sub, start=0, end=None):
    return np.char.rindex(a, sub, start, end)


def _char_str_len(a):
    return np.char.str_len(a)


def _char_isalpha(a):
    return np.char.isalpha(a)


def _char_isalnum(a):
    return np.char.isalnum(a)


def _char_isdecimal(a):
    return np.char.isdecimal(a)


def _char_isdigit(a):
    return np.char.isdigit(a)


def _char_islower(a):
    return np.char.islower(a)


def _char_isnumeric(a):
    return np.char.isnumeric(a)


def _char_isspace(a):
    return np.char.isspace(a)


def _char_istitle(a):
    return np.char.istitle(a)


def _char_isupper(a):
    return np.char.isupper(a)


_CHAR_INFO_FUNCTIONS = {
    'count': _char_count,
    'endswith': _char_endswith,
    'startswith': _char_startswith,
    'find': _char_find,
    'rfind': _char_rfind,
    'index': _char_index,
    'rindex': _char_rindex,
    'str_len': _char_str_len,
    'isalpha': _char_isalpha,
    'isalnum': _char_isalnum,
    'isdecimal': _char_isdecimal,
    'isdigit': _char_isdigit,
    'islower': _char_islower,
    'isnumeric': _char_isnumeric,
    'isspace': _char_isspace,
    'istitle': _char_istitle,
    'isupper': _char_isupper,
}


# NumPy 1.x exposes these as array-function dispatchers, and Numba has scalar
# builtins that can win before charex overloads. Resolve np.char.<name> to a
# charex wrapper inside compiled code so the same overload owns every shape.
@infer_getattr
class _CharModuleAttrs(AttributeTemplate):
    key = types.Module(np.char)

    def generic_resolve(self, value, attr):
        function = _CHAR_INFO_FUNCTIONS.get(attr)
        if function is not None:
            return self.context.resolve_value_type(function)


def _overload_char_function(numpy_function, wrapper_function):
    def decorate(overload_function):
        overload(numpy_function, **OPTIONS)(overload_function)
        overload(wrapper_function, **OPTIONS)(overload_function)
        return overload_function
    return decorate


@register_jitable(boundscheck=False, forceinline=True,
                  no_cpython_wrapper=True, nogil=True)
def _char_info_scalar_result(value):
    if _CHAR_INFO_SCALARS_AS_ARRAY:
        return np.array(value)
    return value


# ----------------------------------------------------------------------------------------------------------------------
# Comparison Operators


def _ensure_slice(start, end):
    """Ensure start and end slice argument is an integer type."""
    if _NUMPY_MAJOR < 2:
        start_ok = start is None or isinstance(start, (
            int, types.Integer, types.NoneType, types.Omitted
        ))
    else:
        start_ok = isinstance(start, (int, types.Integer, types.Omitted))
    end_ok = end is None or isinstance(end, (int, types.Integer,
                                             types.NoneType, types.Omitted))
    if not start_ok or not end_ok:
        raise NumbaTypeError("slice indices must be integers or None "
                             "or have an __index__ method")
    return 0, np.iinfo(np.int64).max


def _ensure_type(x, exception: NumbaError = None):
    """Ensure argument is a character type with appropriate layout and shape."""
    ndim = -1
    if isinstance(x, types.Array):
        ndim = x.ndim
        if ndim > 1:
            raise NumbaValueError('charex supports only scalars and '
                                  'one-dimensional arrays')
        if x.layout != 'C':
            raise NumbaValueError('charex requires C-contiguous arrays; '
                                  'call numpy.ascontiguousarray')
        x = x.dtype
        if not isinstance(x, (types.CharSeq,
                              types.UnicodeCharSeq)) or not x.count:
            ndim = None
    elif isinstance(x, (types.CharSeq, types.UnicodeCharSeq)):
        ndim = -2
    elif not isinstance(x, (types.Bytes, types.UnicodeType)):
        ndim = None
    if isinstance(exception, NumbaError) and ndim is None:
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


def _register_pair(x1, x2, exception: (NumbaError, int) = None):
    """Determines the call function for the comparison pair, based on type."""
    e = exception or NumbaTypeError("comparison of non-string arrays")
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
                e = NumbaTypeError(f"must be {as_type}, not {_str_type(x2)}")
            else:
                e = NumbaTypeError("string operation on non-string array")
        else:
            e = NumbaNotImplementedError('NotImplemented')
        raise e
    return register_x1, register_x2, x1_dim, x2_dim


def _array_char_width(x):
    if not isinstance(x, types.Array):
        return None, 0
    if isinstance(x.dtype, types.CharSeq):
        return 'bytes', x.dtype.count
    if isinstance(x.dtype, types.UnicodeCharSeq):
        return 'unicode', x.dtype.count
    return None, 0


def _equal_impl(x1, x2):
    """Choose the equality kernel for fixed-width operands."""
    kind1, width1 = _array_char_width(x1)
    kind2, width2 = _array_char_width(x2)
    array_count = int(width1 > 0) + int(width2 > 0)
    width = max(width1, width2)
    same_width = array_count == 2 and kind1 == kind2 and width1 == width2

    if same_width and width < 32:
        if kind1 == 'bytes':
            return equal_sub32_bytes
        if kind1 == 'unicode':
            return equal_sub32_unicode
    return equal


def _register_single(x1, exception: NumbaError = None):
    """Determines the call function for the input, based on type."""
    e = exception or NumbaTypeError("string operation on non-string array")
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
    equal_impl = _equal_impl(x1, x2)
    left_scalar_like = x1_dim <= 0 < x2_dim

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if left_scalar_like:
                return equal_impl(*register_x2(x2, False),
                                  *register_x1(x1, False))
            return equal_impl(*register_x1(x1, False),
                              *register_x2(x2, False))
    else:
        def impl(x1, x2):
            return np.array(equal_impl(*register_x1(x1, False),
                                       *register_x2(x2, False))[0])
    return impl


@overload(np.char.not_equal, **OPTIONS)
def ov_char_not_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    equal_impl = _equal_impl(x1, x2)
    left_scalar_like = x1_dim <= 0 < x2_dim

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if left_scalar_like:
                return ~equal_impl(*register_x2(x2, False),
                                   *register_x1(x1, False))
            return ~equal_impl(*register_x1(x1, False),
                               *register_x2(x2, False))
    else:
        def impl(x1, x2):
            return np.array(~equal_impl(*register_x1(x1, False),
                                        *register_x2(x2, False))[0])
    return impl


@overload(np.char.greater_equal, **OPTIONS)
def ov_char_greater_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    left_scalar_like = x1_dim <= 0 < x2_dim

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if left_scalar_like:
                return greater_equal(*register_x2(x2, False),
                                     *register_x1(x1, False), True)
            return greater_equal(*register_x1(x1, False),
                                 *register_x2(x2, False))
    else:
        def impl(x1, x2):
            return np.array(greater_equal(*register_x1(x1, False),
                                          *register_x2(x2, False))[0])
    return impl


@overload(np.char.greater, **OPTIONS)
def ov_char_greater(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    left_scalar_like = x1_dim <= 0 < x2_dim

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if left_scalar_like:
                return greater(*register_x2(x2, False),
                               *register_x1(x1, False), True)
            return greater(*register_x1(x1, False),
                           *register_x2(x2, False))
    else:
        def impl(x1, x2):
            return np.array(greater(*register_x1(x1, False),
                                    *register_x2(x2, False))[0])
    return impl


@overload(np.char.less, **OPTIONS)
def ov_char_less(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    left_scalar_like = x1_dim <= 0 < x2_dim

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if left_scalar_like:
                return ~greater_equal(*register_x2(x2, False),
                                      *register_x1(x1, False), True)
            return ~greater_equal(*register_x1(x1, False),
                                  *register_x2(x2, False))
    else:
        def impl(x1, x2):
            return np.array(~greater_equal(*register_x1(x1, False),
                                           *register_x2(x2, False))[0])
    return impl


@overload(np.char.less_equal, **OPTIONS)
def ov_char_less_equal(x1, x2):
    register_x1, register_x2, x1_dim, x2_dim = _register_pair(x1, x2)
    left_scalar_like = x1_dim <= 0 < x2_dim

    if x1_dim > 0 or x2_dim > 0:
        def impl(x1, x2):
            if left_scalar_like:
                return ~greater(*register_x2(x2, False),
                                *register_x1(x1, False), True)
            return ~greater(*register_x1(x1, False),
                            *register_x2(x2, False))
    else:
        def impl(x1, x2):
            return np.array(~greater(*register_x1(x1, False),
                                     *register_x2(x2, False))[0])
    return impl


@overload(np.char.compare_chararrays, **OPTIONS)
def ov_char_compare_chararrays(a1, a2, cmp, rstrip):
    if not isinstance(cmp, (types.Bytes, types.UnicodeType)):
        raise NumbaTypeError(f'a bytes-like object is required, not {cmp.name}')

    register_a1, register_a2, a1_dim, a2_dim = _register_pair(a1, a2)
    left_scalar_like = a1_dim <= 0 < a2_dim

    if a1_dim > 0 or a2_dim > 0:
        def impl(a1, a2, cmp, rstrip):
            if left_scalar_like:
                return compare_chararrays(*register_a2(a2, False),
                                          *register_a1(a1, False),
                                          True, cmp, rstrip)
            return compare_chararrays(*register_a1(a1, False),
                                      *register_a2(a2, False),
                                      False, cmp, rstrip)
    else:
        def impl(a1, a2, cmp, rstrip):
            return np.array(compare_chararrays(*register_a1(a1, False),
                                               *register_a2(a2, False),
                                               False, cmp, rstrip)[0])
    return impl


# ----------------------------------------------------------------------------------------------------------------------
# String Information


@_overload_char_function(np.char.count, _char_count)
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
            return _char_info_scalar_result(
                count(*register_a(a, False),
                      *register_sub(sub, False),
                      start, end)[0])
    return impl


@_overload_char_function(np.char.endswith, _char_endswith)
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
            return _char_info_scalar_result(
                endswith(*register_a(a, False),
                         *register_sub(suffix, False),
                         start, end)[0])
    return impl


@_overload_char_function(np.char.startswith, _char_startswith)
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
            return _char_info_scalar_result(
                startswith(*register_a(a, False),
                           *register_sub(prefix, False),
                           start, end)[0])
    return impl


@_overload_char_function(np.char.find, _char_find)
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
            return _char_info_scalar_result(
                find(*register_a(a, False),
                     *register_sub(sub, False),
                     start, end)[0])
    return impl


@_overload_char_function(np.char.rfind, _char_rfind)
def ov_char_rfind(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return rfind(*register_a(a, False),
                         *register_sub(sub, False),
                         start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                rfind(*register_a(a, False),
                      *register_sub(sub, False),
                      start, end)[0])
    return impl


@_overload_char_function(np.char.index, _char_index)
def ov_char_index(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return index(*register_a(a, False),
                         *register_sub(sub, False),
                         start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                index(*register_a(a, False),
                      *register_sub(sub, False),
                      start, end)[0])
    return impl


@_overload_char_function(np.char.rindex, _char_rindex)
def ov_char_rindex(a, sub, start=0, end=None):
    register_a, register_sub, a_dim, sub_dim = _register_pair(a, sub, 1)
    s, e = _ensure_slice(start, end)

    if a_dim > 0 or sub_dim > 0:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return rindex(*register_a(a, False),
                          *register_sub(sub, False),
                          start, end)
    else:
        def impl(a, sub, start=0, end=None):
            start = start or s
            end = e if end is None else end
            return _char_info_scalar_result(
                rindex(*register_a(a, False),
                       *register_sub(sub, False),
                       start, end)[0])
    return impl


@_overload_char_function(np.char.str_len, _char_str_len)
def ov_char_str_len(a):
    register_a, a_dim, as_bytes = _register_single(a)
    array_len = str_len_bytes if as_bytes else str_len
    width = a.dtype.count if isinstance(a, types.Array) else 0

    if a_dim > 0 and width:
        if as_bytes:
            direct_len = _str_len_loop if width <= 8 else str_len_bytes

            def impl(a):
                return direct_len(a.view(np.uint8), a.size, literally(width))
        else:
            direct_len = _str_len_loop if width <= 16 else str_len

            def impl(a):
                return direct_len(a.view(np.int32), a.size, width)
    elif a_dim > 0:
        def impl(a):
            return array_len(*register_a(a, False))
    elif a_dim == -2:
        if as_bytes:
            def impl(a):
                return _char_info_scalar_result(scalar_bytes_len(a))
        else:
            def impl(a):
                return _char_info_scalar_result(scalar_strings_len(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                array_len(*register_a(a, False))[0])
    return impl


@_overload_char_function(np.char.isalpha, _char_isalpha)
def ov_char_isalpha(a):
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isalpha(*register_a(a, False), as_bytes)
    elif a_dim == -2:
        if as_bytes:
            def impl(a):
                return _char_info_scalar_result(scalar_bytes_isalpha(a))
        else:
            def impl(a):
                return _char_info_scalar_result(scalar_strings_isalpha(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                isalpha(*register_a(a, False), as_bytes)[0])
    return impl


@_overload_char_function(np.char.isalnum, _char_isalnum)
def ov_char_isalnum(a):
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isalnum(*register_a(a, False), as_bytes)
    elif a_dim == -2:
        if as_bytes:
            def impl(a):
                return _char_info_scalar_result(scalar_bytes_isalnum(a))
        else:
            def impl(a):
                return _char_info_scalar_result(scalar_strings_isalnum(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                isalnum(*register_a(a, False), as_bytes)[0])
    return impl


@_overload_char_function(np.char.isspace, _char_isspace)
def ov_char_isspace(a):
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isspace(*register_a(a, False), as_bytes)
    elif a_dim == -2:
        if as_bytes:
            def impl(a):
                return _char_info_scalar_result(scalar_bytes_isspace(a))
        else:
            def impl(a):
                return _char_info_scalar_result(scalar_strings_isspace(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                isspace(*register_a(a, False), as_bytes)[0])
    return impl


@_overload_char_function(np.char.isdecimal, _char_isdecimal)
def ov_char_isdecimal(a):
    catch_incompatible = NumbaTypeError("isnumeric is only available for "
                                        "Unicode strings and arrays")
    register_a, a_dim, as_bytes = _register_single(a, catch_incompatible)
    if as_bytes:
        raise catch_incompatible

    if a_dim > 0:
        def impl(a):
            return isdecimal(*register_a(a, False))
    elif a_dim == -2:
        def impl(a):
            return _char_info_scalar_result(scalar_strings_isdecimal(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                isdecimal(*register_a(a, False))[0])
    return impl


@_overload_char_function(np.char.isdigit, _char_isdigit)
def ov_char_isdigit(a):
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isdigit(*register_a(a, False), as_bytes)
    elif a_dim == -2:
        if as_bytes:
            def impl(a):
                return _char_info_scalar_result(scalar_bytes_isdigit(a))
        else:
            def impl(a):
                return _char_info_scalar_result(scalar_strings_isdigit(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                isdigit(*register_a(a, False), as_bytes)[0])
    return impl


@_overload_char_function(np.char.isnumeric, _char_isnumeric)
def ov_char_isnumeric(a):
    catch_incompatible = NumbaTypeError("isnumeric is only available for "
                                        "Unicode strings and arrays")
    register_a, a_dim, as_bytes = _register_single(a, catch_incompatible)
    if as_bytes:
        raise catch_incompatible

    if a_dim > 0:
        def impl(a):
            return isnumeric(*register_a(a, False))
    elif a_dim == -2:
        def impl(a):
            return _char_info_scalar_result(scalar_strings_isnumeric(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                isnumeric(*register_a(a, False))[0])
    return impl


@_overload_char_function(np.char.istitle, _char_istitle)
def ov_char_istitle(a):
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return istitle(*register_a(a, False), as_bytes)
    elif a_dim == -2:
        if as_bytes:
            def impl(a):
                return _char_info_scalar_result(scalar_bytes_istitle(a))
        else:
            def impl(a):
                return _char_info_scalar_result(scalar_strings_istitle(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                istitle(*register_a(a, False), as_bytes)[0])
    return impl


@_overload_char_function(np.char.isupper, _char_isupper)
def ov_char_isupper(a):
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return isupper(*register_a(a, False), as_bytes)
    elif a_dim == -2:
        if as_bytes:
            def impl(a):
                return _char_info_scalar_result(scalar_bytes_isupper(a))
        else:
            def impl(a):
                return _char_info_scalar_result(scalar_strings_isupper(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                isupper(*register_a(a, False), as_bytes)[0])
    return impl


@_overload_char_function(np.char.islower, _char_islower)
def ov_char_islower(a):
    register_a, a_dim, as_bytes = _register_single(a)

    if a_dim > 0:
        def impl(a):
            return islower(*register_a(a, False), as_bytes)
    elif a_dim == -2:
        if as_bytes:
            def impl(a):
                return _char_info_scalar_result(scalar_bytes_islower(a))
        else:
            def impl(a):
                return _char_info_scalar_result(scalar_strings_islower(a))
    else:
        def impl(a):
            return _char_info_scalar_result(
                islower(*register_a(a, False), as_bytes)[0])
    return impl

# ----------------------------------------------------------------------------------------------------------------------
