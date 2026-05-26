"""Numba overloads for NumPy's np.strings routines."""

from charex.core import OPTIONS
from charex.numpy.overloads._shared import (
    equal_dispatch, equal_kernel, order_dispatch, try_register_pair,
)
from charex.numpy.stringdtype import (
    is_stringdtype_array_type, stringdtype_acquire_allocator,
    stringdtype_codepoint_len, stringdtype_equal,
    stringdtype_release_allocator,
)
from charex.numpy.overloads.definitions import (
    equal, equal_sub32_bytes, equal_sub32_unicode, greater, greater_equal,
)
from charex.numpy.overloads.char import _CHAR_INFO_FUNCTIONS, ov_char_str_len
from numba.core import types
from numba.core.errors import NumbaValueError
from numba.core.typing.templates import AttributeTemplate
from numba.extending import infer_getattr, overload
import numpy as np


_STRINGS = getattr(np, 'strings', None)


def _has_string_operand(left, right):
    string_types = (types.Bytes, types.CharSeq,
                    types.UnicodeType, types.UnicodeCharSeq)
    if isinstance(left, types.Array):
        if isinstance(left.dtype, string_types) and left.dtype.count:
            return True
    elif isinstance(left, string_types):
        return True

    if isinstance(right, types.Array):
        if isinstance(right.dtype, string_types) and right.dtype.count:
            return True
    elif isinstance(right, string_types):
        return True
    return False


def _numpy_fallback(op):
    if op == 'equal':
        def impl(left, right):
            return np.equal(left, right)
    elif op == 'not_equal':
        def impl(left, right):
            return np.not_equal(left, right)
    elif op == 'greater_equal':
        def impl(left, right):
            return np.greater_equal(left, right)
    elif op == 'greater':
        def impl(left, right):
            return np.greater(left, right)
    elif op == 'less':
        def impl(left, right):
            return np.less(left, right)
    else:
        def impl(left, right):
            return np.less_equal(left, right)
    return impl


def _overload_equal(left, right, invert):
    if is_stringdtype_array_type(left) or is_stringdtype_array_type(right):
        if not is_stringdtype_array_type(left) \
                or not is_stringdtype_array_type(right):
            return None
        if left.ndim > 1 or right.ndim > 1:
            raise NumbaValueError('charex supports only scalars and '
                                  'one-dimensional arrays')
        if left.layout != 'C' or right.layout != 'C':
            raise NumbaValueError('charex requires C-contiguous arrays; '
                                  'call numpy.ascontiguousarray')

        def impl(left, right):
            if left.size != right.size:
                raise ValueError('shape mismatch: objects cannot be '
                                 'broadcast to a single shape')
            result = np.empty(left.size, np.bool_)
            left_allocator = stringdtype_acquire_allocator(left)
            right_allocator = stringdtype_acquire_allocator(right)
            for i in range(left.size):
                result[i] = stringdtype_equal(
                    left, i, left_allocator, right, i, right_allocator)
            stringdtype_release_allocator(left_allocator)
            stringdtype_release_allocator(right_allocator)
            return ~result if invert else result

        return impl

    registered = try_register_pair(left, right)
    if registered is None:
        if _has_string_operand(left, right):
            return None
        return _numpy_fallback('not_equal' if invert else 'equal')

    register_left, register_right, left_dim, right_dim = registered
    return equal_dispatch(register_left, register_right, left_dim, right_dim,
                          equal_kernel(left, right, equal, equal_sub32_bytes,
                                       equal_sub32_unicode),
                          False, invert)


def _overload_order(left, right, op):
    registered = try_register_pair(left, right)
    if registered is None:
        if _has_string_operand(left, right):
            return None
        return _numpy_fallback(op)

    register_left, register_right, left_dim, right_dim = registered
    return order_dispatch(register_left, register_right, left_dim, right_dim,
                          greater, greater_equal, op, False)


if _STRINGS is not None:
    def _strings_equal(left, right):
        return _STRINGS.equal(left, right)

    def _strings_not_equal(left, right):
        return _STRINGS.not_equal(left, right)

    def _strings_greater_equal(left, right):
        return _STRINGS.greater_equal(left, right)

    def _strings_greater(left, right):
        return _STRINGS.greater(left, right)

    def _strings_less(left, right):
        return _STRINGS.less(left, right)

    def _strings_less_equal(left, right):
        return _STRINGS.less_equal(left, right)

    def _strings_str_len(value):
        return _STRINGS.str_len(value)

    _STRINGS_FUNCTIONS = {
        **_CHAR_INFO_FUNCTIONS,
        'equal': _strings_equal,
        'not_equal': _strings_not_equal,
        'greater_equal': _strings_greater_equal,
        'greater': _strings_greater,
        'less': _strings_less,
        'less_equal': _strings_less_equal,
        'str_len': _strings_str_len,
    }

    @infer_getattr
    class _StringsModuleAttrs(AttributeTemplate):
        key = types.Module(_STRINGS)

        def generic_resolve(self, value, attr):
            function = _STRINGS_FUNCTIONS.get(attr)
            if function is not None:
                return self.context.resolve_value_type(function)

    @overload(_strings_equal, **OPTIONS)
    def ov_strings_equal(left, right):
        return _overload_equal(left, right, False)

    @overload(_strings_not_equal, **OPTIONS)
    def ov_strings_not_equal(left, right):
        return _overload_equal(left, right, True)

    @overload(_strings_greater_equal, **OPTIONS)
    def ov_strings_greater_equal(left, right):
        return _overload_order(left, right, 'greater_equal')

    @overload(_strings_greater, **OPTIONS)
    def ov_strings_greater(left, right):
        return _overload_order(left, right, 'greater')

    @overload(_strings_less, **OPTIONS)
    def ov_strings_less(left, right):
        return _overload_order(left, right, 'less')

    @overload(_strings_less_equal, **OPTIONS)
    def ov_strings_less_equal(left, right):
        return _overload_order(left, right, 'less_equal')

    @overload(_strings_str_len, **OPTIONS)
    def ov_strings_str_len(value):
        if not is_stringdtype_array_type(value):
            return ov_char_str_len(value)

        if value.ndim > 1:
            raise NumbaValueError('charex supports only scalars and '
                                  'one-dimensional arrays')
        if value.layout != 'C':
            raise NumbaValueError('charex requires C-contiguous arrays; '
                                  'call numpy.ascontiguousarray')

        def impl(value):
            result = np.empty(value.size, np.int64)
            allocator = stringdtype_acquire_allocator(value)
            null_string = False
            for i in range(value.size):
                length = stringdtype_codepoint_len(value, i, allocator)
                if length < 0:
                    null_string = True
                    length = 0
                result[i] = length
            stringdtype_release_allocator(allocator)
            if null_string:
                raise ValueError('The length of a null string is undefined')
            return result

        return impl
