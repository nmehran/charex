"""Numba overloads for NumPy's np.strings routines."""

from charex.core import OPTIONS
from charex.numpy.overloads._shared import (
    ensure_slice, equal_dispatch, equal_kernel, order_dispatch,
    try_register_pair,
)
from charex.numpy.stringdtype import (
    is_stringdtype_array_type, stringdtype_acquire_allocator,
    stringdtype_acquire_allocators, stringdtype_codepoint_len_data,
    stringdtype_count_data, stringdtype_data_ptr, stringdtype_endswith_data,
    stringdtype_equal_data, stringdtype_find_data,
    stringdtype_release_allocator, stringdtype_release_allocators,
    stringdtype_rfind_data, stringdtype_startswith_data,
)
from charex.numpy.overloads.definitions import (
    equal, equal_sub32_bytes, equal_sub32_unicode, greater, greater_equal,
)
from charex.numpy.overloads.char import (
    _CHAR_INFO_FUNCTIONS, ov_char_count, ov_char_endswith, ov_char_find,
    ov_char_index, ov_char_rfind, ov_char_rindex, ov_char_startswith,
    ov_char_str_len,
)
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
            raise NumbaValueError('StringDType comparisons currently require '
                                  'two StringDType arrays')
        if left.ndim != 1 or right.ndim != 1:
            raise NumbaValueError('charex StringDType support currently '
                                  'requires one-dimensional arrays')
        if left.layout != 'C' or right.layout != 'C':
            raise NumbaValueError('charex requires C-contiguous arrays; '
                                  'call numpy.ascontiguousarray')

        def impl(left, right):
            if left.size != right.size:
                raise ValueError('shape mismatch: objects cannot be '
                                 'broadcast to a single shape')
            result = np.empty(left.size, np.bool_)
            allocators = stringdtype_acquire_allocators(left, right)
            left_allocator = allocators[0]
            right_allocator = allocators[1]
            left_data = stringdtype_data_ptr(left)
            right_data = stringdtype_data_ptr(right)
            for i in range(left.size):
                result[i] = stringdtype_equal_data(
                    left_data, i, left_allocator,
                    right_data, i, right_allocator,
                )
            stringdtype_release_allocators(allocators)
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


def _overload_affix(value, pattern, start, end, suffix):
    if is_stringdtype_array_type(value) or is_stringdtype_array_type(pattern):
        if not is_stringdtype_array_type(value) \
                or not is_stringdtype_array_type(pattern):
            raise NumbaValueError('StringDType prefix/suffix operations '
                                  'currently require two StringDType arrays')
        if value.ndim != 1 or pattern.ndim != 1:
            raise NumbaValueError('charex StringDType support currently '
                                  'requires one-dimensional arrays')
        if value.layout != 'C' or pattern.layout != 'C':
            raise NumbaValueError('charex requires C-contiguous arrays; '
                                  'call numpy.ascontiguousarray')
        s, e = ensure_slice(start, end)

        def impl(value, pattern, start=0, end=None):
            start = start or s
            end = e if end is None else end
            if value.size != pattern.size:
                raise ValueError('shape mismatch: objects cannot be '
                                 'broadcast to a single shape')
            result = np.empty(value.size, np.bool_)
            allocators = stringdtype_acquire_allocators(value, pattern)
            value_allocator = allocators[0]
            pattern_allocator = allocators[1]
            value_data = stringdtype_data_ptr(value)
            pattern_data = stringdtype_data_ptr(pattern)
            for i in range(value.size):
                if suffix:
                    result[i] = stringdtype_endswith_data(
                        value_data, i, value_allocator,
                        pattern_data, i, pattern_allocator,
                        start, end,
                    )
                else:
                    result[i] = stringdtype_startswith_data(
                        value_data, i, value_allocator,
                        pattern_data, i, pattern_allocator,
                        start, end,
                    )
            stringdtype_release_allocators(allocators)
            return result

        return impl

    if suffix:
        return ov_char_endswith(value, pattern, start, end)
    return ov_char_startswith(value, pattern, start, end)


def _overload_search(value, pattern, start, end, op):
    if is_stringdtype_array_type(value) or is_stringdtype_array_type(pattern):
        if not is_stringdtype_array_type(value) \
                or not is_stringdtype_array_type(pattern):
            raise NumbaValueError('StringDType search operations currently '
                                  'require two StringDType arrays')
        if value.ndim != 1 or pattern.ndim != 1:
            raise NumbaValueError('charex StringDType support currently '
                                  'requires one-dimensional arrays')
        if value.layout != 'C' or pattern.layout != 'C':
            raise NumbaValueError('charex requires C-contiguous arrays; '
                                  'call numpy.ascontiguousarray')
        s, e = ensure_slice(start, end)

        def impl(value, pattern, start=0, end=None):
            start = start or s
            end = e if end is None else end
            if value.size != pattern.size:
                raise ValueError('shape mismatch: objects cannot be '
                                 'broadcast to a single shape')
            result = np.empty(value.size, np.int64)
            if value.size == 0:
                return result
            allocators = stringdtype_acquire_allocators(value, pattern)
            value_allocator = allocators[0]
            pattern_allocator = allocators[1]
            value_data = stringdtype_data_ptr(value)
            pattern_data = stringdtype_data_ptr(pattern)
            forward = op == 'find' or op == 'index'
            reverse = op == 'rfind' or op == 'rindex'
            raise_not_found = op == 'index' or op == 'rindex'
            not_found = False
            for i in range(value.size):
                if forward:
                    found = stringdtype_find_data(
                        value_data, i, value_allocator,
                        pattern_data, i, pattern_allocator,
                        start, end,
                    )
                elif reverse:
                    found = stringdtype_rfind_data(
                        value_data, i, value_allocator,
                        pattern_data, i, pattern_allocator,
                        start, end,
                    )
                else:
                    found = stringdtype_count_data(
                        value_data, i, value_allocator,
                        pattern_data, i, pattern_allocator,
                        start, end,
                    )
                if raise_not_found and found < 0:
                    not_found = True
                    break
                result[i] = found
            stringdtype_release_allocators(allocators)
            if not_found:
                raise ValueError('substring not found')
            return result

        return impl

    if op == 'find':
        return ov_char_find(value, pattern, start, end)
    if op == 'rfind':
        return ov_char_rfind(value, pattern, start, end)
    if op == 'index':
        return ov_char_index(value, pattern, start, end)
    if op == 'rindex':
        return ov_char_rindex(value, pattern, start, end)
    return ov_char_count(value, pattern, start, end)


if _STRINGS is not None:
    def _strings_count(value, sub, start=0, end=None):
        return _STRINGS.count(value, sub, start, end)

    def _strings_equal(left, right):
        return _STRINGS.equal(left, right)

    def _strings_find(value, sub, start=0, end=None):
        return _STRINGS.find(value, sub, start, end)

    def _strings_index(value, sub, start=0, end=None):
        return _STRINGS.index(value, sub, start, end)

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

    def _strings_rfind(value, sub, start=0, end=None):
        return _STRINGS.rfind(value, sub, start, end)

    def _strings_rindex(value, sub, start=0, end=None):
        return _STRINGS.rindex(value, sub, start, end)

    def _strings_endswith(value, suffix, start=0, end=None):
        return _STRINGS.endswith(value, suffix, start, end)

    def _strings_startswith(value, prefix, start=0, end=None):
        return _STRINGS.startswith(value, prefix, start, end)

    def _strings_str_len(value):
        return _STRINGS.str_len(value)

    _STRINGS_FUNCTIONS = {
        **_CHAR_INFO_FUNCTIONS,
        'count': _strings_count,
        'endswith': _strings_endswith,
        'equal': _strings_equal,
        'find': _strings_find,
        'index': _strings_index,
        'not_equal': _strings_not_equal,
        'greater_equal': _strings_greater_equal,
        'greater': _strings_greater,
        'less': _strings_less,
        'less_equal': _strings_less_equal,
        'rfind': _strings_rfind,
        'rindex': _strings_rindex,
        'startswith': _strings_startswith,
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

    @overload(_strings_count, **OPTIONS)
    def ov_strings_count(value, sub, start=0, end=None):
        return _overload_search(value, sub, start, end, 'count')

    @overload(_strings_find, **OPTIONS)
    def ov_strings_find(value, sub, start=0, end=None):
        return _overload_search(value, sub, start, end, 'find')

    @overload(_strings_index, **OPTIONS)
    def ov_strings_index(value, sub, start=0, end=None):
        return _overload_search(value, sub, start, end, 'index')

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

    @overload(_strings_rfind, **OPTIONS)
    def ov_strings_rfind(value, sub, start=0, end=None):
        return _overload_search(value, sub, start, end, 'rfind')

    @overload(_strings_rindex, **OPTIONS)
    def ov_strings_rindex(value, sub, start=0, end=None):
        return _overload_search(value, sub, start, end, 'rindex')

    @overload(_strings_endswith, **OPTIONS)
    def ov_strings_endswith(value, suffix, start=0, end=None):
        return _overload_affix(value, suffix, start, end, True)

    @overload(_strings_startswith, **OPTIONS)
    def ov_strings_startswith(value, prefix, start=0, end=None):
        return _overload_affix(value, prefix, start, end, False)

    @overload(_strings_str_len, **OPTIONS)
    def ov_strings_str_len(value):
        if not is_stringdtype_array_type(value):
            return ov_char_str_len(value)

        if value.ndim != 1:
            raise NumbaValueError('charex StringDType support currently '
                                  'requires one-dimensional arrays')
        if value.layout != 'C':
            raise NumbaValueError('charex requires C-contiguous arrays; '
                                  'call numpy.ascontiguousarray')

        def impl(value):
            result = np.empty(value.size, np.int64)
            allocator = stringdtype_acquire_allocator(value)
            data = stringdtype_data_ptr(value)
            null_string = False
            for i in range(value.size):
                length = stringdtype_codepoint_len_data(data, i, allocator)
                if length < 0:
                    null_string = True
                    length = 0
                result[i] = length
            stringdtype_release_allocator(allocator)
            if null_string:
                raise ValueError('The length of a null string is undefined')
            return result

        return impl
