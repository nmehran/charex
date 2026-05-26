"""Numba overloads for NumPy's np.strings routines."""

from charex.core import OPTIONS
from charex.numpy.overloads._shared import (
    ensure_slice, equal_dispatch, equal_kernel, order_dispatch,
    try_register_pair,
)
from charex.numpy.stringdtype import (
    is_stringdtype_array_type, stringdtype_acquire_allocator,
    stringdtype_acquire_allocators, stringdtype_codepoint_len_data,
    stringdtype_compare_data, stringdtype_compare_unicode_data,
    stringdtype_count_data, stringdtype_data_ptr, stringdtype_endswith_data,
    stringdtype_endswith_unicode_data, stringdtype_equal_data,
    stringdtype_equal_unicode_data, stringdtype_find_data,
    stringdtype_isalnum_data, stringdtype_isalpha_data,
    stringdtype_isdecimal_data, stringdtype_isdigit_data,
    stringdtype_islower_data, stringdtype_isnumeric_data,
    stringdtype_isspace_data, stringdtype_istitle_data,
    stringdtype_isupper_data, stringdtype_release_allocator,
    stringdtype_release_allocators, stringdtype_rfind_data,
    stringdtype_startswith_data, stringdtype_startswith_unicode_data,
    stringdtype_unicode_valid, unicode_endswith_stringdtype_data,
    unicode_startswith_stringdtype_data,
)
from charex.numpy.overloads.definitions import (
    equal, equal_sub32_bytes, equal_sub32_unicode, greater, greater_equal,
)
from charex.numpy.overloads.char import (
    _CHAR_INFO_FUNCTIONS, ov_char_count, ov_char_endswith, ov_char_find,
    ov_char_index, ov_char_isalnum, ov_char_isalpha, ov_char_isdecimal,
    ov_char_isdigit, ov_char_islower, ov_char_isnumeric, ov_char_isspace,
    ov_char_istitle, ov_char_isupper, ov_char_rfind, ov_char_rindex,
    ov_char_startswith, ov_char_str_len,
)
from numba.core import types
from numba.core.errors import NumbaValueError
from numba.core.typing.templates import AttributeTemplate
from numba.extending import infer_getattr, overload
import numpy as np


_STRINGS = getattr(np, 'strings', None)


def _validate_stringdtype_array(value):
    if value.ndim > 1:
        raise NumbaValueError('charex StringDType support currently '
                              'requires scalar or one-dimensional arrays')
    if value.ndim == 1 and value.layout != 'C':
        raise NumbaValueError('charex requires C-contiguous arrays; '
                              'call numpy.ascontiguousarray')


def _is_unicode_scalar(value):
    return isinstance(value, types.UnicodeType)


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
    left_stringdtype = is_stringdtype_array_type(left)
    right_stringdtype = is_stringdtype_array_type(right)
    if left_stringdtype or right_stringdtype:
        if left_stringdtype and _is_unicode_scalar(right):
            _validate_stringdtype_array(left)
            if left.ndim == 0:
                def impl(left, right):
                    if not stringdtype_unicode_valid(right):
                        raise TypeError('Invalid unicode code point found')
                    allocator = stringdtype_acquire_allocator(left)
                    result = stringdtype_equal_unicode_data(
                        stringdtype_data_ptr(left), 0, allocator, right)
                    stringdtype_release_allocator(allocator)
                    return not result if invert else result

                return impl

            def impl(left, right):
                if not stringdtype_unicode_valid(right):
                    raise TypeError('Invalid unicode code point found')
                result = np.empty(left.size, np.bool_)
                if left.size == 0:
                    return ~result if invert else result
                allocator = stringdtype_acquire_allocator(left)
                data = stringdtype_data_ptr(left)
                for i in range(left.size):
                    result[i] = stringdtype_equal_unicode_data(
                        data, i, allocator, right)
                stringdtype_release_allocator(allocator)
                return ~result if invert else result

            return impl

        if _is_unicode_scalar(left) and right_stringdtype:
            _validate_stringdtype_array(right)
            if right.ndim == 0:
                def impl(left, right):
                    if not stringdtype_unicode_valid(left):
                        raise TypeError('Invalid unicode code point found')
                    allocator = stringdtype_acquire_allocator(right)
                    result = stringdtype_equal_unicode_data(
                        stringdtype_data_ptr(right), 0, allocator, left)
                    stringdtype_release_allocator(allocator)
                    return not result if invert else result

                return impl

            def impl(left, right):
                if not stringdtype_unicode_valid(left):
                    raise TypeError('Invalid unicode code point found')
                result = np.empty(right.size, np.bool_)
                if right.size == 0:
                    return ~result if invert else result
                allocator = stringdtype_acquire_allocator(right)
                data = stringdtype_data_ptr(right)
                for i in range(right.size):
                    result[i] = stringdtype_equal_unicode_data(
                        data, i, allocator, left)
                stringdtype_release_allocator(allocator)
                return ~result if invert else result

            return impl

        if not left_stringdtype or not right_stringdtype:
            raise NumbaValueError('StringDType comparisons currently require '
                                  'two StringDType arrays')
        _validate_stringdtype_array(left)
        _validate_stringdtype_array(right)

        if left.ndim == 0 and right.ndim == 0:
            def impl(left, right):
                allocators = stringdtype_acquire_allocators(left, right)
                equal_result = stringdtype_equal_data(
                    stringdtype_data_ptr(left), 0, allocators[0],
                    stringdtype_data_ptr(right), 0, allocators[1],
                )
                stringdtype_release_allocators(allocators)
                return not equal_result if invert else equal_result

            return impl

        left_scalar = left.ndim == 0
        right_scalar = right.ndim == 0

        def impl(left, right):
            if not left_scalar and not right_scalar and left.size != right.size:
                raise ValueError('shape mismatch: objects cannot be '
                                 'broadcast to a single shape')
            size = right.size if left_scalar else left.size
            result = np.empty(size, np.bool_)
            if size == 0:
                return ~result if invert else result
            allocators = stringdtype_acquire_allocators(left, right)
            left_allocator = allocators[0]
            right_allocator = allocators[1]
            left_data = stringdtype_data_ptr(left)
            right_data = stringdtype_data_ptr(right)
            for i in range(size):
                result[i] = stringdtype_equal_data(
                    left_data, 0 if left_scalar else i, left_allocator,
                    right_data, 0 if right_scalar else i, right_allocator,
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
    left_stringdtype = is_stringdtype_array_type(left)
    right_stringdtype = is_stringdtype_array_type(right)
    if left_stringdtype or right_stringdtype:
        if left_stringdtype and _is_unicode_scalar(right):
            _validate_stringdtype_array(left)
            if left.ndim == 0:
                def impl(left, right):
                    if not stringdtype_unicode_valid(right):
                        raise TypeError('Invalid unicode code point found')
                    allocator = stringdtype_acquire_allocator(left)
                    cmp_result = stringdtype_compare_unicode_data(
                        stringdtype_data_ptr(left), 0, allocator, right)
                    stringdtype_release_allocator(allocator)
                    if op == 'greater':
                        return cmp_result > 0
                    if op == 'greater_equal':
                        return cmp_result >= 0
                    if op == 'less':
                        return cmp_result < 0
                    return cmp_result <= 0

                return impl

            def impl(left, right):
                if not stringdtype_unicode_valid(right):
                    raise TypeError('Invalid unicode code point found')
                result = np.empty(left.size, np.bool_)
                if left.size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(left)
                data = stringdtype_data_ptr(left)
                for i in range(left.size):
                    cmp_result = stringdtype_compare_unicode_data(
                        data, i, allocator, right)
                    if op == 'greater':
                        result[i] = cmp_result > 0
                    elif op == 'greater_equal':
                        result[i] = cmp_result >= 0
                    elif op == 'less':
                        result[i] = cmp_result < 0
                    else:
                        result[i] = cmp_result <= 0
                stringdtype_release_allocator(allocator)
                return result

            return impl

        if _is_unicode_scalar(left) and right_stringdtype:
            _validate_stringdtype_array(right)
            if right.ndim == 0:
                def impl(left, right):
                    if not stringdtype_unicode_valid(left):
                        raise TypeError('Invalid unicode code point found')
                    allocator = stringdtype_acquire_allocator(right)
                    cmp_result = -stringdtype_compare_unicode_data(
                        stringdtype_data_ptr(right), 0, allocator, left)
                    stringdtype_release_allocator(allocator)
                    if op == 'greater':
                        return cmp_result > 0
                    if op == 'greater_equal':
                        return cmp_result >= 0
                    if op == 'less':
                        return cmp_result < 0
                    return cmp_result <= 0

                return impl

            def impl(left, right):
                if not stringdtype_unicode_valid(left):
                    raise TypeError('Invalid unicode code point found')
                result = np.empty(right.size, np.bool_)
                if right.size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(right)
                data = stringdtype_data_ptr(right)
                for i in range(right.size):
                    cmp_result = -stringdtype_compare_unicode_data(
                        data, i, allocator, left)
                    if op == 'greater':
                        result[i] = cmp_result > 0
                    elif op == 'greater_equal':
                        result[i] = cmp_result >= 0
                    elif op == 'less':
                        result[i] = cmp_result < 0
                    else:
                        result[i] = cmp_result <= 0
                stringdtype_release_allocator(allocator)
                return result

            return impl

        if not left_stringdtype or not right_stringdtype:
            raise NumbaValueError('StringDType comparisons currently require '
                                  'two StringDType arrays')
        _validate_stringdtype_array(left)
        _validate_stringdtype_array(right)

        if left.ndim == 0 and right.ndim == 0:
            def impl(left, right):
                allocators = stringdtype_acquire_allocators(left, right)
                cmp_result = stringdtype_compare_data(
                    stringdtype_data_ptr(left), 0, allocators[0],
                    stringdtype_data_ptr(right), 0, allocators[1],
                )
                stringdtype_release_allocators(allocators)
                if op == 'greater':
                    return cmp_result > 0
                if op == 'greater_equal':
                    return cmp_result >= 0
                if op == 'less':
                    return cmp_result < 0
                return cmp_result <= 0

            return impl

        left_scalar = left.ndim == 0
        right_scalar = right.ndim == 0

        def impl(left, right):
            if not left_scalar and not right_scalar and left.size != right.size:
                raise ValueError('shape mismatch: objects cannot be '
                                 'broadcast to a single shape')
            size = right.size if left_scalar else left.size
            result = np.empty(size, np.bool_)
            if size == 0:
                return result
            allocators = stringdtype_acquire_allocators(left, right)
            left_allocator = allocators[0]
            right_allocator = allocators[1]
            left_data = stringdtype_data_ptr(left)
            right_data = stringdtype_data_ptr(right)
            for i in range(size):
                cmp_result = stringdtype_compare_data(
                    left_data, 0 if left_scalar else i, left_allocator,
                    right_data, 0 if right_scalar else i, right_allocator,
                )
                if op == 'greater':
                    result[i] = cmp_result > 0
                elif op == 'greater_equal':
                    result[i] = cmp_result >= 0
                elif op == 'less':
                    result[i] = cmp_result < 0
                else:
                    result[i] = cmp_result <= 0
            stringdtype_release_allocators(allocators)
            return result

        return impl

    registered = try_register_pair(left, right)
    if registered is None:
        if _has_string_operand(left, right):
            return None
        return _numpy_fallback(op)

    register_left, register_right, left_dim, right_dim = registered
    return order_dispatch(register_left, register_right, left_dim, right_dim,
                          greater, greater_equal, op, False)


def _overload_affix(value, pattern, start, end, suffix):
    value_stringdtype = is_stringdtype_array_type(value)
    pattern_stringdtype = is_stringdtype_array_type(pattern)
    if value_stringdtype or pattern_stringdtype:
        s, e = ensure_slice(start, end)

        if value_stringdtype and _is_unicode_scalar(pattern):
            _validate_stringdtype_array(value)
            if value.ndim == 0:
                def impl(value, pattern, start=0, end=None):
                    if not stringdtype_unicode_valid(pattern):
                        raise TypeError('Invalid unicode code point found')
                    start = start or s
                    end = e if end is None else end
                    allocator = stringdtype_acquire_allocator(value)
                    data = stringdtype_data_ptr(value)
                    if suffix:
                        result = stringdtype_endswith_unicode_data(
                            data, 0, allocator, pattern, start, end)
                    else:
                        result = stringdtype_startswith_unicode_data(
                            data, 0, allocator, pattern, start, end)
                    stringdtype_release_allocator(allocator)
                    return result

                return impl

            def impl(value, pattern, start=0, end=None):
                if not stringdtype_unicode_valid(pattern):
                    raise TypeError('Invalid unicode code point found')
                start = start or s
                end = e if end is None else end
                result = np.empty(value.size, np.bool_)
                if value.size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(value)
                data = stringdtype_data_ptr(value)
                for i in range(value.size):
                    if suffix:
                        result[i] = stringdtype_endswith_unicode_data(
                            data, i, allocator, pattern, start, end)
                    else:
                        result[i] = stringdtype_startswith_unicode_data(
                            data, i, allocator, pattern, start, end)
                stringdtype_release_allocator(allocator)
                return result

            return impl

        if _is_unicode_scalar(value) and pattern_stringdtype:
            _validate_stringdtype_array(pattern)
            if pattern.ndim == 0:
                def impl(value, pattern, start=0, end=None):
                    if not stringdtype_unicode_valid(value):
                        raise TypeError('Invalid unicode code point found')
                    start = start or s
                    end = e if end is None else end
                    allocator = stringdtype_acquire_allocator(pattern)
                    data = stringdtype_data_ptr(pattern)
                    if suffix:
                        result = unicode_endswith_stringdtype_data(
                            value, data, 0, allocator, start, end)
                    else:
                        result = unicode_startswith_stringdtype_data(
                            value, data, 0, allocator, start, end)
                    stringdtype_release_allocator(allocator)
                    return result

                return impl

            def impl(value, pattern, start=0, end=None):
                if not stringdtype_unicode_valid(value):
                    raise TypeError('Invalid unicode code point found')
                start = start or s
                end = e if end is None else end
                result = np.empty(pattern.size, np.bool_)
                if pattern.size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(pattern)
                data = stringdtype_data_ptr(pattern)
                for i in range(pattern.size):
                    if suffix:
                        result[i] = unicode_endswith_stringdtype_data(
                            value, data, i, allocator, start, end)
                    else:
                        result[i] = unicode_startswith_stringdtype_data(
                            value, data, i, allocator, start, end)
                stringdtype_release_allocator(allocator)
                return result

            return impl

        if not value_stringdtype or not pattern_stringdtype:
            raise NumbaValueError('StringDType prefix/suffix operations '
                                  'currently require two StringDType arrays')
        _validate_stringdtype_array(value)
        _validate_stringdtype_array(pattern)

        if value.ndim == 0 and pattern.ndim == 0:
            def impl(value, pattern, start=0, end=None):
                start = start or s
                end = e if end is None else end
                allocators = stringdtype_acquire_allocators(value, pattern)
                if suffix:
                    result = stringdtype_endswith_data(
                        stringdtype_data_ptr(value), 0, allocators[0],
                        stringdtype_data_ptr(pattern), 0, allocators[1],
                        start, end,
                    )
                else:
                    result = stringdtype_startswith_data(
                        stringdtype_data_ptr(value), 0, allocators[0],
                        stringdtype_data_ptr(pattern), 0, allocators[1],
                        start, end,
                    )
                stringdtype_release_allocators(allocators)
                return result

            return impl

        value_scalar = value.ndim == 0
        pattern_scalar = pattern.ndim == 0

        def impl(value, pattern, start=0, end=None):
            start = start or s
            end = e if end is None else end
            if not value_scalar and not pattern_scalar \
                    and value.size != pattern.size:
                raise ValueError('shape mismatch: objects cannot be '
                                 'broadcast to a single shape')
            size = pattern.size if value_scalar else value.size
            result = np.empty(size, np.bool_)
            if size == 0:
                return result
            allocators = stringdtype_acquire_allocators(value, pattern)
            value_allocator = allocators[0]
            pattern_allocator = allocators[1]
            value_data = stringdtype_data_ptr(value)
            pattern_data = stringdtype_data_ptr(pattern)
            for i in range(size):
                if suffix:
                    result[i] = stringdtype_endswith_data(
                        value_data, 0 if value_scalar else i, value_allocator,
                        pattern_data, 0 if pattern_scalar else i,
                        pattern_allocator,
                        start, end,
                    )
                else:
                    result[i] = stringdtype_startswith_data(
                        value_data, 0 if value_scalar else i, value_allocator,
                        pattern_data, 0 if pattern_scalar else i,
                        pattern_allocator,
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
        _validate_stringdtype_array(value)
        _validate_stringdtype_array(pattern)
        s, e = ensure_slice(start, end)

        if value.ndim == 0 and pattern.ndim == 0:
            def impl(value, pattern, start=0, end=None):
                start = start or s
                end = e if end is None else end
                allocators = stringdtype_acquire_allocators(value, pattern)
                value_data = stringdtype_data_ptr(value)
                pattern_data = stringdtype_data_ptr(pattern)
                if op == 'find' or op == 'index':
                    found = stringdtype_find_data(
                        value_data, 0, allocators[0],
                        pattern_data, 0, allocators[1],
                        start, end,
                    )
                elif op == 'rfind' or op == 'rindex':
                    found = stringdtype_rfind_data(
                        value_data, 0, allocators[0],
                        pattern_data, 0, allocators[1],
                        start, end,
                    )
                else:
                    found = stringdtype_count_data(
                        value_data, 0, allocators[0],
                        pattern_data, 0, allocators[1],
                        start, end,
                    )
                stringdtype_release_allocators(allocators)
                if (op == 'index' or op == 'rindex') and found < 0:
                    raise ValueError('substring not found')
                return found

            return impl

        value_scalar = value.ndim == 0
        pattern_scalar = pattern.ndim == 0

        def impl(value, pattern, start=0, end=None):
            start = start or s
            end = e if end is None else end
            if not value_scalar and not pattern_scalar \
                    and value.size != pattern.size:
                raise ValueError('shape mismatch: objects cannot be '
                                 'broadcast to a single shape')
            size = pattern.size if value_scalar else value.size
            result = np.empty(size, np.int64)
            if size == 0:
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
            for i in range(size):
                value_index = 0 if value_scalar else i
                pattern_index = 0 if pattern_scalar else i
                if forward:
                    found = stringdtype_find_data(
                        value_data, value_index, value_allocator,
                        pattern_data, pattern_index, pattern_allocator,
                        start, end,
                    )
                elif reverse:
                    found = stringdtype_rfind_data(
                        value_data, value_index, value_allocator,
                        pattern_data, pattern_index, pattern_allocator,
                        start, end,
                    )
                else:
                    found = stringdtype_count_data(
                        value_data, value_index, value_allocator,
                        pattern_data, pattern_index, pattern_allocator,
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


_CHAR_PREDICATE_OVERLOADS = {
    'isalpha': ov_char_isalpha,
    'isalnum': ov_char_isalnum,
    'isdecimal': ov_char_isdecimal,
    'isdigit': ov_char_isdigit,
    'isnumeric': ov_char_isnumeric,
    'isspace': ov_char_isspace,
    'islower': ov_char_islower,
    'isupper': ov_char_isupper,
    'istitle': ov_char_istitle,
}


def _overload_predicate(value, op):
    if not is_stringdtype_array_type(value):
        return _CHAR_PREDICATE_OVERLOADS[op](value)

    _validate_stringdtype_array(value)

    if value.ndim == 0:
        def impl(value):
            allocator = stringdtype_acquire_allocator(value)
            data = stringdtype_data_ptr(value)
            if op == 'isalpha':
                result = stringdtype_isalpha_data(data, 0, allocator)
            elif op == 'isalnum':
                result = stringdtype_isalnum_data(data, 0, allocator)
            elif op == 'isdecimal':
                result = stringdtype_isdecimal_data(data, 0, allocator)
            elif op == 'isdigit':
                result = stringdtype_isdigit_data(data, 0, allocator)
            elif op == 'isnumeric':
                result = stringdtype_isnumeric_data(data, 0, allocator)
            elif op == 'isspace':
                result = stringdtype_isspace_data(data, 0, allocator)
            elif op == 'islower':
                result = stringdtype_islower_data(data, 0, allocator)
            elif op == 'isupper':
                result = stringdtype_isupper_data(data, 0, allocator)
            else:
                result = stringdtype_istitle_data(data, 0, allocator)
            stringdtype_release_allocator(allocator)
            return result

        return impl

    def impl(value):
        result = np.empty(value.size, np.bool_)
        if value.size == 0:
            return result
        allocator = stringdtype_acquire_allocator(value)
        data = stringdtype_data_ptr(value)
        for i in range(value.size):
            if op == 'isalpha':
                result[i] = stringdtype_isalpha_data(data, i, allocator)
            elif op == 'isalnum':
                result[i] = stringdtype_isalnum_data(data, i, allocator)
            elif op == 'isdecimal':
                result[i] = stringdtype_isdecimal_data(data, i, allocator)
            elif op == 'isdigit':
                result[i] = stringdtype_isdigit_data(data, i, allocator)
            elif op == 'isnumeric':
                result[i] = stringdtype_isnumeric_data(data, i, allocator)
            elif op == 'isspace':
                result[i] = stringdtype_isspace_data(data, i, allocator)
            elif op == 'islower':
                result[i] = stringdtype_islower_data(data, i, allocator)
            elif op == 'isupper':
                result[i] = stringdtype_isupper_data(data, i, allocator)
            else:
                result[i] = stringdtype_istitle_data(data, i, allocator)
        stringdtype_release_allocator(allocator)
        return result

    return impl


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

    def _strings_isalpha(value):
        return _STRINGS.isalpha(value)

    def _strings_isalnum(value):
        return _STRINGS.isalnum(value)

    def _strings_isdecimal(value):
        return _STRINGS.isdecimal(value)

    def _strings_isdigit(value):
        return _STRINGS.isdigit(value)

    def _strings_islower(value):
        return _STRINGS.islower(value)

    def _strings_isnumeric(value):
        return _STRINGS.isnumeric(value)

    def _strings_isspace(value):
        return _STRINGS.isspace(value)

    def _strings_istitle(value):
        return _STRINGS.istitle(value)

    def _strings_isupper(value):
        return _STRINGS.isupper(value)

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
        'isalnum': _strings_isalnum,
        'isalpha': _strings_isalpha,
        'isdecimal': _strings_isdecimal,
        'isdigit': _strings_isdigit,
        'islower': _strings_islower,
        'isnumeric': _strings_isnumeric,
        'isspace': _strings_isspace,
        'istitle': _strings_istitle,
        'isupper': _strings_isupper,
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

        _validate_stringdtype_array(value)

        if value.ndim == 0:
            def impl(value):
                allocator = stringdtype_acquire_allocator(value)
                length = stringdtype_codepoint_len_data(
                    stringdtype_data_ptr(value), 0, allocator)
                stringdtype_release_allocator(allocator)
                if length < 0:
                    raise ValueError('The length of a null string is undefined')
                return length

            return impl

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

    @overload(_strings_isalpha, **OPTIONS)
    def ov_strings_isalpha(value):
        return _overload_predicate(value, 'isalpha')

    @overload(_strings_isalnum, **OPTIONS)
    def ov_strings_isalnum(value):
        return _overload_predicate(value, 'isalnum')

    @overload(_strings_isdecimal, **OPTIONS)
    def ov_strings_isdecimal(value):
        return _overload_predicate(value, 'isdecimal')

    @overload(_strings_isdigit, **OPTIONS)
    def ov_strings_isdigit(value):
        return _overload_predicate(value, 'isdigit')

    @overload(_strings_islower, **OPTIONS)
    def ov_strings_islower(value):
        return _overload_predicate(value, 'islower')

    @overload(_strings_isnumeric, **OPTIONS)
    def ov_strings_isnumeric(value):
        return _overload_predicate(value, 'isnumeric')

    @overload(_strings_isspace, **OPTIONS)
    def ov_strings_isspace(value):
        return _overload_predicate(value, 'isspace')

    @overload(_strings_istitle, **OPTIONS)
    def ov_strings_istitle(value):
        return _overload_predicate(value, 'istitle')

    @overload(_strings_isupper, **OPTIONS)
    def ov_strings_isupper(value):
        return _overload_predicate(value, 'isupper')
