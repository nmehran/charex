"""Numba overloads for NumPy's np.strings routines."""

from charex.core import JIT_OPTIONS, OPTIONS
from charex.numpy.overloads._shared import (
    ensure_slice, equal_dispatch, equal_kernel, order_dispatch,
    try_register_pair,
)
from charex.numpy.stringdtype import (
    _PACKED_STRING_SIZE, is_stringdtype_array_type,
    STRINGDTYPE_ORDER_ERROR, STRINGDTYPE_ORDER_FALSE,
    stringdtype_acquire_allocator, stringdtype_acquire_allocators,
    stringdtype_codepoint_len_data, stringdtype_compare_data,
    stringdtype_compare_na_data,
    stringdtype_compare_unicode_data, stringdtype_compare_unicode_na_data,
    stringdtype_compare_utf8_data,
    stringdtype_count_data, stringdtype_count_unicode_data,
    stringdtype_count_utf8_data,
    stringdtype_data_ptr, stringdtype_endswith_data,
    stringdtype_endswith_unicode_data, stringdtype_endswith_utf8_data,
    stringdtype_equal_data, stringdtype_equal_na_data,
    stringdtype_equal_unicode_data, stringdtype_equal_unicode_na_data,
    stringdtype_equal_utf8_data,
    stringdtype_find_data, stringdtype_find_unicode_data,
    stringdtype_find_utf8_data,
    stringdtype_free_utf8_span,
    stringdtype_isalnum_data, stringdtype_isalnum_na_data,
    stringdtype_isalpha_data, stringdtype_isalpha_na_data,
    stringdtype_isdecimal_data, stringdtype_isdecimal_na_data,
    stringdtype_isdigit_data, stringdtype_isdigit_na_data,
    stringdtype_islower_data, stringdtype_islower_na_data,
    stringdtype_isnumeric_data, stringdtype_isnumeric_na_data,
    stringdtype_isspace_data, stringdtype_isspace_na_data,
    stringdtype_istitle_data, stringdtype_istitle_na_data,
    stringdtype_isupper_data, stringdtype_isupper_na_data,
    stringdtype_na_name, stringdtype_not_equal_na_data,
    stringdtype_not_equal_unicode_na_data,
    stringdtype_release_allocator,
    stringdtype_release_allocators, stringdtype_rfind_data,
    stringdtype_rfind_unicode_data, stringdtype_rfind_utf8_data,
    stringdtype_startswith_data,
    stringdtype_startswith_unicode_data, stringdtype_startswith_utf8_data,
    stringdtype_codepoint_len_na_data, stringdtype_unicode_parts,
    stringdtype_unicode_utf8_span, stringdtype_unicode_valid,
    stringdtype_utf8_search_slice, stringdtype_utf8_slice,
    utf8_count_stringdtype_sliced_data,
    utf8_endswith_stringdtype_sliced_data,
    utf8_find_stringdtype_sliced_data,
    utf8_rfind_stringdtype_sliced_data,
    utf8_startswith_stringdtype_sliced_data,
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
from numba.extending import infer_getattr, overload, register_jitable
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


def _is_unicode_array_scalar(value):
    return isinstance(value, types.Array) and value.ndim == 0 \
        and isinstance(value.dtype, types.UnicodeCharSeq) \
        and value.dtype.count


def _is_unicode_array(value):
    return isinstance(value, types.Array) \
        and isinstance(value.dtype, types.UnicodeCharSeq) \
        and value.dtype.count


def _is_none(value):
    return isinstance(value, types.NoneType)


def _is_unicode_scalar_like(value):
    return _is_unicode_scalar(value) or _is_unicode_array_scalar(value)


def _validate_unicode_array(value):
    if value.ndim > 1:
        raise NumbaValueError('charex StringDType support currently '
                              'requires scalar or one-dimensional arrays')


def _stringdtype_na_kind(value):
    if is_stringdtype_array_type(value):
        return value.dtype.na_kind
    return 0


def _stringdtype_na_name_type(value):
    if is_stringdtype_array_type(value):
        return value.dtype.na_name
    return b''


def _compatible_stringdtype_na(left, right):
    left_kind = _stringdtype_na_kind(left)
    right_kind = _stringdtype_na_kind(right)
    if left_kind == 0 or right_kind == 0:
        return True
    return left_kind == right_kind \
        and _stringdtype_na_name_type(left) == _stringdtype_na_name_type(right)


def _has_stringdtype_na(*values):
    for value in values:
        if _stringdtype_na_kind(value) != 0:
            return True
    return False


def _reject_stringdtype_na(operation):
    raise NumbaValueError(
        f'StringDType na_object support for {operation} is still '
        'being prototyped',
    )


def _unicode_scalar_value(value):
    return value


@overload(_unicode_scalar_value, **OPTIONS)
def ov_unicode_scalar_value(value):
    if _is_unicode_array_scalar(value):
        def impl(value):
            return str(value[()])
        return impl
    if _is_unicode_scalar(value):
        def impl(value):
            return value
        return impl
    if isinstance(value, types.UnicodeCharSeq):
        def impl(value):
            return str(value)
        return impl


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


@register_jitable(**JIT_OPTIONS)
def _order_result(cmp_result, op_code):
    if op_code == 0:
        return cmp_result > 0
    if op_code == 1:
        return cmp_result >= 0
    if op_code == 2:
        return cmp_result < 0
    return cmp_result <= 0


@register_jitable(**JIT_OPTIONS)
def _stringdtype_order_result(cmp_result, op_code):
    if cmp_result == STRINGDTYPE_ORDER_ERROR:
        raise ValueError(
            'StringDType ordering is not supported for this null value')
    if cmp_result == STRINGDTYPE_ORDER_FALSE:
        return False
    return _order_result(cmp_result, op_code)


@register_jitable(**JIT_OPTIONS)
def _stringdtype_unicode_affix(data, index, allocator, pattern,
                               pattern_length, pattern_size, start, end,
                               suffix):
    if suffix:
        return stringdtype_endswith_unicode_data(
            data, index, allocator, pattern, pattern_length, pattern_size,
            start, end)
    return stringdtype_startswith_unicode_data(
        data, index, allocator, pattern, pattern_length, pattern_size, start,
        end)


@register_jitable(**JIT_OPTIONS)
def _stringdtype_utf8_affix(data, index, allocator, pattern_data,
                            pattern_size, start, end, suffix):
    if suffix:
        return stringdtype_endswith_utf8_data(
            data, index, allocator, pattern_data, pattern_size, start, end)
    return stringdtype_startswith_utf8_data(
        data, index, allocator, pattern_data, pattern_size, start, end)


@register_jitable(**JIT_OPTIONS)
def _utf8_stringdtype_affix(value_data, start_byte, end_byte, start_index,
                            pattern_data, pattern_index, allocator, suffix):
    if suffix:
        return utf8_endswith_stringdtype_sliced_data(
            value_data, start_byte, end_byte, start_index, pattern_data,
            pattern_index, allocator)
    return utf8_startswith_stringdtype_sliced_data(
        value_data, start_byte, end_byte, start_index, pattern_data,
        pattern_index, allocator)


@register_jitable(**JIT_OPTIONS)
def _stringdtype_unicode_search(data, index, allocator, pattern,
                                pattern_length, pattern_size, start, end,
                                search_op):
    if search_op == 0:
        return stringdtype_find_unicode_data(
            data, index, allocator, pattern, pattern_length, pattern_size,
            start, end)
    if search_op == 1:
        return stringdtype_rfind_unicode_data(
            data, index, allocator, pattern, pattern_length, pattern_size,
            start, end)
    return stringdtype_count_unicode_data(
        data, index, allocator, pattern, pattern_length, pattern_size, start,
        end)


@register_jitable(**JIT_OPTIONS)
def _stringdtype_utf8_search(data, index, allocator, pattern_data,
                             pattern_size, start, end, search_op):
    if search_op == 0:
        return stringdtype_find_utf8_data(
            data, index, allocator, pattern_data, pattern_size, start, end)
    if search_op == 1:
        return stringdtype_rfind_utf8_data(
            data, index, allocator, pattern_data, pattern_size, start, end)
    return stringdtype_count_utf8_data(
        data, index, allocator, pattern_data, pattern_size, start, end)


@register_jitable(**JIT_OPTIONS)
def _utf8_stringdtype_search(value_data, start_byte, end_byte, start_index,
                             end_index, start_offset, pattern_data,
                             pattern_index, allocator, search_op):
    if search_op == 0:
        return utf8_find_stringdtype_sliced_data(
            value_data, start_byte, end_byte, start_index, end_index,
            start_offset, pattern_data, pattern_index, allocator)
    if search_op == 1:
        return utf8_rfind_stringdtype_sliced_data(
            value_data, start_byte, end_byte, start_index, end_index,
            start_offset, pattern_data, pattern_index, allocator)
    return utf8_count_stringdtype_sliced_data(
        value_data, start_byte, end_byte, start_index, end_index,
        start_offset, pattern_data, pattern_index, allocator)


def _overload_equal(left, right, invert):
    left_stringdtype = is_stringdtype_array_type(left)
    right_stringdtype = is_stringdtype_array_type(right)
    if left_stringdtype or right_stringdtype:
        use_na = _has_stringdtype_na(left, right)
        if use_na and left_stringdtype and right_stringdtype:
            if not _compatible_stringdtype_na(left, right):
                def impl(left, right):
                    raise TypeError(
                        'Cannot find a compatible null string value')

                return impl

        if left_stringdtype and _is_unicode_scalar_like(right):
            _validate_stringdtype_array(left)
            left_na_kind = _stringdtype_na_kind(left)
            if left.ndim == 0:
                def impl(left, right):
                    right_value = _unicode_scalar_value(right)
                    if not stringdtype_unicode_valid(right_value):
                        raise TypeError('Invalid unicode code point found')
                    right_parts = stringdtype_unicode_parts(right_value)
                    allocator = stringdtype_acquire_allocator(left)
                    if use_na:
                        left_na = stringdtype_na_name(left)
                        if invert:
                            result = stringdtype_not_equal_unicode_na_data(
                                stringdtype_data_ptr(left), 0, allocator,
                                left_na_kind, left_na[0], left_na[1],
                                right_value, right_parts[0], right_parts[1],
                                False)
                        else:
                            result = stringdtype_equal_unicode_na_data(
                                stringdtype_data_ptr(left), 0, allocator,
                                left_na_kind, left_na[0], left_na[1],
                                right_value, right_parts[0], right_parts[1],
                                False)
                    elif right_parts[1] > _PACKED_STRING_SIZE:
                        right_span = stringdtype_unicode_utf8_span(
                            right_value, right_parts[0], right_parts[1])
                        result = stringdtype_equal_utf8_data(
                            stringdtype_data_ptr(left), 0, allocator,
                            right_span[0], right_span[1])
                        stringdtype_free_utf8_span(right_span[0],
                                                   right_span[2])
                    else:
                        result = stringdtype_equal_unicode_data(
                            stringdtype_data_ptr(left), 0, allocator,
                            right_value, right_parts[0], right_parts[1])
                    stringdtype_release_allocator(allocator)
                    return result if use_na else (
                        not result if invert else result)

                return impl

            def impl(left, right):
                right_value = _unicode_scalar_value(right)
                if not stringdtype_unicode_valid(right_value):
                    raise TypeError('Invalid unicode code point found')
                right_parts = stringdtype_unicode_parts(right_value)
                result = np.empty(left.size, np.bool_)
                if left.size == 0:
                    return ~result if invert else result
                allocator = stringdtype_acquire_allocator(left)
                data = stringdtype_data_ptr(left)
                if use_na:
                    left_na = stringdtype_na_name(left)
                    for i in range(left.size):
                        if invert:
                            result[i] = stringdtype_not_equal_unicode_na_data(
                                data, i, allocator, left_na_kind, left_na[0],
                                left_na[1], right_value, right_parts[0],
                                right_parts[1], False)
                        else:
                            result[i] = stringdtype_equal_unicode_na_data(
                                data, i, allocator, left_na_kind, left_na[0],
                                left_na[1], right_value, right_parts[0],
                                right_parts[1], False)
                elif right_parts[1] > _PACKED_STRING_SIZE:
                    right_span = stringdtype_unicode_utf8_span(
                        right_value, right_parts[0], right_parts[1])
                    for i in range(left.size):
                        result[i] = stringdtype_equal_utf8_data(
                            data, i, allocator, right_span[0],
                            right_span[1])
                    stringdtype_free_utf8_span(right_span[0], right_span[2])
                else:
                    for i in range(left.size):
                        result[i] = stringdtype_equal_unicode_data(
                            data, i, allocator, right_value, right_parts[0],
                            right_parts[1])
                stringdtype_release_allocator(allocator)
                return result if use_na else (~result if invert else result)

            return impl

        if _is_unicode_scalar_like(left) and right_stringdtype:
            _validate_stringdtype_array(right)
            right_na_kind = _stringdtype_na_kind(right)
            if right.ndim == 0:
                def impl(left, right):
                    left_value = _unicode_scalar_value(left)
                    if not stringdtype_unicode_valid(left_value):
                        raise TypeError('Invalid unicode code point found')
                    left_parts = stringdtype_unicode_parts(left_value)
                    allocator = stringdtype_acquire_allocator(right)
                    if use_na:
                        right_na = stringdtype_na_name(right)
                        if invert:
                            result = stringdtype_not_equal_unicode_na_data(
                                stringdtype_data_ptr(right), 0, allocator,
                                right_na_kind, right_na[0], right_na[1],
                                left_value, left_parts[0], left_parts[1],
                                True)
                        else:
                            result = stringdtype_equal_unicode_na_data(
                                stringdtype_data_ptr(right), 0, allocator,
                                right_na_kind, right_na[0], right_na[1],
                                left_value, left_parts[0], left_parts[1],
                                True)
                    elif left_parts[1] > _PACKED_STRING_SIZE:
                        left_span = stringdtype_unicode_utf8_span(
                            left_value, left_parts[0], left_parts[1])
                        result = stringdtype_equal_utf8_data(
                            stringdtype_data_ptr(right), 0, allocator,
                            left_span[0], left_span[1])
                        stringdtype_free_utf8_span(left_span[0],
                                                   left_span[2])
                    else:
                        result = stringdtype_equal_unicode_data(
                            stringdtype_data_ptr(right), 0, allocator,
                            left_value, left_parts[0], left_parts[1])
                    stringdtype_release_allocator(allocator)
                    return result if use_na else (
                        not result if invert else result)

                return impl

            def impl(left, right):
                left_value = _unicode_scalar_value(left)
                if not stringdtype_unicode_valid(left_value):
                    raise TypeError('Invalid unicode code point found')
                left_parts = stringdtype_unicode_parts(left_value)
                result = np.empty(right.size, np.bool_)
                if right.size == 0:
                    return ~result if invert else result
                allocator = stringdtype_acquire_allocator(right)
                data = stringdtype_data_ptr(right)
                if use_na:
                    right_na = stringdtype_na_name(right)
                    for i in range(right.size):
                        if invert:
                            result[i] = stringdtype_not_equal_unicode_na_data(
                                data, i, allocator, right_na_kind,
                                right_na[0], right_na[1], left_value,
                                left_parts[0], left_parts[1], True)
                        else:
                            result[i] = stringdtype_equal_unicode_na_data(
                                data, i, allocator, right_na_kind,
                                right_na[0], right_na[1], left_value,
                                left_parts[0], left_parts[1], True)
                elif left_parts[1] > _PACKED_STRING_SIZE:
                    left_span = stringdtype_unicode_utf8_span(
                        left_value, left_parts[0], left_parts[1])
                    for i in range(right.size):
                        result[i] = stringdtype_equal_utf8_data(
                            data, i, allocator, left_span[0], left_span[1])
                    stringdtype_free_utf8_span(left_span[0], left_span[2])
                else:
                    for i in range(right.size):
                        result[i] = stringdtype_equal_unicode_data(
                            data, i, allocator, left_value, left_parts[0],
                            left_parts[1])
                stringdtype_release_allocator(allocator)
                return result if use_na else (~result if invert else result)

            return impl

        if left_stringdtype and _is_unicode_array(right) and right.ndim == 1:
            _validate_stringdtype_array(left)
            _validate_unicode_array(right)
            left_scalar = left.ndim == 0
            left_na_kind = _stringdtype_na_kind(left)

            def impl(left, right):
                if not left_scalar and left.size != right.size:
                    raise ValueError('shape mismatch: objects cannot be '
                                     'broadcast to a single shape')
                size = right.size if left_scalar else left.size
                result = np.empty(size, np.bool_)
                if size == 0:
                    return ~result if invert else result
                allocator = stringdtype_acquire_allocator(left)
                data = stringdtype_data_ptr(left)
                if use_na:
                    left_na = stringdtype_na_name(left)
                for i in range(size):
                    right_value = _unicode_scalar_value(right[i])
                    if not stringdtype_unicode_valid(right_value):
                        stringdtype_release_allocator(allocator)
                        raise TypeError('Invalid unicode code point found')
                    right_parts = stringdtype_unicode_parts(right_value)
                    if use_na:
                        if invert:
                            result[i] = stringdtype_not_equal_unicode_na_data(
                                data, 0 if left_scalar else i, allocator,
                                left_na_kind, left_na[0], left_na[1],
                                right_value, right_parts[0], right_parts[1],
                                False)
                        else:
                            result[i] = stringdtype_equal_unicode_na_data(
                                data, 0 if left_scalar else i, allocator,
                                left_na_kind, left_na[0], left_na[1],
                                right_value, right_parts[0], right_parts[1],
                                False)
                    else:
                        result[i] = stringdtype_equal_unicode_data(
                            data, 0 if left_scalar else i, allocator,
                            right_value, right_parts[0], right_parts[1])
                stringdtype_release_allocator(allocator)
                return result if use_na else (~result if invert else result)

            return impl

        if _is_unicode_array(left) and left.ndim == 1 and right_stringdtype:
            _validate_unicode_array(left)
            _validate_stringdtype_array(right)
            right_scalar = right.ndim == 0
            right_na_kind = _stringdtype_na_kind(right)

            def impl(left, right):
                if not right_scalar and left.size != right.size:
                    raise ValueError('shape mismatch: objects cannot be '
                                     'broadcast to a single shape')
                size = left.size
                result = np.empty(size, np.bool_)
                if size == 0:
                    return ~result if invert else result
                allocator = stringdtype_acquire_allocator(right)
                data = stringdtype_data_ptr(right)
                if use_na:
                    right_na = stringdtype_na_name(right)
                for i in range(size):
                    left_value = _unicode_scalar_value(left[i])
                    if not stringdtype_unicode_valid(left_value):
                        stringdtype_release_allocator(allocator)
                        raise TypeError('Invalid unicode code point found')
                    left_parts = stringdtype_unicode_parts(left_value)
                    if use_na:
                        if invert:
                            result[i] = stringdtype_not_equal_unicode_na_data(
                                data, 0 if right_scalar else i, allocator,
                                right_na_kind, right_na[0], right_na[1],
                                left_value, left_parts[0], left_parts[1],
                                True)
                        else:
                            result[i] = stringdtype_equal_unicode_na_data(
                                data, 0 if right_scalar else i, allocator,
                                right_na_kind, right_na[0], right_na[1],
                                left_value, left_parts[0], left_parts[1],
                                True)
                    else:
                        result[i] = stringdtype_equal_unicode_data(
                            data, 0 if right_scalar else i, allocator,
                            left_value, left_parts[0], left_parts[1])
                stringdtype_release_allocator(allocator)
                return result if use_na else (~result if invert else result)

            return impl

        if left_stringdtype and _is_none(right):
            _validate_stringdtype_array(left)
            if left.ndim == 0:
                def impl(left, right):
                    return True if invert else False

                return impl

            def impl(left, right):
                result = np.empty(left.size, np.bool_)
                for i in range(left.size):
                    result[i] = True if invert else False
                return result

            return impl

        if _is_none(left) and right_stringdtype:
            _validate_stringdtype_array(right)
            if right.ndim == 0:
                def impl(left, right):
                    return True if invert else False

                return impl

            def impl(left, right):
                result = np.empty(right.size, np.bool_)
                for i in range(right.size):
                    result[i] = True if invert else False
                return result

            return impl

        if not left_stringdtype or not right_stringdtype:
            raise NumbaValueError('StringDType comparisons currently require '
                                  'two StringDType arrays')
        _validate_stringdtype_array(left)
        _validate_stringdtype_array(right)
        left_na_kind = _stringdtype_na_kind(left)
        right_na_kind = _stringdtype_na_kind(right)

        if left.ndim == 0 and right.ndim == 0:
            def impl(left, right):
                allocators = stringdtype_acquire_allocators(left, right)
                if use_na:
                    left_na = stringdtype_na_name(left)
                    right_na = stringdtype_na_name(right)
                    if invert:
                        equal_result = stringdtype_not_equal_na_data(
                            stringdtype_data_ptr(left), 0, allocators[0],
                            left_na_kind, left_na[0], left_na[1],
                            stringdtype_data_ptr(right), 0, allocators[1],
                            right_na_kind, right_na[0], right_na[1],
                        )
                    else:
                        equal_result = stringdtype_equal_na_data(
                            stringdtype_data_ptr(left), 0, allocators[0],
                            left_na_kind, left_na[0], left_na[1],
                            stringdtype_data_ptr(right), 0, allocators[1],
                            right_na_kind, right_na[0], right_na[1],
                        )
                else:
                    equal_result = stringdtype_equal_data(
                        stringdtype_data_ptr(left), 0, allocators[0],
                        stringdtype_data_ptr(right), 0, allocators[1],
                    )
                stringdtype_release_allocators(allocators)
                return equal_result if use_na \
                    else (not equal_result if invert else equal_result)

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
            if use_na:
                left_na = stringdtype_na_name(left)
                right_na = stringdtype_na_name(right)
            for i in range(size):
                if use_na:
                    if invert:
                        result[i] = stringdtype_not_equal_na_data(
                            left_data, 0 if left_scalar else i,
                            left_allocator, left_na_kind, left_na[0],
                            left_na[1], right_data,
                            0 if right_scalar else i, right_allocator,
                            right_na_kind, right_na[0], right_na[1],
                        )
                    else:
                        result[i] = stringdtype_equal_na_data(
                            left_data, 0 if left_scalar else i,
                            left_allocator, left_na_kind, left_na[0],
                            left_na[1], right_data,
                            0 if right_scalar else i, right_allocator,
                            right_na_kind, right_na[0], right_na[1],
                        )
                else:
                    result[i] = stringdtype_equal_data(
                        left_data, 0 if left_scalar else i, left_allocator,
                        right_data, 0 if right_scalar else i,
                        right_allocator,
                    )
            stringdtype_release_allocators(allocators)
            return result if use_na else (~result if invert else result)

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
        use_na = _has_stringdtype_na(left, right)
        if use_na and left_stringdtype and right_stringdtype:
            if not _compatible_stringdtype_na(left, right):
                def impl(left, right):
                    raise TypeError(
                        'Cannot find a compatible null string value')

                return impl

        if op == 'greater':
            op_code = 0
        elif op == 'greater_equal':
            op_code = 1
        elif op == 'less':
            op_code = 2
        else:
            op_code = 3

        if left_stringdtype and _is_unicode_scalar_like(right):
            _validate_stringdtype_array(left)
            left_na_kind = _stringdtype_na_kind(left)
            if left.ndim == 0:
                def impl(left, right):
                    right_value = _unicode_scalar_value(right)
                    if not stringdtype_unicode_valid(right_value):
                        raise TypeError('Invalid unicode code point found')
                    right_parts = stringdtype_unicode_parts(right_value)
                    allocator = stringdtype_acquire_allocator(left)
                    if use_na:
                        left_na = stringdtype_na_name(left)
                        cmp_result = stringdtype_compare_unicode_na_data(
                            stringdtype_data_ptr(left), 0, allocator,
                            left_na_kind, left_na[0], left_na[1],
                            right_value, right_parts[0], right_parts[1],
                            False)
                    elif right_parts[1] > _PACKED_STRING_SIZE:
                        right_span = stringdtype_unicode_utf8_span(
                            right_value, right_parts[0], right_parts[1])
                        cmp_result = stringdtype_compare_utf8_data(
                            stringdtype_data_ptr(left), 0, allocator,
                            right_span[0], right_span[1])
                        stringdtype_free_utf8_span(right_span[0],
                                                   right_span[2])
                    else:
                        cmp_result = stringdtype_compare_unicode_data(
                            stringdtype_data_ptr(left), 0, allocator,
                            right_value, right_parts[0], right_parts[1])
                    stringdtype_release_allocator(allocator)
                    return _stringdtype_order_result(cmp_result, op_code) \
                        if use_na else _order_result(cmp_result, op_code)

                return impl

            def impl(left, right):
                right_value = _unicode_scalar_value(right)
                if not stringdtype_unicode_valid(right_value):
                    raise TypeError('Invalid unicode code point found')
                right_parts = stringdtype_unicode_parts(right_value)
                result = np.empty(left.size, np.bool_)
                if left.size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(left)
                data = stringdtype_data_ptr(left)
                bad_order = False
                if use_na:
                    left_na = stringdtype_na_name(left)
                    for i in range(left.size):
                        cmp_result = stringdtype_compare_unicode_na_data(
                            data, i, allocator, left_na_kind, left_na[0],
                            left_na[1], right_value, right_parts[0],
                            right_parts[1], False)
                        if cmp_result == STRINGDTYPE_ORDER_ERROR:
                            bad_order = True
                            break
                        if cmp_result == STRINGDTYPE_ORDER_FALSE:
                            result[i] = False
                        else:
                            result[i] = _order_result(cmp_result, op_code)
                elif right_parts[1] > _PACKED_STRING_SIZE:
                    right_span = stringdtype_unicode_utf8_span(
                        right_value, right_parts[0], right_parts[1])
                    for i in range(left.size):
                        cmp_result = stringdtype_compare_utf8_data(
                            data, i, allocator, right_span[0],
                            right_span[1])
                        result[i] = _order_result(cmp_result, op_code)
                    stringdtype_free_utf8_span(right_span[0], right_span[2])
                else:
                    for i in range(left.size):
                        cmp_result = stringdtype_compare_unicode_data(
                            data, i, allocator, right_value, right_parts[0],
                            right_parts[1])
                        result[i] = _order_result(cmp_result, op_code)
                stringdtype_release_allocator(allocator)
                if use_na and bad_order:
                    raise ValueError(
                        'StringDType ordering is not supported for this null '
                        'value')
                return result

            return impl

        if _is_unicode_scalar_like(left) and right_stringdtype:
            _validate_stringdtype_array(right)
            right_na_kind = _stringdtype_na_kind(right)
            if right.ndim == 0:
                def impl(left, right):
                    left_value = _unicode_scalar_value(left)
                    if not stringdtype_unicode_valid(left_value):
                        raise TypeError('Invalid unicode code point found')
                    left_parts = stringdtype_unicode_parts(left_value)
                    allocator = stringdtype_acquire_allocator(right)
                    if use_na:
                        right_na = stringdtype_na_name(right)
                        cmp_result = stringdtype_compare_unicode_na_data(
                            stringdtype_data_ptr(right), 0, allocator,
                            right_na_kind, right_na[0], right_na[1],
                            left_value, left_parts[0], left_parts[1], True)
                        if cmp_result != STRINGDTYPE_ORDER_ERROR \
                                and cmp_result != STRINGDTYPE_ORDER_FALSE:
                            cmp_result = -cmp_result
                    elif left_parts[1] > _PACKED_STRING_SIZE:
                        left_span = stringdtype_unicode_utf8_span(
                            left_value, left_parts[0], left_parts[1])
                        cmp_result = -stringdtype_compare_utf8_data(
                            stringdtype_data_ptr(right), 0, allocator,
                            left_span[0], left_span[1])
                        stringdtype_free_utf8_span(left_span[0],
                                                   left_span[2])
                    else:
                        cmp_result = -stringdtype_compare_unicode_data(
                            stringdtype_data_ptr(right), 0, allocator,
                            left_value, left_parts[0], left_parts[1])
                    stringdtype_release_allocator(allocator)
                    return _stringdtype_order_result(cmp_result, op_code) \
                        if use_na else _order_result(cmp_result, op_code)

                return impl

            def impl(left, right):
                left_value = _unicode_scalar_value(left)
                if not stringdtype_unicode_valid(left_value):
                    raise TypeError('Invalid unicode code point found')
                left_parts = stringdtype_unicode_parts(left_value)
                result = np.empty(right.size, np.bool_)
                if right.size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(right)
                data = stringdtype_data_ptr(right)
                bad_order = False
                if use_na:
                    right_na = stringdtype_na_name(right)
                    for i in range(right.size):
                        cmp_result = stringdtype_compare_unicode_na_data(
                            data, i, allocator, right_na_kind, right_na[0],
                            right_na[1], left_value, left_parts[0],
                            left_parts[1], True)
                        if cmp_result == STRINGDTYPE_ORDER_ERROR:
                            bad_order = True
                            break
                        if cmp_result == STRINGDTYPE_ORDER_FALSE:
                            result[i] = False
                        else:
                            result[i] = _order_result(-cmp_result, op_code)
                elif left_parts[1] > _PACKED_STRING_SIZE:
                    left_span = stringdtype_unicode_utf8_span(
                        left_value, left_parts[0], left_parts[1])
                    for i in range(right.size):
                        cmp_result = -stringdtype_compare_utf8_data(
                            data, i, allocator, left_span[0], left_span[1])
                        result[i] = _order_result(cmp_result, op_code)
                    stringdtype_free_utf8_span(left_span[0], left_span[2])
                else:
                    for i in range(right.size):
                        cmp_result = -stringdtype_compare_unicode_data(
                            data, i, allocator, left_value, left_parts[0],
                            left_parts[1])
                        result[i] = _order_result(cmp_result, op_code)
                stringdtype_release_allocator(allocator)
                return result

            return impl

        if left_stringdtype and _is_unicode_array(right) and right.ndim == 1:
            _validate_stringdtype_array(left)
            _validate_unicode_array(right)
            left_scalar = left.ndim == 0
            left_na_kind = _stringdtype_na_kind(left)

            def impl(left, right):
                if not left_scalar and left.size != right.size:
                    raise ValueError('shape mismatch: objects cannot be '
                                     'broadcast to a single shape')
                size = right.size if left_scalar else left.size
                result = np.empty(size, np.bool_)
                if size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(left)
                data = stringdtype_data_ptr(left)
                bad_order = False
                if use_na:
                    left_na = stringdtype_na_name(left)
                for i in range(size):
                    right_value = _unicode_scalar_value(right[i])
                    if not stringdtype_unicode_valid(right_value):
                        stringdtype_release_allocator(allocator)
                        raise TypeError('Invalid unicode code point found')
                    right_parts = stringdtype_unicode_parts(right_value)
                    if use_na:
                        cmp_result = stringdtype_compare_unicode_na_data(
                            data, 0 if left_scalar else i, allocator,
                            left_na_kind, left_na[0], left_na[1],
                            right_value, right_parts[0], right_parts[1],
                            False)
                        if cmp_result == STRINGDTYPE_ORDER_ERROR:
                            bad_order = True
                            break
                        if cmp_result == STRINGDTYPE_ORDER_FALSE:
                            result[i] = False
                        else:
                            result[i] = _order_result(cmp_result, op_code)
                    else:
                        cmp_result = stringdtype_compare_unicode_data(
                            data, 0 if left_scalar else i, allocator,
                            right_value, right_parts[0], right_parts[1])
                        result[i] = _order_result(cmp_result, op_code)
                stringdtype_release_allocator(allocator)
                if use_na and bad_order:
                    raise ValueError(
                        'StringDType ordering is not supported for this null '
                        'value')
                return result

            return impl

        if _is_unicode_array(left) and left.ndim == 1 and right_stringdtype:
            _validate_unicode_array(left)
            _validate_stringdtype_array(right)
            right_scalar = right.ndim == 0
            right_na_kind = _stringdtype_na_kind(right)

            def impl(left, right):
                if not right_scalar and left.size != right.size:
                    raise ValueError('shape mismatch: objects cannot be '
                                     'broadcast to a single shape')
                size = left.size
                result = np.empty(size, np.bool_)
                if size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(right)
                data = stringdtype_data_ptr(right)
                bad_order = False
                if use_na:
                    right_na = stringdtype_na_name(right)
                for i in range(size):
                    left_value = _unicode_scalar_value(left[i])
                    if not stringdtype_unicode_valid(left_value):
                        stringdtype_release_allocator(allocator)
                        raise TypeError('Invalid unicode code point found')
                    left_parts = stringdtype_unicode_parts(left_value)
                    if use_na:
                        cmp_result = stringdtype_compare_unicode_na_data(
                            data, 0 if right_scalar else i, allocator,
                            right_na_kind, right_na[0], right_na[1],
                            left_value, left_parts[0], left_parts[1], True)
                        if cmp_result == STRINGDTYPE_ORDER_ERROR:
                            bad_order = True
                            break
                        if cmp_result == STRINGDTYPE_ORDER_FALSE:
                            result[i] = False
                        else:
                            result[i] = _order_result(-cmp_result, op_code)
                    else:
                        cmp_result = -stringdtype_compare_unicode_data(
                            data, 0 if right_scalar else i, allocator,
                            left_value, left_parts[0], left_parts[1])
                        result[i] = _order_result(cmp_result, op_code)
                stringdtype_release_allocator(allocator)
                if use_na and bad_order:
                    raise ValueError(
                        'StringDType ordering is not supported for this null '
                        'value')
                return result

            return impl

        if not left_stringdtype or not right_stringdtype:
            raise NumbaValueError('StringDType comparisons currently require '
                                  'two StringDType arrays')
        _validate_stringdtype_array(left)
        _validate_stringdtype_array(right)
        left_na_kind = _stringdtype_na_kind(left)
        right_na_kind = _stringdtype_na_kind(right)

        if left.ndim == 0 and right.ndim == 0:
            def impl(left, right):
                allocators = stringdtype_acquire_allocators(left, right)
                if use_na:
                    left_na = stringdtype_na_name(left)
                    right_na = stringdtype_na_name(right)
                    cmp_result = stringdtype_compare_na_data(
                        stringdtype_data_ptr(left), 0, allocators[0],
                        left_na_kind, left_na[0], left_na[1],
                        stringdtype_data_ptr(right), 0, allocators[1],
                        right_na_kind, right_na[0], right_na[1],
                    )
                else:
                    cmp_result = stringdtype_compare_data(
                        stringdtype_data_ptr(left), 0, allocators[0],
                        stringdtype_data_ptr(right), 0, allocators[1],
                    )
                stringdtype_release_allocators(allocators)
                return _stringdtype_order_result(cmp_result, op_code) \
                    if use_na else _order_result(cmp_result, op_code)

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
            bad_order = False
            if use_na:
                left_na = stringdtype_na_name(left)
                right_na = stringdtype_na_name(right)
            for i in range(size):
                if use_na:
                    cmp_result = stringdtype_compare_na_data(
                        left_data, 0 if left_scalar else i, left_allocator,
                        left_na_kind, left_na[0], left_na[1],
                        right_data, 0 if right_scalar else i,
                        right_allocator, right_na_kind, right_na[0],
                        right_na[1],
                    )
                    if cmp_result == STRINGDTYPE_ORDER_ERROR:
                        bad_order = True
                        break
                    if cmp_result == STRINGDTYPE_ORDER_FALSE:
                        result[i] = False
                    else:
                        result[i] = _order_result(cmp_result, op_code)
                else:
                    cmp_result = stringdtype_compare_data(
                        left_data, 0 if left_scalar else i, left_allocator,
                        right_data, 0 if right_scalar else i,
                        right_allocator,
                    )
                    result[i] = _order_result(cmp_result, op_code)
            stringdtype_release_allocators(allocators)
            if use_na and bad_order:
                raise ValueError(
                    'StringDType ordering is not supported for this null value')
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
        if _has_stringdtype_na(value, pattern):
            _reject_stringdtype_na('prefix/suffix operations')

        s, e = ensure_slice(start, end)

        if value_stringdtype and _is_unicode_scalar_like(pattern):
            _validate_stringdtype_array(value)
            if value.ndim == 0:
                def impl(value, pattern, start=0, end=None):
                    pattern_value = _unicode_scalar_value(pattern)
                    if not stringdtype_unicode_valid(pattern_value):
                        raise TypeError('Invalid unicode code point found')
                    pattern_parts = stringdtype_unicode_parts(pattern_value)
                    start = start or s
                    end = e if end is None else end
                    allocator = stringdtype_acquire_allocator(value)
                    data = stringdtype_data_ptr(value)
                    if pattern_parts[1] > _PACKED_STRING_SIZE:
                        pattern_span = stringdtype_unicode_utf8_span(
                            pattern_value, pattern_parts[0],
                            pattern_parts[1])
                        result = _stringdtype_utf8_affix(
                            data, 0, allocator, pattern_span[0],
                            pattern_span[1], start, end, suffix)
                        stringdtype_free_utf8_span(pattern_span[0],
                                                   pattern_span[2])
                    else:
                        result = _stringdtype_unicode_affix(
                            data, 0, allocator, pattern_value,
                            pattern_parts[0], pattern_parts[1], start, end,
                            suffix)
                    stringdtype_release_allocator(allocator)
                    return result

                return impl

            def impl(value, pattern, start=0, end=None):
                pattern_value = _unicode_scalar_value(pattern)
                if not stringdtype_unicode_valid(pattern_value):
                    raise TypeError('Invalid unicode code point found')
                pattern_parts = stringdtype_unicode_parts(pattern_value)
                start = start or s
                end = e if end is None else end
                result = np.empty(value.size, np.bool_)
                if value.size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(value)
                data = stringdtype_data_ptr(value)
                if pattern_parts[1] > _PACKED_STRING_SIZE:
                    pattern_span = stringdtype_unicode_utf8_span(
                        pattern_value, pattern_parts[0], pattern_parts[1])
                    for i in range(value.size):
                        result[i] = _stringdtype_utf8_affix(
                            data, i, allocator, pattern_span[0],
                            pattern_span[1], start, end, suffix)
                    stringdtype_free_utf8_span(pattern_span[0],
                                               pattern_span[2])
                else:
                    for i in range(value.size):
                        result[i] = _stringdtype_unicode_affix(
                            data, i, allocator, pattern_value,
                            pattern_parts[0], pattern_parts[1], start, end,
                            suffix)
                stringdtype_release_allocator(allocator)
                return result

            return impl

        if _is_unicode_scalar_like(value) and pattern_stringdtype:
            _validate_stringdtype_array(pattern)
            if pattern.ndim == 0:
                def impl(value, pattern, start=0, end=None):
                    value_value = _unicode_scalar_value(value)
                    if not stringdtype_unicode_valid(value_value):
                        raise TypeError('Invalid unicode code point found')
                    value_parts = stringdtype_unicode_parts(value_value)
                    start = start or s
                    end = e if end is None else end
                    value_span = stringdtype_unicode_utf8_span(
                        value_value, value_parts[0], value_parts[1])
                    slice_parts = stringdtype_utf8_slice(
                        value_span[0], value_span[1], start, end)
                    allocator = stringdtype_acquire_allocator(pattern)
                    data = stringdtype_data_ptr(pattern)
                    result = _utf8_stringdtype_affix(
                        value_span[0], slice_parts[0], slice_parts[1],
                        slice_parts[2], data, 0, allocator, suffix)
                    stringdtype_release_allocator(allocator)
                    stringdtype_free_utf8_span(value_span[0], value_span[2])
                    return result

                return impl

            def impl(value, pattern, start=0, end=None):
                value_value = _unicode_scalar_value(value)
                if not stringdtype_unicode_valid(value_value):
                    raise TypeError('Invalid unicode code point found')
                value_parts = stringdtype_unicode_parts(value_value)
                start = start or s
                end = e if end is None else end
                result = np.empty(pattern.size, np.bool_)
                if pattern.size == 0:
                    return result
                value_span = stringdtype_unicode_utf8_span(
                    value_value, value_parts[0], value_parts[1])
                slice_parts = stringdtype_utf8_slice(
                    value_span[0], value_span[1], start, end)
                allocator = stringdtype_acquire_allocator(pattern)
                data = stringdtype_data_ptr(pattern)
                for i in range(pattern.size):
                    result[i] = _utf8_stringdtype_affix(
                        value_span[0], slice_parts[0], slice_parts[1],
                        slice_parts[2], data, i, allocator, suffix)
                stringdtype_release_allocator(allocator)
                stringdtype_free_utf8_span(value_span[0], value_span[2])
                return result

            return impl

        if value_stringdtype and _is_unicode_array(pattern) \
                and pattern.ndim == 1:
            _validate_stringdtype_array(value)
            _validate_unicode_array(pattern)
            value_scalar = value.ndim == 0

            def impl(value, pattern, start=0, end=None):
                start = start or s
                end = e if end is None else end
                if not value_scalar and value.size != pattern.size:
                    raise ValueError('shape mismatch: objects cannot be '
                                     'broadcast to a single shape')
                size = pattern.size if value_scalar else value.size
                result = np.empty(size, np.bool_)
                if size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(value)
                data = stringdtype_data_ptr(value)
                for i in range(size):
                    pattern_value = _unicode_scalar_value(pattern[i])
                    if not stringdtype_unicode_valid(pattern_value):
                        stringdtype_release_allocator(allocator)
                        raise TypeError('Invalid unicode code point found')
                    pattern_parts = stringdtype_unicode_parts(pattern_value)
                    result[i] = _stringdtype_unicode_affix(
                        data, 0 if value_scalar else i, allocator,
                        pattern_value, pattern_parts[0], pattern_parts[1],
                        start, end, suffix)
                stringdtype_release_allocator(allocator)
                return result

            return impl

        if _is_unicode_array(value) and value.ndim == 1 \
                and pattern_stringdtype:
            _validate_unicode_array(value)
            _validate_stringdtype_array(pattern)
            pattern_scalar = pattern.ndim == 0

            def impl(value, pattern, start=0, end=None):
                start = start or s
                end = e if end is None else end
                if not pattern_scalar and value.size != pattern.size:
                    raise ValueError('shape mismatch: objects cannot be '
                                     'broadcast to a single shape')
                size = value.size
                result = np.empty(size, np.bool_)
                if size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(pattern)
                data = stringdtype_data_ptr(pattern)
                for i in range(size):
                    value_value = _unicode_scalar_value(value[i])
                    if not stringdtype_unicode_valid(value_value):
                        stringdtype_release_allocator(allocator)
                        raise TypeError('Invalid unicode code point found')
                    value_parts = stringdtype_unicode_parts(value_value)
                    value_span = stringdtype_unicode_utf8_span(
                        value_value, value_parts[0], value_parts[1])
                    slice_parts = stringdtype_utf8_slice(
                        value_span[0], value_span[1], start, end)
                    result[i] = _utf8_stringdtype_affix(
                        value_span[0], slice_parts[0], slice_parts[1],
                        slice_parts[2], data, 0 if pattern_scalar else i,
                        allocator, suffix)
                    stringdtype_free_utf8_span(value_span[0], value_span[2])
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
    value_stringdtype = is_stringdtype_array_type(value)
    pattern_stringdtype = is_stringdtype_array_type(pattern)
    if value_stringdtype or pattern_stringdtype:
        if _has_stringdtype_na(value, pattern):
            _reject_stringdtype_na('search operations')

        s, e = ensure_slice(start, end)
        forward = op == 'find' or op == 'index'
        reverse = op == 'rfind' or op == 'rindex'
        raise_not_found = op == 'index' or op == 'rindex'
        search_op = 0 if forward else 1 if reverse else 2

        if value_stringdtype and _is_unicode_scalar_like(pattern):
            _validate_stringdtype_array(value)
            if value.ndim == 0:
                def impl(value, pattern, start=0, end=None):
                    pattern_value = _unicode_scalar_value(pattern)
                    if not stringdtype_unicode_valid(pattern_value):
                        raise TypeError('Invalid unicode code point found')
                    pattern_parts = stringdtype_unicode_parts(pattern_value)
                    start = start or s
                    end = e if end is None else end
                    allocator = stringdtype_acquire_allocator(value)
                    data = stringdtype_data_ptr(value)
                    if pattern_parts[1] > _PACKED_STRING_SIZE:
                        pattern_span = stringdtype_unicode_utf8_span(
                            pattern_value, pattern_parts[0],
                            pattern_parts[1])
                        found = _stringdtype_utf8_search(
                            data, 0, allocator, pattern_span[0],
                            pattern_span[1], start, end, search_op)
                        stringdtype_free_utf8_span(pattern_span[0],
                                                   pattern_span[2])
                    else:
                        found = _stringdtype_unicode_search(
                            data, 0, allocator, pattern_value,
                            pattern_parts[0], pattern_parts[1], start, end,
                            search_op)
                    stringdtype_release_allocator(allocator)
                    if raise_not_found and found < 0:
                        raise ValueError('substring not found')
                    return found

                return impl

            def impl(value, pattern, start=0, end=None):
                pattern_value = _unicode_scalar_value(pattern)
                if not stringdtype_unicode_valid(pattern_value):
                    raise TypeError('Invalid unicode code point found')
                pattern_parts = stringdtype_unicode_parts(pattern_value)
                start = start or s
                end = e if end is None else end
                result = np.empty(value.size, np.int64)
                if value.size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(value)
                data = stringdtype_data_ptr(value)
                not_found = False
                if pattern_parts[1] > _PACKED_STRING_SIZE:
                    pattern_span = stringdtype_unicode_utf8_span(
                        pattern_value, pattern_parts[0], pattern_parts[1])
                    for i in range(value.size):
                        found = _stringdtype_utf8_search(
                            data, i, allocator, pattern_span[0],
                            pattern_span[1], start, end, search_op)
                        if raise_not_found and found < 0:
                            not_found = True
                            break
                        result[i] = found
                    stringdtype_free_utf8_span(pattern_span[0],
                                               pattern_span[2])
                else:
                    for i in range(value.size):
                        found = _stringdtype_unicode_search(
                            data, i, allocator, pattern_value,
                            pattern_parts[0], pattern_parts[1], start, end,
                            search_op)
                        if raise_not_found and found < 0:
                            not_found = True
                            break
                        result[i] = found
                stringdtype_release_allocator(allocator)
                if not_found:
                    raise ValueError('substring not found')
                return result

            return impl

        if _is_unicode_scalar_like(value) and pattern_stringdtype:
            _validate_stringdtype_array(pattern)
            if pattern.ndim == 0:
                def impl(value, pattern, start=0, end=None):
                    value_value = _unicode_scalar_value(value)
                    if not stringdtype_unicode_valid(value_value):
                        raise TypeError('Invalid unicode code point found')
                    value_parts = stringdtype_unicode_parts(value_value)
                    start = start or s
                    end = e if end is None else end
                    value_span = stringdtype_unicode_utf8_span(
                        value_value, value_parts[0], value_parts[1])
                    slice_parts = stringdtype_utf8_search_slice(
                        value_span[0], value_span[1], start, end)
                    allocator = stringdtype_acquire_allocator(pattern)
                    data = stringdtype_data_ptr(pattern)
                    found = _utf8_stringdtype_search(
                        value_span[0], slice_parts[0], slice_parts[1],
                        slice_parts[2], slice_parts[3], slice_parts[4],
                        data, 0, allocator, search_op)
                    stringdtype_release_allocator(allocator)
                    stringdtype_free_utf8_span(value_span[0], value_span[2])
                    if raise_not_found and found < 0:
                        raise ValueError('substring not found')
                    return found

                return impl

            def impl(value, pattern, start=0, end=None):
                value_value = _unicode_scalar_value(value)
                if not stringdtype_unicode_valid(value_value):
                    raise TypeError('Invalid unicode code point found')
                value_parts = stringdtype_unicode_parts(value_value)
                start = start or s
                end = e if end is None else end
                result = np.empty(pattern.size, np.int64)
                if pattern.size == 0:
                    return result
                value_span = stringdtype_unicode_utf8_span(
                    value_value, value_parts[0], value_parts[1])
                slice_parts = stringdtype_utf8_search_slice(
                    value_span[0], value_span[1], start, end)
                allocator = stringdtype_acquire_allocator(pattern)
                data = stringdtype_data_ptr(pattern)
                not_found = False
                for i in range(pattern.size):
                    found = _utf8_stringdtype_search(
                        value_span[0], slice_parts[0], slice_parts[1],
                        slice_parts[2], slice_parts[3], slice_parts[4],
                        data, i, allocator, search_op)
                    if raise_not_found and found < 0:
                        not_found = True
                        break
                    result[i] = found
                stringdtype_release_allocator(allocator)
                stringdtype_free_utf8_span(value_span[0], value_span[2])
                if not_found:
                    raise ValueError('substring not found')
                return result

            return impl

        if value_stringdtype and _is_unicode_array(pattern) \
                and pattern.ndim == 1:
            _validate_stringdtype_array(value)
            _validate_unicode_array(pattern)
            value_scalar = value.ndim == 0

            def impl(value, pattern, start=0, end=None):
                start = start or s
                end = e if end is None else end
                if not value_scalar and value.size != pattern.size:
                    raise ValueError('shape mismatch: objects cannot be '
                                     'broadcast to a single shape')
                size = pattern.size if value_scalar else value.size
                result = np.empty(size, np.int64)
                if size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(value)
                data = stringdtype_data_ptr(value)
                not_found = False
                for i in range(size):
                    pattern_value = _unicode_scalar_value(pattern[i])
                    if not stringdtype_unicode_valid(pattern_value):
                        stringdtype_release_allocator(allocator)
                        raise TypeError('Invalid unicode code point found')
                    pattern_parts = stringdtype_unicode_parts(pattern_value)
                    found = _stringdtype_unicode_search(
                        data, 0 if value_scalar else i, allocator,
                        pattern_value, pattern_parts[0], pattern_parts[1],
                        start, end, search_op)
                    if raise_not_found and found < 0:
                        not_found = True
                        break
                    result[i] = found
                stringdtype_release_allocator(allocator)
                if not_found:
                    raise ValueError('substring not found')
                return result

            return impl

        if _is_unicode_array(value) and value.ndim == 1 \
                and pattern_stringdtype:
            _validate_unicode_array(value)
            _validate_stringdtype_array(pattern)
            pattern_scalar = pattern.ndim == 0

            def impl(value, pattern, start=0, end=None):
                start = start or s
                end = e if end is None else end
                if not pattern_scalar and value.size != pattern.size:
                    raise ValueError('shape mismatch: objects cannot be '
                                     'broadcast to a single shape')
                size = value.size
                result = np.empty(size, np.int64)
                if size == 0:
                    return result
                allocator = stringdtype_acquire_allocator(pattern)
                data = stringdtype_data_ptr(pattern)
                not_found = False
                for i in range(size):
                    value_value = _unicode_scalar_value(value[i])
                    if not stringdtype_unicode_valid(value_value):
                        stringdtype_release_allocator(allocator)
                        raise TypeError('Invalid unicode code point found')
                    value_parts = stringdtype_unicode_parts(value_value)
                    value_span = stringdtype_unicode_utf8_span(
                        value_value, value_parts[0], value_parts[1])
                    slice_parts = stringdtype_utf8_search_slice(
                        value_span[0], value_span[1], start, end)
                    found = _utf8_stringdtype_search(
                        value_span[0], slice_parts[0], slice_parts[1],
                        slice_parts[2], slice_parts[3], slice_parts[4],
                        data, 0 if pattern_scalar else i, allocator,
                        search_op)
                    stringdtype_free_utf8_span(value_span[0], value_span[2])
                    if raise_not_found and found < 0:
                        not_found = True
                        break
                    result[i] = found
                stringdtype_release_allocator(allocator)
                if not_found:
                    raise ValueError('substring not found')
                return result

            return impl

        if not value_stringdtype or not pattern_stringdtype:
            raise NumbaValueError('StringDType search operations currently '
                                  'require two StringDType arrays')
        _validate_stringdtype_array(value)
        _validate_stringdtype_array(pattern)

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
    na_kind = value.dtype.na_kind

    if value.ndim == 0:
        if na_kind == 0:
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
            allocator = stringdtype_acquire_allocator(value)
            data = stringdtype_data_ptr(value)
            na_name = stringdtype_na_name(value)
            if op == 'isalpha':
                result = stringdtype_isalpha_na_data(
                    data, 0, allocator, na_kind, na_name[0], na_name[1])
            elif op == 'isalnum':
                result = stringdtype_isalnum_na_data(
                    data, 0, allocator, na_kind, na_name[0], na_name[1])
            elif op == 'isdecimal':
                result = stringdtype_isdecimal_na_data(
                    data, 0, allocator, na_kind, na_name[0], na_name[1])
            elif op == 'isdigit':
                result = stringdtype_isdigit_na_data(
                    data, 0, allocator, na_kind, na_name[0], na_name[1])
            elif op == 'isnumeric':
                result = stringdtype_isnumeric_na_data(
                    data, 0, allocator, na_kind, na_name[0], na_name[1])
            elif op == 'isspace':
                result = stringdtype_isspace_na_data(
                    data, 0, allocator, na_kind, na_name[0], na_name[1])
            elif op == 'islower':
                result = stringdtype_islower_na_data(
                    data, 0, allocator, na_kind, na_name[0], na_name[1])
            elif op == 'isupper':
                result = stringdtype_isupper_na_data(
                    data, 0, allocator, na_kind, na_name[0], na_name[1])
            else:
                result = stringdtype_istitle_na_data(
                    data, 0, allocator, na_kind, na_name[0], na_name[1])
            stringdtype_release_allocator(allocator)
            if result < 0:
                raise ValueError(
                    f'Cannot use the {op} function with a null that is '
                    'not a nan-like value')
            return bool(result)

        return impl

    if na_kind != 0:
        def impl(value):
            result = np.empty(value.size, np.bool_)
            if value.size == 0:
                return result
            allocator = stringdtype_acquire_allocator(value)
            data = stringdtype_data_ptr(value)
            na_name = stringdtype_na_name(value)
            null_string = False
            for i in range(value.size):
                if op == 'isalpha':
                    predicate = stringdtype_isalpha_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                elif op == 'isalnum':
                    predicate = stringdtype_isalnum_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                elif op == 'isdecimal':
                    predicate = stringdtype_isdecimal_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                elif op == 'isdigit':
                    predicate = stringdtype_isdigit_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                elif op == 'isnumeric':
                    predicate = stringdtype_isnumeric_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                elif op == 'isspace':
                    predicate = stringdtype_isspace_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                elif op == 'islower':
                    predicate = stringdtype_islower_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                elif op == 'isupper':
                    predicate = stringdtype_isupper_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                else:
                    predicate = stringdtype_istitle_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                if predicate < 0:
                    null_string = True
                    predicate = 0
                result[i] = bool(predicate)
            stringdtype_release_allocator(allocator)
            if null_string:
                raise ValueError(
                    f'Cannot use the {op} function with a null that is '
                    'not a nan-like value')
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
        na_kind = value.dtype.na_kind

        if value.ndim == 0:
            if na_kind == 0:
                def impl(value):
                    allocator = stringdtype_acquire_allocator(value)
                    length = stringdtype_codepoint_len_data(
                        stringdtype_data_ptr(value), 0, allocator)
                    stringdtype_release_allocator(allocator)
                    if length < 0:
                        raise ValueError(
                            'The length of a null string is undefined')
                    return length

                return impl

            def impl(value):
                allocator = stringdtype_acquire_allocator(value)
                na_name = stringdtype_na_name(value)
                length = stringdtype_codepoint_len_na_data(
                    stringdtype_data_ptr(value), 0, allocator, na_kind,
                    na_name[0], na_name[1])
                stringdtype_release_allocator(allocator)
                if length < 0:
                    raise ValueError('The length of a null string is undefined')
                return length

            return impl

        if na_kind != 0:
            def impl(value):
                result = np.empty(value.size, np.int64)
                allocator = stringdtype_acquire_allocator(value)
                data = stringdtype_data_ptr(value)
                na_name = stringdtype_na_name(value)
                null_string = False
                for i in range(value.size):
                    length = stringdtype_codepoint_len_na_data(
                        data, i, allocator, na_kind, na_name[0], na_name[1])
                    if length < 0:
                        null_string = True
                        length = 0
                    result[i] = length
                stringdtype_release_allocator(allocator)
                if null_string:
                    raise ValueError(
                        'The length of a null string is undefined')
                return result

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
