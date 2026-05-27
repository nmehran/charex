"""Audit NumPy/charex string array shape and layout parity.

This is an exploration harness, not a test suite.  It records NumPy behavior
and the current charex-in-Numba behavior for representative string operations
across dtype, API, dimensionality, and stride/layout cases.
"""

from __future__ import annotations

from argparse import ArgumentParser
from csv import DictWriter
from dataclasses import dataclass
from pathlib import Path
from platform import python_version
from typing import Callable
import sys

import charex  # noqa: F401
import numba
from numba import njit
import numpy as np


STRINGS = getattr(np, 'strings', None)
STRING_DTYPE = getattr(getattr(np, 'dtypes', None), 'StringDType', None)


@njit(nogil=True, cache=False)
def jit_char_equal(left, right):
    return np.char.equal(left, right)


@njit(nogil=True, cache=False)
def jit_char_not_equal(left, right):
    return np.char.not_equal(left, right)


@njit(nogil=True, cache=False)
def jit_char_greater(left, right):
    return np.char.greater(left, right)


@njit(nogil=True, cache=False)
def jit_char_greater_equal(left, right):
    return np.char.greater_equal(left, right)


@njit(nogil=True, cache=False)
def jit_char_less(left, right):
    return np.char.less(left, right)


@njit(nogil=True, cache=False)
def jit_char_less_equal(left, right):
    return np.char.less_equal(left, right)


@njit(nogil=True, cache=False)
def jit_char_count(value, pattern):
    return np.char.count(value, pattern)


@njit(nogil=True, cache=False)
def jit_char_find(value, pattern):
    return np.char.find(value, pattern)


@njit(nogil=True, cache=False)
def jit_char_rfind(value, pattern):
    return np.char.rfind(value, pattern)


@njit(nogil=True, cache=False)
def jit_char_index(value, pattern):
    return np.char.index(value, pattern)


@njit(nogil=True, cache=False)
def jit_char_rindex(value, pattern):
    return np.char.rindex(value, pattern)


@njit(nogil=True, cache=False)
def jit_char_startswith(value, pattern):
    return np.char.startswith(value, pattern)


@njit(nogil=True, cache=False)
def jit_char_endswith(value, pattern):
    return np.char.endswith(value, pattern)


@njit(nogil=True, cache=False)
def jit_char_str_len(value):
    return np.char.str_len(value)


@njit(nogil=True, cache=False)
def jit_char_isalpha(value):
    return np.char.isalpha(value)


@njit(nogil=True, cache=False)
def jit_char_isalnum(value):
    return np.char.isalnum(value)


@njit(nogil=True, cache=False)
def jit_char_isdecimal(value):
    return np.char.isdecimal(value)


@njit(nogil=True, cache=False)
def jit_char_isdigit(value):
    return np.char.isdigit(value)


@njit(nogil=True, cache=False)
def jit_char_islower(value):
    return np.char.islower(value)


@njit(nogil=True, cache=False)
def jit_char_isnumeric(value):
    return np.char.isnumeric(value)


@njit(nogil=True, cache=False)
def jit_char_isspace(value):
    return np.char.isspace(value)


@njit(nogil=True, cache=False)
def jit_char_istitle(value):
    return np.char.istitle(value)


@njit(nogil=True, cache=False)
def jit_char_isupper(value):
    return np.char.isupper(value)


if STRINGS is not None:
    @njit(nogil=True, cache=False)
    def jit_strings_equal(left, right):
        return np.strings.equal(left, right)

    @njit(nogil=True, cache=False)
    def jit_strings_not_equal(left, right):
        return np.strings.not_equal(left, right)

    @njit(nogil=True, cache=False)
    def jit_strings_greater(left, right):
        return np.strings.greater(left, right)

    @njit(nogil=True, cache=False)
    def jit_strings_greater_equal(left, right):
        return np.strings.greater_equal(left, right)

    @njit(nogil=True, cache=False)
    def jit_strings_less(left, right):
        return np.strings.less(left, right)

    @njit(nogil=True, cache=False)
    def jit_strings_less_equal(left, right):
        return np.strings.less_equal(left, right)

    @njit(nogil=True, cache=False)
    def jit_strings_count(value, pattern):
        return np.strings.count(value, pattern)

    @njit(nogil=True, cache=False)
    def jit_strings_find(value, pattern):
        return np.strings.find(value, pattern)

    @njit(nogil=True, cache=False)
    def jit_strings_rfind(value, pattern):
        return np.strings.rfind(value, pattern)

    @njit(nogil=True, cache=False)
    def jit_strings_index(value, pattern):
        return np.strings.index(value, pattern)

    @njit(nogil=True, cache=False)
    def jit_strings_rindex(value, pattern):
        return np.strings.rindex(value, pattern)

    @njit(nogil=True, cache=False)
    def jit_strings_startswith(value, pattern):
        return np.strings.startswith(value, pattern)

    @njit(nogil=True, cache=False)
    def jit_strings_endswith(value, pattern):
        return np.strings.endswith(value, pattern)

    @njit(nogil=True, cache=False)
    def jit_strings_str_len(value):
        return np.strings.str_len(value)

    @njit(nogil=True, cache=False)
    def jit_strings_isalpha(value):
        return np.strings.isalpha(value)

    @njit(nogil=True, cache=False)
    def jit_strings_isalnum(value):
        return np.strings.isalnum(value)

    @njit(nogil=True, cache=False)
    def jit_strings_isdecimal(value):
        return np.strings.isdecimal(value)

    @njit(nogil=True, cache=False)
    def jit_strings_isdigit(value):
        return np.strings.isdigit(value)

    @njit(nogil=True, cache=False)
    def jit_strings_islower(value):
        return np.strings.islower(value)

    @njit(nogil=True, cache=False)
    def jit_strings_isnumeric(value):
        return np.strings.isnumeric(value)

    @njit(nogil=True, cache=False)
    def jit_strings_isspace(value):
        return np.strings.isspace(value)

    @njit(nogil=True, cache=False)
    def jit_strings_istitle(value):
        return np.strings.istitle(value)

    @njit(nogil=True, cache=False)
    def jit_strings_isupper(value):
        return np.strings.isupper(value)


@dataclass(frozen=True)
class Operation:
    family: str
    name: str
    nargs: int
    char_jit: Callable | None
    strings_jit: Callable | None
    char_numpy: Callable | None
    strings_numpy: Callable | None


@dataclass(frozen=True)
class Case:
    dtype: str
    layout: str
    builder: Callable[[], tuple]


OPS = [
    Operation('comparison', 'equal', 2, jit_char_equal,
              globals().get('jit_strings_equal'), np.char.equal,
              getattr(STRINGS, 'equal', None)),
    Operation('comparison', 'not_equal', 2, jit_char_not_equal,
              globals().get('jit_strings_not_equal'), np.char.not_equal,
              getattr(STRINGS, 'not_equal', None)),
    Operation('comparison', 'greater', 2, jit_char_greater,
              globals().get('jit_strings_greater'), np.char.greater,
              getattr(STRINGS, 'greater', None)),
    Operation('comparison', 'greater_equal', 2, jit_char_greater_equal,
              globals().get('jit_strings_greater_equal'),
              np.char.greater_equal, getattr(STRINGS, 'greater_equal', None)),
    Operation('comparison', 'less', 2, jit_char_less,
              globals().get('jit_strings_less'), np.char.less,
              getattr(STRINGS, 'less', None)),
    Operation('comparison', 'less_equal', 2, jit_char_less_equal,
              globals().get('jit_strings_less_equal'), np.char.less_equal,
              getattr(STRINGS, 'less_equal', None)),
    Operation('occurrence', 'count', 2, jit_char_count,
              globals().get('jit_strings_count'), np.char.count,
              getattr(STRINGS, 'count', None)),
    Operation('occurrence', 'find', 2, jit_char_find,
              globals().get('jit_strings_find'), np.char.find,
              getattr(STRINGS, 'find', None)),
    Operation('occurrence', 'rfind', 2, jit_char_rfind,
              globals().get('jit_strings_rfind'), np.char.rfind,
              getattr(STRINGS, 'rfind', None)),
    Operation('occurrence', 'index', 2, jit_char_index,
              globals().get('jit_strings_index'), np.char.index,
              getattr(STRINGS, 'index', None)),
    Operation('occurrence', 'rindex', 2, jit_char_rindex,
              globals().get('jit_strings_rindex'), np.char.rindex,
              getattr(STRINGS, 'rindex', None)),
    Operation('occurrence', 'startswith', 2, jit_char_startswith,
              globals().get('jit_strings_startswith'), np.char.startswith,
              getattr(STRINGS, 'startswith', None)),
    Operation('occurrence', 'endswith', 2, jit_char_endswith,
              globals().get('jit_strings_endswith'), np.char.endswith,
              getattr(STRINGS, 'endswith', None)),
    Operation('property', 'str_len', 1, jit_char_str_len,
              globals().get('jit_strings_str_len'), np.char.str_len,
              getattr(STRINGS, 'str_len', None)),
    Operation('property', 'isalpha', 1, jit_char_isalpha,
              globals().get('jit_strings_isalpha'), np.char.isalpha,
              getattr(STRINGS, 'isalpha', None)),
    Operation('property', 'isalnum', 1, jit_char_isalnum,
              globals().get('jit_strings_isalnum'), np.char.isalnum,
              getattr(STRINGS, 'isalnum', None)),
    Operation('property', 'isdecimal', 1, jit_char_isdecimal,
              globals().get('jit_strings_isdecimal'), np.char.isdecimal,
              getattr(STRINGS, 'isdecimal', None)),
    Operation('property', 'isdigit', 1, jit_char_isdigit,
              globals().get('jit_strings_isdigit'), np.char.isdigit,
              getattr(STRINGS, 'isdigit', None)),
    Operation('property', 'islower', 1, jit_char_islower,
              globals().get('jit_strings_islower'), np.char.islower,
              getattr(STRINGS, 'islower', None)),
    Operation('property', 'isnumeric', 1, jit_char_isnumeric,
              globals().get('jit_strings_isnumeric'), np.char.isnumeric,
              getattr(STRINGS, 'isnumeric', None)),
    Operation('property', 'isspace', 1, jit_char_isspace,
              globals().get('jit_strings_isspace'), np.char.isspace,
              getattr(STRINGS, 'isspace', None)),
    Operation('property', 'istitle', 1, jit_char_istitle,
              globals().get('jit_strings_istitle'), np.char.istitle,
              getattr(STRINGS, 'istitle', None)),
    Operation('property', 'isupper', 1, jit_char_isupper,
              globals().get('jit_strings_isupper'), np.char.isupper,
              getattr(STRINGS, 'isupper', None)),
]

REPRESENTATIVE_METHODS = {
    'equal', 'greater', 'startswith', 'find', 'index', 'str_len', 'isalpha',
    'isdecimal',
}


def _array(values, dtype_name):
    if dtype_name == 'S':
        return np.array([value.encode('ascii') for value in values], dtype='S8')
    if dtype_name == 'U':
        return np.array(values, dtype='U8')
    if dtype_name == 'T':
        return np.array(values, dtype=STRING_DTYPE())
    if dtype_name == 'T-na-string':
        return np.array(values, dtype=STRING_DTYPE(na_object='MISSING'))
    if dtype_name == 'T-na-nan':
        return np.array(values, dtype=STRING_DTYPE(na_object=np.nan))
    raise ValueError(dtype_name)


def _values(dtype_name):
    values = ['abcabc', 'xabc', 'abcx', '', 'alpha', 'ABC123']
    if dtype_name == 'T-na-string':
        values[3] = 'MISSING'
    elif dtype_name == 'T-na-nan':
        values[3] = np.nan
    return _array(values, dtype_name)


def _patterns(dtype_name):
    values = ['abc', 'x', 'x', '', 'a', 'ABC']
    if dtype_name == 'T-na-string':
        values[3] = 'MISSING'
    elif dtype_name == 'T-na-nan':
        values[3] = np.nan
    return _array(values, dtype_name)


def _scalar_value(dtype_name):
    if dtype_name == 'S':
        return b'abcabc'
    if dtype_name == 'U':
        return 'abcabc'
    return np.array('abcabc', dtype=STRING_DTYPE())


def _scalar_pattern(dtype_name):
    if dtype_name == 'S':
        return b'abc'
    if dtype_name == 'U':
        return 'abc'
    return np.array('abc', dtype=STRING_DTYPE())


def _readonly(value):
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    return value


def _zero_dim_from(array, index=0):
    return np.array(array[index], dtype=array.dtype)


def _case_builder(dtype_name, layout, nargs):
    def pair(left, right):
        return (left,) if nargs == 1 else (left, right)

    def build():
        values = _values(dtype_name)
        patterns = _patterns(dtype_name)
        if layout == 'python-scalar':
            return pair(_scalar_value(dtype_name), _scalar_pattern(dtype_name))
        if layout == 'zero-dimensional':
            return pair(_zero_dim_from(values), _zero_dim_from(patterns))
        if layout == 'contiguous-1d':
            return pair(values[:4].copy(), patterns[:4].copy())
        if layout == 'readonly-1d':
            return pair(_readonly(values[:4].copy()),
                        _readonly(patterns[:4].copy()))
        if layout == 'positive-stride-1d':
            return pair(values[::2], patterns[::2])
        if layout == 'negative-stride-1d':
            return pair(values[::-2], patterns[::-2])
        if layout == 'zero-stride-1d':
            return pair(np.broadcast_to(values[:1], (4,)),
                        np.broadcast_to(patterns[:1], (4,)))
        if layout == 'empty-stride-1d':
            return pair(values[:0:2], patterns[:0:2])
        if layout == 'contiguous-2d':
            return pair(values.reshape(2, 3), patterns.reshape(2, 3))
        if layout == 'broadcast-2d':
            return pair(values[:2].reshape(2, 1), patterns[:3].reshape(1, 3))
        if layout == 'shape-mismatch-1d':
            return pair(values[:3].copy(), patterns[:2].copy())
        raise ValueError(layout)

    return build


def cases(methods):
    dtype_names = ['S', 'U']
    if STRING_DTYPE is not None:
        dtype_names.extend(['T', 'T-na-string', 'T-na-nan'])

    layouts = [
        'python-scalar',
        'zero-dimensional',
        'contiguous-1d',
        'readonly-1d',
        'positive-stride-1d',
        'negative-stride-1d',
        'zero-stride-1d',
        'empty-stride-1d',
        'contiguous-2d',
        'broadcast-2d',
        'shape-mismatch-1d',
    ]

    for op in OPS:
        if methods == 'representative' and op.name not in REPRESENTATIVE_METHODS:
            continue
        for dtype_name in dtype_names:
            for layout in layouts:
                if layout == 'python-scalar' and dtype_name.startswith('T'):
                    continue
                yield op, Case(dtype_name, layout,
                               _case_builder(dtype_name, layout, op.nargs))


def _array_snapshot(args):
    snapshot = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            snapshot.append(arg.copy())
        else:
            snapshot.append(None)
    return snapshot


def _mutated(args, before):
    for arg, expected in zip(args, before):
        if expected is None:
            continue
        if np.array_equal(arg, expected):
            continue
        try:
            unchanged = np.array_equal(arg, expected, equal_nan=True)
        except TypeError:
            unchanged = False
        if not unchanged:
            return True
    return False


def _result_summary(value):
    array = np.asarray(value)
    if array.shape == ():
        try:
            sample = array.item()
        except ValueError:
            sample = '<scalar>'
    elif array.size:
        sample = array.reshape(-1)[0]
    else:
        sample = '<empty>'
    return {
        'status': 'ok',
        'result_shape': repr(array.shape),
        'result_dtype': str(array.dtype),
        'result_sample': repr(sample),
    }


def _exception_summary(exc):
    return {
        'status': 'error',
        'error_type': type(exc).__name__,
        'error_message': str(exc).splitlines()[0] if str(exc) else '',
    }


def _run(func, args):
    before = _array_snapshot(args)
    try:
        result = func(*args)
    except Exception as exc:  # noqa: BLE001 - this is an audit harness.
        summary = _exception_summary(exc)
    else:
        summary = _result_summary(result)
    summary['mutated'] = _mutated(args, before)
    return summary


def _same_outcome(numpy_summary, charex_summary):
    if numpy_summary['status'] != charex_summary['status']:
        return False
    if numpy_summary['status'] == 'error':
        return numpy_summary.get('error_type') == charex_summary.get(
            'error_type')
    return numpy_summary['result_shape'] == charex_summary['result_shape'] \
        and numpy_summary['result_dtype'] == charex_summary['result_dtype']


def _compare_values(numpy_func, charex_func, builder):
    left = builder()
    right = builder()
    expected = _run(numpy_func, left)
    actual = _run(charex_func, right)
    values_match = ''
    if expected['status'] == 'ok' and actual['status'] == 'ok':
        try:
            np.testing.assert_array_equal(
                np.asarray(charex_func(*builder())),
                np.asarray(numpy_func(*builder())),
            )
        except Exception as exc:  # noqa: BLE001 - audit mismatch detail.
            values_match = f'false:{type(exc).__name__}'
        else:
            values_match = 'true'
    return expected, actual, values_match


def _args_summary(args):
    parts = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            parts.append(
                f'shape={arg.shape},strides={arg.strides},dtype={arg.dtype},'
                f'writeable={arg.flags.writeable}'
            )
        else:
            parts.append(f'{type(arg).__name__}:{arg!r}')
    return ' | '.join(parts)


def audit_rows(methods, api_filter, dtype_filter):
    rows = []
    for op, case in cases(methods):
        if dtype_filter != 'all' and case.dtype not in dtype_filter.split(','):
            continue
        api_pairs = []
        if op.char_jit is not None and op.char_numpy is not None:
            api_pairs.append(('char', op.char_jit, op.char_numpy))
        if op.strings_jit is not None and op.strings_numpy is not None:
            api_pairs.append(('strings', op.strings_jit, op.strings_numpy))
        for api, charex_func, numpy_func in api_pairs:
            if api_filter != 'all' and api not in api_filter.split(','):
                continue
            if api == 'char' and case.dtype.startswith('T'):
                continue
            args = case.builder()
            expected, actual, values_match = _compare_values(
                numpy_func, charex_func, case.builder)
            rows.append({
                'api': api,
                'family': op.family,
                'method': op.name,
                'dtype': case.dtype,
                'layout': case.layout,
                'args': _args_summary(args),
                'numpy_status': expected['status'],
                'numpy_shape': expected.get('result_shape', ''),
                'numpy_dtype': expected.get('result_dtype', ''),
                'numpy_error': expected.get('error_type', ''),
                'numpy_message': expected.get('error_message', ''),
                'numpy_mutated': expected['mutated'],
                'charex_status': actual['status'],
                'charex_shape': actual.get('result_shape', ''),
                'charex_dtype': actual.get('result_dtype', ''),
                'charex_error': actual.get('error_type', ''),
                'charex_message': actual.get('error_message', ''),
                'charex_mutated': actual['mutated'],
                'same_outcome': _same_outcome(expected, actual),
                'values_match': values_match,
            })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0]),
                            lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    total = len(rows)
    same = sum(row['same_outcome']
               and not row['values_match'].startswith('false')
               for row in rows)
    numpy_ok_charex_error = [
        row for row in rows
        if row['numpy_status'] == 'ok' and row['charex_status'] == 'error'
    ]
    mismatches = [
        row for row in rows
        if not row['same_outcome'] or row['values_match'].startswith('false')
    ]
    print(f'Python {python_version()} | NumPy {np.__version__} | '
          f'Numba {numba.__version__}')
    print(f'rows={total} matching={same} mismatches={len(mismatches)}')
    print(f'numpy-ok/charex-error={len(numpy_ok_charex_error)}')
    print()
    print('Top mismatch groups:')
    grouped = {}
    for row in mismatches:
        key = (row['api'], row['dtype'], row['layout'], row['family'])
        grouped[key] = grouped.get(key, 0) + 1
    for key, count in sorted(grouped.items(), key=lambda item: -item[1])[:20]:
        print(f'  {count:4d} {key[0]:8s} {key[1]:11s} '
              f'{key[2]:22s} {key[3]}')

    print()
    print('Representative mismatches:')
    for row in mismatches[:25]:
        print(
            f"  {row['api']:8s} {row['dtype']:11s} {row['layout']:22s} "
            f"{row['method']:13s} numpy={row['numpy_status']}"
            f"/{row['numpy_shape'] or row['numpy_error']} "
            f"charex={row['charex_status']}"
            f"/{row['charex_shape'] or row['charex_error']} "
            f"values={row['values_match']}"
        )


def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument('--methods', choices=['representative', 'all'],
                        default='representative')
    parser.add_argument('--api', default='all',
                        help='all, char, strings, or comma-separated values')
    parser.add_argument('--dtype', default='all',
                        help='all or comma-separated S,U,T,T-na-string,T-na-nan')
    parser.add_argument('--csv', type=Path,
                        default=Path('docs/exploration/'
                                     'string_array_shape_audit.csv'))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = audit_rows(args.methods, args.api, args.dtype)
    if not rows:
        raise SystemExit('no rows selected')
    write_csv(args.csv, rows)
    summarize(rows)
    print()
    print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
