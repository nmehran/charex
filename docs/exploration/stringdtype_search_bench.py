"""Benchmark active StringDType substring-search paths."""

from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import charex  # noqa: F401
from charex.tests.definitions import StringsInformation


def make_values(value_pattern, pattern_pattern, n):
    value_repeats = (n + len(value_pattern) - 1) // len(value_pattern)
    pattern_repeats = (n + len(pattern_pattern) - 1) // len(pattern_pattern)
    dtype = np.dtypes.StringDType()
    return (
        np.array((value_pattern * value_repeats)[:n], dtype=dtype),
        np.array((pattern_pattern * pattern_repeats)[:n], dtype=dtype),
    )


def call_numpy(op, values, patterns, start, end):
    if end is None:
        return op(values, patterns, start)
    return op(values, patterns, start, end)


def call_charex(method, values, patterns, start, end):
    if end is None:
        return method(values, patterns, start)
    return method(values, patterns, start, end)


def bench_pair(label, numpy_op, charex_method, values, patterns,
               start=0, end=None, repeat=11):
    expected = call_numpy(numpy_op, values, patterns, start, end)
    np.testing.assert_array_equal(
        call_charex(charex_method, values, patterns, start, end),
        expected,
    )

    timings = {'numpy': [], 'charex': []}
    order = ['numpy', 'charex'] * repeat
    random.Random(1337).shuffle(order)
    for name in order:
        begin = time.perf_counter_ns()
        if name == 'numpy':
            call_numpy(numpy_op, values, patterns, start, end)
        else:
            call_charex(charex_method, values, patterns, start, end)
        timings[name].append((time.perf_counter_ns() - begin) / 1e6)

    numpy_median = statistics.median(timings['numpy'])
    charex_median = statistics.median(timings['charex'])
    print(f'{label:<24} numpy min {min(timings["numpy"]):8.3f} ms  '
          f'median {numpy_median:8.3f} ms')
    print(f'{"":<24} charex min {min(timings["charex"]):7.3f} ms  '
          f'median {charex_median:8.3f} ms  '
          f'speedup {numpy_median / charex_median:7.3f}x')


def main():
    strings = StringsInformation()
    cases = [
        ('short-default', 0, None,
         ['abcabc', 'xabc', 'miss', 'aaaa', ''],
         ['bc', 'ab', 'zz', 'aa', '']),
        ('short-slice', 1, -1,
         ['xabcx', 'xéfgx', 'x🙂abcx', 'xa\x00bcx'],
         ['abc', 'éf', '🙂a', 'a\x00b']),
        ('empty-pattern', 0, None,
         ['abc', '', '\x00', '🙂'],
         ['', '\x00', '\x00\x00', '']),
        ('embedded-nul', 0, None,
         ['a\x00bc\x00bc', 'ab\x00c', '\x00abc', 'abc\x00x'],
         ['\x00bc', 'b\x00c', '\x00a', 'bc\x00']),
        ('trailing-nul-pattern', 0, None,
         ['a', 'ab', 'ab\x00x', 'é', 'é\x00x', 'x🙂\x00y'],
         ['a\x00', 'ab\x00', 'ab\x00', 'é\x00',
          'é\x00', '🙂\x00']),
        ('unicode', 0, None,
         ['éclairé', 'κόσμεκό', '🙂abc🙂', '漢字abc漢字'],
         ['é', 'κό', '🙂', '漢字']),
        ('long-first', 0, None,
         ['needle' + 'a' * 256, 'é' * 16 + 'x' * 256,
          '🙂' * 8 + 'x' * 256],
         ['needle', 'é' * 8, '🙂' * 4]),
        ('long-last', 0, None,
         ['a' * 256 + 'needle', 'x' * 256 + 'é' * 16,
          'x' * 256 + '🙂' * 8],
         ['needle', 'é' * 8, '🙂' * 4]),
        ('long-no-match', 0, None,
         ['a' * 256 + 'x', 'é' * 128 + 'x', '🙂' * 64 + 'x'],
         ['z' * 8, 'ø' * 8, '🙃' * 4]),
        ('long-repeated', 0, None,
         ['a' * 256, 'é' * 128, '🙂' * 64],
         ['aa', 'éé', '🙂🙂']),
    ]

    operations = [
        ('find', np.strings.find, strings.strings_find, False),
        ('rfind', np.strings.rfind, strings.strings_rfind, False),
        ('count', np.strings.count, strings.strings_count, False),
        ('index', np.strings.index, strings.strings_index, True),
        ('rindex', np.strings.rindex, strings.strings_rindex, True),
    ]
    index_cases = {
        'short-slice', 'empty-pattern', 'embedded-nul', 'unicode',
        'long-first', 'long-last', 'long-repeated',
    }

    for n in [1000, 100000]:
        print(f'\nn={n}')
        for case_name, start, end, values_pattern, patterns_pattern in cases:
            values, patterns = make_values(values_pattern, patterns_pattern, n)
            for op_name, numpy_op, charex_method, requires_success in \
                    operations:
                if requires_success and case_name not in index_cases:
                    continue
                label = f'{op_name}:{case_name}'
                bench_pair(label, numpy_op, charex_method,
                           values, patterns, start, end)


if __name__ == '__main__':
    main()
