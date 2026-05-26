"""Benchmark active StringDType ordering comparison paths."""

from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import charex  # noqa: F401
from charex.tests.definitions import StringsComparisonOperators


def make_values(left_pattern, right_pattern, n):
    left_repeats = (n + len(left_pattern) - 1) // len(left_pattern)
    right_repeats = (n + len(right_pattern) - 1) // len(right_pattern)
    dtype = np.dtypes.StringDType()
    return (
        np.array((left_pattern * left_repeats)[:n], dtype=dtype),
        np.array((right_pattern * right_repeats)[:n], dtype=dtype),
    )


def bench_pair(label, numpy_op, charex_method, left, right, repeat=11):
    expected = numpy_op(left, right)
    np.testing.assert_array_equal(charex_method(left, right), expected)

    timings = {'numpy': [], 'charex': []}
    order = ['numpy', 'charex'] * repeat
    random.Random(1337).shuffle(order)
    for name in order:
        begin = time.perf_counter_ns()
        if name == 'numpy':
            numpy_op(left, right)
        else:
            charex_method(left, right)
        timings[name].append((time.perf_counter_ns() - begin) / 1e6)

    numpy_median = statistics.median(timings['numpy'])
    charex_median = statistics.median(timings['charex'])
    print(f'{label:<28} numpy min {min(timings["numpy"]):8.3f} ms  '
          f'median {numpy_median:8.3f} ms')
    print(f'{"":<28} charex min {min(timings["charex"]):7.3f} ms  '
          f'median {charex_median:8.3f} ms  '
          f'speedup {numpy_median / charex_median:7.3f}x')


def main():
    strings = StringsComparisonOperators()
    cases = [
        ('ascii-mixed',
         ['a', 'b', 'aa', '', 'alpha'],
         ['a', 'a', 'a', 'a', 'beta']),
        ('unicode-mixed',
         ['é', 'α', '🙂', '一', '漢字'],
         ['e', 'β', '🙃', '二', '漢語']),
        ('embedded-nul',
         ['a\x00x', 'a\x00y', 'ab\x00', '\x00a', 'é\x00x'],
         ['a\x00y', 'a\x00x', 'ab\x00x', '\x00b', 'é']),
        ('one-sided-nul',
         ['a\x00x', 'abx', '\x00x', 'é\x00'],
         ['abx', 'a\x00x', 'a', 'é']),
        ('long-first',
         ['needle' + 'a' * 256, 'é' * 16 + 'x' * 256,
          '🙂' * 8 + 'x' * 256],
         ['meedle' + 'a' * 256, 'é' * 15 + 'x' * 257,
          '🙂' * 7 + 'x' * 260]),
        ('long-last',
         ['a' * 256 + 'x', 'é' * 128 + 'x', '🙂' * 64 + 'x'],
         ['a' * 256 + 'y', 'é' * 128 + 'y', '🙂' * 64 + 'y']),
        ('long-equal',
         ['a' * 256, 'é' * 128, '🙂' * 64],
         ['a' * 256, 'é' * 128, '🙂' * 64]),
    ]
    operations = [
        ('greater', np.strings.greater, strings.strings_greater),
        ('greater_equal', np.strings.greater_equal,
         strings.strings_greater_equal),
        ('less', np.strings.less, strings.strings_less),
        ('less_equal', np.strings.less_equal, strings.strings_less_equal),
    ]

    for n in [1000, 100000]:
        print(f'\nn={n}')
        for case_name, left_pattern, right_pattern in cases:
            left, right = make_values(left_pattern, right_pattern, n)
            for op_name, numpy_op, charex_method in operations:
                bench_pair(f'{op_name}:{case_name}', numpy_op,
                           charex_method, left, right)


if __name__ == '__main__':
    main()
