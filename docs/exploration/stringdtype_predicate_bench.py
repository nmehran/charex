"""Benchmark active StringDType predicate paths."""

from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import charex  # noqa: F401
from charex.tests.definitions import StringsInformation


def make_values(pattern, n):
    repeats = (n + len(pattern) - 1) // len(pattern)
    return np.array((pattern * repeats)[:n], dtype=np.dtypes.StringDType())


def bench_pair(label, numpy_op, charex_method, values, repeat=11):
    expected = numpy_op(values)
    np.testing.assert_array_equal(charex_method(values), expected)

    timings = {'numpy': [], 'charex': []}
    order = ['numpy', 'charex'] * repeat
    random.Random(1337).shuffle(order)
    for name in order:
        begin = time.perf_counter_ns()
        if name == 'numpy':
            numpy_op(values)
        else:
            charex_method(values)
        timings[name].append((time.perf_counter_ns() - begin) / 1e6)

    numpy_median = statistics.median(timings['numpy'])
    charex_median = statistics.median(timings['charex'])
    print(f'{label:<28} numpy min {min(timings["numpy"]):8.3f} ms  '
          f'median {numpy_median:8.3f} ms')
    print(f'{"":<28} charex min {min(timings["charex"]):7.3f} ms  '
          f'median {charex_median:8.3f} ms  '
          f'speedup {numpy_median / charex_median:7.3f}x')


def main():
    strings = StringsInformation()
    cases = [
        ('ascii-alpha', ['alpha', 'Beta', 'GAMMA', 'delta']),
        ('unicode-alpha', ['αβγ', 'éclair', '漢字', 'ǅuro']),
        ('numeric', ['123', '１２', '١٢٣', 'Ⅷ', '⅕', '²']),
        ('space', [' ', '\t', '\n', '\x1c', '\x1f']),
        ('cased', ['lower', 'UPPER', 'Title Case', 'Title case', 'ǅuro']),
        ('mixed', ['alpha1', '🙂', '', 'abc\x00', 'ab\x00cd', '\x00abc']),
        ('long-ascii', ['a' * 256, 'A' * 256, 'Title ' * 48]),
        ('long-unicode', ['é' * 128, '漢字' * 128, '🙂' * 64]),
    ]
    operations = [
        ('isalpha', np.strings.isalpha, strings.strings_isalpha),
        ('isalnum', np.strings.isalnum, strings.strings_isalnum),
        ('isdecimal', np.strings.isdecimal, strings.strings_isdecimal),
        ('isdigit', np.strings.isdigit, strings.strings_isdigit),
        ('isnumeric', np.strings.isnumeric, strings.strings_isnumeric),
        ('isspace', np.strings.isspace, strings.strings_isspace),
        ('islower', np.strings.islower, strings.strings_islower),
        ('isupper', np.strings.isupper, strings.strings_isupper),
        ('istitle', np.strings.istitle, strings.strings_istitle),
    ]

    for n in [1000, 100000]:
        print(f'\nn={n}')
        for case_name, pattern in cases:
            values = make_values(pattern, n)
            for op_name, numpy_op, charex_method in operations:
                bench_pair(f'{op_name}:{case_name}', numpy_op,
                           charex_method, values)


if __name__ == '__main__':
    main()
