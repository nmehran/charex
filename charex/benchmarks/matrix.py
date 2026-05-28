from argparse import ArgumentParser
from csv import DictWriter
from pathlib import Path
from platform import python_version
from time import perf_counter
import gc

import charex  # noqa: F401
import llvmlite
import numba
import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def jit_equal(left, right):
    return np.char.equal(left, right)


@njit(nogil=True, cache=True)
def jit_not_equal(left, right):
    return np.char.not_equal(left, right)


@njit(nogil=True, cache=True)
def jit_greater(left, right):
    return np.char.greater(left, right)


@njit(nogil=True, cache=True)
def jit_greater_equal(left, right):
    return np.char.greater_equal(left, right)


@njit(nogil=True, cache=True)
def jit_less(left, right):
    return np.char.less(left, right)


@njit(nogil=True, cache=True)
def jit_less_equal(left, right):
    return np.char.less_equal(left, right)


@njit(nogil=True, cache=True)
def jit_count(values, sub):
    return np.char.count(values, sub)


@njit(nogil=True, cache=True)
def jit_find(values, sub):
    return np.char.find(values, sub)


@njit(nogil=True, cache=True)
def jit_rfind(values, sub):
    return np.char.rfind(values, sub)


@njit(nogil=True, cache=True)
def jit_startswith(values, sub):
    return np.char.startswith(values, sub)


@njit(nogil=True, cache=True)
def jit_endswith(values, sub):
    return np.char.endswith(values, sub)


@njit(nogil=True, cache=True)
def jit_str_len(values):
    return np.char.str_len(values)


@njit(nogil=True, cache=True)
def jit_isalpha(values):
    return np.char.isalpha(values)


@njit(nogil=True, cache=True)
def jit_isalnum(values):
    return np.char.isalnum(values)


@njit(nogil=True, cache=True)
def jit_isdigit(values):
    return np.char.isdigit(values)


@njit(nogil=True, cache=True)
def jit_islower(values):
    return np.char.islower(values)


@njit(nogil=True, cache=True)
def jit_isspace(values):
    return np.char.isspace(values)


@njit(nogil=True, cache=True)
def jit_istitle(values):
    return np.char.istitle(values)


@njit(nogil=True, cache=True)
def jit_isupper(values):
    return np.char.isupper(values)


@njit(nogil=True, cache=True)
def jit_isdecimal(values):
    return np.char.isdecimal(values)


@njit(nogil=True, cache=True)
def jit_isnumeric(values):
    return np.char.isnumeric(values)


@njit(nogil=True, cache=True)
def jit_strings_equal(left, right):
    return np.strings.equal(left, right)


@njit(nogil=True, cache=True)
def jit_strings_not_equal(left, right):
    return np.strings.not_equal(left, right)


@njit(nogil=True, cache=True)
def jit_strings_greater(left, right):
    return np.strings.greater(left, right)


@njit(nogil=True, cache=True)
def jit_strings_greater_equal(left, right):
    return np.strings.greater_equal(left, right)


@njit(nogil=True, cache=True)
def jit_strings_less(left, right):
    return np.strings.less(left, right)


@njit(nogil=True, cache=True)
def jit_strings_less_equal(left, right):
    return np.strings.less_equal(left, right)


@njit(nogil=True, cache=True)
def jit_strings_count(values, sub):
    return np.strings.count(values, sub)


@njit(nogil=True, cache=True)
def jit_strings_find(values, sub):
    return np.strings.find(values, sub)


@njit(nogil=True, cache=True)
def jit_strings_rfind(values, sub):
    return np.strings.rfind(values, sub)


@njit(nogil=True, cache=True)
def jit_strings_startswith(values, sub):
    return np.strings.startswith(values, sub)


@njit(nogil=True, cache=True)
def jit_strings_endswith(values, sub):
    return np.strings.endswith(values, sub)


@njit(nogil=True, cache=True)
def jit_strings_str_len(values):
    return np.strings.str_len(values)


@njit(nogil=True, cache=True)
def jit_strings_isalpha(values):
    return np.strings.isalpha(values)


@njit(nogil=True, cache=True)
def jit_strings_isalnum(values):
    return np.strings.isalnum(values)


@njit(nogil=True, cache=True)
def jit_strings_isdigit(values):
    return np.strings.isdigit(values)


@njit(nogil=True, cache=True)
def jit_strings_islower(values):
    return np.strings.islower(values)


@njit(nogil=True, cache=True)
def jit_strings_isspace(values):
    return np.strings.isspace(values)


@njit(nogil=True, cache=True)
def jit_strings_istitle(values):
    return np.strings.istitle(values)


@njit(nogil=True, cache=True)
def jit_strings_isupper(values):
    return np.strings.isupper(values)


@njit(nogil=True, cache=True)
def jit_strings_isdecimal(values):
    return np.strings.isdecimal(values)


@njit(nogil=True, cache=True)
def jit_strings_isnumeric(values):
    return np.strings.isnumeric(values)


COMPARISON_FUNCS = [
    ('equal', jit_equal, np.char.equal),
    ('not_equal', jit_not_equal, np.char.not_equal),
    ('greater', jit_greater, np.char.greater),
    ('greater_equal', jit_greater_equal, np.char.greater_equal),
    ('less', jit_less, np.char.less),
    ('less_equal', jit_less_equal, np.char.less_equal),
]

OCCURRENCE_FUNCS = [
    ('count', jit_count, np.char.count),
    ('find', jit_find, np.char.find),
    ('rfind', jit_rfind, np.char.rfind),
    ('startswith', jit_startswith, np.char.startswith),
    ('endswith', jit_endswith, np.char.endswith),
]

PROPERTY_FUNCS = [
    ('str_len', jit_str_len, np.char.str_len),
    ('isalpha', jit_isalpha, np.char.isalpha),
    ('isalnum', jit_isalnum, np.char.isalnum),
    ('isdigit', jit_isdigit, np.char.isdigit),
    ('islower', jit_islower, np.char.islower),
    ('isspace', jit_isspace, np.char.isspace),
    ('istitle', jit_istitle, np.char.istitle),
    ('isupper', jit_isupper, np.char.isupper),
]

NUMERIC_STRING_FUNCS = [
    ('isdecimal', jit_isdecimal, np.char.isdecimal),
    ('isdigit', jit_isdigit, np.char.isdigit),
    ('isnumeric', jit_isnumeric, np.char.isnumeric),
]


_STRINGS = getattr(np, 'strings', None)
_STRING_DTYPE = getattr(getattr(np, 'dtypes', None), 'StringDType', None)

if _STRINGS is None:
    STRINGS_COMPARISON_FUNCS = []
    STRINGS_OCCURRENCE_FUNCS = []
    STRINGS_PROPERTY_FUNCS = []
    STRINGS_NUMERIC_FUNCS = []
else:
    STRINGS_COMPARISON_FUNCS = [
        ('equal', jit_strings_equal, _STRINGS.equal),
        ('not_equal', jit_strings_not_equal, _STRINGS.not_equal),
        ('greater', jit_strings_greater, _STRINGS.greater),
        ('greater_equal', jit_strings_greater_equal, _STRINGS.greater_equal),
        ('less', jit_strings_less, _STRINGS.less),
        ('less_equal', jit_strings_less_equal, _STRINGS.less_equal),
    ]

    STRINGS_OCCURRENCE_FUNCS = [
        ('count', jit_strings_count, _STRINGS.count),
        ('find', jit_strings_find, _STRINGS.find),
        ('rfind', jit_strings_rfind, _STRINGS.rfind),
        ('startswith', jit_strings_startswith, _STRINGS.startswith),
        ('endswith', jit_strings_endswith, _STRINGS.endswith),
    ]

    STRINGS_PROPERTY_FUNCS = [
        ('str_len', jit_strings_str_len, _STRINGS.str_len),
        ('isalpha', jit_strings_isalpha, _STRINGS.isalpha),
        ('isalnum', jit_strings_isalnum, _STRINGS.isalnum),
        ('isdigit', jit_strings_isdigit, _STRINGS.isdigit),
        ('islower', jit_strings_islower, _STRINGS.islower),
        ('isspace', jit_strings_isspace, _STRINGS.isspace),
        ('istitle', jit_strings_istitle, _STRINGS.istitle),
        ('isupper', jit_strings_isupper, _STRINGS.isupper),
    ]

    STRINGS_NUMERIC_FUNCS = [
        ('isdecimal', jit_strings_isdecimal, _STRINGS.isdecimal),
        ('isdigit', jit_strings_isdigit, _STRINGS.isdigit),
        ('isnumeric', jit_strings_isnumeric, _STRINGS.isnumeric),
    ]


TIMING_BATCH = 3


def _time_call(func, args):
    start = perf_counter()
    for _ in range(TIMING_BATCH):
        func(*args)
    return (perf_counter() - start) / TIMING_BATCH


def paired_median_times(jit_func, numpy_func, args, repeat):
    jit_times = np.empty(repeat, dtype=np.float64)
    numpy_times = np.empty(repeat, dtype=np.float64)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(repeat):
            if i % 2:
                numpy_times[i] = _time_call(numpy_func, args)
                jit_times[i] = _time_call(jit_func, args)
            else:
                jit_times[i] = _time_call(jit_func, args)
                numpy_times[i] = _time_call(numpy_func, args)
    finally:
        if gc_enabled:
            gc.enable()
    return float(np.median(jit_times)), float(np.median(numpy_times))


def bench(group, kind, method, case, jit_func, numpy_func, args, repeat):
    expected = numpy_func(*args)
    actual = jit_func(*args)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    jit_time, numpy_time = paired_median_times(
        jit_func, numpy_func, args, repeat)
    speedup = numpy_time / jit_time if jit_time else float('inf')
    print(f'{group:11} {kind:7} {method:13} {case:12} '
          f'charex={jit_time * 1000:9.3f} ms  '
          f'numpy={numpy_time * 1000:9.3f} ms  speedup={speedup:6.2f}x')
    return {
        'group': group,
        'kind': kind,
        'method': method,
        'case': case,
        'charex_ms': jit_time * 1000,
        'numpy_ms': numpy_time * 1000,
        'speedup': speedup,
    }


def string_values(size):
    base16 = np.array(['item%012d' % i for i in range(size)], dtype='U16')
    same16 = base16.copy()
    first16 = np.array(['X' + value[1:] for value in base16], dtype='U16')
    last16 = np.array([value[:-1] + 'X' for value in base16], dtype='U16')
    trail16 = np.array([value[:8] + ' ' * 8 for value in base16], dtype='U16')

    base64 = np.array([('item%012d' % i).ljust(64, 'a')
                       for i in range(size)], dtype='U64')
    last64 = np.array([value[:-1] + 'X' for value in base64], dtype='U64')

    occurrence = np.array(['alpha beta alpha %06d' % i
                           for i in range(size)], dtype='U32')
    properties = np.resize(
        np.array(['Alpha', 'alpha', 'abc123', 'Title Case', 'UPPER',
                  'lower', ' ', '', 'αβγ', '١٢٣', 'Ⅷ', '一二'],
                 dtype='U10'),
        size,
    )
    numerics = np.resize(
        np.array(['123', '١٢٣', 'Ⅷ', '一二', 'abc', '', '3.14'], dtype='U4'),
        size,
    )
    return {
        'comparison': [
            ('equal16', base16, same16),
            ('first16', base16, first16),
            ('last16', base16, last16),
            ('trail16', trail16, base16),
            ('last64', base64, last64),
        ],
        'occurrence': occurrence,
        'properties': properties,
        'numerics': numerics,
    }


def byte_values(size):
    base16 = np.array([('item%012d' % i).encode()
                       for i in range(size)], dtype='S16')
    same16 = base16.copy()
    first16 = np.array([b'X' + value[1:] for value in base16], dtype='S16')
    last16 = np.array([value[:-1] + b'X' for value in base16], dtype='S16')
    trail16 = np.array([value[:8] + b' ' * 8 for value in base16], dtype='S16')

    base64 = np.array([('item%012d' % i).ljust(64, 'a').encode()
                       for i in range(size)], dtype='S64')
    last64 = np.array([value[:-1] + b'X' for value in base64], dtype='S64')

    occurrence = np.array([('alpha beta alpha %06d' % i).encode()
                           for i in range(size)], dtype='S32')
    properties = np.resize(
        np.array([b'Alpha', b'alpha', b'abc123', b'Title Case', b'UPPER',
                  b'lower', b' ', b'', b'123'], dtype='S10'),
        size,
    )
    return {
        'comparison': [
            ('equal16', base16, same16),
            ('first16', base16, first16),
            ('last16', base16, last16),
            ('trail16', trail16, base16),
            ('last64', base64, last64),
        ],
        'occurrence': occurrence,
        'properties': properties,
    }


def stringdtype_values(size):
    dtype = _STRING_DTYPE()
    base16 = np.array(['item%012d' % i for i in range(size)], dtype=dtype)
    same16 = base16.copy()
    first16 = np.array(['X' + value[1:] for value in base16], dtype=dtype)
    last16 = np.array([value[:-1] + 'X' for value in base16], dtype=dtype)
    trail16 = np.array([value[:8] + ' ' * 8 for value in base16],
                       dtype=dtype)

    base64 = np.array([('item%012d' % i).ljust(64, 'a')
                       for i in range(size)], dtype=dtype)
    last64 = np.array([value[:-1] + 'X' for value in base64], dtype=dtype)

    occurrence = np.array(['alpha beta alpha %06d' % i
                           for i in range(size)], dtype=dtype)
    properties = np.resize(
        np.array(['Alpha', 'alpha', 'abc123', 'Title Case', 'UPPER',
                  'lower', ' ', '', 'αβγ', '١٢٣', 'Ⅷ', '一二'],
                 dtype=dtype),
        size,
    )
    numerics = np.resize(
        np.array(['123', '١٢٣', 'Ⅷ', '一二', 'abc', '', '3.14'],
                 dtype=dtype),
        size,
    )
    return {
        'comparison': [
            ('equal16', base16, same16),
            ('first16', base16, first16),
            ('last16', base16, last16),
            ('trail16', trail16, base16),
            ('last64', base64, last64),
        ],
        'occurrence': occurrence,
        'properties': properties,
        'numerics': numerics,
    }


def comparison_records(kind, values, repeat, funcs=COMPARISON_FUNCS):
    records = []
    for method, jit_func, numpy_func in funcs:
        for case, left, right in values['comparison']:
            records.append(bench('comparison', kind, method, case,
                                 jit_func, numpy_func, (left, right), repeat))
    return records


def occurrence_records(kind, values, repeat, funcs=OCCURRENCE_FUNCS, sub=None):
    if sub is None:
        sub = 'alpha' if kind != 'bytes' else b'alpha'
    records = []
    for method, jit_func, numpy_func in funcs:
        records.append(bench('occurrence', kind, method, 'hit',
                             jit_func, numpy_func,
                             (values['occurrence'], sub), repeat))
    return records


def property_records(kind, values, repeat, funcs=PROPERTY_FUNCS):
    records = []
    for method, jit_func, numpy_func in funcs:
        records.append(bench('properties', kind, method, 'mixed',
                             jit_func, numpy_func,
                             (values['properties'],), repeat))
    return records


def numeric_records(values, repeat, kind='strings', funcs=NUMERIC_STRING_FUNCS):
    records = []
    for method, jit_func, numpy_func in funcs:
        records.append(bench('numerics', kind, method, 'mixed',
                             jit_func, numpy_func,
                             (values['numerics'],), repeat))
    return records


def write_csv(records, output_dir, size, repeat):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'matrix.csv'
    fieldnames = [
        'python', 'numpy', 'numba', 'llvmlite', 'size', 'repeat',
        'group', 'kind', 'method', 'case', 'charex_ms', 'numpy_ms', 'speedup',
    ]
    with output_path.open('w', newline='') as file:
        writer = DictWriter(file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for record in records:
            writer.writerow({
                'python': python_version(),
                'numpy': np.__version__,
                'numba': numba.__version__,
                'llvmlite': llvmlite.__version__,
                'size': size,
                'repeat': repeat,
                **record,
            })
    return output_path


def _format_log_tick(value, _):
    if value <= 0:
        return ''
    if value >= 1:
        text = f'{value:.2f}'
    elif value >= 0.01:
        text = f'{value:.3f}'
    else:
        text = f'{value:.4f}'
    return text.rstrip('0').rstrip('.')


def _format_ratio_tick(value, position):
    return f'{_format_log_tick(value, position)}x'


def _set_log_ticks(axis, values, ratio=False):
    from matplotlib.ticker import FuncFormatter, NullFormatter

    axis.set_yscale('log')
    lower, upper = axis.get_ylim()

    def candidate_ticks(multiples):
        ticks = []
        for exponent in range(-4, 5):
            scale = 10.0 ** exponent
            for multiple in multiples:
                value = multiple * scale
                if lower <= value <= upper:
                    ticks.append(value)
        return ticks

    ticks = candidate_ticks((1, 2, 3, 4, 5, 10))
    if ratio and len(ticks) < 4:
        ticks = candidate_ticks(
            (1, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75,
             2, 2.5, 3, 4, 5, 7.5, 10)
        )
    if ratio and len(ticks) < 4:
        step = (upper - lower) / 3
        if step > 0:
            magnitude = 10.0 ** np.floor(np.log10(step))
            for unit in (1, 2, 2.5, 5, 10):
                nice_step = unit * magnitude
                if nice_step >= step:
                    break
            start = np.ceil(lower / nice_step) * nice_step
            value = start
            while value <= upper:
                if value > 0:
                    ticks.append(value)
                value += nice_step
    if not ticks:
        ticks = [value for value in values if value > 0]
    axis.set_yticks(sorted(set(ticks)))
    formatter = _format_ratio_tick if ratio else _format_log_tick
    axis.yaxis.set_major_formatter(FuncFormatter(formatter))
    axis.yaxis.set_minor_formatter(NullFormatter())


def write_plot(records, output_dir, filename, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('Install benchmark plotting deps with .[bench]') \
            from exc

    labels = [f"{record['method']}:{record['case']}" for record in records]
    charex_ms = [record['charex_ms'] for record in records]
    numpy_ms = [record['numpy_ms'] for record in records]
    speedups = [record['speedup'] for record in records]
    x = np.arange(len(records))
    width = 0.38

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].bar(x - width / 2, charex_ms, width, label='charex')
    axes[0].bar(x + width / 2, numpy_ms, width, label='numpy')
    _set_log_ticks(axes[0], charex_ms + numpy_ms)
    axes[0].set_ylabel('median ms, log scale')
    axes[0].legend()

    axes[1].bar(x, speedups, color='#5470c6')
    axes[1].axhline(1, color='#333333', linewidth=1)
    _set_log_ticks(axes[1], speedups, ratio=True)
    axes[1].set_ylabel('speedup vs NumPy, log scale')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=80, ha='right', fontsize=7)

    fig.suptitle(title)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_plots(records, output_dir, size, repeat):
    written = []
    title_suffix = (
        f'Python {python_version()}, NumPy {np.__version__}, '
        f'Numba {numba.__version__}, n={size}, repeat={repeat}'
    )
    for group in ('comparison', 'occurrence', 'properties'):
        for kind in ('bytes', 'strings'):
            subset = [
                record for record in records
                if record['group'] == group and record['kind'] == kind
            ]
            if subset:
                filename = {
                    'comparison': 'comparison-operators',
                    'occurrence': 'char-occurrence',
                    'properties': 'char-properties',
                }[group] + f'-{kind}.png'
                written.append(write_plot(
                    subset, output_dir, filename,
                    f'charex {group} {kind} ({title_suffix})',
                ))

    for group in ('comparison', 'occurrence', 'properties'):
        subset = [
            record for record in records
            if record['group'] == group and record['kind'] == 'stringdtype'
        ]
        if subset:
            filename = f'stringdtype-{group}.png'
            written.append(write_plot(
                subset, output_dir, filename,
                f'charex StringDType {group} ({title_suffix})',
            ))

    numerics = [
        record for record in records
        if record['group'] == 'numerics' and record['kind'] == 'strings'
    ]
    if numerics:
        written.append(write_plot(
            numerics, output_dir, 'char-numerics-strings.png',
            f'charex numeric string predicates ({title_suffix})',
        ))
    stringdtype_numerics = [
        record for record in records
        if record['group'] == 'numerics' and record['kind'] == 'stringdtype'
    ]
    if stringdtype_numerics:
        written.append(write_plot(
            stringdtype_numerics, output_dir, 'stringdtype-numerics.png',
            f'charex StringDType numeric predicates ({title_suffix})',
        ))
    return written


def write_readme(output_dir, size, repeat):
    output_path = output_dir / 'README.md'
    stringdtype = ''
    if _STRINGS is not None and _STRING_DTYPE is not None:
        stringdtype = (
            '\nThis matrix includes fixed-width `np.char` inputs and '
            'NumPy 2.x `StringDType` inputs through `np.strings`.\n'
        )
    output_path.write_text(
        '# charex benchmark matrix\n\n'
        f'- Python: `{python_version()}`\n'
        f'- NumPy: `{np.__version__}`\n'
        f'- Numba: `{numba.__version__}`\n'
        f'- llvmlite: `{llvmlite.__version__}`\n'
        f'- Size: `{size}`\n'
        f'- Repeat: `{repeat}`\n\n'
        f'Timings are medians from interleaved charex/NumPy calls with '
        f'`{TIMING_BATCH}` calls per timed sample.\n'
        f'{stringdtype}\n'
        'Regenerate from the repository root:\n\n'
        '```bash\n'
        'python -m pip install -e ".[bench]"\n'
        f'python charex/benchmarks/matrix.py --size {size} --repeat {repeat}\n'
        '```\n',
        encoding='utf-8',
    )
    return output_path


def main():
    parser = ArgumentParser()
    parser.add_argument('--size', type=int, default=50_000)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('docs/benchmarks/numba-v-0.65.1'))
    args = parser.parse_args()

    print(f'python/numpy/numba/llvmlite: {python_version()} / '
          f'{np.__version__} / {numba.__version__} / {llvmlite.__version__}')
    print(f'size={args.size} repeat={args.repeat}')

    strings = string_values(args.size)
    bytes_ = byte_values(args.size)
    records = []
    records.extend(comparison_records('strings', strings, args.repeat))
    records.extend(comparison_records('bytes', bytes_, args.repeat))
    records.extend(occurrence_records('strings', strings, args.repeat))
    records.extend(occurrence_records('bytes', bytes_, args.repeat))
    records.extend(property_records('strings', strings, args.repeat))
    records.extend(property_records('bytes', bytes_, args.repeat))
    records.extend(numeric_records(strings, args.repeat))
    if _STRINGS is not None and _STRING_DTYPE is not None:
        stringdtype = stringdtype_values(args.size)
        records.extend(comparison_records(
            'stringdtype', stringdtype, args.repeat, STRINGS_COMPARISON_FUNCS))
        records.extend(occurrence_records(
            'stringdtype', stringdtype, args.repeat, STRINGS_OCCURRENCE_FUNCS))
        records.extend(property_records(
            'stringdtype', stringdtype, args.repeat, STRINGS_PROPERTY_FUNCS))
        records.extend(numeric_records(
            stringdtype, args.repeat, 'stringdtype', STRINGS_NUMERIC_FUNCS))

    print(f'wrote {write_csv(records, args.output_dir, args.size, args.repeat)}')
    for path in write_plots(records, args.output_dir, args.size, args.repeat):
        print(f'wrote {path}')
    print(f'wrote {write_readme(args.output_dir, args.size, args.repeat)}')


if __name__ == '__main__':
    main()
