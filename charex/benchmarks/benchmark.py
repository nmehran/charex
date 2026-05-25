from argparse import ArgumentParser
from csv import DictWriter
from pathlib import Path
from platform import python_version
from time import perf_counter

import charex  # noqa: F401
import llvmlite
import numba
import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def jit_equal(left, right):
    return np.char.equal(left, right)


@njit(nogil=True, cache=True)
def jit_count(values, sub):
    return np.char.count(values, sub)


@njit(nogil=True, cache=True)
def jit_isalpha(values):
    return np.char.isalpha(values)


def median_time(func, args, repeat):
    times = np.empty(repeat, dtype=np.float64)
    for i in range(repeat):
        start = perf_counter()
        result = func(*args)
        times[i] = perf_counter() - start
    return float(np.median(times)), result


def bench(label, jit_func, numpy_func, args, repeat):
    expected = numpy_func(*args)
    actual = jit_func(*args)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    jit_time, _ = median_time(jit_func, args, repeat)
    numpy_time, _ = median_time(numpy_func, args, repeat)
    ratio = numpy_time / jit_time if jit_time else float('inf')
    print(f'{label:18} charex={jit_time * 1000:9.3f} ms  '
          f'numpy={numpy_time * 1000:9.3f} ms  speedup={ratio:6.2f}x')
    return {
        'label': label,
        'charex_ms': jit_time * 1000,
        'numpy_ms': numpy_time * 1000,
        'speedup': ratio,
    }


def file_stem():
    return (
        f'py-{python_version()}_numpy-{np.__version__}_'
        f'numba-{numba.__version__}'
    ).replace('.', '-')


def write_csv(records, output_dir, size, repeat):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{file_stem()}.csv'
    fieldnames = [
        'python', 'numpy', 'numba', 'llvmlite', 'size', 'repeat',
        'label', 'charex_ms', 'numpy_ms', 'speedup',
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
    ticks = []
    for exponent in range(-4, 5):
        scale = 10.0 ** exponent
        for multiple in (1, 2, 3, 4, 5, 10):
            value = multiple * scale
            if lower <= value <= upper:
                ticks.append(value)
    if not ticks:
        ticks = [value for value in values if value > 0]
    axis.set_yticks(sorted(set(ticks)))
    formatter = _format_ratio_tick if ratio else _format_log_tick
    axis.yaxis.set_major_formatter(FuncFormatter(formatter))
    axis.yaxis.set_minor_formatter(NullFormatter())


def write_plot(records, output_dir, size, repeat):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('Install benchmark plotting deps with .[bench]') \
            from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [record['label'] for record in records]
    charex_ms = [record['charex_ms'] for record in records]
    numpy_ms = [record['numpy_ms'] for record in records]
    speedups = [record['speedup'] for record in records]
    x = np.arange(len(records))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
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
    axes[1].set_xticklabels(labels, rotation=30, ha='right')

    fig.suptitle(
        'charex benchmark '
        f'({python_version()}, NumPy {np.__version__}, '
        f'Numba {numba.__version__}, n={size}, repeat={repeat})'
    )
    output_path = output_dir / f'{file_stem()}.png'
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def make_strings(size):
    left = np.array([f'item-{i:06d}' for i in range(size)], dtype='U16')
    right = left.copy()
    right[::3] = np.array([f'item-{i:06d}x' for i in range(0, size, 3)],
                          dtype='U16')
    properties = np.resize(
        np.array(['Alpha', 'αβγ', '١٢٣', 'Ⅷ', '一二', 'abc123', ''],
                 dtype='U8'),
        size,
    )
    return left, right, properties


def make_bytes(size):
    left = np.array([f'item-{i:06d}'.encode() for i in range(size)],
                    dtype='S16')
    right = left.copy()
    right[::3] = np.array([f'item-{i:06d}x'.encode()
                           for i in range(0, size, 3)], dtype='S16')
    properties = np.resize(
        np.array([b'Alpha', b'abc', b'123', b'abc123', b'', b' '],
                 dtype='S8'),
        size,
    )
    return left, right, properties


def main():
    parser = ArgumentParser()
    parser.add_argument('--size', type=int, default=50_000)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    print(f'python/numpy/numba/llvmlite: {python_version()} / '
          f'{np.__version__} / {numba.__version__} / {llvmlite.__version__}')
    print(f'size={args.size} repeat={args.repeat}')

    strings_left, strings_right, strings_props = make_strings(args.size)
    bytes_left, bytes_right, bytes_props = make_bytes(args.size)

    records = [
        bench('strings equal', jit_equal, np.char.equal,
              (strings_left, strings_right), args.repeat),
        bench('strings count', jit_count, np.char.count,
              (strings_left, 'item'), args.repeat),
        bench('strings isalpha', jit_isalpha, np.char.isalpha,
              (strings_props,), args.repeat),
        bench('bytes equal', jit_equal, np.char.equal,
              (bytes_left, bytes_right), args.repeat),
        bench('bytes count', jit_count, np.char.count,
              (bytes_left, b'item'), args.repeat),
        bench('bytes isalpha', jit_isalpha, np.char.isalpha,
              (bytes_props,), args.repeat),
    ]

    output_dir = args.output_dir
    if args.plot and output_dir is None:
        output_dir = Path('charex') / 'benchmarks' / f'numba-v-{numba.__version__}'
    if output_dir:
        print(f'wrote {write_csv(records, output_dir, args.size, args.repeat)}')
    if args.plot:
        print(f'wrote {write_plot(records, output_dir, args.size, args.repeat)}')


if __name__ == '__main__':
    main()
