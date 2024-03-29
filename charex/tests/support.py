from itertools import product
from math import ceil
from numpy import all, array_split, median, ndarray, zeros
from pandas import DataFrame
from time import perf_counter
from types import GeneratorType
import matplotlib.pyplot as plt
import numpy as np


class CharacterTest:
    """Methods for testing, measuring, and graphing performance of string array operations against their baseline."""
    def __init__(self, impl=None, base=None, byte_args=None, string_args=None):
        self.test_count = 0
        self.name_impl, self.name_base = [], []
        self.measurements: list = [[], []]
        self.signatures: list = [[], []]
        self.plots = []

        self.byte_args = byte_args
        self.string_args = string_args
        self.impl = self.base = None
        self.is_tested = self.is_measured = False
        self.__set_args(impl, base, byte_args, string_args)

    def __set_args(self, impl, base, byte_args, string_args):
        if impl or base:
            assert impl and base, 'Must set both implementation and baseline.'
            self.impl = impl
            self.base = base
            if not self.name_impl or self.name_impl[-1] != impl:
                self.is_tested = False
                self.is_measured = False
            self.name_impl.append(impl.__name__)
            self.name_base.append(base.__name__)

        self.byte_args = byte_args or self.byte_args
        self.string_args = string_args or self.string_args
        if isinstance(self.byte_args, GeneratorType):
            self.byte_args = list(byte_args)
        if isinstance(self.string_args, GeneratorType):
            self.string_args = list(string_args)

    @staticmethod
    def __describe_args(args):
        return ', '.join(str(a) if isinstance(a, (int, float, type(None)))
                         else str(a) if isinstance(a, (bytes, str)) and len(a) <= 10
                         else signature(a) for a in args)

    def test(self, impl=None, base=None, byte_args=None, string_args=None):
        self.__set_args(impl, base, byte_args, string_args)
        if not self.impl:
            raise AttributeError('Implementation not set before calling test function')
        for m, args in enumerate(self.byte_args):
            self.signatures[0].append(self.__describe_args(args))
            run_test(self.impl, self.base, *args,
                     __msg=f"Test={m+1}, base='{self.name_base[-1]}', args=({self.signatures[0][m]})")
        for m, args in enumerate(self.string_args):
            self.signatures[1].append(self.__describe_args(args))
            run_test(self.impl, self.base, *args,
                     __msg=f"Test={m+1}, base='{self.name_base[-1]}', args=({self.signatures[1][m]})")
        self.is_tested = True
        self.test_count += 1

    def measure(self, impl=None, base=None, byte_args=None, string_args=None):
        if not self.is_tested:
            self.test(impl, base, byte_args, string_args)
        else:
            self.__set_args(impl, base, byte_args, string_args)

        for args in self.byte_args:
            self.measurements[0].append(median(measure_test(self.impl, self.base, *args), axis=1)*1000)
        for args in self.string_args:
            self.measurements[1].append(median(measure_test(self.impl, self.base, *args), axis=1)*1000)
        self.is_measured = True

    def graph(self, impl=None, base=None, byte_args=None, string_args=None, main_title=None, columns=2):
        if not self.is_measured:
            self.measure(impl, base, byte_args, string_args)
        else:
            self.__set_args(impl, base, byte_args, string_args)

        self.plots += [
            graph_performance(measurements=self.measurements[0],
                              test_titles=np.char.add('bytes: ', self.name_base),
                              test_labels=self.signatures[0],
                              main_title=main_title or 'Measured Performance (ms): Bytes',
                              columns=columns,
                              x_label='test number',
                              y_label='median time (milliseconds)'),
            graph_performance(measurements=self.measurements[1],
                              test_titles=np.char.add('string: ', self.name_base),
                              test_labels=self.signatures[1],
                              columns=columns,
                              main_title=main_title or 'Measured Performance (ms): Strings',
                              x_label='test number',
                              y_label='median time (milliseconds)')
        ]

    def run(self, method: str, impl=None, base=None, byte_args=None, string_args=None, **method_kwargs):
        methods = {'test': self.test, 'measure': self.measure, 'graph': self.graph}
        run_method = methods.get(method)
        if not run_method:
            print("Method must be in ('test', 'measure', 'graph')")
        else:
            self.is_tested = self.is_measured = False
            self.__set_args(impl, base, byte_args, string_args)
            if not self.impl:
                return 'Implementation not set.'
            run_method(**method_kwargs)


class StandardTest:
    """Methods for testing, measuring, and graphing performance of implementations against their baseline."""
    def __init__(self, impl=None, base=None, args=None):
        self.test_count = 0
        self.name_impl, self.name_base = [], []
        self.measurements = []
        self.signatures = []
        self.plots = []

        self.args = args
        self.__set_args(impl, base, args)

    def __set_args(self, impl, base, args):
        if impl or base:
            assert impl and base, 'Must set both implementation and baseline.'
            self.impl = impl
            self.base = base
            if not self.name_impl or self.name_impl[-1] != impl:
                self.is_tested = False
                self.is_measured = False
            self.name_impl.append(impl.__name__)
            self.name_base.append(base.__name__)

        self.args = args or self.args
        if isinstance(self.args, GeneratorType):
            self.args = list(args)

    @staticmethod
    def __describe_args(args):
        return ', '.join(str(a) if isinstance(a, (int, float, type(None)))
                         else str(a) if isinstance(a, (bytes, str)) and len(a) <= 10
                         else signature(a) for a in args)

    def test(self, impl=None, base=None, args=None):
        self.__set_args(impl, base, args)
        if not self.impl:
            raise AttributeError('Implementation not set before calling test function')
        for m, args in enumerate(self.args):
            self.signatures.append(self.__describe_args(args))
            run_test(self.impl, self.base, *args,
                     __msg=f"Test={m+1}, base='{self.name_base[-1]}', args=({self.signatures[m]})")
        self.is_tested = True
        self.test_count += 1

    def measure(self, impl=None, base=None, args=None, scale=1):
        if not self.is_tested:
            self.test(impl, base, args)
        else:
            self.__set_args(impl, base, args)

        for args in self.args:
            self.measurements.append(median(measure_test(self.impl, self.base, *args), axis=1)*scale)
        self.is_measured = True

    def graph(self, impl=None, base=None, args=None,
              main_title=None, prefix=None, suffix=None,
              x_label='test number', y_label='median time'):
        if not self.is_measured:
            self.measure(impl, base, args)
        else:
            self.__set_args(impl, base, args)

        base_names = self.name_base
        if prefix:
            base_names = np.char.add(prefix, self.name_base)
        if suffix:
            base_names = np.char.add(suffix, self.name_base)

        self.plots += [
            graph_performance(measurements=self.measurements,
                              test_titles=base_names,
                              main_title=main_title,
                              x_label=x_label,
                              y_label=y_label),
        ]

    def run(self, method: str, impl=None, base=None, args=None, **method_kwargs):
        methods = {'test': self.test, 'measure': self.measure, 'graph': self.graph}
        run_method = methods.get(method)
        if not run_method:
            print("Method must be in ('test', 'measure', 'graph')")
        else:
            self.is_tested = self.is_measured = False
            self.__set_args(impl, base, args)
            if not self.impl:
                return 'Implementation not set.'
            run_method(**method_kwargs)


def run_test(implementation, baseline, *args, **kwargs) -> None:
    """Assert implementation reflects the baseline."""
    __msg = ''
    if '__msg' in kwargs:
        print(__msg := kwargs.pop('__msg'))
    im = implementation(*args, **kwargs)
    ba = baseline(*args, **kwargs)
    assert im.dtype == ba.dtype, f'{im.dtype} == {ba.dtype} -> {__msg}'
    assert im.shape == ba.shape, f'{im.shape} == {ba.shape} -> {__msg}'
    assert all(im == ba), f'all({implementation.__name__} == {baseline.__name__}) -> {__msg}'


def measure_test(implementation, baseline, *args, **kwargs):
    """Return pair of measurements for implementation and baseline."""
    m1 = measure_performance(implementation, 10, *args, **kwargs)
    m2 = measure_performance(baseline, 10, *args, **kwargs)
    return m1, m2


def measure_performance(func, n_trials: int = 5, *args, **kwargs) -> ndarray:
    """
    measure_performance(func, n=5, *args, **kwargs)

        Return timings of a given function and print the median time.

    Parameters
    ----------
    func: function
    n_trials : int : default = 5
        number of iterations to compute median time
    args : arguments, optional
        function arguments
    kwargs : key-word arguments, optional
        function key-word arguments
    Returns
    -------
    end_times : ndarray
        An array of calculated times.
    """
    result = None
    end_times = zeros(n_trials, dtype='float64')
    for i in range(n_trials):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_times[i] = (perf_counter() - start_time)
    _ = type(result)

    func_name = f'f_{id(func)}' if not hasattr(func, '__name__') else func.__name__
    print(f'{func_name!r}: completed in {median(end_times):.10f} seconds')

    return end_times


def graph_performance(measurements, test_titles: list, test_labels=None, main_title=None,
                      kind='bar', columns=1, x_label=None, y_label=None, show=True):
    """Graph measured performance timings implementations."""
    if not len(measurements):
        return []

    func_count = len(test_titles)
    columns = min(len(test_titles), columns)
    rows = max(ceil(func_count / columns), 1)

    def set_fig_size(test_count, columns_, rows_, scale_factor=0.5, size_y=7):
        x_dim = scale_factor * test_count * columns_
        y_dim = rows_ * size_y
        return max(x_dim, size_y * columns_), y_dim

    def plot_graph(samples, titles):
        functions = array_split(samples, func_count)
        test_count = len(samples) // func_count
        x_ticks = range(test_count)

        fig, axs = plt.subplots(rows, columns)
        fig_size = set_fig_size(test_count, columns, rows)

        if func_count == 1:
            axs = np.array([[axs]], dtype=object)
        axs = axs.reshape(rows, columns)

        f = 0
        for r in range(rows):
            for c in range(columns):
                functions[f].plot(ax=axs[r, c],
                                  figsize=fig_size,
                                  kind=kind,
                                  logy=True,
                                  title=titles[f],
                                  xlabel=x_label,
                                  xticks=x_ticks,
                                  ylabel=y_label)
                f += 1
                if f == func_count:
                    break

        fig.suptitle(main_title, fontsize=max(11, int(fig_size[0]*0.8)))
        if test_labels:
            signatures = ', '.join(f"{k}: '{v}'" for k, v in dict(zip(x_ticks, test_labels)).items())
            fig.text(0.01, 0.01, ''.join(["test arguments: {\n", signatures, "}\n"]),
                     fontsize=8, wrap=True)
        fig.tight_layout(pad=3, rect=[0.05, 0.07, 0.95, 0.99])
        if show:
            fig.show()
        return fig, axs

    figure, axes = plot_graph(samples=DataFrame(measurements, columns=['implementation', 'baseline']),
                              titles=test_titles)
    return figure, axes


def pack_arguments(main_args: (list, tuple), args: (list, tuple)):
    """Generate combinations of arguments for a list of main arguments."""
    arg_product = tuple(product(*args))
    for a in main_args:
        for args in arg_product:
            yield (*a, *args)


def signature(arg):
    """Describe signature of homogenous types."""
    if isinstance(arg, np.ndarray):
        return f"{arg.dtype.name}[{'x'.join(map(str, arg.shape))}]"

    type_arg = type(arg).__name__
    if not hasattr(arg, '__len__') or not len(arg):
        return type_arg

    n = len(arg)
    if isinstance(arg, (bytes, str)):
        return f'{type_arg}{n}'
    if isinstance(arg, (list, tuple)):
        return f'{type_arg}({type(arg[0]).__name__}*{n})'
    if isinstance(arg, set):
        return f'set({type(next(iter(arg))).__name__}*{n})'
    if isinstance(arg, dict):
        sig = next(iter(arg.items()))
        return f'dict({type(sig[0]).__name__}, {type(sig[1]).__name__}])*{n}'
    return type_arg


def arguments_as_bytes(args: (list, tuple)):
    """Yield byte counterparts of string arguments, given a list of arguments."""
    for pair in args:
        as_bytes = []
        for arg in pair:
            if isinstance(arg, np.ndarray):
                as_bytes.append(arg.astype('S'))
            elif isinstance(arg, str):
                as_bytes.append(bytes(arg, 'UTF-8'))
            else:
                as_bytes.append(arg)
        yield as_bytes
