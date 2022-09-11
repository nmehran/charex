from itertools import product
from numpy import all, array_split, median, ndarray, zeros
from pandas import DataFrame
from time import perf_counter
from types import GeneratorType
import matplotlib.pyplot as plt
import numpy as np


class CharacterTest:
    """Methods for testing, measuring, and graphing performance of string array operations against baselines"""
    def __init__(self, impl=None, base=None, byte_args=None, string_args=None):
        self.test_count = 0
        self.name_impl, self.name_base = [], []
        self.measurements: list = [[], []]
        self.signatures: list = [[], []]
        self.plots = []

        self.byte_args = byte_args
        self.string_args = string_args
        self.__set_args(impl, base, byte_args, string_args)

    def __set_args(self, impl, base, byte_args, string_args):
        if impl or base:
            assert impl and base, 'Must set both implementation and base.'
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

    def test(self, impl=None, base=None, byte_args=None, string_args=None):
        self.__set_args(impl, base, byte_args, string_args)
        for m, args in enumerate(self.byte_args):
            self.signatures[0].append(', '.join(signature(a) for a in args))
            run_test(self.impl, self.base, *args, __msg=f"Test={m+1}, args=({self.signatures[0][m]})")
        for m, args in enumerate(self.string_args):
            self.signatures[1].append(', '.join(signature(a) for a in args))
            run_test(self.impl, self.base, *args, __msg=f"Test={m+1}, args=({self.signatures[1][m]})")
        self.is_tested = True
        self.test_count += 1

    def measure(self, impl=None, base=None, byte_args=None, string_args=None):
        if not self.is_tested:
            self.test()
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
                              test_names=np.char.add('bytes: ', self.name_base),
                              columns=columns,
                              main_title=main_title,
                              x_label='test number',
                              y_label='median time (milliseconds)'),
            graph_performance(measurements=self.measurements[1],
                              test_names=np.char.add('string: ', self.name_base),
                              columns=columns,
                              main_title=main_title,
                              x_label='test number',
                              y_label='median time (milliseconds)')
        ]

    def run(self, method: str, impl=None, base=None, byte_args=None, string_args=None, **method_kwargs):
        methods = {'test': self.test, 'measure': self.measure, 'graph': self.graph}
        run_method = methods.get(method)
        if not run_method:
            print("Method must be in ('test', 'measure', 'graph')")
        else:
            self.__set_args(impl, base, byte_args, string_args)
            run_method(**method_kwargs)


class StandardTest:
    """Methods for testing, measuring, and graphing performance of implementations against baselines"""
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
            assert impl and base, 'Must set both implementation and base.'
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

    def test(self, impl=None, base=None, args=None):
        self.__set_args(impl, base, args)
        for m, args in enumerate(self.args):
            self.signatures.append(', '.join(signature(a) for a in args))
            run_test(self.impl, self.base, *args, __msg=f"Test={m+1}, args=({self.signatures[m]})")
        self.is_tested = True
        self.test_count += 1

    def measure(self, impl=None, base=None, args=None, scale=1):
        if not self.is_tested:
            self.test()
        else:
            self.__set_args(impl, base, args)

        for args in self.args:
            self.measurements.append(median(measure_test(self.impl, self.base, *args), axis=1)*scale)
        self.is_measured = True

    def graph(self, impl=None, base=None, args=None,
              main_title=None, prefix=None, suffix=None, x_label='test number', y_label='median time'):
        if not self.is_measured:
            self.measure(impl, base, args)
        else:
            self.__set_args(impl, base, args)

        test_names = self.name_base
        if prefix:
            test_names = np.char.add(prefix, self.name_base)
        if suffix:
            test_names = np.char.add(suffix, self.name_base)

        self.plots += [
            graph_performance(measurements=self.measurements[0],
                              test_names=test_names,
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
            self.is_tested = False
            self.__set_args(impl, base, args)
            run_method(**method_kwargs)


def run_test(implementation, baseline, *args, **kwargs) -> None:
    """Assert implementation reflects the baseline"""
    __msg = ''
    if '__msg' in kwargs:
        print(__msg := kwargs.pop('__msg'))
    comparison = implementation(*args, **kwargs) == baseline(*args, **kwargs)
    assert all(comparison), f'all({implementation.__name__} == {baseline.__name__}) -> {__msg}'


def measure_test(implementation, baseline, *args, **kwargs):
    """Return pair of measurements for implementation and baseline"""
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


def graph_performance(measurements, test_names: list,
                      columns=1, kind='bar', main_title=None, x_label=None, y_label=None, show=True):
    if not len(measurements):
        return []

    func_count = len(test_names)
    columns = min(len(test_names), columns)
    rows = max(func_count // columns, 1)

    def set_fig_size(test_count, columns_, rows_, scale_factor=0.5, size_y=7):
        x_dim = scale_factor * test_count * columns_
        y_dim = rows_ * size_y
        return max(x_dim, size_y * 0.8), y_dim

    def plot_graph(samples, titles):
        test_count = len(samples) // func_count
        functions = array_split(samples, func_count)
        x_ticks = range(test_count)

        fig, axs = plt.subplots(rows, columns)
        fig_size = set_fig_size(test_count, columns, rows)

        if func_count == 1:
            axs = np.array([[axs]], dtype=object)
        axs = axs.reshape(-1, columns)

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
        fig.suptitle(main_title, fontsize=max(11, int(fig_size[0])))
        fig.tight_layout(pad=3, rect=[0.05, 0.02, 0.95, 0.98])
        if show:
            fig.show()
        return fig, axs

    figure, axes = plot_graph(samples=DataFrame(measurements, columns=['implementation', 'baseline']), titles=test_names)
    return figure, axes


def pack_arguments(main_args: (tuple, list), args: (tuple, list)):
    """Generate combinations of arguments for a list of main arguments"""
    arg_product = tuple(product(*args))
    for a in main_args:
        for args in arg_product:
            yield a, *args


def signature(arg):
    """Describe signature of homogenous types"""
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


def arguments_as_bytes(args: list):
    """Yield byte counterparts of string arguments, given a tuple of arguments"""
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
