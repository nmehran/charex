from itertools import product
from numpy import all, array_split, median, ndarray, zeros
from pandas import DataFrame
from time import perf_counter
from types import GeneratorType
import matplotlib.pyplot as plt
import numpy as np


class CharacterTest:
    def __init__(self, impl, base, byte_args, string_args):
        self.name_impl = impl.__name__
        self.name_base = base.__name__
        self.test_count = 0

        self.impl = impl
        self.base = base
        self.byte_args = byte_args
        self.string_args = string_args

        self.is_measured = False
        self.is_tested = False
        self.measurements: list = [[], []]
        self.plots = []

    def test(self):
        for m, args in enumerate(self.byte_args, start=1):
            run_test(self.impl, self.base, *args, __msg=f"Test={m}, args=({', '.join(signature(a) for a in args)})")
        for m, args in enumerate(self.string_args, start=1):
            run_test(self.impl, self.base, *args, __msg=f"Test={m}, args=({', '.join(signature(a) for a in args)})")
        self.is_tested = True

    def measure(self):
        if not self.is_tested:
            if isinstance(self.byte_args, GeneratorType):
                self.byte_args = list(self.byte_args)
            if isinstance(self.byte_args, GeneratorType):
                self.string_args = list(self.string_args)
            self.test()
        for args in self.byte_args:
            self.measurements[0].append(median(measure_test(self.impl, self.base, *args), axis=1)*1000)
        for args in self.string_args:
            self.measurements[1].append(median(measure_test(self.impl, self.base, *args), axis=1)*1000)
        self.is_measured = True

    def graph(self):
        if not self.is_measured:
            self.measure()
        self.plots += [
            graph_performance(measurements=self.measurements[0], func_count=1, test_names=[f'bytes: {self.name_base}']),
            graph_performance(measurements=self.measurements[1], func_count=1, test_names=[f'string: {self.name_base}'])
        ]

    def run(self, method: str):
        methods = {'test': self.test, 'measure': self.measure, 'graph': self.graph}
        run_method = methods.get(method)
        if not run_method:
            print("Method must be in ('test', 'measure', 'graph')")
        else:
            run_method()


class StandardTest:
    def __init__(self, impl, base, args):
        self.name_impl = impl.__name__
        self.name_base = base.__name__
        self.test_count = 0

        self.impl = impl
        self.base = base
        self.args = args

        self.is_measured = False
        self.is_tested = False
        self.measurements: list = [[], []]
        self.plots = []

    def test(self):
        for m, args in enumerate(self.args, start=1):
            run_test(self.impl, self.base, *args, __msg=f"Test={m}, args=({', '.join(signature(a) for a in args)})")
        self.is_tested = True

    def measure(self):
        if not self.is_tested:
            if isinstance(self.args, GeneratorType):
                self.args = list(self.args)
            self.test()
        for args in self.args:
            self.measurements[0].append(median(measure_test(self.impl, self.base, *args), axis=1)*1000)
        self.is_measured = True

    def graph(self, title_prefix='', title_suffix=''):
        if not self.is_measured:
            self.measure()
        self.plots += [
            graph_performance(measurements=self.measurements[0], func_count=1,
                              test_names=[f'{title_prefix}{self.name_base}{title_suffix}'])
        ]

    def run(self, method: str):
        methods = {'test': self.test, 'measure': self.measure, 'graph': self.graph}
        run_method = methods.get(method)
        if not run_method:
            print("Method must be in ('test', 'measure', 'graph')")
        else:
            run_method()


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


def test_char_function(func, baseline, byte_arguments, string_arguments):
    if not hasattr(test_char_function, 'count'):
        test_char_function.count = 0
        test_char_function.test_names = []
        test_char_function.measurements = [[], []]

    test_char_function.count += 1
    test_char_function.test_names.append(baseline.__name__)
    test_char_function.measurements[0] += test_and_measure(func, baseline, byte_arguments, __msg='Byte Tests')
    test_char_function.measurements[1] += test_and_measure(func, baseline, string_arguments, __msg='String Tests')


def test_and_measure(impl, base, *args, **kwargs):
    """
    Test implementation against baseline and yield a list of tuple measurements

    Parameters
    ----------
    impl : function
        implementation of function which mirrors baseline
    base: function
        baseline function which against implementation is tested
    args : arguments, optional
        list of function arguments to be used in tests
    kwargs : key-word arguments, optional
        function key-word arguments to be used in tests
    """

    test_label = impl.__name__
    if '__msg' in kwargs:
        print(f"\n{test_label!r}::{kwargs.pop('__msg')}:")

    for i, arguments in enumerate(*args, start=1):
        run_test(impl, base, *arguments, __msg=i)
        yield (f'{test_label}_{i}',
               *median(measure_test(impl, base, *arguments), axis=1) * 1000)

    for i, arguments in enumerate(*args, start=1):
        run_test(impl, base, *arguments[::-1], __msg=i)
        measure_test(impl, base, *arguments[::-1])


def measure_performance(func, n: int = 5, *args, **kwargs) -> ndarray:
    """
    measure_performance(func, n=5, *args, **kwargs)

        Return timings of a given function and print the median time.

    Parameters
    ----------
    func: function
    n : int : default = 5
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
    end_times = zeros(n, dtype='float64')
    for i in range(n):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_times[i] = (perf_counter() - start_time)
    _ = type(result)

    func_name = f'f_{id(func)}' if not hasattr(func, '__name__') else func.__name__
    print(f'{func_name!r}: completed in {median(end_times):.10f} seconds')

    return end_times


def graph_performance(measurements, func_count: int, test_names: list):
    if not len(measurements):
        return []

    def set_fig_size(test_count, columns, rows, scale_factor=0.5, size_y=7):
        x_dim = scale_factor * test_count * columns
        y_dim = rows * size_y
        return max(x_dim, size_y * 0.8), y_dim

    def plot_graph(graph, titles, columns=1, main_title=None, as_subplot=False):
        samples = len(graph)
        test_count = samples // func_count
        rows = max(func_count // columns, 1)
        x_ticks = range(test_count)

        fig = axes = None
        if func_count == 1:
            functions = [graph]
        else:
            functions = array_split(graph, func_count)
            fig, axes = plt.subplots(rows, columns)

        fig_size = set_fig_size(test_count, columns, rows)
        figs = []
        f = 0
        for r in range(rows):
            for c in range(columns):
                fp = functions[f].plot(ax=axes and axes[r, c],
                                       figsize=fig_size,
                                       kind='bar',
                                       logy=True,
                                       title=titles[f],
                                       xlabel='test number',
                                       xticks=x_ticks,
                                       ylabel='median time (milliseconds)')
                figs.append(fp)
                f += 1
        plt.suptitle(main_title, fontsize=max(11, int(fig_size[0])))
        if axes and as_subplot:
            return fig
        plt.show()
        return figs

    figures = plot_graph(graph=DataFrame(measurements, columns=['implementation', 'baseline']),
                         titles=test_names,
                         main_title='Measured Test Performance (milliseconds)')
    return figures


def pack_arguments(arrays: (tuple, list), args: (tuple, list)):
    arg_product = tuple(product(*args))
    for a in arrays:
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
