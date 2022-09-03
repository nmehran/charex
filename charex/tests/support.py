from time import perf_counter, process_time_ns
from numpy import median, ndarray, zeros


def measure_performance(func, n: int = 5, *args, **kwargs) -> ndarray:
    """
    measure_performance(func, n=5, *args, **kwargs)

        Return timings of a given function and print the median time.

    Parameters
    ----------
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

    if not hasattr(func, '__name__'):
        func.__name__ = f'f_{process_time_ns()}'
    print(f'{func.__name__!r}: completed in {median(end_times):.10f} seconds')

    return end_times
