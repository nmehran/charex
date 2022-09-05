from charex.tests.support import measure_performance
from numba import njit, jit
import numpy as np
import charex


def run_tests(implementation, baseline, *args, **kwargs):
    if '__iter' in kwargs:
        print(__iter := kwargs.pop('__iter'))
    comparison = implementation(*args, **kwargs) == baseline(*args, **kwargs)
    assert np.all(comparison)


def measure_tests(implementation, baseline, *args, **kwargs):
    measure_performance(implementation, 10, *args, **kwargs)
    measure_performance(baseline, 10, *args, **kwargs)


def test_equal():
    """Test numpy.char.equal"""

    @njit(nogil=True, cache=True)
    def numba_char_equal(x1, x2):
        return np.char.equal(x1, x2)

    print('\ntest_equal::Byte Tests:')

    test_arguments = [
        (B, B), (B, C), (B, D), (D, B), (E, F),
        (B, b'hello'), (b'hello', B), (D, b'hello'),
        (b'hello', b'hella'), (b'hello'*1000, b'hello'*1000)
    ]
    for i, arguments in enumerate(test_arguments):
        run_tests(numba_char_equal, np.char.equal, *arguments, __iter=i)
        measure_tests(numba_char_equal, np.char.equal, *arguments)

    print('\ntest_equal::String Tests:')

    test_arguments = [
        (S, S), (S, T), (S, U), (U, S), (V, W),
        (S, 'hello'), ('hello', S), (U, 'hello'),
        ('hello', 'hella'), ('hello'*1000, 'hello'*1000)
    ]
    for i, arguments in enumerate(test_arguments):
        run_tests(numba_char_equal, np.char.equal, *arguments, __iter=i)
        measure_tests(numba_char_equal, np.char.equal, *arguments)


def main():
    test_equal()


if __name__ == '__main__':
    np.random.seed(1)

    B = np.random.choice([b'hello', b'all', b'worlds'], 10_000)
    C = np.random.choice([b'hello', b'all', b'worlds'], 10_000)
    D = np.random.choice([b'hello', b'all', b'worlds'], 10_000).astype('S200')
    E = np.random.choice([chr(__i) for __i in range(1, 255)], 10_000)
    F = np.random.choice([chr(__i) for __i in range(1, 255)], 10_000)

    S = B.astype('U')
    T = C.astype('U')
    U = D.astype('U')
    V = E.astype('U')
    W = F.astype('U')

    main()
