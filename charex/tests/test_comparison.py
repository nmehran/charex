"""Test numpy character comparison operators"""

from charex.tests.support import measure_performance
from numba import njit
import numpy as np
import charex


def run_test(implementation, baseline, *args, **kwargs):
    if '__msg' in kwargs:
        print(__msg := kwargs.pop('__msg'))
    comparison = implementation(*args, **kwargs) == baseline(*args, **kwargs)
    assert np.all(comparison)


def measure_test(implementation, baseline, *args, **kwargs):
    measure_performance(implementation, 10, *args, **kwargs)
    measure_performance(baseline, 10, *args, **kwargs)


def test_equal(byte_arguments, string_arguments):
    """Test numpy.char.equal"""

    @njit(nogil=True, cache=True)
    def numba_char_equal(x1, x2):
        return np.char.equal(x1, x2)

    print('\ntest_equal::Byte Tests:')
    for i, arguments in enumerate(byte_arguments):
        run_test(numba_char_equal, np.char.equal, *arguments, __msg=i)
        measure_test(numba_char_equal, np.char.equal, *arguments)

    print('\ntest_equal::String Tests:')
    for i, arguments in enumerate(string_arguments):
        run_test(numba_char_equal, np.char.equal, *arguments, __msg=i)
        measure_test(numba_char_equal, np.char.equal, *arguments)


def test_not_equal(byte_arguments, string_arguments):
    """Test numpy.char.not_equal"""

    @njit(nogil=True, cache=True)
    def numba_char_not_equal(x1, x2):
        return np.char.not_equal(x1, x2)

    print('\ntest_not_equal::Byte Tests:')
    for i, arguments in enumerate(byte_arguments):
        run_test(numba_char_not_equal, np.char.not_equal, *arguments, __msg=i)
        measure_test(numba_char_not_equal, np.char.not_equal, *arguments)

    print('\ntest_not_equal::String Tests:')
    for i, arguments in enumerate(string_arguments):
        run_test(numba_char_not_equal, np.char.not_equal, *arguments, __msg=i)
        measure_test(numba_char_not_equal, np.char.not_equal, *arguments)


def test_greater(byte_arguments, string_arguments):
    """Test numpy.char.greater"""

    @njit(nogil=True, cache=True)
    def numba_char_greater(x1, x2):
        return np.char.greater(x1, x2)

    print('\ntest_greater::Byte Tests:')
    for i, arguments in enumerate(byte_arguments):
        run_test(numba_char_greater, np.char.greater, *arguments, __msg=i)
        measure_test(numba_char_greater, np.char.greater, *arguments)

    print('\ntest_greater::String Tests:')
    for i, arguments in enumerate(string_arguments):
        run_test(numba_char_greater, np.char.greater, *arguments, __msg=i)
        measure_test(numba_char_greater, np.char.greater, *arguments)


def test_greater_equal(byte_arguments, string_arguments):
    """Test numpy.char.greater_equal"""

    @njit(nogil=True, cache=True)
    def numba_char_greater_equal(x1, x2):
        return np.char.greater_equal(x1, x2)

    print('\ntest_greater_equal::Byte Tests:')
    for i, arguments in enumerate(byte_arguments):
        run_test(numba_char_greater_equal, np.char.greater_equal, *arguments, __msg=i)
        measure_test(numba_char_greater_equal, np.char.greater_equal, *arguments)

    print('\ntest_greater_equal::String Tests:')
    for i, arguments in enumerate(string_arguments):
        run_test(numba_char_greater_equal, np.char.greater_equal, *arguments, __msg=i)
        measure_test(numba_char_greater_equal, np.char.greater_equal, *arguments)


def test_less(byte_arguments, string_arguments):
    """Test numpy.char.less"""

    @njit(nogil=True, cache=True)
    def numba_char_less(x1, x2):
        return np.char.less(x1, x2)

    print('\ntest_less::Byte Tests:')
    for i, arguments in enumerate(byte_arguments):
        run_test(numba_char_less, np.char.less, *arguments, __msg=i)
        measure_test(numba_char_less, np.char.less, *arguments)

    print('\ntest_less::String Tests:')
    for i, arguments in enumerate(string_arguments):
        run_test(numba_char_less, np.char.less, *arguments, __msg=i)
        measure_test(numba_char_less, np.char.less, *arguments)


def test_less_equal(byte_arguments, string_arguments):
    """Test numpy.char.less_equal"""

    @njit(nogil=True, cache=True)
    def numba_char_less_equal(x1, x2):
        return np.char.less_equal(x1, x2)

    print('\ntest_less_equal::Byte Tests:')
    for i, arguments in enumerate(byte_arguments):
        run_test(numba_char_less_equal, np.char.less_equal, *arguments, __msg=i)
        measure_test(numba_char_less_equal, np.char.less_equal, *arguments)

    print('\ntest_less_equal::String Tests:')
    for i, arguments in enumerate(string_arguments):
        run_test(numba_char_less_equal, np.char.less_equal, *arguments, __msg=i)
        measure_test(numba_char_less_equal, np.char.less_equal, *arguments)


def main():
    byte_arguments = [
        (B, B), (B, C), (B, D), (D, B), (E, F), (X.astype('S'), Y.astype('S')),
        (B, b'hello'), (b'hello', B), (D, b'hello'),
        (b'hello', b'hella'), (b'hello' * 1000, b'hello' * 1000)
    ]
    string_arguments = [
        (S, S), (S, T), (S, U), (U, S), (V, W), (X, Y),
        (S, 'hello'), ('hello', S), (U, 'hello'),
        ('hello', 'hella'), ('hello' * 1000, 'hello' * 1000)
    ]
    test_equal(byte_arguments, string_arguments)
    test_not_equal(byte_arguments, string_arguments)
    test_greater(byte_arguments, string_arguments)
    test_greater_equal(byte_arguments, string_arguments)
    test_less(byte_arguments, string_arguments)
    test_less_equal(byte_arguments, string_arguments)


if __name__ == '__main__':
    np.random.seed(1)

    B = np.random.choice([b'hello', b'all', b'worlds'], 10_000)
    C = np.random.choice([b'hello', b'\tFrom', b'all', b'Around', b'\nthe' b' World'], 10_000)
    D = np.random.choice([b'hello', b'all', b'worlds'], 10_000).astype('S200')
    E = np.random.choice([chr(__i) for __i in range(1, 128)], 10_000)
    F = np.random.choice(E, 10_000)

    S = B.astype('U')
    T = C.astype('U')
    U = D.astype('U')
    V = E.astype('U')
    W = F.astype('U')

    # With an efficient implementation of whitespace removal, trailing \t\n\r\f\v characters can be supported.
    X = np.random.choice([''.join([chr(np.random.randint(33, 127)) for _ in range(np.random.randint(1, 50))])
                          for _ in range(100)], 10_000)
    Y = np.random.choice(X, 10_000)

    main()
