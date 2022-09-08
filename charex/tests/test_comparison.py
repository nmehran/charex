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


def test_and_measure(func, baseline, *args, **kwargs):
    if '__msg' in kwargs:
        print(f"\n{func.__name__!r}::{kwargs.pop('__msg')}:")
    for i, arguments in enumerate(*args):
        run_test(func, baseline, *arguments, __msg=i)
        measure_test(func, baseline, *arguments)
    for i, arguments in enumerate(*args):
        run_test(func, baseline, *arguments[::-1], __msg=i)
        measure_test(func, baseline, *arguments[::-1])


def test_char_function(func, baseline, byte_arguments, string_arguments):
    test_and_measure(func, baseline, byte_arguments, __msg='Byte Tests')
    test_and_measure(func, baseline, string_arguments, __msg='String Tests')


@njit(nogil=True, cache=True)
def numba_char_equal(x1, x2):
    return np.char.equal(x1, x2)


@njit(nogil=True, cache=True)
def numba_char_not_equal(x1, x2):
    return np.char.not_equal(x1, x2)


@njit(nogil=True, cache=True)
def numba_char_greater(x1, x2):
    return np.char.greater(x1, x2)


@njit(nogil=True, cache=True)
def numba_char_greater_equal(x1, x2):
    return np.char.greater_equal(x1, x2)


@njit(nogil=True, cache=True)
def numba_char_less(x1, x2):
    return np.char.less(x1, x2)


@njit(nogil=True, cache=True)
def numba_char_less_equal(x1, x2):
    return np.char.less_equal(x1, x2)


@njit(nogil=True, cache=True)
def compare_chararrays(a1, b1, cmp, rstrip):
    return np.char.compare_chararrays(a1, b1, cmp, rstrip)


def main():
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
    G, H = X.astype('S'), Y.astype('S')

    byte_arguments = [
        (B, B), (B, C), (B, D), (E, F), (G, H),
        (B, b'hello'), (D, b'hello'), (D, b'hello'*100), (D, b'hello'*10_000),
        (b'hello', b'hella'), (b'hello', b'aello'),  (b'hello', b'yello'), 
        (b'hello' * 1000, b'hello' * 1000), (b'hello', b'hello'*100), (b'hello', b'jello'*10), (b'hello', b'bello'*10),
        (np.array(b'hello', dtype='S200'), np.array(b'elo', dtype='S60'))
    ]
    string_arguments = [
        (S, S), (S, T), (S, U), (V, W), (X, Y),
        (S, 'hello'), (U, 'hello'), (U, 'hello'*100), (U, 'hello'*10_000),
        ('hello', 'hella'), ('hello', 'aello'),  ('hello', 'yello'),
        ('hello' * 1000, 'hello' * 1000), ('hello', 'hello'*100), ('hello', 'jello'*10), ('hello', 'bello'*10),
        (np.array('hello', dtype='U200'), np.array('elo', dtype='U60'))
    ]
    test_char_function(numba_char_equal, np.char.equal, byte_arguments, string_arguments)
    test_char_function(numba_char_not_equal, np.char.not_equal, byte_arguments, string_arguments)
    test_char_function(numba_char_greater_equal, np.char.greater_equal, byte_arguments, string_arguments)
    test_char_function(numba_char_greater, np.char.greater, byte_arguments, string_arguments)
    test_char_function(numba_char_less, np.char.less, byte_arguments, string_arguments)
    test_char_function(numba_char_less_equal, np.char.less_equal, byte_arguments, string_arguments)


if __name__ == '__main__':
    main()
