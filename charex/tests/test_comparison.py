from charex.tests.support import measure_performance
from numba import njit
import numpy as np
import charex


def test_equal():
    """Test numpy.char.equal"""

    @njit(nogil=True, cache=True)
    def numba_char_equal(x1, x2):
        return np.char.equal(x1, x2)

    print('\ntest_equal::Byte Tests:')

    comparison = numba_char_equal(B, B) == np.char.equal(B, B)
    assert np.all(comparison)

    comparison = numba_char_equal(B, C) == np.char.equal(B, C)
    assert np.all(comparison)

    comparison = numba_char_equal(C, B) == np.char.equal(C, B)
    assert np.all(comparison)

    comparison = numba_char_equal(B, b'hello') == np.char.equal(B, b'hello')
    assert np.all(comparison)

    comparison = numba_char_equal(b'hello', B) == np.char.equal(b'hello', B)
    assert np.all(comparison)

    comparison = numba_char_equal(b'hello', b'hella') == np.char.equal(b'hello', b'hella')
    assert np.all(comparison)

    comparison = numba_char_equal(b'hello'*100, b'hello'*100) == np.char.equal(b'hello'*100, b'hello'*100)
    assert np.all(comparison)

    comparison = numba_char_equal(D, E) == np.char.equal(D, E)
    assert np.all(comparison)

    measure_performance(numba_char_equal, 10, B, B)
    measure_performance(np.char.equal, 10, B, B)

    measure_performance(numba_char_equal, 10, B, C)
    measure_performance(np.char.equal, 10, B, C)

    measure_performance(numba_char_equal, 10, C, B)
    measure_performance(np.char.equal, 10, C, B)

    measure_performance(numba_char_equal, 10, B, b'hello')
    measure_performance(np.char.equal, 10, B, b'hello')

    measure_performance(numba_char_equal, 10, b'hello', B)
    measure_performance(np.char.equal, 10, b'hello', B)

    measure_performance(numba_char_equal, 10, b'hello', b'hella')
    measure_performance(np.char.equal, 10, b'hello', b'hella')

    measure_performance(numba_char_equal, 10, b'hello'*1000, b'hello'*1000)
    measure_performance(np.char.equal, 10, b'hello'*1000, b'hello'*1000)

    measure_performance(numba_char_equal, 10, D, E)
    measure_performance(np.char.equal, 10, D, E)

    print('\ntest_equal::String Tests:')

    comparison = numba_char_equal(S, S) == np.char.equal(S, S)
    assert np.all(comparison)

    comparison = numba_char_equal(S, T) == np.char.equal(S, T)
    assert np.all(comparison)

    comparison = numba_char_equal(T, S) == np.char.equal(T, S)
    assert np.all(comparison)

    comparison = numba_char_equal(S, 'hello') == np.char.equal(S, 'hello')
    assert np.all(comparison)

    comparison = numba_char_equal('hello', S) == np.char.equal('hello', S)
    assert np.all(comparison)

    comparison = numba_char_equal('hello', 'hella') == np.char.equal('hello', 'hella')
    assert np.all(comparison)

    comparison = numba_char_equal('hello'*100, 'hello'*100) == np.char.equal('hello'*100, 'hello'*100)
    assert np.all(comparison)

    comparison = numba_char_equal(U, V) == np.char.equal(U, V)
    assert np.all(comparison)

    measure_performance(numba_char_equal, 10, S, S)
    measure_performance(np.char.equal, 10, S, S)

    measure_performance(numba_char_equal, 10, S, T)
    measure_performance(np.char.equal, 10, S, T)

    measure_performance(numba_char_equal, 10, T, S)
    measure_performance(np.char.equal, 10, T, S)

    measure_performance(numba_char_equal, 10, S, 'hello')
    measure_performance(np.char.equal, 10, S, 'hello')

    measure_performance(numba_char_equal, 10, 'hello', S)
    measure_performance(np.char.equal, 10, 'hello', S)

    measure_performance(numba_char_equal, 10, 'hello', 'hella')
    measure_performance(np.char.equal, 10, 'hello', 'hella')

    measure_performance(numba_char_equal, 10, 'hello'*1000, 'hello'*1000)
    measure_performance(np.char.equal, 10, 'hello'*1000, 'hello'*1000)

    measure_performance(numba_char_equal, 10, U, V)
    measure_performance(np.char.equal, 10, U, V)


def main():
    test_equal()


if __name__ == '__main__':
    np.random.seed(1)

    B = np.random.choice([b'hello', b'all', b'worlds'], 10_000)
    C = np.random.choice([b'hello', b'all', b'worlds'], 10_000).astype('S200')
    D = np.random.choice([chr(__i) for __i in range(1, 255)], 10_000)
    E = np.random.choice([chr(__i) for __i in range(1, 255)], 10_000)

    S = B.astype('U')
    T = C.astype('U')
    U = D.astype('U')
    V = E.astype('U')

    main()
