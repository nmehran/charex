"""Test numpy character comparison operators"""

from charex.tests.support import CharacterFunctionTest
from numba import njit
import numpy as np
import charex


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


def main(method='graph'):
    np.random.seed(1)

    length = 10_000

    b = np.random.choice([b'hello', b'all', b'worlds'], length)
    c = np.random.choice([b'hello', b'\tFrom', b'all', b'Around', b'\nthe' b' World'], length)
    d = np.random.choice([b'hello', b'all', b'worlds'], length).astype('S200')
    e = np.random.choice([chr(__i) for __i in range(1, 128)], length)
    f = np.random.choice(e, length)

    s = b.astype('U')
    t = c.astype('U')
    u = d.astype('U')
    v = e.astype('U')
    w = f.astype('U')

    # With an efficient implementation of whitespace removal, trailing \t\n\r\f\v characters can be supported.
    x = np.random.choice([''.join([chr(np.random.randint(33, 127)) for _ in range(np.random.randint(1, 50))])
                          for _ in range(100)], length)
    y = np.random.choice(x, 10_000)
    g, h = x.astype('S'), y.astype('S')

    byte_arguments = [
        (b, b), (b, c), (b, d), (c, d), (e, f), (g, h),
        (b, b'hello'), (d, b'hello'), (d, b'hello'*100), (d, b'hello'*10_000),
        (np.array(b'hello', dtype='S200'), np.array(b'ello', dtype='S60')),
        (b'hello', b'hella'), (b'hello', b'aello'),  (b'hello', b'yello'),
        (b'hello' * 1000, b'hello' * 1000), (b'hello', b'hello'*100), (b'hello', b'jello'*10), (b'hello', b'bello'*10),
    ]
    string_arguments = [
        (s, s), (s, t), (s, u), (t, u), (v, w), (x, y),
        (s, 'hello'), (u, 'hello'), (u, 'hello' * 100), (u, 'hello' * 10_000),
        (np.array('hello', dtype='U200'), np.array('ello', dtype='U60')),
        ('hello', 'hella'), ('hello', 'aello'),  ('hello', 'yello'),
        ('hello' * 1000, 'hello' * 1000), ('hello', 'hello'*100), ('hello', 'jello'*10), ('hello', 'bello'*10),
    ]

    CharacterFunctionTest(numba_char_equal, np.char.equal, byte_arguments, string_arguments).run(method)
    CharacterFunctionTest(numba_char_not_equal, np.char.not_equal, byte_arguments, string_arguments).run(method)
    CharacterFunctionTest(numba_char_greater_equal, np.char.greater_equal, byte_arguments, string_arguments).run(method)
    CharacterFunctionTest(numba_char_greater, np.char.greater, byte_arguments, string_arguments).run(method)
    CharacterFunctionTest(numba_char_less, np.char.less, byte_arguments, string_arguments).run(method)
    CharacterFunctionTest(numba_char_less_equal, np.char.less_equal, byte_arguments, string_arguments).run(method)


if __name__ == '__main__':
    main(method='graph')

    # np.recarray(measurements, [('label', 'U50'), ('byte_timings', 'object'), ('string_timings', 'object')])
