"""Test numpy character comparison operators"""

from charex.tests.definitions import ComparisonOperators
from charex.tests.support import arguments_as_bytes, CharacterTest
import numpy as np

TEST_BYTES = True
TEST_STRINGS = True


def test(method='test'):
    np.random.seed(1)
    ch = ComparisonOperators()
    character_test = CharacterTest()

    length = 10_001
    # Generate 100 ASCII strings of length 1 to 50
    r = np.random.choice([''.join([chr(np.random.randint(1, 127)) for _ in range(np.random.randint(1, 50))])
                          for _ in range(100)], length)
    s: tuple = np.array(['abc', 'def'] * length), np.array(['cba', 'fed'] * length),
    t: tuple = np.array(['ab', 'bc'] * length), np.array(['bc', 'ab'] * length),
    u: tuple = np.array(['ba', 'cb'] * length), np.array(['cb', 'ba'] * length)
    v = np.random.choice(['abcd', 'abc', 'abcde'], length)
    w = np.random.choice([chr(__i) for __i in range(1, 128)], length)

    # With an efficient implementation of whitespace removal, trailing \t\n\r\f\v characters can be supported
    x = np.char.add(r, chr(np.random.randint(33, 127)))

    arrays = [
        s, t, u,
        (v, np.random.choice(v, length)),
        (w, np.random.choice(w, length)),
        (w, w), (x, x),
        (x, np.random.choice(x, length)),
        (x, 'abcdefg'), (x, 'gfedcba'),
        ('abc', 'abc'), ('abc', 'abd'), ('abc', 'abb'),
        ('abc', 'abc'*100), ('ab', 'ba'),
    ]

    # Character buffers of different length
    arrays += [
        (x, x.astype('U200')),
        (s[0].astype('U20'), s[1].astype('U40')),
        (x.astype('U60'), x.astype('U61')),
        (np.array('hello'*100, dtype='U200'), np.array('hello'*100, dtype='U100'))
    ]

    byte_args, string_args = [], []
    if TEST_STRINGS:
        string_args = arrays
    if TEST_BYTES:
        byte_args = arguments_as_bytes(arrays)

    m = 'measure' if method == 'graph' else method

    character_test.run(m, ch.numba_char_equal, np.char.equal, byte_args, string_args)
    character_test.run(m, ch.numba_char_not_equal, np.char.not_equal)
    character_test.run(m, ch.numba_char_greater_equal, np.char.greater_equal)
    character_test.run(m, ch.numba_char_greater, np.char.greater)
    character_test.run(m, ch.numba_char_less, np.char.less)
    character_test.run(m, ch.numba_char_less_equal, np.char.less_equal)

    if method == 'graph':
        character_test.graph(main_title='Measured Test Performance (milliseconds)')


if __name__ == '__main__':
    test(method='graph')
