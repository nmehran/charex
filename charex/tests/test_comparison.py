"""Test numpy character comparison operators"""

from charex.tests.definitions import ComparisonOperators
from charex.tests.support import arguments_as_bytes, pack_arguments, CharacterTest
from sys import maxunicode
import numpy as np

TEST_BYTES = True
TEST_STRINGS = True


def test(method='test'):
    ch = ComparisonOperators()

    def test_comparison_operators(byte_args_, string_args_, method_=method):
        character_test = CharacterTest(byte_args=byte_args_, string_args=string_args_)
        m = 'measure' if method_ == 'graph' else method_

        character_test.run(m, ch.numba_char_equal, np.char.equal)
        character_test.run(m, ch.numba_char_not_equal, np.char.not_equal)
        character_test.run(m, ch.numba_char_greater_equal, np.char.greater_equal)
        character_test.run(m, ch.numba_char_greater, np.char.greater)
        character_test.run(m, ch.numba_char_less, np.char.less)
        character_test.run(m, ch.numba_char_less_equal, np.char.less_equal)

        if method_ == 'graph':
            character_test.graph()

        byte_args_ = list(pack_arguments(byte_args_, [('==', '!=', '>=', '>', '<', '<='), (True, False)]))
        string_args_ = list(pack_arguments(string_args_, [('==', '!=', '>=', '>', '<', '<='), (True, False)]))
        CharacterTest(byte_args=byte_args_, string_args=string_args_).test(ch.numba_compare_chararrays,
                                                                           np.compare_chararrays)
        return character_test

    length = 10001
    # Generate 500 UTF-32 strings of length 1 to 200 in range(1, sys.maxunicode)
    np.random.seed(1)
    q = np.random.choice([''.join([chr(np.random.randint(1, maxunicode)) for _ in range(np.random.randint(1, 200))])
                          for _ in range(500)], length)
    # Generate 100 ASCII strings of length 1 to 50
    r = np.random.choice([''.join([chr(np.random.randint(1, 128)) for _ in range(np.random.randint(1, 50))])
                          for _ in range(100)], length)
    s: tuple = np.array(['abc', 'def'] * length), np.array(['cba', 'fed'] * length),
    t: tuple = np.array(['ab', 'bc'] * length), np.array(['bc', 'ab'] * length),
    u: tuple = np.array(['ba', 'cb'] * length), np.array(['cb', 'ba'] * length)
    v = np.random.choice(['abcd', 'abc', 'abcde'], length)

    # Add whitespace to end of strings, and single ASCII characters in range(0, 33)
    w = [chr(i) for i in range(33)]
    x = np.concatenate([w, np.char.add(r, np.random.choice(w, length))])

    # Generate single ASCII characters
    c = np.random.choice([chr(__i) for __i in range(128)], length)

    arrays = [
        s, t, u,
        (c, np.random.choice(c, c.size)),
        (v, np.random.choice(v, v.size)),
        (x, x), (x, np.random.choice(x, x.size)),
        (x, 'abcdefg'), (x, 'abcdefg'*50),
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

    # UTF-32
    arrays += [
        (q, np.random.choice(q)),
        (q, np.random.choice(q, q.size)),
        (q, np.char.add(q, np.random.choice(w, q.size)))
    ]

    byte_args, string_args = [], []
    if TEST_STRINGS:
        string_args = arrays
    if TEST_BYTES:
        byte_args = list(arguments_as_bytes(arrays[:-3]))

    test_comparison_operators(byte_args_=[a[::-1] for a in byte_args],
                              string_args_=[a[::-1] for a in string_args],
                              method_='test')
    test_comparison_operators(byte_args, string_args, method)


if __name__ == '__main__':
    test(method='graph')
