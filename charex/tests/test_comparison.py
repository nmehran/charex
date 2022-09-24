"""Test numpy character comparison operators."""

from charex.tests.definitions import ComparisonOperators
from charex.tests.support import CharacterTest, arguments_as_bytes, pack_arguments
from sys import maxunicode
import numpy as np

TEST_BYTES = True
TEST_STRINGS = True


def test(method='test'):
    ch = ComparisonOperators()

    def test_comparison_operators(byte_args_, string_args_, method_=method):
        character_test = CharacterTest(byte_args=byte_args_, string_args=string_args_)
        m = 'measure' if method_ == 'graph' else method_

        character_test.run(m, ch.char_equal, np.char.equal)
        character_test.run(m, ch.char_not_equal, np.char.not_equal)
        character_test.run(m, ch.char_greater_equal, np.char.greater_equal)
        character_test.run(m, ch.char_greater, np.char.greater)
        character_test.run(m, ch.char_less, np.char.less)
        character_test.run(m, ch.char_less_equal, np.char.less_equal)

        if method_ == 'graph':
            character_test.graph()

        byte_args_ = list(pack_arguments(byte_args_, [('==', '!=', '>=', '>', '<', '<='), (True, False)]))
        string_args_ = list(pack_arguments(string_args_, [('==', '!=', '>=', '>', '<', '<='), (True, False)]))
        CharacterTest(byte_args=byte_args_, string_args=string_args_)\
            .test(ch.compare_chararrays, np.compare_chararrays)

        return character_test

    length = 10001
    np.random.seed(1)
    # ASCII strings of length 0 to 50
    s = np.array([''.join([chr(np.random.randint(1, 127))
                           for _ in range(np.random.randint(0, 50))])
                  for _ in range(length)])
    # UTF-32 strings of length 1 to 200 in range(1, sys.maxunicode)
    # Python 3.7 can not decode unicode in range(55296, 57344)
    u = np.array([''.join([chr(np.random.randint(1, 55295)) if i % 2
                           else chr(np.random.randint(57345, maxunicode))
                           for i in range(np.random.randint(1, 200))])
                  for _ in range(length)])
    # Whitespace to end of strings & single ASCII characters in range(0, 33)
    w = [chr(i) for i in range(33)]
    x = np.concatenate([w, np.char.add(s, np.random.choice(w, length))])

    # Single ASCII characters
    c = np.random.choice([chr(i) for i in range(128)], length)

    generics = [
        (c, np.random.choice(c, c.size)),
        (x, np.random.choice(x, x.size)),
        (x[:2], x[:2]),
    ]

    # Scalar Comparisons
    scalars = [
        (x, 'abcd ' * 20),
        ('abc', 'abc '), ('abc', 'abc' * 2),
        ('abc', 'abd'), ('abc', 'abb'), ('ab', 'ba'),
    ]

    # Character buffers of different length
    buffers = [
        (s[:1].astype('U20'), s[:1].astype('U40')),
        (x[:5].astype('U60'), x[:5].astype('U61')),
        (x[:5], x[:5].astype('U100')),
        (np.array('hello ' * 5, dtype='U30'),
         np.array('hello ' * 10, dtype='U60')),
    ]

    # UTF-32
    utf32 = [
        (u, np.random.choice(u)),
        (u, np.random.choice(u, len(u))),
        (u, np.char.add(u, np.random.choice(w, len(u))))
    ]

    byte_args, string_args = [], []
    if TEST_STRINGS:
        string_args = generics + scalars + buffers + utf32
    if TEST_BYTES:
        byte_args = list(arguments_as_bytes(generics + scalars + buffers))

    test_comparison_operators(byte_args_=[a[::-1] for a in byte_args],
                              string_args_=[a[::-1] for a in string_args],
                              method_='test')
    test_comparison_operators(byte_args, string_args, method)


if __name__ == '__main__':
    test(method='graph')
