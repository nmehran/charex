"""Test numpy string information operations."""

from charex.tests.definitions import StringInformation
from charex.tests.support import CharacterTest, arguments_as_bytes, pack_arguments
from sys import maxunicode
import numpy as np

TEST_BYTES = True
TEST_STRINGS = True


def test(method='test'):
    ch = StringInformation()

    def test_string_information(byte_args_, string_args_, method_=method):

        ba = list(pack_arguments(byte_args_, [(0, 1, 2,  -500, 500, None), (-0, -1, -2,  -500, 500, None)]))
        sa = list(pack_arguments(string_args_, [(0, 1, 2,  -500, 500, None), (-0, -1, -2,  -500, 500, None)]))

        CharacterTest(byte_args=ba,
                      string_args=sa).run(method_, ch.char_count, np.char.count)

        CharacterTest(byte_args=ba,
                      string_args=sa).run(method_, ch.char_endswith, np.char.endswith)

        CharacterTest(byte_args=ba,
                      string_args=sa).run(method_, ch.char_find, np.char.find)

        CharacterTest(byte_args=ba,
                      string_args=sa).run(method_, ch.char_rfind, np.char.rfind)

        CharacterTest(byte_args=ba,
                      string_args=sa).run(method_, ch.char_startswith, np.char.startswith)

        CharacterTest(byte_args=[ba[:1] for ba in byte_args_],
                      string_args=[sa[:1] for sa in string_args_]).run(method_, ch.str_len, np.char.str_len)

    length = 10000
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
        (x, x),
    ]

    # Scalar Comparisons
    scalars = [
        (x, 'abcd ' * 20),
        ('abc', 'abc '), ('abc', 'abc' * 2),
        ('abc', 'abd'), ('abc', 'abb'), ('ab', 'ba'),
        ('abc', '')
    ]

    # Character buffers of different length
    buffers = [
        (s.astype('U20'), s.astype('U40')),
        (x.astype('U60'), x.astype('U61')),
        (x, x.astype('U100')),
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

    test_string_information(byte_args_=[a[::-1] for a in byte_args],
                            string_args_=[a[::-1] for a in string_args],
                            method_='test')
    test_string_information(byte_args, string_args, method)


if __name__ == '__main__':
    test(method='graph')
