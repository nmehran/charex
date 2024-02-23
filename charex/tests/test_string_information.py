"""Test numpy string information operations."""

from charex.tests.definitions import StringInformation
from charex.tests.support import CharacterTest, arguments_as_bytes, pack_arguments
from sys import maxunicode
import numpy as np

TEST_BYTES = True
TEST_STRINGS = True


def test(method='test'):
    ch = StringInformation()

    def test_string_occurrence(byte_args_, string_args_, method_=method):
        m = 'measure' if method_ == 'graph' else method_

        ba = list(pack_arguments(byte_args_, [(None, 1, 2, -500, 500), (0, -1, -2, -500, None)]))
        sa = list(pack_arguments(string_args_, [(None, 1, 2, -500, 500), (0, -1, -2, -500, None)]))

        test_occurrence = CharacterTest(byte_args=ba, string_args=sa)

        test_occurrence.run(m, ch.char_count, np.char.count)
        test_occurrence.run(m, ch.char_find, np.char.find)
        test_occurrence.run(m, ch.char_rfind, np.char.rfind)
        test_occurrence.run(m, ch.char_startswith, np.char.startswith)
        test_occurrence.run(m, ch.char_endswith, np.char.endswith)

        if method_ == 'graph':
            test_occurrence.graph(columns=1)

        return test_occurrence

    def test_string_properties(byte_args_, string_args_, method_=method):
        m = 'measure' if method_ == 'graph' else method_

        test_properties = CharacterTest(byte_args=byte_args_, string_args=string_args_)

        test_properties.run(m, ch.char_str_len, np.char.str_len)
        test_properties.run(m, ch.char_isalpha, np.char.isalpha)
        test_properties.run(m, ch.char_isalnum, np.char.isalnum)
        test_properties.run(m, ch.char_isdigit, np.char.isdigit)
        test_properties.run(m, ch.char_islower, np.char.islower)
        test_properties.run(m, ch.char_isspace, np.char.isspace)
        test_properties.run(m, ch.char_istitle, np.char.istitle)
        test_properties.run(m, ch.char_isupper, np.char.isupper)

        if method_ == 'graph':
            test_properties.graph(columns=3)

        # Methods which do not support bytes
        test_properties = CharacterTest(byte_args=[], string_args=string_args_)
        test_properties.run(m, ch.char_isnumeric, np.char.isnumeric)
        test_properties.run(m, ch.char_isdecimal, np.char.isdecimal)

        if method_ == 'graph':
            test_properties.graph(columns=3)

        return test_properties

    length = 10000
    np.random.seed(1)

    # Whitespace
    w = [0, 9, 10, 11, 12, 13, 28, 29, 30, 31, 32]

    # ASCII word pairs
    a = np.array(['aAaAaA', '  aA  ', 'abBABba', 'AbbAbbbbAbb', ' aA Aa aa AA A1 1A 2a 33 Aa-aa '])
    p = np.array([chr(np.random.choice(w)).join([''.join([chr(np.random.randint(48, 127))
                                                          for _ in range(3)]) for _ in range(2)])
                  for _ in range(length)])

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
    # Single ASCII characters
    c = np.random.choice([chr(i) for i in range(128)], length)

    generics = [
        (c, np.random.choice(c, c.size)),
        (s, np.random.choice(s, s.size)),
    ]

    # Scalar Comparisons
    scalars = [
        (a, 'aA'), (a, 'Abb'),
        ('abc' * 2, 'abc'), ('abc', ''),
    ]

    # Character buffers of different length
    buffers = [
        (a.astype('U35'), np.array('A', 'U40')),
        (a, np.array('', 'U10'))
    ]

    # UTF-32
    utf32 = [
        (u, np.random.choice(u)),
        (u, np.random.choice(u, u.size)),
    ]

    # Test Occurrence Methods
    byte_args, string_args = [], []
    if TEST_STRINGS:
        string_args = generics + scalars + buffers + utf32
    if TEST_BYTES:
        byte_args = list(arguments_as_bytes(generics + scalars + buffers))

    test_string_occurrence(byte_args_=[a[::-1] for a in byte_args],
                           string_args_=[a[::-1] for a in string_args],
                           method_='test')
    test_string_occurrence(byte_args, string_args, method)

    # Test Property Methods
    byte_args, string_args = [], []
    if TEST_STRINGS:
        string_args = [(z,) for z in [a, c, p, s]]
    if TEST_BYTES:
        byte_args = list(arguments_as_bytes([(z,) for z in [a, c, p, s]]))

    test_string_properties(byte_args, string_args, method)


if __name__ == '__main__':
    test(method='graph')
