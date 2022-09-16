from charex.core import JIT_OPTIONS
from numba.extending import register_jitable
from numpy import dtype, empty, frombuffer, ravel


@register_jitable(**JIT_OPTIONS)
def register_bytes(b, rstrip=True):
    """Expose bytes to their numerical representation."""
    if isinstance(b, bytes):
        len_chr = 1
        size_chr = len(b)
    else:
        len_chr = b.size
        size_chr = b.itemsize
    if rstrip:
        return rstrip_inner(frombuffer(b, 'int8').copy(), size_chr), len_chr, size_chr
    return frombuffer(b, 'int8'), len_chr, size_chr


@register_jitable(**JIT_OPTIONS)
def register_strings(s, rstrip=True):
    """Expose UTF-32 strings to their numerical representation."""
    if isinstance(s, str):
        len_chr = 1
        size_chr = len(s)
        chr_array = empty(size_chr, 'int32')
        for i in range(size_chr):
            chr_array[i] = ord(s[i])
    else:
        len_chr = s.size
        size_chr = s.itemsize // 4
        chr_array = ravel(s).view(dtype('int32'))
    if rstrip:
        return rstrip_inner(chr_array, size_chr), len_chr, size_chr
    return chr_array, len_chr, size_chr


@register_jitable(**JIT_OPTIONS)
def rstrip_inner(chr_array, size_chr):
    r"""
    Removes trailing \t\n\r\f\v\s characters.
    As is the case when used on character comparison operators, this variation ignores the first character.
    """
    if size_chr == 1:
        return chr_array

    def bisect_null(a, j, k):
        """Bisect null right-padded strings with the form '\x00'."""
        while j < k:
            m = (k + j) // 2
            c = a[m]
            if c != 0:
                j = m + 1
            elif c == 0:
                k = m
            else:
                return m
        return j

    whitespace = {0, 9, 10, 11, 12, 13, 32}
    size_stride = size_chr - 1

    for i in range(size_stride, chr_array.size, size_chr):
        if chr_array[i] not in whitespace:
            continue

        o = i - size_stride
        p = bisect_null(chr_array, o, i - 1)
        while p > o and chr_array[p] in whitespace:
            p -= 1

        chr_array[p + 1: i + 1] = 0

    return chr_array
