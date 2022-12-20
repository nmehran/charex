from charex.core import JIT_OPTIONS
from numba.extending import register_jitable
from numpy import dtype, empty, frombuffer, ravel


# -----------------------------------------------------------------------------
# Support Functions


@register_jitable(**JIT_OPTIONS)
def register_array_bytes(b, rstrip=True):
    """Expose the ordinal representation of ASCII array bytes."""
    len_chr = b.size
    size_chr = b.itemsize
    if rstrip and size_chr > 1:
        return (
            _rstrip_inner(frombuffer(b, 'uint8').copy(), size_chr),
            len_chr,
            size_chr
        )
    return frombuffer(b, 'uint8'), len_chr, size_chr


@register_jitable(**JIT_OPTIONS)
def register_scalar_bytes(b, rstrip=True):
    """Expose the ordinal representation of scalar ASCII bytes."""
    len_chr = 1
    size_chr = len(b)
    if rstrip and size_chr > 1:
        return (
            _rstrip_inner(frombuffer(b, 'uint8').copy(), size_chr, True),
            len_chr,
            size_chr
        )
    return frombuffer(b, 'uint8'), len_chr, size_chr


@register_jitable(**JIT_OPTIONS)
def register_array_strings(s, rstrip=True):
    """Expose the ordinal representation of UTF-32 array strings."""
    len_chr = s.size
    size_chr = s.itemsize // 4
    chr_array = ravel(s).view(dtype('int32'))
    if rstrip and size_chr > 1:
        return _rstrip_inner(chr_array, size_chr), len_chr, size_chr
    return chr_array, len_chr, size_chr


@register_jitable(**JIT_OPTIONS)
def register_scalar_strings(s, rstrip=True):
    """Expose the ordinal representation of scalar UTF-32 strings."""
    len_chr = 1
    size_chr = len(s)
    chr_array = empty(size_chr, 'int32')
    for i in range(size_chr):
        chr_array[i] = ord(s[i])
    if rstrip and size_chr > 1:
        return _rstrip_inner(chr_array, size_chr, True), len_chr, size_chr
    return chr_array, len_chr, size_chr


@register_jitable(**JIT_OPTIONS)
def bisect_null(a, j, k):
    """Bisect null right-padded strings with the form '\x00'."""
    while j < k:
        m = (k + j) // 2
        if a[m]:
            j = m + 1
        else:
            k = m
    return j


@register_jitable(**JIT_OPTIONS)
def _rstrip_inner(chr_array, size_chr, is_scalar=False):
    r"""
    Removes trailing \t\n\r\f\v\s characters.
    As is the case when used on character comparison operators, this variation
    ignores the first character.
    """
    if size_chr == 1:
        return chr_array

    whitespace = {0, 9, 10, 11, 12, 13, 32}
    size_stride = size_chr - 1

    if is_scalar or size_chr < 9:
        for i in range(size_stride, chr_array.size, size_chr):
            for p in range(i, i - size_stride, -1):
                if chr_array[p] not in whitespace:
                    break
                chr_array[p] = 0
    else:
        for i in range(size_stride, chr_array.size, size_chr):
            if chr_array[i] in whitespace:
                o = i - size_stride
                p = bisect_null(chr_array, o, i - 1)
                while p > o and chr_array[p] in whitespace:
                    p -= 1
                chr_array[p + 1: i + 1] = 0
    return chr_array
