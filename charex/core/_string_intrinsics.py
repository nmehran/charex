from charex.core import JIT_OPTIONS
from numba.extending import register_jitable
from numba import objmode
from numpy import frombuffer


@register_jitable(**JIT_OPTIONS)
def register_bytes(b, rstrip=True):
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
    """Restricted to ASCII encoding (can be expanded)."""
    if isinstance(s, str):
        len_chr = 1
        size_chr = len(s)
        with objmode(chr_array='int8[::1]'):
            # objmode in this instance is faster than ordinal mapping.
            chr_array = frombuffer(bytes(s, 'ASCII'), 'int8', size_chr)
    else:
        len_chr = s.size
        size_chr = s.itemsize // 4
        chr_array = frombuffer(s, 'int8')[::4].ravel()
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

    whitespace = {0, 9, 10, 11, 12, 13, 32}
    size_stride = size_chr - 1

    for i in range(size_stride, chr_array.size, size_chr):
        if chr_array[i] not in whitespace:
            continue
        j = i - 1
        while j > i - size_stride and chr_array[j] in whitespace:
            j -= 1
        chr_array[j + 1: i + 1] = 0
    return chr_array
