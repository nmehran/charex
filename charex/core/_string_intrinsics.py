from charex.core import JIT_OPTIONS
from numba.extending import register_jitable
from numba import objmode
from numpy import frombuffer


@register_jitable('int8[::1](int8[::1])', **JIT_OPTIONS)
def register_bytes(b):
    if isinstance(b, bytes):
        len_chr = 1
        size_chr = len(b)
    else:
        len_chr = b.size
        size_chr = b.itemsize
    return frombuffer(b, dtype='int8'), len_chr, size_chr


@register_jitable(**JIT_OPTIONS)
def register_strings(s):
    """Restricted to ASCII encoding (expandable with performance penalty)."""
    if isinstance(s, str):
        len_chr = 1
        size_chr = len(s)
        with objmode(chr_array='int8[::1]'):
            chr_array = frombuffer(bytes(s, 'ASCII'), dtype='int8')
    else:
        len_chr = s.size
        size_chr = s.itemsize // 4
        chr_array = frombuffer(s, dtype='int8')[::4].ravel()
    return chr_array, len_chr, size_chr
