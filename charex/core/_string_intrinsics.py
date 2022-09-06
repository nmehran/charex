from charex.core import JIT_OPTIONS
from numba.extending import register_jitable
from numpy import empty, frombuffer


@register_jitable(**JIT_OPTIONS)
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
    if isinstance(s, str):
        len_chr = 1
        size_chr = len(s)
        chr_array = empty(size_chr, dtype='int8')
        for i in range(size_chr):
            chr_array[i] = ord(s[i])
    else:
        len_chr = s.size
        size_chr = s.itemsize // 4
        chr_array = frombuffer(s, dtype='int8')[::4].ravel()
    return chr_array, len_chr, size_chr
