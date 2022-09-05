from numba.extending import register_jitable
from numpy import array, frombuffer


@register_jitable
def register_bytes(x1, x2):
    if isinstance(x1, bytes):
        len_chr = 1
        size_chr = len(x1)
    else:
        len_chr = x1.size
        size_chr = x1.itemsize

    if isinstance(x2, bytes):
        len_cmp = 1
        size_cmp = len(x2)
    else:
        len_cmp = x2.size
        size_cmp = x2.itemsize

    chr_array = frombuffer(x1, dtype='uint8')
    cmp_array = frombuffer(x2, dtype='uint8')
    return chr_array, cmp_array, len_chr, len_cmp, size_chr, size_cmp


@register_jitable
def register_strings(x1, x2):
    if isinstance(x1, str):
        len_chr = 1
        size_chr = len(x1)
        chr_array = array(list(map(ord, x1)), dtype='uint8')
    else:
        len_chr = x1.size
        size_chr = x1.itemsize // 4
        chr_array = frombuffer(x1, dtype='uint8')[::4].ravel()

    if isinstance(x2, str):
        len_cmp = 1
        size_cmp = len(x2)
        cmp_array = array(list(map(ord, x2)), dtype='uint8')
    else:
        len_cmp = x2.size
        size_cmp = x2.itemsize // 4
        cmp_array = frombuffer(x2, dtype='uint8')[::4].ravel()

    return chr_array, cmp_array, len_chr, len_cmp, size_chr, size_cmp