from charex.core._string_intrinsics import register_bytes, register_strings
from numba.extending import overload, register_jitable
from numba.core import types
import numpy as np

OPTIONS = dict(
    jit_options=dict(
        nogil=True
    ),
    prefer_literal=True,
    strict=True
)


@overload(np.char.equal, **OPTIONS)
def ov_nb_char_equal(x1, x2):
    """Native Implementation of np.character.equal"""

    accepted_types = (types.Array, types.Bytes, types.UnicodeType)
    if not isinstance(x1, accepted_types) or not isinstance(x2, accepted_types):
        raise TypeError('comparison of non-string arrays')

    if isinstance(x1, types.UnicodeType) and isinstance(x2, types.UnicodeType):
        def impl(x1, x2):
            return np.array(len(x1) == len(x2) and x1 == x2, dtype='bool')
        return impl

    @register_jitable
    def character_equal(chr_array, cmp_array, len_chr, len_cmp, size_chr, size_cmp):
        equal_to = np.zeros(len_chr, dtype='bool')
        ix = 0
        if len_cmp == 1:
            if size_chr < size_cmp:
                return equal_to
            elif size_chr > size_cmp:
                for i in range(len_chr):
                    equal_to[i] = chr_array[ix + size_cmp] == 0 and (chr_array[ix:ix + size_cmp] - cmp_array).sum() == 0
                    ix += size_chr
            else:
                for i in range(len_chr):
                    equal_to[i] = (chr_array[ix:ix + size_chr] - cmp_array).sum() == 0
                    ix += size_chr
        elif len_chr == len_cmp:
            iy = 0
            if size_chr < size_cmp:
                for i in range(len_chr):
                    equal_to[i] = cmp_array[iy + size_chr] == 0 and (
                            chr_array[ix:ix + size_chr] - cmp_array[iy:iy + size_chr]).sum() == 0
                    ix += size_chr
                    iy += size_cmp
            elif size_chr > size_cmp:
                for i in range(len_chr):
                    equal_to[i] = chr_array[ix + size_cmp] == 0 and (
                            chr_array[ix:ix + size_cmp] - cmp_array[iy:iy + size_cmp]).sum() == 0
                    ix += size_chr
                    iy += size_cmp
            else:
                for i in range(len_chr):
                    equal_to[i] = (chr_array[ix:ix + size_chr] - cmp_array[ix:ix + size_chr]).sum() == 0
                    ix += size_chr
        else:
            msg = 'shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 and arg 1.'
            raise ValueError(msg)
        return equal_to

    byte_types = (types.Bytes, types.CharSeq)
    str_types = (types.UnicodeType, types.UnicodeCharSeq)

    x1_type = x1.dtype if isinstance(x1, types.Array) else x1
    x2_type = x2.dtype if isinstance(x2, types.Array) else x2

    if isinstance(x1_type, byte_types) and isinstance(x2_type, byte_types):
        register_type = register_bytes
        cmp_type = bytes
    elif isinstance(x1_type, str_types) and isinstance(x2_type, str_types):
        register_type = register_strings
        cmp_type = str
    else:
        def impl(x1, x2):
            raise NotImplementedError('NotImplemented')
        return impl

    def impl(x1, x2):
        if isinstance(x1, cmp_type) and not isinstance(x2, cmp_type):
            return character_equal(*register_type(x2, x1))
        return character_equal(*register_type(x1, x2))

    return impl