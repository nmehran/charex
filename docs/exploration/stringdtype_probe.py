"""Probe NumPy StringDType support points for charex.

Run with a NumPy 2.x / Numba environment, for example:

    /tmp/charex-numba-065-audit/bin/python docs/exploration/stringdtype_probe.py

This script is intentionally isolated from charex imports. It monkey-patches
Numba's ndarray typeof hook in-process to test feasibility.
"""

import ctypes

import numpy as np
from llvmlite import ir
from numba import njit, typeof
from numba.core import types
from numba.core.datamodel import models, register_default
from numba.core.typing.typeof import typeof_impl
from numba.np import numpy_support


def print_dtype_facts():
    dtype = np.dtypes.StringDType()
    arr = np.array(['a', 'é', '🙂', '', 'a\x00b'], dtype=dtype)

    print('dtype:', repr(dtype))
    print('kind/char/num:', dtype.kind, dtype.char, dtype.num)
    print('itemsize/hasobject:', dtype.itemsize, dtype.hasobject)
    print('shape/strides/nbytes:', arr.shape, arr.strides, arr.nbytes)
    print('np.strings.str_len:', np.strings.str_len(arr).tolist())

    try:
        print('numba typeof:', typeof(arr))
    except Exception as exc:
        print('numba typeof error:', type(exc).__name__, exc)


def numpy_c_api_slots():
    capsule = np._core.multiarray._ARRAY_API
    getptr = ctypes.pythonapi.PyCapsule_GetPointer
    getptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
    getptr.restype = ctypes.c_void_p
    api = getptr(capsule, None)
    return (ctypes.c_void_p * 400).from_address(api)


def print_c_api_unpacking():
    slots = numpy_c_api_slots()

    class StaticString(ctypes.Structure):
        _fields_ = [('size', ctypes.c_size_t), ('buf', ctypes.c_void_p)]

    acquire = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(slots[316])
    release = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(slots[318])
    load = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(StaticString),
    )(slots[313])

    for dtype in [
        np.dtypes.StringDType(),
        np.dtypes.StringDType(na_object=None),
        np.dtypes.StringDType(na_object=np.nan),
    ]:
        values = ['a', '', 'b'] if not hasattr(dtype, 'na_object') else [
            'a', dtype.na_object, 'b',
        ]
        arr = np.array(values, dtype=dtype)
        allocator = acquire(id(arr.dtype))
        base = arr.__array_interface__['data'][0]
        unpacked = []

        for i in range(arr.size):
            out = StaticString()
            rc = load(allocator, base + i * arr.itemsize, ctypes.byref(out))
            data = ctypes.string_at(out.buf, out.size) \
                if out.buf and out.size else b''
            unpacked.append((rc, out.size, data))

        release(allocator)
        print('unpack:', repr(dtype), unpacked)


class StringDTypePacket(types.Type):
    """Prototype Numba dtype for StringDType's 16-byte packed record."""

    def __init__(self):
        super().__init__('StringDTypePacket')


stringdtype_packet = StringDTypePacket()


@register_default(StringDTypePacket)
class StringDTypePacketModel(models.DataModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type)
        self._be_type = ir.ArrayType(ir.IntType(64), 2)

    def get_value_type(self):
        return self._be_type

    def get_data_type(self):
        return self._be_type

    def as_data(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value


def install_typeof_probe():
    old_ndarray_typeof = typeof_impl.registry[np.ndarray]

    @typeof_impl.register(np.ndarray)
    def _stringdtype_ndarray_typeof(val, context):
        if getattr(val.dtype, 'char', None) == 'T':
            layout = numpy_support.map_layout(val)
            readonly = not val.flags.writeable
            return types.Array(
                stringdtype_packet,
                val.ndim,
                layout,
                readonly=readonly,
            )
        return old_ndarray_typeof(val, context)

    typeof_impl._clear_cache()


def print_typeof_probe():
    install_typeof_probe()
    arr = np.array(['a', 'bb'], dtype=np.dtypes.StringDType())

    @njit
    def shape_info(x):
        return x.shape[0], x.itemsize, x.strides[0]

    @njit
    def item_load_compiles(x):
        _ = x[0]
        return 1

    print('patched typeof:', typeof(arr))
    print('shape_info:', shape_info(arr))
    print('item_load_compiles:', item_load_compiles(arr))


def main():
    print_dtype_facts()
    print_c_api_unpacking()
    print_typeof_probe()


if __name__ == '__main__':
    main()
