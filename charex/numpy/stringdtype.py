"""Experimental NumPy StringDType typing support."""

import ctypes

from llvmlite import ir
from numba.core import types
from numba.core.datamodel import models, register_default
from numba.core.typing import signature
from numba.core.typing.typeof import typeof_impl
from numba.extending import intrinsic
from numba.np import numpy_support
import numpy as np


_STRING_DTYPE = getattr(getattr(np, 'dtypes', None), 'StringDType', None)


class StringDTypePacket(types.Type):
    """Numba type for StringDType's 16-byte packed element record."""

    def __init__(self):
        super().__init__('StringDTypePacket')


stringdtype_packet = StringDTypePacket()


def _numpy_api_slots():
    capsule = np._core.multiarray._ARRAY_API
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    api = get_pointer(capsule, None)
    return (ctypes.c_void_p * 400).from_address(api)


class _StaticString(ctypes.Structure):
    _fields_ = [('size', ctypes.c_size_t), ('buf', ctypes.c_void_p)]


if _STRING_DTYPE is not None:
    _API_SLOTS = _numpy_api_slots()
    _NpyString_load = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_StaticString),
    )(_API_SLOTS[313])
    _NpyString_acquire_allocator = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p,
    )(_API_SLOTS[316])
    _NpyString_release_allocator = ctypes.CFUNCTYPE(
        None, ctypes.c_void_p,
    )(_API_SLOTS[318])
else:
    _NpyString_load = None
    _NpyString_acquire_allocator = None
    _NpyString_release_allocator = None


def _count_utf8_codepoints(data):
    count = 0
    for byte in data:
        count += (byte & 0xc0) != 0x80
    return count


_LEN_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t,
)


@_LEN_CALLBACK_TYPE
def _codepoint_len_callback(parent_addr, packed_addr):
    parent = ctypes.cast(parent_addr, ctypes.py_object).value
    allocator = _NpyString_acquire_allocator(id(parent.dtype))
    try:
        out = _StaticString()
        status = _NpyString_load(allocator, packed_addr, ctypes.byref(out))
        if status != 0:
            return -1
        data = ctypes.string_at(out.buf, out.size) \
            if out.buf and out.size else b''
        return _count_utf8_codepoints(data)
    finally:
        _NpyString_release_allocator(allocator)


_CODEPOINT_LEN_ADDR = ctypes.cast(
    _codepoint_len_callback, ctypes.c_void_p,
).value


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


def is_stringdtype(dtype):
    return _STRING_DTYPE is not None and getattr(dtype, 'char', None) == 'T'


def is_stringdtype_array_type(value):
    return isinstance(value, types.Array) \
        and isinstance(value.dtype, StringDTypePacket)


@intrinsic
def stringdtype_codepoint_len(typingctx, array, index):
    if not is_stringdtype_array_type(array) \
            or not isinstance(index, types.Integer):
        return None

    sig = signature(types.intp, array, types.intp)

    def codegen(context, builder, signature, args):
        array_type = signature.args[0]
        array_value, index_value = args
        array_struct = context.make_array(array_type)(
            context, builder, array_value,
        )

        intp = context.get_value_type(types.intp)
        uintp = context.get_value_type(types.uintp)
        byte_ptr = ir.IntType(8).as_pointer()
        data = builder.bitcast(array_struct.data, byte_ptr)
        stride = builder.extract_value(array_struct.strides, 0)
        offset = builder.mul(index_value, stride)
        packed = builder.gep(data, [offset])

        callback_type = ir.FunctionType(intp, [uintp, uintp])
        callback_addr = context.get_constant(
            types.uintp, _CODEPOINT_LEN_ADDR,
        )
        callback = builder.inttoptr(
            callback_addr, callback_type.as_pointer(),
        )
        parent = builder.ptrtoint(array_struct.parent, uintp)
        packed = builder.ptrtoint(packed, uintp)
        return builder.call(callback, [parent, packed])

    return sig, codegen


def _install_typeof():
    if _STRING_DTYPE is None:
        return
    if getattr(typeof_impl, '_charex_stringdtype_installed', False):
        return

    old_ndarray_typeof = typeof_impl.registry[np.ndarray]
    typeof_impl._charex_stringdtype_old_ndarray_typeof = old_ndarray_typeof

    @typeof_impl.register(np.ndarray)
    def _stringdtype_ndarray_typeof(value, context):
        if is_stringdtype(value.dtype):
            layout = numpy_support.map_layout(value)
            readonly = not value.flags.writeable
            return types.Array(
                stringdtype_packet,
                value.ndim,
                layout,
                readonly=readonly,
            )
        return old_ndarray_typeof(value, context)

    typeof_impl._charex_stringdtype_installed = True
    typeof_impl._clear_cache()


_install_typeof()
