"""Compare StringDType descriptor/allocator access strategies.

This is exploratory code for Tranche 1. It intentionally tests approaches that
may be rejected from the final implementation.
"""

import ctypes
from pathlib import Path
import random
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time

from llvmlite import ir
from numba import njit
from numba.core import types
from numba.core.typing import signature
from numba.extending import intrinsic
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import charex  # noqa: F401
from charex.numpy.stringdtype import (
    is_stringdtype_array_type,
    stringdtype_acquire_allocator,
    stringdtype_codepoint_len,
    stringdtype_release_allocator,
)


_SLOTS = charex.numpy.stringdtype._API_SLOTS
_NpyString_acquire_allocator = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p,
)(_SLOTS[316])
_DEFAULT_DTYPE = np.dtype('T')
_DEFAULT_DTYPE_ADDR = id(_DEFAULT_DTYPE)
_C_HELPER_ACQUIRE_ADDR = 0
_C_HELPER_LIBRARY = None


_ACQUIRE_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)


@_ACQUIRE_CALLBACK
def _callback_acquire(parent_addr):
    parent = ctypes.cast(parent_addr, ctypes.py_object).value
    return _NpyString_acquire_allocator(id(parent.dtype))


_CALLBACK_ACQUIRE_ADDR = ctypes.cast(
    _callback_acquire, ctypes.c_void_p,
).value


@intrinsic
def callback_acquire_allocator(typingctx, array):
    if not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        array_struct = context.make_array(signature.args[0])(
            context, builder, args[0],
        )
        uintp = context.get_value_type(types.uintp)
        callback_type = ir.FunctionType(ir.IntType(8).as_pointer(), [uintp])
        callback_addr = context.get_constant(
            types.uintp, _CALLBACK_ACQUIRE_ADDR,
        )
        callback = builder.inttoptr(
            callback_addr, callback_type.as_pointer(),
        )
        parent_addr = builder.ptrtoint(array_struct.parent, uintp)
        return builder.call(callback, [parent_addr])

    return sig, codegen


@intrinsic
def default_descriptor_acquire_allocator(typingctx, array):
    if not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        byte_ptr = ir.IntType(8).as_pointer()
        acquire_type = ir.FunctionType(byte_ptr, [byte_ptr])
        acquire = builder.module.globals.get(
            'charex_NpyString_acquire_allocator',
        )
        if acquire is None:
            acquire = ir.Function(
                builder.module,
                acquire_type,
                'charex_NpyString_acquire_allocator',
            )
        descriptor_addr = context.get_constant(
            types.uintp, _DEFAULT_DTYPE_ADDR,
        )
        descriptor = builder.inttoptr(descriptor_addr, byte_ptr)
        return builder.call(acquire, [descriptor])

    return sig, codegen


def build_c_helper():
    source = r'''
#define PY_SSIZE_T_CLEAN
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

int charex_bench_import_array(void) {
    import_array1(-1);
    return 0;
}

void *charex_bench_acquire_allocator(PyObject *array) {
    PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *)array);
    return NpyString_acquire_allocator((PyArray_StringDTypeObject *)descr);
}
'''
    build_dir = Path(tempfile.mkdtemp(prefix='charex-stringdtype-c-'))
    source_path = build_dir / 'helper.c'
    output_path = build_dir / 'helper.so'
    source_path.write_text(source)
    includes = [
        sysconfig.get_path('include'),
        sysconfig.get_path('platinclude'),
        np.get_include(),
    ]
    command = [
        'gcc', '-shared', '-fPIC', '-O3',
        *(f'-I{path}' for path in includes if path),
        str(source_path), '-o', str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    library = ctypes.PyDLL(str(output_path))
    library.charex_bench_import_array.argtypes = []
    library.charex_bench_import_array.restype = ctypes.c_int
    if library.charex_bench_import_array() != 0:
        raise RuntimeError('failed to initialize NumPy C API')
    library.charex_bench_acquire_allocator.argtypes = [ctypes.py_object]
    library.charex_bench_acquire_allocator.restype = ctypes.c_void_p
    address = ctypes.cast(
        library.charex_bench_acquire_allocator,
        ctypes.c_void_p,
    ).value
    return library, address


try:
    _C_HELPER_LIBRARY, _C_HELPER_ACQUIRE_ADDR = build_c_helper()
except Exception as exc:  # pragma: no cover - exploration-only fallback
    print(f'C helper unavailable: {type(exc).__name__}: {exc}')


@intrinsic
def c_helper_acquire_allocator(typingctx, array):
    if not _C_HELPER_ACQUIRE_ADDR or not is_stringdtype_array_type(array):
        return None

    sig = signature(types.voidptr, array)

    def codegen(context, builder, signature, args):
        array_struct = context.make_array(signature.args[0])(
            context, builder, args[0],
        )
        uintp = context.get_value_type(types.uintp)
        callback_type = ir.FunctionType(ir.IntType(8).as_pointer(), [uintp])
        callback_addr = context.get_constant(
            types.uintp, _C_HELPER_ACQUIRE_ADDR,
        )
        callback = builder.inttoptr(
            callback_addr, callback_type.as_pointer(),
        )
        parent_addr = builder.ptrtoint(array_struct.parent, uintp)
        return builder.call(callback, [parent_addr])

    return sig, codegen


@njit(nogil=True, cache=False)
def current_strlen(values):
    return np.strings.str_len(values)


@njit(nogil=True, cache=False)
def current_per_element_strlen(values):
    result = np.empty(values.size, np.int64)
    for i in range(values.size):
        allocator = stringdtype_acquire_allocator(values)
        result[i] = stringdtype_codepoint_len(values, i, allocator)
        stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def callback_acquire_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = callback_acquire_allocator(values)
    for i in range(values.size):
        result[i] = stringdtype_codepoint_len(values, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def default_descriptor_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = default_descriptor_acquire_allocator(values)
    for i in range(values.size):
        result[i] = stringdtype_codepoint_len(values, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


@njit(nogil=True, cache=False)
def c_helper_strlen(values):
    result = np.empty(values.size, np.int64)
    allocator = c_helper_acquire_allocator(values)
    for i in range(values.size):
        result[i] = stringdtype_codepoint_len(values, i, allocator)
    stringdtype_release_allocator(allocator)
    return result


def bench_methods(methods, values, repeat=15):
    number = max(1, min(1000, 100000 // max(values.size, 1)))
    samples = {label: [] for label, _ in methods}
    order = list(methods)
    rng = random.Random(1009 + values.size)

    for _, fn in methods:
        np.testing.assert_array_equal(fn(values), np.strings.str_len(values))

    for _ in range(repeat):
        rng.shuffle(order)
        for label, fn in order:
            start = time.perf_counter()
            for _ in range(number):
                fn(values)
            samples[label].append((time.perf_counter() - start) / number)

    results = {}
    for label in samples:
        result = {
            'min': min(samples[label]) * 1000,
            'median': statistics.median(samples[label]) * 1000,
        }
        results[label] = result
        print(
            f'{label:28} min {result["min"]:8.3f} ms  '
            f'median {result["median"]:8.3f} ms'
        )
    return results


def main():
    for n in [100, 1000, 10000, 100000]:
        values = np.array(
            ['alpha', 'é', '🙂', '', 'a\x00b'] * (n // 5),
            dtype='T',
        )
        print(f'\nn={n}')
        methods = [
            ('numpy', np.strings.str_len),
            ('current parent-offset', current_strlen),
            ('acquire per element', current_per_element_strlen),
            ('callback acquire', callback_acquire_strlen),
            ('default descriptor', default_descriptor_strlen),
        ]
        if _C_HELPER_ACQUIRE_ADDR:
            methods.append(('C helper acquire', c_helper_strlen))
        results = bench_methods(methods, values)
        baseline = results['numpy']['median']
        for label in results:
            if label != 'numpy':
                print(
                    f'{label:28} median speedup '
                    f'{baseline / results[label]["median"]:8.3f}x'
                )


if __name__ == '__main__':
    main()
