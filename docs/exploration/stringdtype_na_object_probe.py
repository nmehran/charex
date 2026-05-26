"""Probe NumPy StringDType ``na_object`` semantics.

This script intentionally uses NumPy directly.  It records operation-level
behavior before charex prototypes any sentinel handling, because StringDType
missing values are not governed by one universal propagation rule.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np


StringDType = np.dtypes.StringDType


UNARY_OPS = [
    ("str_len", np.strings.str_len),
    ("isalpha", np.strings.isalpha),
    ("isalnum", np.strings.isalnum),
    ("isdecimal", np.strings.isdecimal),
    ("isdigit", np.strings.isdigit),
    ("isnumeric", np.strings.isnumeric),
    ("isspace", np.strings.isspace),
    ("istitle", np.strings.istitle),
    ("isupper", np.strings.isupper),
    ("islower", np.strings.islower),
]


BINARY_OPS = [
    ("equal", np.strings.equal),
    ("not_equal", np.strings.not_equal),
    ("greater", np.strings.greater),
    ("greater_equal", np.strings.greater_equal),
    ("less", np.strings.less),
    ("less_equal", np.strings.less_equal),
]


AFFIX_OPS = [
    ("startswith", np.strings.startswith),
    ("endswith", np.strings.endswith),
]


SEARCH_OPS = [
    ("find", np.strings.find),
    ("rfind", np.strings.rfind),
    ("count", np.strings.count),
    ("index", np.strings.index),
    ("rindex", np.strings.rindex),
]


def format_scalar(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return repr(value)


def format_value(value):
    if isinstance(value, np.ndarray):
        return "[" + ", ".join(format_scalar(v) for v in value.tolist()) + "]"
    return format_scalar(value)


def call_repr(fn, *args):
    try:
        return "OK " + format_value(fn(*args))
    except Exception as exc:
        return f"ERR {type(exc).__name__}: {str(exc).splitlines()[0]}"


def array_for(na_object, values):
    return np.array(values, dtype=StringDType(na_object=na_object))


def print_operation_truth_tables():
    for na_object in (None, np.nan, "MISSING"):
        dtype = StringDType(na_object=na_object)
        values = array_for(na_object, ["a", na_object, "", "MISSING", "aa"])
        patterns = array_for(na_object, ["a", "a", na_object, "", "z"])

        print(f"\n## na_object={format_scalar(na_object)}")
        print("values  ", format_value(values), repr(values.dtype))
        print("patterns", format_value(patterns), repr(patterns.dtype))

        print("\n### unary")
        for name, op in UNARY_OPS:
            print(f"{name:14} {call_repr(op, values)}")

        print("\n### binary self")
        for name, op in BINARY_OPS:
            print(f"{name:14} {call_repr(op, values, values)}")

        print("\n### binary scalar 'a'")
        for name, op in BINARY_OPS:
            print(f"{name:14} {call_repr(op, values, 'a')}")

        print("\n### affix scalar 'a'")
        for name, op in AFFIX_OPS:
            print(f"{name:14} {call_repr(op, values, 'a')}")

        print("\n### search scalar 'a'")
        for name, op in SEARCH_OPS:
            print(f"{name:14} {call_repr(op, values, 'a')}")

        print("\n### pattern-side sentinel")
        for name, op in AFFIX_OPS + SEARCH_OPS:
            print(f"{name:14} {call_repr(op, values, patterns)}")


def numpy_c_api_slots():
    capsule = np._core.multiarray._ARRAY_API
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    api = get_pointer(capsule, None)
    return (ctypes.c_void_p * 400).from_address(api)


class StaticString(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("buf", ctypes.c_void_p)]


def print_low_level_load_truth_table():
    slots = numpy_c_api_slots()
    acquire = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(slots[316])
    release = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(slots[318])
    load = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(StaticString),
    )(slots[313])

    print("\n## low-level NpyString_load")
    for na_object in (None, np.nan, "MISSING"):
        arr = array_for(na_object, ["a", na_object, "", "MISSING", "aa"])
        display = format_value(arr)
        base = arr.__array_interface__["data"][0]
        allocator = acquire(id(arr.dtype))
        rows = []
        for i in range(arr.size):
            out = StaticString()
            status = load(
                allocator,
                base + i * arr.itemsize,
                ctypes.byref(out),
            )
            data = ctypes.string_at(out.buf, out.size) \
                if out.buf and out.size else b""
            rows.append((status, out.size, data))
        release(allocator)

        print(f"\n### na_object={format_scalar(na_object)} {display}")
        for i, row in enumerate(rows):
            print(f"{i}: status={row[0]} size={row[1]} data={row[2]!r}")


def main():
    print_operation_truth_tables()
    print_low_level_load_truth_table()


if __name__ == "__main__":
    main()
