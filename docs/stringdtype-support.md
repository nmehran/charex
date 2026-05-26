# StringDType Support Exploration

This branch is exploratory. The immediate goal is to learn the smallest
correct implementation shape for NumPy 2.x `StringDType`, then distill the
result into a compact merge-ready patch.

## Current Facts

Verified locally with NumPy 2.4.6 and Numba 0.65.1:

- `np.dtypes.StringDType()` has `kind == 'T'`, `char == 'T'`,
  `itemsize == 16`, and `hasobject is True`.
- `numba.typeof(np.array(['a'], dtype=np.dtypes.StringDType()))` raises
  `NumbaValueError: Unsupported array dtype: StringDType()`.
- `@njit` functions fail before charex overload dispatch can run.
- StringDType scalars currently come back to Python as normal `str` objects,
  so scalar support is not the blocker. Array typing is the blocker.
- `np.strings.str_len` counts Unicode code points, not UTF-8 bytes.

Reproducible probe:

```bash
python docs/exploration/stringdtype_probe.py
```

NumPy exposes a C API for unpacking packed StringDType records:

- `NpyString_load`
- `NpyString_pack`
- `NpyString_pack_null`
- `NpyString_acquire_allocator`
- `NpyString_release_allocator`

The raw ndarray payload is not a simple UTF-8 byte buffer. Viewing it as bytes
raises `TypeError: Cannot change data-type for array of references.`

## Feasibility Notes

charex can intercept Numba's ndarray typeof hook at import time. A prototype
proved that `dtype.char == 'T'` can be mapped to `types.Array(custom_dtype, ...)`
and that shape/stride access compiles if the custom dtype has a 16-byte data
model.

That is enough to make Numba accept the array shape. It is not enough to read
strings correctly.

The hard part is element unpacking:

- Numba's normal `Array` model stores `data`, shape, strides, itemsize, parent,
  and meminfo, but not the NumPy dtype descriptor.
- `NpyString_load` needs both the packed 16-byte record and the dtype allocator.
- The allocator can be acquired from the `PyArray_StringDTypeObject` descriptor.
- Missing sentinels return `rc == 1` from `NpyString_load`; normal strings
  return `rc == 0`.
- `na_object` variants have operation-specific semantics. For example,
  `StringDType(na_object='MISSING')` makes `np.strings.str_len` return `7`
  for the sentinel value even though the low-level load path reports the null
  status.
- Calling back into NumPy/Python-level element access while holding a
  StringDType allocator can deadlock. Keep C-API unpacking probes and future
  native helpers clear about lock lifetime.

The first implementation should reject non-default or missing-sentinel
StringDType variants unless we can handle them exactly. Correctness is the
contract.

## Tranche 1: Access Layer

Exploratory benchmark:

```bash
python docs/exploration/stringdtype_access_bench.py
```

This compares allocator access strategies for `np.strings.str_len` on mixed
short `StringDType` values. The benchmark randomizes measurement order and
reports per-call median timings.

Local result on Python 3.12.8, NumPy 2.4.6, Numba 0.65.1:

| rows | NumPy | parent offset | C helper | default descriptor | callback | per element acquire |
| ---- | ----- | ------------- | -------- | ------------------ | -------- | ------------------- |
| 100 | 0.001 ms | 0.009 ms | 0.009 ms | 0.009 ms | 0.010 ms | 0.010 ms |
| 1,000 | 0.006 ms | 0.013 ms | 0.014 ms | 0.013 ms | 0.015 ms | 0.026 ms |
| 10,000 | 0.058 ms | 0.060 ms | 0.065 ms | 0.060 ms | 0.064 ms | 0.186 ms |
| 100,000 | 0.575 ms | 0.509 ms | 0.526 ms | 0.514 ms | 0.530 ms | 1.743 ms |

Findings:

- Acquiring the allocator once per operation is required. Per-element acquire
  is roughly 3x slower at larger sizes.
- A small C helper that reads `PyArray_DESCR(array)` performs close to the
  current parent-offset prototype and is much cleaner.
- A Python callback is slower and should not be part of the final shape.
- A default-descriptor constant is fast, but it is not general enough for
  dtype variants and missing-sentinel semantics.
- The current parent-offset prototype is useful for exploration only. It reads
  NumPy object layout discovered at import time and should be replaced before a
  merge-ready implementation.
- Until sentinel propagation is implemented exactly, charex rejects
  `StringDType(na_object=...)` arrays during Numba typing.

## Prototype Order

1. Type recognition only:
   - override Numba ndarray typeof for `dtype.char == 'T'`;
   - represent the element storage as a 16-byte packet;
   - prove shape and strides still compile.
2. Native unpack helper:
   - get a UTF-8 byte span for one element through NumPy's `NpyString_*` API;
   - define lifetime/locking rules explicitly;
   - reject or handle missing values deliberately.
  - The first callback prototype was correct but hundreds of times slower
    than NumPy.
  - Current prototype calls NumPy's `NpyString_*` C API directly from LLVM and
    acquires the allocator once per array operation. This is the right
    performance direction, but it currently discovers the ndarray descriptor
    offset at import time instead of using a compiled C helper.
3. First operation:
   - `np.strings.str_len` for one-dimensional C-contiguous arrays;
   - count Unicode code points from UTF-8 bytes.
  - Current prototype matches NumPy for normal values and raises
    `NumbaValueError` for `StringDType(na_object=...)` variants.
  - Quick local timing on mixed short strings: still slower below about 1k
    rows, but faster than NumPy at 10k+ rows.
4. First comparison:
   - `np.strings.equal`;
   - then `not_equal`.
   - Current prototype supports non-scalar StringDType array-array
     `equal`/`not_equal`.
   - Observed NumPy behavior: StringDType equality requires the stored byte
     lengths to match, then compares through the first NUL byte. Bytes after
     the first NUL do not affect equality if the stored byte lengths match.
5. Broader operations:
   - `startswith`, `endswith`;
   - `find`, `rfind`, `count`, `index`, `rindex`;
   - predicates.

## Open Questions

- Can the native helper safely call the NumPy C API from Numba nopython code
  while preserving charex's current `nogil` posture?
- Should StringDType kernels acquire the allocator once per call, once per
  outer loop, or once per element?
- Can the dtype descriptor be reached safely from the array `parent` object in
  lowering, or do we need a small compiled extension helper?
- How should missing sentinels propagate for each `np.strings` operation?
- Is this suitable for charex as an extension, or should part of it be upstream
  Numba dtype support?
