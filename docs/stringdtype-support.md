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

Second pass findings:

- The default-descriptor shortcut is correctness-invalid for longer
  heap-backed strings. It returns the low-level load failure path because those
  values require the array's real allocator.
- Hoisting the array data pointer and using the known 16-byte packed element
  stride slightly improves the current two-pass kernel and reduces generated
  LLVM size.
- One-pass length kernels are faster for short strings, especially ASCII and
  trailing-NUL-heavy data, but slower for longer strings. They also did not
  generate LLVM vector loops in the local build.
- The current/two-pass shape does generate vector loops and is the strongest
  long-string path.
- A `size <= 16` hybrid is the best balanced prototype so far. The threshold is
  tied to the packed element size, not to one benchmark case, but it still adds
  both loop bodies and a per-element branch.
- A whole-loop C helper is not attractive. It hides the hot loop from
  Numba/LLVM and was slower than the generated-loop variants, especially on
  longer strings.

Representative 100k-row medians from the second pass:

| case | current | one-pass best | two-pass data | hybrid16 | best |
| ---- | ------- | ------------- | ------------- | -------- | ---- |
| mixed short | 0.506 ms | 0.380 ms | 0.509 ms | 0.438 ms | one-pass |
| ASCII short | 0.685 ms | 0.559 ms | 0.684 ms | 0.589 ms | one-pass |
| NUL heavy | 0.402 ms | 0.326 ms | 0.380 ms | 0.378 ms | one-pass |
| Unicode short | 0.742 ms | 0.589 ms | 0.754 ms | 0.621 ms | one-pass |
| long mixed | 2.271 ms | 3.358 ms | 2.188 ms | 2.229 ms | two-pass |

Near-term candidate: replace parent-offset descriptor access with a C helper,
keep allocator acquisition at operation scope, and choose between a minimal
two-pass data-pointer kernel or the `size <= 16` hybrid after broader operation
coverage shows whether the extra surface is justified.

Third pass findings:

- Backward trimming is a major improvement over the forward effective-size
  scan. It checks trailing NUL bytes from the end, then counts code points over
  the effective prefix. This is especially strong for long heap-backed strings
  and Unicode-heavy short strings.
- A safe peeled-16 small-string path is the fastest inline-size path. It uses
  fixed lanes guarded by `size > lane`, so it does not read beyond the
  `NpyString_load` span.
- The fastest short and fastest long kernels do not combine for free. A
  peeled-16/backward hybrid is strong, but code layout and the per-element
  branch cost mean it can be slower on short-only data than a peeled-16/two-pass
  hybrid, even when the fallback is never taken.
- The C-helper allocator path tracks the parent-offset prototype closely with
  these kernels, so the final access direction remains valid.

Representative 100k-row medians from the third pass:

| case | current | backward | peeled16 | peeled16/backward | best |
| ---- | ------- | -------- | -------- | ----------------- | ---- |
| mixed short | 0.505 ms | 0.363 ms | 0.328 ms | 0.349 ms | peeled16 |
| ASCII short | 0.687 ms | 0.456 ms | 0.434 ms | 0.462 ms | peeled16 |
| NUL heavy | 0.402 ms | 0.325 ms | 0.286 ms | 0.332 ms | peeled16 |
| Unicode short | 0.741 ms | 0.480 ms | 0.580 ms | 0.575 ms | backward |
| long mixed | 2.280 ms | 1.277 ms | 2.250 ms | 1.300 ms | backward |

Near-term candidate after this pass: use a C helper only for descriptor access,
hoist the data pointer, and treat backward trim as the default long-string
kernel. Continue evaluating whether peeled-16 should be used as the universal
small-string path or only exposed through specialized fast paths where the code
layout cost is acceptable.

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

## Resolved Decisions

- Call NumPy's `NpyString_*` C API directly from nopython-generated code for
  element unpacking.
- Keep C code limited to descriptor/allocator access unless a later benchmark
  shows a specific whole-loop helper wins. The whole-loop helper tried here was
  slower than Numba-generated loops.
- Acquire each StringDType allocator once per array operation, outside the
  element loop. Per-element acquire is too expensive.
- Replace the parent-object descriptor offset probe with a compiled helper
  before any merge-ready implementation. The offset probe is exploratory only.
- Reject `StringDType(na_object=...)` arrays until sentinel propagation is
  implemented exactly.
- Keep this in charex while prototyping. The dtype recognition and helper
  approach are enough for exploration; a future upstream Numba proposal can be
  cut from the distilled implementation if it proves generally useful.

## Remaining Questions

- What is the final `str_len` kernel shape: backward trim only, peeled-16 for
  small strings plus backward fallback, or separate specialized fast paths that
  avoid slowing the small path with a larger fallback body?
- What exact propagation rules do missing sentinels require for each
  `np.strings` operation?
- How much of the access/helper layer should become reusable infrastructure
  across comparison, occurrence, predicate, and transformation kernels?
- What packaging shape cleanly provides the compiled helper while preserving
  NumPy 1.x compatibility?
