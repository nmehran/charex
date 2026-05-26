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

This compares active `np.strings.str_len` access candidates. Rejected Tranche 1
experiments and their rationale are kept in
`docs/exploration/stringdtype_access_rejected.md`. The benchmark randomizes
measurement order and reports per-call median timings.

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

Fourth pass findings:

- A SWAR-style continuation counter is the strongest long-string idea tested.
  It trims backward, then counts UTF-8 continuation bytes eight at a time with
  unaligned `i64` loads and `llvm.ctpop.i64`; length is
  `effective_size - continuation_count`.
- The SWAR path is excellent for long and Unicode-heavy strings, but its fixed
  overhead loses on short ASCII and NUL-heavy inline data.
- Size-threshold hybrids for the SWAR path did not improve the clean story.
  The extra branch and larger generated function made results noisier and did
  not consistently beat the simpler candidates.
- The SWAR path uses unaligned word loads. That is viable as a prototype on the
  local x86_64 target, but needs portability review before becoming public
  production surface.

Candidate-only 100k-row medians from the fourth pass:

| case | current | backward | backward+SWAR | peeled16 | peeled16+SWAR | best |
| ---- | ------- | -------- | ------------- | -------- | ------------- | ---- |
| mixed short | 0.506 ms | 0.365 ms | 0.380 ms | 0.328 ms | 0.322 ms | peeled16+SWAR |
| ASCII short | 0.689 ms | 0.458 ms | 0.466 ms | 0.435 ms | 0.436 ms | peeled16 |
| NUL heavy | 0.403 ms | 0.339 ms | 0.352 ms | 0.286 ms | 0.309 ms | peeled16 |
| Unicode short | 0.743 ms | 0.485 ms | 0.457 ms | 0.581 ms | 0.577 ms | backward+SWAR |
| long mixed | 2.300 ms | 1.288 ms | 0.685 ms | 2.277 ms | 0.704 ms | backward+SWAR |

The durable result is that backward trim remains the clean minimal baseline,
while SWAR continuation counting is the main max-performance long-string and
Unicode-heavy candidate.

Fifth pass findings:

- ASCII-skipping SWAR helps some short mostly-ASCII distributions, but it is
  not a cleaner universal replacement for plain SWAR. It adds a branch in the
  word loop and loses on long mixed/ASCII cases.
- Four-word unrolling is rejected. It increased generated LLVM size
  substantially and was slower or noisier in every relevant distribution.
- Word-level trailing-NUL trim is a real but narrow win. It is excellent when
  the physical StringDType payload has long trailing NUL runs, but it is slower
  than normal backward trim on ordinary mixed and ASCII payloads.
- None of these variants produce a new compiler-level vectorization story. The
  durable speedups still come from algorithmic shape and explicit word loads,
  not from LLVM discovering a better loop form.

Representative 100k-row medians from the fifth pass:

| case | current | backward | SWAR | ASCII-skip SWAR | unroll4 SWAR | word-trim SWAR | best |
| ---- | ------- | -------- | ---- | --------------- | ------------ | -------------- | ---- |
| mixed short | 0.504 ms | 0.385 ms | 0.381 ms | 0.358 ms | 0.410 ms | 0.392 ms | peeled16/SWAR |
| Unicode short | 0.741 ms | 0.484 ms | 0.455 ms | 0.476 ms | 0.476 ms | 0.473 ms | SWAR |
| long mixed | 2.276 ms | 1.273 ms | 0.673 ms | 0.692 ms | 0.929 ms | 0.706 ms | SWAR |
| long ASCII | 5.286 ms | 2.735 ms | 1.034 ms | 1.091 ms | 1.217 ms | 1.084 ms | SWAR |
| long NUL-tail | 1.586 ms | 3.142 ms | 2.220 ms | 2.257 ms | 2.112 ms | 0.769 ms | word-trim SWAR |

Generated-code inspection from this pass:

| variant | LLVM size | ctpop | ctlz | note |
| ------- | --------- | ----- | ---- | ---- |
| backward byte-count | 37.7 KB | 0 | 0 | clean minimal baseline |
| backward SWAR | 43.1 KB | 8 | 0 | best broad max-performance candidate |
| ASCII-skip SWAR | 44.4 KB | 8 | 0 | distribution-sensitive |
| unroll4 SWAR | 63.7 KB | 32 | 0 | rejected |
| word-trim SWAR | 44.8 KB | 8 | 2 | narrow NUL-tail candidate |

Final tranche-1 recommendation:

- Minimal production candidate: descriptor C helper, operation-scope allocator,
  hoisted data pointer, trailing-NUL trim, and byte-wise continuation counting.
  The first distilled runtime pass uses all-zero word skipping for the trim
  loop, then falls back to byte trimming for the final partial word. This keeps
  the no-threshold shape while avoiding the long trailing-NUL pathology of pure
  byte-wise backward trim.
- Max-performance candidate: add plain SWAR continuation counting after
  portability and code-layout costs are resolved.
- Keep the stronger `ctlz` word-trim variant as a documented special-case
  candidate, not as the default kernel.
- Keep peeled-16 as a specialized inline-string candidate, not part of the base
  kernel until broader operations show it pays for its surface area.

Runtime distillation checkpoint:

- `np.strings.str_len` now hoists the packed StringDType data pointer once per
  call instead of rebuilding the array data pointer inside each element access.
- The length kernel trims all-zero trailing words, trims the final partial word
  byte-wise, then counts UTF-8 code points with the byte-wise continuation
  predicate.
- A 100k-row exploration-harness run on Python 3.12.8, NumPy 2.4.6,
  and Numba 0.65.1:

| case | charex | NumPy | speedup |
| ---- | ------ | ----- | ------- |
| mixed short | 0.381 ms | 0.586 ms | 1.54x |
| ASCII short | 0.487 ms | 0.807 ms | 1.66x |
| NUL heavy | 0.365 ms | 0.518 ms | 1.42x |
| Unicode short | 0.522 ms | 1.128 ms | 2.16x |
| long mixed | 1.300 ms | 14.489 ms | 11.15x |
| long ASCII | 3.098 ms | 40.529 ms | 13.08x |
| long NUL-tail | 0.901 ms | 6.430 ms | 7.14x |

## Tranche 2: Equality

Exploratory benchmark:

```bash
python docs/exploration/stringdtype_equal_bench.py
```

This compares active `np.strings.equal` candidates on default `StringDType`
arrays. The benchmark randomizes measurement order and validates every
candidate against NumPy before timing.

Implementation checkpoint:

- `equal` and `not_equal` support one-dimensional C-contiguous default
  `StringDType` arrays.
- Scalar/StringDType mixed comparisons are rejected for now.
- Allocators are acquired as a pair for binary operations. Acquiring the same
  allocator twice can deadlock on same-array comparisons.
- The hot loop hoists both ndarray data pointers and passes packed-record
  pointers to the equality intrinsic.
- The equality intrinsic uses full-buffer `memcmp` first, then falls back to
  first-NUL-prefix comparison only when the full buffer differs. This preserves
  NumPy's observed embedded-NUL behavior without the old byte-by-byte long
  string cliff.

Rejected equality candidates:

- The original branchless byte loop is clean for short strings, but scans the
  entire UTF-8 byte span in generated LLVM and was about 0.09x NumPy on 100k
  long equal and late-mismatch cases.
- `memchr` before `memcmp` fixes long strings, but pays an unnecessary extra
  scan for fully equal strings.
- A packed-size hybrid (`size <= 16`) is defensible, but did not clearly beat
  the simpler full-`memcmp` shape after broader cases were added.

Representative post-checkpoint 100k-row medians:

| case | NumPy | charex | speedup |
| ---- | ----- | ------ | ------- |
| equal short | 0.862 ms | 0.787 ms | 1.10x |
| first mismatch | 0.940 ms | 0.908 ms | 1.04x |
| late mismatch | 0.917 ms | 0.897 ms | 1.02x |
| unequal byte length | 0.886 ms | 0.553 ms | 1.60x |
| embedded NUL | 0.932 ms | 0.945 ms | 0.99x |
| Unicode equal | 0.908 ms | 0.744 ms | 1.22x |
| long equal | 1.808 ms | 1.661 ms | 1.09x |
| long late mismatch | 1.908 ms | 1.836 ms | 1.04x |
| long first mismatch | 3.028 ms | 3.246 ms | 0.93x |

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
    acquires the allocator once per array operation. StringDType support now
    requires the small compiled helper for descriptor-to-allocator access.
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
- Require the compiled helper for descriptor-to-allocator access. If the helper
  is unavailable, fail during StringDType typing instead of reading private
  NumPy object layout.
- Reject `StringDType(na_object=...)` arrays until sentinel propagation is
  implemented exactly.
- Keep this in charex while prototyping. The dtype recognition and helper
  approach are enough for exploration; a future upstream Numba proposal can be
  cut from the distilled implementation if it proves generally useful.

## Remaining Questions

- Whether the max-performance `str_len` path should add SWAR continuation
  counting or peeled-16 inline handling after portability, code-layout, and
  broader operation reuse are understood.
- What exact propagation rules do missing sentinels require for each
  `np.strings` operation?
- How much of the access/helper layer should become reusable infrastructure
  across comparison, occurrence, predicate, and transformation kernels?
