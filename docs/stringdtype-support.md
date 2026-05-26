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

## Tranche 3: Prefix/Suffix And Span Slicing

The next tranche should not jump directly to the full occurrence catalog. The
shared primitive is Python-style `start`/`end` slicing over a UTF-8
StringDType span. `startswith` and `endswith` are the smallest useful public
operations that exercise that primitive without also committing to substring
search strategy.

Target operations:

- `np.strings.startswith`
- `np.strings.endswith`

Initial runtime scope:

- default `StringDType` only, still rejecting `na_object` variants;
- one-dimensional C-contiguous arrays;
- array-array same-shape inputs first;
- scalar prefix/suffix support only after the scalar UTF-8 access path is
  understood and measured;
- `start` and `end` with positive, negative, omitted, and out-of-range values.

Correctness questions to answer before implementation:

- Whether `start` and `end` are always Python codepoint offsets for
  StringDType, including non-BMP characters.
- How trailing NUL trimming interacts with prefix/suffix and slicing.
- Whether embedded NUL comparison follows Python string behavior for these
  operations, unlike equality's first-NUL-prefix behavior.
- Exact empty-prefix and empty-suffix behavior for empty and non-empty spans.
- Whether NumPy accepts mixed `StringDType` array and Python/NumPy scalar
  pattern inputs, and what error shape charex should use before supporting
  them.

Implementation shape to prototype:

- Reuse operation-scope allocator acquisition and hoisted packed data pointers.
- Add a small internal span representation: status, byte pointer, raw byte
  size, and effective byte size after operation-specific trimming.
- Add a slice-normalization helper that converts codepoint `start`/`end` into
  byte offsets. Keep this byte-wise and simple first; aggressive SWAR or index
  caching belongs in a later optimization pass.
- Implement prefix/suffix checks with length guards plus `memcmp`.
- Keep the first public overload narrow and explicit. Unsupported scalar,
  dimensionality, layout, and shape cases should fail early with targeted
  messages.

Benchmark and test harness:

- Add a focused exploration benchmark for `startswith` and `endswith`.
- Randomize measurement order and validate every candidate against NumPy before
  timing, matching the previous exploration harnesses.
- Cover ASCII, multibyte Unicode, non-BMP characters, empty pattern, empty
  value, trailing NUL, embedded NUL, positive/negative `start`/`end`, early
  mismatch, late mismatch, and long strings.
- Include same-array, readonly, empty-array, shape mismatch, non-contiguous,
  multidimensional, and unsupported mixed-input tests.

Acceptance bar for this tranche:

- Exact NumPy behavior for the supported surface.
- No mutation of inputs.
- No per-element allocator acquisition.
- No arbitrary performance gates. Any branch must follow from semantics or a
  stable representation boundary, not from one benchmark case.
- No broad helper abstraction unless `find`/`count` reuse is already obvious
  from the prototype.

Runtime checkpoint:

- `np.strings.startswith` and `np.strings.endswith` support same-shape,
  one-dimensional C-contiguous default `StringDType` arrays.
- 0-D arrays, multidimensional arrays, non-contiguous arrays, mixed scalar/array
  inputs, and `na_object` variants are rejected for now.
- The implementation reuses operation-scope paired allocator acquisition and
  hoisted packed data pointers.
- Slice normalization converts codepoint `start`/`end` to UTF-8 byte offsets,
  trims trailing NUL bytes consistently with `StringDType` string length, and
  uses `memcmp` for prefix/suffix comparison.
- The first codepoint-offset implementation had a multibyte suffix bug: it
  stopped after the leading byte of the target codepoint. The current helper
  advances through UTF-8 continuation bytes before returning the next
  codepoint boundary.

Exploratory benchmark:

```bash
python docs/exploration/stringdtype_affix_bench.py
```

Representative 100k-row medians on Python 3.12.8, NumPy 2.4.6,
Numba 0.65.1:

| case | startswith | endswith |
| ---- | ---------- | -------- |
| short default | 1.34x | 1.17x |
| short slice | 1.56x | 1.39x |
| empty pattern | 1.11x | 1.18x |
| embedded NUL | 1.71x | 1.44x |
| Unicode | 1.56x | 1.34x |
| long equal | 3.35x | 4.80x |
| long late mismatch | 7.57x | 7.61x |

Review-pass finding:

- A combined `start`/`end` byte-offset scan looked attractive because it would
  replace two UTF-8 scans with one. In practice it made the loop body branchier
  and regressed the focused affix benchmark, especially short `endswith`.
  Keep the simpler independent offset scans until a more substantial
  slice-indexing strategy is justified.

Mixed Python `str` scalar checkpoint:

- `StringDType` value array with Python `str` scalar prefix/suffix keeps the
  existing codepoint bridge for scalar patterns up to the 16-byte packed-record
  size and uses one pre-encoded UTF-8 scalar span for longer patterns.
- Python `str` scalar value with `StringDType` pattern array uses one
  pre-encoded UTF-8 value span and normalizes the scalar slice once per call.
  The rejected alternative normalized the scalar slice inside every element
  check and lost badly except on trailing-NUL-heavy cases.
- The 16-byte boundary is the stable `StringDType` packed-record size, matching
  the scalar comparison bridge. It is not fitted to one benchmark case.

Exploratory benchmark:

```bash
python docs/exploration/stringdtype_scalar_affix_bench.py
```

Representative 100k-row medians on Python 3.12.8, NumPy 2.4.6,
Numba 0.65.1:

| direction | case | current | NumPy | current speedup | prior unicode bridge |
| --------- | ---- | ------- | ----- | --------------- | -------------------- |
| value array / scalar pattern | startswith long prefix | 19.61 ms | 108.34 ms | 5.53x | 22.94 ms |
| value array / scalar pattern | endswith long NUL | 10.96 ms | 104.81 ms | 9.57x | 19.74 ms |
| scalar value / pattern array | startswith short slice | 0.27 ms | 1.26 ms | 4.66x | 0.38 ms |
| scalar value / pattern array | endswith long prefix | 0.29 ms | 59.14 ms | 204.11x | 2.54 ms |
| scalar value / pattern array | startswith long NUL | 0.60 ms | 104.75 ms | 175.59x | 24.70 ms |

## Tranche 4: Substring Search

The next tranche should build on the codepoint span-slicing primitive from
Tranche 3 and add substring search before moving to predicates or
transformations. `find`, `rfind`, and `count` share most of the required
search machinery. `index` and `rindex` should follow after not-found exception
behavior is exact.

Target operations:

- `np.strings.find`
- `np.strings.rfind`
- `np.strings.count`
- then `np.strings.index` and `np.strings.rindex`

Initial runtime scope:

- default `StringDType` only, still rejecting `na_object` variants;
- one-dimensional C-contiguous arrays;
- array-array same-shape inputs first;
- scalar substring support deferred until the scalar UTF-8 access strategy is
  designed;
- `start` and `end` with positive, negative, omitted, and out-of-range values.

Correctness questions to answer before implementation:

- Exact empty-substring behavior:
  - `find` returns the normalized start offset when the slice is valid;
  - `rfind` returns the normalized end offset;
  - `count` returns the number of codepoint insertion positions in the slice.
- Whether result offsets are always codepoint offsets, including multibyte
  Unicode and non-BMP characters.
- How trailing NUL trimming and embedded NUL bytes interact with search and
  returned offsets.
- Whether `count` uses non-overlapping matches, matching Python/NumPy string
  semantics.
- `index`/`rindex` exception semantics: any not-found element should raise
  `ValueError`, not return a partially useful result.
- Whether the broad search primitive should return byte offsets, codepoint
  offsets, or both to avoid repeated UTF-8 scans.

Implementation shape to prototype:

- Reuse operation-scope paired allocator acquisition and hoisted packed data
  pointers.
- Reuse the existing slice normalization helper for `start`/`end`.
- Start with a simple byte-wise search over the normalized slice and convert
  match byte offsets back to codepoint offsets. Keep this correct and readable
  before trying a vectorized or word-based strategy.
- Use `memcmp` for candidate verification after a first-byte check.
- Keep `find` and `rfind` as the first public surface. Add `count` once the
  shared forward iteration is stable. Add `index`/`rindex` last with explicit
  not-found aggregation.

Benchmark and test harness:

- Add a focused exploration benchmark for `find`, `rfind`, and `count`.
- Randomize measurement order and validate every candidate against NumPy before
  timing.
- Cover ASCII, multibyte Unicode, non-BMP characters, empty substring, empty
  value, trailing NUL, embedded NUL, positive/negative `start`/`end`, no-match,
  first match, last match, repeated matches, overlapping-looking matches, and
  long strings.
- Include same-array, readonly, empty-array, shape mismatch, non-contiguous,
  multidimensional, unsupported mixed-input, and not-found exception tests.

Acceptance bar for this tranche:

- Exact NumPy behavior for the supported surface.
- Codepoint offsets, not UTF-8 byte offsets, in public results.
- No mutation of inputs.
- No per-element allocator acquisition.
- No arbitrary performance gates. Search strategy can branch for semantic
  cases such as empty substring or direction, but not for benchmark-fitted
  widths.
- Clear documentation of rejected search strategies so later optimization work
  does not repeat failed experiments.

Runtime checkpoint:

- `np.strings.find`, `np.strings.rfind`, and `np.strings.count` support
  same-shape, one-dimensional C-contiguous default `StringDType` arrays.
- 0-D arrays, multidimensional arrays, non-contiguous arrays, mixed
  scalar/array inputs, and `na_object` variants are rejected for now.
- Results are public codepoint offsets. The internal scan works over UTF-8 byte
  spans and converts match byte positions back to codepoint positions.
- Empty substring behavior follows NumPy: `find` returns the normalized start,
  `rfind` returns the normalized end, and `count` returns the number of
  insertion positions in the normalized slice.
- Search uses a byte-wise scan with a first-byte guard and `memcmp`
  verification. `count` advances by the matched byte span and therefore uses
  non-overlapping matches.
- Trailing-NUL substring behavior is intentionally operation-specific to match
  NumPy. All-NUL substrings are empty. For non-empty substrings, `count` uses
  the raw substring bytes, while `find`/`rfind` use the raw bytes except for a
  single-byte effective substring, which NumPy treats as that one byte.
- `index` and `rindex` are handled in Tranche 5 because they need array-wide
  not-found aggregation before raising.

Exploratory benchmark:

```bash
python docs/exploration/stringdtype_search_bench.py
```

Representative 100k-row speedups on Python 3.12.8, NumPy 2.4.6,
Numba 0.65.1:

| case | find | rfind | count |
| ---- | ---- | ----- | ----- |
| short default | 1.88x | 1.76x | 1.82x |
| short slice | 1.67x | 1.67x | 1.62x |
| empty pattern | 1.37x | 1.26x | 1.36x |
| embedded NUL | 2.00x | 1.84x | 1.79x |
| trailing-NUL pattern | 1.74x | 1.83x | 1.73x |
| Unicode | 2.00x | 2.11x | 1.95x |
| long first | 5.35x | 4.01x | 4.33x |
| long last | 6.47x | 7.60x | 4.37x |
| long no match | 2.49x | 2.33x | 2.41x |
| long repeated | 4.10x | 5.31x | 2.95x |

Review-pass findings:

- A randomized search stress pass over ASCII, multibyte Unicode, non-BMP
  characters, embedded NULs, all-NUL substrings, and positive/negative slices
  matched NumPy for `find`, `rfind`, and `count`.
- Empty supported search inputs now return before acquiring StringDType
  allocators. The broader same-shape limitation is unchanged.
- No additional search strategy was promoted in this pass. More aggressive
  forward-position tracking, byte-to-codepoint indexing, or word/SWAR scans
  belong in the later optimization tranche.

Mixed Python `str` scalar checkpoint:

- `StringDType` value array with Python `str` scalar substring keeps the
  existing codepoint bridge for scalar patterns up to the 16-byte packed-record
  size and uses one pre-encoded UTF-8 scalar span for longer patterns.
- Python `str` scalar value with `StringDType` pattern array uses one
  pre-encoded UTF-8 value span and normalizes the scalar slice once per call.
- `index` and `rindex` use the same optimized `find`/`rfind` paths, but still
  release allocators and UTF-8 spans before raising on not-found results.

Exploratory benchmark:

```bash
python docs/exploration/stringdtype_scalar_search_bench.py
```

Representative 100k-row medians on Python 3.12.8, NumPy 2.4.6,
Numba 0.65.1:

| direction | case | operation | current | NumPy | current speedup | prior unicode bridge |
| --------- | ---- | --------- | ------- | ----- | --------------- | -------------------- |
| value array / scalar pattern | long first | find | 20.02 ms | 125.42 ms | 6.27x | 23.82 ms |
| value array / scalar pattern | long first | rfind | 37.58 ms | 146.49 ms | 3.90x | 147.02 ms |
| value array / scalar pattern | long first | count | 28.69 ms | 126.62 ms | 4.41x | 92.61 ms |
| scalar value / pattern array | long NUL | find | 0.71 ms | 79.98 ms | 113.49x | 24.43 ms |
| scalar value / pattern array | long NUL | count | 0.71 ms | 78.73 ms | 111.68x | 21.86 ms |

## Tranche 5: Index/Rindex Exceptions

This tranche should complete the substring-search information family before
moving to predicates or scalar support. `index` and `rindex` should reuse the
`find`/`rfind` semantics from Tranche 4, but they need exact array-wide
not-found behavior instead of returning `-1`.

Target operations:

- `np.strings.index`
- `np.strings.rindex`

Initial runtime scope:

- default `StringDType` only, still rejecting `na_object` variants;
- one-dimensional C-contiguous arrays;
- array-array same-shape inputs first;
- scalar substring support deferred;
- `start` and `end` with the same accepted forms as `find`/`rfind`.

Correctness questions to answer before implementation:

- Whether NumPy raises `ValueError` if any element is not found, and whether it
  discards all partial results in that case.
- Exact exception message text and type for `index` and `rindex`.
- Whether empty substring and all-NUL substring behavior is identical to
  `find`/`rfind` before applying the not-found check.
- Whether unsupported-input errors should continue matching the narrow
  StringDType search messages from Tranche 4.

Implementation shape to prototype:

- Keep a shared search intrinsic returning `-1` for not found.
- In the array wrapper, fill the result array while tracking a `not_found`
  flag. Release paired allocators before raising.
- Raise once after the loop if any element was not found.
- Avoid per-element exception paths inside the intrinsic; they complicate
  allocator lifetime and make the hot loop harder to reason about.
- Keep this separate from aggressive search optimization. This tranche is about
  exact exception semantics and clean control flow.

Benchmark and test harness:

- Add success and failure tests for `index` and `rindex` covering ASCII,
  multibyte Unicode, non-BMP characters, empty substring, all-NUL substring,
  trailing-NUL patterns, embedded NULs, positive/negative slices, empty arrays,
  readonly arrays, shape mismatch, non-contiguous arrays, multidimensional
  arrays, and mixed scalar/array rejection.
- Add explicit tests that allocator release happens before the not-found
  exception path by exercising same-array inputs after a failing call.
- Benchmark only successful calls in the focused search harness; failure timing
  is less useful and can distort the search-path numbers.

Acceptance bar for this tranche:

- Exact NumPy behavior for all supported `index`/`rindex` success and failure
  cases.
- No allocator leaks or deadlocks after failed searches.
- No mutation of inputs.
- No new search algorithm gates.
- No public support expansion beyond the agreed array-array StringDType scope.

Runtime checkpoint:

- `np.strings.index` and `np.strings.rindex` now support same-shape,
  one-dimensional C-contiguous default `StringDType` arrays.
- Successful calls reuse the `find`/`rfind` search semantics and return
  codepoint offsets.
- Failing calls track a not-found flag across the array, release the paired
  StringDType allocators, then raise `ValueError('substring not found')`.
- Empty arrays return an empty integer result. Mixed scalar/array inputs,
  non-contiguous arrays, multidimensional arrays, 0-D arrays, and `na_object`
  variants remain rejected.
- A failure-path regression test runs a successful StringDType search
  immediately after a failing `index`/`rindex` call to guard allocator release.

Representative successful-call 100k-row speedups on Python 3.12.8,
NumPy 2.4.6, Numba 0.65.1:

| case | index | rindex |
| ---- | ----- | ------ |
| short slice | 1.64x | 1.63x |
| empty pattern | 1.41x | 1.37x |
| embedded NUL | 1.91x | 1.83x |
| Unicode | 1.89x | 2.00x |
| long first | 5.40x | 4.07x |
| long last | 6.41x | 7.63x |
| long repeated | 4.04x | 5.29x |

Review-pass findings:

- `index` and `rindex` now stop scanning after the first not-found element.
  This keeps the allocator release and single exception point, while avoiding
  wasted work on failure-only results.
- The operation classification is kept inside the overload implementation.
  Hoisting it outside looked cleaner, but Numba 0.65.1 did not capture those
  closure booleans reliably during overload typing.
- Failure coverage now includes both a later not-found element and a
  first-element not-found case, followed by a successful search to guard
  allocator release.

## Tranche 6: Predicate Information

This tranche adds the remaining read-only boolean information methods before
moving into scalar support, missing sentinels, or multidimensional behavior.

Target operations:

- `np.strings.isalpha`
- `np.strings.isalnum`
- `np.strings.isdecimal`
- `np.strings.isdigit`
- `np.strings.isnumeric`
- `np.strings.isspace`
- `np.strings.islower`
- `np.strings.isupper`
- `np.strings.istitle`

Initial runtime scope:

- default `StringDType` only, still rejecting `na_object` variants;
- one-dimensional C-contiguous arrays;
- array input only;
- scalar support deferred to the broader scalar StringDType tranche.

Implementation shape:

- Reuse the existing allocator-once-per-call and packed-data-pointer path.
- Trim trailing NUL bytes once per element, matching the behavior already
  established for StringDType length and information routines.
- Decode UTF-8 into Unicode code points inside a shared intrinsic loop.
- Reuse Numba/Python Unicode category helpers for exact predicate semantics.
- Keep simple predicates and cased predicates in one shared loop shape so the
  public overload layer remains small.

Runtime checkpoint:

- All nine predicate operations now match NumPy for Unicode letters, decimal
  digits, digit/numeric-only characters, whitespace control characters,
  titlecase characters, emoji, empty strings, embedded NULs, and trailing NULs.
- Empty arrays and readonly arrays match NumPy.
- Non-contiguous arrays, multidimensional arrays, 0-D arrays, and
  `na_object` variants remain rejected consistently with the current
  StringDType scope.

Benchmark:

```bash
python docs/exploration/stringdtype_predicate_bench.py
```

Representative 100k-row speedups on Python 3.12.8, NumPy 2.4.6,
Numba 0.65.1:

| case | isalpha | isalnum | isdecimal | isdigit | isnumeric | isspace | islower | isupper | istitle |
| ---- | ------- | ------- | --------- | ------- | --------- | ------- | ------- | ------- | ------- |
| ASCII alpha | 3.05x | 2.07x | 2.94x | 3.28x | 2.95x | 2.91x | 6.45x | 4.94x | 4.56x |
| Unicode alpha | 2.01x | 1.55x | 1.96x | 2.04x | 1.98x | 1.98x | 1.48x | 1.06x | 1.61x |
| Numeric mix | 1.95x | 1.45x | 1.83x | 1.79x | 1.93x | 2.00x | 1.49x | 1.56x | 1.44x |
| Long ASCII | 5.78x | 3.16x | 66.15x | 54.73x | 65.85x | 37.97x | 14.21x | 13.21x | 9.82x |

Review-pass notes to revisit before distillation:

- The UTF-8 decode helper is the largest code surface added in this tranche.
  It is shared by all predicates and is the likely reusable primitive for later
  scalar and transformation work.
- The current implementation favors one readable decode loop over specialized
  branchless/SWAR variants. Aggressive predicate optimization is deferred until
  the full StringDType information surface is stable.
- The slowest relative case in this benchmark is `isupper` on mixed Unicode
  alpha input, where the current path is near parity rather than a large win.

## Tranche 7: Ordering Comparisons

This tranche completes the StringDType comparison family for array-array inputs.

Target operations:

- `np.strings.greater`
- `np.strings.greater_equal`
- `np.strings.less`
- `np.strings.less_equal`

Initial runtime scope:

- default `StringDType` only, still rejecting `na_object` variants;
- one-dimensional C-contiguous arrays;
- array-array same-shape inputs;
- scalar and multidimensional support deferred.

Observed NumPy semantics:

- Ordering compares stored UTF-8 bytes lexicographically.
- If the common byte prefix reaches a simultaneous NUL byte, bytes after that
  NUL are ignored for content comparison.
- Stored byte length breaks ties, including simultaneous-NUL ties.
- This means `a\0x` and `a\0y` compare equal, while `a\0` is less than
  `a\0x`, and `a\0x` is less than `abx`.

Implementation shape:

- Add one shared StringDType comparison intrinsic returning `-1`, `0`, or `1`.
- Use C `strncmp` for the content comparison because it matches the
  simultaneous-NUL stop rule directly.
- Apply the stored byte-length tie-break when `strncmp` returns equality.
- Drive all four public order operations from the same intrinsic.

Benchmark:

```bash
python docs/exploration/stringdtype_order_bench.py
```

Representative 100k-row speedup ranges on Python 3.12.8, NumPy 2.4.6,
Numba 0.65.1:

| case | speedup range across order ops |
| ---- | ------------------------------ |
| ASCII mixed | 1.22x-1.28x |
| Unicode mixed | 1.16x-1.35x |
| embedded NUL | 1.11x-1.24x |
| one-sided NUL | 1.31x-1.33x |
| long first mismatch | 1.00x-1.02x |
| long last mismatch | 1.00x-1.04x |
| long equal | 1.00x-1.02x |

Review-pass notes to revisit before distillation:

- Small 1k-row benchmarks remain dominated by call overhead and are slower than
  NumPy. The 100k-row path is the meaningful public target for now.
- A future max-performance branch could evaluate a short-string inline path,
  but this tranche intentionally avoids a threshold gate.

Review-pass findings:

- A randomized ordering stress pass over ASCII, multibyte Unicode, non-BMP
  characters, embedded NULs, one-sided NULs, and low control bytes matched
  NumPy for all four order operations.
- No ordering algorithm changes were promoted. The `strncmp` plus stored-length
  tie-break remains the cleanest current shape.
- Same-shape empty prefix/suffix calls now return before acquiring paired
  StringDType allocators, matching the empty-array fast paths used by the newer
  comparison, search, and predicate overloads.

## Tranche 8: Scalar And Mixed Inputs

This tranche should make the existing StringDType information and comparison
surface useful in the scalar-broadcast cases NumPy accepts, without taking on
general multidimensional broadcasting yet.

Observed NumPy 2.4.6 behavior:

- StringDType array with Python `str` scalar broadcasts over the array.
- Python `str` scalar with StringDType array also broadcasts over the array.
- 0-D `StringDType` arrays behave like scalar operands in binary operations.
- Unary operations accept Python `str` scalars and 0-D `StringDType` arrays and
  return scalar NumPy values.
- `index` and `rindex` still raise `ValueError('substring not found')` if any
  broadcasted element is not found.

Target operations:

- comparisons: `equal`, `not_equal`, `greater`, `greater_equal`, `less`,
  `less_equal`;
- affix/search: `startswith`, `endswith`, `find`, `rfind`, `count`, `index`,
  `rindex`;
- unary information: `str_len` and the nine predicate functions.

Initial runtime scope:

- default `StringDType` arrays only, still rejecting `na_object` variants;
- one-dimensional C-contiguous arrays plus scalar/0-D operands;
- scalar broadcasting only: array-scalar, scalar-array, 0-D-array, and
  scalar-only cases where NumPy accepts them;
- no N-D broadcasting, no non-contiguous layout support, and no missing
  sentinel propagation in this tranche.

Correctness questions to answer before implementation:

- Whether Python `str`, NumPy scalar strings, and 0-D `StringDType` arrays share
  the same trailing-NUL, embedded-NUL, ordering, and search semantics in every
  operation.
- Whether scalar Python strings need a dedicated UTF-8 bridge, or whether a
  small temporary/default `StringDType` pack path is cleaner.
- Exact return types for scalar-only and 0-D-only calls: NumPy scalar vs 0-D
  ndarray vs 1-D ndarray after broadcasting.
- Whether scalar `None`, bytes, fixed-width `S/U`, and non-string scalars should
  route to existing fixed-width overloads, NumPy fallback, or explicit
  rejection.
- How to guarantee allocator release on `index`/`rindex` failure when only one
  side requires a StringDType allocator.

Implementation shape to prototype:

- Keep the current array-array kernels intact.
- Add a narrow scalar span abstraction that can feed the same equality,
  ordering, affix, and search intrinsics without duplicating operation logic.
- Prefer one explicit broadcast loop per operation family over generalized
  broadcasting machinery.
- Handle 0-D `StringDType` arrays by loading the single packed element with the
  normal allocator path.
- For Python `str` scalars, prototype the smallest safe UTF-8 access path first;
  do not depend on private CPython layout unless there is no public or Numba
  helper path.
- Keep scalar-only calls separate until return-type semantics are proven.

Benchmark and test harness:

- Add a focused scalar/mixed benchmark covering array-scalar and scalar-array
  calls for comparisons, affix/search, and predicates.
- Validate every benchmark case against NumPy before timing.
- Cover ASCII, multibyte Unicode, non-BMP characters, empty strings,
  embedded/trailing NULs, all-NUL patterns, positive/negative slices,
  not-found `index`/`rindex`, readonly arrays, empty arrays, and 0-D
  `StringDType` arrays.
- Add direct `@njit` tests for `np.strings.*` calls, not only calls through the
  existing wrapper classes.

Acceptance bar for this tranche:

- Exact NumPy behavior for supported scalar-broadcast cases.
- No regression in existing array-array behavior or performance shape.
- No mutation of inputs.
- No allocator leaks or deadlocks after scalar-mixed failures.
- No generalized N-D broadcasting code.
- No benchmark-fitted thresholds.

Checkpoint:

- 0-D default `StringDType` operands now use the same packed-element allocator
  path as one-dimensional arrays. Unary 0-D calls return scalar values; binary
  0-D/0-D calls return scalar values; 0-D/1-D calls broadcast over the 1-D side.
- Python `str` mixed comparisons now use a Unicode-scalar bridge. It trims
  trailing NULs on the Python scalar side, preserves StringDType's NUL-prefix
  comparison behavior, and rejects invalid surrogate code points like NumPy.
- Python `str` mixed `startswith` and `endswith` now use the same scalar bridge
  with explicit slice normalization for the Unicode-value side.
- Python `str` mixed `find`, `rfind`, `count`, `index`, and `rindex` now use
  explicit scalar-search paths. They preserve the distinct trailing-NUL handling
  for `count` versus `find`/`rfind`, return codepoint indexes, and release
  StringDType allocators before `index`/`rindex` not-found failures.
- NumPy `str_` scalars are typed by Numba as normal Unicode scalars and use the
  same bridge as Python `str`.
- 0-D fixed-width Unicode arrays mixed with `StringDType` now unbox through
  `value[()]` and use the same Unicode scalar bridge for comparisons,
  prefix/suffix checks, search, `index`, and `rindex`.

Scalar bridge distillation:

- Mixed Python `str` comparisons now keep the direct Unicode/codepoint bridge
  for scalars whose trimmed UTF-8 size fits in StringDType's 16-byte packed
  record boundary.
- Longer mixed comparison scalars use a UTF-8 span built once per operation,
  outside the StringDType element loop. ASCII scalars use the existing Unicode
  one-byte buffer directly; non-ASCII scalars use a stack span when small and
  heap storage only when needed.
- Long mixed equality uses a first-word decision path with a `strncmp`
  fallback. This keeps the strong mid-mismatch result without taking the larger
  hybrid candidate.
- Long mixed ordering uses a first-byte prefilter and `strncmp` with stored
  byte-length tie-breaks. Word-wide ordering was rejected for this checkpoint:
  it added surface without a clear enough broad win.
- Rejected cleanup: moving the scalar bridge choice into a per-element helper.
  It made the overload code smaller but would allocate/free the UTF-8 span
  inside the hot loop for long scalars.

500k-row sanity medians after distillation on Python 3.12.8, NumPy 2.4.6,
Numba 0.65.1:

| case | charex | NumPy | speedup |
| ---- | ------ | ----- | ------- |
| equal short ASCII | 1.503 ms | 3.118 ms | 2.07x |
| equal long ASCII | 5.453 ms | 6.397 ms | 1.17x |
| equal long mid mismatch | 2.756 ms | 9.003 ms | 3.27x |
| equal long NUL | 3.950 ms | 5.461 ms | 1.38x |
| equal long Unicode | 4.989 ms | 6.671 ms | 1.34x |
| equal long emoji | 5.277 ms | 7.160 ms | 1.36x |
| greater long first mismatch | 4.488 ms | 12.915 ms | 2.88x |
| greater long mid mismatch | 6.200 ms | 6.862 ms | 1.11x |
| greater long late mismatch | 5.975 ms | 6.880 ms | 1.15x |
| greater long NUL | 3.754 ms | 5.443 ms | 1.45x |

Remaining scalar bridge work:

- Scalar-only return types are covered where NumPy accepts Python `str`,
  NumPy `str_`, and 0-D fixed-width Unicode inputs.
- Non-scalar fixed-width Unicode arrays mixed with `StringDType` now use the
  same Unicode bridge one element at a time for comparisons, prefix/suffix
  checks, search, `index`, and `rindex`.
- `None` is supported for `equal` and `not_equal`, matching NumPy's all-False
  / all-True result. Ordering, prefix/suffix, and search reject `None`, as
  NumPy does.
- Fixed-width bytes (`S`) and bytes scalars are intentionally rejected because
  NumPy has no `StringDType` loop for those mixed operands.
- Object arrays/scalars are still not part of the nopython StringDType bridge.
  NumPy accepts object operands for comparison through Python object semantics,
  but prefix/suffix and search reject them. Supporting the accepted comparison
  subset would require a separate object-mode design, not the current
  `StringDType` fast path.
- Keep future scalar optimizations operation-specific. The comparison bridge is
  not automatically the right shape for search or transformations.

Fixed-width Unicode array checkpoint:

- Implemented one-dimensional fixed-width Unicode arrays mixed with
  scalar/0-D/1-D default `StringDType`.
- The first implementation uses the existing Unicode scalar bridge per element:
  `str(value[i])`, then the same StringDType-vs-Unicode intrinsics already used
  by Python `str` and 0-D fixed-width Unicode operands.
- This keeps the code path simple and exact, with no new raw fixed-width `U`
  memory intrinsics yet. A future performance branch can revisit direct `U`
  buffer access if broader benchmarking justifies the extra surface.
- Fixed-width Unicode arrays do not need to be C-contiguous for this path
  because the kernels only use normal indexed access on the Unicode side.
- Distillation pass kept the explicit operation-family branches, but moved the
  repeated `greater`/`less`, prefix/suffix, and search operation dispatch into
  tiny force-inlined helpers. This reduced duplicated hot-loop control flow
  without moving allocator acquisition or UTF-8 span allocation into per-element
  helpers.

200k-row sanity medians after distillation on Python 3.12.8, NumPy 2.4.6,
Numba 0.65.1:

| case | charex | NumPy | speedup |
| ---- | ------ | ----- | ------- |
| equal StringDType/Unicode | 5.735 ms | 6.883 ms | 1.20x |
| equal Unicode/StringDType | 7.757 ms | 9.579 ms | 1.23x |
| greater StringDType/Unicode | 6.234 ms | 7.006 ms | 1.12x |
| startswith StringDType/Unicode | 7.120 ms | 8.763 ms | 1.23x |
| find StringDType/Unicode | 7.458 ms | 9.012 ms | 1.21x |
| find Unicode/StringDType | 10.347 ms | 11.430 ms | 1.10x |
| greater StringDType/StringDType | 1.082 ms | 1.337 ms | 1.24x |
| startswith StringDType/StringDType | 1.827 ms | 3.111 ms | 1.70x |
| find StringDType/StringDType | 1.947 ms | 3.905 ms | 2.01x |
| count StringDType/StringDType | 2.036 ms | 3.293 ms | 1.62x |

## Tranche 9: Missing Sentinels And `na_object`

This tranche should replace the current blanket rejection of
`StringDType(na_object=...)` with exact NumPy behavior. It should not begin
until the supported default-`StringDType` operation surface is stable enough
that sentinel propagation can be audited operation by operation.

Currently unsupported variants:

- `np.dtypes.StringDType(na_object=None)`
- `np.dtypes.StringDType(na_object=np.nan)`
- `np.dtypes.StringDType(na_object="MISSING")`
- other user-provided sentinel values accepted by NumPy

Known risk:

- `NpyString_load` can report a null-string status for sentinel values, while
  NumPy operation semantics are not uniformly "return missing" or "raise".
  For example, a string sentinel can behave like a normal string in operations
  such as `np.strings.str_len`.

Correctness questions to answer before implementation:

- Exact result and exception behavior for every currently supported operation:
  `str_len`, predicates, equality/order comparisons, prefix/suffix, search,
  `index`, and `rindex`.
- Whether `None`, `np.nan`, and string sentinels have different propagation
  rules.
- Whether sentinel values compare equal to themselves, to normal strings with
  the same text, or to neither, per operation.
- Whether search-like operations treat a sentinel substring as missing,
  ordinary text, empty, or an error.
- Whether scalar/0-D sentinel behavior differs from array element behavior.
- Whether result dtypes change when missing propagation is involved.

Implementation shape to prototype:

- Keep the current default-`StringDType` fast path unchanged.
- Add sentinel-aware load metadata rather than treating load status alone as
  final operation semantics.
- Keep operation-specific propagation explicit until a smaller shared rule is
  proven by tests.
- Preserve allocator release discipline on every success and failure path.

Benchmark and test harness:

- Build an operation-by-operation truth table against NumPy for `na_object=None`,
  `na_object=np.nan`, and a string sentinel.
- Cover sentinel in the value operand, pattern/right operand, both operands,
  mixed sentinel and normal text, empty arrays, readonly arrays, and failure
  paths.
- Benchmark only after correctness is settled; sentinel handling should not
  slow the default-`StringDType()` path.

Acceptance bar for this tranche:

- Exact NumPy behavior for each supported `na_object` variant and operation.
- No change in behavior or performance shape for default `StringDType()`.
- No sentinel approximation, especially no treating every `rc == 1` load as
  the same operation-level result.
- No mutation of inputs.
- No allocator leaks or deadlocks.

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
   - `startswith`, `endswith` after a shared codepoint span-slicing primitive;
   - `find`, `rfind`, `count`, `index`, `rindex` after substring search
     semantics and codepoint result offsets are exact;
   - predicates;
   - ordering comparisons.
6. Scalar and mixed inputs:
   - scalar broadcast for the existing StringDType operation families;
   - direct 0-D StringDType behavior;
   - scalar-only return types after semantics are proven.
7. Missing sentinels:
   - exact `na_object` propagation per operation;
   - preserve the default-`StringDType()` fast path;
   - support only after operation-specific NumPy behavior is fully mapped.
8. Layout and dimensionality:
   - non-contiguous arrays through strides;
   - multidimensional arrays and NumPy-compatible broadcasting;
   - keep the one-dimensional contiguous fast path intact.
9. Full catalog:
   - transformation methods such as replace/case conversion/padding;
   - exact output allocation, dtype sizing, and error behavior;
   - only after read-only information methods and broadcasting are stable.

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
- In mixed Python `str` comparisons, validate the scalar once and hoist its
  trimmed codepoint length plus UTF-8 byte size once per operation. Short
  scalars stay on the Unicode/codepoint bridge; longer scalars use a UTF-8
  span outside the element loop.
- Reject `StringDType(na_object=...)` arrays until Tranche 9 implements
  sentinel propagation exactly.
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
- Whether Python string scalars should use a dedicated UTF-8 scalar bridge, a
  small temporary StringDType pack path, or a Numba-provided unicode access
  helper.
- Whether substring search should keep a reusable byte-to-codepoint mapping per
  element, or whether repeated linear scans are cleaner until broader operation
  coverage justifies the extra surface.
