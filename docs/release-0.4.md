# charex 0.4.0

`charex` 0.4.0 is the pre-N-D checkpoint for the Numba 0.65.1 compatibility
work. It updates the package for the NumPy 1.x/2.x support window and lands the
current scalar, 0-D, and 1-D string operation surface before the larger
N-D/broadcasting work begins.

## Compatibility

- Python `>=3.10,<3.15`
- Numba `>=0.65.1,<0.66`
- NumPy supported/tested window: `>=1.22,<1.27` or `>=2.0,<2.5`
- llvmlite `0.47.x`

`np.strings` support is conditional on NumPy 2.x.

## Highlights

- Adds NumPy 2.x `np.strings` overloads for the read-only catalog.
- Adds NumPy 2.x `StringDType` support for scalar, 0-D, and 1-D inputs.
- Supports default `StringDType()` and `StringDType(na_object=...)` variants.
- Supports StringDType comparisons, occurrence/search, and information
  predicates for the same read-only method families as fixed-width strings.
- Supports fixed-width `S`/`U` scalar, 0-D, and 1-D inputs, including strided,
  reversed, zero-stride, read-only, and empty views.
- Preserves separate `np.char` and `np.strings` semantics, including the
  trailing whitespace/NUL behavior difference.
- Adds packaging metadata and the native `charex._stringdtype` helper required
  for StringDType access.

## Supported Read-Only Catalog

- comparisons: `equal`, `not_equal`, `greater`, `greater_equal`, `less`,
  `less_equal`;
- occurrence/search: `count`, `startswith`, `endswith`, `find`, `rfind`,
  `index`, `rindex`;
- information/predicates: `str_len`, `isalpha`, `isalnum`, `isdigit`,
  `isdecimal`, `isnumeric`, `isspace`, `islower`, `isupper`, `istitle`;
- `np.char.compare_chararrays` for fixed-width `S`/`U`.

## Not In Scope

- N-D arrays and general broadcasting.
- Transformation/output-producing operations such as replace, case conversion,
  strip, pad, join, split, encode, and decode.
- Object array bridges.
- Max-performance experimental kernels that have not been distilled.

## Next Milestone

`0.5.0` is reserved for the shape/layout release: fixed-width and StringDType
N-D same-shape support, general broadcasting, and the corresponding audit and
benchmark refresh.
