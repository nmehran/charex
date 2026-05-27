# String Array Shape Parity

This branch audits NumPy shape, stride, and broadcasting behavior before
changing the implementation. The target contract is exact NumPy behavior across
the string APIs charex overloads:

- `np.char` fixed-width `S` and `U`;
- `np.strings` fixed-width `S` and `U`;
- `np.strings` `StringDType`.

Regenerate the current audit matrix:

```bash
python docs/exploration/string_array_shape_audit.py --methods all
```

The script writes `docs/exploration/string_array_shape_audit.csv`.

## Current Audit

Environment:

- Python 3.12.8
- NumPy 2.4.6
- Numba 0.65.1

Full matrix:

- rows: 1702
- matching rows: 1344
- mismatches: 358
- NumPy accepts but charex currently rejects: 294
- NumPy raises but charex currently returns a value: 0
- both raise but with different error type/status: 64
- successful value mismatches: 0

## Findings

The dominant remaining gap is N-D shape support. NumPy accepts contiguous 2-D
and broadcasted 2-D fixed-width `S`/`U` arrays for both `np.char` and
`np.strings`; charex still rejects these paths at typing.

Fixed-width scalar, 0-D, 1-D contiguous, read-only, positive-stride,
negative-stride, zero-stride, and empty strided views are now covered for the
audited `np.char` and fixed-width `np.strings` operations.

`StringDType` is in better shape: scalar, 0-D, 1-D contiguous, and 1-D strided
paths are covered. Remaining `StringDType` shape gaps are N-D same-shape and
N-D broadcasting.

The first correctness fix on this branch closed an existing fixed-width
shape-mismatch bug in occurrence methods. For 1-D mismatched fixed-width
arrays, NumPy raises a broadcast `ValueError`; charex now raises instead of
returning a result.

Byte-only Unicode predicates need a separate error-shape decision. NumPy raises
`UFuncTypeError` for bytes with `isdecimal` and `isnumeric`; charex raises a
Numba typing error. This is lower priority than the accepted-input gaps but
should remain visible in the parity matrix.

## Next Iteration

1. Add fixed-width N-D same-shape support.
   Do this before general broadcasting so the index mapping stays reviewable.

2. Add broadcast-compatible shape support for fixed-width and `StringDType`.
   General broadcasting should be a separate path and should not slow the
   scalar/0-D/1-D fast paths.

3. Re-run the full audit matrix after each tranche and use the CSV diff to
   decide the next smallest correctness slice.
