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

## Initial Audit

Environment:

- Python 3.12.8
- NumPy 2.4.6
- Numba 0.65.1

Full matrix:

- rows: 1702
- matching rows: 1056
- mismatches: 646
- NumPy accepts but charex currently rejects: 558
- NumPy raises but charex currently returns a value: 24
- both raise but with different error type/status: 64
- successful value mismatches: 0

## Findings

The dominant gap is fixed-width array shape/layout support. NumPy accepts
positive-stride, negative-stride, zero-stride, contiguous 2-D, and broadcasted
2-D fixed-width `S`/`U` arrays for both `np.char` and `np.strings`. charex
currently rejects these paths at typing because fixed-width registration is
still scalar or one-dimensional C-contiguous only.

`StringDType` is in better shape: scalar, 0-D, 1-D contiguous, and 1-D strided
paths are covered. Remaining `StringDType` shape gaps are N-D same-shape and
N-D broadcasting.

There is also an existing fixed-width shape-mismatch correctness bug in
occurrence methods. For 1-D mismatched fixed-width arrays, NumPy raises a
broadcast `ValueError`, but charex currently returns a result for:

- `count`
- `find`
- `rfind`
- `index` / `rindex` through `np.strings`
- `startswith`
- `endswith`

Byte-only Unicode predicates need a separate error-shape decision. NumPy raises
`UFuncTypeError` for bytes with `isdecimal` and `isnumeric`; charex raises a
Numba typing error. This is lower priority than the accepted-input gaps but
should remain visible in the parity matrix.

## Next Iteration

1. Fix fixed-width 1-D shape mismatch for occurrence methods.
   This is a correctness bug and should be smaller than general broadcasting.

2. Add fixed-width 1-D strided support for `np.char` and `np.strings`.
   Preserve the current C-contiguous fast path and add a stride-aware fixed-width
   access path at the registration boundary.

3. Add fixed-width N-D same-shape support.
   Do this before general broadcasting so the index mapping stays reviewable.

4. Add broadcast-compatible shape support for fixed-width and `StringDType`.
   General broadcasting should be a separate path and should not slow the
   scalar/0-D/1-D fast paths.

5. Re-run the full audit matrix after each tranche and use the CSV diff to
   decide the next smallest correctness slice.
