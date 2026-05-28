# charex

String array extensions for Numba.

Importing `charex` registers Numba overloads for NumPy's `np.char` and
`np.strings` string APIs:

```python
import charex
```

The implementation aims to match NumPy behavior exactly while balancing strong
fixed-width string performance with a compact code surface. Fast paths are kept
when they are broadly useful and still leave the implementation simple enough to
review.

## Compatibility

`charex` targets Numba 0.65.1 and the NumPy ranges tested by that Numba
release:

- Python `>=3.10,<3.15`
- Numba `>=0.65.1,<0.66`
- NumPy `>=1.22,<1.27` or `>=2.0,<2.5`
- llvmlite `0.47.x`

`np.strings` overloads are available on NumPy 2.x. NumPy 1.x does not expose
`np.strings`, so that registration is conditional.

## Supported Operations

Supported for `np.char` on all supported NumPy versions, and for `np.strings`
on NumPy 2.x, except `compare_chararrays`, which is `np.char` only:

- `equal`
- `not_equal`
- `greater_equal`
- `less_equal`
- `greater`
- `less`
- `count`
- `endswith`
- `startswith`
- `find`
- `rfind`
- `index`
- `rindex`
- `str_len`
- `isalpha`
- `isalnum`
- `isspace`
- `isdecimal`
- `isdigit`
- `isnumeric`
- `istitle`
- `isupper`
- `islower`

Additional `np.char` operation:

- `compare_chararrays`

Fixed-width `S`/`U` inputs may be scalars, 0-D arrays, or 1-D arrays,
including contiguous, read-only, positive-stride, negative-stride,
zero-stride, and empty views.

On NumPy 2.x, `np.strings` also supports `StringDType` inputs for the same
read-only operation catalog. The supported `StringDType` shape scope is scalar,
0-D, and 1-D arrays, including strided views. Default `StringDType()` and
`StringDType(na_object=...)` variants are supported with NumPy-matching
operation-specific null behavior.

N-D arrays, general broadcasting, and transformation/output-producing string
operations are not part of this release.

See [docs/release-0.4.md](docs/release-0.4.md) for the release scope.

## Performance Matrix

Current Numba 0.65.1 `np.char` benchmark artifacts are in
[docs/benchmarks/numba-v-0.65.1](docs/benchmarks/numba-v-0.65.1/).

Regenerate the matrix from the repository root:

```bash
python -m pip install -e ".[bench]"
python charex/benchmarks/matrix.py --size 50000 --repeat 5
```

### Comparison Operators

![comparison-operators-bytes.png](docs/benchmarks/numba-v-0.65.1/comparison-operators-bytes.png)
![comparison-operators-strings.png](docs/benchmarks/numba-v-0.65.1/comparison-operators-strings.png)

### Occurrence Information

![char-occurrence-bytes.png](docs/benchmarks/numba-v-0.65.1/char-occurrence-bytes.png)
![char-occurrence-strings.png](docs/benchmarks/numba-v-0.65.1/char-occurrence-strings.png)

### Property Information

![char-properties-bytes.png](docs/benchmarks/numba-v-0.65.1/char-properties-bytes.png)
![char-properties-strings.png](docs/benchmarks/numba-v-0.65.1/char-properties-strings.png)
![char-numerics-strings.png](docs/benchmarks/numba-v-0.65.1/char-numerics-strings.png)

The previous Numba 0.59 matrix is archived under
[charex/benchmarks/numba-v-0.59](charex/benchmarks/numba-v-0.59/).

## Development

Install test dependencies:

```bash
python -m pip install -e ".[test]"
```

Run tests:

```bash
pytest -q
```

Run the benchmark smoke test:

```bash
python charex/benchmarks/benchmark.py --size 50000 --repeat 5
```

Install benchmark plotting dependencies and write CSV/PNG output:

```bash
python -m pip install -e ".[bench]"
python charex/benchmarks/benchmark.py --size 50000 --repeat 5 --plot
```

Last locally tested 2026-05-25 on Python 3.12.8 with:

- Numba 0.65.1, llvmlite 0.47.0, NumPy 1.26.4
- Numba 0.65.1, llvmlite 0.47.0, NumPy 2.4.6
