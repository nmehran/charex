# charex benchmark matrix

- Python: `3.12.8`
- NumPy: `2.4.6`
- Numba: `0.65.1`
- llvmlite: `0.47.0`
- Size: `250000`
- Repeat: `7`

Timings are medians from interleaved charex/NumPy calls with `3` calls per timed sample.

This matrix includes fixed-width `np.char` inputs and NumPy 2.x `StringDType` inputs through `np.strings`.

Regenerate from the repository root:

```bash
python -m pip install -e ".[bench]"
python charex/benchmarks/matrix.py --size 250000 --repeat 7
```
