# charex benchmark matrix

- Python: `3.12.8`
- NumPy: `2.4.6`
- Numba: `0.65.1`
- llvmlite: `0.47.0`
- Size: `50000`
- Repeat: `5`

Regenerate from the repository root:

```bash
python -m pip install -e ".[bench]"
python charex/benchmarks/matrix.py --size 50000 --repeat 5
```
