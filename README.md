# charex

### String Array Extensions for Numba

Enhance Numba with NumPy's string processing features by importing charex:

```python
import charex
```

### Comparison operations:
##### https://numpy.org/doc/stable/reference/routines.char.html#comparison

- `char.equal`
- `char.not_equal`
- `char.greater_equal`
- `char.less_equal`
- `char.greater`
- `char.less`
- `char.compare_chararrays`

### Occurrence and Property information:
##### https://numpy.org/doc/stable/reference/routines.char.html#string-information

- `char.count`
- `char.endswith`
- `char.startswith`
- `char.find`
- `char.rfind`
- `char.index`
- `char.rindex`
- `char.str_len`
- `char.isalpha`
- `char.isalnum`
- `char.isspace`
- `char.isdecimal`
- `char.isdigit`
- `char.isnumeric`
- `char.istitle`
- `char.isupper`
- `char.islower`

#### Includes support for UTF-32 strings and ASCII bytes on contiguous arrays of 1-dimension and scalars.

## Benchmarks

Despite a minor initial overhead from Numba's LLVM initialization, `charex` offsets this with increased data scale, outperforming NumPy in handling occurrence and property information.

### Comparison Operators
![comparison-operators-bytes.png](charex%2Fbenchmarks%2Fnumba-v-0.59%2Fcomparison-operators-bytes.png)
![comparison-operators-strings.png](charex%2Fbenchmarks%2Fnumba-v-0.59%2Fcomparison-operators-strings.png)

### Occurrence Information
![char-occurrence-bytes.png](charex%2Fbenchmarks%2Fnumba-v-0.59%2Fchar-occurrence-bytes.png)
![char-occurrence-strings.png](charex%2Fbenchmarks%2Fnumba-v-0.59%2Fchar-occurrence-strings.png)

### Property Information
![char-properties-bytes.png](charex%2Fbenchmarks%2Fnumba-v-0.59%2Fchar-properties-bytes.png)
![char-properties-strings.png](charex%2Fbenchmarks%2Fnumba-v-0.59%2Fchar-properties-strings.png)
![char-numerics-strings.png](charex%2Fbenchmarks%2Fnumba-v-0.59%2Fchar-numerics-strings.png)

The benchmarks are generated during testing using `charex/tests/test_comparison.py` and `charex/tests/test_string_information.py`. 

Last tested 2024-02-23: Numba 0.59.0, NumPy 1.26.3