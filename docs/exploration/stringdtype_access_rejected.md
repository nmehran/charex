# Rejected StringDType Access Experiments

This file keeps the rejected Tranche 1 ideas out of the active benchmark
surface while preserving the rationale. The historical code is still available
in the exploratory commits before the candidate cleanup, especially:

- `9695434 perf(stringdtype): document allocator access strategies`
- `cdabdac perf(stringdtype): deepen access layer exploration`
- `317379a perf(stringdtype): explore peeled and backward length kernels`
- `1edaee5 perf(stringdtype): explore SWAR length counting`
- `988ec69 perf(stringdtype): probe final length refinements`

## Rejected Access Strategies

- Python callback allocator acquisition:
  Correct, but too slow and crosses back through Python during native work.
- Default descriptor constant:
  Fast on some short cases, but correctness-invalid for dtype variants and
  missing-sentinel semantics.
- Per-element allocator acquisition:
  Simple, but much slower than acquiring once per array operation.
- Whole-loop C helpers:
  Not attractive. They hide the hot loop from Numba and were slower than
  Numba-generated loops in the tested shapes.
- Duplicate C-helper benchmark variants:
  Removed from the active harness after `charex._stringdtype` became mandatory
  for StringDType support.

## Rejected Length Kernels

- One-pass effective count:
  Good on some short distributions, but poor on long payloads. The final
  runtime shape prefers trailing trim plus a separate code-point count.
- One-pass effective size:
  Same broad issue as one-pass effective count, with no cleaner production
  contract.
- Unchecked one-pass:
  Exploration-only. It did not offer enough speedup to justify a less explicit
  shape.
- Hybrid16 and hybrid32:
  Empirical thresholds. They are not a clean universal policy.
- Backward word threshold32 and threshold64:
  Also threshold-fitted. Plain SWAR is the cleaner max-performance candidate.
- ASCII-skip SWAR:
  Distribution-sensitive. It helps some short mostly-ASCII cases but loses on
  long mixed/ASCII cases.
- Four-word unrolled SWAR:
  Rejected. It grew generated LLVM substantially and was slower or noisier in
  relevant distributions.
- Byte-count plus word-level trim:
  Good for long trailing-NUL payloads but not broad enough. The active surface
  keeps the stronger word-trim plus SWAR candidate instead.

## Still Active

- Current runtime:
  mandatory helper, operation-scope allocator, hoisted data pointer,
  all-zero word trim, byte-wise UTF-8 continuation count.
- Backward data:
  minimal byte-wise reference candidate.
- Backward word:
  plain SWAR continuation-count max-performance candidate.
- Word-trim word:
  stronger trailing-NUL special-case candidate.
- Peeled16 word-hybrid:
  short inline candidate combined with SWAR fallback.
