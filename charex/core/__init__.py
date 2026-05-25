JIT_OPTIONS = dict(
    boundscheck=False,
    cache=True,
    forceinline=True,
    no_cpython_wrapper=True,
    nogil=True,
)

OVERLOAD_JIT_OPTIONS = dict(JIT_OPTIONS, cache=False)

OPTIONS = dict(
    jit_options=OVERLOAD_JIT_OPTIONS,
    prefer_literal=True,
    strict=False,
)
