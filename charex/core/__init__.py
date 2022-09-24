JIT_OPTIONS = dict(
    boundscheck=False,
    cache=True,
    forceinline=True,
    no_cpython_wrapper=True,
    nogil=True,
)

OPTIONS = dict(
    jit_options=JIT_OPTIONS,
    prefer_literal=True,
    strict=False,
)
