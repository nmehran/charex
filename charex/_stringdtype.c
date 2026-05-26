#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

#define CHAREX_NUMPY_2_0_API_VERSION 0x00000012
#define CHAREX_STRINGDTYPE_ACQUIRE_SLOT 316

#if NPY_API_VERSION >= CHAREX_NUMPY_2_0_API_VERSION
#define CHAREX_HAS_STRINGDTYPE_API 1
#else
#define CHAREX_HAS_STRINGDTYPE_API 0
#endif

void *
charex_stringdtype_acquire_allocator(PyObject *array)
{
#if CHAREX_HAS_STRINGDTYPE_API
    PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *)array);
    typedef void *(*acquire_allocator_func)(const void *);
    acquire_allocator_func acquire = (acquire_allocator_func)
        PyArray_API[CHAREX_STRINGDTYPE_ACQUIRE_SLOT];
    return acquire((const void *)descr);
#else
    (void)array;
    return NULL;
#endif
}


static PyObject *
has_stringdtype_api(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
#if CHAREX_HAS_STRINGDTYPE_API
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}


static PyMethodDef methods[] = {
    {
        "has_stringdtype_api",
        has_stringdtype_api,
        METH_NOARGS,
        "Return whether this helper was built with NumPy StringDType API.",
    },
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_stringdtype",
    NULL,
    -1,
    methods,
};


PyMODINIT_FUNC
PyInit__stringdtype(void)
{
    import_array();
    return PyModule_Create(&module);
}
