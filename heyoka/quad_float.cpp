// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <new>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include <mp++/real128.hpp>

namespace
{

// Small RAII helper that will decref a Python object
// on destruction.
struct auto_decref {
    PyObject *m_ptr;

    explicit auto_decref(PyObject *ptr) : m_ptr(ptr) {}
    auto_decref(const auto_decref &) = delete;
    auto_decref(auto_decref &&) = delete;
    auto_decref &operator=(const auto_decref &) = delete;
    auto_decref &operator=(auto_decref &&) = delete;
    ~auto_decref()
    {
        Py_XDECREF(m_ptr);
    }
};

// The Python type that will represent mppp::real128.
struct py_quad_float {
    PyObject_HEAD alignas(mppp::real128) unsigned char m_storage[sizeof(mppp::real128)];
};

// Initialise the Python type corresponding to py_quad_float.
// NOTE: more type properties will be filled in when initing the module.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyTypeObject py_quad_float_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

// Methods for the number protocol.
PyNumberMethods py_quad_float_as_number = {};

// Initialise the module object.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyModuleDef quad_float{PyModuleDef_HEAD_INIT};

// Check if the input object is a quad float.
bool py_quad_float_check(PyObject *ob)
{
    return PyObject_IsInstance(ob, reinterpret_cast<PyObject *>(&py_quad_float_type)) != 0;
}

// Create a quad float containing an mppp::real128
// constructed from the input set of arguments.
template <typename... Args>
PyObject *py_quad_float_from_real128(Args &&...args)
{
    // Acquire the storage for a py_quad_float.
    auto *p = reinterpret_cast<py_quad_float *>(py_quad_float_type.tp_alloc(&py_quad_float_type, 0));

    // Setup its internal data.
    if (p != nullptr) {
        ::new (p->m_storage) mppp::real128(std::forward<Args>(args)...);
    }

    return reinterpret_cast<PyObject *>(p);
}

// Helpers to reach the mppp::real128 stored inside a py_quad_float.
mppp::real128 *get_val(py_quad_float *self)
{
    return std::launder(reinterpret_cast<mppp::real128 *>(self->m_storage));
}

mppp::real128 *get_val(PyObject *self)
{
    assert(py_quad_float_check(self));

    return get_val(reinterpret_cast<py_quad_float *>(self));
}

// __new__() implementation.
PyObject *py_quad_float_new([[maybe_unused]] PyTypeObject *type, PyObject *, PyObject *)
{
    assert(type == &py_quad_float_type);

    return py_quad_float_from_real128();
}

// Helper to convert a Python integer to real128.
mppp::real128 py_int_to_real128(PyObject *arg)
{
    assert(PyObject_IsInstance(arg, (PyObject *)&PyLong_Type));

    // Try to see if nptr fits in a long long.
    int overflow = 0;
    const auto candidate = PyLong_AsLongLongAndOverflow(arg, &overflow);

    if (overflow == 0) {
        return candidate;
    }

    // Need to construct a real128 from the limb array.
    auto *nptr = reinterpret_cast<PyLongObject *>(arg);

    // Get the signed size of nptr.
    const auto ob_size = nptr->ob_base.ob_size;
    assert(ob_size != 0);

    // Get the limbs array.
    const auto *ob_digit = nptr->ob_digit;

    // Is it negative?
    const auto neg = ob_size < 0;

    // Compute the unsigned size.
    using size_type = std::make_unsigned_t<std::remove_const_t<decltype(ob_size)>>;
    auto abs_ob_size = neg ? -static_cast<size_type>(ob_size) : static_cast<size_type>(ob_size);

    // Init the retval with the first (most significant) digit. The Python integer is nonzero, so this is safe.
    mppp::real128 retval = ob_digit[--abs_ob_size];

    // Init the number of digits consumed so far.
    auto ncdigits = PyLong_SHIFT;

    while (ncdigits < std::numeric_limits<mppp::real128>::digits && abs_ob_size != 0u) {
        retval = scalbn(retval, PyLong_SHIFT);
        retval += ob_digit[--abs_ob_size];
    }

    if (abs_ob_size != 0u) {
        // TODO finish, overflow check - how? std::optional perhaps?
    }

    if (neg) {
        return -retval;
    } else {
        return retval;
    }
}

// __init__() implementation.
int py_quad_float_init(py_quad_float *self, PyObject *args, PyObject *)
{
    PyObject *arg = nullptr;

    if (PyArg_ParseTuple(args, "|O", &arg) == 0) {
        return -1;
    }

    if (arg == nullptr) {
        return 0;
    }

    if (PyFloat_Check(arg)) {
        auto fp_arg = PyFloat_AsDouble(arg);

        if (PyErr_Occurred() == nullptr) {
            *get_val(self) = fp_arg;
        } else {
            return -1;
        }
    } else if (PyLong_Check(arg)) {
        *get_val(self) = py_int_to_real128(arg);
    } else if (py_quad_float_check(arg)) {
        *get_val(self) = *get_val(arg);
    } else if (PyUnicode_Check(arg)) {
        const auto *str = PyUnicode_AsUTF8(arg);

        if (str == nullptr) {
            return -1;
        }

        // TODO handle other exceptions.
        try {
            *get_val(self) = mppp::real128(str);
        } catch (const std::invalid_argument &ia) {
            PyErr_SetString(PyExc_ValueError, ia.what());

            return -1;
        }
    } else {
        // TODO raise error if construction argument type is not supported.
    }

    return 0;
}

// Deallocation.
void py_quad_float_dealloc(py_quad_float *self)
{
    // Invoke the destructor.
    get_val(self)->~real128();

    // Free the memory.
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// __repr__().
// TODO exception throwing?
// TODO repr vs str: repr should alway print all significant digits,
// what should str do? Use the G format to autodetect best representation,
// always using 34 digits?
PyObject *py_quad_repr(PyObject *self)
{
    // TODO handle potential runtime_error (return null on failure, test it).
    return PyUnicode_FromString(get_val(self)->to_string().c_str());
}

// Generic implementation of binary operations.
template <typename F>
PyObject *py_quad_float_binop(PyObject *a, PyObject *b, const F &op)
{
    if (py_quad_float_check(a) && py_quad_float_check(b)) {
        const auto *x = get_val(a);
        const auto *y = get_val(b);

        return py_quad_float_from_real128(op(*x, *y));
    } else {
        Py_RETURN_NOTIMPLEMENTED;
    }
}

// NumPy array function.
PyArray_ArrFuncs npy_py_quad_float_arr_funcs = {};

// NumPy type descriptor.
PyArray_Descr npy_py_quad_float_descr = {PyObject_HEAD_INIT(0) & py_quad_float_type};

// Array getitem.
PyObject *npy_py_quad_float_getitem(void *data, void *)
{
    mppp::real128 q;
    std::memcpy(&q, data, sizeof(q));
    return py_quad_float_from_real128(q);
}

// Array setitem.
int npy_py_quad_float_setitem(PyObject *item, void *data, void *)
{
    if (py_quad_float_check(item)) {
        const auto *q = get_val(item);
        std::memcpy(data, q, sizeof(mppp::real128));
        return 0;
    } else {
        PyErr_Format(PyExc_TypeError, "%s: expected real128, got %s", __func__, item->ob_type->tp_name);
        return -1;
    }
}

// Byteswap a real128 in place.
void byteswap(mppp::real128 *x)
{
    static_assert(sizeof(mppp::real128) == sizeof(__float128));

    const auto bs = sizeof(mppp::real128);

    auto *cptr = reinterpret_cast<char *>(x);
    for (std::size_t i = 0; i < bs / 2u; ++i) {
        const auto j = bs - i - 1u;
        const auto t = cptr[i];
        cptr[i] = cptr[j];
        cptr[j] = t;
    }
}

// Copyswap primitive.
void npy_py_quad_float_copyswap(void *dst, void *src, int swap, void *)
{
    assert(dst != nullptr);

    auto *r = static_cast<mppp::real128 *>(dst);

    if (src != nullptr) {
        // NOTE: not sure if src and dst can overlap here,
        // use memmove() just in case.
        std::memmove(r, src, sizeof(mppp::real128));
    }

    if (swap != 0) {
        byteswap(r);
    }
}

// Nonzero primitive.
npy_bool npy_py_quad_float_nonzero(void *data, void *)
{
    mppp::real128 q;
    std::memcpy(&q, data, sizeof(mppp::real128));

    return q != 0 ? NPY_TRUE : NPY_FALSE;
}

// Generic NumPy binary operation.
template <typename F>
void py_quad_float_ufunc_binary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *, const F &f)
{
    npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
    char *i0 = args[0], *i1 = args[1], *o = args[2];

    for (npy_intp k = 0; k < n; ++k) {
        const auto &x = *reinterpret_cast<mppp::real128 *>(i0);
        const auto &y = *reinterpret_cast<mppp::real128 *>(i1);
        *reinterpret_cast<mppp::real128 *>(o) = f(x, y);
        i0 += is0;
        i1 += is1;
        o += os;
    }
}

// NOTE: this is an integer used to represnt
// real128 *after* it has been registered in the
// NumPy dtype system. This is needed to expose
// ufuncs and it will be set up during module
// initialisation.
int npy_registered_py_quad_float = 0;

template <typename... Types>
void register_ufunc(PyObject *numpy, const char *name, PyUFuncGenericFunction func, Types... types_)
{
    auto *ufunc = reinterpret_cast<PyUFuncObject *>(PyObject_GetAttrString(numpy, name));
    assert(ufunc != nullptr);

    int types[] = {types_...};
    assert(sizeof(types) / sizeof(int) == ufunc->nargs);

    [[maybe_unused]] const auto ret = PyUFunc_RegisterLoopForType(ufunc, npy_registered_py_quad_float, func, types, 0);

    assert(ret >= 0);
}

} // namespace

PyMODINIT_FUNC PyInit_quad_float(void)
{
    import_array();
    if (PyErr_Occurred() != nullptr) {
        return nullptr;
    }

    import_umath();
    if (PyErr_Occurred() != nullptr) {
        return nullptr;
    }

    // Fill out the entries of py_quad_float_type.
    py_quad_float_type.tp_base = &PyGenericArrType_Type;
    py_quad_float_type.tp_name = "quad_float.real128";
    py_quad_float_type.tp_basicsize = sizeof(py_quad_float);
    py_quad_float_type.tp_itemsize = 0;
    py_quad_float_type.tp_flags = Py_TPFLAGS_DEFAULT;
    py_quad_float_type.tp_doc = PyDoc_STR("");
    py_quad_float_type.tp_new = py_quad_float_new;
    py_quad_float_type.tp_init = (initproc)py_quad_float_init;
    py_quad_float_type.tp_dealloc = (destructor)py_quad_float_dealloc;
    py_quad_float_type.tp_repr = py_quad_repr;
    py_quad_float_type.tp_as_number = &py_quad_float_as_number;

    // Fill out the functions for the number protocol.
    py_quad_float_as_number.nb_add
        = [](PyObject *a, PyObject *b) { return py_quad_float_binop(a, b, [](auto x, auto y) { return x + y; }); };
    py_quad_float_as_number.nb_subtract
        = [](PyObject *a, PyObject *b) { return py_quad_float_binop(a, b, [](auto x, auto y) { return x - y; }); };
    py_quad_float_as_number.nb_multiply
        = [](PyObject *a, PyObject *b) { return py_quad_float_binop(a, b, [](auto x, auto y) { return x * y; }); };
    py_quad_float_as_number.nb_true_divide
        = [](PyObject *a, PyObject *b) { return py_quad_float_binop(a, b, [](auto x, auto y) { return x / y; }); };

    // Finalize py_quad_float_type.
    if (PyType_Ready(&py_quad_float_type) < 0) {
        return nullptr;
    }

    // Fill out the NumPy descriptor.
    npy_py_quad_float_descr.kind = 'f';
    npy_py_quad_float_descr.type = 'q';
    npy_py_quad_float_descr.byteorder = '=';
    npy_py_quad_float_descr.flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
    npy_py_quad_float_descr.elsize = sizeof(mppp::real128);
    npy_py_quad_float_descr.alignment = alignof(mppp::real128);
    npy_py_quad_float_descr.f = &npy_py_quad_float_arr_funcs;

    // Setup the basic NumPy array functions.
    // NOTE: not 100% sure PyArray_InitArrFuncs is needed, as npy_py_quad_float_arr_funcs
    // is already cleared out on creation.
    PyArray_InitArrFuncs(&npy_py_quad_float_arr_funcs);
    npy_py_quad_float_arr_funcs.getitem = npy_py_quad_float_getitem;
    npy_py_quad_float_arr_funcs.setitem = npy_py_quad_float_setitem;
    npy_py_quad_float_arr_funcs.copyswap = npy_py_quad_float_copyswap;
    npy_py_quad_float_arr_funcs.nonzero = npy_py_quad_float_nonzero;

    Py_TYPE(&npy_py_quad_float_descr) = &PyArrayDescr_Type;
    npy_registered_py_quad_float = PyArray_RegisterDataType(&npy_py_quad_float_descr);
    if (npy_registered_py_quad_float < 0) {
        return nullptr;
    }

    // Support dtype(real128) syntax.
    if (PyDict_SetItemString(py_quad_float_type.tp_dict, "dtype", (PyObject *)&npy_py_quad_float_descr) < 0) {
        return nullptr;
    }

    // if (register_cast_functions(npy_registered_quadnum) < 0) {
    //     return;
    // }

    // NOTE: need access to the numpy module to register ufuncs.
    auto numpy_str = auto_decref(PyUnicode_FromString("numpy"));
    if (numpy_str.m_ptr == nullptr) {
        return nullptr;
    }
    auto numpy = auto_decref(PyImport_Import(numpy_str.m_ptr));
    if (numpy.m_ptr == nullptr) {
        return nullptr;
    }

    register_ufunc(
        numpy.m_ptr, "add",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            py_quad_float_ufunc_binary(args, dimensions, steps, data, [](auto x, auto y) { return x + y; });
        },
        npy_registered_py_quad_float, npy_registered_py_quad_float, npy_registered_py_quad_float);
    register_ufunc(
        numpy.m_ptr, "subtract",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            py_quad_float_ufunc_binary(args, dimensions, steps, data, [](auto x, auto y) { return x - y; });
        },
        npy_registered_py_quad_float, npy_registered_py_quad_float, npy_registered_py_quad_float);
    register_ufunc(
        numpy.m_ptr, "multiply",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            py_quad_float_ufunc_binary(args, dimensions, steps, data, [](auto x, auto y) { return x * y; });
        },
        npy_registered_py_quad_float, npy_registered_py_quad_float, npy_registered_py_quad_float);
    register_ufunc(
        numpy.m_ptr, "divide",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            py_quad_float_ufunc_binary(args, dimensions, steps, data, [](auto x, auto y) { return x / y; });
        },
        npy_registered_py_quad_float, npy_registered_py_quad_float, npy_registered_py_quad_float);

    // Setup and create the module.
    quad_float.m_name = "quad_float";
    quad_float.m_doc = "";
    quad_float.m_size = -1;
    auto *m = PyModule_Create(&quad_float);
    if (m == nullptr) {
        return nullptr;
    }

    // Add py_quad_float_type to the module.
    Py_INCREF(&py_quad_float_type);
    if (PyModule_AddObject(m, "real128", (PyObject *)&py_quad_float_type) < 0) {
        Py_DECREF(&py_quad_float_type);
        Py_DECREF(m);
        return nullptr;
    }

    return m;
}
