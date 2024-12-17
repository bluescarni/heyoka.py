// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <pybind11/pybind11.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/dtype_api.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include <mp++/config.hpp>
#include <mp++/integer.hpp>
#include <mp++/real128.hpp>

#include "common_utils.hpp"

#endif

#include "custom_casters.hpp"
#include "dtypes.hpp"
#include "expose_real128.hpp"

#if defined(HEYOKA_HAVE_REAL)

#include "expose_real.hpp"

#endif

namespace heyoka_py
{

namespace py = pybind11;

#if defined(HEYOKA_HAVE_REAL128)

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-align"

#if !defined(__clang__)

#pragma GCC diagnostic ignored "-Wcast-function-type"

#endif

#endif

// NOTE: more type properties will be filled in when initing the module.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyTypeObject py_real128_type = {PyVarObject_HEAD_INIT(nullptr, 0)};

// NOTE: this is an integer used to represent the
// real128 type *after* it has been registered in the
// NumPy dtype system. This is needed to expose
// ufuncs and it will be set up during module
// initialisation.
// TODO still true?
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
int npy_registered_py_real128 = 0;

namespace detail
{

namespace
{

// Double check that malloc() aligns memory suitably
// for py_real128. See:
// https://en.cppreference.com/w/cpp/types/max_align_t
static_assert(alignof(py_real128) <= alignof(std::max_align_t));

// A few function object to implement basic mathematical
// operations in a generic fashion.
const auto identity_func = [](auto x) { return x; };

const auto negation_func = [](auto x) { return -x; };

const auto abs_func = [](auto x) {
    using std::abs;

    return abs(x);
};

const auto square_func = [](auto x) { return x * x; };

const auto floor_divide_func = [](auto x, auto y) {
    using std::floor;

    return floor(x / y);
};

const auto sqrt_func = [](auto x) {
    using std::sqrt;

    return sqrt(x);
};

const auto cbrt_func = [](auto x) {
    using std::cbrt;

    return cbrt(x);
};

const auto sin_func = [](auto x) {
    using std::sin;

    return sin(x);
};

const auto cos_func = [](auto x) {
    using std::cos;

    return cos(x);
};

const auto tan_func = [](auto x) {
    using std::tan;

    return tan(x);
};

const auto asin_func = [](auto x) {
    using std::asin;

    return asin(x);
};

const auto acos_func = [](auto x) {
    using std::acos;

    return acos(x);
};

const auto atan_func = [](auto x) {
    using std::atan;

    return atan(x);
};

const auto atan2_func = [](auto y, auto x) {
    using std::atan2;

    return atan2(y, x);
};

const auto sinh_func = [](auto x) {
    using std::sinh;

    return sinh(x);
};

const auto cosh_func = [](auto x) {
    using std::cosh;

    return cosh(x);
};

const auto tanh_func = [](auto x) {
    using std::tanh;

    return tanh(x);
};

const auto asinh_func = [](auto x) {
    using std::asinh;

    return asinh(x);
};

const auto acosh_func = [](auto x) {
    using std::acosh;

    return acosh(x);
};

const auto atanh_func = [](auto x) {
    using std::atanh;

    return atanh(x);
};

// NOTE: this is 2*pi/360 computed in octuple precision.
const auto deg2rad_const = mppp::real128("0.01745329251994329576923690768488612713442871888541725456097191440171005");

const auto deg2rad_func = [](mppp::real128 x) { return deg2rad_const * x; };

// NOTE: this is 360/(2*pi) computed in octuple precision.
const auto rad2deg_constant = mppp::real128("57.295779513082320876798154814105170332405472466564321549160243861202985");

const auto rad2deg_func = [](mppp::real128 x) { return rad2deg_constant * x; };

const auto exp_func = [](auto x) {
    using std::exp;

    return exp(x);
};

#if defined(MPPP_QUADMATH_HAVE_EXP2Q)

const auto exp2_func = [](auto x) {
    using std::exp2;

    return exp2(x);
};

#endif

const auto expm1_func = [](auto x) {
    using std::expm1;

    return expm1(x);
};

const auto log_func = [](auto x) {
    using std::log;

    return log(x);
};

const auto log2_func = [](auto x) {
    using std::log2;

    return log2(x);
};

const auto log10_func = [](auto x) {
    using std::log10;

    return log10(x);
};

const auto log1p_func = [](auto x) {
    using std::log1p;

    return log1p(x);
};

const auto isfinite_func = [](auto x) {
    using std::isfinite;

    return isfinite(x);
};

const auto sign_func = [](auto x) {
    using std::isnan;

    if (isnan(x)) {
        return x;
    }

    return static_cast<decltype(x)>((0 < x) - (x < 0));
};

const auto isnan_func = [](auto x) {
    using std::isnan;

    return isnan(x);
};

const auto isinf_func = [](auto x) {
    using std::isinf;

    return isinf(x);
};

const auto pow_func = [](auto x, auto y) {
    using std::pow;

    return pow(x, y);
};

const auto max_func = [](auto a, auto b) { return std::max(a, b); };

const auto min_func = [](auto a, auto b) { return std::min(a, b); };

// Methods for the number protocol.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyNumberMethods py_real128_as_number = {};

// Create a py_real128 containing an mppp::real128
// constructed from the input set of arguments.
template <typename... Args>
PyObject *py_real128_from_args(Args &&...args)
{
    // Acquire the storage for a py_real128.
    void *pv = py_real128_type.tp_alloc(&py_real128_type, 0);
    if (pv == nullptr) {
        return nullptr;
    }

    // Construct the py_real128 instance.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto *p = ::new (pv) py_real128;

    if (with_pybind11_eh([&]() {
            // Setup its internal data.
            ::new (p->m_storage) mppp::real128(std::forward<Args>(args)...);
        })) {

        // Clean up.
        py_real128_type.tp_free(pv);

        return nullptr;
    }

    return reinterpret_cast<PyObject *>(p);
}

// __new__() implementation.
PyObject *py_real128_new([[maybe_unused]] PyTypeObject *type, PyObject *, PyObject *)
{
    assert(type == &py_real128_type);

    return py_real128_from_args();
}

// Helper to convert a Python integer to real128.
// NOTE: this currently uses a string conversion for
// large integers.
std::optional<mppp::real128> py_int_to_real128(PyObject *arg)
{
    assert(PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLong_Type)));

    // Try to see if arg fits in a long long.
    int overflow = 0;
    const auto candidate = PyLong_AsLongLongAndOverflow(arg, &overflow);

    // NOTE: PyLong_AsLongLongAndOverflow() can raise exceptions in principle.
    if (PyErr_Occurred() != nullptr) {
        return {};
    }

    if (overflow == 0) {
        return candidate;
    }

    // Go through a string conversion.
    std::optional<mppp::real128> retval;

    with_pybind11_eh([&]() {
        auto *str_rep = PyObject_Str(arg);
        if (str_rep == nullptr) {
            return;
        }
        assert(PyUnicode_Check(str_rep) != 0);

        // NOTE: str remains valid as long as str_rep is alive.
        const auto *str = PyUnicode_AsUTF8(str_rep);
        if (str == nullptr) {
            Py_DECREF(str_rep);
            return;
        }

        try {
            retval.emplace(str);
        } catch (...) {
            // Ensure proper cleanup of str_rep
            // before continuing.
            Py_DECREF(str_rep);
            throw;
        }

        Py_DECREF(str_rep);
    });

    return retval;
}

// __init__() implementation.
int py_real128_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *arg = nullptr;

    const char *kwlist[] = {"value", nullptr};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "|O", const_cast<char **>(&kwlist[0]), &arg) == 0) {
        return -1;
    }

    if (arg == nullptr) {
        return 0;
    }

    if (PyFloat_Check(arg)) {
        auto fp_arg = PyFloat_AsDouble(arg);

        if (PyErr_Occurred() == nullptr) {
            *get_real128_val(self) = fp_arg;
        } else {
            return -1;
        }
    } else if (PyLong_Check(arg)) {
        if (auto opt = py_int_to_real128(arg)) {
            *get_real128_val(self) = *opt;
        } else {
            return -1;
        }
    } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyFloat32ArrType_Type)) != 0) {
        *get_real128_val(self) = reinterpret_cast<PyFloat32ScalarObject *>(arg)->obval;
#if defined(MPPP_FLOAT128_WITH_LONG_DOUBLE)
    } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLongDoubleArrType_Type)) != 0) {
        *get_real128_val(self) = reinterpret_cast<PyLongDoubleScalarObject *>(arg)->obval;
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (py_real_check(arg)) {
        // NOTE: conversion of real to real128 cannot throw.
        *get_real128_val(self) = static_cast<mppp::real128>(*get_real_val(arg));
#endif
    } else if (py_real128_check(arg)) {
        *get_real128_val(self) = *get_real128_val(arg);
    } else if (PyUnicode_Check(arg)) {
        const auto *str = PyUnicode_AsUTF8(arg);

        if (str == nullptr) {
            return -1;
        }

        if (with_pybind11_eh([&]() { *get_real128_val(self) = mppp::real128(str); })) {
            return -1;
        }
    } else {
        PyErr_Format(PyExc_TypeError, "Cannot construct a real128 from an object of type \"%s\"",
                     Py_TYPE(arg)->tp_name);

        return -1;
    }

    return 0;
}

// Deallocation.
void py_real128_dealloc(PyObject *self)
{
    assert(py_real128_check(self));

    // Invoke the destructor.
    get_real128_val(self)->~real128();

    // Free the memory.
    Py_TYPE(self)->tp_free(self);
}

// Helper to construct a real128 from one of the
// supported Pythonic numerical types:
// - int,
// - float32,
// - float,
// - long double,
// - real.
// If the conversion is successful, returns {x, true}. If some error
// is encountered, returns {<empty>, false}. If the type of arg is not
// supported, returns {<empty>, true}.
std::pair<std::optional<mppp::real128>, bool> real128_from_ob(PyObject *arg)
{
    if (PyFloat_Check(arg)) {
        auto fp_arg = PyFloat_AsDouble(arg);

        if (PyErr_Occurred() == nullptr) {
            return {fp_arg, true};
        } else {
            return {{}, false};
        }
    } else if (PyLong_Check(arg)) {
        if (auto opt = py_int_to_real128(arg)) {
            return {*opt, true};
        } else {
            return {{}, false};
        }
    } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyFloat32ArrType_Type)) != 0) {
        return {reinterpret_cast<PyFloat32ScalarObject *>(arg)->obval, true};
#if defined(MPPP_FLOAT128_WITH_LONG_DOUBLE)
    } else if (PyObject_IsInstance(arg, reinterpret_cast<PyObject *>(&PyLongDoubleArrType_Type)) != 0) {
        return {reinterpret_cast<PyLongDoubleScalarObject *>(arg)->obval, true};
#endif
#if defined(HEYOKA_HAVE_REAL)
    } else if (py_real_check(arg)) {
        // NOTE: conversion of real to real128 cannot throw.
        return {static_cast<mppp::real128>(*get_real_val(arg)), true};
#endif
    } else {
        return {{}, true};
    }
}

PyObject *py_real128_copy(PyObject *self, [[maybe_unused]] PyObject *args)
{
    assert(args == nullptr);

    return py_real128_from_args(*get_real128_val(self));
}

PyObject *py_real128_deepcopy(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *memo_arg = nullptr;

    const char *kwlist[] = {"memo", nullptr};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    if (PyArg_ParseTupleAndKeywords(args, kwargs, "|O", const_cast<char **>(&kwlist[0]), &memo_arg) == 0) {
        return nullptr;
    }

    return py_real128_from_args(*get_real128_val(self));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyMethodDef py_real128_methods[]
    = {{"__copy__", py_real128_copy, METH_NOARGS, nullptr},
       {"__deepcopy__", reinterpret_cast<PyCFunction>(py_real128_deepcopy), METH_VARARGS | METH_KEYWORDS, nullptr},
       {nullptr}};

// __repr__().
PyObject *py_real128_repr(PyObject *self)
{
    PyObject *retval = nullptr;

    // NOTE: C++ exceptions here can be thrown only *before*
    // the invocation of PyUnicode_FromString(). Hence, in case of
    // exceptions, retval will remain nullptr.
    with_pybind11_eh([&]() { retval = PyUnicode_FromString(get_real128_val(self)->to_string().c_str()); });

    return retval;
}

// Generic implementation of unary operations.
template <typename F>
PyObject *py_real128_unop(PyObject *a, const F &op)
{
    if (py_real128_check(a)) {
        const auto *x = get_real128_val(a);
        return py_real128_from_args(op(*x));
    } else {
        Py_RETURN_NOTIMPLEMENTED;
    }
}

// Generic implementation of binary operations.
template <typename F>
PyObject *py_real128_binop(PyObject *a, PyObject *b, const F &op)
{
    const auto a_is_real128 = py_real128_check(a);
    const auto b_is_real128 = py_real128_check(b);

    if (a_is_real128 && b_is_real128) {
        // Both operands are real128.
        const auto *x = get_real128_val(a);
        const auto *y = get_real128_val(b);

        return py_real128_from_args(op(*x, *y));
    }

    if (a_is_real128) {
        // a is a real128, b is not.

#if defined(HEYOKA_HAVE_REAL)

        // NOTE: if one of the operands is a real,
        // the result will be a real too.
        if (py_real_check(b)) {
            PyObject *retval = nullptr;

            with_pybind11_eh([&]() { retval = pyreal_from_real(op(*get_real128_val(a), *get_real_val(b))); });

            return retval;
        }

#endif
        auto [r, flag] = real128_from_ob(b);

        if (r) {
            // The conversion was successful, do the op.
            return py_real128_from_args(op(*get_real128_val(a), *r));
        }

        if (flag) {
            // b's type is not supported.
            Py_RETURN_NOTIMPLEMENTED;
        }

        // Attempting to convert b to a real128 generated an error.
        return nullptr;
    }

    if (b_is_real128) {
        // The mirror of the previous case.

#if defined(HEYOKA_HAVE_REAL)

        if (py_real_check(a)) {
            PyObject *retval = nullptr;

            with_pybind11_eh([&]() { retval = pyreal_from_real(op(*get_real_val(a), *get_real128_val(b))); });

            return retval;
        }

#endif

        auto [r, flag] = real128_from_ob(a);

        if (r) {
            return py_real128_from_args(op(*r, *get_real128_val(b)));
        }

        if (flag) {
            Py_RETURN_NOTIMPLEMENTED;
        }

        return nullptr;
    }

    // Neither a nor b are real128.
    Py_RETURN_NOTIMPLEMENTED;
}

// Rich comparison operator.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *py_real128_rcmp(PyObject *a, PyObject *b, int op)
{
    auto impl = [a, b](const auto &func) -> PyObject * {
        const auto a_is_real128 = py_real128_check(a);
        const auto b_is_real128 = py_real128_check(b);

        if (a_is_real128 && b_is_real128) {
            // Both operands are real128.
            const auto *x = get_real128_val(a);
            const auto *y = get_real128_val(b);

            if (func(*x, *y)) {
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
        }

        if (a_is_real128) {
            // a is a real128, b is not.

#if defined(HEYOKA_HAVE_REAL)

            if (py_real_check(b)) {
                // NOTE: comparisons cannot throw.
                if (func(*get_real128_val(a), *get_real_val(b))) {
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }

#endif

            auto [r, flag] = real128_from_ob(b);

            if (r) {
                // The conversion was successful, do the op.
                if (func(*get_real128_val(a), *r)) {
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }

            if (flag) {
                // b's type is not supported.
                Py_RETURN_NOTIMPLEMENTED;
            }

            // Attempting to convert b to a real128 generated an error.
            return nullptr;
        }

        if (b_is_real128) {
            // The mirror of the previous case.

#if defined(HEYOKA_HAVE_REAL)

            if (py_real_check(a)) {
                if (func(*get_real_val(a), *get_real128_val(b))) {
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }

#endif

            auto [r, flag] = real128_from_ob(a);

            if (r) {
                if (func(*r, *get_real128_val(b))) {
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }

            if (flag) {
                Py_RETURN_NOTIMPLEMENTED;
            }

            return nullptr;
        }

        Py_RETURN_NOTIMPLEMENTED;
    };

    switch (op) {
        case Py_LT:
            return impl(std::less{});
        case Py_LE:
            return impl(std::less_equal{});
        case Py_EQ:
            return impl(std::equal_to{});
        case Py_NE:
            return impl(std::not_equal_to{});
        case Py_GT:
            return impl(std::greater{});
        default:
            assert(op == Py_GE);
            return impl(std::greater_equal{});
    }
}

// Numpy setup.

// The real128 dtype. It does not need to store any information,
// it just derives from PyArray_Descr.
struct real128_dtype_object {
    PyArray_Descr base;
};

// The Python type descriptor for the real128 dtype.
// NOTE: this is a complex type descriptor, containing a superclass
// which in turn contains the PyTypeObject. The setup of the PyTypeObject
// is accomplished by init_real128_dtype().
PyArray_DTypeMeta real128_dtype{};

// Byteswap a real128 in place.
void byteswap(mppp::real128 *x)
{
    static_assert(sizeof(mppp::real128) == sizeof(x->m_value));

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
void npy_real128_copyswap(void *dst, void *src, int swap, void *)
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

// Helper to create a new instance of the real128 dtype.
PyObject *real128_dtype_new_instance()
{
    // Attempt to create the dtype, invoking the base __new__().
    auto *ret = reinterpret_cast<real128_dtype_object *>(
        PyArrayDescr_Type.tp_new(reinterpret_cast<PyTypeObject *>(&real128_dtype), nullptr, nullptr));
    if (ret == nullptr) {
        return nullptr;
    }

    // Setup the dtype properties.
    ret->base.kind = 'f';
    ret->base.type = 'q';
    ret->base.byteorder = '=';
    // NOTE: NPY_NEEDS_PYAPI should be set if the "data-type requires the Python C-API during access":
    //
    // https://numpy.org/doc/stable/reference/c-api/types-and-structures.html#c.NPY_NEEDS_PYAPI
    //
    // However I do not understand what this means precisely. On the one hand, our get/set item implementations
    // need the Python C API, e.g., for type checking. Does that count as "access"? On the other hand,
    // the examples from numpy-user-dtypes almost never flip NPY_NEEDS_PYAPI on. It may be that "access"
    // in this context means only ufuncs/casting. In any case, let us leave this on for the time being
    // for safety.
    ret->base.flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
    // NOTE: these are the size/alignment requirements for the C type represented
    // by this dtype.
    ret->base.elsize = sizeof(mppp::real128);
    ret->base.alignment = alignof(mppp::real128);

    // NOTE: it seems like the only way to set up the copyswap primitive is
    // to go through the legacy array functions API. Note that, as mentioned in
    // the numpy-user-dtypes repo, these functions slots are the same for all dtype instances.
    PyDataType_GetArrFuncs(&ret->base)->copyswap = &npy_real128_copyswap;

    return reinterpret_cast<PyObject *>(ret);
}

// Implementation of __new__() for the real128 dtype.
PyObject *real128_dtype_new(PyTypeObject *, PyObject *args, PyObject *kwargs)
{
    // No arguments must have been provided in input.
    const char *kwlist[] = {nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", const_cast<char **>(&kwlist[0]))) {
        return nullptr;
    }

    // Create a new instance of the real128 dtype.
    return real128_dtype_new_instance();
}

// Helper to initialise the PyTypeObject in real128_dtype.
auto init_real128_dtype()
{
    // Basic properties: name, type, base, size.
    PyTypeObject ret{PyVarObject_HEAD_INIT(nullptr, 0)};
    ret.tp_name = "heyoka.core.real128_dtype";
    ret.tp_basicsize = sizeof(real128_dtype_object);
    ret.tp_base = &PyArrayDescr_Type;
    Py_SET_TYPE(&ret, &PyArrayDTypeMeta_Type);

    // __new__().
    ret.tp_new = real128_dtype_new;
    // __repr__() and __str__().
    auto repr_func = [](PyObject *) -> PyObject * { return PyUnicode_FromString("real128_dtype"); };
    ret.tp_repr = repr_func;
    ret.tp_str = repr_func;

    return ret;
}

// Returns the “canonical” representation for a descriptor instance.
PyArray_Descr *npy_real128_ensure_canonical(PyArray_Descr *self)
{
    Py_INCREF(self);
    return self;
}

// Common instance.
PyArray_Descr *npy_real128_common_instance(PyArray_Descr *dtype1, PyArray_Descr *)
{
    Py_INCREF(dtype1);
    return dtype1;
}

// Given two input DTypes, determines the appropriate “common” DType that can store values for both types.
PyArray_DTypeMeta *npy_real128_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // Promote integer and floating-point types to real128.
    if (other->type_num >= 0 && (PyTypeNum_ISINTEGER(other->type_num) || PyTypeNum_ISFLOAT(other->type_num))) {
        Py_INCREF(cls);
        return cls;
    }

    // Don't promote complex types
    if (PyTypeNum_ISCOMPLEX(other->type_num)) {
        Py_INCREF(Py_NotImplemented);
        return reinterpret_cast<PyArray_DTypeMeta *>(Py_NotImplemented);
    }

    Py_INCREF(Py_NotImplemented);
    return reinterpret_cast<PyArray_DTypeMeta *>(Py_NotImplemented);
}

// Used during DType inference to find the correct DType for a given PyObject.
PyArray_Descr *npy_real128_discover_descriptor_from_pyobject(PyArray_DTypeMeta *, PyObject *obj)
{
    if (Py_TYPE(obj) != &py_real128_type) {
        PyErr_SetString(PyExc_TypeError, "Can only store real128 in a real128_dtype array.");
        return nullptr;
    }

    return reinterpret_cast<PyArray_Descr *>(real128_dtype_new_instance());
}

// Returns the default descriptor instance for the DType.
PyArray_Descr *npy_real128_default_descr(PyArray_DTypeMeta *)
{
    return reinterpret_cast<PyArray_Descr *>(real128_dtype_new_instance());
}

// Array getitem.
PyObject *npy_real128_getitem(PyArray_Descr *, char *data)
{
    mppp::real128 q;
    std::memcpy(&q, data, sizeof(q));
    return py_real128_from_args(q);
}

// Array setitem.
int npy_real128_setitem(PyArray_Descr *, PyObject *obj, char *itemptr)
{
    if (py_real128_check(obj)) {
        const auto *q = get_real128_val(obj);
        std::memcpy(itemptr, q, sizeof(mppp::real128));
        return 0;
    }

    // obj is not a real128, but it might be convertible to a real128.
    auto [r, flag] = real128_from_ob(obj);

    if (r) {
        // Conversion successful.
        std::memcpy(itemptr, &*r, sizeof(mppp::real128));
        return 0;
    }

    if (flag) {
        // Cannot convert obj to a real128 because the type of obj is not supported.
        PyErr_Format(PyExc_TypeError,
                     "Cannot invoke __setitem__() on a real128 array with an input value of type \"%s\"",
                     Py_TYPE(obj)->tp_name);
        return -1;
    }

    // Attempting a conversion to real128 generated an error.
    // The error flag has already been set by real128_from_ob(),
    // just return -1.
    return -1;
}

// Nonzero primitive.
npy_bool npy_real128_nonzero(void *data, void *)
{
    mppp::real128 q;
    std::memcpy(&q, data, sizeof(mppp::real128));

    return q != 0 ? NPY_TRUE : NPY_FALSE;
}

// argmin/argmax implementation.
template <auto Cmp>
int npy_real128_argminmax(void *data_, npy_intp n, npy_intp *max_ind)
{
    if (n == 0) {
        return 0;
    }

    const auto *data = static_cast<mppp::real128 *>(data_);

    npy_intp best_i = 0;
    auto best_r = data[0];

    for (npy_intp i = 1; i < n; ++i) {
        if (Cmp(data[i], best_r)) {
            best_i = i;
            best_r = data[i];
        }
    }

    *max_ind = best_i;

    return 0;
}

// Fill primitive (e.g., used for arange()).
int npy_real128_fill(void *data_, npy_intp length, void *)
{
    // NOTE: not sure if this is possible, let's stay on the safe side.
    if (length < 2) {
        return 0;
    }

    auto *data = static_cast<mppp::real128 *>(data_);
    const auto delta = data[1] - data[0];
    auto r = data[1];

    for (npy_intp i = 2; i < length; ++i) {
        r += delta;
        data[i] = r;
    }

    return 0;
}

// Fill with scalar primitive.
int npy_real128_fillwithscalar(void *buffer_, npy_intp length, void *value, void *)
{
    const auto r = *static_cast<mppp::real128 *>(value);
    auto *buffer = static_cast<mppp::real128 *>(buffer_);

    for (npy_intp i = 0; i < length; ++i) {
        buffer[i] = r;
    }

    return 0;
}

// Dot product.
void npy_real128_dot(void *ip0_, npy_intp is0, void *ip1_, npy_intp is1, void *op, npy_intp n, void *)
{
    using std::fma;

    mppp::real128 r = 0;
    const auto *ip0 = static_cast<const char *>(ip0_), *ip1 = static_cast<const char *>(ip1_);

    for (npy_intp i = 0; i < n; ++i) {
        r = fma(*reinterpret_cast<const mppp::real128 *>(ip0), *reinterpret_cast<const mppp::real128 *>(ip1), r);

        ip0 += is0;
        ip1 += is1;
    }

    *static_cast<mppp::real128 *>(op) = r;
}

// Comparison primitive, used for sorting.
// NOTE: use special comparisons to ensure NaNs are put at the end
// of an array when sorting.
int npy_real128_compare(const void *d0, const void *d1, void *)
{
    const auto &x = *static_cast<const mppp::real128 *>(d0);
    const auto &y = *static_cast<const mppp::real128 *>(d1);

    // NOTE: no exceptions are possible in these comparisons.
    if (mppp::real128_lt(x, y)) {
        return -1;
    }

    if (mppp::real128_equal_to(x, y)) {
        return 0;
    }

    return 1;
}

// Slots for basic numpy functionality.
PyType_Slot real128_slots[]
    = {{NPY_DT_ensure_canonical, reinterpret_cast<void *>(&npy_real128_ensure_canonical)},
       {NPY_DT_common_instance, reinterpret_cast<void *>(&npy_real128_common_instance)},
       {NPY_DT_common_dtype, reinterpret_cast<void *>(&npy_real128_common_dtype)},
       {NPY_DT_discover_descr_from_pyobject, reinterpret_cast<void *>(&npy_real128_discover_descriptor_from_pyobject)},
       {NPY_DT_default_descr, reinterpret_cast<void *>(&npy_real128_default_descr)},
       {NPY_DT_getitem, reinterpret_cast<void *>(&npy_real128_getitem)},
       {NPY_DT_setitem, reinterpret_cast<void *>(&npy_real128_setitem)},
       {NPY_DT_PyArray_ArrFuncs_nonzero, reinterpret_cast<void *>(&npy_real128_nonzero)},
       {NPY_DT_PyArray_ArrFuncs_compare, reinterpret_cast<void *>(&npy_real128_compare)},
       {NPY_DT_PyArray_ArrFuncs_argmin, reinterpret_cast<void *>(&npy_real128_argminmax<std::less{}>)},
       {NPY_DT_PyArray_ArrFuncs_argmax, reinterpret_cast<void *>(&npy_real128_argminmax<std::greater{}>)},
       {NPY_DT_PyArray_ArrFuncs_fill, reinterpret_cast<void *>(&npy_real128_fill)},
       {NPY_DT_PyArray_ArrFuncs_fillwithscalar, reinterpret_cast<void *>(&npy_real128_fillwithscalar)},
       {NPY_DT_PyArray_ArrFuncs_dotfunc, reinterpret_cast<void *>(&npy_real128_dot)},
       {0, nullptr}};

// Vector to hold the casting functions.
std::vector<PyArrayMethod_Spec *> real128_casts;

// Descriptors resolver for real128 to real128 casting.
NPY_CASTING
real128_to_real128_resolve_descriptors(PyArrayMethodObject_tag *, PyArray_DTypeMeta *const *,
                                       PyArray_Descr *const *given_descrs, PyArray_Descr **loop_descrs,
                                       npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == nullptr) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    } else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    *view_offset = 0;

    return NPY_NO_CASTING;
}

// real128 to real128 casting.
template <bool Aligned>
int real128_to_real128_strided_loop(PyArrayMethod_Context *, char *const *data, const npy_intp *dimensions,
                                    const npy_intp *strides, NpyAuxData *)
{
    auto N = dimensions[0];
    const auto *in_ptr = data[0];
    auto *out_ptr = data[1];
    const auto in_stride = strides[0];
    const auto out_stride = strides[1];

    while (N--) {
        if constexpr (Aligned) {
            *reinterpret_cast<mppp::real128 *>(out_ptr) = *reinterpret_cast<const mppp::real128 *>(in_ptr);
        } else {
            std::memcpy(out_ptr, in_ptr, sizeof(mppp::real128));
        }

        in_ptr += in_stride;
        out_ptr += out_stride;
    }

    return 0;
}

// Descriptors resolver for numpy to real128 casting.
template <typename T>
NPY_CASTING numpy_to_real128_resolve_descriptors(PyArrayMethodObject_tag *, PyArray_DTypeMeta *const *dtypes,
                                                 PyArray_Descr *const *given_descrs, PyArray_Descr **loop_descrs,
                                                 npy_intp *)
{
    if (given_descrs[1] == nullptr) {
        loop_descrs[1] = reinterpret_cast<PyArray_Descr *>(real128_dtype_new_instance());
        if (loop_descrs[1] == nullptr) {
            return static_cast<NPY_CASTING>(-1);
        }
    } else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    loop_descrs[0] = PyArray_GetDefaultDescr(dtypes[0]);

    return NPY_SAFE_CASTING;
}

// numpy to real128 casting.
template <typename T>
int numpy_to_real128_strided_loop(PyArrayMethod_Context *, char *const *data, const npy_intp *dimensions,
                                  const npy_intp *strides, NpyAuxData *)
{
    auto N = dimensions[0];
    const auto *in_ptr = data[0];
    auto *out_ptr = data[1];

    while (N--) {
        const auto in_val = *reinterpret_cast<const T *>(in_ptr);
        *reinterpret_cast<mppp::real128 *>(out_ptr) = static_cast<mppp::real128>(in_val);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }

    return 0;
}

// Helper to implement casting from numpy to real128.
template <typename T>
void add_cast_to(PyArray_DTypeMeta *from)
{
    static PyArray_DTypeMeta *dtypes[] = {from, &real128_dtype};
    static PyType_Slot slots[]
        = {{NPY_METH_resolve_descriptors, reinterpret_cast<void *>(&numpy_to_real128_resolve_descriptors<T>)},
           {NPY_METH_strided_loop, reinterpret_cast<void *>(&numpy_to_real128_strided_loop<T>)},
           {0, nullptr}};
    static PyArrayMethod_Spec spec = {
        .name = "cast_numpy_to_real128",
        .nin = 1,
        .nout = 1,
        .casting = NPY_SAFE_CASTING,
        .dtypes = dtypes,
        .slots = slots,
    };

    real128_casts.push_back(&spec);
}

// Descriptors resolver for real128 to numpy casting.
template <typename T>
NPY_CASTING real128_to_numpy_resolve_descriptors(PyArrayMethodObject_tag *, PyArray_DTypeMeta *const *dtypes,
                                                 PyArray_Descr *const *given_descrs, PyArray_Descr **loop_descrs,
                                                 npy_intp *)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    loop_descrs[1] = PyArray_GetDefaultDescr(dtypes[1]);

    return NPY_UNSAFE_CASTING;
}

// real128 to numpy casting.
template <typename T>
int real128_to_numpy_strided_loop(PyArrayMethod_Context *, char *const *data, const npy_intp *dimensions,
                                  const npy_intp *strides, NpyAuxData *)
{
    auto N = dimensions[0];
    const auto *in_ptr = data[0];
    auto *out_ptr = data[1];

    while (N--) {
        const auto in_val = *reinterpret_cast<const mppp::real128 *>(in_ptr);
        *reinterpret_cast<T *>(out_ptr) = static_cast<T>(in_val);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }

    return 0;
}

// Helper to implement casting from real128 to numpy.
template <typename T>
void add_cast_from(PyArray_DTypeMeta *to)
{
    static PyArray_DTypeMeta *dtypes[] = {&real128_dtype, to};
    static PyType_Slot slots[]
        = {{NPY_METH_resolve_descriptors, reinterpret_cast<void *>(&real128_to_numpy_resolve_descriptors<T>)},
           {NPY_METH_strided_loop, reinterpret_cast<void *>(&real128_to_numpy_strided_loop<T>)},
           {0, nullptr}};
    static PyArrayMethod_Spec spec = {
        .name = "cast_real128_to_numpy",
        .nin = 1,
        .nout = 1,
        .casting = NPY_UNSAFE_CASTING,
        .dtypes = dtypes,
        .slots = slots,
    };

    real128_casts.push_back(&spec);
}

// Setup of the casting methods.
PyArrayMethod_Spec **npy_real128_setup_casts()
{
    // Cast to self.
    static PyArray_DTypeMeta *real128_to_real128_dtypes[] = {nullptr, nullptr};
    static PyType_Slot real128_to_real128_slots[]
        = {{NPY_METH_resolve_descriptors, reinterpret_cast<void *>(&real128_to_real128_resolve_descriptors)},
           {NPY_METH_strided_loop, reinterpret_cast<void *>(&real128_to_real128_strided_loop<true>)},
           {NPY_METH_unaligned_strided_loop, reinterpret_cast<void *>(&real128_to_real128_strided_loop<false>)},
           {0, nullptr}};
    static PyArrayMethod_Spec real128_to_real128_spec = {
        .name = "cast_real128_to_real128",
        .nin = 1,
        .nout = 1,
        .casting = NPY_NO_CASTING,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = real128_to_real128_dtypes,
        .slots = real128_to_real128_slots,
    };
    real128_casts.push_back(&real128_to_real128_spec);

    // Casts from numpy to real128.
    add_cast_to<npy_bool>(&PyArray_BoolDType);
    add_cast_to<npy_byte>(&PyArray_ByteDType);
    add_cast_to<npy_short>(&PyArray_ShortDType);
    add_cast_to<npy_ushort>(&PyArray_UShortDType);
    add_cast_to<npy_int>(&PyArray_IntDType);
    add_cast_to<npy_uint>(&PyArray_UIntDType);
    add_cast_to<npy_long>(&PyArray_LongDType);
    add_cast_to<npy_ulong>(&PyArray_ULongDType);
    add_cast_to<npy_longlong>(&PyArray_LongLongDType);
    add_cast_to<npy_ulonglong>(&PyArray_ULongLongDType);
    add_cast_to<float>(&PyArray_FloatDType);
    add_cast_to<double>(&PyArray_DoubleDType);
    add_cast_to<long double>(&PyArray_LongDoubleDType);

    // Casts from real128 to numpy.
    add_cast_from<npy_bool>(&PyArray_BoolDType);
    add_cast_from<npy_byte>(&PyArray_ByteDType);
    add_cast_from<npy_short>(&PyArray_ShortDType);
    add_cast_from<npy_ushort>(&PyArray_UShortDType);
    add_cast_from<npy_int>(&PyArray_IntDType);
    add_cast_from<npy_uint>(&PyArray_UIntDType);
    add_cast_from<npy_long>(&PyArray_LongDType);
    add_cast_from<npy_ulong>(&PyArray_ULongDType);
    add_cast_from<npy_longlong>(&PyArray_LongLongDType);
    add_cast_from<npy_ulonglong>(&PyArray_ULongLongDType);
    add_cast_from<float>(&PyArray_FloatDType);
    add_cast_from<double>(&PyArray_DoubleDType);
    add_cast_from<long double>(&PyArray_LongDoubleDType);

    // Finalise.
    real128_casts.push_back(nullptr);

    return real128_casts.data();
}

// ufuncs setup.

// Descriptors resolver for unary ufuncs.
NPY_CASTING
real128_unary_op_resolve_descriptors(PyArrayMethodObject_tag *, PyArray_DTypeMeta *const *,
                                     PyArray_Descr *const *given_descrs, PyArray_Descr **loop_descrs, npy_intp *)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    } else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    return NPY_NO_CASTING;
}

// Implementation of a unary loop function.
template <auto UnaryFunc>
int real128_generic_unary_op_strided_loop(PyArrayMethod_Context *, char *const *data, const npy_intp *dimensions,
                                          const npy_intp *strides, NpyAuxData *)
{
    auto N = dimensions[0];
    const auto *in_ptr = data[0];
    auto *out_ptr = data[1];
    const auto in_stride = strides[0];
    const auto out_stride = strides[1];

    while (N--) {
        *reinterpret_cast<mppp::real128 *>(out_ptr) = UnaryFunc(*reinterpret_cast<const mppp::real128 *>(in_ptr));

        in_ptr += in_stride;
        out_ptr += out_stride;
    }

    return 0;
}

// Create a single unary func.
template <auto UnaryFunc>
void create_real128_unary_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == nullptr) {
        py_throw(PyExc_TypeError, fmt::format("The name '{}' in the NumPy module is not a ufunc", ufunc_name).c_str());
    }

    PyArray_DTypeMeta *dtypes[2] = {&real128_dtype, &real128_dtype};

    PyType_Slot slots[]
        = {{NPY_METH_resolve_descriptors, reinterpret_cast<void *>(&real128_unary_op_resolve_descriptors)},
           {NPY_METH_strided_loop, reinterpret_cast<void *>(&real128_generic_unary_op_strided_loop<UnaryFunc>)},
           {0, nullptr}};

    PyArrayMethod_Spec spec = {
        .name = "real128_unary_op",
        .nin = 1,
        .nout = 1,
        .casting = NPY_NO_CASTING,
        .dtypes = dtypes,
        .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &spec) < 0) {
        py_throw(PyExc_RuntimeError,
                 fmt::format("Failed to add the NumPy unary function '{}' for real128", ufunc_name).c_str());
    }
}

// Create all unary funcs.
void create_real128_unary_ufuncs(PyObject *numpy)
{
    create_real128_unary_ufunc<identity_func>(numpy, "positive");
    create_real128_unary_ufunc<negation_func>(numpy, "negative");
    create_real128_unary_ufunc<square_func>(numpy, "square");
    create_real128_unary_ufunc<abs_func>(numpy, "absolute");
    create_real128_unary_ufunc<abs_func>(numpy, "fabs");
    create_real128_unary_ufunc<sqrt_func>(numpy, "sqrt");
    create_real128_unary_ufunc<cbrt_func>(numpy, "cbrt");
    create_real128_unary_ufunc<sin_func>(numpy, "sin");
    create_real128_unary_ufunc<cos_func>(numpy, "cos");
    create_real128_unary_ufunc<tan_func>(numpy, "tan");
    create_real128_unary_ufunc<asin_func>(numpy, "arcsin");
    create_real128_unary_ufunc<acos_func>(numpy, "arccos");
    create_real128_unary_ufunc<atan_func>(numpy, "arctan");
    create_real128_unary_ufunc<sinh_func>(numpy, "sinh");
    create_real128_unary_ufunc<cosh_func>(numpy, "cosh");
    create_real128_unary_ufunc<tanh_func>(numpy, "tanh");
    create_real128_unary_ufunc<asinh_func>(numpy, "arcsinh");
    create_real128_unary_ufunc<acosh_func>(numpy, "arccosh");
    create_real128_unary_ufunc<atanh_func>(numpy, "arctanh");
    create_real128_unary_ufunc<deg2rad_func>(numpy, "deg2rad");
    create_real128_unary_ufunc<deg2rad_func>(numpy, "radians");
    create_real128_unary_ufunc<rad2deg_func>(numpy, "rad2deg");
    create_real128_unary_ufunc<rad2deg_func>(numpy, "degrees");
    create_real128_unary_ufunc<exp_func>(numpy, "exp");
#if defined(MPPP_QUADMATH_HAVE_EXP2Q)
    create_real128_unary_ufunc<exp2_func>(numpy, "exp2");
#endif
    create_real128_unary_ufunc<log_func>(numpy, "log");
    create_real128_unary_ufunc<log2_func>(numpy, "log2");
    create_real128_unary_ufunc<log10_func>(numpy, "log10");
    create_real128_unary_ufunc<log1p_func>(numpy, "log1p");
    create_real128_unary_ufunc<sign_func>(numpy, "sign");
}

// NOTE: I honestly have no idea what this does exactly. I just lifted this directly
// from the numpy-user-dtypes repository with minimal changes. I *think* this adds
// some extra type promotion logic which is somehow needed when handling types in binary
// ufuncs. This promoter function is supposed to be added into a capsule with a specific
// name and then passed to PyUFunc_AddPromoter(). See also the docs here:
//
// https://numpy.org/doc/stable/reference/c-api/array.html#c.PyUFunc_AddPromoter
int real128_ufunc_promoter(PyObject *ufunc_, PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
                           PyArray_DTypeMeta *new_op_dtypes[])
{
    auto *ufunc = reinterpret_cast<PyUFuncObject *>(ufunc_);

    const auto nin = ufunc->nin;
    const auto nargs = ufunc->nargs;
    PyArray_DTypeMeta *common = nullptr;
    bool has_real128 = false;

    // Handle the special case for reductions
    if (op_dtypes[0] == nullptr) {
        // Must be reduction.
        assert(nin == 2 && ufunc->nout == 1);
        for (int i = 0; i < 3; i++) {
            Py_INCREF(op_dtypes[1]);
            new_op_dtypes[i] = op_dtypes[1];
        }
        return 0;
    }

    // Check if any input or signature is real128.
    for (int i = 0; i < nin; i++) {
        if (op_dtypes[i] == &real128_dtype) {
            has_real128 = true;
        }
    }

    if (has_real128) {
        common = &real128_dtype;
    } else {
        for (int i = nin; i < nargs; i++) {
            if (signature[i] != nullptr) {
                if (common == nullptr) {
                    Py_INCREF(signature[i]);
                    common = signature[i];
                } else if (common != signature[i]) {
                    // Not homogeneous, unset common.
                    Py_CLEAR(common);
                    break;
                }
            }
        }
    }

    // If no common output dtype, use standard promotion for inputs.
    if (common == nullptr) {
        common = PyArray_PromoteDTypeSequence(nin, const_cast<PyArray_DTypeMeta **>(op_dtypes));
        if (common == nullptr) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                // Do not propagate normal promotion errors
                PyErr_Clear();
            }

            return -1;
        }
    }

    // Set all new_op_dtypes to the common dtype
    for (int i = 0; i < nargs; i++) {
        if (signature[i]) {
            // If signature is specified for this argument, use it
            Py_INCREF(signature[i]);
            new_op_dtypes[i] = signature[i];
        } else {
            // Otherwise, use the common dtype
            Py_INCREF(common);

            new_op_dtypes[i] = common;
        }
    }

    Py_XDECREF(common);

    return 0;
}

// Descriptors resolver for binary ufuncs.
NPY_CASTING real128_binary_op_resolve_descriptors(PyArrayMethodObject_tag *, PyArray_DTypeMeta *const *,
                                                  PyArray_Descr *const *given_descrs, PyArray_Descr **loop_descrs,
                                                  npy_intp *)
{
    // Set up input descriptors.
    for (int i = 0; i < 2; i++) {
        Py_INCREF(given_descrs[i]);
        loop_descrs[i] = given_descrs[i];
    }

    // Set up output descriptor.
    if (given_descrs[2] == NULL) {
        loop_descrs[2] = reinterpret_cast<PyArray_Descr *>(real128_dtype_new_instance());
        if (!loop_descrs[2]) {
            return static_cast<NPY_CASTING>(-1);
        }
    } else {
        Py_INCREF(given_descrs[2]);
        loop_descrs[2] = given_descrs[2];
    }

    return NPY_NO_CASTING;
}

// Implementation of a binary loop function.
template <auto BinaryFunc>
int real128_generic_binop_strided_loop(PyArrayMethod_Context *, char *const *data, const npy_intp *dimensions,
                                       const npy_intp *strides, NpyAuxData *)
{
    auto N = dimensions[0];
    const auto *in1_ptr = data[0], *in2_ptr = data[1];
    auto *out_ptr = data[2];
    const auto in1_stride = strides[0];
    const auto in2_stride = strides[1];
    const auto out_stride = strides[2];

    while (N--) {
        *reinterpret_cast<mppp::real128 *>(out_ptr) = BinaryFunc(*reinterpret_cast<const mppp::real128 *>(in1_ptr),
                                                                 *reinterpret_cast<const mppp::real128 *>(in2_ptr));

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }

    return 0;
}

// Create a single binary func.
template <auto BinaryFunc>
void create_real128_binary_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == nullptr) {
        py_throw(PyExc_TypeError, fmt::format("The name '{}' in the NumPy module is not a ufunc", ufunc_name).c_str());
    }

    PyArray_DTypeMeta *dtypes[3] = {&real128_dtype, &real128_dtype, &real128_dtype};

    PyType_Slot slots[]
        = {{NPY_METH_resolve_descriptors, reinterpret_cast<void *>(&real128_binary_op_resolve_descriptors)},
           {NPY_METH_strided_loop, reinterpret_cast<void *>(&real128_generic_binop_strided_loop<BinaryFunc>)},
           {0, nullptr}};

    PyArrayMethod_Spec spec = {
        .name = "real128_binop",
        .nin = 2,
        .nout = 1,
        .casting = NPY_NO_CASTING,
        .flags = NPY_METH_IS_REORDERABLE,
        .dtypes = dtypes,
        .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &spec) < 0) {
        py_throw(PyExc_RuntimeError,
                 fmt::format("Failed to add the NumPy binary function '{}' for real128", ufunc_name).c_str());
    }

    PyObject *promoter_capsule
        = PyCapsule_New(reinterpret_cast<void *>(&real128_ufunc_promoter), "numpy._ufunc_promoter", nullptr);
    if (promoter_capsule == nullptr) {
        py_throw(
            PyExc_RuntimeError,
            fmt::format("Failed to create the promoter capsule for the binary function '{}' for real128", ufunc_name)
                .c_str());
    }

    PyObject *DTypes = PyTuple_Pack(3, &PyArrayDescr_Type, &PyArrayDescr_Type, &PyArrayDescr_Type);
    if (DTypes == 0) {
        Py_DECREF(promoter_capsule);
        py_throw(PyExc_RuntimeError,
                 fmt::format("Failed to create the dtypes tuple for the binary function '{}' for real128", ufunc_name)
                     .c_str());
    }

    if (PyUFunc_AddPromoter(ufunc, DTypes, promoter_capsule) < 0) {
        Py_DECREF(promoter_capsule);
        Py_DECREF(DTypes);
        py_throw(
            PyExc_RuntimeError,
            fmt::format("Failed to add the promoter for the binary function '{}' for real128", ufunc_name).c_str());
    }

    Py_DECREF(promoter_capsule);
    Py_DECREF(DTypes);
}

// Create all binary funcs.
void create_real128_binary_ufuncs(PyObject *numpy)
{
    create_real128_binary_ufunc<std::plus{}>(numpy, "add");
    create_real128_binary_ufunc<std::minus{}>(numpy, "subtract");
    create_real128_binary_ufunc<std::multiplies{}>(numpy, "multiply");
    create_real128_binary_ufunc<std::divides{}>(numpy, "divide");
    create_real128_binary_ufunc<floor_divide_func>(numpy, "floor_divide");
    create_real128_binary_ufunc<pow_func>(numpy, "power");
    create_real128_binary_ufunc<atan2_func>(numpy, "arctan2");
    create_real128_binary_ufunc<min_func>(numpy, "minimum");
    create_real128_binary_ufunc<max_func>(numpy, "maximum");
}

// NOTE: I honestly have no idea what this does exactly. I just lifted this directly
// from the numpy-user-dtypes repository with minimal changes. I *think* this adds
// some extra type promotion logic which is somehow needed when handling types in binary
// ufuncs. This promoter function is supposed to be added into a capsule with a specific
// name and then passed to PyUFunc_AddPromoter(). See also the docs here:
//
// https://numpy.org/doc/stable/reference/c-api/array.html#c.PyUFunc_AddPromoter
int real128_comparison_ufunc_promoter(PyObject *ufunc_, PyArray_DTypeMeta *const op_dtypes[],
                                      PyArray_DTypeMeta *const signature[], PyArray_DTypeMeta *new_op_dtypes[])
{
    PyArray_DTypeMeta *new_signature[NPY_MAXARGS];
    std::memcpy(new_signature, signature, 3 * sizeof(PyArray_DTypeMeta *));
    new_signature[2] = nullptr;

    int res = real128_ufunc_promoter(ufunc_, op_dtypes, new_signature, new_op_dtypes);
    if (res < 0) {
        return -1;
    }

    Py_XSETREF(new_op_dtypes[2], &PyArray_BoolDType);

    return 0;
}

// Descriptors resolver for comparison operators.
NPY_CASTING real128_comparison_op_resolve_descriptors(PyArrayMethodObject_tag *, PyArray_DTypeMeta *const *,
                                                      PyArray_Descr *const *given_descrs, PyArray_Descr **loop_descrs,
                                                      npy_intp *)
{
    // Set up input descriptors.
    for (int i = 0; i < 2; i++) {
        Py_INCREF(given_descrs[i]);
        loop_descrs[i] = given_descrs[i];
    }

    // Set up output descriptor.
    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);
    if (!loop_descrs[2]) {
        return static_cast<NPY_CASTING>(-1);
    }

    return NPY_NO_CASTING;
}

// Implementation of a comparison loop function.
template <auto CompFunc>
int real128_generic_comp_strided_loop(PyArrayMethod_Context *, char *const *data, const npy_intp *dimensions,
                                      const npy_intp *strides, NpyAuxData *)
{
    auto N = dimensions[0];
    const auto *in1_ptr = data[0], *in2_ptr = data[1];
    auto *out_ptr = data[2];
    const auto in1_stride = strides[0];
    const auto in2_stride = strides[1];
    const auto out_stride = strides[2];

    while (N--) {
        const auto in1 = *reinterpret_cast<const mppp::real128 *>(in1_ptr);
        const auto in2 = *reinterpret_cast<const mppp::real128 *>(in2_ptr);
        *reinterpret_cast<npy_bool *>(out_ptr) = CompFunc(in1, in2);

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }

    return 0;
}

// Create a single comparison ufunc.
template <auto CompFunc>
void create_real128_comparison_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == nullptr) {
        py_throw(PyExc_TypeError, fmt::format("The name '{}' in the NumPy module is not a ufunc", ufunc_name).c_str());
    }

    PyArray_DTypeMeta *dtypes[3] = {&real128_dtype, &real128_dtype, &PyArray_BoolDType};

    PyType_Slot slots[]
        = {{NPY_METH_resolve_descriptors, reinterpret_cast<void *>(&real128_comparison_op_resolve_descriptors)},
           {NPY_METH_strided_loop, reinterpret_cast<void *>(&real128_generic_comp_strided_loop<CompFunc>)},
           {0, nullptr}};

    PyArrayMethod_Spec spec = {
        .name = "real128_comp",
        .nin = 2,
        .nout = 1,
        .casting = NPY_SAFE_CASTING,
        .dtypes = dtypes,
        .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &spec) < 0) {
        py_throw(PyExc_RuntimeError,
                 fmt::format("Failed to add the NumPy comparison function '{}' for real128", ufunc_name).c_str());
    }

    PyObject *promoter_capsule
        = PyCapsule_New(reinterpret_cast<void *>(&real128_comparison_ufunc_promoter), "numpy._ufunc_promoter", nullptr);
    if (promoter_capsule == nullptr) {
        py_throw(PyExc_RuntimeError,
                 fmt::format("Failed to create the promoter capsule for the comparison function '{}' for real128",
                             ufunc_name)
                     .c_str());
    }

    PyObject *DTypes = PyTuple_Pack(3, &PyArrayDescr_Type, &PyArrayDescr_Type, &PyArray_BoolDType);
    if (DTypes == 0) {
        Py_DECREF(promoter_capsule);
        py_throw(
            PyExc_RuntimeError,
            fmt::format("Failed to create the dtypes tuple for the comparison function '{}' for real128", ufunc_name)
                .c_str());
    }

    if (PyUFunc_AddPromoter(ufunc, DTypes, promoter_capsule) < 0) {
        Py_DECREF(promoter_capsule);
        Py_DECREF(DTypes);
        py_throw(
            PyExc_RuntimeError,
            fmt::format("Failed to add the promoter for the comparison function '{}' for real128", ufunc_name).c_str());
    }

    Py_DECREF(promoter_capsule);
    Py_DECREF(DTypes);
}

// Create all comparison funcs.
void create_real128_comparison_ufuncs(PyObject *numpy)
{
    create_real128_comparison_ufunc<std::equal_to{}>(numpy, "equal");
    create_real128_comparison_ufunc<std::not_equal_to{}>(numpy, "not_equal");
    create_real128_comparison_ufunc<std::greater{}>(numpy, "greater");
    create_real128_comparison_ufunc<std::greater_equal{}>(numpy, "greater_equal");
    create_real128_comparison_ufunc<std::less{}>(numpy, "less");
    create_real128_comparison_ufunc<std::less_equal{}>(numpy, "less_equal");
}

#if 0

// NumPy array function.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyArray_ArrFuncs npy_py_real128_arr_funcs = {};

// NumPy type descriptor.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyArray_Descr npy_py_real128_descr = {PyObject_HEAD_INIT(0) & py_real128_type};

// Array getitem.
PyObject *npy_py_real128_getitem(void *data, void *)
{
    mppp::real128 q;
    std::memcpy(&q, data, sizeof(q));
    return py_real128_from_args(q);
}

// Array setitem.
int npy_py_real128_setitem(PyObject *item, void *data, void *)
{
    if (py_real128_check(item)) {
        const auto *q = get_real128_val(item);
        std::memcpy(data, q, sizeof(mppp::real128));
        return 0;
    }

    // item is not a real128, but it might be convertible to a real128.
    auto [r, flag] = real128_from_ob(item);

    if (r) {
        // Conversion successful.
        std::memcpy(data, &*r, sizeof(mppp::real128));
        return 0;
    }

    if (flag) {
        // Cannot convert item to a real128 because the type of item is not supported.
        PyErr_Format(PyExc_TypeError,
                     "Cannot invoke __setitem__() on a real128 array with an input value of type \"%s\"",
                     Py_TYPE(item)->tp_name);
        return -1;
    }

    // Attempting a conversion to real128 generated an error.
    // The error flag has already been set by real128_from_ob(),
    // just return -1.
    return -1;
}

// Byteswap a real128 in place.
void byteswap(mppp::real128 *x)
{
    static_assert(sizeof(mppp::real128) == sizeof(x->m_value));

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
void npy_py_real128_copyswap(void *dst, void *src, int swap, void *)
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

// Copyswapn primitive.
// NOTE: although the NumPy docs mention that you *must* implement copyswapn, it seems
// that - at least in recent NumPy versions - a synthesized default implementation of
// copyswapn built on top of copyswap is used if copyswapn is not defined. Perhaps
// then we can remove this in the future.
void npy_py_real128_copyswapn(void *dst_, npy_intp dstride, void *src_, npy_intp sstride, npy_intp n, int swap, void *)
{
    char *dst = static_cast<char *>(dst_), *src = static_cast<char *>(src_);

    for (npy_intp i = 0; i < n; ++i) {
        auto *r = reinterpret_cast<mppp::real128 *>(dst + dstride * i);

        if (src != nullptr) {
            // NOTE: not sure if src and dst can overlap here,
            // use memmove() just in case.
            std::memmove(r, src + sstride * i, sizeof(mppp::real128));
        }

        if (swap != 0) {
            byteswap(r);
        }
    }
}

// Nonzero primitive.
npy_bool npy_py_real128_nonzero(void *data, void *)
{
    mppp::real128 q;
    std::memcpy(&q, data, sizeof(mppp::real128));

    return q != 0 ? NPY_TRUE : NPY_FALSE;
}

// Comparison primitive, used for sorting.
// NOTE: use special comparisons to ensure NaNs are put at the end
// of an array when sorting.
int npy_py_real128_compare(const void *d0, const void *d1, void *)
{
    const auto &x = *static_cast<const mppp::real128 *>(d0);
    const auto &y = *static_cast<const mppp::real128 *>(d1);

    // NOTE: no exceptions are possible in these comparisons.
    if (mppp::real128_lt(x, y)) {
        return -1;
    }

    if (mppp::real128_equal_to(x, y)) {
        return 0;
    }

    return 1;
}

// argmin/argmax implementation.
template <typename F>
int npy_py_real128_argminmax(void *data_, npy_intp n, npy_intp *max_ind, const F &cmp)
{
    if (n == 0) {
        return 0;
    }

    const auto *data = static_cast<mppp::real128 *>(data_);

    npy_intp best_i = 0;
    auto best_r = data[0];

    for (npy_intp i = 1; i < n; ++i) {
        if (cmp(data[i], best_r)) {
            best_i = i;
            best_r = data[i];
        }
    }

    *max_ind = best_i;

    return 0;
}

// Fill primitive (e.g., used for arange()).
int npy_py_real128_fill(void *data_, npy_intp length, void *)
{
    // NOTE: not sure if this is possible, let's stay on the safe side.
    if (length < 2) {
        return 0;
    }

    auto *data = static_cast<mppp::real128 *>(data_);
    const auto delta = data[1] - data[0];
    auto r = data[1];

    for (npy_intp i = 2; i < length; ++i) {
        r += delta;
        data[i] = r;
    }

    return 0;
}

// Fill with scalar primitive.
int npy_py_real128_fillwithscalar(void *buffer_, npy_intp length, void *value, void *)
{
    const auto r = *static_cast<mppp::real128 *>(value);
    auto *buffer = static_cast<mppp::real128 *>(buffer_);

    for (npy_intp i = 0; i < length; ++i) {
        buffer[i] = r;
    }

    return 0;
}

// Dot product.
void npy_py_real128_dot(void *ip0_, npy_intp is0, void *ip1_, npy_intp is1, void *op, npy_intp n, void *)
{
    using std::fma;

    mppp::real128 r = 0;
    const auto *ip0 = static_cast<const char *>(ip0_), *ip1 = static_cast<const char *>(ip1_);

    for (npy_intp i = 0; i < n; ++i) {
        r = fma(*reinterpret_cast<const mppp::real128 *>(ip0), *reinterpret_cast<const mppp::real128 *>(ip1), r);

        ip0 += is0;
        ip1 += is1;
    }

    *static_cast<mppp::real128 *>(op) = r;
}

// Implementation of matrix multiplication.
void npy_py_real128_matrix_multiply(char **args, const npy_intp *dimensions, const npy_intp *steps)
{
    // Pointers to data for input and output arrays.
    char *ip1 = args[0];
    char *ip2 = args[1];
    char *op = args[2];

    // Lengths of core dimensions.
    npy_intp dm = dimensions[0];
    npy_intp dn = dimensions[1];
    npy_intp dp = dimensions[2];

    // Striding over core dimensions.
    npy_intp is1_m = steps[0];
    npy_intp is1_n = steps[1];
    npy_intp is2_n = steps[2];
    npy_intp is2_p = steps[3];
    npy_intp os_m = steps[4];
    npy_intp os_p = steps[5];

    // Calculate dot product for each row/column vector pair.
    for (npy_intp m = 0; m < dm; ++m) {
        npy_intp p = 0;
        for (; p < dp; ++p) {
            npy_py_real128_dot(ip1, is1_n, ip2, is2_n, op, dn, nullptr);

            // Advance to next column of 2nd input array and output array.
            ip2 += is2_p;
            op += os_p;
        }

        // Reset to first column of 2nd input array and output array.
        ip2 -= is2_p * p;
        op -= os_p * p;

        // Advance to next row of 1st input array and output array.
        ip1 += is1_m;
        op += os_m;
    }
}

void npy_py_real128_gufunc_matrix_multiply(char **args, const npy_intp *dimensions, const npy_intp *steps, void *)
{
    // Length of flattened outer dimensions.
    npy_intp dN = dimensions[0];

    // Striding over flattened outer dimensions for input and output arrays.
    npy_intp s0 = steps[0];
    npy_intp s1 = steps[1];
    npy_intp s2 = steps[2];

    // Loop through outer dimensions, performing matrix multiply on core dimensions for each loop.
    for (npy_intp N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {
        npy_py_real128_matrix_multiply(args, dimensions + 1, steps + 3);
    }
}

// Generic NumPy conversion function to real128.
template <typename From>
void npy_cast_to_real128(void *from, void *to, npy_intp n, void *, void *)
{
    // Sanity check, we don't want to implement real128 -> real128 conversion.
    static_assert(!std::is_same_v<From, mppp::real128>);

    const auto *typed_from = static_cast<const From *>(from);
    auto *typed_to = static_cast<mppp::real128 *>(to);

    for (npy_intp i = 0; i < n; ++i) {
        typed_to[i] = typed_from[i];
    }
}

// Generic NumPy conversion function from real128.
template <typename To>
void npy_cast_from_real128(void *from, void *to, npy_intp n, void *, void *)
{
    // Sanity check, we don't want to implement real128 -> real128 conversion.
    static_assert(!std::is_same_v<To, mppp::real128>);

    const auto *typed_from = static_cast<const mppp::real128 *>(from);
    auto *typed_to = static_cast<To *>(to);

    for (npy_intp i = 0; i < n; ++i) {
        typed_to[i] = static_cast<To>(typed_from[i]);
    }
}

// Helper to register NumPy casting functions to/from T.
template <typename T>
void npy_register_cast_functions()
{
    if (PyArray_RegisterCastFunc(PyArray_DescrFromType(get_dtype<T>()), npy_registered_py_real128,
                                 &npy_cast_to_real128<T>)
        < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    // NOTE: this is to signal that conversion of any scalar type to real128 is safe.
    if (PyArray_RegisterCanCast(PyArray_DescrFromType(get_dtype<T>()), npy_registered_py_real128, NPY_NOSCALAR) < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    if (PyArray_RegisterCastFunc(&npy_py_real128_descr, get_dtype<T>(), &npy_cast_from_real128<T>) < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }
}

// Generic NumPy unary operation.
template <typename T, typename F>
void py_real128_ufunc_unary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *, const F &f)
{
    npy_intp is1 = steps[0], os1 = steps[1], n = dimensions[0];
    char *ip1 = args[0], *op1 = args[1];

    for (npy_intp i = 0; i < n; ++i, ip1 += is1, op1 += os1) {
        const auto &x = *reinterpret_cast<mppp::real128 *>(ip1);
        *reinterpret_cast<T *>(op1) = static_cast<T>(f(x));
    };
}

template <typename F>
void py_real128_ufunc_unary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *p, const F &f)
{
    py_real128_ufunc_unary<mppp::real128>(args, dimensions, steps, p, f);
}

// Generic NumPy binary operation.
template <typename T, typename F>
void py_real128_ufunc_binary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *, const F &f)
{
    npy_intp is0 = steps[0], is1 = steps[1], os = steps[2], n = *dimensions;
    char *i0 = args[0], *i1 = args[1], *o = args[2];

    for (npy_intp k = 0; k < n; ++k) {
        const auto &x = *reinterpret_cast<mppp::real128 *>(i0);
        const auto &y = *reinterpret_cast<mppp::real128 *>(i1);
        *reinterpret_cast<T *>(o) = static_cast<T>(f(x, y));
        i0 += is0;
        i1 += is1;
        o += os;
    }
}

template <typename F>
void py_real128_ufunc_binary(char **args, const npy_intp *dimensions, const npy_intp *steps, void *p, const F &f)
{
    py_real128_ufunc_binary<mppp::real128>(args, dimensions, steps, p, f);
}

// Helper to register a NumPy ufunc.
template <typename... Types>
void npy_register_ufunc(py::module_ &numpy, const char *name, PyUFuncGenericFunction func, Types... types_)
{
    py::object ufunc_ob = numpy.attr(name);
    auto *ufunc_ = ufunc_ob.ptr();
    if (PyObject_IsInstance(ufunc_, reinterpret_cast<PyObject *>(&PyUFunc_Type)) == 0) {
        py_throw(PyExc_TypeError, fmt::format("The name '{}' in the NumPy module is not a ufunc", name).c_str());
    }
    auto *ufunc = reinterpret_cast<PyUFuncObject *>(ufunc_);

    int types[] = {types_...};
    if (std::size(types) != boost::numeric_cast<std::size_t>(ufunc->nargs)) {
        py_throw(PyExc_TypeError, fmt::format("Invalid arity for the ufunc '{}': the NumPy function expects {} "
                                              "arguments, but {} arguments were provided instead",
                                              name, ufunc->nargs, std::size(types))
                                      .c_str());
    }

    if (PyUFunc_RegisterLoopForType(ufunc, npy_registered_py_real128, func, types, nullptr) < 0) {
        py_throw(PyExc_TypeError, fmt::format("The registration of the ufunc '{}' failed", name).c_str());
    }
}

#endif

} // namespace

} // namespace detail

// Check if the input object is a py_real128.
bool py_real128_check(PyObject *ob)
{
    return PyObject_IsInstance(ob, reinterpret_cast<PyObject *>(&py_real128_type)) != 0;
}

// Helper to reach the mppp::real128 stored inside a py_real128.
mppp::real128 *get_real128_val(PyObject *self)
{
    assert(py_real128_check(self));

    return std::launder(reinterpret_cast<mppp::real128 *>(reinterpret_cast<py_real128 *>(self)->m_storage));
}

// Helper to create a pyreal128 from a real128.
PyObject *pyreal128_from_real128(const mppp::real128 &src)
{
    return detail::py_real128_from_args(src);
}

void expose_real128(py::module_ &m)
{
    // Fill out the entries of py_real128_type.
    py_real128_type.tp_name = "heyoka.core.real128";
    py_real128_type.tp_basicsize = sizeof(py_real128);
    py_real128_type.tp_flags = Py_TPFLAGS_DEFAULT;
    py_real128_type.tp_doc = PyDoc_STR("");
    py_real128_type.tp_new = detail::py_real128_new;
    py_real128_type.tp_init = detail::py_real128_init;
    py_real128_type.tp_dealloc = detail::py_real128_dealloc;
    py_real128_type.tp_repr = detail::py_real128_repr;
    py_real128_type.tp_as_number = &detail::py_real128_as_number;
    py_real128_type.tp_richcompare = &detail::py_real128_rcmp;
    py_real128_type.tp_methods = detail::py_real128_methods;

    // Fill out the functions for the number protocol. See:
    // https://docs.python.org/3/c-api/number.html
    detail::py_real128_as_number.nb_negative
        = [](PyObject *a) { return detail::py_real128_unop(a, detail::negation_func); };
    detail::py_real128_as_number.nb_positive
        = [](PyObject *a) { return detail::py_real128_unop(a, detail::identity_func); };
    detail::py_real128_as_number.nb_absolute = [](PyObject *a) { return detail::py_real128_unop(a, detail::abs_func); };
    detail::py_real128_as_number.nb_add
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, std::plus{}); };
    detail::py_real128_as_number.nb_subtract
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, std::minus{}); };
    detail::py_real128_as_number.nb_multiply
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, std::multiplies{}); };
    detail::py_real128_as_number.nb_true_divide
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, std::divides{}); };
    detail::py_real128_as_number.nb_floor_divide
        = [](PyObject *a, PyObject *b) { return detail::py_real128_binop(a, b, detail::floor_divide_func); };
    detail::py_real128_as_number.nb_power = [](PyObject *a, PyObject *b, PyObject *mod) -> PyObject * {
        if (mod != Py_None) {
            PyErr_SetString(PyExc_ValueError, "Modular exponentiation is not supported for real128");

            return nullptr;
        }

        return detail::py_real128_binop(a, b, detail::pow_func);
    };
    detail::py_real128_as_number.nb_bool
        = [](PyObject *arg) { return static_cast<int>(static_cast<bool>(*get_real128_val(arg))); };
    detail::py_real128_as_number.nb_float
        = [](PyObject *arg) { return PyFloat_FromDouble(static_cast<double>(*get_real128_val(arg))); };
    // NOTE: for large integers, this goes through a string conversion.
    // Of course, this can be implemented much faster.
    detail::py_real128_as_number.nb_int = [](PyObject *arg) -> PyObject * {
        using std::isfinite;
        using std::isnan;

        const auto val = *get_real128_val(arg);

        if (isnan(val)) {
            PyErr_SetString(PyExc_ValueError, "Cannot convert real128 NaN to integer");
            return nullptr;
        }

        if (!isfinite(val)) {
            PyErr_SetString(PyExc_OverflowError, "Cannot convert real128 infinity to integer");
            return nullptr;
        }

        PyObject *retval = nullptr;

        with_pybind11_eh([&]() {
            const auto val_int = mppp::integer<1>{val};
            long long candidate = 0;
            if (val_int.get(candidate)) {
                retval = PyLong_FromLongLong(candidate);
            } else {
                retval = PyLong_FromString(val_int.to_string().c_str(), nullptr, 10);
            }
        });

        return retval;
    };

    // Finalize py_real128_type.
    if (PyType_Ready(&py_real128_type) < 0) {
        // NOTE: PyType_Ready() already sets the exception flag.
        throw py::error_already_set();
    }

    // Add py_real128_type to the module.
    Py_INCREF(&py_real128_type);
    if (PyModule_AddObject(m.ptr(), "real128", reinterpret_cast<PyObject *>(&py_real128_type)) < 0) {
        Py_DECREF(&py_real128_type);
        py_throw(PyExc_TypeError, "Could not add the real128 type to the module");
    }

    // Numpy setup.

    // Struct used to post-finalise the real128 dtype.
    PyArrayDTypeMeta_Spec real128_dtype_spec = {
        // Here we specify the Python scalar associated
        // to the dtype.
        .typeobj = &py_real128_type,
        // Signal that real128 is a numeric dtype.
        .flags = NPY_DT_NUMERIC,
        // The casting functions.
        .casts = detail::npy_real128_setup_casts(),
        // The basic numpy functions.
        .slots = detail::real128_slots,
    };

    // Init the real128 dtype descriptor.
    detail::real128_dtype.super.ht_type = detail::init_real128_dtype();

    // Finalise the real128 dtype as a Python object.
    if (PyType_Ready(reinterpret_cast<PyTypeObject *>(&detail::real128_dtype)) < 0) {
        // NOTE: PyType_Ready() already sets the exception flag.
        throw py::error_already_set();
    }

    // Post-finalisation of the real128 dtype.
    if (PyArrayInitDTypeMeta_FromSpec(&detail::real128_dtype, &real128_dtype_spec) < 0) {
        // NOTE: not sure if PyArrayInitDTypeMeta_FromSpec() sets an exception flag here.
        throw py::error_already_set();
    }

    // Add it to the module.
    Py_INCREF(&detail::real128_dtype);
    if (PyModule_AddObject(m.ptr(), "real128_dtype", reinterpret_cast<PyObject *>(&detail::real128_dtype)) < 0) {
        Py_DECREF(&detail::real128_dtype);
        py_throw(PyExc_TypeError, "Could not add the real128 dtype to the module");
    }

    // NOTE: need access to the numpy module to register ufuncs.
    auto *numpy_mod = py::module_::import("numpy").ptr();

    // Create the unary ufuncs.
    detail::create_real128_unary_ufuncs(numpy_mod);

    // Create the binary ufuncs.
    detail::create_real128_binary_ufuncs(numpy_mod);

    // Create the comparison ufuncs.
    detail::create_real128_comparison_ufuncs(numpy_mod);

#if 0
    // Fill out the NumPy descriptor.
    detail::npy_py_real128_descr.kind = 'f';
    detail::npy_py_real128_descr.type = 'q';
    detail::npy_py_real128_descr.byteorder = '=';
    detail::npy_py_real128_descr.flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM;
    detail::npy_py_real128_descr.elsize = sizeof(mppp::real128);
    detail::npy_py_real128_descr.alignment = alignof(mppp::real128);
    detail::npy_py_real128_descr.f = &detail::npy_py_real128_arr_funcs;

    // Setup the basic NumPy array functions.
    // NOTE: not 100% sure PyArray_InitArrFuncs() is needed, as npy_py_real128_arr_funcs
    // is already cleared out on creation.
    PyArray_InitArrFuncs(&detail::npy_py_real128_arr_funcs);
    // NOTE: get/set item, copyswap and nonzero are the functions that
    // need to be able to deal with "misbehaved" array:
    // https://numpy.org/doc/stable/reference/c-api/types-and-structures.html
    // Hence, we group the together at the beginning.
    detail::npy_py_real128_arr_funcs.getitem = detail::npy_py_real128_getitem;
    detail::npy_py_real128_arr_funcs.setitem = detail::npy_py_real128_setitem;
    detail::npy_py_real128_arr_funcs.copyswap = detail::npy_py_real128_copyswap;
    detail::npy_py_real128_arr_funcs.copyswapn = detail::npy_py_real128_copyswapn;
    detail::npy_py_real128_arr_funcs.nonzero = detail::npy_py_real128_nonzero;
    detail::npy_py_real128_arr_funcs.compare = detail::npy_py_real128_compare;
    detail::npy_py_real128_arr_funcs.argmin = [](void *data, npy_intp n, npy_intp *max_ind, void *) {
        return detail::npy_py_real128_argminmax(data, n, max_ind, std::less{});
    };
    detail::npy_py_real128_arr_funcs.argmax = [](void *data, npy_intp n, npy_intp *max_ind, void *) {
        return detail::npy_py_real128_argminmax(data, n, max_ind, std::greater{});
    };
    detail::npy_py_real128_arr_funcs.fill = detail::npy_py_real128_fill;
    detail::npy_py_real128_arr_funcs.fillwithscalar = detail::npy_py_real128_fillwithscalar;
    detail::npy_py_real128_arr_funcs.dotfunc = detail::npy_py_real128_dot;
    // NOTE: not sure if this is needed - it does not seem to have
    // any effect and the online examples of user dtypes do not set it.
    // Let's leave it commented out at this time.
    // detail::npy_py_real128_arr_funcs.scalarkind = [](void *) -> int { return NPY_FLOAT_SCALAR; };

    // Register the NumPy data type.
    Py_SET_TYPE(&detail::npy_py_real128_descr, &PyArrayDescr_Type);
    npy_registered_py_real128 = PyArray_RegisterDataType(&detail::npy_py_real128_descr);
    if (npy_registered_py_real128 < 0) {
        // NOTE: PyArray_RegisterDataType() already sets the error flag.
        throw py::error_already_set();
    }

    // Support the dtype(real128) syntax.
    if (PyDict_SetItemString(py_real128_type.tp_dict, "dtype",
                             reinterpret_cast<PyObject *>(&detail::npy_py_real128_descr))
        < 0) {
        py_throw(PyExc_TypeError, "Cannot add the 'dtype' field to the real128 class");
    }

    // NOTE: need access to the numpy module to register ufuncs.
    auto numpy_mod = py::module_::import("numpy");

    // Arithmetics.
    detail::npy_register_ufunc(
        numpy_mod, "add",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, std::plus{});
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "subtract",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, std::minus{});
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "multiply",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, std::multiplies{});
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "square",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::square_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "divide",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, std::divides{});
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "floor_divide",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, detail::floor_divide_func);
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "absolute",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::abs_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "fabs",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::abs_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "positive",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::identity_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "negative",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::negation_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    // Power/roots.
    detail::npy_register_ufunc(
        numpy_mod, "power",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, detail::pow_func);
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "sqrt",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::sqrt_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "cbrt",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::cbrt_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    // Trigonometry.
    detail::npy_register_ufunc(
        numpy_mod, "sin",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::sin_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "cos",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::cos_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "tan",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::tan_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "arcsin",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::asin_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "arccos",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::acos_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "arctan",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::atan_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "arctan2",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, detail::atan2_func);
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "sinh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::sinh_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "cosh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::cosh_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "tanh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::tanh_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "arcsinh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::asinh_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "arccosh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::acosh_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "arctanh",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::atanh_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "deg2rad",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::deg2rad_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "radians",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::deg2rad_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "rad2deg",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::rad2deg_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "degrees",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::rad2deg_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    // Exponentials and logarithms.
    detail::npy_register_ufunc(
        numpy_mod, "exp",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::exp_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
#if defined(MPPP_QUADMATH_HAVE_EXP2Q)
    detail::npy_register_ufunc(
        numpy_mod, "exp2",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::exp2_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
#endif
    detail::npy_register_ufunc(
        numpy_mod, "expm1",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::expm1_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "log",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::log_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "log2",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::log2_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "log10",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::log10_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "log1p",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::log1p_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    // Comparisons.
    detail::npy_register_ufunc(
        numpy_mod, "less",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::less{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "less_equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::less_equal{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::equal_to{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "not_equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::not_equal_to{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "greater",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::greater{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "greater_equal",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary<npy_bool>(args, dimensions, steps, data, std::greater_equal{});
        },
        npy_registered_py_real128, npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "sign",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary(args, dimensions, steps, data, detail::sign_func);
        },
        npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "maximum",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, detail::max_func);
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "minimum",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_binary(args, dimensions, steps, data, detail::min_func);
        },
        npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);
    detail::npy_register_ufunc(
        numpy_mod, "isnan",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary<npy_bool>(args, dimensions, steps, data, detail::isnan_func);
        },
        npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "isinf",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary<npy_bool>(args, dimensions, steps, data, detail::isinf_func);
        },
        npy_registered_py_real128, NPY_BOOL);
    detail::npy_register_ufunc(
        numpy_mod, "isfinite",
        [](char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            detail::py_real128_ufunc_unary<npy_bool>(args, dimensions, steps, data, detail::isfinite_func);
        },
        npy_registered_py_real128, NPY_BOOL);
    // Matrix multiplication.
    detail::npy_register_ufunc(numpy_mod, "matmul", detail::npy_py_real128_gufunc_matrix_multiply,
                               npy_registered_py_real128, npy_registered_py_real128, npy_registered_py_real128);

    // Casting.
    // NOTE: casting to/from real is already implemented
    // in the real exposition.
    detail::npy_register_cast_functions<float>();
    detail::npy_register_cast_functions<double>();

#if defined(MPPP_FLOAT128_WITH_LONG_DOUBLE)

    // NOTE: registering conversions to/from long double has several
    // adverse effects on the casting rules. Unclear at this time if
    // such issues are from our code or NumPy's. Let's leave it commented
    // at this time.
    // detail::npy_register_cast_functions<long double>();

#endif

    detail::npy_register_cast_functions<npy_int8>();
    detail::npy_register_cast_functions<npy_int16>();
    detail::npy_register_cast_functions<npy_int32>();
    detail::npy_register_cast_functions<npy_int64>();

    detail::npy_register_cast_functions<npy_uint8>();
    detail::npy_register_cast_functions<npy_uint16>();
    detail::npy_register_cast_functions<npy_uint32>();
    detail::npy_register_cast_functions<npy_uint64>();

    // NOTE: need to do bool manually as it typically overlaps
    // with another C++ type (e.g., uint8).
    if (PyArray_RegisterCastFunc(PyArray_DescrFromType(NPY_BOOL), npy_registered_py_real128,
                                 &detail::npy_cast_to_real128<npy_bool>)
        < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    // NOTE: this is to signal that conversion of bool to real128 is safe.
    if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BOOL), npy_registered_py_real128, NPY_NOSCALAR) < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

    if (PyArray_RegisterCastFunc(&detail::npy_py_real128_descr, NPY_BOOL, &detail::npy_cast_from_real128<npy_bool>)
        < 0) {
        py_throw(PyExc_TypeError, "The registration of a NumPy casting function failed");
    }

#endif

    // Helper to return the epsilon of real128.
    m.def("_get_real128_eps", []() { return std::numeric_limits<mppp::real128>::epsilon(); });
}

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#else

void expose_real128(py::module_ &) {}

#endif

} // namespace heyoka_py
