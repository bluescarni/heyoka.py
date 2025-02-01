// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>

#include <pybind11/pybind11.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include "common_utils.hpp"
#include "custom_casters.hpp"

#if defined(HEYOKA_HAVE_REAL128)

#include "expose_real128.hpp"

#endif

#if defined(HEYOKA_HAVE_REAL)

#include "expose_real.hpp"

#endif

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace pybind11::detail
{

bool type_caster<float>::load(handle src, bool)
{
    if (PyObject_IsInstance(src.ptr(), reinterpret_cast<PyObject *>(&PyFloat32ArrType_Type)) == 0) {
        return false;
    }

    value = reinterpret_cast<PyFloat32ScalarObject *>(src.ptr())->obval;

    return true;
}

handle type_caster<float>::cast(const float &src, return_value_policy, handle)
{
    auto *ret_ob = PyArrayScalar_New(Float32);
    if (ret_ob == nullptr) {
        heyoka_py::py_throw(PyExc_RuntimeError, "Unable to obtain storage for a NumPy float32 object");
    }

    assert(PyObject_IsInstance(ret_ob, reinterpret_cast<PyObject *>(&PyFloat32ArrType_Type)) != 0);

    reinterpret_cast<PyFloat32ScalarObject *>(ret_ob)->obval = src;

    return ret_ob;
}

bool type_caster<long double>::load(handle src, bool)
{
    if (PyObject_IsInstance(src.ptr(), reinterpret_cast<PyObject *>(&PyLongDoubleArrType_Type)) == 0) {
        return false;
    }

    value = reinterpret_cast<PyLongDoubleScalarObject *>(src.ptr())->obval;

    return true;
}

handle type_caster<long double>::cast(const long double &src, return_value_policy, handle)
{
    auto *ret_ob = PyArrayScalar_New(LongDouble);
    if (ret_ob == nullptr) {
        heyoka_py::py_throw(PyExc_RuntimeError, "Unable to obtain storage for a NumPy longdouble object");
    }

    assert(PyObject_IsInstance(ret_ob, reinterpret_cast<PyObject *>(&PyLongDoubleArrType_Type)) != 0);

    reinterpret_cast<PyLongDoubleScalarObject *>(ret_ob)->obval = src;

    return ret_ob;
}

#if defined(HEYOKA_HAVE_REAL128)

bool type_caster<mppp::real128>::load(handle src, bool)
{
    namespace heypy = heyoka_py;

    if (!heypy::py_real128_check(src.ptr())) {
        return false;
    }

    value = *heypy::get_real128_val(src.ptr());

    return true;
}

handle type_caster<mppp::real128>::cast(const mppp::real128 &src, return_value_policy, handle)
{
    namespace heypy = heyoka_py;

    auto *ret_ob = heypy::pyreal128_from_real128(src);
    if (ret_ob == nullptr) {
        heyoka_py::py_throw(PyExc_RuntimeError, "Unable to obtain storage for a real128 object");
    }

    return ret_ob;
}

#endif

#if defined(HEYOKA_HAVE_REAL)

bool type_caster<mppp::real>::load(handle src, bool)
{
    namespace heypy = heyoka_py;

    if (!heypy::py_real_check(src.ptr())) {
        return false;
    }

    value = *heypy::get_real_val(src.ptr());

    return true;
}

handle type_caster<mppp::real>::cast(const mppp::real &src, return_value_policy, handle)
{
    namespace heypy = heyoka_py;

    auto *ret_ob = heypy::pyreal_from_real(src);
    if (ret_ob == nullptr) {
        heyoka_py::py_throw(PyExc_RuntimeError, "Unable to obtain storage for a real object");
    }

    return ret_ob;
}

#endif

} // namespace pybind11::detail

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
