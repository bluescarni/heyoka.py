// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>

#include <pybind11/pybind11.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include "common_utils.hpp"
#include "custom_casters.hpp"

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace pybind11::detail
{

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

} // namespace pybind11::detail

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif
