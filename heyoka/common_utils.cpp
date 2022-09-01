// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <exception>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/number.hpp>

#include "common_utils.hpp"
#include "custom_casters.hpp"

namespace heyoka_py
{

py::object builtins()
{
    return py::module_::import("builtins");
}

py::object type(const py::handle &o)
{
    return builtins().attr("type")(o);
}

std::string str(const py::handle &o)
{
    return py::cast<std::string>(py::str(o));
}

void py_throw(PyObject *type, const char *msg)
{
    PyErr_SetString(type, msg);
    throw py::error_already_set();
}

heyoka::number to_number(const py::handle &o)
{
    // NOTE: investigate if these try/catch
    // blocks can be replaced by something more
    // efficient.
    // NOTE: the real128/long double casters
    // will fail if o is not exactly of type
    // real128/long double. The caster to double
    // will be more tolerant.
#if defined(HEYOKA_HAVE_REAL128)
    try {
        return heyoka::number{py::cast<mppp::real128>(o)};
    } catch (const py::cast_error &) {
    }
#endif

    try {
        return heyoka::number{py::cast<long double>(o)};
    } catch (const py::cast_error &) {
    }

    return heyoka::number{py::cast<double>(o)};
}

// Detect if o is a callable object.
bool callable(const py::handle &o)
{
    return py::cast<bool>(builtins().attr("callable")(o));
}

bool may_share_memory(const py::array &a, const py::array &b)
{
    return py::module_::import("numpy").attr("may_share_memory")(a, b).cast<bool>();
}

// Helper to check if a numpy array is a NPY_ARRAY_CARRAY (i.e., C-style
// contiguous and with properly aligned storage). The flag signals whether
// the array must also be writeable or not.
bool is_npy_array_carray(const py::array &arr, bool writeable)
{
    assert(PyObject_IsInstance(arr.ptr(), reinterpret_cast<PyObject *>(&PyArray_Type)));

    // NOTE: NPY_ARRAY_CARRAY is NPY_ARRAY_CARRAY_RO + writeable flag.
    if (writeable) {
        return PyArray_CHKFLAGS(reinterpret_cast<const PyArrayObject *>(arr.ptr()), NPY_ARRAY_CARRAY) != 0;
    } else {
        return PyArray_CHKFLAGS(reinterpret_cast<const PyArrayObject *>(arr.ptr()), NPY_ARRAY_CARRAY_RO) != 0;
    }
}

} // namespace heyoka_py
