// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <exception>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
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

// Detect if mpmath is available.
bool mpmath_available()
{
    try {
        py::module_::import("mpmath");
    } catch (...) {
        return false;
    }

    return true;
}

// RAII helper to temporarily change the mpmath precision
// to 113 bits. The original precision will be restored upon
// destruction.
scoped_quadprec_setter::scoped_quadprec_setter() : has_mpmath(mpmath_available())
{
    if (has_mpmath) {
        auto mpmod = py::module_::import("mpmath");

        orig_prec = py::cast<int>(mpmod.attr("mp").attr("prec"));
        mpmod.attr("mp").attr("prec") = 113;
    }
}

scoped_quadprec_setter::~scoped_quadprec_setter()
{
    if (has_mpmath) {
        // Restore the original precision.
        py::module_::import("mpmath").attr("mp").attr("prec") = orig_prec;
    }
}

} // namespace heyoka_py
