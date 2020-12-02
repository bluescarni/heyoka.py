// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <string>

#include <pybind11/pybind11.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/number.hpp>

#include "common_utils.hpp"
#include "long_double_caster.hpp"

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

} // namespace heyoka_py
