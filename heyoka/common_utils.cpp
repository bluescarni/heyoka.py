// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

#include <Python.h>

#include "common_utils.hpp"

namespace heyoka_py
{

py::object builtins()
{
    return py::module::import("builtins");
}

py::object type(const py::object &o)
{
    return builtins().attr("type")(o);
}

std::string str(const py::object &o)
{
    return py::cast<std::string>(py::str(o));
}

[[noreturn]] void py_throw(PyObject *type, const char *msg)
{
    PyErr_SetString(type, msg);
    throw py::error_already_set();
}

bool is_numpy_ld(const py::object &o)
{
    return type(o).is(py::module::import("numpy").attr("longdouble"));
}

long double from_numpy_ld(const py::object &o)
{
    assert(is_numpy_ld(o));

    py::bytes b = o.attr("data").attr("tobytes")();
    const auto str = static_cast<std::string>(b);

    if (str.size() != sizeof(long double)) {
        throw std::runtime_error(
            "error while converting a numpy.longdouble to a C++ long double: the size of the bytes array ("
            + std::to_string(str.size()) + ") does not match the size of the long double type ("
            + std::to_string(sizeof(long double)) + ")");
    }

    long double retval;
    std::transform(str.begin(), str.end(), reinterpret_cast<unsigned char *>(&retval),
                   [](auto c) { return static_cast<unsigned char>(c); });

    return retval;
}

} // namespace heyoka_py
