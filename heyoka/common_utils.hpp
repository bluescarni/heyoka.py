// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_COMMON_UTILS_HPP
#define HEYOKA_PY_COMMON_UTILS_HPP

#include <string>

#include <pybind11/pybind11.h>

#include <Python.h>

namespace heyoka_py
{

namespace py = pybind11;

py::object builtins();

py::object type(const py::object &);

std::string str(const py::object &);

[[noreturn]] void py_throw(PyObject *, const char *);

bool is_numpy_ld(const py::object &);

long double from_numpy_ld(const py::object &);

} // namespace heyoka_py

#endif
