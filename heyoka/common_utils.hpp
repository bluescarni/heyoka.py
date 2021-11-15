// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <heyoka/number.hpp>

namespace heyoka_py
{

namespace py = pybind11;

py::object builtins();

py::object type(const py::handle &);

std::string str(const py::handle &);

[[noreturn]] void py_throw(PyObject *, const char *);

heyoka::number to_number(const py::handle &);

bool callable(const py::handle &);

bool mpmath_available();

struct scoped_quadprec_setter {
    scoped_quadprec_setter();
    ~scoped_quadprec_setter();

    bool has_mpmath;
    int orig_prec = 0;
};

// Helper to expose the llvm_state getter
// for a generic object.
template <typename T>
inline void expose_llvm_state_property(py::class_<T> &c)
{
    c.def_property_readonly("llvm_state", &T::get_llvm_state);
}

} // namespace heyoka_py

#endif
