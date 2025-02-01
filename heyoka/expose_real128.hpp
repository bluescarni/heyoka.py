// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_EXPOSE_REAL_128_HPP
#define HEYOKA_PY_EXPOSE_REAL_128_HPP

#include <heyoka/config.hpp>

#include <pybind11/pybind11.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <Python.h>

#include <mp++/real128.hpp>

#endif

namespace heyoka_py
{

#if defined(HEYOKA_HAVE_REAL128)

// The Python type that will represent mppp::real128.
struct py_real128 {
    PyObject_HEAD alignas(mppp::real128) unsigned char m_storage[sizeof(mppp::real128)];
};

// The Python type descriptor for py_real128.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern PyTypeObject py_real128_type;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern int npy_registered_py_real128;

bool py_real128_check(PyObject *);
mppp::real128 *get_real128_val(PyObject *);
PyObject *pyreal128_from_real128(const mppp::real128 &);

#endif

namespace py = pybind11;

void expose_real128(py::module_ &);

} // namespace heyoka_py

#endif
