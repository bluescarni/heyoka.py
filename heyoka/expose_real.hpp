// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_EXPOSE_REAL_HPP
#define HEYOKA_PY_EXPOSE_REAL_HPP

#include <heyoka/config.hpp>

#include <pybind11/pybind11.h>

#if defined(HEYOKA_HAVE_REAL)

#include <pybind11/numpy.h>

#include <Python.h>

#include <mp++/real.hpp>

#endif

namespace heyoka_py
{

namespace py = pybind11;

#if defined(HEYOKA_HAVE_REAL)

// The Python type that will represent mppp::real.
struct py_real {
    PyObject_HEAD alignas(mppp::real) unsigned char m_storage[sizeof(mppp::real)];
};

// The Python type descriptor for py_real.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern PyTypeObject py_real_type;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern int npy_registered_py_real;

bool py_real_check(PyObject *);
mppp::real *get_real_val(PyObject *);
PyObject *pyreal_from_real(const mppp::real &);
PyObject *pyreal_from_real(mppp::real &&);

void pyreal_check_array(const py::array &, mpfr_prec_t = 0);
void pyreal_ensure_array(py::array &, mpfr_prec_t);

#endif

void expose_real(py::module_ &);

} // namespace heyoka_py

#endif
