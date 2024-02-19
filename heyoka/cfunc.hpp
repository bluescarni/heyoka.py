// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_CFUNC_HPP
#define HEYOKA_PY_CFUNC_HPP

#include <pybind11/pybind11.h>

#include <heyoka/config.hpp>

namespace heyoka_py
{

namespace py = pybind11;

void expose_add_cfunc_flt(py::module &);
void expose_add_cfunc_dbl(py::module &);
void expose_add_cfunc_ldbl(py::module &);

#if defined(HEYOKA_HAVE_REAL128)

void expose_add_cfunc_f128(py::module &);

#endif

#if defined(HEYOKA_HAVE_REAL)

void expose_add_cfunc_real(py::module &);

#endif

} // namespace heyoka_py

#endif
