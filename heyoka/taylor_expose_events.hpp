// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_TAYLOR_EXPOSE_EVENTS_HPP
#define HEYOKA_PY_TAYLOR_EXPOSE_EVENTS_HPP

#include <pybind11/pybind11.h>

#include <heyoka/config.hpp>

namespace heyoka_py
{

namespace py = pybind11;

void expose_taylor_t_event_flt(py::module &);
void expose_taylor_t_event_dbl(py::module &);
void expose_taylor_t_event_ldbl(py::module &);

#if defined(HEYOKA_HAVE_REAL128)

void expose_taylor_t_event_f128(py::module &);

#endif

#if defined(HEYOKA_HAVE_REAL)

void expose_taylor_t_event_real(py::module &);

#endif

void expose_taylor_nt_event_flt(py::module &);
void expose_taylor_nt_event_dbl(py::module &);
void expose_taylor_nt_event_ldbl(py::module &);

#if defined(HEYOKA_HAVE_REAL128)

void expose_taylor_nt_event_f128(py::module &);

#endif

#if defined(HEYOKA_HAVE_REAL)

void expose_taylor_nt_event_real(py::module &);

#endif

// Batch mode (only float and double).
void expose_taylor_t_event_batch_flt(py::module &);
void expose_taylor_nt_event_batch_flt(py::module &);

void expose_taylor_t_event_batch_dbl(py::module &);
void expose_taylor_nt_event_batch_dbl(py::module &);

} // namespace heyoka_py

#endif
