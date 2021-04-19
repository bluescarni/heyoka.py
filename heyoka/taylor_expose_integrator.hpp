// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_TAYLOR_EXPOSE_INTEGRATOR_HPP
#define HEYOKA_PY_TAYLOR_EXPOSE_INTEGRATOR_HPP

#include <functional>
#include <utility>

#include <pybind11/pybind11.h>

#include <heyoka/config.hpp>

namespace heyoka_py
{

namespace py = pybind11;

void expose_taylor_integrator_dbl(py::module &);
void expose_taylor_integrator_ldbl(py::module &);

#if defined(HEYOKA_HAVE_REAL128)

void expose_taylor_integrator_f128(py::module &);

#endif

// NOTE: this helper wraps a callback for the propagate_*()
// functions ensuring that the GIL is acquired before invoking the callback.
// Additionally, the returned wrapper will contain a const reference to the
// original callback. This ensures that copying the wrapper does not
// copy the original callback, so that copying the wrapper
// never ends up calling into the Python interpreter.
// If cb is an empty callback, a copy of cb will be returned.
template <typename T>
inline auto make_prop_cb(const std::function<void(T &)> &cb)
{
    if (cb) {
        auto ret = [&cb](T &ta) {
            py::gil_scoped_acquire acquire;

            cb(ta);
        };

        return std::function<void(T &)>(std::move(ret));
    } else {
        return cb;
    }
}

} // namespace heyoka_py

#endif
