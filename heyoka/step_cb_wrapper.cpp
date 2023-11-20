// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <utility>

#include <fmt/core.h>

#include <pybind11/pybind11.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/taylor.hpp>

#include "common_utils.hpp"
#include "step_cb_wrapper.hpp"

namespace heyoka_py
{

namespace py = pybind11;

// NOTE: m_obj will be a reference to the original Python object.
// Unlike the event callbacks we don't want to do any deep copy
// here - in ensemble propagations, we will be manually deep-copying
// the Python callbacks before invoking the propagate_*() member
// functions.
step_cb_wrapper::step_cb_wrapper(py::object obj) : m_obj(std::move(obj))
{
    if (!callable(m_obj)) {
        py_throw(PyExc_TypeError,
                 fmt::format("An object of type '{}' cannot be used as a step callback because it is not callable",
                             str(type(m_obj)))
                     .c_str());
    }
}

template <typename TA>
bool step_cb_wrapper::operator()(TA &ta)
{
    py::gil_scoped_acquire acquire;

    // NOTE: this will return a reference
    // to the Python object wrapping ta.
    auto ta_obj = py::cast(&ta);

    // Attempt to invoke the call operator of the
    // Pythonic callback.
    auto ret = m_obj(ta_obj);

    // NOTE: we want to manually check the conversion
    // of the return value because if that fails
    // the pybind11 error message is not very helpful, and thus
    // we try to provide a more detailed error message.
    try {
        return py::cast<bool>(ret);
    } catch (const py::cast_error &) {
        py_throw(PyExc_TypeError, (fmt::format("The call operator of a step callback is expected to return a boolean, "
                                               "but a value of type '{}' was returned instead",
                                               str(type(ret))))
                                      .c_str());
    }
}

template <typename TA>
void step_cb_wrapper::pre_hook(TA &ta)
{
    py::gil_scoped_acquire acquire;

    if (py::hasattr(m_obj, "pre_hook")) {
        auto ta_obj = py::cast(&ta);

        m_obj.attr("pre_hook")(ta_obj);
    }
}

// Explicit instantiations.

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<float> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<float> &);

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<double> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<double> &);

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<long double> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<mppp::real128> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<mppp::real128> &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<mppp::real> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<mppp::real> &);

#endif

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive_batch<float> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive_batch<float> &);

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive_batch<double> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive_batch<double> &);

} // namespace heyoka_py
