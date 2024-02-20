// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_STEP_CB_UTILS_HPP
#define HEYOKA_PY_STEP_CB_UTILS_HPP

#include <variant>
#include <vector>

#include <pybind11/pybind11.h>

#include <heyoka/callbacks.hpp>

namespace heyoka_py
{

namespace py = pybind11;

// Wrapper for a python step callback.
// NOTE: the purpose of this wrapper is to enforce
// GIL acquisition when invoking the call operator
// or the pre_hook() function. This is needed because
// the GIL is released during the invocation of the
// propagate_*() functions. See also the event callback
// wrapper, which does the same thing.
struct step_cb_wrapper {
    // NOTE: it is *essential* here that we capture
    // the Python object by reference, rather than
    // by value. The reason for this is that in
    // the propagate_*() functions we may end up
    // calling the destructor of a step_cb_wrapper
    // if the call operator or pre_hook() throw
    // an exception. In such a case, we would be calling
    // into the Python interpreter without holding the GIL.
    py::object *m_obj_ptr = nullptr;

    step_cb_wrapper() = delete;
    explicit step_cb_wrapper(py::object *);
    step_cb_wrapper(const step_cb_wrapper &);
    step_cb_wrapper(step_cb_wrapper &&) noexcept;
    step_cb_wrapper &operator=(const step_cb_wrapper &) = delete;
    step_cb_wrapper &operator=(step_cb_wrapper &&) noexcept = delete;
    ~step_cb_wrapper();

    template <typename TA>
    bool operator()(TA &);

    template <typename TA>
    void pre_hook(TA &);
};

// Union of step callback types.
using scb_t = std::variant<heyoka::callback::angle_reducer, py::object>;

// Union of types that can be used to pass a step callback
// to a propagate_*() function: either a single scb_t
// or a list of scb_t objects.
// NOTE: place the list first, otherwise a list argument
// will match the py::object in scb_t.
using scb_arg_t = std::variant<std::vector<scb_t>, scb_t>;

// Helper to turn an scb_arg_t into a step callback
// usable by the propagate_*() functions.
template <typename StepCallback>
StepCallback scb_arg_to_step_callback(scb_arg_t &);

// Helper to turn a step callback back into an scb_arg_t.
// The type stored in the callback is inferred by the active
// type(s) of the scb_arg_t argument.
template <typename StepCallback>
scb_arg_t step_callback_to_scb_arg_t(const scb_arg_t &, StepCallback &);

} // namespace heyoka_py

#endif
