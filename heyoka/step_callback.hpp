// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_STEP_CALLBACK_HPP
#define HEYOKA_PY_STEP_CALLBACK_HPP

#include <pybind11/pybind11.h>

namespace heyoka_py
{

namespace py = pybind11;

// NOTE: the purpose of this wrapper is to enforce
// GIL acquisition when invoking the call operator
// or the pre_hook() function. This is needed because
// the GIL is released during the invocation of the
// propagate_*() functions. See also the event callback
// wrapper, which does the same thing.
class step_cb_wrapper
{
    py::object m_obj;

public:
    explicit step_cb_wrapper(py::object);

    template <typename TA>
    bool operator()(TA &);

    template <typename TA>
    void pre_hook(TA &);
};

} // namespace heyoka_py

#endif
