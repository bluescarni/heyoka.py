// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>
#include <vector>

#include <fmt/core.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/callbacks.hpp>
#include <heyoka/expression.hpp>

#include "common_utils.hpp"
#include "expose_callbacks.hpp"
#include "pickle_wrappers.hpp"

namespace heyoka_py
{

namespace py = pybind11;

namespace detail
{

namespace
{

// NOTE: possible improvements:
// - expose operator()/pre_hook()? Might be handy in step-by-step
//   integrations. If we do it we probably need to expose them
//   *after* having exposed the integrators, otherwise we have the usual
//   issue of the signatures with the "wrong" names.
template <typename StepCallback>
py::class_<StepCallback> expose_step_callback(py::module_ &m, const char *name)
{
    // NOLINTNEXTLINE(google-build-using-namespace)
    using namespace pybind11::literals;

    py::class_<StepCallback> ret(m, fmt::format("_callback_{}", name).c_str(), py::dynamic_attr{});
    ret.def(py::init<>());
    // Repr.
    ret.def("__repr__", [](const StepCallback &cb) {
        std::ostringstream oss;
        oss << cb;
        return oss.str();
    });
    // Copy/deepcopy.
    ret.def("__copy__", copy_wrapper<StepCallback>);
    ret.def("__deepcopy__", deepcopy_wrapper<StepCallback>, "memo"_a);
    // Pickle support.
    ret.def(py::pickle(&pickle_getstate_wrapper<StepCallback>, &pickle_setstate_wrapper<StepCallback>));

    return ret;
}

} // namespace

} // namespace detail

void expose_callbacks(py::module_ &m)
{
    // Angle reducer.
    auto ar_class = detail::expose_step_callback<heyoka::callback::angle_reducer>(m, "angle_reducer");
    ar_class.def(py::init<std::vector<heyoka::expression>>());
}

} // namespace heyoka_py
