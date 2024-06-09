// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <utility>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/var_ode_sys.hpp>

#include "common_utils.hpp"
#include "expose_var_ode_sys.hpp"
#include "pickle_wrappers.hpp"

namespace heyoka_py
{

namespace py = pybind11;

void expose_var_ode_sys(py::module_ &m)
{
    namespace hey = heyoka;
    // NOLINTNEXTLINE(google-build-using-namespace)
    using namespace pybind11::literals;

    // var_args enum.
    py::enum_<hey::var_args>(m, "var_args", py::arithmetic())
        .value("vars", hey::var_args::vars)
        .value("params", hey::var_args::params)
        .value("time", hey::var_args::time)
        .value("all", hey::var_args::all)
        .def("__or__", [](hey::var_args a, hey::var_args b) { return a | b; })
        .def("__and__", [](hey::var_args a, hey::var_args b) { return a & b; });

    // var_ode_sys class.
    py::class_<hey::var_ode_sys> v_cl(m, "var_ode_sys", py::dynamic_attr{});
    v_cl.def(py::init([](const std::vector<std::pair<hey::expression, hey::expression>> &sys,
                         const std::variant<hey::var_args, std::vector<hey::expression>> &args,
                         std::uint32_t order) { return hey::var_ode_sys(sys, args, order); }),
             "sys"_a, "args"_a, "order"_a = static_cast<std::uint32_t>(1));
    v_cl.def_property_readonly("sys", &hey::var_ode_sys::get_sys);
    v_cl.def_property_readonly("vargs", &hey::var_ode_sys::get_vargs);
    v_cl.def_property_readonly("n_orig_sv", &hey::var_ode_sys::get_n_orig_sv);
    v_cl.def_property_readonly("order", &hey::var_ode_sys::get_order);
    // Copy/deepcopy.
    v_cl.def("__copy__", copy_wrapper<hey::var_ode_sys>);
    v_cl.def("__deepcopy__", deepcopy_wrapper<hey::var_ode_sys>, "memo"_a);
    // Pickle support.
    v_cl.def(py::pickle(&pickle_getstate_wrapper<hey::var_ode_sys>, &pickle_setstate_wrapper<hey::var_ode_sys>));
}

} // namespace heyoka_py
