// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/step_callback.hpp>

#include "common_utils.hpp"
#include "expose_step_callback_set.hpp"
#include "step_cb_wrapper.hpp"

namespace heyoka_py
{

namespace py = pybind11;

namespace detail
{

namespace
{

template <typename T, bool Batch>
void expose_step_callback_set_impl(py::module_ &m, const std::string &suffix)
{
    namespace hey = heyoka;

    using cb_t = std::conditional_t<Batch, hey::step_callback_batch<T>, hey::step_callback<T>>;
    using scb_t = std::conditional_t<Batch, hey::step_callback_batch_set<T>, hey::step_callback_set<T>>;

    constexpr auto scb_name = Batch ? "step_callback_batch_set_{}" : "step_callback_set_{}";

    py::class_<scb_t> scb_c(m, fmt::format(scb_name, suffix).c_str(), py::dynamic_attr{});

    // Constructor.
    scb_c.def(py::init([](const std::vector<py::object> &cbs) {
        std::vector<cb_t> v_cbs;
        v_cbs.reserve(cbs.size());

        for (const auto &cb : cbs) {
            v_cbs.emplace_back(step_cb_wrapper(cb));
        }

        return scb_t(std::move(v_cbs));
    }));

    // Size.
    scb_c.def("__len__", &scb_t::size);
}

} // namespace

} // namespace detail

void expose_step_callback_set(py::module_ &m)
{
    detail::expose_step_callback_set_impl<float, false>(m, "flt");
    detail::expose_step_callback_set_impl<double, false>(m, "dbl");
    detail::expose_step_callback_set_impl<long double, false>(m, "ldbl");

#if defined(HEYOKA_HAVE_REAL128)

    detail::expose_step_callback_set_impl<mppp::real128, false>(m, "f128");

#endif

#if defined(HEYOKA_HAVE_REAL)

    detail::expose_step_callback_set_impl<mppp::real, false>(m, "real");

#endif

    detail::expose_step_callback_set_impl<float, true>(m, "flt");
    detail::expose_step_callback_set_impl<double, true>(m, "dbl");
}

} // namespace heyoka_py
