// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <sstream>
#include <string>
#include <utility>

#include <fmt/format.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/taylor.hpp>

#include "common_utils.hpp"
#include "long_double_caster.hpp"
#include "taylor_expose_events.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;
namespace heypy = heyoka_py;

namespace detail
{

namespace
{

// Helper to expose non-terminal events.
template <typename T>
void expose_taylor_nt_event_impl(py::module &m, const std::string &suffix)
{
    using namespace pybind11::literals;
    using fmt::literals::operator""_format;

    using ev_t = hey::nt_event<T>;
    using callback_t = typename ev_t::callback_t;

    py::class_<ev_t>(m, ("_nt_event_{}"_format(suffix)).c_str())
        .def(py::init([](hey::expression ex, callback_t callback, hey::event_direction dir) {
                 auto cbl = [cb = std::move(callback)](hey::taylor_adaptive<T> &ta, T time) {
                     // Make sure we lock the GIL before calling into the
                     // interpreter, as the callbacks may be invoked in long-running
                     // propagate functions which release the GIL.
                     py::gil_scoped_acquire acquire;

                     cb(ta, time);
                 };

                 return ev_t(std::move(ex), std::move(cbl), dir);
             }),
             "expression"_a, "callback"_a, "direction"_a = hey::event_direction::any)
        // Repr.
        .def("__repr__",
             [](const ev_t &e) {
                 std::ostringstream oss;
                 oss << e;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", [](const ev_t &e) { return e; })
        .def(
            "__deepcopy__", [](const ev_t &e, py::dict) { return e; }, "memo"_a);
}

// Helper to expose terminal events.
template <typename T>
void expose_taylor_t_event_impl(py::module &m, const std::string &suffix)
{
    using namespace pybind11::literals;
    using fmt::literals::operator""_format;
    namespace kw = hey::kw;

    using ev_t = hey::t_event<T>;
    using callback_t = typename ev_t::callback_t;

    py::class_<ev_t>(m, ("_t_event_{}"_format(suffix)).c_str())
        .def(py::init([](hey::expression ex, callback_t callback, hey::event_direction dir, T cooldown) {
                 if (callback) {
                     auto cbl = [cb = std::move(callback)](hey::taylor_adaptive<T> &ta, T time, bool mr) {
                         // Make sure we lock the GIL before calling into the
                         // interpreter, as the callbacks may be invoked in long-running
                         // propagate functions which release the GIL.
                         py::gil_scoped_acquire acquire;

                         cb(ta, time, mr);
                     };

                     return ev_t(std::move(ex), kw::callback = std::move(cbl), kw::direction = dir,
                                 kw::cooldown = cooldown);
                 } else {
                     return ev_t(std::move(ex), kw::direction = dir, kw::cooldown = cooldown);
                 }
             }),
             "expression"_a, "callback"_a = py::none{}, "direction"_a = hey::event_direction::any, "cooldown"_a = T(-1))
        // Repr.
        .def("__repr__",
             [](const ev_t &e) {
                 std::ostringstream oss;
                 oss << e;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", [](const ev_t &e) { return e; })
        .def(
            "__deepcopy__", [](const ev_t &e, py::dict) { return e; }, "memo"_a);
}

} // namespace

} // namespace detail

void expose_taylor_t_event_dbl(py::module &m)
{
    detail::expose_taylor_t_event_impl<double>(m, "dbl");
}

void expose_taylor_t_event_ldbl(py::module &m)
{
    detail::expose_taylor_t_event_impl<long double>(m, "ldbl");
}

#if defined(HEYOKA_HAVE_REAL128)

void expose_taylor_t_event_f128(py::module &m)
{
    // NOTE: we need to temporarily alter
    // the precision in mpmath to successfully
    // construct the default values of the parameters
    // for the constructor.
    scoped_quadprec_setter qs;

    detail::expose_taylor_t_event_impl<mppp::real128>(m, "f128");
}

#endif

void expose_taylor_nt_event_dbl(py::module &m)
{
    detail::expose_taylor_nt_event_impl<double>(m, "dbl");
}

void expose_taylor_nt_event_ldbl(py::module &m)
{
    detail::expose_taylor_nt_event_impl<long double>(m, "ldbl");
}

#if defined(HEYOKA_HAVE_REAL128)

void expose_taylor_nt_event_f128(py::module &m)
{
    // NOTE: we need to temporarily alter
    // the precision in mpmath to successfully
    // construct the default values of the parameters
    // for the constructor.
    scoped_quadprec_setter qs;

    detail::expose_taylor_nt_event_impl<mppp::real128>(m, "f128");
}

#endif

} // namespace heyoka_py
