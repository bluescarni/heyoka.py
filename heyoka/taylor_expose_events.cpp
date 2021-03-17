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

    struct callback_wrapper {
        callback_wrapper(py::object obj) : m_obj(std::move(obj)) {}

        callback_wrapper(callback_wrapper &&) noexcept = default;
        // NOTE: the purpose of this is to enforce deep copy semantics
        // when a copy of the wrapper is requested.
        callback_wrapper(const callback_wrapper &c) : m_obj(py::module_::import("copy").attr("deepcopy")(c.m_obj)) {}

        // Delete the rest.
        callback_wrapper &operator=(const callback_wrapper &) = delete;
        callback_wrapper &operator=(callback_wrapper &&) noexcept = delete;

        void operator()(hey::taylor_adaptive<T> &ta, T time) const
        {
            // Make sure we lock the GIL before calling into the
            // interpreter, as the callbacks may be invoked in long-running
            // propagate functions which release the GIL.
            py::gil_scoped_acquire acquire;

            py::cast<callback_t>(m_obj)(ta, time);
        }

        py::object m_obj;
    };

    py::class_<ev_t>(m, ("_nt_event_{}"_format(suffix)).c_str())
        .def(py::init([](hey::expression ex, py::object callback, hey::event_direction dir) {
                 if (!heypy::callable(callback)) {
                     heypy::py_throw(PyExc_TypeError,
                                     "Cannot create a non-terminal event with a callback of type '{}', "
                                     "which is not callable"_format(heypy::str(heypy::type(callback)))
                                         .c_str());
                 }

                 return ev_t(std::move(ex), callback_wrapper(std::move(callback)), dir);
             }),
             "expression"_a, "callback"_a, "direction"_a = hey::event_direction::any)
        .def_property_readonly("expression", &ev_t::get_expression)
        .def_property_readonly("callback", &ev_t::get_callback)
        .def_property_readonly("direction", &ev_t::get_direction)
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

    struct callback_wrapper {
        callback_wrapper(py::object obj) : m_obj(std::move(obj)) {}

        callback_wrapper(callback_wrapper &&) noexcept = default;
        // NOTE: the purpose of this is to enforce deep copy semantics
        // when a copy of the wrapper is requested.
        callback_wrapper(const callback_wrapper &c) : m_obj(py::module_::import("copy").attr("deepcopy")(c.m_obj)) {}

        // Delete the rest.
        callback_wrapper &operator=(const callback_wrapper &) = delete;
        callback_wrapper &operator=(callback_wrapper &&) noexcept = delete;

        void operator()(hey::taylor_adaptive<T> &ta, T time, bool mr) const
        {
            // Make sure we lock the GIL before calling into the
            // interpreter, as the callbacks may be invoked in long-running
            // propagate functions which release the GIL.
            py::gil_scoped_acquire acquire;

            py::cast<callback_t>(m_obj)(ta, time, mr);
        }

        py::object m_obj;
    };

    py::class_<ev_t>(m, ("_t_event_{}"_format(suffix)).c_str())
        .def(py::init([](hey::expression ex, py::object callback, hey::event_direction dir, T cooldown) {
                 if (callback.is_none()) {
                     return ev_t(std::move(ex), kw::direction = dir, kw::cooldown = cooldown);
                 } else {
                     if (!heypy::callable(callback)) {
                         heypy::py_throw(PyExc_TypeError,
                                         "Cannot create a terminal event with a callback of type '{}', "
                                         "which is not callable"_format(heypy::str(heypy::type(callback)))
                                             .c_str());
                     }

                     return ev_t(std::move(ex), kw::callback = callback_wrapper(std::move(callback)),
                                 kw::direction = dir, kw::cooldown = cooldown);
                 }
             }),
             "expression"_a, "callback"_a = py::none{}, "direction"_a = hey::event_direction::any, "cooldown"_a = T(-1))
        .def_property_readonly("expression", &ev_t::get_expression)
        .def_property_readonly("callback", &ev_t::get_callback)
        .def_property_readonly("direction", &ev_t::get_direction)
        .def_property_readonly("cooldown", &ev_t::get_cooldown)
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
