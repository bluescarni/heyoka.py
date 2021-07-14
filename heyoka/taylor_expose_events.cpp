// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/serialization/binary_object.hpp>

#include <fmt/format.h>

#include <pybind11/pybind11.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#include "common_utils.hpp"
#include "long_double_caster.hpp"
#include "pickle_wrappers.hpp"
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

// A wrapper for event callbacks implemented in Python.
// The Python callback is stored as the m_obj data member.
// This wrapper accomplishes the following goals:
// - provide value semantics (i.e., the copy ctor
//   performs a deep copy),
// - ensure the GIL is acquired in the call operator,
// - provide serialisation capabilities.
template <typename Ret, typename... Args>
struct ev_callback {
    py::object m_obj;

    ev_callback() = default;
    explicit ev_callback(py::object o) : m_obj(std::move(o)) {}
    ev_callback(const ev_callback &c) : m_obj(py::module_::import("copy").attr("deepcopy")(c.m_obj)) {}
    ev_callback(ev_callback &&) noexcept = default;
    ev_callback &operator=(const ev_callback &c)
    {
        if (this != &c) {
            *this = ev_callback(c);
        }

        return *this;
    }
    ev_callback &operator=(ev_callback &&) noexcept = default;

    Ret operator()(Args... args) const
    {
        // Make sure we lock the GIL before calling into the
        // interpreter, as the callbacks may be invoked in long-running
        // propagate functions which release the GIL.
        py::gil_scoped_acquire acquire;

        // NOTE: the conversion of the input arguments to Python
        // objects should always work, because all callback arguments
        // are guaranteed to have conversions to Python. We want to manually
        // check the conversion of the return value because if that fails
        // the pybind11 error message is not very helpful, and thus
        // we try to provide a more detailed error message.

        if constexpr (std::is_same_v<void, Ret>) {
            m_obj(std::forward<Args>(args)...);
        } else {
            auto ret = m_obj(std::forward<Args>(args)...);

            try {
                return py::cast<Ret>(ret);
            } catch (const py::cast_error &) {
                using fmt::literals::operator""_format;

                py_throw(PyExc_TypeError, ("Unable to convert a Python object of type '{}' to the C++ type '{}' "
                                           "in the construction of the return value of an event callback"_format(
                                               heypy::str(heypy::type(ret)), boost::core::demangle(typeid(Ret).name())))
                                              .c_str());
            }
        }
    }

private:
    // Make the callback serialisable.
    friend class boost::serialization::access;
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        // Dump the Python callable into a bytes object.
        auto tmp = py::module::import("heyoka").attr("get_serialization_backend")().attr("dumps")(m_obj);

        // This gives a null-terminated char * to the internal
        // content of the bytes object.
        auto ptr = PyBytes_AsString(tmp.ptr());
        if (!ptr) {
            py_throw(PyExc_TypeError, "The serialization backend's dumps() function did not return a bytes object");
        }

        // NOTE: this will be the length of the bytes object *without* the terminator.
        const auto size = boost::numeric_cast<std::size_t>(py::len(tmp));

        // Save the binary size.
        ar << size;

        // Save the binary object.
        ar << boost::serialization::make_binary_object(ptr, size);
    }
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        // Recover the size.
        std::size_t size{};
        ar >> size;

        // Recover the binary object.
        std::vector<char> tmp;
        tmp.resize(boost::numeric_cast<decltype(tmp.size())>(size));
        ar >> boost::serialization::make_binary_object(tmp.data(), size);

        // Deserialise and assign.
        auto b = py::bytes(tmp.data(), boost::numeric_cast<py::size_t>(size));
        m_obj = py::module::import("heyoka").attr("get_serialization_backend")().attr("loads")(b);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

// Helper to expose non-terminal events.
template <typename T>
void expose_taylor_nt_event_impl(py::module &m, const std::string &suffix)
{
    using namespace pybind11::literals;
    using fmt::literals::operator""_format;
    namespace kw = hey::kw;

    using ev_t = hey::nt_event<T>;

    py::class_<ev_t>(m, ("_nt_event_{}"_format(suffix)).c_str())
        .def(py::init([](hey::expression ex, py::object callback, hey::event_direction dir) {
                 if (!heypy::callable(callback)) {
                     heypy::py_throw(
                         PyExc_TypeError,
                         "An object of type '{}' cannot be used as an event callback because it is not callable"_format(
                             heypy::str(heypy::type(callback)))
                             .c_str());
                 }

                 return ev_t(std::move(ex), ev_callback<void, hey::taylor_adaptive<T> &, T, int>{std::move(callback)},
                             kw::direction = dir);
             }),
             "expression"_a, "callback"_a, "direction"_a = hey::event_direction::any)
        // Repr.
        .def("__repr__",
             [](const ev_t &e) {
                 std::ostringstream oss;
                 oss << e;
                 return oss.str();
             })
        // Expression.
        .def_property_readonly("expression", &ev_t::get_expression)
        // Callback.
        .def_property_readonly(
            "callback",
            [](const ev_t &e) {
                const auto ptr
                    = e.get_callback().template extract<ev_callback<void, hey::taylor_adaptive<T> &, T, int>>();

                // NOTE: ptr should never be null, unless
                // the event was unpickled from a broken archive.
                assert(ptr);
                return ptr->m_obj;
            })
        // Direction.
        .def_property_readonly("direction", &ev_t::get_direction)
        // Copy/deepcopy.
        .def("__copy__", [](const ev_t &e) { return e; })
        .def(
            "__deepcopy__", [](const ev_t &e, py::dict) { return e; }, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pickle_getstate_wrapper<ev_t>, &pickle_setstate_wrapper<ev_t>));
}

// Helper to expose terminal events.
template <typename T>
void expose_taylor_t_event_impl(py::module &m, const std::string &suffix)
{
    using namespace pybind11::literals;
    using fmt::literals::operator""_format;
    namespace kw = hey::kw;

    using ev_t = hey::t_event<T>;

    py::class_<ev_t>(m, ("_t_event_{}"_format(suffix)).c_str())
        .def(
            py::init([](hey::expression ex, py::object callback, hey::event_direction dir, T cooldown) {
                if (callback.is_none()) {
                    return ev_t(std::move(ex), kw::direction = dir, kw::cooldown = cooldown);
                } else {
                    if (!heypy::callable(callback)) {
                        heypy::py_throw(
                            PyExc_TypeError,
                            "An object of type '{}' cannot be used as an event callback because it is not callable"_format(
                                heypy::str(heypy::type(callback)))
                                .c_str());
                    }

                    return ev_t(std::move(ex),
                                kw::callback
                                = ev_callback<bool, hey::taylor_adaptive<T> &, bool, int>{std::move(callback)},
                                kw::direction = dir, kw::cooldown = cooldown);
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
        // Expression.
        .def_property_readonly("expression", &ev_t::get_expression)
        // Callback.
        .def_property_readonly(
            "callback",
            [](const ev_t &e) -> py::object {
                const auto ptr
                    = e.get_callback().template extract<ev_callback<bool, hey::taylor_adaptive<T> &, bool, int>>();

                // NOTE: the callback could be empty, in which case
                // extraction returns a null pointer.
                if (ptr) {
                    return ptr->m_obj;
                } else {
                    return py::none{};
                }
            })
        // Direction.
        .def_property_readonly("direction", &ev_t::get_direction)
        // Cooldown.
        .def_property_readonly("cooldown", &ev_t::get_cooldown)
        // Copy/deepcopy.
        .def("__copy__", [](const ev_t &e) { return e; })
        .def(
            "__deepcopy__", [](const ev_t &e, py::dict) { return e; }, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pickle_getstate_wrapper<ev_t>, &pickle_setstate_wrapper<ev_t>));
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

// NOTE: create shortcuts for the event callback wrappers,
// because if we use their full name we ran into issues with the
// Boost.Serialization library complaining that the class name is too long.
using nt_cb_dbl = detail::ev_callback<void, heyoka::taylor_adaptive<double> &, double, int>;
using nt_cb_ldbl = detail::ev_callback<void, heyoka::taylor_adaptive<long double> &, long double, int>;

#if defined(HEYOKA_HAVE_REAL128)

using nt_cb_f128 = detail::ev_callback<void, heyoka::taylor_adaptive<mppp::real128> &, mppp::real128, int>;

#endif

using t_cb_dbl = detail::ev_callback<bool, heyoka::taylor_adaptive<double> &, bool, int>;
using t_cb_ldbl = detail::ev_callback<bool, heyoka::taylor_adaptive<long double> &, bool, int>;

#if defined(HEYOKA_HAVE_REAL128)

using t_cb_f128 = detail::ev_callback<bool, heyoka::taylor_adaptive<mppp::real128> &, bool, int>;

#endif

} // namespace heyoka_py

// Register the callback wrappers in the serialisation system.
HEYOKA_S11N_CALLABLE_EXPORT(heyoka_py::nt_cb_dbl, void, heyoka::taylor_adaptive<double> &, double, int)
HEYOKA_S11N_CALLABLE_EXPORT(heyoka_py::nt_cb_ldbl, void, heyoka::taylor_adaptive<long double> &, long double, int)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_CALLABLE_EXPORT(heyoka_py::nt_cb_f128, void, heyoka::taylor_adaptive<mppp::real128> &, mppp::real128, int)

#endif

HEYOKA_S11N_CALLABLE_EXPORT(heyoka_py::t_cb_dbl, bool, heyoka::taylor_adaptive<double> &, bool, int)
HEYOKA_S11N_CALLABLE_EXPORT(heyoka_py::t_cb_ldbl, bool, heyoka::taylor_adaptive<long double> &, bool, int)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_CALLABLE_EXPORT(heyoka_py::t_cb_f128, bool, heyoka::taylor_adaptive<mppp::real128> &, bool, int)

#endif
