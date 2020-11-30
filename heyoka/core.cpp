// Copyright 2019-2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the obake.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>

namespace py = pybind11;
namespace hey = heyoka;

namespace heyoka_py
{

namespace
{

py::object builtins()
{
    return py::module::import("builtins");
}

py::object type(const py::object &o)
{
    return builtins().attr("type")(o);
}

std::string str(const py::object &o)
{
    return py::cast<std::string>(py::str(o));
}

[[noreturn]] void py_throw(PyObject *type, const char *msg)
{
    PyErr_SetString(type, msg);
    throw py::error_already_set();
}

bool is_numpy_ld(const py::object &o)
{
    return type(o).is(py::module::import("numpy").attr("longdouble"));
}

long double from_numpy_ld(const py::object &o)
{
    assert(is_numpy_ld(o));

    py::bytes b = o.attr("data").attr("tobytes")();
    const auto str = static_cast<std::string>(b);

    if (str.size() != sizeof(long double)) {
        throw std::runtime_error(
            "error while converting a numpy.longdouble to a C++ long double: the size of the bytes array ("
            + std::to_string(str.size()) + ") does not match the size of the long double type ("
            + std::to_string(sizeof(long double)) + ")");
    }

    long double retval;
    std::transform(str.begin(), str.end(), reinterpret_cast<unsigned char *>(&retval),
                   [](auto c) { return static_cast<unsigned char>(c); });

    return retval;
}

} // namespace

} // namespace heyoka_py

namespace heypy = heyoka_py;

PYBIND11_MODULE(core, m)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Init the pybind11 integration for this module.
    mppp_pybind11::init();
#endif

    py::class_<hey::expression>(m, "expression")
        .def(py::init<>())
        .def(py::init<double>())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::init<mppp::real128>())
#endif
        .def(py::init<std::string>())
        .def(py::init([](const py::object &o) {
            if (heypy::is_numpy_ld(o)) {
                return hey::expression{heypy::from_numpy_ld(o)};
            } else {
                heypy::py_throw(PyExc_TypeError, ("cannot construct an expression from an object of type \""
                                                  + heypy::str(heypy::type(o)) + "\"")
                                                     .c_str());
            }
        }))
        .def(py::self + py::self)
        .def(py::self + double())
        .def(double() + py::self)
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self + mppp::real128())
        .def(mppp::real128() + py::self)
#endif
        .def(
            "__add__",
            [](const hey::expression &ex, const py::object &o) {
                if (heypy::is_numpy_ld(o)) {
                    return ex + heypy::from_numpy_ld(o);
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot add an expression to an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\"")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def(
            "__add__",
            [](const py::object &o, const hey::expression &ex) {
                if (heypy::is_numpy_ld(o)) {
                    return heypy::from_numpy_ld(o) + ex;
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot add an expression to an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\"")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def("__repr__", [](const hey::expression &e) {
            std::ostringstream oss;
            oss << e;
            return oss.str();
        });
}
