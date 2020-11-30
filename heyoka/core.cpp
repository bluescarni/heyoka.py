// Copyright 2019-2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the obake.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>

#include "common_utils.hpp"

namespace py = pybind11;
namespace hey = heyoka;

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
        // Unary operators.
        .def(-py::self)
        .def(+py::self)
        // Binary operators.
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
        .def(py::self - py::self)
        .def(py::self - double())
        .def(double() - py::self)
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self - mppp::real128())
        .def(mppp::real128() - py::self)
#endif
        .def(
            "__sub__",
            [](const hey::expression &ex, const py::object &o) {
                if (heypy::is_numpy_ld(o)) {
                    return ex - heypy::from_numpy_ld(o);
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot subtract an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\" from an expression")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def(
            "__sub__",
            [](const py::object &o, const hey::expression &ex) {
                if (heypy::is_numpy_ld(o)) {
                    return heypy::from_numpy_ld(o) - ex;
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot subtract an expression from an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\"")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self)
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self * mppp::real128())
        .def(mppp::real128() * py::self)
#endif
        .def(
            "__mul__",
            [](const hey::expression &ex, const py::object &o) {
                if (heypy::is_numpy_ld(o)) {
                    return ex * heypy::from_numpy_ld(o);
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot multiply an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\" by an expression")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const py::object &o, const hey::expression &ex) {
                if (heypy::is_numpy_ld(o)) {
                    return heypy::from_numpy_ld(o) * ex;
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot multiply an expression by an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\"")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def(py::self / py::self)
        .def(py::self / double())
        .def(double() / py::self)
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self / mppp::real128())
        .def(mppp::real128() / py::self)
#endif
        .def(
            "__div__",
            [](const hey::expression &ex, const py::object &o) {
                if (heypy::is_numpy_ld(o)) {
                    return ex / heypy::from_numpy_ld(o);
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot divide an expression by an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\"")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def(
            "__div__",
            [](const py::object &o, const hey::expression &ex) {
                if (heypy::is_numpy_ld(o)) {
                    return heypy::from_numpy_ld(o) / ex;
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot divide an object of type \"" + heypy::str(heypy::type(o))
                                                      + "\" by an expression")
                                                         .c_str());
                }
            },
            py::is_operator())
        // In-place operators.
        .def(py::self += py::self)
        .def(py::self += double())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self += mppp::real128())
#endif
        .def(
            "__iadd__",
            [](hey::expression &ex, const py::object &o) -> hey::expression & {
                if (heypy::is_numpy_ld(o)) {
                    return ex += heypy::from_numpy_ld(o);
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot add in-place an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\" to an expression")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def(py::self -= py::self)
        .def(py::self -= double())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self -= mppp::real128())
#endif
        .def(
            "__isub__",
            [](hey::expression &ex, const py::object &o) -> hey::expression & {
                if (heypy::is_numpy_ld(o)) {
                    return ex -= heypy::from_numpy_ld(o);
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot subtract in-place an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\" from an expression")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def(py::self *= py::self)
        .def(py::self *= double())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self *= mppp::real128())
#endif
        .def(
            "__imul__",
            [](hey::expression &ex, const py::object &o) -> hey::expression & {
                if (heypy::is_numpy_ld(o)) {
                    return ex *= heypy::from_numpy_ld(o);
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot multiply in-place an expression by an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\"")
                                                         .c_str());
                }
            },
            py::is_operator())
        .def(py::self /= py::self)
        .def(py::self /= double())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self /= mppp::real128())
#endif
        .def(
            "__idiv__",
            [](hey::expression &ex, const py::object &o) -> hey::expression & {
                if (heypy::is_numpy_ld(o)) {
                    return ex /= heypy::from_numpy_ld(o);
                } else {
                    heypy::py_throw(PyExc_TypeError, ("cannot divide in-place an expression by an object of type \""
                                                      + heypy::str(heypy::type(o)) + "\"")
                                                         .c_str());
                }
            },
            py::is_operator())
        // Comparisons.
        .def(py::self == py::self)
        .def(py::self != py::self)
        // Repr.
        .def("__repr__", [](const hey::expression &e) {
            std::ostringstream oss;
            oss << e;
            return oss.str();
        });

    m.def("pairwise_sum", [](const std::vector<hey::expression> &v_ex) { return hey::pairwise_sum(v_ex); });

    m.def("make_vars", [](py::args v_str) {
        std::vector<hey::expression> retval;
        std::transform(v_str.begin(), v_str.end(), std::back_inserter(retval),
                       [](const auto &o) { return hey::expression(py::cast<std::string>(o)); });
        return retval;
    });
}
