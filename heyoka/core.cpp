// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/number.hpp>
#include <heyoka/square.hpp>
#include <heyoka/taylor.hpp>

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

    using namespace pybind11::literals;

    m.doc() = "The core heyoka module";

    // Flag the presence of real128.
    m.attr("with_real128") =
#if defined(HEYOKA_HAVE_REAL128)
        true
#else
        false
#endif
        ;

    // Export the heyoka version.
    m.attr("_heyoka_cpp_version_major") = HEYOKA_VERSION_MAJOR;
    m.attr("_heyoka_cpp_version_minor") = HEYOKA_VERSION_MINOR;
    m.attr("_heyoka_cpp_version_patch") = HEYOKA_VERSION_PATCH;

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
        // pow().
        .def("__pow__", [](const hey::expression &b, const hey::expression &e) { return hey::pow(b, e); })
        .def("__pow__", [](const hey::expression &b, double e) { return hey::pow(b, e); })
#if defined(HEYOKA_HAVE_REAL128)
        .def("__pow__", [](const hey::expression &b, mppp::real128 e) { return hey::pow(b, e); })
#endif
        .def("__pow__",
             [](const hey::expression &b, const py::object &e) {
                 if (heypy::is_numpy_ld(e)) {
                     return hey::pow(b, heypy::from_numpy_ld(e));
                 } else {
                     heypy::py_throw(PyExc_TypeError, ("cannot raise an expression to the power of an object of type \""
                                                       + heypy::str(heypy::type(e)) + "\"")
                                                          .c_str());
                 }
             })
        // Repr.
        .def("__repr__",
             [](const hey::expression &e) {
                 std::ostringstream oss;
                 oss << e;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", [](const hey::expression &e) { return e; })
        .def(
            "__deepcopy__", [](const hey::expression &e, py::dict) { return e; }, "memo"_a);

    // Pairwise sum.
    m.def("pairwise_sum", [](const std::vector<hey::expression> &v_ex) { return hey::pairwise_sum(v_ex); });

    // make_vars() helper.
    m.def("make_vars", [](py::args v_str) {
        py::list retval;
        for (const auto &o : v_str) {
            retval.append(hey::expression(py::cast<std::string>(o)));
        }
        return retval;
    });

    // Math functions.
    m.def("sin", &hey::sin);
    m.def("cos", &hey::cos);
    m.def("log", &hey::log);
    m.def("exp", &hey::exp);
    m.def("sqrt", &hey::sqrt);
    m.def("square", &hey::square);

    // N-body builder.
    m.def("make_nbody_sys", [](std::uint32_t n, py::kwargs kwargs) {
        if (kwargs) {
            // NOTE: Gconst's default value is 1, if not provided.
            const auto Gconst = kwargs.contains("Gconst") ? heypy::to_number(kwargs["Gconst"]) : hey::number{1.};

            std::vector<hey::number> m_vec;
            if (kwargs.contains("masses")) {
                auto masses = py::cast<py::iterable>(kwargs["masses"]);
                for (const auto &ms : masses) {
                    m_vec.push_back(heypy::to_number(ms));
                }
            } else {
                // If masses are not provided, all masses are 1.
                m_vec.resize(static_cast<decltype(m_vec.size())>(n), hey::number{1.});
            }

            namespace kw = hey::kw;
            return hey::make_nbody_sys(n, kw::Gconst = Gconst, kw::masses = m_vec);
        }

        return hey::make_nbody_sys(n);
    });

    // Adaptive taylor integrators.
    py::class_<hey::taylor_adaptive<double>>(m, "taylor_adaptive_double")
        .def(py::init(
            [](const std::vector<std::pair<hey::expression, hey::expression>> &sys, py::iterable state, py::kwargs) {
                // Build the initial state vector.
                std::vector<double> s_vector;
                for (const auto &x : state) {
                    s_vector.push_back(py::cast<double>(x));
                }

                return hey::taylor_adaptive<double>{sys, std::move(s_vector)};
            }))
        .def_property_readonly("state", [](py::object &o) {
            auto *ta = py::cast<hey::taylor_adaptive<double> *>(o);
            return py::array_t<double>({boost::numeric_cast<py::ssize_t>(ta->get_state().size())}, ta->get_state_data(),
                                       o);
        });

    py::class_<hey::taylor_adaptive<long double>>(m, "taylor_adaptive_long_double")
        .def(py::init(
            [](const std::vector<std::pair<hey::expression, hey::expression>> &sys, py::iterable state, py::kwargs) {
                // Build the initial state vector.
                std::vector<long double> s_vector;
                for (const auto &x : state) {
                    if (heypy::is_numpy_ld(x)) {
                        s_vector.push_back(heypy::from_numpy_ld(x));
                    } else {
                        s_vector.push_back(py::cast<double>(x));
                    }
                }

                return hey::taylor_adaptive<long double>{sys, std::move(s_vector)};
            }))
        .def_property_readonly("state", [](py::object &o) {
            auto *ta = py::cast<hey::taylor_adaptive<long double> *>(o);
            return py::array_t<long double>({boost::numeric_cast<py::ssize_t>(ta->get_state().size())},
                                            ta->get_state_data(), o);
        });

#if defined(HEYOKA_HAVE_REAL128)
    py::class_<hey::taylor_adaptive<mppp::real128>>(m, "taylor_adaptive_real128")
        .def(py::init(
            [](const std::vector<std::pair<hey::expression, hey::expression>> &sys, py::iterable state, py::kwargs) {
                // Build the initial state vector.
                std::vector<mppp::real128> s_vector;
                for (const auto &x : state) {
                    s_vector.push_back(py::cast<mppp::real128>(x));
                }

                return hey::taylor_adaptive<mppp::real128>{sys, std::move(s_vector)};
            }))
        // TODO return an array somehow?
        .def("get_state", [](const hey::taylor_adaptive<mppp::real128> &ta) { return ta.get_state(); });
#endif
}
