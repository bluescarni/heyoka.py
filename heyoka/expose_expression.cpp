// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "pickle_wrappers.hpp"

namespace heyoka_py
{

namespace py = pybind11;

void expose_expression(py::module_ &m)
{
    namespace hey = heyoka;
    namespace heypy = heyoka_py;
    // NOLINTNEXTLINE(google-build-using-namespace)
    using namespace pybind11::literals;

    // NOTE: typedef to avoid complications in the
    // exposition of the operators.
    using ld_t = long double;

    // NOTE: this is used in the implementation of
    // copy/deepcopy for expression. We need this because
    // in order to perform a true copy of an expression
    // we need to use an external function, as the copy ctor
    // performs a shallow copy.
    struct ex_copy_func {
        hey::expression operator()(const hey::expression &ex) const
        {
            return hey::copy(ex);
        }
    };

    py::class_<hey::expression>(m, "expression", py::dynamic_attr{})
        .def(py::init<>())
        .def(py::init([](std::int32_t x) { return hey::expression{static_cast<double>(x)}; }), "x"_a.noconvert())
        .def(py::init<double>(), "x"_a.noconvert())
        .def(py::init<long double>(), "x"_a.noconvert())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::init<mppp::real128>(), "x"_a.noconvert())
#endif
        .def(py::init<std::string>(), "x"_a)
        // Unary operators.
        .def(-py::self)
        .def(+py::self)
        // Binary operators.
        .def(py::self + py::self, "x"_a)
        .def(
            "__add__", [](const hey::expression &ex, std::int32_t x) { return ex + static_cast<double>(x); },
            "x"_a.noconvert())
        .def(
            "__radd__", [](const hey::expression &ex, std::int32_t x) { return ex + static_cast<double>(x); },
            "x"_a.noconvert())
        .def(py::self + double(), "x"_a.noconvert())
        .def(double() + py::self, "x"_a.noconvert())
        .def(py::self + ld_t(), "x"_a.noconvert())
        .def(ld_t() + py::self, "x"_a.noconvert())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self + mppp::real128(), "x"_a.noconvert())
        .def(mppp::real128() + py::self, "x"_a.noconvert())
#endif
        // NOLINTNEXTLINE(misc-redundant-expression)
        .def(py::self - py::self, "x"_a)
        .def(
            "__sub__", [](const hey::expression &ex, std::int32_t x) { return ex - static_cast<double>(x); },
            "x"_a.noconvert())
        .def(
            "__rsub__", [](const hey::expression &ex, std::int32_t x) { return static_cast<double>(x) - ex; },
            "x"_a.noconvert())
        .def(py::self - double(), "x"_a.noconvert())
        .def(double() - py::self, "x"_a.noconvert())
        .def(py::self - ld_t(), "x"_a.noconvert())
        .def(ld_t() - py::self, "x"_a.noconvert())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self - mppp::real128(), "x"_a.noconvert())
        .def(mppp::real128() - py::self, "x"_a.noconvert())
#endif
        .def(py::self * py::self, "x"_a)
        .def(
            "__mul__", [](const hey::expression &ex, std::int32_t x) { return ex * static_cast<double>(x); },
            "x"_a.noconvert())
        .def(
            "__rmul__", [](const hey::expression &ex, std::int32_t x) { return ex * static_cast<double>(x); },
            "x"_a.noconvert())
        .def(py::self * double(), "x"_a.noconvert())
        .def(double() * py::self, "x"_a.noconvert())
        .def(py::self * ld_t(), "x"_a.noconvert())
        .def(ld_t() * py::self, "x"_a.noconvert())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self * mppp::real128(), "x"_a.noconvert())
        .def(mppp::real128() * py::self, "x"_a.noconvert())
#endif
        .def(py::self / py::self, "x"_a)
        .def(
            "__truediv__", [](const hey::expression &ex, std::int32_t x) { return ex / static_cast<double>(x); },
            "x"_a.noconvert())
        .def(
            "__rtruediv__", [](const hey::expression &ex, std::int32_t x) { return static_cast<double>(x) / ex; },
            "x"_a.noconvert())
        .def(py::self / double(), "x"_a.noconvert())
        .def(double() / py::self, "x"_a.noconvert())
        .def(py::self / ld_t(), "x"_a.noconvert())
        .def(ld_t() / py::self, "x"_a.noconvert())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self / mppp::real128(), "x"_a.noconvert())
        .def(mppp::real128() / py::self, "x"_a.noconvert())
#endif
        // Comparisons.
        // NOLINTNEXTLINE(misc-redundant-expression)
        .def(py::self == py::self, "x"_a)
        // NOLINTNEXTLINE(misc-redundant-expression)
        .def(py::self != py::self, "x"_a)
        // pow().
        .def(
            "__pow__", [](const hey::expression &b, const hey::expression &e) { return hey::pow(b, e); }, "e"_a)
        .def(
            "__pow__", [](const hey::expression &b, std::int32_t e) { return hey::pow(b, static_cast<double>(e)); },
            "e"_a.noconvert())
        .def(
            "__pow__", [](const hey::expression &b, double e) { return hey::pow(b, e); }, "e"_a.noconvert())
        .def(
            "__pow__", [](const hey::expression &b, long double e) { return hey::pow(b, e); }, "e"_a.noconvert())
#if defined(HEYOKA_HAVE_REAL128)
        .def(
            "__pow__", [](const hey::expression &b, mppp::real128 e) { return hey::pow(b, e); }, "e"_a.noconvert())
#endif
        // Expression size.
        .def("__len__", [](const hey::expression &e) { return hey::get_n_nodes(e); })
        // Repr.
        .def("__repr__",
             [](const hey::expression &e) {
                 std::ostringstream oss;
                 oss << e;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", heypy::copy_wrapper<hey::expression, ex_copy_func>)
        .def("__deepcopy__", heypy::deepcopy_wrapper<hey::expression, ex_copy_func>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&heypy::pickle_getstate_wrapper<hey::expression>,
                        &heypy::pickle_setstate_wrapper<hey::expression>));

    // Eval
    m.def("_eval_dbl", [](const hey::expression &e, const std::unordered_map<std::string, double> &map,
                          const std::vector<double> &pars) { return hey::eval<double>(e, map, pars); });
    m.def("_eval_ldbl", [](const hey::expression &e, const std::unordered_map<std::string, long double> &map,
                           const std::vector<long double> &pars) { return hey::eval<long double>(e, map, pars); });
#if defined(HEYOKA_HAVE_REAL128)
    m.def("_eval_f128", [](const hey::expression &e, const std::unordered_map<std::string, mppp::real128> &map,
                           const std::vector<mppp::real128> &pars) { return hey::eval<mppp::real128>(e, map, pars); });
#endif

    // Sum.
    m.def(
        "sum",
        [](std::vector<hey::expression> terms, std::uint32_t split) { return hey::sum(std::move(terms), split); },
        "terms"_a, "split"_a = hey::detail::default_sum_split);

    // Sum of squares.
    m.def(
        "sum_sq",
        [](std::vector<hey::expression> terms, std::uint32_t split) { return hey::sum_sq(std::move(terms), split); },
        "terms"_a, "split"_a = hey::detail::default_sum_sq_split);

    // Pairwise prod.
    m.def("pairwise_prod", &hey::pairwise_prod, "terms"_a);

    // Subs.
    m.def("subs", [](const hey::expression &e, const std::unordered_map<std::string, hey::expression> &smap) {
        return hey::subs(e, smap);
    });

    // make_vars() helper.
    m.def("make_vars", [](const py::args &v_str) {
        py::list retval;
        for (auto o : v_str) {
            retval.append(hey::expression(py::cast<std::string>(o)));
        }
        return retval;
    });

    // Math functions.
    m.def("square", &hey::square);
    m.def("sqrt", &hey::sqrt);
    m.def("log", &hey::log);
    m.def("exp", [](hey::expression e) {return hey::exp(std::move(e));});
    m.def("sin", &hey::sin);
    m.def("cos", &hey::cos);
    m.def("tan", &hey::tan);
    m.def("asin", &hey::asin);
    m.def("acos", &hey::acos);
    m.def("atan", &hey::atan);
    m.def("sinh", &hey::sinh);
    m.def("cosh", &hey::cosh);
    m.def("tanh", &hey::tanh);
    m.def("asinh", &hey::asinh);
    m.def("acosh", &hey::acosh);
    m.def("atanh", &hey::atanh);
    m.def("sigmoid", &hey::sigmoid);
    m.def("erf", &hey::erf);
    m.def("powi", &hey::powi);

    // kepE().
    m.def(
        "kepE", [](hey::expression e, hey::expression M) { return hey::kepE(std::move(e), std::move(M)); }, "e"_a,
        "M"_a);
    m.def(
        "kepE", [](double e, hey::expression M) { return hey::kepE(e, std::move(M)); }, "e"_a.noconvert(), "M"_a);
    m.def(
        "kepE", [](long double e, hey::expression M) { return hey::kepE(e, std::move(M)); }, "e"_a.noconvert(), "M"_a);
#if defined(HEYOKA_HAVE_REAL128)
    m.def(
        "kepE", [](mppp::real128 e, hey::expression M) { return hey::kepE(e, std::move(M)); }, "e"_a.noconvert(),
        "M"_a);
#endif

    m.def(
        "kepE", [](hey::expression e, double M) { return hey::kepE(std::move(e), M); }, "e"_a, "M"_a.noconvert());
    m.def(
        "kepE", [](hey::expression e, long double M) { return hey::kepE(std::move(e), M); }, "e"_a, "M"_a.noconvert());
#if defined(HEYOKA_HAVE_REAL128)
    m.def(
        "kepE", [](hey::expression e, mppp::real128 M) { return hey::kepE(std::move(e), M); }, "e"_a,
        "M"_a.noconvert());
#endif

    // atan2().
    m.def(
        "atan2", [](hey::expression y, hey::expression x) { return hey::atan2(std::move(y), std::move(x)); }, "y"_a,
        "x"_a);

    m.def(
        "atan2", [](double y, hey::expression x) { return hey::atan2(y, std::move(x)); }, "y"_a.noconvert(), "x"_a);
    m.def(
        "atan2", [](long double y, hey::expression x) { return hey::atan2(y, std::move(x)); }, "y"_a.noconvert(),
        "x"_a);
#if defined(HEYOKA_HAVE_REAL128)
    m.def(
        "atan2", [](mppp::real128 y, hey::expression x) { return hey::atan2(y, std::move(x)); }, "y"_a.noconvert(),
        "x"_a);
#endif

    m.def(
        "atan2", [](hey::expression y, double x) { return hey::atan2(std::move(y), x); }, "y"_a, "x"_a.noconvert());
    m.def(
        "atan2", [](hey::expression y, long double x) { return hey::atan2(std::move(y), x); }, "y"_a,
        "x"_a.noconvert());
#if defined(HEYOKA_HAVE_REAL128)
    m.def(
        "atan2", [](hey::expression y, mppp::real128 x) { return hey::atan2(std::move(y), x); }, "y"_a,
        "x"_a.noconvert());
#endif

    // Time.
    m.attr("time") = hey::time;

    // pi.
    m.attr("pi") = hey::pi;

    // tpoly().
    m.def("tpoly", &hey::tpoly);

    // Diff.
    m.def(
        "diff", [](const hey::expression &ex, const std::string &s) { return hey::diff(ex, s); }, "ex"_a, "var"_a);
    m.def(
        "diff", [](const hey::expression &ex, const hey::expression &var) { return hey::diff(ex, var); }, "ex"_a,
        "var"_a);

    // Syntax sugar for creating parameters.
    py::class_<hey::detail::par_impl>(m, "_par_generator").def("__getitem__", &hey::detail::par_impl::operator[]);
    m.attr("par") = hey::detail::par_impl{};
}

} // namespace heyoka_py
