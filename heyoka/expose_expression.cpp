// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/expression.hpp>
#include <heyoka/func_args.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "docstrings.hpp"
#include "expose_expression.hpp"
#include "pickle_wrappers.hpp"

namespace heyoka_py
{

namespace py = pybind11;

// NOTE: regarding single-precision support: we only expose the expression ctor,
// but not the arithmetic operators. The reason for this is that np.float32 already
// has math operators defined which sometimes take the precedence over our own exposed
// operators, leading to implicit conversions to float64 and general inconsistent
// behaviour (e.g., when constant folding is involved). In a similar fashion and for
// consistency, we do not expose float32 overloads for multivariate functions.
// NOTE: I am not 100% sure why this happens, as the same problem does not seem to
// be there when doing mixed-mode arithmetic between real and float32 for instance.
// Perhaps something to do with pybind11's conversion machinery (since the exposition
// of real does not use pybind11)?
// NOTE: this may be solved in the new NumPy 2 dtype API, need to check at one point.
void expose_expression(py::module_ &m)
{
    namespace hey = heyoka;
    // NOLINTNEXTLINE(google-build-using-namespace)
    using namespace pybind11::literals;

    // NOTE: typedef to avoid complications in the
    // exposition of the operators.
    using ld_t = long double;

    // Variant holding either an expression or a list of expressions.
    using v_ex_t = std::variant<hey::expression, std::vector<hey::expression>>;

    py::class_<hey::expression>(m, "expression", py::dynamic_attr{}, docstrings::expression().c_str())
        .def(py::init<>(), docstrings::expression_init().c_str())
        .def(py::init([](std::int32_t x) { return hey::expression{static_cast<double>(x)}; }), "x"_a.noconvert())
        .def(py::init<float>(), "x"_a.noconvert())
        .def(py::init<double>(), "x"_a.noconvert())
        .def(py::init<ld_t>(), "x"_a.noconvert())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::init<mppp::real128>(), "x"_a.noconvert())
#endif
#if defined(HEYOKA_HAVE_REAL)
        .def(py::init<mppp::real>(), "x"_a.noconvert())
#endif
        .def(py::init<std::string>(), "x"_a)
        // Unary operators.
        .def(-py::self)
        .def(+py::self)
        // Binary operators.
        .def(py::self + py::self, "x"_a)
        // NOTE: provide a custom implementation of
        // these convenience overloads so that
        // ex + int/int + ex is allowed only if
        // the int fits 32-bit integers, which
        // automatically guarantees they are representable
        // exactly as doubles.
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
#if defined(HEYOKA_HAVE_REAL)
        .def(py::self + mppp::real(), "x"_a.noconvert())
        .def(mppp::real() + py::self, "x"_a.noconvert())
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
#if defined(HEYOKA_HAVE_REAL)
        .def(py::self - mppp::real(), "x"_a.noconvert())
        .def(mppp::real() - py::self, "x"_a.noconvert())
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
#if defined(HEYOKA_HAVE_REAL)
        .def(py::self * mppp::real(), "x"_a.noconvert())
        .def(mppp::real() * py::self, "x"_a.noconvert())
#endif
        // NOLINTNEXTLINE(misc-redundant-expression)
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
#if defined(HEYOKA_HAVE_REAL)
        .def(py::self / mppp::real(), "x"_a.noconvert())
        .def(mppp::real() / py::self, "x"_a.noconvert())
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
#if defined(HEYOKA_HAVE_REAL)
        .def(
            "__pow__", [](const hey::expression &b, mppp::real e) { return hey::pow(b, std::move(e)); },
            "e"_a.noconvert())
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
        .def("__copy__", copy_wrapper<hey::expression>)
        .def("__deepcopy__", deepcopy_wrapper<hey::expression>, "memo"_a)
        // Hashing.
        .def("__hash__", [](const heyoka::expression &e) { return std::hash<heyoka::expression>{}(e); })
        // Pickle support.
        .def(py::pickle(&pickle_getstate_wrapper<hey::expression>, &pickle_setstate_wrapper<hey::expression>));

    // get_variables().
    m.def(
        "get_variables",
        [](const v_ex_t &arg) { return std::visit([](const auto &v) { return hey::get_variables(v); }, arg); },
        "arg"_a);

    // get_params().
    m.def(
        "get_params",
        [](const v_ex_t &arg) { return std::visit([](const auto &v) { return hey::get_params(v); }, arg); }, "arg"_a);

    // rename_variables().
    m.def(
        "rename_variables",
        [](const v_ex_t &arg, const std::unordered_map<std::string, std::string> &d) {
            return std::visit([&d](const auto &v) -> v_ex_t { return hey::rename_variables(v, d); }, arg);
        },
        "arg"_a, "d"_a);

    // subs().
    m.def(
        "subs",
        [](const v_ex_t &arg, const std::variant<std::unordered_map<std::string, hey::expression>,
                                                 std::map<hey::expression, hey::expression>> &smap) {
            return std::visit([](const auto &a, const auto &m) -> v_ex_t { return hey::subs(a, m); }, arg, smap);
        },
        "arg"_a, "smap"_a, docstrings::subs().c_str());

    // make_vars() helper.
    m.def(
        "make_vars",
        [](const py::args &v_str) -> std::variant<hey::expression, py::list> {
            if (py::len(v_str) == 0u) {
                py_throw(PyExc_ValueError, "At least one argument is required when invoking 'make_vars()'");
            }

            if (py::len(v_str) == 1u) {
                return hey::expression(py::cast<std::string>(v_str[0]));
            }

            py::list retval;
            for (auto o : v_str) {
                retval.append(hey::expression(py::cast<std::string>(o)));
            }
            return retval;
        },
        docstrings::make_vars().c_str());

    // Math functions.

    // Sum.
    m.def("sum", &hey::sum, "terms"_a, docstrings::sum().c_str());

    // Prod.
    m.def("prod", &hey::prod, "terms"_a, docstrings::prod().c_str());

    // NOTE: need explicit casts for sqrt and exp due to the presence of overloads for number.
    m.def("sqrt", static_cast<hey::expression (*)(hey::expression)>(&hey::sqrt), "arg"_a);
    m.def("exp", static_cast<hey::expression (*)(hey::expression)>(&hey::exp), "arg"_a);
    m.def("log", &hey::log, "arg"_a);
    m.def("sin", &hey::sin, "arg"_a);
    m.def("cos", &hey::cos, "arg"_a);
    m.def("tan", &hey::tan, "arg"_a);
    m.def("asin", &hey::asin, "arg"_a);
    m.def("acos", &hey::acos, "arg"_a);
    m.def("atan", &hey::atan, "arg"_a);
    m.def("sinh", &hey::sinh, "arg"_a);
    m.def("cosh", &hey::cosh, "arg"_a);
    m.def("tanh", &hey::tanh, "arg"_a);
    m.def("asinh", &hey::asinh, "arg"_a);
    m.def("acosh", &hey::acosh, "arg"_a);
    m.def("atanh", &hey::atanh, "arg"_a);
    m.def("sigmoid", &hey::sigmoid, "arg"_a);
    m.def("erf", &hey::erf, "arg"_a);
    m.def("relu", &hey::relu, "arg"_a, "slope"_a = 0.);
    m.def("relup", &hey::relup, "arg"_a, "slope"_a = 0.);

    // Leaky relu wrappers.
    py::class_<hey::leaky_relu> lr_class(m, "leaky_relu", py::dynamic_attr{});
    lr_class.def(py::init([](double slope) { return hey::leaky_relu(slope); }), "slope"_a);
    lr_class.def("__call__", &hey::leaky_relu::operator(), "arg"_a);

    py::class_<hey::leaky_relup> lrp_class(m, "leaky_relup", py::dynamic_attr{});
    lrp_class.def(py::init([](double slope) { return hey::leaky_relup(slope); }), "slope"_a);
    lrp_class.def("__call__", &hey::leaky_relup::operator(), "arg"_a);

    // NOTE: when exposing multivariate functions, we want to be able to pass
    // in numerical arguments for convenience. Thus, we expose such functions taking
    // in input a union of expression and supported numerical types.
    using mvf_arg = std::variant<hey::expression, double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
#if defined(HEYOKA_HAVE_REAL)
                                 ,
                                 mppp::real
#endif
                                 >;

    // Relational operators.
#define HEYOKA_PY_EXPOSE_REL(op)                                                                                       \
    m.def(                                                                                                             \
        #op,                                                                                                           \
        [](const mvf_arg &x, const mvf_arg &y) {                                                                       \
            return std::visit(                                                                                         \
                []<typename T, typename U>(const T &a, const U &b) -> hey::expression {                                \
                    if constexpr (!std::same_as<T, hey::expression> && !std::same_as<U, hey::expression>) {            \
                        py_throw(PyExc_TypeError, "At least one of the arguments of " #op "() must be an expression"); \
                    } else {                                                                                           \
                        return hey::op(a, b);                                                                          \
                    }                                                                                                  \
                },                                                                                                     \
                x, y);                                                                                                 \
        },                                                                                                             \
        "x"_a.noconvert(), "y"_a.noconvert())

    HEYOKA_PY_EXPOSE_REL(eq);
    HEYOKA_PY_EXPOSE_REL(neq);
    HEYOKA_PY_EXPOSE_REL(lt);
    HEYOKA_PY_EXPOSE_REL(gt);
    HEYOKA_PY_EXPOSE_REL(lte);
    HEYOKA_PY_EXPOSE_REL(gte);

#undef HEYOKA_PY_EXPOSE_REL

    // Logical operators.
    m.def("logical_and", &hey::logical_and, "terms"_a);
    m.def("logical_or", &hey::logical_or, "terms"_a);

    // select().
    m.def(
        "select",
        [](const mvf_arg &c, const mvf_arg &t, const mvf_arg &f) {
            return std::visit(
                []<typename T, typename U, typename V>(const T &a, const U &b, const V &c) -> hey::expression {
                    constexpr auto tp1_num = static_cast<int>(!std::same_as<T, hey::expression>);
                    constexpr auto tp2_num = static_cast<int>(!std::same_as<U, hey::expression>);
                    constexpr auto tp3_num = static_cast<int>(!std::same_as<V, hey::expression>);

                    constexpr auto n_num = tp1_num + tp2_num + tp3_num;

                    if constexpr (n_num == 3) {
                        py_throw(PyExc_TypeError, "At least one of the arguments of select() must be an expression");
                    } else if constexpr (n_num == 2) {
                        constexpr auto flag = tp1_num + (tp2_num << 1) + (tp3_num << 2);

                        if constexpr (flag == 6 && std::same_as<V, U>) {
                            return hey::select(a, b, c);
                        } else if constexpr (flag == 5 && std::same_as<T, V>) {
                            return hey::select(a, b, c);
                        } else if constexpr (flag == 3 && std::same_as<T, U>) {
                            return hey::select(a, b, c);
                        } else {
                            py_throw(PyExc_TypeError,
                                     "The numerical arguments of select() must be all of the same type");
                        }
                    } else {
                        return hey::select(a, b, c);
                    }
                },
                c, t, f);
        },
        "c"_a.noconvert(), "t"_a.noconvert(), "f"_a.noconvert());

    // kepE().
    m.def(
        "kepE",
        [](const mvf_arg &e, const mvf_arg &M) {
            return std::visit(
                [](const auto &a, const auto &b) -> hey::expression {
                    using tp1 = std::remove_cvref_t<decltype(a)>;
                    using tp2 = std::remove_cvref_t<decltype(b)>;

                    if constexpr (!std::is_same_v<tp1, hey::expression> && !std::is_same_v<tp2, hey::expression>) {
                        py_throw(PyExc_TypeError, "At least one of the arguments of kepE() must be an expression");
                    } else {
                        return hey::kepE(a, b);
                    }
                },
                e, M);
        },
        "e"_a.noconvert(), "M"_a.noconvert());

    // kepF().
    m.def(
        "kepF",
        [](const mvf_arg &h, const mvf_arg &k, const mvf_arg &lam) {
            return std::visit(
                [](const auto &a, const auto &b, const auto &c) -> hey::expression {
                    using tp1 = std::remove_cvref_t<decltype(a)>;
                    using tp2 = std::remove_cvref_t<decltype(b)>;
                    using tp3 = std::remove_cvref_t<decltype(c)>;

                    constexpr auto tp1_num = static_cast<int>(!std::is_same_v<tp1, hey::expression>);
                    constexpr auto tp2_num = static_cast<int>(!std::is_same_v<tp2, hey::expression>);
                    constexpr auto tp3_num = static_cast<int>(!std::is_same_v<tp3, hey::expression>);

                    constexpr auto n_num = tp1_num + tp2_num + tp3_num;

                    if constexpr (n_num == 3) {
                        py_throw(PyExc_TypeError, "At least one of the arguments of kepF() must be an expression");
                    } else if constexpr (n_num == 2) {
                        constexpr auto flag = tp1_num + (tp2_num << 1) + (tp3_num << 2);

                        if constexpr (flag == 6 && std::is_same_v<tp3, tp2>) {
                            return hey::kepF(a, b, c);
                        } else if constexpr (flag == 5 && std::is_same_v<tp1, tp3>) {
                            return hey::kepF(a, b, c);
                        } else if constexpr (flag == 3 && std::is_same_v<tp1, tp2>) {
                            return hey::kepF(a, b, c);
                        } else {
                            py_throw(PyExc_TypeError, "The numerical arguments of kepF() must be all of the same type");
                        }
                    } else {
                        return hey::kepF(a, b, c);
                    }
                },
                h, k, lam);
        },
        "h"_a.noconvert(), "k"_a.noconvert(), "lam"_a.noconvert());

    // kepDE().
    m.def(
        "kepDE",
        [](const mvf_arg &s0, const mvf_arg &c0, const mvf_arg &DM) {
            return std::visit(
                [](const auto &a, const auto &b, const auto &c) -> hey::expression {
                    using tp1 = std::remove_cvref_t<decltype(a)>;
                    using tp2 = std::remove_cvref_t<decltype(b)>;
                    using tp3 = std::remove_cvref_t<decltype(c)>;

                    constexpr auto tp1_num = static_cast<int>(!std::is_same_v<tp1, hey::expression>);
                    constexpr auto tp2_num = static_cast<int>(!std::is_same_v<tp2, hey::expression>);
                    constexpr auto tp3_num = static_cast<int>(!std::is_same_v<tp3, hey::expression>);

                    constexpr auto n_num = tp1_num + tp2_num + tp3_num;

                    if constexpr (n_num == 3) {
                        py_throw(PyExc_TypeError, "At least one of the arguments of kepDE() must be an expression");
                    } else if constexpr (n_num == 2) {
                        constexpr auto flag = tp1_num + (tp2_num << 1) + (tp3_num << 2);

                        if constexpr (flag == 6 && std::is_same_v<tp3, tp2>) {
                            return hey::kepDE(a, b, c);
                        } else if constexpr (flag == 5 && std::is_same_v<tp1, tp3>) {
                            return hey::kepDE(a, b, c);
                        } else if constexpr (flag == 3 && std::is_same_v<tp1, tp2>) {
                            return hey::kepDE(a, b, c);
                        } else {
                            py_throw(PyExc_TypeError,
                                     "The numerical arguments of kepDE() must be all of the same type");
                        }
                    } else {
                        return hey::kepDE(a, b, c);
                    }
                },
                s0, c0, DM);
        },
        "s0"_a.noconvert(), "c0"_a.noconvert(), "DM"_a.noconvert());

    // atan2().
    m.def(
        "atan2",
        [](const mvf_arg &y, const mvf_arg &x) {
            return std::visit(
                [](const auto &a, const auto &b) -> hey::expression {
                    using tp1 = std::remove_cvref_t<decltype(a)>;
                    using tp2 = std::remove_cvref_t<decltype(b)>;

                    if constexpr (!std::is_same_v<tp1, hey::expression> && !std::is_same_v<tp2, hey::expression>) {
                        py_throw(PyExc_TypeError, "At least one of the arguments of atan2() must be an expression");
                    } else {
                        return hey::atan2(a, b);
                    }
                },
                y, x);
        },
        "y"_a.noconvert(), "x"_a.noconvert());

    // dfun().
    m.def(
        "dfun",
        [](std::string name, std::vector<hey::expression> args,
           std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>> didx) {
            if (didx) {
                return hey::dfun(std::move(name), std::move(args), std::move(*didx));
            } else {
                return hey::dfun(std::move(name), std::move(args));
            }
        },
        "name"_a, "args"_a, "didx"_a = py::none{});

    // Time.
    m.attr("_time") = hey::time;

    // pi.
    m.attr("pi") = hey::pi;

    // Diff.
    m.def(
        "diff",
        [](const v_ex_t &arg, const std::variant<std::string, hey::expression> &var) {
            return std::visit([](const auto &a, const auto &v) -> v_ex_t { return hey::diff(a, v); }, arg, var);
        },
        "arg"_a, "var"_a);

    // Syntax sugar for creating parameters.
    py::class_<hey::detail::par_impl>(m, "_par_generator")
        .def(py::init<>())
        .def("__getitem__", &hey::detail::par_impl::operator[]);

    // dtens.
    py::class_<hey::dtens> dtens_cl(m, "dtens", py::dynamic_attr{}, docstrings::dtens().c_str());
    dtens_cl.def(py::init<>(), docstrings::dtens_init().c_str());
    // Total number of derivatives.
    dtens_cl.def("__len__", &hey::dtens::size);
    // Repr.
    dtens_cl.def("__repr__", [](const hey::dtens &dt) {
        std::ostringstream oss;
        oss << dt;
        return oss.str();
    });
    // Read-only properties.
    dtens_cl.def_property_readonly("order", &hey::dtens::get_order, docstrings::dtens_order().c_str());
    dtens_cl.def_property_readonly("nargs", &hey::dtens::get_nargs, docstrings::dtens_nargs().c_str());
    dtens_cl.def_property_readonly("nouts", &hey::dtens::get_nouts, docstrings::dtens_nouts().c_str());
    dtens_cl.def_property_readonly("args", &hey::dtens::get_args, docstrings::dtens_args().c_str());
    // Lookup/contains.
    dtens_cl.def(
        "__getitem__", [](const hey::dtens &dt, const std::variant<hey::dtens::v_idx_t, hey::dtens::sv_idx_t> &v_idx_) {
            return std::visit(
                [&](const auto &v_idx) {
                    const auto it = dt.find(v_idx);

                    if (it == dt.end()) {
                        py_throw(PyExc_KeyError,
                                 fmt::format("Cannot locate the derivative corresponding the the vector of indices {}",
                                             v_idx)
                                     .c_str());
                    }

                    return it->second;
                },
                v_idx_);
        });
    dtens_cl.def("__getitem__", [](const hey::dtens &dt, hey::dtens::size_type idx) {
        if (idx >= dt.size()) {
            py_throw(PyExc_IndexError,
                     fmt::format("The derivative at index {} was requested, but the total number of derivatives is {}",
                                 idx, dt.size())
                         .c_str());
        }

        const auto s_idx = boost::numeric_cast<std::iterator_traits<hey::dtens::iterator>::difference_type>(idx);

        return dtens_t_it{&dt}(dt.begin()[s_idx]);
    });
    dtens_cl.def("__contains__",
                 [](const hey::dtens &dt, const std::variant<hey::dtens::v_idx_t, hey::dtens::sv_idx_t> &v_idx_) {
                     return std::visit([&](const auto &v_idx) { return dt.find(v_idx) != dt.end(); }, v_idx_);
                 });
    // Iterator.
    dtens_cl.def(
        "__iter__",
        [](const hey::dtens &dt) {
            auto t_begin = boost::iterators::make_transform_iterator(dt.begin(), dtens_t_it{&dt});
            auto t_end = boost::iterators::make_transform_iterator(dt.end(), dtens_t_it{&dt});

            return py::make_key_iterator(t_begin, t_end);
        },
        // NOTE: the calling dtens (argument index 1) needs to be kept alive at least until
        // the return value (argument index 0) is freed by the garbage collector.
        // This ensures that if we fetch an iterator and then delete the originating dtens object,
        // the iterator still points to valid data.
        py::keep_alive<0, 1>{});
    // index_of().
    dtens_cl.def(
        "index_of",
        [](const hey::dtens &dt, const std::variant<hey::dtens::v_idx_t, hey::dtens::sv_idx_t> &v_idx_) {
            return std::visit([&](const auto &v_idx) { return dt.index_of(v_idx); }, v_idx_);
        },
        "vidx"_a, docstrings::dtens_index_of().c_str());
    // get_derivatives().
    dtens_cl.def(
        "get_derivatives",
        [](const hey::dtens &dt, std::uint32_t order, std::optional<std::uint32_t> component) {
            const auto sr = component ? dt.get_derivatives(*component, order) : dt.get_derivatives(order);

            auto t_begin = boost::iterators::make_transform_iterator(sr.begin(), dtens_t_it{&dt});
            auto t_end = boost::iterators::make_transform_iterator(sr.end(), dtens_t_it{&dt});

            return std::vector(t_begin, t_end);
        },
        "diff_order"_a, "component"_a = py::none{}, docstrings::dtens_get_derivatives().c_str());
    // Gradient.
    dtens_cl.def_property_readonly("gradient", &hey::dtens::get_gradient, docstrings::dtens_gradient().c_str());
    // Jacobian.
    dtens_cl.def_property_readonly(
        "jacobian",
        [](const hey::dtens &dt) {
            auto jac = py::array(py::cast(dt.get_jacobian()));

            return jac.reshape(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(dt.get_nouts()),
                                                         boost::numeric_cast<py::ssize_t>(dt.get_nargs())});
        },
        docstrings::dtens_jacobian().c_str());
    // Hessian.
    dtens_cl.def(
        "hessian",
        [](const hey::dtens &dt, std::uint32_t component) {
            py::list h = py::cast(dt.get_hessian(component));

            // Reconstruct the Hessian from its representation as a list
            // of component of the upper triangular part.
            const auto nargs = dt.get_nargs();

            auto np = py::module_::import("numpy");
            auto arr1 = np.attr("full")(py::make_tuple(nargs, nargs), 0., "dtype"_a = "object");

            auto ui = np.attr("triu_indices")(nargs);
            arr1[ui] = h;

            auto arr2 = arr1.attr("T").attr("copy")();
            auto di = np.attr("diag_indices")(nargs);

            arr2[di] = 0.;

            return arr1 + arr2;
        },
        "component"_a, docstrings::dtens_hessian().c_str());
    // Copy/deepcopy.
    dtens_cl.def("__copy__", copy_wrapper<hey::dtens>);
    dtens_cl.def("__deepcopy__", deepcopy_wrapper<hey::dtens>, "memo"_a);
    // Pickle support.
    dtens_cl.def(py::pickle(&pickle_getstate_wrapper<hey::dtens>, &pickle_setstate_wrapper<hey::dtens>));

    // diff_args enum.
    py::enum_<hey::diff_args>(m, "diff_args", docstrings::diff_args().c_str())
        .value("vars", hey::diff_args::vars, docstrings::diff_args_vars().c_str())
        .value("params", hey::diff_args::params, docstrings::diff_args_pars().c_str())
        .value("all", hey::diff_args::all, docstrings::diff_args_all().c_str());

    // diff_tensors().
    m.def(
        "diff_tensors",
        [](const std::vector<hey::expression> &v_ex,
           const std::variant<hey::diff_args, std::vector<hey::expression>> &diff_args,
           std::uint32_t diff_order) { return hey::diff_tensors(v_ex, diff_args, hey::kw::diff_order = diff_order); },
        "func"_a, "diff_args"_a, "diff_order"_a = static_cast<std::uint32_t>(1), docstrings::diff_tensors().c_str());

    // func_args class.
    py::class_<hey::func_args> func_args_cl(m, "func_args", py::dynamic_attr{}, docstrings::func_args().c_str());
    func_args_cl.def(py::init([](std::vector<hey::expression> args, bool shared) {
                         return hey::func_args(std::move(args), shared);
                     }),
                     "args"_a = std::vector<hey::expression>{}, "shared"_a = false,
                     docstrings::func_args_init().c_str());
    func_args_cl.def_property_readonly(
        "args", [](const hey::func_args &fa) { return fa.get_args(); }, docstrings::func_args_args().c_str());
    func_args_cl.def_property_readonly(
        "is_shared", [](const hey::func_args &fa) { return static_cast<bool>(fa.get_shared_args()); },
        docstrings::func_args_is_shared().c_str());
    // Copy/deepcopy.
    func_args_cl.def("__copy__", copy_wrapper<hey::func_args>);
    func_args_cl.def("__deepcopy__", deepcopy_wrapper<hey::func_args>, "memo"_a);
    // Pickle support.
    func_args_cl.def(py::pickle(&pickle_getstate_wrapper<hey::func_args>, &pickle_setstate_wrapper<hey::func_args>));
}

} // namespace heyoka_py
