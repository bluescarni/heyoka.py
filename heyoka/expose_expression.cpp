// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/expression.hpp>
#include <heyoka/math.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "pickle_wrappers.hpp"

namespace heyoka_py
{

namespace detail
{

namespace
{

template <typename T>
using uncvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

} // namespace

} // namespace detail

namespace py = pybind11;

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

    py::class_<hey::expression>(m, "expression", py::dynamic_attr{})
        .def(py::init<>())
        .def(py::init([](std::int32_t x) { return hey::expression{static_cast<double>(x)}; }), "x"_a.noconvert())
        .def(py::init<double>(), "x"_a.noconvert())
        .def(py::init<long double>(), "x"_a.noconvert())
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
        [](const v_ex_t &arg,
           const std::variant<std::unordered_map<std::string, hey::expression>,
                              std::unordered_map<hey::expression, hey::expression>> &smap,
           bool normalise) {
            return std::visit(
                [normalise](const auto &a, const auto &m) -> v_ex_t { return hey::subs(a, m, normalise); }, arg, smap);
        },
        "arg"_a, "smap"_a, "normalise"_a = false);

    // fix()/unfix().
    m.def("fix", &hey::fix, "arg"_a);
    m.def("fix_nn", &hey::fix_nn, "arg"_a);
    m.def(
        "unfix",
        [](const v_ex_t &arg) { return std::visit([](const auto &v) -> v_ex_t { return hey::unfix(v); }, arg); },
        "arg"_a);

    // normalise().
    m.def(
        "normalise",
        [](const v_ex_t &arg) { return std::visit([](const auto &v) -> v_ex_t { return hey::normalise(v); }, arg); },
        "arg"_a);

    // make_vars() helper.
    m.def("make_vars", [](const py::args &v_str) {
        py::list retval;
        for (auto o : v_str) {
            retval.append(hey::expression(py::cast<std::string>(o)));
        }
        return retval;
    });

    // Math functions.

    // Sum.
    m.def("sum", &hey::sum, "terms"_a);

    // Prod.
    m.def("prod", &hey::prod, "terms"_a);

    // NOTE: need explicit casts for sqrt and exp due to the presence of overloads for number.
    m.def("sqrt", static_cast<hey::expression (*)(hey::expression)>(&hey::sqrt));
    m.def("exp", static_cast<hey::expression (*)(hey::expression)>(&hey::exp));
    m.def("log", &hey::log);
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

    // kepE().
    m.def(
        "kepE",
        [](const mvf_arg &e, const mvf_arg &M) {
            return std::visit(
                [](const auto &a, const auto &b) -> hey::expression {
                    using tp1 = detail::uncvref_t<decltype(a)>;
                    using tp2 = detail::uncvref_t<decltype(b)>;

                    if constexpr (!std::is_same_v<tp1, hey::expression> && !std::is_same_v<tp2, hey::expression>) {
                        py_throw(PyExc_TypeError, "At least one of the arguments of kepE() must be an expression");
                    } else {
                        return hey::kepE(a, b);
                    }
                },
                e, M);
        },
        "e"_a, "M"_a);

    // kepF().
    m.def(
        "kepF",
        [](const mvf_arg &h, const mvf_arg &k, const mvf_arg &lam) {
            return std::visit(
                [](const auto &a, const auto &b, const auto &c) -> hey::expression {
                    using tp1 = detail::uncvref_t<decltype(a)>;
                    using tp2 = detail::uncvref_t<decltype(b)>;
                    using tp3 = detail::uncvref_t<decltype(c)>;

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
        "h"_a, "k"_a, "lam"_a);

    // kepDE().
    m.def(
        "kepDE",
        [](const mvf_arg &s0, const mvf_arg &c0, const mvf_arg &DM) {
            return std::visit(
                [](const auto &a, const auto &b, const auto &c) -> hey::expression {
                    using tp1 = detail::uncvref_t<decltype(a)>;
                    using tp2 = detail::uncvref_t<decltype(b)>;
                    using tp3 = detail::uncvref_t<decltype(c)>;

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
        "s0"_a, "c0"_a, "DM"_a);

    // atan2().
    m.def(
        "atan2",
        [](const mvf_arg &y, const mvf_arg &x) {
            return std::visit(
                [](const auto &a, const auto &b) -> hey::expression {
                    using tp1 = detail::uncvref_t<decltype(a)>;
                    using tp2 = detail::uncvref_t<decltype(b)>;

                    if constexpr (!std::is_same_v<tp1, hey::expression> && !std::is_same_v<tp2, hey::expression>) {
                        py_throw(PyExc_TypeError, "At least one of the arguments of atan2() must be an expression");
                    } else {
                        return hey::atan2(a, b);
                    }
                },
                y, x);
        },
        "y"_a, "x"_a);

    // Time.
    m.attr("time") = hey::time;

    // pi.
    m.attr("pi") = hey::pi;

    // Diff.
    m.def(
        "diff",
        [](const hey::expression &ex, const std::variant<std::string, hey::expression> &var) {
            return std::visit([&ex](const auto &v) { return hey::diff(ex, v); }, var);
        },
        "ex"_a, "var"_a);

    // Syntax sugar for creating parameters.
    py::class_<hey::detail::par_impl>(m, "_par_generator").def("__getitem__", &hey::detail::par_impl::operator[]);
    m.attr("par") = hey::detail::par_impl{};

    // dtens.
    py::class_<hey::dtens> dtens_cl(m, "dtens", py::dynamic_attr{});
    dtens_cl.def(py::init<>());
    // Total number of derivatives.
    dtens_cl.def("__len__", &hey::dtens::size);
    // Repr.
    dtens_cl.def("__repr__", [](const hey::dtens &dt) {
        std::ostringstream oss;
        oss << dt;
        return oss.str();
    });
    // Read-only properties.
    dtens_cl.def_property_readonly("order", &hey::dtens::get_order);
    dtens_cl.def_property_readonly("nvars", &hey::dtens::get_nvars);
    dtens_cl.def_property_readonly("nouts", &hey::dtens::get_nouts);
    dtens_cl.def_property_readonly("args", &hey::dtens::get_args);
    // Lookup/contains.
    dtens_cl.def("__getitem__", [](const hey::dtens &dt, const hey::dtens::v_idx_t &v_idx) {
        const auto it = dt.find(v_idx);

        if (it == dt.end()) {
            py_throw(
                PyExc_KeyError,
                fmt::format("Cannot locate the derivative corresponding the the vector of indices {}", v_idx).c_str());
        }

        return it->second;
    });
    dtens_cl.def("__getitem__", [](const hey::dtens &dt, hey::dtens::size_type idx) {
        if (idx >= dt.size()) {
            py_throw(PyExc_IndexError,
                     fmt::format("The derivative at index {} was requested, but the total number of derivatives is {}",
                                 idx, dt.size())
                         .c_str());
        }

        const auto s_idx = boost::numeric_cast<std::iterator_traits<hey::dtens::iterator>::difference_type>(idx);

        return dt.begin()[s_idx];
    });
    dtens_cl.def("__contains__",
                 [](const hey::dtens &dt, const hey::dtens::v_idx_t &v_idx) { return dt.find(v_idx) != dt.end(); });
    // Iterator.
    dtens_cl.def(
        "__iter__", [](const hey::dtens &dt) { return py::make_key_iterator(dt.begin(), dt.end()); },
        // NOTE: the calling dtens (argument index 1) needs to be kept alive at least until
        // the return value (argument index 0) is freed by the garbage collector.
        // This ensures that if we fetch an iterator and then delete the originating dtens object,
        // the iterator still points to valid data.
        py::keep_alive<0, 1>{});
    // index_of().
    dtens_cl.def(
        "index_of", [](const hey::dtens &dt, const hey::dtens::v_idx_t &v_idx) { return dt.index_of(v_idx); },
        "vidx"_a);
    // get_derivatives().
    dtens_cl.def(
        "get_derivatives",
        [](const hey::dtens &dt, std::uint32_t order, std::optional<std::uint32_t> component) {
            const auto sr = component ? dt.get_derivatives(*component, order) : dt.get_derivatives(order);

            return std::vector(sr.begin(), sr.end());
        },
        "diff_order"_a, "component"_a = py::none{});
    // Gradient.
    dtens_cl.def_property_readonly("gradient", &hey::dtens::get_gradient);
    // Jacobian.
    dtens_cl.def_property_readonly("jacobian", [](const hey::dtens &dt) {
        auto jac = py::array(py::cast(dt.get_jacobian()));

        return jac.reshape(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(dt.get_nouts()),
                                                     boost::numeric_cast<py::ssize_t>(dt.get_nvars())});
    });
    // Copy/deepcopy.
    dtens_cl.def("__copy__", copy_wrapper<hey::dtens>);
    dtens_cl.def("__deepcopy__", deepcopy_wrapper<hey::dtens>, "memo"_a);
    // Pickle support.
    dtens_cl.def(py::pickle(&pickle_getstate_wrapper<hey::dtens>, &pickle_setstate_wrapper<hey::dtens>));

    // diff_args enum.
    py::enum_<hey::diff_args>(m, "diff_args")
        .value("vars", hey::diff_args::vars)
        .value("params", hey::diff_args::params)
        .value("all", hey::diff_args::all);

    // diff_tensors().
    m.def(
        "diff_tensors",
        [](const std::vector<hey::expression> &v_ex,
           const std::variant<hey::diff_args, std::vector<hey::expression>> &diff_args, std::uint32_t diff_order) {
            return std::visit(
                [&v_ex, diff_order](const auto &v) {
                    return hey::diff_tensors(v_ex, hey::kw::diff_args = v, hey::kw::diff_order = diff_order);
                },
                diff_args);
        },
        "func"_a, "diff_args"_a = hey::diff_args::vars, "diff_order"_a = static_cast<std::uint32_t>(1));
}

} // namespace heyoka_py
