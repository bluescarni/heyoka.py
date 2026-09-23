// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <pybind11/native_enum.h>
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

// NOTE: regarding single-precision support: we only expose the expression ctor, but not the arithmetic operators or
// other multivariate functions. The reason for this is that np.float32 already has math operators defined which
// sometimes take the precedence over our own exposed operators, leading to implicit conversions to float64 and general
// inconsistent behaviour (e.g., when constant folding is involved). Note however that, since the operators and the
// multivariate functions allow conversions to double, np.float32 arguments will be silently converted to double (unless
// the np.float32 is the left operand of a binary operator, in which case NumPy's operator is tried first).
//
// NOTE: I am not 100% sure why this happens, as the same problem does not seem to be there when doing mixed-mode
// arithmetic between real and float32 for instance. Perhaps something to do with pybind11's conversion machinery (since
// the exposition of real does not use pybind11)?
//
// NOTE: this may be solved in the new NumPy 2 dtype API, need to check at one point.
void expose_expression(py::module_ &m)
{
    namespace hey = heyoka;
    // NOLINTNEXTLINE(google-build-using-namespace)
    using namespace pybind11::literals;

    // Variant holding either an expression or a list of expressions.
    using v_ex_t = std::variant<hey::expression, std::vector<hey::expression>>;

    // NOTE: when exposing multivariate functions, we want to be able to pass in numerical arguments for convenience.
    // Thus, we expose such functions taking in input a union of expression and supported numerical types.
    using mvf_arg_t = std::variant<hey::expression, double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                   ,
                                   mppp::real128
#endif
#if defined(HEYOKA_HAVE_REAL)
                                   ,
                                   mppp::real
#endif
                                   >;

    // Construction argument type for expression.
    using ex_ctor_arg_t = std::variant<hey::expression, std::string, float, double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                       ,
                                       mppp::real128
#endif
#if defined(HEYOKA_HAVE_REAL)
                                       ,
                                       mppp::real
#endif
                                       >;

    py::class_<hey::expression> ex_class(m, "expression", py::dynamic_attr{}, docstrings::expression().c_str());

    // Ctor.
    ex_class.def(py::init([](const ex_ctor_arg_t &x) {
                     return std::visit([](const auto &arg) { return hey::expression{arg}; }, x);
                 }),
                 "x"_a = 0., docstrings::expression_init().c_str());

    // Unary operators.
    ex_class.def(-py::self).def(+py::self);

#define HEYOKA_PY_EXPOSE_BINARY_OPERATOR(op_name, op)                                                                  \
    ex_class                                                                                                           \
        .def(                                                                                                          \
            "__" #op_name "__",                                                                                        \
            [](const hey::expression &a, const mvf_arg_t &b) {                                                         \
                return std::visit([&a](const auto &v) { return a op v; }, b);                                          \
            },                                                                                                         \
            py::is_operator(), "x"_a)                                                                                  \
        .def(                                                                                                          \
            "__r" #op_name "__",                                                                                       \
            [](const hey::expression &a, const mvf_arg_t &b) {                                                         \
                return std::visit([&a](const auto &v) { return v op a; }, b);                                          \
            },                                                                                                         \
            py::is_operator(), "x"_a)

    // Binary operators.
    HEYOKA_PY_EXPOSE_BINARY_OPERATOR(add, +);
    HEYOKA_PY_EXPOSE_BINARY_OPERATOR(sub, -);
    HEYOKA_PY_EXPOSE_BINARY_OPERATOR(mul, *);
    HEYOKA_PY_EXPOSE_BINARY_OPERATOR(truediv, /);

#undef HEYOKA_PY_EXPOSE_BINARY_OPERATOR

    // Comparisons.
    ex_class
        // NOLINTNEXTLINE(misc-redundant-expression)
        .def(py::self == py::self, "x"_a)
        // NOLINTNEXTLINE(misc-redundant-expression)
        .def(py::self != py::self, "x"_a);

    // pow().
    ex_class.def(
        "__pow__",
        [](const hey::expression &b, const mvf_arg_t &e) {
            return std::visit([&b](const auto &arg) { return hey::pow(b, arg); }, e);
        },
        py::is_operator(), "e"_a);

    // Expression size.
    ex_class.def("__len__", [](const hey::expression &e) { return hey::get_n_nodes(e); });

    // Repr.
    ex_class.def("__repr__", [](const hey::expression &e) {
        std::ostringstream oss;
        oss << e;
        return oss.str();
    });

    // Copy/deepcopy.
    ex_class.def("__copy__", copy_wrapper<hey::expression>)
        .def("__deepcopy__", deepcopy_wrapper<hey::expression>, "memo"_a);

    // Hashing.
    ex_class.def("__hash__", [](const heyoka::expression &e) { return std::hash<heyoka::expression>{}(e); });

    // Pickle support.
    ex_class.def(py::pickle(&pickle_getstate_wrapper<hey::expression>, &pickle_setstate_wrapper<hey::expression>));

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
    m.def("sqrt", static_cast<hey::expression (*)(const hey::expression &)>(&hey::sqrt), "arg"_a);
    m.def("exp", static_cast<hey::expression (*)(hey::expression)>(&hey::exp), "arg"_a);
    m.def("expm1", &hey::expm1, "arg"_a);
    m.def("log", &hey::log, "arg"_a);
    m.def("log1p", &hey::log1p, "arg"_a);
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
    m.def("erfc", &hey::erfc, "arg"_a);
    m.def("relu", &hey::relu, "arg"_a, "slope"_a = 0.);
    m.def("relup", &hey::relup, "arg"_a, "slope"_a = 0.);

    // Leaky relu wrappers.
    py::class_<hey::leaky_relu> lr_class(m, "leaky_relu", py::dynamic_attr{});
    lr_class.def(py::init([](double slope) { return hey::leaky_relu(slope); }), "slope"_a);
    lr_class.def("__call__", &hey::leaky_relu::operator(), "arg"_a);

    py::class_<hey::leaky_relup> lrp_class(m, "leaky_relup", py::dynamic_attr{});
    lrp_class.def(py::init([](double slope) { return hey::leaky_relup(slope); }), "slope"_a);
    lrp_class.def("__call__", &hey::leaky_relup::operator(), "arg"_a);

    // Relational operators.
#define HEYOKA_PY_EXPOSE_REL(op)                                                                                       \
    m.def(                                                                                                             \
        #op,                                                                                                           \
        [](const mvf_arg_t &x, const mvf_arg_t &y) {                                                                   \
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
        "x"_a, "y"_a)

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
        [](const mvf_arg_t &c, const mvf_arg_t &t, const mvf_arg_t &f) {
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
        "c"_a, "t"_a, "f"_a);

    // kepE().
    m.def(
        "kepE",
        [](const mvf_arg_t &e, const mvf_arg_t &M) {
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
        "e"_a, "M"_a);

    // kepF().
    m.def(
        "kepF",
        [](const mvf_arg_t &h, const mvf_arg_t &k, const mvf_arg_t &lam) {
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
        "h"_a, "k"_a, "lam"_a);

    // kepDE().
    m.def(
        "kepDE",
        [](const mvf_arg_t &s0, const mvf_arg_t &c0, const mvf_arg_t &DM) {
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
        "s0"_a, "c0"_a, "DM"_a);

    // atan2().
    m.def(
        "atan2",
        [](const mvf_arg_t &y, const mvf_arg_t &x) {
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
        "y"_a, "x"_a);

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
    m.attr("_par") = hey::detail::par_impl{};

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
    py::native_enum<hey::diff_args>(m, "diff_args", "enum.Enum", docstrings::diff_args().c_str())
        .value("vars", hey::diff_args::vars, docstrings::diff_args_vars().c_str())
        .value("params", hey::diff_args::params, docstrings::diff_args_pars().c_str())
        .value("all", hey::diff_args::all, docstrings::diff_args_all().c_str())
        .finalize();

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
