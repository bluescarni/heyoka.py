// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <variant>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128) || defined(HEYOKA_HAVE_REAL)

#include <mp++/integer.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "setup_sympy.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hy = heyoka;

namespace detail
{

namespace
{

// The dictionary for mapping heyoka functions either directly to sympy functions
// or to callbacks for the creation of sympy wrappers.
std::unordered_map<std::type_index,
                   std::variant<py::object, std::function<py::object(std::unordered_map<const void *, py::object> &,
                                                                     const hy::func &)>>>
    fmap;

// Global variable that will contain the sympy module,
// if available.
std::optional<py::object> spy;

// Fwd-declare the main conversion function.
py::object to_sympy_impl(std::unordered_map<const void *, py::object> &, const hy::expression &);

// Implementation of the conversion functions for the node types.
py::object to_sympy_impl(std::unordered_map<const void *, py::object> &, const hy::variable &var)
{
    assert(spy);

    // NOTE: heyoka symbols can assume only
    // real values.
    py::kwargs kwa;
    kwa["real"] = true;

    return spy->attr("Symbol")(var.name(), **kwa);
}

py::object to_sympy_impl(std::unordered_map<const void *, py::object> &, const hy::param &par)
{
    assert(spy);

    // NOTE: params are converted to symbolic variables
    // following a naming convention.

    // NOTE: heyoka params can assume only
    // real values.
    py::kwargs kwa;
    kwa["real"] = true;

    return spy->attr("Symbol")(fmt::format("par[{}]", par.idx()), **kwa);
}

// Small helper to check if n stores
// an integral value.
bool is_integer(const hy::number &n)
{
    return std::visit(
        [](const auto &arg) {
            using std::trunc;
            using std::isfinite;

            return isfinite(arg) && trunc(arg) == arg;
        },
        n.value());
}

// Number conversion corresponds to creating a SymPy number from the
// original value.
py::object to_sympy_impl(std::unordered_map<const void *, py::object> &, const hy::number &num)
{
    // NOTE: if num contains an integral value, we want to convert it into a SymPy integer,
    // since several simplifications are disabled for floating-point constants:
    // https://github.com/sympy/sympy/issues/23040
    const auto is_int = is_integer(num);

    return std::visit(
        [&num, is_int]<typename T>(const T &x) -> py::object {
            using std::isfinite;

            // NOTE: forbid conversion if the value is not finite.
            if (!isfinite(x)) {
                py_throw(PyExc_ValueError,
                         (fmt::format("Cannot convert to sympy the nonfinite number {}", hy::expression{num})).c_str());
            }

#if defined(HEYOKA_HAVE_REAL128)
            if constexpr (std::is_same_v<T, mppp::real128>) {
                if (is_int) {
                    return spy->attr("Integer")(static_cast<mppp::integer<1>>(x).to_string());
                } else {
                    return spy->attr("Float")(x.to_string(), py::none{}, std::numeric_limits<mppp::real128>::digits);
                }
            }
#endif

#if defined(HEYOKA_HAVE_REAL)
            if constexpr (std::is_same_v<T, mppp::real>) {
                if (is_int) {
                    return spy->attr("Integer")(static_cast<mppp::integer<1>>(x).to_string());
                } else {
                    return spy->attr("Float")(x.to_string(), py::none{}, x.get_prec());
                }
            }
#endif

            if constexpr (std::is_floating_point_v<T>) {
                if (is_int) {
#if defined(HEYOKA_HAVE_REAL128) || defined(HEYOKA_HAVE_REAL)
                    return spy->attr("Integer")(static_cast<mppp::integer<1>>(x).to_string());
#else
                    // NOTE: if we cannot leverage mppp::integer for the conversion, let's just
                    // try with a 64-bit int. Clearly this could overflow and raise an exception,
                    // if this becomes an issue we will have to find a better solution.
                    return spy->attr("Integer")(boost::numeric_cast<std::int64_t>(x));
#endif
                } else {
                    return spy->attr("Float")(py::cast(x));
                }
            }

            // NOTE: we should never end up here.
            throw std::invalid_argument(
                "An unsupported C++ floating-point type was detected while trying to convert an expression to sympy");
        },
        num.value());
}

py::object to_sympy_impl(std::unordered_map<const void *, py::object> &func_map, const hy::func &f)
{
    const auto *const f_id = f.get_ptr();

    if (auto it = func_map.find(f_id); it != func_map.end()) {
        // We already converted the current function, return the
        // cached result.
        return it->second;
    }

    auto it = fmap.find(f.get_type_index());

    if (it == fmap.end()) {
        py_throw(PyExc_TypeError,
                 (fmt::format("Cannot convert to sympy the heyoka function {}", hy::expression{f})).c_str());
    }

    py::object retval;
    if (auto *pobj = std::get_if<0>(&it->second)) {
        // We can use directly a sympy function. Convert
        // the function arguments and invoke the function.
        py::list args;
        for (const auto &arg : f.args()) {
            args.append(to_sympy_impl(func_map, arg));
        }

        retval = (*pobj)(*args);
    } else {
        // Cannot use directly a sympy function, invoke the wrapper.
        retval = std::get<1>(it->second)(func_map, f);
    }

    // Update the cache.
    [[maybe_unused]] const auto [_, flag] = func_map.insert(std::pair{f_id, retval});
    // NOTE: an expression cannot contain itself.
    assert(flag);

    return retval;
}

py::object to_sympy_impl(std::unordered_map<const void *, py::object> &func_map, const hy::expression &ex)
{
    return std::visit([&func_map](const auto &v) { return to_sympy_impl(func_map, v); }, ex.value());
}

py::object to_sympy(const hy::expression &ex)
{
    std::unordered_map<const void *, py::object> func_map;

    return to_sympy_impl(func_map, ex);
}

py::list to_sympy(const std::vector<hy::expression> &v_ex)
{
    std::unordered_map<const void *, py::object> func_map;

    py::list retval;

    for (const auto &ex : v_ex) {
        retval.append(to_sympy_impl(func_map, ex));
    }

    return retval;
}

} // namespace

} // namespace detail

// Helper to setup the sympy integration bits
// on the C++ side.
void setup_sympy(py::module &m)
{
    bool has_sympy = true;

    try {
        detail::spy = py::module_::import("sympy");
    } catch (...) {
        has_sympy = false;
    }

    if (has_sympy) {
        // Fill in the function map.
        detail::fmap[typeid(hy::detail::acos_impl)] = py::object(detail::spy->attr("acos"));
        detail::fmap[typeid(hy::detail::acosh_impl)] = py::object(detail::spy->attr("acosh"));
        detail::fmap[typeid(hy::detail::asin_impl)] = py::object(detail::spy->attr("asin"));
        detail::fmap[typeid(hy::detail::asinh_impl)] = py::object(detail::spy->attr("asinh"));
        detail::fmap[typeid(hy::detail::atan_impl)] = py::object(detail::spy->attr("atan"));
        detail::fmap[typeid(hy::detail::atan2_impl)] = py::object(detail::spy->attr("atan2"));
        detail::fmap[typeid(hy::detail::atanh_impl)] = py::object(detail::spy->attr("atanh"));
        detail::fmap[typeid(hy::detail::cos_impl)] = py::object(detail::spy->attr("cos"));
        detail::fmap[typeid(hy::detail::cosh_impl)] = py::object(detail::spy->attr("cosh"));
        detail::fmap[typeid(hy::detail::erf_impl)] = py::object(detail::spy->attr("erf"));
        detail::fmap[typeid(hy::detail::exp_impl)] = py::object(detail::spy->attr("exp"));
        detail::fmap[typeid(hy::detail::log_impl)] = py::object(detail::spy->attr("log"));
        detail::fmap[typeid(hy::detail::sin_impl)] = py::object(detail::spy->attr("sin"));
        detail::fmap[typeid(hy::detail::sinh_impl)] = py::object(detail::spy->attr("sinh"));
        detail::fmap[typeid(hy::detail::tan_impl)] = py::object(detail::spy->attr("tan"));
        detail::fmap[typeid(hy::detail::tanh_impl)] = py::object(detail::spy->attr("tanh"));
        detail::fmap[typeid(hy::detail::sum_impl)] = py::object(detail::spy->attr("Add"));
        detail::fmap[typeid(hy::detail::prod_impl)] = py::object(detail::spy->attr("Mul"));

        // Special case pow() to detect sqrt().
        detail::fmap[typeid(hy::detail::pow_impl)]
            = [](std::unordered_map<const void *, py::object> &func_map, const hy::func &f) -> py::object {
            assert(f.args().size() == 2u);

            const auto &base = f.args()[0];
            const auto &expo = f.args()[1];

            if (const auto *num_ptr = std::get_if<hy::number>(&expo.value());
                num_ptr != nullptr && std::visit([](const auto &v) { return v == .5; }, num_ptr->value())) {
                return detail::spy->attr("sqrt")(detail::to_sympy_impl(func_map, base));
            } else {
                return detail::spy->attr("Pow")(detail::to_sympy_impl(func_map, base),
                                                detail::to_sympy_impl(func_map, expo));
            }
        };

        // kepE, kepF, kepDE.
        // NOTE: these will remain unevaluated functions.
        auto sympy_kepE = py::object(detail::spy->attr("Function")("heyoka_kepE"));
        detail::fmap[typeid(hy::detail::kepE_impl)] = sympy_kepE;

        auto sympy_kepF = py::object(detail::spy->attr("Function")("heyoka_kepF"));
        detail::fmap[typeid(hy::detail::kepF_impl)] = sympy_kepF;

        auto sympy_kepDE = py::object(detail::spy->attr("Function")("heyoka_kepDE"));
        detail::fmap[typeid(hy::detail::kepDE_impl)] = sympy_kepDE;

        // relu, relup and leaky variants.
        // NOTE: these are implemented as piecewise functions:
        // https://medium.com/@mathcube7/piecewise-functions-in-pythons-sympy-83f857948d3
        detail::fmap[typeid(hy::detail::relu_impl)]
            = [](std::unordered_map<const void *, py::object> &func_map, const hy::func &f) -> py::object {
            assert(f.args().size() == 1u);

            // Convert the argument to SymPy.
            auto s_arg = detail::to_sympy_impl(func_map, f.args()[0]);

            // Fetch the slope value.
            const auto slope = f.extract<hy::detail::relu_impl>()->get_slope();

            // Create the condition arg > 0.
            auto cond = s_arg.attr("__gt__")(0);

            // Fetch the piecewise function.
            auto pw = detail::spy->attr("Piecewise");

            if (slope == 0) {
                return pw(py::make_tuple(s_arg, cond), py::make_tuple(0., true));
            } else {
                return pw(py::make_tuple(s_arg, cond), py::make_tuple(py::cast(slope) * s_arg, true));
            }
        };

        detail::fmap[typeid(hy::detail::relup_impl)]
            = [](std::unordered_map<const void *, py::object> &func_map, const hy::func &f) -> py::object {
            assert(f.args().size() == 1u);

            // Convert the argument to SymPy.
            auto s_arg = detail::to_sympy_impl(func_map, f.args()[0]);

            // Fetch the slope value.
            const auto slope = f.extract<hy::detail::relup_impl>()->get_slope();

            // Create the condition arg > 0.
            auto cond = s_arg.attr("__gt__")(0);

            // Fetch the piecewise function.
            auto pw = detail::spy->attr("Piecewise");

            if (slope == 0) {
                return pw(py::make_tuple(1., cond), py::make_tuple(0., true));
            } else {
                return pw(py::make_tuple(1., cond), py::make_tuple(slope, true));
            }
        };

        // sigmoid.
        detail::fmap[typeid(hy::detail::sigmoid_impl)]
            = [](std::unordered_map<const void *, py::object> &func_map, const hy::func &f) {
                  assert(f.args().size() == 1u);

                  return py::cast(1.)
                         / (py::cast(1.) + detail::spy->attr("exp")(-detail::to_sympy_impl(func_map, f.args()[0])));
              };

        // time.
        // NOTE: this will remain an unevaluated nullary function.
        auto sympy_time = py::object(detail::spy->attr("Function")("heyoka_time"));
        detail::fmap[typeid(hy::detail::time_impl)] = sympy_time;

        // Constants.
        detail::fmap[typeid(hy::constant)] = [](std::unordered_map<const void *, py::object> &, const hy::func &f) {
            const auto *cptr = f.extract<hy::constant>();
            assert(cptr != nullptr);

            if (cptr->get_str_func_t() == typeid(hy::detail::pi_constant_func)) {
                return py::object(detail::spy->attr("pi"));
            }

            // Translate other constants as unevaluated nullary functions.
            return py::object(detail::spy->attr("Function")(f.get_name().c_str()));
        };

        // Expose the conversion function.
        m.def("to_sympy", [](const std::variant<hy::expression, std::vector<hy::expression>> &arg) {
            return std::visit([](const auto &v) -> std::variant<py::object, py::list> { return detail::to_sympy(v); },
                              arg);
        });

        // Register a cleanup function to destroy the global variables at shutdown.
        auto atexit = py::module_::import("atexit");
        atexit.attr("register")(py::cpp_function([]() {
#if !defined(NDEBUG)
            std::cout << "Cleaning up sympy conversion data." << std::endl;
#endif

            detail::spy.reset();
            detail::fmap.clear();
        }));
    } else {
        m.def("to_sympy", [](const hy::expression &) {
            py_throw(PyExc_ImportError, "The 'to_sympy()' function is not available because sympy is not installed");
        });
    }
}

} // namespace heyoka_py
