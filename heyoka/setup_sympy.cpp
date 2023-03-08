// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <variant>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#include <fmt/format.h>

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
    // NOTE: heyoka symbols can assume only
    // real values.
    py::kwargs kwa;
    kwa["real"] = true;

    return spy->attr("Symbol")(var.name(), **kwa);
}

py::object to_sympy_impl(std::unordered_map<const void *, py::object> &, const hy::param &par)
{
    // NOTE: params are converted to symbolic variables
    // following a naming convention.

    // NOTE: heyoka params can assume only
    // real values.
    py::kwargs kwa;
    kwa["real"] = true;

    return spy->attr("Symbol")(fmt::format("par[{}]", par.idx()), **kwa);
}

// Number conversion corresponds to casting to python
// the numerical value.
py::object to_sympy_impl(std::unordered_map<const void *, py::object> &, const hy::number &num)
{
    return std::visit(
        [&num](const auto &x) -> py::object {
            using std::isfinite;

            // NOTE: forbid conversion if the value is not finite.
            if (!isfinite(x)) {
                py_throw(PyExc_ValueError,
                         (fmt::format("Cannot convert to sympy the nonfinite number {}", hy::expression{num})).c_str());
            }

#if defined(HEYOKA_HAVE_REAL128)
            if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(x)>>, mppp::real128>) {
                return spy->attr("Float")(x.to_string(), py::none{}, std::numeric_limits<mppp::real128>::digits);
            }
#endif

#if defined(HEYOKA_HAVE_REAL)
            if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(x)>>, mppp::real>) {
                return spy->attr("Float")(x.to_string(), py::none{}, x.get_prec());
            }
#endif

            return py::cast(x);
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
        detail::fmap[typeid(hy::detail::sqrt_impl)] = py::object(detail::spy->attr("sqrt"));
        detail::fmap[typeid(hy::detail::tan_impl)] = py::object(detail::spy->attr("tan"));
        detail::fmap[typeid(hy::detail::tanh_impl)] = py::object(detail::spy->attr("tanh"));
        detail::fmap[typeid(hy::detail::pow_impl)] = py::object(detail::spy->attr("Pow"));
        detail::fmap[typeid(hy::detail::sum_impl)] = py::object(detail::spy->attr("Add"));

        // sum_sq.
        detail::fmap[typeid(hy::detail::sum_sq_impl)]
            = [](std::unordered_map<const void *, py::object> &func_map, const hy::func &f) {
                  // Convert all arguments to Python objects.
                  py::list py_args;
                  for (const auto &arg : f.args()) {
                      auto tmp = detail::to_sympy_impl(func_map, arg);
                      py_args.append(tmp * tmp);
                  }

                  return py::object(detail::spy->attr("Add"))(*py_args);
              };

        // Binary op.
        detail::fmap[typeid(hy::detail::binary_op)]
            = [](std::unordered_map<const void *, py::object> &func_map, const hy::func &f) {
                  assert(f.args().size() == 2u);

                  auto op0 = detail::to_sympy_impl(func_map, f.args()[0]);
                  auto op1 = detail::to_sympy_impl(func_map, f.args()[1]);

                  const auto op_type = f.extract<hy::detail::binary_op>()->op();

                  switch (op_type) {
                      case hy::detail::binary_op::type::add:
                          return op0 + op1;
                      case hy::detail::binary_op::type::sub:
                          return op0 - op1;
                      case hy::detail::binary_op::type::mul:
                          return op0 * op1;
                      default:
                          assert(op_type == hy::detail::binary_op::type::div);
                          return op0 / op1;
                  }
              };

        // kepE.
        // NOTE: this will remain an unevaluated binary function.
        auto sympy_kepE = py::object(detail::spy->attr("Function")("heyoka_kepE"));
        detail::fmap[typeid(hy::detail::kepE_impl)] = sympy_kepE;

        // neg.
        detail::fmap[typeid(hy::detail::neg_impl)]
            = [](std::unordered_map<const void *, py::object> &func_map, const hy::func &f) {
                  assert(f.args().size() == 1u);

                  return -detail::to_sympy_impl(func_map, f.args()[0]);
              };

        // sigmoid.
        detail::fmap[typeid(hy::detail::sigmoid_impl)]
            = [](std::unordered_map<const void *, py::object> &func_map, const hy::func &f) {
                  assert(f.args().size() == 1u);

                  return py::cast(1.)
                         / (py::cast(1.) + detail::spy->attr("exp")(-detail::to_sympy_impl(func_map, f.args()[0])));
              };

        // square.
        detail::fmap[typeid(hy::detail::square_impl)]
            = [](std::unordered_map<const void *, py::object> &func_map, const hy::func &f) {
                  assert(f.args().size() == 1u);

                  auto op = detail::to_sympy_impl(func_map, f.args()[0]);
                  return op * op;
              };

        // time.
        // NOTE: this will remain an unevaluated nullary function.
        auto sympy_time = py::object(detail::spy->attr("Function")("heyoka_time"));
        detail::fmap[typeid(hy::detail::time_impl)] = sympy_time;

        // tpoly.
        // NOTE: this will remain an unevaluated binary function.
        auto sympy_tpoly = py::object(detail::spy->attr("Function")("heyoka_tpoly"));
        detail::fmap[typeid(hy::detail::tpoly_impl)] = sympy_tpoly;

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
        m.def("to_sympy", &detail::to_sympy);

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
