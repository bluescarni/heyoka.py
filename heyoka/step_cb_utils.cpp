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
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <pybind11/pybind11.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

#include "common_utils.hpp"
#include "step_cb_utils.hpp"

namespace heyoka_py
{

namespace py = pybind11;

step_cb_wrapper::step_cb_wrapper(py::object *obj_ptr) : m_obj_ptr(obj_ptr)
{
    assert(m_obj_ptr != nullptr);

    if (!callable(*m_obj_ptr)) {
        py_throw(PyExc_TypeError,
                 fmt::format("An object of type '{}' cannot be used as a step callback because it is not callable",
                             str(type(*m_obj_ptr)))
                     .c_str());
    }
}

step_cb_wrapper::step_cb_wrapper(const step_cb_wrapper &)
{
    // NOTE: we should never end up copying step_cb_wrapper.
    // Unfortunately, we cannot delete this constructor as step_callback
    // requires copy constructibility of the inner object.
    assert(false);
}

step_cb_wrapper::step_cb_wrapper(step_cb_wrapper &&) noexcept = default;

step_cb_wrapper::~step_cb_wrapper() = default;

template <typename TA>
bool step_cb_wrapper::operator()(TA &ta)
{
    assert(m_obj_ptr != nullptr);
    auto &obj = *m_obj_ptr;

    py::gil_scoped_acquire acquire;

    // NOTE: this will return a reference
    // to the Python object wrapping ta.
    auto ta_obj = py::cast(&ta);

    // Attempt to invoke the call operator of the
    // Pythonic callback.
    auto ret = obj(ta_obj);

    // NOTE: we want to manually check the conversion
    // of the return value because if that fails
    // the pybind11 error message is not very helpful, and thus
    // we try to provide a more detailed error message.
    try {
        return py::cast<bool>(ret);
    } catch (const py::cast_error &) {
        py_throw(PyExc_TypeError, (fmt::format("The call operator of a step callback is expected to return a boolean, "
                                               "but a value of type '{}' was returned instead",
                                               str(type(ret))))
                                      .c_str());
    }
}

template <typename TA>
void step_cb_wrapper::pre_hook(TA &ta)
{
    assert(m_obj_ptr != nullptr);
    auto &obj = *m_obj_ptr;

    py::gil_scoped_acquire acquire;

    if (py::hasattr(obj, "pre_hook")) {
        auto ta_obj = py::cast(&ta);

        obj.attr("pre_hook")(ta_obj);
    }
}

namespace detail
{

namespace
{

// Helper to turn an scb_t into a step_callback/step_callback_batch.
template <typename StepCallback>
StepCallback scb_to_step_callback(scb_t &s)
{
    return std::visit(
        []<typename T>(T &a) -> StepCallback {
            if constexpr (std::same_as<T, py::object>) {
                return step_cb_wrapper(&a);
            } else {
                return std::move(a);
            }
        },
        s);
}

// Helper to turn the input step_callback/step_callback_batch
// into an scb_t. The type stored in cb is inferred from the
// active type of s.
template <typename StepCallback>
scb_t step_callback_to_scb(const scb_t &s, StepCallback &cb)
{
    return std::visit(
        [&cb]<typename T>(const T &) -> scb_t {
            if constexpr (std::same_as<T, py::object>) {
                assert(value_isa<step_cb_wrapper>(cb));
                assert(value_ref<step_cb_wrapper>(cb).m_obj_ptr != nullptr);

                return *value_ref<step_cb_wrapper>(cb).m_obj_ptr;
            } else {
                assert(value_isa<T>(cb));

                return std::move(value_ref<T>(cb));
            }
        },
        s);
}

// Metaprogramming to select the step callback
// set corresponding to the input step callback type.
template <typename StepCallback>
struct make_callback_set {
};

template <typename T>
struct make_callback_set<heyoka::step_callback<T>> {
    using type = heyoka::step_callback_set<T>;
};

template <typename T>
struct make_callback_set<heyoka::step_callback_batch<T>> {
    using type = heyoka::step_callback_batch_set<T>;
};

} // namespace

} // namespace detail

// Helper to turn the input scb_arg_t into a step_callback/step_callback_batch.
template <typename StepCallback>
StepCallback scb_arg_to_step_callback(scb_arg_t &arg)
{
    // Select the step callback set corresponding to StepCallback.
    using scs_t = typename detail::make_callback_set<StepCallback>::type;

    return std::visit(
        []<typename T>(T &a) -> StepCallback {
            if constexpr (std::same_as<T, scb_t>) {
                // Single callback.
                return detail::scb_to_step_callback<StepCallback>(a);
            } else {
                // List of callbacks.
                std::vector<StepCallback> vec;
                vec.reserve(a.size());

                for (auto &s : a) {
                    vec.push_back(detail::scb_to_step_callback<StepCallback>(s));
                }

                return scs_t(std::move(vec));
            }
        },
        arg);
}

// Helper to turn the input step_callback/step_callback_batch into an scb_arg_t.
// The type stored in cb is inferred from the active type of scb_arg.
template <typename StepCallback>
scb_arg_t step_callback_to_scb_arg_t(const scb_arg_t &scb_arg, StepCallback &cb)
{
    // Select the step callback set corresponding to StepCallback.
    using scs_t = typename detail::make_callback_set<StepCallback>::type;

    return std::visit(
        [&cb]<typename T>(const T &a) -> scb_arg_t {
            if constexpr (std::same_as<T, scb_t>) {
                // Single callback.
                assert(!value_isa<scs_t>(cb));

                return detail::step_callback_to_scb(a, cb);
            } else {
                assert(value_isa<scs_t>(cb));

                auto &scs = value_ref<scs_t>(cb);

                assert(scs.size() == a.size());

                std::vector<scb_t> vec;
                vec.reserve(scs.size());

                for (decltype(scs.size()) i = 0; i < scs.size(); ++i) {
                    vec.push_back(detail::step_callback_to_scb(a[i], scs[i]));
                }

                return vec;
            }
        },
        scb_arg);
}

// Explicit instantiations.

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<float> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<float> &);

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<double> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<double> &);

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<long double> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<mppp::real128> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<mppp::real128> &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive<mppp::real> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive<mppp::real> &);

#endif

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive_batch<float> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive_batch<float> &);

template bool step_cb_wrapper::operator()(heyoka::taylor_adaptive_batch<double> &);
template void step_cb_wrapper::pre_hook(heyoka::taylor_adaptive_batch<double> &);

template heyoka::step_callback<float> scb_arg_to_step_callback(scb_arg_t &);
template scb_arg_t step_callback_to_scb_arg_t(const scb_arg_t &, heyoka::step_callback<float> &);

template heyoka::step_callback<double> scb_arg_to_step_callback(scb_arg_t &);
template scb_arg_t step_callback_to_scb_arg_t(const scb_arg_t &, heyoka::step_callback<double> &);

template heyoka::step_callback<long double> scb_arg_to_step_callback(scb_arg_t &);
template scb_arg_t step_callback_to_scb_arg_t(const scb_arg_t &, heyoka::step_callback<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template heyoka::step_callback<mppp::real128> scb_arg_to_step_callback(scb_arg_t &);
template scb_arg_t step_callback_to_scb_arg_t(const scb_arg_t &, heyoka::step_callback<mppp::real128> &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template heyoka::step_callback<mppp::real> scb_arg_to_step_callback(scb_arg_t &);
template scb_arg_t step_callback_to_scb_arg_t(const scb_arg_t &, heyoka::step_callback<mppp::real> &);

#endif

template heyoka::step_callback_batch<float> scb_arg_to_step_callback(scb_arg_t &);
template scb_arg_t step_callback_to_scb_arg_t(const scb_arg_t &, heyoka::step_callback_batch<float> &);

template heyoka::step_callback_batch<double> scb_arg_to_step_callback(scb_arg_t &);
template scb_arg_t step_callback_to_scb_arg_t(const scb_arg_t &, heyoka::step_callback_batch<double> &);

} // namespace heyoka_py
