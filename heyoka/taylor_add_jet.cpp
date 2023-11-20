// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/taylor.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "dtypes.hpp"
#include "taylor_add_jet.hpp"

#if defined(HEYOKA_HAVE_REAL)

#include "expose_real.hpp"

#endif

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;

namespace detail
{

namespace
{

template <typename T>
constexpr bool default_cm =
#if defined(HEYOKA_HAVE_REAL)
    std::is_same_v<T, mppp::real>
#else
    false
#endif
    ;

// Common helper to check the shapes/dimensions
// of the arrays used as inputs for the function
// computing the jet of derivatives.
template <typename T, typename Arr>
void taylor_add_jet_array_check(const Arr &state, const std::optional<Arr> &pars, const std::optional<Arr> &time,
                                std::uint32_t n_params, std::uint32_t order, std::uint32_t tot_n_eq,
                                std::uint32_t batch_size)
{
    // Distinguish the batch and scalar cases when checking
    // dimensions and shapes.
    if (batch_size == 1u) {
        // We need a state array with shape (order + 1, tot_n_eq).
        if (state.ndim() != 2) {
            py_throw(PyExc_ValueError,
                     fmt::format("Invalid state vector passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the number of dimensions must be 2, but it is "
                                 "{} instead",
                                 state.ndim())
                         .c_str());
        }

        if (state.shape(0) != boost::numeric_cast<py::ssize_t>(order + 1u)
            || state.shape(1) != boost::numeric_cast<py::ssize_t>(tot_n_eq)) {
            py_throw(PyExc_ValueError,
                     fmt::format("Invalid state vector passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the shape must be ({}, {}), but it is "
                                 "({}, {}) instead",
                                 order + 1u, tot_n_eq, state.shape(0), state.shape(1))
                         .c_str());
        }

        if (pars) {
            // The pars array must have shape (n_params).
            if (pars->ndim() != 1) {
                py_throw(PyExc_ValueError,
                         fmt::format("Invalid parameters vector passed to a function for the computation of "
                                     "the jet of "
                                     "Taylor derivatives: the number of dimensions must be 1, but it is "
                                     "{} instead",
                                     pars->ndim())
                             .c_str());
            }

            if (pars->shape(0) != boost::numeric_cast<py::ssize_t>(n_params)) {
                py_throw(PyExc_ValueError, fmt::format("Invalid parameters vector passed to a function for the "
                                                       "computation of the jet of "
                                                       "Taylor derivatives: the shape must be ({}, ), but it is "
                                                       "({}) instead",
                                                       n_params, pars->shape(0))
                                               .c_str());
            }
        }

        if (time) {
            // The time vector must have shape (1).
            if (time->ndim() != 1) {
                py_throw(PyExc_ValueError,
                         fmt::format("Invalid time vector passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the number of dimensions must be 1, but it is "
                                     "{} instead",
                                     time->ndim())
                             .c_str());
            }

            if (time->shape(0) != 1) {
                py_throw(PyExc_ValueError,
                         fmt::format("Invalid time vector passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the shape must be (1, ), but it is "
                                     "({}) instead",
                                     time->shape(0))
                             .c_str());
            }
        }
    } else {
        // We need a state array with shape (order + 1, tot_n_eq, batch_size).
        if (state.ndim() != 3) {
            py_throw(PyExc_ValueError,
                     fmt::format("Invalid state vector passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the number of dimensions must be 3, but it is "
                                 "{} instead",
                                 state.ndim())
                         .c_str());
        }

        if (state.shape(0) != boost::numeric_cast<py::ssize_t>(order + 1u)
            || state.shape(1) != boost::numeric_cast<py::ssize_t>(tot_n_eq)
            || state.shape(2) != boost::numeric_cast<py::ssize_t>(batch_size)) {
            py_throw(PyExc_ValueError,
                     fmt::format("Invalid state vector passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the shape must be ({}, {}, {}), but it is "
                                 "({}, {}, {}) instead",
                                 order + 1u, tot_n_eq, batch_size, state.shape(0), state.shape(1), state.shape(2))
                         .c_str());
        }

        if (pars) {
            // The pars array must have shape (n_params, batch_size).
            if (pars->ndim() != 2) {
                py_throw(PyExc_ValueError,
                         fmt::format("Invalid parameters vector passed to a function for the computation of "
                                     "the jet of "
                                     "Taylor derivatives: the number of dimensions must be 2, but it is "
                                     "{} instead",
                                     pars->ndim())
                             .c_str());
            }

            if (pars->shape(0) != boost::numeric_cast<py::ssize_t>(n_params)
                || pars->shape(1) != boost::numeric_cast<py::ssize_t>(batch_size)) {
                py_throw(PyExc_ValueError,
                         fmt::format("Invalid parameters vector passed to a function for the computation of the jet "
                                     "of "
                                     "Taylor derivatives: the shape must be ({}, {}), but it is "
                                     "({}, {}) instead",
                                     n_params, batch_size, pars->shape(0), pars->shape(1))
                             .c_str());
            }
        }

        if (time) {
            // The time vector must have shape (batch_size).
            if (time->ndim() != 1) {
                py_throw(PyExc_ValueError,
                         fmt::format("Invalid time vector passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the number of dimensions must be 1, but it is "
                                     "{} instead",
                                     time->ndim())
                             .c_str());
            }

            if (time->shape(0) != boost::numeric_cast<py::ssize_t>(batch_size)) {
                py_throw(PyExc_ValueError,
                         fmt::format("Invalid time vector passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the shape must be ({}, ), but it is "
                                     "({}) instead",
                                     batch_size, time->shape(0))
                             .c_str());
            }
        }
    }
}

template <typename T, typename U>
void expose_taylor_add_jet_impl(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    namespace kw = hey::kw;

    m.def(
        name,
        [](const U &sys, std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
           const std::vector<hey::expression> &sv_funcs, bool parallel_mode, unsigned opt_level, bool force_avx512,
           bool slp_vectorize, bool fast_math, long long prec) {
            // Forbid batch sizes > 1 for everything but double.
            if (!std::is_same_v<T, double> && batch_size > 1u) {
                py_throw(PyExc_ValueError, "Batch sizes greater than 1 are not supported for this floating-point type");
            }

            // Let's figure out if sys contains params and/or time.
            bool is_time_dependent = false;
            std::uint32_t n_params = 0;

            if constexpr (std::is_same_v<U, std::vector<std::pair<hey::expression, hey::expression>>>) {
                for (const auto &[_, ex] : sys) {
                    is_time_dependent = is_time_dependent || hey::is_time_dependent(ex);
                    n_params = std::max<std::uint32_t>(n_params, hey::get_param_size(ex));
                }
            } else {
                for (const auto &ex : sys) {
                    is_time_dependent = is_time_dependent || hey::is_time_dependent(ex);
                    n_params = std::max<std::uint32_t>(n_params, hey::get_param_size(ex));
                }
            }

            // Cache the number of equations and sv_funcs.
            const auto n_eq = sys.size();
            const auto n_sv_funcs = sv_funcs.size();

            // Add the jet function.
            using jptr_t = void (*)(T *, const T *, const T *);
            jptr_t jptr = nullptr;
            hey::llvm_state s{kw::opt_level = opt_level, kw::force_avx512 = force_avx512,
                              kw::slp_vectorize = slp_vectorize, kw::fast_math = fast_math};

            {
                // NOTE: release the GIL during compilation.
                py::gil_scoped_release release;

                // NOTE: this will throw in case of an invalid prec value.
                hey::taylor_add_jet<T>(s, "jet", sys, order, batch_size, high_accuracy, compact_mode, sv_funcs,
                                       parallel_mode, prec);

                s.compile();

                jptr = reinterpret_cast<jptr_t>(s.jit_lookup("jet"));
            }

            // Build and return the Python wrapper for jptr.
            return py::cpp_function(
                [s = std::move(s), batch_size, order, is_time_dependent, n_params,
                 tot_n_eq = static_cast<std::uint32_t>(n_eq) + static_cast<std::uint32_t>(n_sv_funcs), jptr,
                 prec](const py::iterable &state_ob, std::optional<py::iterable> pars_ob,
                       std::optional<py::iterable> time_ob) {
                    (void)prec;

                    // Attempt to turn the input objects into arrays.
                    py::array state = state_ob;
                    std::optional<py::array> pars = pars_ob ? *pars_ob : std::optional<py::array>{};
                    std::optional<py::array> time = time_ob ? *time_ob : std::optional<py::array>{};

                    // Enforce the correct dtype for all arrays.
                    const auto dt = get_dtype<T>();
                    if (state.dtype().num() != dt) {
                        state = state.attr("astype")(py::dtype(dt), "casting"_a = "safe");
                    }
                    if (pars && pars->dtype().num() != dt) {
                        *pars = pars->attr("astype")(py::dtype(dt), "casting"_a = "safe");
                    }
                    if (time && time->dtype().num() != dt) {
                        *time = time->attr("astype")(py::dtype(dt), "casting"_a = "safe");
                    }

                    // Check the input arrays. They must all be C-style contiguous and properly aligned,
                    // and they cannot share any memory.
                    if (!is_npy_array_carray(state, true)) {
                        py_throw(PyExc_ValueError,
                                 "Invalid state vector passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the NumPy array is not C contiguous or not writeable");
                    }
                    if (pars && !is_npy_array_carray(*pars)) {
                        py_throw(PyExc_ValueError,
                                 "Invalid parameters vector passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the NumPy array is not C contiguous");
                    }
                    if (time && !is_npy_array_carray(*time)) {
                        py_throw(PyExc_ValueError,
                                 "Invalid time vector passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the NumPy array is not C contiguous");
                    }

                    bool maybe_share_memory = false;
                    if (pars && time) {
                        maybe_share_memory = may_share_memory(state, *pars, *time);
                    } else if (pars) {
                        maybe_share_memory = may_share_memory(state, *pars);
                    } else if (time) {
                        maybe_share_memory = may_share_memory(state, *time);
                    }

                    if (maybe_share_memory) {
                        py_throw(PyExc_ValueError,
                                 "Invalid vectors passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the NumPy arrays must not share any memory");
                    }

                    // Get out the raw pointers
                    auto *s_ptr = static_cast<T *>(state.mutable_data());
                    const auto *p_ptr = pars ? static_cast<const T *>(pars->data()) : nullptr;
                    const auto *t_ptr = time ? static_cast<const T *>(time->data()) : nullptr;

                    // Check that, if there are params or time, the corresponding
                    // arrays have been provided.
                    if (n_params > 0u && p_ptr == nullptr) {
                        py_throw(PyExc_ValueError,
                                 "Invalid vectors passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the ODE system contains parameters, but no parameter array was "
                                 "passed as input argument");
                    }

                    if (is_time_dependent && t_ptr == nullptr) {
                        py_throw(PyExc_ValueError,
                                 "Invalid vectors passed to a function for the computation of the jet of "
                                 "Taylor derivatives: the ODE system is non-autonomous, but no time array was "
                                 "passed as input argument");
                    }

                    // Check the shapes/dims of the input arrays.
                    taylor_add_jet_array_check<T>(state, pars, time, n_params, order, tot_n_eq, batch_size);

#if defined(HEYOKA_HAVE_REAL)

                    if constexpr (std::is_same_v<T, mppp::real>) {
                        // For mppp::real, check that all elements in the provided arrays
                        // have been initialised with the expected precision.
                        // NOTE: here we will error out even if no pars/time are needed
                        // but the user provided them anyway, uninited or with incorrect precision.
                        pyreal_check_array(state, boost::numeric_cast<mpfr_prec_t>(prec));
                        if (pars) {
                            pyreal_check_array(*pars, boost::numeric_cast<mpfr_prec_t>(prec));
                        }
                        if (time) {
                            pyreal_check_array(*time, boost::numeric_cast<mpfr_prec_t>(prec));
                        }
                    }

#endif

                    // NOTE: here it may be that p_ptr and/or t_ptr are provided
                    // but the system has not time/params. In that case, the pointers
                    // will never be used in jptr.
                    jptr(s_ptr, p_ptr, t_ptr);

                    return state;
                },
                "state"_a, "pars"_a = py::none{}, "time"_a = py::none{});
        },
        "sys"_a, "order"_a, "batch_size"_a = 1u, "high_accuracy"_a = false, "compact_mode"_a = default_cm<T>,
        "sv_funcs"_a = py::list{}, "parallel_mode"_a = false, "opt_level"_a.noconvert() = 3,
        "force_avx512"_a.noconvert() = false, "slp_vectorize"_a.noconvert() = false, "fast_math"_a.noconvert() = false,
        "prec"_a.noconvert() = 0);
}

} // namespace

} // namespace detail

void expose_taylor_add_jet_flt(py::module &m)
{
    detail::expose_taylor_add_jet_impl<float, std::vector<std::pair<hey::expression, hey::expression>>>(
        m, "_taylor_add_jet_flt");
    detail::expose_taylor_add_jet_impl<float, std::vector<hey::expression>>(m, "_taylor_add_jet_flt");
}

void expose_taylor_add_jet_dbl(py::module &m)
{
    detail::expose_taylor_add_jet_impl<double, std::vector<std::pair<hey::expression, hey::expression>>>(
        m, "_taylor_add_jet_dbl");
    detail::expose_taylor_add_jet_impl<double, std::vector<hey::expression>>(m, "_taylor_add_jet_dbl");
}

void expose_taylor_add_jet_ldbl(py::module &m)
{
    detail::expose_taylor_add_jet_impl<long double, std::vector<std::pair<hey::expression, hey::expression>>>(
        m, "_taylor_add_jet_ldbl");
    detail::expose_taylor_add_jet_impl<long double, std::vector<hey::expression>>(m, "_taylor_add_jet_ldbl");
}

#if defined(HEYOKA_HAVE_REAL128)

void expose_taylor_add_jet_f128(py::module &m)
{
    detail::expose_taylor_add_jet_impl<mppp::real128, std::vector<std::pair<hey::expression, hey::expression>>>(
        m, "_taylor_add_jet_f128");
    detail::expose_taylor_add_jet_impl<mppp::real128, std::vector<hey::expression>>(m, "_taylor_add_jet_f128");
}

#endif

#if defined(HEYOKA_HAVE_REAL)

void expose_taylor_add_jet_real(py::module &m)
{
    detail::expose_taylor_add_jet_impl<mppp::real, std::vector<std::pair<hey::expression, hey::expression>>>(
        m, "_taylor_add_jet_real");
    detail::expose_taylor_add_jet_impl<mppp::real, std::vector<hey::expression>>(m, "_taylor_add_jet_real");
}

#endif

} // namespace heyoka_py
