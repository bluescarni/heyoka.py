// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include "common_utils.hpp"
#include "long_double_caster.hpp"
#include "taylor_add_jet.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;

namespace detail
{

namespace
{

// Common helper to check the shapes/dimensions
// of the arrays used as inputs for the function
// computing the jet of derivatives.
template <typename Arr>
void taylor_add_jet_array_check(const Arr &state, const std::optional<Arr> &pars, const std::optional<Arr> &time,
                                std::uint32_t n_params, bool has_time, std::uint32_t order, std::uint32_t tot_n_eq,
                                std::uint32_t batch_size)
{
    using fmt::literals::operator""_format;

    // Distinguish the batch and scalar cases when checking
    // dimensions and shapes.
    if (batch_size == 1u) {
        // We need a state array with shape (order + 1, tot_n_eq).
        if (state.ndim() != 2) {
            py_throw(PyExc_ValueError, "Invalid state vector passed to a function for the computation of the jet of "
                                       "Taylor derivatives: the number of dimensions must be 2, but it is "
                                       "{} instead"_format(state.ndim())
                                           .c_str());
        }

        if (state.shape(0) != boost::numeric_cast<py::ssize_t>(order + 1u)
            || state.shape(1) != boost::numeric_cast<py::ssize_t>(tot_n_eq)) {
            py_throw(PyExc_ValueError, "Invalid state vector passed to a function for the computation of the jet of "
                                       "Taylor derivatives: the shape must be ({}, {}), but it is "
                                       "({}, {}) instead"_format(order + 1u, tot_n_eq, state.shape(0), state.shape(1))
                                           .c_str());
        }

        if (n_params > 0u) {
            // The pars array must have shape (n_params).
            if (pars->ndim() != 1) {
                py_throw(PyExc_ValueError, "Invalid parameters vector passed to a function for the computation of "
                                           "the jet of "
                                           "Taylor derivatives: the number of dimensions must be 1, but it is "
                                           "{} instead"_format(pars->ndim())
                                               .c_str());
            }

            if (pars->shape(0) != boost::numeric_cast<py::ssize_t>(n_params)) {
                py_throw(PyExc_ValueError, "Invalid parameters vector passed to a function for the "
                                           "computation of the jet of "
                                           "Taylor derivatives: the shape must be ({}), but it is "
                                           "({}) instead"_format(n_params, pars->shape(0))
                                               .c_str());
            }
        }

        if (has_time) {
            // The time vector must have shape (1).
            if (time->ndim() != 1) {
                py_throw(PyExc_ValueError, "Invalid time vector passed to a function for the computation of the jet of "
                                           "Taylor derivatives: the number of dimensions must be 1, but it is "
                                           "{} instead"_format(time->ndim())
                                               .c_str());
            }

            if (time->shape(0) != 1) {
                py_throw(PyExc_ValueError, "Invalid time vector passed to a function for the computation of the jet of "
                                           "Taylor derivatives: the shape must be (1), but it is "
                                           "({}) instead"_format(time->shape(0))
                                               .c_str());
            }
        }
    } else {
        // We need a state array with shape (order + 1, tot_n_eq, batch_size).
        if (state.ndim() != 3) {
            py_throw(PyExc_ValueError, "Invalid state vector passed to a function for the computation of the jet of "
                                       "Taylor derivatives: the number of dimensions must be 3, but it is "
                                       "{} instead"_format(state.ndim())
                                           .c_str());
        }

        if (state.shape(0) != boost::numeric_cast<py::ssize_t>(order + 1u)
            || state.shape(1) != boost::numeric_cast<py::ssize_t>(tot_n_eq)
            || state.shape(2) != boost::numeric_cast<py::ssize_t>(batch_size)) {
            py_throw(PyExc_ValueError, "Invalid state vector passed to a function for the computation of the jet of "
                                       "Taylor derivatives: the shape must be ({}, {}, {}), but it is "
                                       "({}, {}, {}) instead"_format(order + 1u, tot_n_eq, batch_size, state.shape(0),
                                                                     state.shape(1), state.shape(2))
                                           .c_str());
        }

        if (n_params > 0u) {
            // The pars array must have shape (n_params, batch_size).
            if (pars->ndim() != 2) {
                py_throw(PyExc_ValueError, "Invalid parameters vector passed to a function for the computation of "
                                           "the jet of "
                                           "Taylor derivatives: the number of dimensions must be 2, but it is "
                                           "{} instead"_format(pars->ndim())
                                               .c_str());
            }

            if (pars->shape(0) != boost::numeric_cast<py::ssize_t>(n_params)
                || pars->shape(1) != boost::numeric_cast<py::ssize_t>(batch_size)) {
                py_throw(PyExc_ValueError,
                         "Invalid parameters vector passed to a function for the computation of the jet "
                         "of "
                         "Taylor derivatives: the shape must be ({}, {}), but it is "
                         "({}, {}) instead"_format(n_params, batch_size, pars->shape(0), pars->shape(1))
                             .c_str());
            }
        }

        if (has_time) {
            // The time vector must have shape (batch_size).
            if (time->ndim() != 1) {
                py_throw(PyExc_ValueError, "Invalid time vector passed to a function for the computation of the jet of "
                                           "Taylor derivatives: the number of dimensions must be 1, but it is "
                                           "{} instead"_format(time->ndim())
                                               .c_str());
            }

            if (time->shape(0) != boost::numeric_cast<py::ssize_t>(batch_size)) {
                py_throw(PyExc_ValueError, "Invalid time vector passed to a function for the computation of the jet of "
                                           "Taylor derivatives: the shape must be ({}), but it is "
                                           "({}) instead"_format(batch_size, time->shape(0))
                                               .c_str());
            }
        }
    }
}

template <typename T, typename U>
void expose_taylor_add_jet_impl(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    m.def(
        name,
        [](U sys, std::uint32_t order, std::uint32_t batch_size, bool high_accuracy, bool compact_mode,
           std::vector<hey::expression> sv_funcs) {
            // Let's figure out if sys contains params and/or time.
            bool has_time = false;
            std::uint32_t n_params = 0;

            if constexpr (std::is_same_v<U, std::vector<std::pair<hey::expression, hey::expression>>>) {
                for (const auto &[_, ex] : sys) {
                    has_time = has_time || hey::has_time(ex);
                    n_params = std::max<std::uint32_t>(n_params, hey::get_param_size(ex));
                }
            } else {
                for (const auto &ex : sys) {
                    has_time = has_time || hey::has_time(ex);
                    n_params = std::max<std::uint32_t>(n_params, hey::get_param_size(ex));
                }
            }

            // Cache the number of equations and sv_funcs.
            const auto n_eq = sys.size();
            const auto n_sv_funcs = sv_funcs.size();

            // Add the jet function.
            using jptr_t = void (*)(T *, const T *, const T *);
            jptr_t jptr;
            hey::llvm_state s;

            {
                // NOTE: release the GIL during compilation.
                py::gil_scoped_release release;

                hey::taylor_add_jet<T>(s, "jet", std::move(sys), order, batch_size, high_accuracy, compact_mode,
                                       std::move(sv_funcs));

                s.compile();

                jptr = reinterpret_cast<jptr_t>(s.jit_lookup("jet"));
            }

#if defined(HEYOKA_HAVE_REAL128)
            if constexpr (std::is_same_v<T, mppp::real128>) {
                // Build and return the Python wrapper for jptr.
                return py::cpp_function(
                    [s = std::move(s), batch_size, order, has_time, n_params,
                     tot_n_eq = static_cast<std::uint32_t>(n_eq) + static_cast<std::uint32_t>(n_sv_funcs),
                     jptr](py::array state, std::optional<py::array> pars, std::optional<py::array> time) {
                        // Check that, if there are params or time, the corresponding
                        // arrays have been provided.
                        if (n_params > 0u && !pars) {
                            py_throw(
                                PyExc_ValueError,
                                "Invalid vectors passed to a function for the computation of the jet of "
                                "Taylor derivatives: the ODE system contains parameters, but no parameter array was "
                                "passed as input argument");
                        }

                        if (has_time && !time) {
                            py_throw(PyExc_ValueError,
                                     "Invalid vectors passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the ODE system is non-autonomous, but no time array was "
                                     "passed as input argument");
                        }

                        // Check the shapes/dims of the input arrays.
                        taylor_add_jet_array_check(state, pars, time, n_params, has_time, order, tot_n_eq, batch_size);

                        // Convert the arrays into vectors.
                        auto state_vec = py::cast<std::vector<mppp::real128>>(state.attr("flatten")());

                        std::vector<mppp::real128> pars_vec, time_vec;
                        if (n_params > 0u) {
                            pars_vec = py::cast<std::vector<mppp::real128>>(pars->attr("flatten")());
                        }
                        if (has_time) {
                            time_vec = py::cast<std::vector<mppp::real128>>(time->attr("flatten")());
                        }

                        jptr(state_vec.data(), pars_vec.data(), time_vec.data());

                        auto ret = py::array(py::cast(state_vec));

                        if (batch_size == 1u) {
                            ret.resize(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(order + 1u),
                                                                 boost::numeric_cast<py::ssize_t>(tot_n_eq)});
                        } else {
                            ret.resize(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(order + 1u),
                                                                 boost::numeric_cast<py::ssize_t>(tot_n_eq),
                                                                 boost::numeric_cast<py::ssize_t>(batch_size)});
                        }

                        return ret;
                    },
                    "state"_a, "pars"_a = py::none{}, "time"_a = py::none{});
            } else {
#endif
                // Build and return the Python wrapper for jptr.
                return py::cpp_function(
                    [s = std::move(s), batch_size, order, has_time, n_params,
                     tot_n_eq = static_cast<std::uint32_t>(n_eq) + static_cast<std::uint32_t>(n_sv_funcs), jptr](
                        py::array_t<T> state, std::optional<py::array_t<T>> pars, std::optional<py::array_t<T>> time) {
                        // Check the input arrays.
                        // NOTE: it looks like c_style does not necessarily imply well-aligned:
                        // https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_FromAny.NPY_ARRAY_CARRAY
                        // If this becomes a problem, we can tap into the NumPy C API and do additional
                        // flag checking:
                        // https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_CHKFLAGS
                        if (!(state.flags() & py::array::c_style)) {
                            py_throw(PyExc_ValueError,
                                     "Invalid state vector passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the NumPy array is not C contiguous");
                        }
                        if (!state.writeable()) {
                            py_throw(PyExc_ValueError,
                                     "Invalid state vector passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the NumPy array is not writeable");
                        }
                        if (pars && !(pars->flags() & py::array::c_style)) {
                            py_throw(PyExc_ValueError,
                                     "Invalid parameters vector passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the NumPy array is not C contiguous");
                        }
                        if (time && !(time->flags() & py::array::c_style)) {
                            py_throw(PyExc_ValueError,
                                     "Invalid time vector passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the NumPy array is not C contiguous");
                        }

                        // Get out the raw pointers and make sure they are distinct.
                        auto s_ptr = state.mutable_data();
                        auto p_ptr = pars ? pars->data() : nullptr;
                        auto t_ptr = time ? time->data() : nullptr;

                        if (s_ptr == p_ptr || s_ptr == t_ptr || (p_ptr && t_ptr && p_ptr == t_ptr)) {
                            py_throw(PyExc_ValueError,
                                     "Invalid vectors passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the NumPy arrays must all be distinct");
                        }

                        // Check that, if there are params or time, the corresponding
                        // arrays have been provided.
                        if (n_params > 0u && p_ptr == nullptr) {
                            py_throw(
                                PyExc_ValueError,
                                "Invalid vectors passed to a function for the computation of the jet of "
                                "Taylor derivatives: the ODE system contains parameters, but no parameter array was "
                                "passed as input argument");
                        }

                        if (has_time && t_ptr == nullptr) {
                            py_throw(PyExc_ValueError,
                                     "Invalid vectors passed to a function for the computation of the jet of "
                                     "Taylor derivatives: the ODE system is non-autonomous, but no time array was "
                                     "passed as input argument");
                        }

                        // Check the shapes/dims of the input arrays.
                        taylor_add_jet_array_check(state, pars, time, n_params, has_time, order, tot_n_eq, batch_size);

                        // NOTE: here it may be that p_ptr and/or t_ptr are provided
                        // but the system has not time/params. In that case, the pointers
                        // will never be used in jptr.
                        jptr(s_ptr, p_ptr, t_ptr);

                        return state;
                    },
                    "state"_a, "pars"_a = py::none{}, "time"_a = py::none{});
#if defined(HEYOKA_HAVE_REAL128)
            }
#endif
        },
        "sys"_a, "order"_a, "batch_size"_a = 1u, "high_accuracy"_a = false, "compact_mode"_a = false,
        "sv_funcs"_a = py::list{});
}

} // namespace

} // namespace detail

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

} // namespace heyoka_py
