// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <iostream>
#include <optional>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "dtypes.hpp"
#include "expose_M2E.hpp"

namespace heyoka_py
{

namespace py = pybind11;

namespace detail
{

namespace
{

// The function type for the C++ implementations of M2E().
template <typename T>
using kepE_func_t = void (*)(T *, const T *, const T *);

// The pointers to the scalar C++ implementations.
template <typename T>
kepE_func_t<T> kepE_scal_f = nullptr;

// The pointers to the batch C++ implementations.
template <typename T>
kepE_func_t<T> kepE_batch_f = nullptr;

// Scalar implementation of Python's M2E().
template <typename T>
T kepE_scalar_wrapper(T e, T M)
{
    assert(kepE_scal_f<T> != nullptr);

    T out;
    kepE_scal_f<T>(&out, &e, &M);
    return out;
}

// Helper for the vector implementation of M2E().
template <typename T>
py::array kepE_vector_impl(py::array e, py::array M)
{
    namespace hey = heyoka;

    assert(kepE_batch_f<T> != nullptr);
    assert(e.dtype().num() == get_dtype<T>());
    assert(M.dtype().num() == get_dtype<T>());

    if (e.ndim() != 1) {
        py_throw(PyExc_ValueError,
                 fmt::format("Invalid eccentricity array passed to M2E(): "
                             "a one-dimensional array is expected, but the input array has {} dimensions",
                             e.ndim())
                     .c_str());
    }

    const auto arr_size = e.shape(0);

    if (M.ndim() != 1) {
        py_throw(PyExc_ValueError,
                 fmt::format("Invalid mean anomaly array passed to M2E(): "
                             "a one-dimensional array is expected, but the input array has {} dimensions",
                             M.ndim())
                     .c_str());
    }

    if (M.shape(0) != arr_size) {
        py_throw(PyExc_ValueError, fmt::format("Invalid arrays passed to M2E(): "
                                               "the eccentricity array has a size of {}, but the mean anomaly "
                                               "array has a size of {} (the sizes must be equal)",
                                               arr_size, M.shape(0))
                                       .c_str());
    }

    // Prepare the return value.
    py::array retval(e.dtype(), py::array::ShapeContainer{arr_size});
    auto *out_ptr = static_cast<T *>(retval.mutable_data());

    // Temporary buffers to store the input/output of the batch-mode
    // M2E() C++ implementation.
    const auto simd_size = hey::recommended_simd_size<T>();
    std::vector<T> tmp_e, tmp_M, tmp_out;
    tmp_e.resize(boost::numeric_cast<typename std::vector<T>::size_type>(simd_size));
    tmp_M.resize(boost::numeric_cast<typename std::vector<T>::size_type>(simd_size));
    tmp_out.resize(boost::numeric_cast<typename std::vector<T>::size_type>(simd_size));

    const auto tmp_e_ptr = tmp_e.data();
    const auto tmp_M_ptr = tmp_M.data();
    const auto tmp_out_ptr = tmp_out.data();

    // Signed version of the recommended simd size.
    const auto ss_size = boost::numeric_cast<py::ssize_t>(simd_size);

    // Number of simd blocks in the input arrays.
    const auto n_simd_blocks = arr_size / ss_size;

    // Unchecked access to e and M.
    auto e_acc = e.template unchecked<T, 1>();
    auto M_acc = M.template unchecked<T, 1>();

    // Iterate over the simd blocks, executing the
    // batch-mode C++ implementation of M2E().
    for (py::ssize_t i = 0; i < n_simd_blocks; ++i) {
        for (py::ssize_t j = 0; j < ss_size; ++j) {
            tmp_e_ptr[j] = e_acc(i * ss_size + j);
            tmp_M_ptr[j] = M_acc(i * ss_size + j);
        }

        kepE_batch_f<T>(out_ptr + i * ss_size, tmp_e_ptr, tmp_M_ptr);
    }

    // Handle the remainder, if present.
    if (n_simd_blocks * ss_size != arr_size) {
        py::ssize_t j = 0;
        for (py::ssize_t i = n_simd_blocks * ss_size; i < arr_size; ++i, ++j) {
            tmp_e_ptr[j] = e_acc(i);
            tmp_M_ptr[j] = M_acc(i);
        }
        // Pad with zeroes.
        for (; j < ss_size; ++j) {
            tmp_e_ptr[j] = 0;
            tmp_M_ptr[j] = 0;
        }

        // NOTE: the result for the remainder is written
        // to a temporary buffer and copied into out_ptr
        // below.
        kepE_batch_f<T>(tmp_out_ptr, tmp_e_ptr, tmp_M_ptr);

        for (py::ssize_t i = n_simd_blocks * ss_size; i < arr_size; ++i) {
            out_ptr[i] = tmp_out_ptr[i - n_simd_blocks * ss_size];
        }
    }

    return retval;
}

// Vector implementation of Python's M2E() via the batch mode C++ M2E() primitive.
// NOTE: this can probably be optimised in case "e" and "M" have contiguous C-like storage.
py::array kepE_vector_wrapper(const py::iterable &e_ob, const py::iterable &M_ob)
{
    // Attempt to convert the input arguments into arrays.
    py::array e = e_ob, M = M_ob;

    // Check the two arrays have the same dtype.
    if (e.dtype().num() != M.dtype().num()) {
        py_throw(
            PyExc_TypeError,
            fmt::format(
                "Inconsistent dtypes detected in the vectorised M2E() implementation: the eccentricity array has "
                "dtype \"{}\", while the mean anomaly array has dtype \"{}\" (the arrays must have the same dtype)",
                str(e.dtype()), str(M.dtype()))
                .c_str());
    }

    if (e.dtype().num() == get_dtype<double>()) {
        return kepE_vector_impl<double>(e, M);
#if !defined(HEYOKA_ARCH_PPC)
    } else if (e.dtype().num() == get_dtype<long double>()) {
        return kepE_vector_impl<long double>(e, M);
#endif
#if defined(HEYOKA_HAVE_REAL128)
    } else if (e.dtype().num() == get_dtype<mppp::real128>()) {
        return kepE_vector_impl<mppp::real128>(e, M);
#endif
    } else {
        py_throw(
            PyExc_TypeError,
            fmt::format(R"(Unsupported dtype "{}" for the vectorised M2E() implementation)", str(e.dtype())).c_str());
    }
}

// The llvm_state containing the JIT-compiled C++ implementations
// of kepE().
std::optional<heyoka::llvm_state> kepE_state;

} // namespace

} // namespace detail

void expose_M2E(py::module_ &m)
{
    namespace hey = heyoka;
    using namespace pybind11::literals;

    // Setup the JIT compiled C++ implementations of M2E().
    detail::kepE_state.emplace();

    // Scalar implementations.
    hey::detail::llvm_add_inv_kep_E_wrapper<double>(*detail::kepE_state, 1, "kepE_scal_dbl");
#if !defined(HEYOKA_ARCH_PPC)
    hey::detail::llvm_add_inv_kep_E_wrapper<long double>(*detail::kepE_state, 1, "kepE_scal_ldbl");
#endif
#if defined(HEYOKA_HAVE_REAL128)
    hey::detail::llvm_add_inv_kep_E_wrapper<mppp::real128>(*detail::kepE_state, 1, "kepE_scal_f128");
#endif

    // Batch implementations.
    hey::detail::llvm_add_inv_kep_E_wrapper<double>(*detail::kepE_state, hey::recommended_simd_size<double>(),
                                                    "kepE_batch_dbl");
#if !defined(HEYOKA_ARCH_PPC)
    hey::detail::llvm_add_inv_kep_E_wrapper<long double>(*detail::kepE_state, hey::recommended_simd_size<long double>(),
                                                         "kepE_batch_ldbl");
#endif
#if defined(HEYOKA_HAVE_REAL128)
    hey::detail::llvm_add_inv_kep_E_wrapper<mppp::real128>(
        *detail::kepE_state, hey::recommended_simd_size<mppp::real128>(), "kepE_batch_f128");
#endif

    detail::kepE_state->optimise();
    detail::kepE_state->compile();

    // Fetch and assign the scalar implementations.
    detail::kepE_scal_f<double> = reinterpret_cast<detail::kepE_func_t<double>>(
        detail::kepE_state->jit_lookup("kepE_scal_dbl"));
#if !defined(HEYOKA_ARCH_PPC)
    detail::kepE_scal_f<long double> = reinterpret_cast<detail::kepE_func_t<long double>>(
        detail::kepE_state->jit_lookup("kepE_scal_ldbl"));
#endif
#if defined(HEYOKA_HAVE_REAL128)
    detail::kepE_scal_f<mppp::real128> = reinterpret_cast<detail::kepE_func_t<mppp::real128>>(
        detail::kepE_state->jit_lookup("kepE_scal_f128"));
#endif

    // Fetch and assign the batch implementations.
    detail::kepE_batch_f<double> = reinterpret_cast<detail::kepE_func_t<double>>(
        detail::kepE_state->jit_lookup("kepE_batch_dbl"));
#if !defined(HEYOKA_ARCH_PPC)
    detail::kepE_batch_f<long double> = reinterpret_cast<detail::kepE_func_t<long double>>(
        detail::kepE_state->jit_lookup("kepE_batch_ldbl"));
#endif
#if defined(HEYOKA_HAVE_REAL128)
    detail::kepE_batch_f<mppp::real128> = reinterpret_cast<detail::kepE_func_t<mppp::real128>>(
        detail::kepE_state->jit_lookup("kepE_batch_f128"));
#endif

    // Expose the Python scalar implementations.
    m.def("M2E", &detail::kepE_scalar_wrapper<double>, "e"_a.noconvert(), "M"_a.noconvert());
#if !defined(HEYOKA_ARCH_PPC)
    m.def("M2E", &detail::kepE_scalar_wrapper<long double>, "e"_a.noconvert(), "M"_a.noconvert());
#endif
#if defined(HEYOKA_HAVE_REAL128)
    m.def("M2E", &detail::kepE_scalar_wrapper<mppp::real128>, "e"_a.noconvert(), "M"_a.noconvert());
#endif

    // Expose the vector implementation.
    // NOTE: type dispatching is done internally.
    m.def("M2E", &detail::kepE_vector_wrapper, "e"_a.noconvert(), "M"_a.noconvert());

    // Make sure the TBB control structure is cleaned
    // up before shutdown.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
#if !defined(NDEBUG)
        std::cout << "Cleaning up the M2E llvm state" << std::endl;
#endif
        detail::kepE_state.reset();
    }));
}

} // namespace heyoka_py
