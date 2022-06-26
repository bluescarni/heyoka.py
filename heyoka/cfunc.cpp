// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <oneapi/tbb/parallel_invoke.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include "cfunc.hpp"
#include "common_utils.hpp"
#include "long_double_caster.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;
namespace heypy = heyoka_py;

namespace detail
{

namespace
{

template <typename T>
void expose_add_cfunc_impl(py::module &m, const char *name)
{
    using namespace pybind11::literals;

    m.def(
        name,
        [](const std::vector<hey::expression> &fn, const std::optional<std::vector<hey::expression>> &vars,
           bool high_accuracy, bool compact_mode, bool parallel_mode) {
            // Fetch the recommended SIMD size.
            const auto simd_size = hey::recommended_simd_size<T>();

            // Add the compiled functions.
            using ptr_t = void (*)(T *, const T *, const T *) noexcept;
            ptr_t fptr_scal = nullptr, fptr_batch = nullptr;
            hey::llvm_state s_scal, s_batch;

            {
                // NOTE: release the GIL during compilation.
                py::gil_scoped_release release;

                oneapi::tbb::parallel_invoke(
                    [&]() {
                        // Scalar.
                        if (vars) {
                            hey::add_cfunc<T>(s_scal, "cfunc", fn, *vars, 1, high_accuracy, compact_mode,
                                              parallel_mode);
                        } else {
                            hey::add_cfunc<T>(s_scal, "cfunc", fn, 1, high_accuracy, compact_mode, parallel_mode);
                        }

                        s_scal.compile();

                        fptr_scal = reinterpret_cast<ptr_t>(s_scal.jit_lookup("cfunc"));
                    },
                    [&]() {
                        // Batch.
                        if (vars) {
                            hey::add_cfunc<T>(s_batch, "cfunc", fn, *vars, simd_size, high_accuracy, compact_mode,
                                              parallel_mode);
                        } else {
                            hey::add_cfunc<T>(s_batch, "cfunc", fn, simd_size, high_accuracy, compact_mode,
                                              parallel_mode);
                        }

                        s_batch.compile();

                        fptr_batch = reinterpret_cast<ptr_t>(s_batch.jit_lookup("cfunc"));
                    });
            }

            // Let's figure out if fn contains params.
            std::uint32_t nparams = 0;
            for (const auto &ex : fn) {
                nparams = std::max<std::uint32_t>(nparams, hey::get_param_size(ex));
            }

            // Cache the number of variables and outputs.
            // NOTE: static casts are fine, because add_cfunc()
            // succeeded and that guarantees that the number of vars and outputs
            // fits in a 32-bit int.
            const auto nouts = static_cast<std::uint32_t>(fn.size());

            std::uint32_t nvars = 0;
            if (vars) {
                nvars = static_cast<std::uint32_t>(vars->size());
            } else {
                // NOTE: this is a bit of repetition from add_cfunc().
                // If this becomes an issue, we can consider in the
                // future changing add_cfunc() to return also the number
                // of detected variables.
                std::set<std::string> dvars;
                for (const auto &ex : fn) {
                    for (const auto &var : hey::get_variables(ex)) {
                        dvars.emplace(var);
                    }
                }

                nvars = static_cast<std::uint32_t>(dvars.size());
            }

            // Prepare local buffers to store inputs, outputs and pars
            // during the invocation of the compiled functions.
            std::vector<T> buf_in, buf_out, buf_pars;
            // NOTE: the multiplications are safe because
            // the overflow checks we run during the compilation
            // of the function in batch mode did not raise errors.
            buf_in.resize(boost::numeric_cast<decltype(buf_in.size())>(nvars * simd_size));
            buf_out.resize(boost::numeric_cast<decltype(buf_out.size())>(nouts * simd_size));
            buf_pars.resize(boost::numeric_cast<decltype(buf_pars.size())>(nparams * simd_size));

#if defined(HEYOKA_HAVE_REAL128)
            if constexpr (std::is_same_v<T, mppp::real128>) {
                return py::cpp_function([s_scal = std::move(s_scal), s_batch = std::move(s_batch), simd_size, nparams,
                                         nouts, nvars, fptr_scal, fptr_batch, buf_in = std::move(buf_in),
                                         buf_out = std::move(buf_out), buf_pars = std::move(buf_pars)](
                                            py::array inputs, std::optional<py::array_t<T>> outputs_,
                                            std::optional<py::array> pars) mutable {
                    if (outputs_) {
                        heypy::py_throw(PyExc_ValueError,
                                        "Specifying the output array for a compiled function is not supported in "
                                        "quadruple precision");
                    }

                    // Fetch pointers to the buffers, to decrease typing.
                    const auto in_ptr = buf_in.data();
                    const auto out_ptr = buf_out.data();
                    const auto par_ptr = buf_pars.data();

                    // If we have params in the function, we must be provided
                    // with an array of parameter values.
                    if (nparams > 0u && !pars) {
                        heypy::py_throw(PyExc_ValueError,
                                        fmt::format("The compiled function contains {} parameter(s), but no array "
                                                    "of parameter values was provided for evaluation",
                                                    nparams)
                                            .c_str());
                    }

                    // Validate the number of dimensions for the inputs.
                    if (inputs.ndim() != 1 && inputs.ndim() != 2) {
                        heypy::py_throw(PyExc_ValueError,
                                        fmt::format("The array of inputs provided for the evaluation "
                                                    "of a compiled function has {} dimensions, "
                                                    "but it must have either 1 or 2 dimensions instead",
                                                    inputs.ndim())
                                            .c_str());
                    }

                    // Check the number of inputs.
                    if (boost::numeric_cast<std::uint32_t>(inputs.shape(0)) != nvars) {
                        heypy::py_throw(PyExc_ValueError,
                                        fmt::format("The array of inputs provided for the evaluation "
                                                    "of a compiled function has size {} in the first dimension, "
                                                    "but it must have a size of {} instead (i.e., the size in the "
                                                    "first dimension must be equal to the number of variables)",
                                                    inputs.shape(0), nvars)
                                            .c_str());
                    }

                    // Determine if we are running one or more evaluations.
                    const auto multi_eval = inputs.ndim() == 2;

                    // Convert the arrays into vectors.
                    auto inputs_vec = py::cast<std::vector<mppp::real128>>(inputs.attr("flatten")());
                    std::vector<mppp::real128> pars_vec;
                    if (pars) {
                        pars_vec = py::cast<std::vector<mppp::real128>>(pars->attr("flatten")());
                    }

                    // Prepare the outputs vector.
                    std::vector<mppp::real128> outputs_vec;
                    using vec_size_t = decltype(outputs_vec.size());
                    if (multi_eval) {
                        // Overflow checking.
                        const auto nevals = boost::numeric_cast<vec_size_t>(inputs.shape(1));
                        assert(nouts > 0u);
                        if (nevals > std::numeric_limits<vec_size_t>::max() / nouts) {
                            heypy::py_throw(PyExc_OverflowError, "An overflow condition was detected while trying to "
                                                                 "evaluate a compiled function in quadruple precision");
                        }

                        outputs_vec.resize(nouts * nevals);
                    } else {
                        outputs_vec.resize(boost::numeric_cast<vec_size_t>(nouts));
                    }

                    // Check the pars array, if necessary.
                    if (pars) {
                        // Validate the number of dimensions.
                        if (pars->ndim() != inputs.ndim()) {
                            heypy::py_throw(PyExc_ValueError,
                                            fmt::format("The array of parameter values provided for the evaluation "
                                                        "of a compiled function has {} dimensions, "
                                                        "but it must have {} dimensions instead (i.e., the same "
                                                        "number of dimensions as the array of inputs)",
                                                        pars->ndim(), inputs.ndim())
                                                .c_str());
                        }

                        // Check the number of pars.
                        if (boost::numeric_cast<std::uint32_t>(pars->shape(0)) != nparams) {
                            heypy::py_throw(
                                PyExc_ValueError,
                                fmt::format(
                                    "The array of parameter values provided for the evaluation "
                                    "of a compiled function has size {} in the first dimension, "
                                    "but it must have a size of {} instead (i.e., the size in the "
                                    "first dimension must be equal to the number of parameters in the function)",
                                    pars->shape(0), nparams)
                                    .c_str());
                        }

                        // If we are running multiple evaluations, the number must
                        // be consistent between inputs and pars.
                        if (multi_eval && pars->shape(1) != inputs.shape(1)) {
                            heypy::py_throw(
                                PyExc_ValueError,
                                fmt::format("The size in the second dimension for the array of parameter values "
                                            "provided for "
                                            "the evaluation of a compiled function ({}) must match the size in the "
                                            "second dimension for the array of inputs ({})",
                                            pars->shape(1), inputs.shape(1))
                                    .c_str());
                        }
                    }

                    // Run the evaluation.
                    if (multi_eval) {
                        // vec_size_t version of the recommended simd size.
                        const auto ss_size = boost::numeric_cast<vec_size_t>(simd_size);

                        // Cache the number of evals.
                        // NOTE: static cast is fine because inputs was successfully
                        // converted to std::vector.
                        const auto nevals = static_cast<vec_size_t>(inputs.shape(1));

                        // Number of simd blocks in the arrays.
                        const auto n_simd_blocks = nevals / ss_size;

                        // Evaluate over the simd blocks.
                        for (vec_size_t k = 0; k < n_simd_blocks; ++k) {
                            // Copy over the input data.
                            for (vec_size_t i = 0; i < static_cast<vec_size_t>(nvars); ++i) {
                                for (vec_size_t j = 0; j < ss_size; ++j) {
                                    in_ptr[i * ss_size + j] = inputs_vec[i * nevals + k * ss_size + j];
                                }
                            }

                            // Copy over the pars.
                            if (pars) {
                                for (vec_size_t i = 0; i < static_cast<vec_size_t>(nparams); ++i) {
                                    for (vec_size_t j = 0; j < ss_size; ++j) {
                                        par_ptr[i * ss_size + j] = pars_vec[i * nevals + k * ss_size + j];
                                    }
                                }
                            }

                            // Run the evaluation.
                            fptr_batch(out_ptr, in_ptr, par_ptr);

                            // Write the outputs.
                            for (vec_size_t i = 0; i < static_cast<vec_size_t>(nouts); ++i) {
                                for (vec_size_t j = 0; j < ss_size; ++j) {
                                    outputs_vec[i * nevals + k * ss_size + j] = out_ptr[i * ss_size + j];
                                }
                            }
                        }

                        // Handle the remainder, if present.
                        for (auto k = n_simd_blocks * ss_size; k < nevals; ++k) {
                            for (vec_size_t i = 0; i < static_cast<vec_size_t>(nvars); ++i) {
                                in_ptr[i] = inputs_vec[i * nevals + k];
                            }

                            if (pars) {
                                for (vec_size_t i = 0; i < static_cast<vec_size_t>(nparams); ++i) {
                                    par_ptr[i] = pars_vec[i * nevals + k];
                                }
                            }

                            fptr_scal(out_ptr, in_ptr, par_ptr);

                            for (vec_size_t i = 0; i < static_cast<vec_size_t>(nouts); ++i) {
                                outputs_vec[i * nevals + k] = out_ptr[i];
                            }
                        }
                    } else {
                        // Copy over the input data.
                        for (vec_size_t i = 0; i < static_cast<vec_size_t>(nvars); ++i) {
                            in_ptr[i] = inputs_vec[i];
                        }

                        // Copy over the pars.
                        if (pars) {
                            for (vec_size_t i = 0; i < static_cast<vec_size_t>(nparams); ++i) {
                                par_ptr[i] = pars_vec[i];
                            }
                        }

                        // Run the evaluation.
                        fptr_scal(out_ptr, in_ptr, par_ptr);

                        // Write the outputs.
                        for (vec_size_t i = 0; i < static_cast<vec_size_t>(nouts); ++i) {
                            outputs_vec[i] = out_ptr[i];
                        }
                    }

                    // Convert to numpy array.
                    auto ret = py::array(py::cast(outputs_vec));

                    // Reshape if necessary.
                    if (multi_eval) {
                        // NOTE: resize() operates in-place, and won't allocate
                        // new memory because here it's just a reshape.
                        ret.resize(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(nouts), inputs.shape(1)});
                    }

                    return ret;
                });
            } else {
#endif
                return py::cpp_function(
                    [s_scal = std::move(s_scal), s_batch = std::move(s_batch), simd_size, nparams, nouts, nvars,
                     fptr_scal, fptr_batch, buf_in = std::move(buf_in), buf_out = std::move(buf_out),
                     buf_pars = std::move(buf_pars)](py::array_t<T> inputs, std::optional<py::array_t<T>> outputs_,
                                                     std::optional<py::array_t<T>> pars) mutable {
                        // Fetch pointers to the buffers, to decrease typing.
                        const auto in_ptr = buf_in.data();
                        const auto out_ptr = buf_out.data();
                        const auto par_ptr = buf_pars.data();

                        // If we have params in the function, we must be provided
                        // with an array of parameter values.
                        if (nparams > 0u && !pars) {
                            heypy::py_throw(PyExc_ValueError,
                                            fmt::format("The compiled function contains {} parameter(s), but no array "
                                                        "of parameter values was provided for evaluation",
                                                        nparams)
                                                .c_str());
                        }

                        // Validate the number of dimensions for the inputs.
                        if (inputs.ndim() != 1 && inputs.ndim() != 2) {
                            heypy::py_throw(PyExc_ValueError,
                                            fmt::format("The array of inputs provided for the evaluation "
                                                        "of a compiled function has {} dimensions, "
                                                        "but it must have either 1 or 2 dimensions instead",
                                                        inputs.ndim())
                                                .c_str());
                        }

                        // Check the number of inputs.
                        if (boost::numeric_cast<std::uint32_t>(inputs.shape(0)) != nvars) {
                            heypy::py_throw(PyExc_ValueError,
                                            fmt::format("The array of inputs provided for the evaluation "
                                                        "of a compiled function has size {} in the first dimension, "
                                                        "but it must have a size of {} instead (i.e., the size in the "
                                                        "first dimension must be equal to the number of variables)",
                                                        inputs.shape(0), nvars)
                                                .c_str());
                        }

                        // Determine if we are running one or more evaluations.
                        const auto multi_eval = inputs.ndim() == 2;

                        // Prepare the array of outputs.
                        auto outputs = [&]() {
                            if (outputs_) {
                                // The outputs array was provided, check it.

                                // Validate the number of dimensions for the outputs.
                                if (outputs_->ndim() != inputs.ndim()) {
                                    heypy::py_throw(
                                        PyExc_ValueError,
                                        fmt::format("The array of outputs provided for the evaluation "
                                                    "of a compiled function has {} dimension(s), "
                                                    "but it must have {} dimension(s) instead (i.e., the same "
                                                    "number of dimensions as the array of inputs)",
                                                    outputs_->ndim(), inputs.ndim())
                                            .c_str());
                                }

                                // Check the number of outputs.
                                if (boost::numeric_cast<std::uint32_t>(outputs_->shape(0)) != nouts) {
                                    heypy::py_throw(
                                        PyExc_ValueError,
                                        fmt::format("The array of outputs provided for the evaluation "
                                                    "of a compiled function has size {} in the first dimension, "
                                                    "but it must have a size of {} instead (i.e., the size in the "
                                                    "first dimension must be equal to the number of outputs)",
                                                    outputs_->shape(0), nouts)
                                            .c_str());
                                }

                                // If we are running multiple evaluations, the number must
                                // be consistent between inputs and outputs.
                                if (multi_eval && outputs_->shape(1) != inputs.shape(1)) {
                                    heypy::py_throw(
                                        PyExc_ValueError,
                                        fmt::format(
                                            "The size in the second dimension for the output array provided for "
                                            "the evaluation of a compiled function ({}) must match the size in the "
                                            "second dimension for the array of inputs ({})",
                                            outputs_->shape(1), inputs.shape(1))
                                            .c_str());
                                }

                                return std::move(*outputs_);
                            } else {
                                // Create the outputs array.
                                if (multi_eval) {
                                    return py::array_t<T>(py::array::ShapeContainer{
                                        boost::numeric_cast<py::ssize_t>(nouts), inputs.shape(1)});
                                } else {
                                    return py::array_t<T>(
                                        py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(nouts)});
                                }
                            }
                        }();

                        // Check the pars array, if necessary.
                        if (pars) {
                            // Validate the number of dimensions.
                            if (pars->ndim() != inputs.ndim()) {
                                heypy::py_throw(PyExc_ValueError,
                                                fmt::format("The array of parameter values provided for the evaluation "
                                                            "of a compiled function has {} dimension(s), "
                                                            "but it must have {} dimension(s) instead (i.e., the same "
                                                            "number of dimensions as the array of inputs)",
                                                            pars->ndim(), inputs.ndim())
                                                    .c_str());
                            }

                            // Check the number of pars.
                            if (boost::numeric_cast<std::uint32_t>(pars->shape(0)) != nparams) {
                                heypy::py_throw(
                                    PyExc_ValueError,
                                    fmt::format(
                                        "The array of parameter values provided for the evaluation "
                                        "of a compiled function has size {} in the first dimension, "
                                        "but it must have a size of {} instead (i.e., the size in the "
                                        "first dimension must be equal to the number of parameters in the function)",
                                        pars->shape(0), nparams)
                                        .c_str());
                            }

                            // If we are running multiple evaluations, the number must
                            // be consistent between inputs and pars.
                            if (multi_eval && pars->shape(1) != inputs.shape(1)) {
                                heypy::py_throw(
                                    PyExc_ValueError,
                                    fmt::format("The size in the second dimension for the array of parameter values "
                                                "provided for "
                                                "the evaluation of a compiled function ({}) must match the size in the "
                                                "second dimension for the array of inputs ({})",
                                                pars->shape(1), inputs.shape(1))
                                        .c_str());
                            }
                        }

                        // Run the evaluation.
                        if (multi_eval) {
                            // Signed version of the recommended simd size.
                            const auto ss_size = boost::numeric_cast<py::ssize_t>(simd_size);

                            // Cache the number of evals.
                            const auto nevals = inputs.shape(1);

                            // Number of simd blocks in the arrays.
                            const auto n_simd_blocks = nevals / ss_size;

                            // Unchecked access to inputs and outputs.
                            auto u_inputs = inputs.template unchecked<2>();
                            auto u_outputs = outputs.template mutable_unchecked<2>();

                            // Evaluate over the simd blocks.
                            for (py::ssize_t k = 0; k < n_simd_blocks; ++k) {
                                // Copy over the input data.
                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nvars); ++i) {
                                    for (py::ssize_t j = 0; j < ss_size; ++j) {
                                        in_ptr[i * ss_size + j] = u_inputs(i, k * ss_size + j);
                                    }
                                }

                                // Copy over the pars.
                                if (pars) {
                                    auto u_pars = pars->template unchecked<2>();

                                    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nparams); ++i) {
                                        for (py::ssize_t j = 0; j < ss_size; ++j) {
                                            par_ptr[i * ss_size + j] = u_pars(i, k * ss_size + j);
                                        }
                                    }
                                }

                                // Run the evaluation.
                                fptr_batch(out_ptr, in_ptr, par_ptr);

                                // Write the outputs.
                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nouts); ++i) {
                                    for (py::ssize_t j = 0; j < ss_size; ++j) {
                                        u_outputs(i, k * ss_size + j) = out_ptr[i * ss_size + j];
                                    }
                                }
                            }

                            // Handle the remainder, if present.
                            for (auto k = n_simd_blocks * ss_size; k < nevals; ++k) {
                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nvars); ++i) {
                                    in_ptr[i] = u_inputs(i, k);
                                }

                                if (pars) {
                                    auto u_pars = pars->template unchecked<2>();

                                    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nparams); ++i) {
                                        par_ptr[i] = u_pars(i, k);
                                    }
                                }

                                fptr_scal(out_ptr, in_ptr, par_ptr);

                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nouts); ++i) {
                                    u_outputs(i, k) = out_ptr[i];
                                }
                            }
                        } else {
                            // Copy over the input data.
                            auto u_inputs = inputs.template unchecked<1>();
                            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nvars); ++i) {
                                in_ptr[i] = u_inputs(i);
                            }

                            // Copy over the pars.
                            if (pars) {
                                auto u_pars = pars->template unchecked<1>();

                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nparams); ++i) {
                                    par_ptr[i] = u_pars(i);
                                }
                            }

                            // Run the evaluation.
                            fptr_scal(out_ptr, in_ptr, par_ptr);

                            // Write the outputs.
                            auto u_outputs = outputs.template mutable_unchecked<1>();
                            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nouts); ++i) {
                                u_outputs(i) = out_ptr[i];
                            }
                        }

                        return outputs;
                    },
                    "inputs"_a, "outputs"_a = py::none{}, "pars"_a = py::none{});
#if defined(HEYOKA_HAVE_REAL128)
            }
#endif
        },
        "fn"_a, "vars"_a = py::none{}, "high_accuracy"_a = false, "compact_mode"_a = false, "parallel_mode"_a = false);
}

} // namespace

} // namespace detail

void expose_add_cfunc_dbl(py::module &m)
{
    detail::expose_add_cfunc_impl<double>(m, "_add_cfunc_dbl");
}

void expose_add_cfunc_ldbl(py::module &m)
{
    detail::expose_add_cfunc_impl<long double>(m, "_add_cfunc_ldbl");
}

#if defined(HEYOKA_HAVE_REAL128)

void expose_add_cfunc_f128(py::module &m)
{
    detail::expose_add_cfunc_impl<mppp::real128>(m, "_add_cfunc_f128");
}

#endif

} // namespace heyoka_py
