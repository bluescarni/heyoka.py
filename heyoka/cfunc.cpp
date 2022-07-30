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
#include <cstddef>
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
#include "custom_casters.hpp"
#include "dtypes.hpp"

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
    namespace kw = hey::kw;

    m.def(
        name,
        [](const std::vector<hey::expression> &fn, const std::optional<std::vector<hey::expression>> &vars,
           bool high_accuracy, bool compact_mode, bool parallel_mode) {
            // Fetch the recommended SIMD size.
            const auto simd_size = hey::recommended_simd_size<T>();

            // Add the compiled functions.
            using ptr_t = void (*)(T *, const T *, const T *) noexcept;
            ptr_t fptr_scal = nullptr, fptr_batch = nullptr;
            using ptr_s_t = void (*)(T *, const T *, const T *, std::size_t) noexcept;
            ptr_s_t fptr_scal_s = nullptr, fptr_batch_s = nullptr;

            hey::llvm_state s_scal, s_batch;

            {
                // NOTE: release the GIL during compilation.
                py::gil_scoped_release release;

                oneapi::tbb::parallel_invoke(
                    [&]() {
                        // Scalar.
                        if (vars) {
                            hey::add_cfunc<T>(s_scal, "cfunc", fn, kw::vars = *vars, kw::high_accuracy = high_accuracy,
                                              kw::compact_mode = compact_mode, kw::parallel_mode = parallel_mode);
                        } else {
                            hey::add_cfunc<T>(s_scal, "cfunc", fn, kw::high_accuracy = high_accuracy,
                                              kw::compact_mode = compact_mode, kw::parallel_mode = parallel_mode);
                        }

                        s_scal.compile();

                        fptr_scal = reinterpret_cast<ptr_t>(s_scal.jit_lookup("cfunc"));
                        fptr_scal_s = reinterpret_cast<ptr_s_t>(s_scal.jit_lookup("cfunc.strided"));
                    },
                    [&]() {
                        // Batch.
                        if (vars) {
                            hey::add_cfunc<T>(s_batch, "cfunc", fn, kw::vars = *vars, kw::batch_size = simd_size,
                                              kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode,
                                              kw::parallel_mode = parallel_mode);
                        } else {
                            hey::add_cfunc<T>(s_batch, "cfunc", fn, kw::batch_size = simd_size,
                                              kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode,
                                              kw::parallel_mode = parallel_mode);
                        }

                        s_batch.compile();

                        fptr_batch = reinterpret_cast<ptr_t>(s_batch.jit_lookup("cfunc"));
                        fptr_batch_s = reinterpret_cast<ptr_s_t>(s_batch.jit_lookup("cfunc.strided"));
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
            // These are used only if we cannot read from/write to
            // the numpy arrays directly.
            std::vector<T> buf_in, buf_out, buf_pars;
            // NOTE: the multiplications are safe because
            // the overflow checks we run during the compilation
            // of the function in batch mode did not raise errors.
            buf_in.resize(boost::numeric_cast<decltype(buf_in.size())>(nvars * simd_size));
            buf_out.resize(boost::numeric_cast<decltype(buf_out.size())>(nouts * simd_size));
            buf_pars.resize(boost::numeric_cast<decltype(buf_pars.size())>(nparams * simd_size));

            return py::cpp_function(
                [s_scal = std::move(s_scal), s_batch = std::move(s_batch), simd_size, nparams, nouts, nvars, fptr_scal,
                 fptr_scal_s, fptr_batch, fptr_batch_s, buf_in = std::move(buf_in), buf_out = std::move(buf_out),
                 buf_pars = std::move(buf_pars)](py::object inputs_ob, std::optional<py::object> outputs_ob,
                                                 std::optional<py::object> pars_ob) mutable {
                    // Attempt to convert the input arguments into arrays.
                    py::array inputs = inputs_ob;
                    std::optional<py::array> outputs_ = outputs_ob ? *outputs_ob : std::optional<py::array>{};
                    std::optional<py::array> pars = pars_ob ? *pars_ob : std::optional<py::array>{};

                    // Enforce the correct dtype for all arrays.
                    const auto dt = get_dtype<T>();
                    if (inputs.dtype().num() != dt) {
                        inputs = inputs.attr("astype")(py::dtype(dt));
                    }
                    if (outputs_ && outputs_->dtype().num() != dt) {
                        *outputs_ = outputs_->attr("astype")(py::dtype(dt));
                    }
                    if (pars && pars->dtype().num() != dt) {
                        *pars = pars->attr("astype")(py::dtype(dt));
                    }

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

                            // Check if we can write to the outputs.
                            if (!outputs_->writeable()) {
                                heypy::py_throw(PyExc_ValueError, "The array of outputs provided for the evaluation "
                                                                  "of a compiled function is not writeable");
                            }

                            // Validate the number of dimensions for the outputs.
                            if (outputs_->ndim() != inputs.ndim()) {
                                heypy::py_throw(PyExc_ValueError,
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
                                    fmt::format("The size in the second dimension for the output array provided for "
                                                "the evaluation of a compiled function ({}) must match the size in the "
                                                "second dimension for the array of inputs ({})",
                                                outputs_->shape(1), inputs.shape(1))
                                        .c_str());
                            }

                            return std::move(*outputs_);
                        } else {
                            // Create the outputs array.
                            if (multi_eval) {
                                return py::array(inputs.dtype(),
                                                 py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(nouts),
                                                                           inputs.shape(1)});
                            } else {
                                return py::array(inputs.dtype(),
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

                    // Check if we can use a zero-copy implementation. This is enabled
                    // for distinct C style arrays who own their data.

                    // All C style?
                    bool zero_copy = (inputs.flags() & py::array::c_style) && (outputs.flags() & py::array::c_style)
                                     && (!pars || (pars->flags() & py::array::c_style));
                    // Do they all own their data?
                    zero_copy = zero_copy && (inputs.owndata() && outputs.owndata() && (!pars || pars->owndata()));
                    if (zero_copy) {
                        auto *out_data = outputs.data();
                        auto *in_data = inputs.data();
                        auto *par_data = pars ? pars->data() : nullptr;

                        // Are they all distinct from each other?
                        // NOTE: while out_data can never be possibly null (as we are sure there's
                        // always at least one output), I am not 100% sure what happens with empty
                        // inputs and/or pars. Just to be on the safe side, we check in_data == par_data
                        // only if both in_data and par_data are not null, so that both pointers
                        // being null does not prevent the zero-copy implementation.
                        if (out_data == in_data || out_data == par_data
                            || (in_data && par_data && in_data == par_data)) {
                            zero_copy = false;
                        }
                    }

                    // Fetch pointers to the buffers, to decrease typing.
                    const auto buf_in_ptr = buf_in.data();
                    const auto buf_out_ptr = buf_out.data();
                    const auto buf_par_ptr = buf_pars.data();

                    // Run the evaluation.
                    if (multi_eval) {
                        // Signed version of the recommended simd size.
                        const auto ss_size = boost::numeric_cast<py::ssize_t>(simd_size);

                        // Cache the number of evals.
                        const auto nevals = inputs.shape(1);

                        // Number of simd blocks in the arrays.
                        const auto n_simd_blocks = nevals / ss_size;

                        if (zero_copy) {
                            // Safely cast nevals to size_t to compute
                            // the stride value.
                            const auto stride = boost::numeric_cast<std::size_t>(nevals);

                            // Cache pointers.
                            auto *out_data = static_cast<T *>(outputs.mutable_data());
                            auto *in_data = static_cast<const T *>(inputs.data());
                            auto *par_data = pars ? static_cast<const T *>(pars->data()) : nullptr;
                            // NOTE: we define these two boolean variables in order
                            // to avoid doing pointer arithmetic (when invoking fptr_batch_s)
                            // on bogus pointers (such as nullptr). This could happen for instance
                            // if the function has no inputs, or if an empty pars array was provided
                            // (that is, in both cases we would be dealing with numpy arrays
                            // with shape (0, nevals)). In other words, while we assume
                            // calling .data() on any numpy array is always safe, we are taking
                            // precautions when doing arithmetics on the pointer returned by .data().
                            const auto with_inputs = nvars > 0u;
                            const auto with_pars = nparams > 0u;

                            // Evaluate over the simd blocks.
                            for (py::ssize_t k = 0; k < n_simd_blocks; ++k) {
                                const auto start_offset = k * ss_size;

                                // Run the evaluation.
                                fptr_batch_s(out_data + start_offset, with_inputs ? in_data + start_offset : nullptr,
                                             with_pars ? par_data + start_offset : nullptr, stride);
                            }

                            // Handle the remainder, if present.
                            for (auto k = n_simd_blocks * ss_size; k < nevals; ++k) {
                                fptr_scal_s(out_data + k, with_inputs ? in_data + k : nullptr,
                                            with_pars ? par_data + k : nullptr, stride);
                            }
                        } else {
                            // Unchecked access to inputs and outputs.
                            auto u_inputs = inputs.template unchecked<T, 2>();
                            auto u_outputs = outputs.template mutable_unchecked<T, 2>();

                            // Evaluate over the simd blocks.
                            for (py::ssize_t k = 0; k < n_simd_blocks; ++k) {
                                // Copy over the input data.
                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nvars); ++i) {
                                    for (py::ssize_t j = 0; j < ss_size; ++j) {
                                        buf_in_ptr[i * ss_size + j] = u_inputs(i, k * ss_size + j);
                                    }
                                }

                                // Copy over the pars.
                                if (pars) {
                                    auto u_pars = pars->template unchecked<T, 2>();

                                    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nparams); ++i) {
                                        for (py::ssize_t j = 0; j < ss_size; ++j) {
                                            buf_par_ptr[i * ss_size + j] = u_pars(i, k * ss_size + j);
                                        }
                                    }
                                }

                                // Run the evaluation.
                                fptr_batch(buf_out_ptr, buf_in_ptr, buf_par_ptr);

                                // Write the outputs.
                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nouts); ++i) {
                                    for (py::ssize_t j = 0; j < ss_size; ++j) {
                                        u_outputs(i, k * ss_size + j) = buf_out_ptr[i * ss_size + j];
                                    }
                                }
                            }

                            // Handle the remainder, if present.
                            for (auto k = n_simd_blocks * ss_size; k < nevals; ++k) {
                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nvars); ++i) {
                                    buf_in_ptr[i] = u_inputs(i, k);
                                }

                                if (pars) {
                                    auto u_pars = pars->template unchecked<T, 2>();

                                    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nparams); ++i) {
                                        buf_par_ptr[i] = u_pars(i, k);
                                    }
                                }

                                fptr_scal(buf_out_ptr, buf_in_ptr, buf_par_ptr);

                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nouts); ++i) {
                                    u_outputs(i, k) = buf_out_ptr[i];
                                }
                            }
                        }
                    } else {
                        if (zero_copy) {
                            fptr_scal(static_cast<T *>(outputs.mutable_data()), static_cast<const T *>(inputs.data()),
                                      pars ? static_cast<const T *>(pars->data()) : nullptr);
                        } else {
                            // Copy over the input data.
                            auto u_inputs = inputs.template unchecked<T, 1>();
                            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nvars); ++i) {
                                buf_in_ptr[i] = u_inputs(i);
                            }

                            // Copy over the pars.
                            if (pars) {
                                auto u_pars = pars->template unchecked<T, 1>();

                                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nparams); ++i) {
                                    buf_par_ptr[i] = u_pars(i);
                                }
                            }

                            // Run the evaluation.
                            fptr_scal(buf_out_ptr, buf_in_ptr, buf_par_ptr);

                            // Write the outputs.
                            auto u_outputs = outputs.template mutable_unchecked<T, 1>();
                            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nouts); ++i) {
                                u_outputs(i) = buf_out_ptr[i];
                            }
                        }
                    }

                    return outputs;
                },
                "inputs"_a, "outputs"_a = py::none{}, "pars"_a = py::none{});
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
