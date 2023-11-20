// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <iterator>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <oneapi/tbb/parallel_invoke.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/variable.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include "cfunc.hpp"
#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "dtypes.hpp"
#include "pickle_wrappers.hpp"

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

// The compiled function types.
template <typename T>
using cfunc_ptr_t = void (*)(T *, const T *, const T *, const T *) noexcept;

template <typename T>
using cfunc_ptr_s_t = void (*)(T *, const T *, const T *, const T *, std::size_t) noexcept;

// The compiled function wrapper.
template <typename T>
struct cfunc {
    using ptr_t = cfunc_ptr_t<T>;
    using ptr_s_t = cfunc_ptr_s_t<T>;

    hey::llvm_state s_scal, s_batch;
    std::uint32_t simd_size = 0, nparams = 0, nouts = 0, nvars = 0;
    bool is_time_dependent = false;
    ptr_t fptr_scal = nullptr, fptr_batch = nullptr;
    ptr_s_t fptr_scal_s = nullptr, fptr_batch_s = nullptr;
    std::vector<T> buf_in, buf_out, buf_pars, buf_time;
    long long prec = 0;
    std::vector<hey::expression> list_var, fn, dc;

    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        ar << s_scal;
        ar << s_batch;
        ar << simd_size;
        ar << nparams;
        ar << nouts;
        ar << nvars;
        ar << is_time_dependent;
        ar << buf_in;
        ar << buf_out;
        ar << buf_pars;
        ar << buf_time;
        ar << prec;
        ar << list_var;
        ar << fn;
        ar << dc;
    }
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        // NOTE: make a copy of this for exception safety.
        auto copy_this = *this;

        try {
            ar >> s_scal;
            ar >> s_batch;
            ar >> simd_size;
            ar >> nparams;
            ar >> nouts;
            ar >> nvars;
            ar >> is_time_dependent;
            ar >> buf_in;
            ar >> buf_out;
            ar >> buf_pars;
            ar >> buf_time;
            ar >> prec;
            ar >> list_var;
            ar >> fn;
            ar >> dc;

            fptr_scal = reinterpret_cast<ptr_t>(s_scal.jit_lookup("cfunc"));
            fptr_batch = reinterpret_cast<ptr_t>(s_batch.jit_lookup("cfunc"));
            fptr_scal_s = reinterpret_cast<ptr_s_t>(s_scal.jit_lookup("cfunc.strided"));
            fptr_batch_s = reinterpret_cast<ptr_s_t>(s_batch.jit_lookup("cfunc.strided"));
        } catch (...) {
            // Restore before rethrowing.
            *this = std::move(copy_this);

            throw;
        }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    cfunc() = default;
    explicit cfunc(hey::llvm_state s_scal, hey::llvm_state s_batch, std::uint32_t simd_size, std::uint32_t nparams,
                   std::uint32_t nouts, std::uint32_t nvars, bool is_time_dependent, ptr_t fptr_scal, ptr_t fptr_batch,
                   ptr_s_t fptr_scal_s, ptr_s_t fptr_batch_s, std::vector<T> buf_in, std::vector<T> buf_out,
                   std::vector<T> buf_pars, std::vector<T> buf_time, long long prec,
                   std::vector<hey::expression> list_var, std::vector<hey::expression> fn,
                   std::vector<hey::expression> dc)
        : s_scal(std::move(s_scal)), s_batch(std::move(s_batch)), simd_size(simd_size), nparams(nparams), nouts(nouts),
          nvars(nvars), is_time_dependent(is_time_dependent), fptr_scal(fptr_scal), fptr_batch(fptr_batch),
          fptr_scal_s(fptr_scal_s), fptr_batch_s(fptr_batch_s), buf_in(std::move(buf_in)), buf_out(std::move(buf_out)),
          buf_pars(std::move(buf_pars)), buf_time(std::move(buf_time)), prec(prec), list_var(std::move(list_var)),
          fn(std::move(fn)), dc(std::move(dc))
    {
    }
    cfunc(const cfunc &other)
        : s_scal(other.s_scal), s_batch(other.s_batch), simd_size(other.simd_size), nparams(other.nparams),
          nouts(other.nouts), nvars(other.nvars), is_time_dependent(other.is_time_dependent), buf_in(other.buf_in),
          buf_out(other.buf_out), buf_pars(other.buf_pars), buf_time(other.buf_time), prec(other.prec),
          list_var(other.list_var), fn(other.fn), dc(other.dc)
    {
        // NOTE: don't lookup the pointers if we are copying a def-cted cfunc,
        // just leave them null.
        if (other.fptr_scal != nullptr) {
            fptr_scal = reinterpret_cast<ptr_t>(s_scal.jit_lookup("cfunc"));
            fptr_batch = reinterpret_cast<ptr_t>(s_batch.jit_lookup("cfunc"));
            fptr_scal_s = reinterpret_cast<ptr_s_t>(s_scal.jit_lookup("cfunc.strided"));
            fptr_batch_s = reinterpret_cast<ptr_s_t>(s_batch.jit_lookup("cfunc.strided"));
        }
    }
    cfunc(cfunc &&) noexcept = default;
    cfunc &operator=(const cfunc &other)
    {
        if (this != &other) {
            *this = cfunc(other);
        }
        return *this;
    }
    cfunc &operator=(cfunc &&) noexcept = default;
    ~cfunc() = default;

    auto operator()(const py::iterable &inputs_ob, std::optional<py::iterable> outputs_ob,
                    std::optional<py::iterable> pars_ob, std::optional<std::variant<T, py::iterable>> time_ob)
    {
        if (fptr_scal == nullptr) {
            py_throw(PyExc_ValueError, "Cannot invoke a default-constructed compiled function");
        }

        using namespace pybind11::literals;

        // Fetch the dtype corresponding to T.
        const auto dt = get_dtype<T>();

        // Attempt to convert the input arguments into arrays.
        py::array inputs = inputs_ob;
        std::optional<py::array> outputs_ = outputs_ob ? *outputs_ob : std::optional<py::array>{};
        std::optional<py::array> pars = pars_ob ? *pars_ob : std::optional<py::array>{};
        // NOTE: transform a time scalar on-the-fly into a numpy array for ease of handling
        // in the logic.
        std::optional<py::array> time = time_ob ? std::visit(
                                            [](auto &v) {
                                                if constexpr (std::is_same_v<T &, decltype(v)>) {
                                                    // NOTE: for the scalar case, go through
                                                    // a list conversion.
                                                    py::list ret;
                                                    ret.append(v);

                                                    return py::array(ret);
                                                } else {
                                                    return py::array(v);
                                                }
                                            },
                                            *time_ob)
                                                : std::optional<py::array>{};

        // Enforce the correct dtype for all arrays.
        if (inputs.dtype().num() != dt) {
            inputs = inputs.attr("astype")(py::dtype(dt), "casting"_a = "safe");
        }
        if (outputs_ && outputs_->dtype().num() != dt) {
            *outputs_ = outputs_->attr("astype")(py::dtype(dt), "casting"_a = "safe");
        }
        if (pars && pars->dtype().num() != dt) {
            *pars = pars->attr("astype")(py::dtype(dt), "casting"_a = "safe");
        }
        if (time && time->dtype().num() != dt) {
            *time = time->attr("astype")(py::dtype(dt), "casting"_a = "safe");
        }

        // If we have params in the function, we must be provided
        // with an array of parameter values.
        if (nparams > 0u && !pars) {
            py_throw(PyExc_ValueError, fmt::format("The compiled function contains {} parameter(s), but no array "
                                                   "of parameter values was provided for evaluation",
                                                   nparams)
                                           .c_str());
        }

        // If the function is time-dependent, we must be provided
        // with an array of time values.
        if (is_time_dependent && !time) {
            py_throw(PyExc_ValueError, "The compiled function is time-dependent, but no "
                                       "time value(s) were provided for evaluation");
        }

        // Validate the number of dimensions for the inputs.
        if (inputs.ndim() != 1 && inputs.ndim() != 2) {
            py_throw(PyExc_ValueError, fmt::format("The array of inputs provided for the evaluation "
                                                   "of a compiled function has {} dimensions, "
                                                   "but it must have either 1 or 2 dimensions instead",
                                                   inputs.ndim())
                                           .c_str());
        }

        // Check the number of inputs.
        if (boost::numeric_cast<std::uint32_t>(inputs.shape(0)) != nvars) {
            py_throw(PyExc_ValueError, fmt::format("The array of inputs provided for the evaluation "
                                                   "of a compiled function has size {} in the first dimension, "
                                                   "but it must have a size of {} instead (i.e., the size in the "
                                                   "first dimension must be equal to the number of variables)",
                                                   inputs.shape(0), nvars)
                                           .c_str());
        }

        // Determine if we are running one or more evaluations.
        const auto multi_eval = inputs.ndim() == 2;

        // Check that if we are doing a single evaluation, a scalar
        // time value was passed.
        if (time && !multi_eval && !std::holds_alternative<T>(*time_ob)) {
            py_throw(PyExc_ValueError, "When performing a single evaluation of a compiled function, a scalar time "
                                       "value must be provided, but an iterable object was passed instead");
        }

        // Prepare the array of outputs.
        auto outputs = [&]() {
            if (outputs_) {
                // The outputs array was provided, check it.

                // Check if we can write to the outputs.
                if (!outputs_->writeable()) {
                    py_throw(PyExc_ValueError, "The array of outputs provided for the evaluation "
                                               "of a compiled function is not writeable");
                }

                // Validate the number of dimensions for the outputs.
                if (outputs_->ndim() != inputs.ndim()) {
                    py_throw(PyExc_ValueError, fmt::format("The array of outputs provided for the evaluation "
                                                           "of a compiled function has {} dimension(s), "
                                                           "but it must have {} dimension(s) instead (i.e., the same "
                                                           "number of dimensions as the array of inputs)",
                                                           outputs_->ndim(), inputs.ndim())
                                                   .c_str());
                }

                // Check the number of outputs.
                if (boost::numeric_cast<std::uint32_t>(outputs_->shape(0)) != nouts) {
                    py_throw(PyExc_ValueError,
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
                    py_throw(PyExc_ValueError,
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
                    return py::array(inputs.dtype(), py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(nouts),
                                                                               inputs.shape(1)});
                } else {
                    return py::array(inputs.dtype(),
                                     py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(nouts)});
                }
            }
        }();

#if defined(HEYOKA_HAVE_REAL)

        if constexpr (std::is_same_v<T, mppp::real>) {
            // For mppp::real:
            // - check that the inputs array contains values with the correct precision,
            // - ensure that the outputs array contains constructed values with the correct
            //   precision.
            pyreal_check_array(inputs, boost::numeric_cast<mpfr_prec_t>(prec));
            pyreal_ensure_array(outputs, boost::numeric_cast<mpfr_prec_t>(prec));
        }

#endif

        // Check the pars array, if necessary.
        if (pars) {
            // Validate the number of dimensions.
            if (pars->ndim() != inputs.ndim()) {
                py_throw(PyExc_ValueError, fmt::format("The array of parameter values provided for the evaluation "
                                                       "of a compiled function has {} dimension(s), "
                                                       "but it must have {} dimension(s) instead (i.e., the same "
                                                       "number of dimensions as the array of inputs)",
                                                       pars->ndim(), inputs.ndim())
                                               .c_str());
            }

            // Check the number of pars.
            if (boost::numeric_cast<std::uint32_t>(pars->shape(0)) != nparams) {
                py_throw(PyExc_ValueError,
                         fmt::format("The array of parameter values provided for the evaluation "
                                     "of a compiled function has size {} in the first dimension, "
                                     "but it must have a size of {} instead (i.e., the size in the "
                                     "first dimension must be equal to the number of parameters in the function)",
                                     pars->shape(0), nparams)
                             .c_str());
            }

            // If we are running multiple evaluations, the number must
            // be consistent between inputs and pars.
            if (multi_eval && pars->shape(1) != inputs.shape(1)) {
                py_throw(PyExc_ValueError,
                         fmt::format("The size in the second dimension for the array of parameter values "
                                     "provided for "
                                     "the evaluation of a compiled function ({}) must match the size in the "
                                     "second dimension for the array of inputs ({})",
                                     pars->shape(1), inputs.shape(1))
                             .c_str());
            }

#if defined(HEYOKA_HAVE_REAL)

            if constexpr (std::is_same_v<T, mppp::real>) {
                // For mppp::real, check that the pars array is filled
                // with constructed values with the correct precision.
                pyreal_check_array(*pars, boost::numeric_cast<mpfr_prec_t>(prec));
            }

#endif
        }

        // Check the time array, if necessary.
        if (time) {
            // NOTE: the time array must be one-dimensional: if we are in a single-eval
            // situation, the time was originally a scalar which was converted into a 1D
            // array, otherwise the time was originally an iterable which was converted
            // into an array. In the latter case, we must ensure the user did not
            // provide a multi-dimensional array in input.
            if (time->ndim() != 1) {
                py_throw(PyExc_ValueError,
                         fmt::format("An invalid time argument was passed to a compiled function: the time "
                                     "array must be one-dimensional, but instead it has {} dimensions",
                                     time->ndim())
                             .c_str());
            }

            // If we are running multiple evaluations, the number must
            // be consistent between inputs and time.
            if (multi_eval && time->shape(0) != inputs.shape(1)) {
                py_throw(PyExc_ValueError,
                         fmt::format("The size of the array of time values provided for "
                                     "the evaluation of a compiled function ({}) must match the size in the "
                                     "second dimension for the array of inputs ({})",
                                     time->shape(0), inputs.shape(1))
                             .c_str());
            }

            if (!multi_eval) {
                // NOTE: in single-eval, the time array was created from a single scalar.
                assert(time->shape(0) == 1);
            }

#if defined(HEYOKA_HAVE_REAL)

            if constexpr (std::is_same_v<T, mppp::real>) {
                // For mppp::real, check that the time array is filled
                // with constructed values with the correct precision.
                pyreal_check_array(*time, boost::numeric_cast<mpfr_prec_t>(prec));
            }

#endif
        }

        // Check if we can use a zero-copy implementation. This is enabled
        // for C-style contiguous aligned arrays guaranteed not to share any data.
        bool zero_copy = is_npy_array_carray(inputs) && is_npy_array_carray(outputs)
                         && (!pars || is_npy_array_carray(*pars)) && (!time || is_npy_array_carray(*time));
        if (zero_copy) {
            bool maybe_share_memory{};
            if (pars) {
                if (time) {
                    maybe_share_memory = may_share_memory(inputs, outputs, *pars, *time);
                } else {
                    maybe_share_memory = may_share_memory(inputs, outputs, *pars);
                }
            } else {
                if (time) {
                    maybe_share_memory = may_share_memory(inputs, outputs, *time);
                } else {
                    maybe_share_memory = may_share_memory(inputs, outputs);
                }
            }

            if (maybe_share_memory) {
                zero_copy = false;
            }
        }

        // Fetch pointers to the buffers, to decrease typing.
        const auto buf_in_ptr = buf_in.data();
        const auto buf_out_ptr = buf_out.data();
        const auto buf_par_ptr = buf_pars.data();
        const auto buf_time_ptr = buf_time.data();

        // Run the evaluation.
        if (multi_eval) {
            // Signed version of the simd size.
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
                auto *time_data = time ? static_cast<const T *>(time->data()) : nullptr;
                // NOTE: the idea of these booleans is that we want to do arithmetics
                // on the inputs/pars/time pointers only if we **must** read from them,
                // in which case the validation steps taken earlier ensure that
                // arithmetics on them is safe. Otherwise, there are certain corner cases in which
                // we might end up doing pointer arithmetics which leads to UB. For instance,
                // if the function has no inputs and/or no parameters, then we are dealing
                // with input and/or pars arrays of shape (0, nevals). I am not sure what
                // kind of data pointer NumPy returns in such a case, but if NumPy, e.g.,
                // returns nullptr, then we are committing UB.
                // NOTE: if nevals is zero, then the two for loops below are never
                // entered and we never end up doing arithmetics on potentially-null
                // pointers.
                const auto read_inputs = nvars > 0u;
                const auto read_pars = nparams > 0u;
                const auto read_time = is_time_dependent;

                // Evaluate over the simd blocks.
                for (py::ssize_t k = 0; k < n_simd_blocks; ++k) {
                    const auto start_offset = k * ss_size;

                    // Run the evaluation.
                    fptr_batch_s(out_data + start_offset, read_inputs ? in_data + start_offset : nullptr,
                                 read_pars ? par_data + start_offset : nullptr,
                                 read_time ? time_data + start_offset : nullptr, stride);
                }

                // Handle the remainder, if present.
                for (auto k = n_simd_blocks * ss_size; k < nevals; ++k) {
                    fptr_scal_s(out_data + k, read_inputs ? in_data + k : nullptr, read_pars ? par_data + k : nullptr,
                                read_time ? time_data + k : nullptr, stride);
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

                    // Copy over the time values.
                    if (time) {
                        auto u_time = time->template unchecked<T, 1>();

                        for (py::ssize_t j = 0; j < ss_size; ++j) {
                            buf_time_ptr[j] = u_time(k * ss_size + j);
                        }
                    }

                    // Run the evaluation.
                    fptr_batch(buf_out_ptr, buf_in_ptr, buf_par_ptr, buf_time_ptr);

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

                    if (time) {
                        auto u_time = time->template unchecked<T, 1>();

                        buf_time_ptr[0] = u_time(k);
                    }

                    fptr_scal(buf_out_ptr, buf_in_ptr, buf_par_ptr, buf_time_ptr);

                    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nouts); ++i) {
                        u_outputs(i, k) = buf_out_ptr[i];
                    }
                }
            }
        } else {
            if (zero_copy) {
                fptr_scal(static_cast<T *>(outputs.mutable_data()), static_cast<const T *>(inputs.data()),
                          pars ? static_cast<const T *>(pars->data()) : nullptr,
                          time ? static_cast<const T *>(time->data()) : nullptr);
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

                // Copy over the time.
                if (time) {
                    auto u_time = time->template unchecked<T, 1>();

                    buf_time_ptr[0] = u_time(0);
                }

                // Run the evaluation.
                fptr_scal(buf_out_ptr, buf_in_ptr, buf_par_ptr, buf_time_ptr);

                // Write the outputs.
                auto u_outputs = outputs.template mutable_unchecked<T, 1>();
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(nouts); ++i) {
                    u_outputs(i) = buf_out_ptr[i];
                }
            }
        }

        return outputs;
    }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const cfunc<T> &cf)
{
#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        os << fmt::format("Precision: {}\n", cf.prec);
    }

#endif

    os << fmt::format("Variables: {}\n", cf.list_var);

    for (decltype(cf.fn.size()) i = 0; i < cf.fn.size(); ++i) {
        os << fmt::format("Output #{}: {}\n", i, cf.fn[i]);
    }

    return os;
}

template <typename T>
void expose_add_cfunc_impl(py::module &m, const char *suffix)
{
    using namespace pybind11::literals;
    namespace kw = hey::kw;

    py::class_<cfunc<T>> cfunc_inst(m, fmt::format("cfunc_{}", suffix).c_str(), py::dynamic_attr{});
    cfunc_inst.def(py::init<>());
    cfunc_inst.def("__call__", &cfunc<T>::operator(), "inputs"_a, "outputs"_a = py::none{}, "pars"_a = py::none{},
                   "time"_a = py::none{});
    cfunc_inst.def_readonly("param_size", &cfunc<T>::nparams);
    cfunc_inst.def_readonly("is_time_dependent", &cfunc<T>::is_time_dependent);
    // NOTE: these can be probably simplified into def_readonly().
    cfunc_inst.def_property_readonly("llvm_state_scalar",
                                     [](const cfunc<T> &cf) -> const hey::llvm_state & { return cf.s_scal; });
    cfunc_inst.def_property_readonly("llvm_state_batch",
                                     [](const cfunc<T> &cf) -> const hey::llvm_state & { return cf.s_batch; });
    cfunc_inst.def_property_readonly("list_var", [](const cfunc<T> &cf) { return cf.list_var; });
    cfunc_inst.def_property_readonly("fn", [](const cfunc<T> &cf) { return cf.fn; });
    cfunc_inst.def_property_readonly("decomposition", [](const cfunc<T> &cf) { return cf.dc; });
#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<T, mppp::real>) {
        cfunc_inst.def_property_readonly("prec", [](const cfunc<T> &cf) { return cf.prec; });
    }
#endif
    // Repr.
    cfunc_inst.def("__repr__", [](const cfunc<T> &cf) {
        std::ostringstream oss;
        oss << cf;
        return oss.str();
    });
    // Copy/deepcopy.
    cfunc_inst.def("__copy__", copy_wrapper<cfunc<T>>);
    cfunc_inst.def("__deepcopy__", deepcopy_wrapper<cfunc<T>>, "memo"_a);
    // Pickle support.
    cfunc_inst.def(py::pickle(&pickle_getstate_wrapper<cfunc<T>>, &pickle_setstate_wrapper<cfunc<T>>));

    m.def(
        fmt::format("_add_cfunc_{}", suffix).c_str(),
        [](std::vector<hey::expression> fn, std::optional<std::vector<hey::expression>> vars, bool high_accuracy,
           bool compact_mode, bool parallel_mode, unsigned opt_level, bool force_avx512, bool slp_vectorize,
           std::optional<std::uint32_t> batch_size, bool fast_math, long long prec) {
            // Compute the SIMD size.
            const auto simd_size = batch_size ? *batch_size : hey::recommended_simd_size<T>();

            // Forbid batch sizes > 1 for everything but double.
            if (!std::is_same_v<T, double> && simd_size > 1u) {
                py_throw(PyExc_ValueError, "Batch sizes greater than 1 are not supported for this floating-point type");
            }

            // Add the compiled functions.
            using ptr_t = cfunc_ptr_t<T>;
            ptr_t fptr_scal = nullptr, fptr_batch = nullptr;
            using ptr_s_t = cfunc_ptr_s_t<T>;
            ptr_s_t fptr_scal_s = nullptr, fptr_batch_s = nullptr;

            hey::llvm_state s_scal{kw::opt_level = opt_level, kw::force_avx512 = force_avx512,
                                   kw::slp_vectorize = slp_vectorize, kw::fast_math = fast_math},
                s_batch{kw::opt_level = opt_level, kw::force_avx512 = force_avx512, kw::slp_vectorize = slp_vectorize,
                        kw::fast_math = fast_math};

            // Variable to store the decomposition.
            std::vector<hey::expression> dc;

            {
                // NOTE: release the GIL during compilation.
                py::gil_scoped_release release;

                oneapi::tbb::parallel_invoke(
                    [&]() {
                        // Scalar.
                        // NOTE: we fetch the decomposition from the scalar invocation
                        // of add_cfunc().
                        if (vars) {
                            dc = hey::add_cfunc<T>(s_scal, "cfunc", fn, kw::vars = *vars,
                                                   kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode,
                                                   kw::parallel_mode = parallel_mode, kw::prec = prec);
                        } else {
                            dc = hey::add_cfunc<T>(s_scal, "cfunc", fn, kw::high_accuracy = high_accuracy,
                                                   kw::compact_mode = compact_mode, kw::parallel_mode = parallel_mode,
                                                   kw::prec = prec);
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
                                              kw::parallel_mode = parallel_mode, kw::prec = prec);
                        } else {
                            hey::add_cfunc<T>(s_batch, "cfunc", fn, kw::batch_size = simd_size,
                                              kw::high_accuracy = high_accuracy, kw::compact_mode = compact_mode,
                                              kw::parallel_mode = parallel_mode, kw::prec = prec);
                        }

                        s_batch.compile();

                        fptr_batch = reinterpret_cast<ptr_t>(s_batch.jit_lookup("cfunc"));
                        fptr_batch_s = reinterpret_cast<ptr_s_t>(s_batch.jit_lookup("cfunc.strided"));
                    });
            }

            // Let's figure out if fn contains params and if it is time-dependent.
            const auto nparams = hey::get_param_size(fn);
            const auto is_time_dependent = hey::is_time_dependent(fn);

            // Cache the number of variables and outputs.
            // NOTE: static casts are fine, because add_cfunc()
            // succeeded and that guarantees that the number of vars and outputs
            // fits in a 32-bit int.
            const auto nouts = static_cast<std::uint32_t>(fn.size());

            // NOTE: save as well the list of variables, so
            // that we can pass it to the cfunc ctor.
            std::vector<hey::expression> list_var;

            std::uint32_t nvars = 0;
            if (vars) {
                nvars = static_cast<std::uint32_t>(vars->size());

                list_var = std::move(*vars);
            } else {
                // NOTE: get_variables() returns an ordered list of strings,
                // we need to convert it into a list of expressions.
                const auto var_slist = hey::get_variables(fn);
                list_var.reserve(var_slist.size());
                std::transform(var_slist.begin(), var_slist.end(), std::back_inserter(list_var),
                               [](const auto &name) { return hey::expression{name}; });

                nvars = static_cast<std::uint32_t>(list_var.size());
            }

            // Prepare local buffers to store inputs, outputs, pars and time
            // during the invocation of the compiled functions.
            // These are used only if we cannot read from/write to
            // the numpy arrays directly.
            std::vector<T> buf_in, buf_out, buf_pars, buf_time;
            // NOTE: the multiplications are safe because
            // the overflow checks we run during the compilation
            // of the function in batch mode did not raise errors.
            buf_in.resize(boost::numeric_cast<decltype(buf_in.size())>(nvars * simd_size));
            buf_out.resize(boost::numeric_cast<decltype(buf_out.size())>(nouts * simd_size));
            buf_pars.resize(boost::numeric_cast<decltype(buf_pars.size())>(nparams * simd_size));
            buf_time.resize(boost::numeric_cast<decltype(buf_time.size())>(simd_size));

#if defined(HEYOKA_HAVE_REAL)

            if constexpr (std::is_same_v<T, mppp::real>) {
                // For mppp::real, ensure that all buffers contain
                // values with the correct precision.

                for (auto &val : buf_in) {
                    val.set_prec(boost::numeric_cast<mpfr_prec_t>(prec));
                }

                for (auto &val : buf_out) {
                    val.set_prec(boost::numeric_cast<mpfr_prec_t>(prec));
                }

                for (auto &val : buf_pars) {
                    val.set_prec(boost::numeric_cast<mpfr_prec_t>(prec));
                }

                for (auto &val : buf_time) {
                    val.set_prec(boost::numeric_cast<mpfr_prec_t>(prec));
                }
            }

#endif

            return cfunc<T>{std::move(s_scal),
                            std::move(s_batch),
                            simd_size,
                            nparams,
                            nouts,
                            nvars,
                            is_time_dependent,
                            fptr_scal,
                            fptr_batch,
                            fptr_scal_s,
                            fptr_batch_s,
                            std::move(buf_in),
                            std::move(buf_out),
                            std::move(buf_pars),
                            std::move(buf_time),
                            prec,
                            std::move(list_var),
                            std::move(fn),
                            std::move(dc)};
        },
        "fn"_a, "vars"_a = py::none{}, "high_accuracy"_a.noconvert() = false,
        "compact_mode"_a.noconvert() = default_cm<T>, "parallel_mode"_a.noconvert() = false,
        "opt_level"_a.noconvert() = 3, "force_avx512"_a.noconvert() = false, "slp_vectorize"_a.noconvert() = false,
        "batch_size"_a.noconvert() = py::none{}, "fast_math"_a.noconvert() = false, "prec"_a.noconvert() = 0);
}

} // namespace

} // namespace detail

void expose_add_cfunc_flt(py::module &m)
{
    detail::expose_add_cfunc_impl<float>(m, "flt");
}

void expose_add_cfunc_dbl(py::module &m)
{
    detail::expose_add_cfunc_impl<double>(m, "dbl");
}

void expose_add_cfunc_ldbl(py::module &m)
{
    detail::expose_add_cfunc_impl<long double>(m, "ldbl");
}

#if defined(HEYOKA_HAVE_REAL128)

void expose_add_cfunc_f128(py::module &m)
{
    detail::expose_add_cfunc_impl<mppp::real128>(m, "f128");
}

#endif

#if defined(HEYOKA_HAVE_REAL)

void expose_add_cfunc_real(py::module &m)
{
    detail::expose_add_cfunc_impl<mppp::real>(m, "real");
}

#endif

} // namespace heyoka_py
