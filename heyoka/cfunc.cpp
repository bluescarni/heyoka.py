// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>

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

template <typename T>
void expose_add_cfunc_impl(py::module &m, const char *suffix)
{
    using namespace pybind11::literals;
    namespace kw = hey::kw;

    py::class_<hey::cfunc<T>> cfunc_inst(m, fmt::format("cfunc_{}", suffix).c_str(), py::dynamic_attr{});
    cfunc_inst.def(
        py::init([](std::vector<hey::expression> fn, std::vector<hey::expression> vars, bool high_accuracy,
                    bool compact_mode, bool parallel_mode, std::uint32_t batch_size, long long prec, unsigned opt_level,
                    bool force_avx512, bool slp_vectorize, bool fast_math, hey::code_model code_model, bool parjit) {
            // Forbid batch sizes > 1 for everything but double and float.
            // NOTE: there is a similar check on the C++ side regarding mppp::real, but in Python
            // specifically we want to be pragmatic and allow for batch operations only if we know that it
            // makes sense performance-wise (and we also want to avoid buggy batch operations on long
            // double).
            if (!std::is_same_v<T, double> && !std::is_same_v<T, float> && batch_size > 1u) [[unlikely]] {
                py_throw(PyExc_ValueError, "Batch sizes greater than 1 are not supported for this floating-point type");
            }

            // NOTE: release the GIL during compilation.
            py::gil_scoped_release release;

            return hey::cfunc<T>{std::move(fn), std::move(vars), kw::high_accuracy = high_accuracy,
                                 kw::compact_mode = compact_mode, kw::parallel_mode = parallel_mode,
                                 kw::opt_level = opt_level, kw::force_avx512 = force_avx512,
                                 kw::slp_vectorize = slp_vectorize, kw::batch_size = batch_size,
                                 kw::fast_math = fast_math, kw::prec = prec,
                                 // NOTE: it is important to disable the prec checking
                                 // here as we will have our own custom implementation
                                 // of precision checking to deal with NumPy arrays.
                                 kw::check_prec = false, kw::code_model = code_model, kw::parjit = parjit};
        }),
        "fn"_a, "vars"_a, HEYOKA_PY_CFUNC_ARGS(default_cm<T>), HEYOKA_PY_LLVM_STATE_ARGS);

    // Typedefs for the call operator.
    using array_or_iter_t = std::variant<py::array, py::iterable>;
    using time_arg_t = std::variant<T, array_or_iter_t>;

    cfunc_inst.def(
        "__call__",
        [](hey::cfunc<T> &self, array_or_iter_t inputs_ob, std::optional<py::array> outputs_ob,
           std::optional<array_or_iter_t> pars_ob, std::optional<time_arg_t> time_ob) {
            // Fetch the dtype corresponding to T.
            const auto dt = get_dtype<T>();

            // Fetch the inputs array.
            // NOTE: this will either fetch the existing array, or convert
            // the input iterable to a py::array on the fly.
            const auto inputs = std::visit(
                [&](auto &v) {
                    if constexpr (std::same_as<std::remove_cvref_t<decltype(v)>, py::array>) {
                        // Check the dtype.
                        if (v.dtype().num() != dt) [[unlikely]] {
                            py_throw(
                                PyExc_TypeError,
                                fmt::format(
                                    "Invalid dtype detected for the inputs of a compiled function: the expected dtype "
                                    "is '{}', but the dtype of the inputs array is '{}' instead",
                                    str(py::dtype(dt)), str(v.dtype()))
                                    .c_str());
                        }

                        // Check that the inputs array is a C-style contiguous array.
                        if (!is_npy_array_carray(v)) [[unlikely]] {
                            py_throw(PyExc_ValueError,
                                     "Invalid inputs array detected: the array is not C-style contiguous, please "
                                     "consider using numpy.ascontiguousarray() to turn it into one");
                        }

                        return std::move(v);
                    } else {
                        return as_carray(v, dt);
                    }
                },
                inputs_ob);

            // Validate the number of dimensions for the inputs.
            // NOTE: this needs to be done regardless of the original type of inputs_ob.
            if (inputs.ndim() != 1 && inputs.ndim() != 2) [[unlikely]] {
                py_throw(PyExc_ValueError, fmt::format("The array of inputs provided for the evaluation "
                                                       "of a compiled function has {} dimensions, "
                                                       "but it must have either 1 or 2 dimensions instead",
                                                       inputs.ndim())
                                               .c_str());
            }

            // Infer if we are in multieval mode from inputs.
            const auto multieval = (inputs.ndim() == 2);

            // Fetch/create/validate the outputs array.
            auto outputs = [&]() {
                if (outputs_ob) {
                    auto &out = *outputs_ob;

                    // Check the dtype.
                    if (out.dtype().num() != dt) [[unlikely]] {
                        py_throw(
                            PyExc_TypeError,
                            fmt::format(
                                "Invalid dtype detected for the outputs of a compiled function: the expected dtype "
                                "is '{}', but the dtype of the outputs array is '{}' instead",
                                str(py::dtype(dt)), str(out.dtype()))
                                .c_str());
                    }

                    // Check C-style contiguous array.
                    if (!is_npy_array_carray(out)) [[unlikely]] {
                        py_throw(PyExc_ValueError,
                                 "Invalid outputs array detected: the array is not C-style contiguous, please "
                                 "consider using numpy.ascontiguousarray() to turn it into one");
                    }

                    // The array must be writeable.
                    if (!out.writeable()) [[unlikely]] {
                        py_throw(PyExc_ValueError, "The array of outputs provided for the evaluation "
                                                   "of a compiled function is not writeable");
                    }

                    // Validate the number of dimensions for the outputs.
                    if (out.ndim() != inputs.ndim()) [[unlikely]] {
                        py_throw(PyExc_ValueError,
                                 fmt::format("The array of outputs provided for the evaluation "
                                             "of a compiled function has {} dimension(s), "
                                             "but it must have {} dimension(s) instead (i.e., the same "
                                             "number of dimensions as the array of inputs)",
                                             out.ndim(), inputs.ndim())
                                     .c_str());
                    }

                    // NOTE: the rest of the validation is done on the C++ side.
                    return std::move(out);
                } else {
                    if (multieval) {
                        return py::array(inputs.dtype(),
                                         py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(self.get_nouts()),
                                                                   inputs.shape(1)});
                    } else {
                        return py::array(inputs.dtype(),
                                         py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(self.get_nouts())});
                    }
                }
            }();

            // Fetch/create/validate the pars array.
            const auto pars = [&]() -> std::optional<py::array> {
                if (pars_ob) {
                    // pars was supplied.
                    auto pars = std::visit(
                        [&](auto &v) {
                            if constexpr (std::same_as<std::remove_cvref_t<decltype(v)>, py::array>) {
                                // Check the dtype.
                                if (v.dtype().num() != dt) [[unlikely]] {
                                    py_throw(
                                        PyExc_TypeError,
                                        fmt::format("Invalid dtype detected for the parameters of a compiled "
                                                    "function: the expected dtype "
                                                    "is '{}', but the dtype of the parameters array is '{}' instead",
                                                    str(py::dtype(dt)), str(v.dtype()))
                                            .c_str());
                                }

                                // Check C-style contiguous array.
                                if (!is_npy_array_carray(v)) [[unlikely]] {
                                    py_throw(PyExc_ValueError,
                                             "Invalid parameters array detected: the array is not C-style contiguous, "
                                             "please consider using numpy.ascontiguousarray() to turn it into one");
                                }

                                return std::move(v);
                            } else {
                                return as_carray(v, dt);
                            }
                        },
                        *pars_ob);

                    // Validate the number of dimensions.
                    // NOTE: this needs to be done regardless of the original type of pars_ob.
                    if (pars.ndim() != inputs.ndim()) [[unlikely]] {
                        py_throw(PyExc_ValueError,
                                 fmt::format("The array of parameters provided for the evaluation "
                                             "of a compiled function has {} dimension(s), "
                                             "but it must have {} dimension(s) instead (i.e., the same "
                                             "number of dimensions as the array of inputs)",
                                             pars.ndim(), inputs.ndim())
                                     .c_str());
                    }

                    return pars;
                } else {
                    // No pars supplied.
                    return {};
                }
            }();

            // Fetch/create/validate the time value/array.
            const auto time = [&]() -> std::optional<std::variant<T, py::array>> {
                if (time_ob) {
                    auto &tm = *time_ob;

                    if (std::holds_alternative<T>(tm)) {
                        if (multieval) [[unlikely]] {
                            py_throw(PyExc_TypeError,
                                     "The time value cannot be a scalar when evaluating a compiled function "
                                     "over batches of inputs, it should be an array of values instead");
                        }

                        return std::move(std::get<T>(tm));
                    }

                    if (!multieval) [[unlikely]] {
                        py_throw(PyExc_TypeError,
                                 "The time value cannot be an array when evaluating a compiled function over a single "
                                 "set of inputs, it should be a scalar instead");
                    }

                    auto time = std::visit(
                        [&](auto &v) {
                            if constexpr (std::same_as<std::remove_cvref_t<decltype(v)>, py::array>) {
                                // Check the dtype.
                                if (v.dtype().num() != dt) [[unlikely]] {
                                    py_throw(PyExc_TypeError,
                                             fmt::format("Invalid dtype detected for the time values of a compiled "
                                                         "function: the expected dtype "
                                                         "is '{}', but the dtype of the time array is '{}' instead",
                                                         str(py::dtype(dt)), str(v.dtype()))
                                                 .c_str());
                                }

                                // Check C-style contiguous array.
                                if (!is_npy_array_carray(v)) [[unlikely]] {
                                    py_throw(PyExc_ValueError,
                                             "Invalid time array detected: the array is not C-style contiguous, "
                                             "please consider using numpy.ascontiguousarray() to turn it into one");
                                }

                                return std::move(v);
                            } else {
                                return as_carray(v, dt);
                            }
                        },
                        std::get<array_or_iter_t>(tm));

                    // Dimensionality check.
                    // NOTE: we know are in multieval mode at this point, we need a 1D array of times.
                    if (time.ndim() != 1) [[unlikely]] {
                        py_throw(PyExc_ValueError, fmt::format("The array of times provided for the evaluation "
                                                               "of a compiled function has {} dimension(s), "
                                                               "but it must be one-dimensional instead",
                                                               time.ndim())
                                                       .c_str());
                    }

                    return time;
                } else {
                    return {};
                }
            }();

#if defined(HEYOKA_HAVE_REAL)

            // Run the checks specific for mppp::real.
            if constexpr (std::is_same_v<T, mppp::real>) {
                // Check that the inputs array contains values with the correct precision.
                pyreal_check_array(inputs, self.get_prec());

                // Ensure that the outputs array contains constructed values with the correct precision.
                pyreal_ensure_array(outputs, self.get_prec());

                if (pars) {
                    // Check that the pars array is filled with constructed values with the correct precision.
                    pyreal_check_array(*pars, self.get_prec());
                }

                if (time) {
                    if (multieval) {
                        // Check that the time array is filled with constructed values with the correct precision.
                        pyreal_check_array(std::get<py::array>(*time), self.get_prec());
                    } else {
                        // Check that the scalar time value has the correct precision.
                        if (std::get<mppp::real>(*time).get_prec() != self.get_prec()) [[unlikely]] {
                            py_throw(PyExc_ValueError,
                                     fmt::format("An invalid time value was passed for the evaluation of a compiled "
                                                 "function in multiprecision mode: the time value has a precision of "
                                                 "{}, while the expected precision is {} instead",
                                                 std::get<mppp::real>(*time).get_prec(), self.get_prec())
                                         .c_str());
                        }
                    }
                }
            }

#endif

            // Run the overlapping memory checks.
            bool maybe_share_memory{};

            if (pars) {
                if (time && multieval) {
                    maybe_share_memory = may_share_memory(inputs, outputs, *pars, std::get<py::array>(*time));
                } else {
                    maybe_share_memory = may_share_memory(inputs, outputs, *pars);
                }
            } else {
                if (time && multieval) {
                    maybe_share_memory = may_share_memory(inputs, outputs, std::get<py::array>(*time));
                } else {
                    maybe_share_memory = may_share_memory(inputs, outputs);
                }
            }

            if (maybe_share_memory) [[unlikely]] {
                py_throw(PyExc_ValueError, "Potential memory overlaps detected when attempting to evaluate a compiled "
                                           "function: please make sure that all input arrays are distinct");
            }

            // Construct the mdspans and invoke the C++ function.
            using in_1d = typename hey::cfunc<T>::in_1d;
            using out_1d = typename hey::cfunc<T>::out_1d;
            using in_2d = typename hey::cfunc<T>::in_2d;
            using out_2d = typename hey::cfunc<T>::out_2d;

            if (multieval) {
                in_2d in_span{static_cast<const T *>(inputs.data()), boost::numeric_cast<std::size_t>(inputs.shape(0)),
                              boost::numeric_cast<std::size_t>(inputs.shape(1))};
                out_2d out_span{static_cast<T *>(outputs.mutable_data()),
                                boost::numeric_cast<std::size_t>(outputs.shape(0)),
                                boost::numeric_cast<std::size_t>(outputs.shape(1))};
                // NOTE: if no pars are supplied, create a nullptr empty span in which the number
                // of rows is zero and the number of columns is equal to the number of columns in inputs. This ensures
                // that the C++ checks on the pars span do not fail, as the C++ code expects a correct shape if pars is
                // supplied.
                in_2d par_span{pars ? static_cast<const T *>(pars->data()) : nullptr,
                               pars ? boost::numeric_cast<std::size_t>(pars->shape(0)) : 0u,
                               pars ? boost::numeric_cast<std::size_t>(pars->shape(1))
                                    : boost::numeric_cast<std::size_t>(inputs.shape(1))};

                // NOTE: we need to branch on the presence of time because if time is not available,
                // it is not possible to construct an empty span that satisfies the checks on the C++ side.
                // If a time span is provided, the C++ code expects its size to be equal to nevals and,
                // because the time span is 1d, we cannot use the same trick used in the par span of
                // constructing an empty nullptr span with the correct second dimension.
                if (time) {
                    // NOTE: I am not 100% sure if pybind11 needs to call into the interpreter
                    // when fetching data/shape from the time array. Better safe than sorry,
                    // delay GIL unlock until next line.
                    in_1d time_span{static_cast<const T *>(std::get<py::array>(*time).data()),
                                    boost::numeric_cast<std::size_t>(std::get<py::array>(*time).shape(0))};

                    // NOTE: release the GIL during evaluation.
                    py::gil_scoped_release release;

                    self(out_span, in_span, kw::pars = par_span, kw::time = time_span);
                } else {
                    // NOTE: release the GIL during evaluation.
                    py::gil_scoped_release release;

                    self(out_span, in_span, kw::pars = par_span);
                }
            } else {
                in_1d in_span{static_cast<const T *>(inputs.data()), boost::numeric_cast<std::size_t>(inputs.shape(0))};
                out_1d out_span{static_cast<T *>(outputs.mutable_data()),
                                boost::numeric_cast<std::size_t>(outputs.shape(0))};
                in_1d par_span{pars ? static_cast<const T *>(pars->data()) : nullptr,
                               pars ? boost::numeric_cast<std::size_t>(pars->shape(0)) : 0u};

                // NOTE: release the GIL during evaluation.
                py::gil_scoped_release release;

                if (time) {
                    self(out_span, in_span, kw::pars = par_span, kw::time = std::move(std::get<T>(*time)));
                } else {
                    self(out_span, in_span, kw::pars = par_span);
                }
            }

            return outputs;
        },
        "inputs"_a.noconvert(), "outputs"_a.noconvert() = py::none{}, "pars"_a.noconvert() = py::none{},
        "time"_a.noconvert() = py::none{});

    cfunc_inst.def_property_readonly("fn", &hey::cfunc<T>::get_fn);
    cfunc_inst.def_property_readonly("vars", &hey::cfunc<T>::get_vars);
    cfunc_inst.def_property_readonly("dc", &hey::cfunc<T>::get_dc);
    cfunc_inst.def_property_readonly("llvm_states", [](const hey::cfunc<T> &self) {
        const auto &st = self.get_llvm_states();

        using ret_t = std::variant<std::reference_wrapper<const std::array<hey::llvm_state, 3>>,
                                   std::reference_wrapper<const hey::llvm_multi_state>>;

        return std::visit([](const auto &v) -> ret_t { return std::cref(v); }, st);
    });
    cfunc_inst.def_property_readonly("high_accuracy", &hey::cfunc<T>::get_high_accuracy);
    cfunc_inst.def_property_readonly("compact_mode", &hey::cfunc<T>::get_compact_mode);
    cfunc_inst.def_property_readonly("parallel_mode", &hey::cfunc<T>::get_parallel_mode);
    cfunc_inst.def_property_readonly("batch_size", &hey::cfunc<T>::get_batch_size);
    cfunc_inst.def_property_readonly("nparams", &hey::cfunc<T>::get_nparams);
    cfunc_inst.def_property_readonly("nvars", &hey::cfunc<T>::get_nvars);
    cfunc_inst.def_property_readonly("nouts", &hey::cfunc<T>::get_nouts);
    cfunc_inst.def_property_readonly("is_time_dependent", &hey::cfunc<T>::is_time_dependent);

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        cfunc_inst.def_property_readonly("prec", &hey::cfunc<T>::get_prec);
    }

#endif

    // Copy/deepcopy.
    cfunc_inst.def("__copy__", copy_wrapper<hey::cfunc<T>>);
    cfunc_inst.def("__deepcopy__", deepcopy_wrapper<hey::cfunc<T>>, "memo"_a);

    // Pickle support.
    cfunc_inst.def(py::pickle(&pickle_getstate_wrapper<hey::cfunc<T>>, &pickle_setstate_wrapper<hey::cfunc<T>>));

    // Repr.
    cfunc_inst.def("__repr__", [](const hey::cfunc<T> &cf) {
        std::ostringstream oss;
        oss << cf;
        return oss.str();
    });
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
