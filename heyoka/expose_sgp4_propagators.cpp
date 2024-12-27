// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/model/sgp4.hpp>

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "docstrings.hpp"
#include "expose_models.hpp"
#include "pickle_wrappers.hpp"

namespace heyoka_py
{

namespace py = pybind11;

namespace detail
{

namespace
{

// Helper to turn a list of sgp4 satellites into a satellite data vector suitable for
// use in the C++ sgp4_propagator API.
template <typename T>
auto sat_list_to_vector(py::list sat_list)
{
    // Import the Satrec class from the sgp4 module.
    py::object satrec = py::module_::import("sgp4.api").attr("Satrec");

    // Prepare the output vector.
    std::vector<T> retval;
    const auto n_sats = boost::safe_numerics::safe<decltype(retval.size())>(py::len(sat_list));
    retval.resize(n_sats * 9);

    // Fill it in.
    py::object isinst = builtins().attr("isinstance");
    for (decltype(retval.size()) i = 0; i < n_sats; ++i) {
        auto sat_obj = sat_list[boost::numeric_cast<py::size_t>(i)];

        if (!py::cast<bool>(isinst(sat_obj, satrec))) [[unlikely]] {
            py_throw(
                PyExc_TypeError,
                fmt::format("Invalid object encountered in the satellite data for an sgp4 propagator: a list of sgp4 "
                            "Satrec objects is expected, but an object of type '{}' was detected instead at index {}",
                            str(type(sat_obj)), i)
                    .c_str());
        }

        retval[i] = static_cast<T>(sat_obj.attr("no_kozai").template cast<double>());
        retval[i + n_sats] = static_cast<T>(sat_obj.attr("ecco").template cast<double>());
        retval[i + n_sats * 2] = static_cast<T>(sat_obj.attr("inclo").template cast<double>());
        retval[i + n_sats * 3] = static_cast<T>(sat_obj.attr("nodeo").template cast<double>());
        retval[i + n_sats * 4] = static_cast<T>(sat_obj.attr("argpo").template cast<double>());
        retval[i + n_sats * 5] = static_cast<T>(sat_obj.attr("mo").template cast<double>());
        retval[i + n_sats * 6] = static_cast<T>(sat_obj.attr("bstar").template cast<double>());
        retval[i + n_sats * 7] = static_cast<T>(sat_obj.attr("jdsatepoch").template cast<double>());
        retval[i + n_sats * 8] = static_cast<T>(sat_obj.attr("jdsatepochF").template cast<double>());
    }

    return retval;
}

// Helper to validate a satellite list provided as a 2D
// NumPy array. A span into v will be returned.
template <typename T>
auto sat_list_array_to_span(const py::array_t<T> &v)
{
    namespace hy = heyoka;

    // Check that the input array is C style and contiguous.
    if (!is_npy_array_carray(v)) [[unlikely]] {
        py_throw(PyExc_ValueError,
                 "Invalid array of input GPEs detected in an sgp4 propagator: the array is not C-style "
                 "contiguous, please consider using numpy.ascontiguousarray() to turn it into one");
    }

    // Check dimensionality and shape.
    if (v.ndim() != 2) [[unlikely]] {
        py_throw(PyExc_ValueError, fmt::format("The array of input GPEs for an sgp4 propagator must have 2 "
                                               "dimensions, but the supplied array has {} dimension(s) instead",
                                               v.ndim())
                                       .c_str());
    }
    if (v.shape(0) != 9) [[unlikely]] {
        py_throw(PyExc_ValueError, fmt::format("The array of input GPEs for an sgp4 propagator must have 9 "
                                               "rows, but the supplied array has {} row(s) instead",
                                               v.shape(0))
                                       .c_str());
    }

    // Create the input span for the constructor.
    using span_t = hy::mdspan<const T, hy::extents<std::size_t, 9, std::dynamic_extent>>;
    return span_t(v.data(), boost::numeric_cast<std::size_t>(v.shape(1)));
}

template <typename T>
void expose_sgp4_propagator_impl(py::module_ &m, const std::string &suffix)
{
    namespace hy = heyoka;
    namespace kw = hy::kw;
    // NOLINTNEXTLINE(google-build-using-namespace)
    using namespace py::literals;
    using prop_t = hy::model::sgp4_propagator<T>;

    // Register the date type as a numpy dtype.
    using date_t = typename prop_t::date;
    PYBIND11_NUMPY_DTYPE(date_t, jd, frac);

    py::class_<prop_t> prop_cl(m, fmt::format("_model_sgp4_propagator_{}", suffix).c_str(), py::dynamic_attr{},
                               docstrings::sgp4_propagator(std::same_as<T, double> ? "double" : "single").c_str());
    prop_cl.def(
        py::init([](std::variant<py::list, py::array_t<T>> sat_list, std::uint32_t diff_order, bool high_accuracy,
                    bool compact_mode, bool parallel_mode, std::uint32_t batch_size, long long, unsigned opt_level,
                    bool force_avx512, bool slp_vectorize, bool fast_math, hy::code_model code_model, bool parjit) {
            return std::visit(
                [&]<typename V>(const V &v) {
                    if constexpr (std::same_as<V, py::list>) {
                        // Check that the sgp4 module is available.
                        try {
                            py::module_::import("sgp4.api");
                        } catch (...) {
                            py_throw(PyExc_ImportError,
                                     "The Python module 'sgp4' must be installed in order to be "
                                     "able to build sgp4 propagators from a list of satellite objects");
                        }

                        // Turn sat_list into a data vector.
                        const auto sat_data = sat_list_to_vector<T>(v);
                        assert(sat_data.size() % 9u == 0u);

                        // Create the input span for the constructor.
                        using span_t = hy::mdspan<const T, hy::extents<std::size_t, 9, std::dynamic_extent>>;
                        const span_t in(sat_data.data(), boost::numeric_cast<std::size_t>(sat_data.size()) / 9u);

                        // NOTE: release the GIL during compilation.
                        py::gil_scoped_release release;

                        return prop_t{in,
                                      kw::diff_order = diff_order,
                                      kw::high_accuracy = high_accuracy,
                                      kw::compact_mode = compact_mode,
                                      kw::parallel_mode = parallel_mode,
                                      kw::batch_size = batch_size,
                                      kw::opt_level = opt_level,
                                      kw::force_avx512 = force_avx512,
                                      kw::slp_vectorize = slp_vectorize,
                                      kw::fast_math = fast_math,
                                      kw::code_model = code_model,
                                      kw::parjit = parjit};
                    } else {
                        // Check v and fetch a span to it.
                        const auto in = sat_list_array_to_span(v);

                        // NOTE: release the GIL during compilation.
                        py::gil_scoped_release release;

                        return prop_t{in,
                                      kw::diff_order = diff_order,
                                      kw::high_accuracy = high_accuracy,
                                      kw::compact_mode = compact_mode,
                                      kw::parallel_mode = parallel_mode,
                                      kw::batch_size = batch_size,
                                      kw::opt_level = opt_level,
                                      kw::force_avx512 = force_avx512,
                                      kw::slp_vectorize = slp_vectorize,
                                      kw::fast_math = fast_math,
                                      kw::code_model = code_model,
                                      kw::parjit = parjit};
                    }
                },
                sat_list);
        }),
        "sat_list"_a.noconvert(), "diff_order"_a.noconvert() = static_cast<std::uint32_t>(0),
        HEYOKA_PY_CFUNC_ARGS(false), HEYOKA_PY_LLVM_STATE_ARGS,
        docstrings::sgp4_propagator_init(std::same_as<T, double> ? "float" : "numpy.single").c_str());
    prop_cl.def_property_readonly(
        "jdtype",
        [](const prop_t &) -> py::object {
            auto np = py::module_::import("numpy");

            py::object dtype = np.attr("dtype");

            py::list l;
            l.append(py::make_tuple("jd", std::same_as<T, double> ? np.attr("double") : np.attr("single")));
            l.append(py::make_tuple("frac", std::same_as<T, double> ? np.attr("double") : np.attr("single")));

            return dtype(l);
        },
        docstrings::sgp4_propagator_jdtype(std::same_as<T, double> ? "float" : "numpy.single").c_str());
    prop_cl.def_property_readonly("nsats", &prop_t::get_nsats, docstrings::sgp4_propagator_nsats().c_str());
    prop_cl.def_property_readonly("nouts", &prop_t::get_nouts, docstrings::sgp4_propagator_nouts().c_str());
    prop_cl.def_property_readonly("diff_args", &prop_t::get_diff_args, docstrings::sgp4_propagator_diff_args().c_str());
    prop_cl.def_property_readonly("diff_order", &prop_t::get_diff_order,
                                  docstrings::sgp4_propagator_diff_order().c_str());
    prop_cl.def_property_readonly(
        "sat_data",
        [](const py::object &o) {
            const auto *prop = py::cast<const prop_t *>(o);

            auto sdata = prop->get_sat_data();

            auto ret = py::array_t<T>(py::array::ShapeContainer{static_cast<py::ssize_t>(9),
                                                                boost::numeric_cast<py::ssize_t>(sdata.extent(1))},
                                      sdata.data_handle(), o);

            // Ensure the returned array is read-only.
            ret.attr("flags").attr("writeable") = false;

            // Return.
            return ret;
        },
        docstrings::sgp4_propagator_sat_data(suffix, std::same_as<T, double> ? "float" : "numpy.single").c_str());
    prop_cl.def(
        "replace_sat_data",
        [](prop_t &prop, std::variant<py::list, py::array_t<T>> sat_list) {
            std::visit(
                [&prop]<typename V>(const V &v) {
                    if constexpr (std::same_as<V, py::list>) {
                        // Turn sat_list into a data vector.
                        const auto sat_data = sat_list_to_vector<T>(v);
                        assert(sat_data.size() % 9u == 0u);

                        // Create the input span for the setter.
                        using span_t = hy::mdspan<const T, hy::extents<std::size_t, 9, std::dynamic_extent>>;
                        const span_t in(sat_data.data(), boost::numeric_cast<std::size_t>(sat_data.size()) / 9u);

                        prop.replace_sat_data(in);
                    } else {
                        // Check v and fetch a span to it.
                        const auto in = sat_list_array_to_span(v);

                        prop.replace_sat_data(in);
                    }
                },
                sat_list);
        },
        "sat_list"_a.noconvert(),
        docstrings::sgp4_propagator_replace_sat_data(suffix, std::same_as<T, double> ? "float" : "numpy.single")
            .c_str());
    prop_cl.def(
        "get_dslice",
        [](const prop_t &prop, std::uint32_t order, std::optional<std::uint32_t> component) {
            const auto ret = component ? prop.get_dslice(*component, order) : prop.get_dslice(order);

            return py::slice(boost::numeric_cast<py::ssize_t>(ret.first), boost::numeric_cast<py::ssize_t>(ret.second),
                             {});
        },
        "order"_a.noconvert(), "component"_a.noconvert() = py::none{},
        docstrings::sgp4_propagator_get_dslice().c_str());
    prop_cl.def(
        "get_mindex",
        [](const prop_t &prop, std::uint32_t i) {
            const auto &ret = prop.get_mindex(i);

            return dtens_t_it::sparse_to_dense(
                ret, boost::numeric_cast<heyoka::dtens::v_idx_t::size_type>(prop.get_diff_args().size()));
        },
        "i"_a.noconvert(), docstrings::sgp4_propagator_get_mindex(suffix).c_str());
    prop_cl.def(
        "__call__",
        [](prop_t &prop, std::variant<py::array_t<T>, py::array_t<date_t>> tm_arr,
           std::optional<py::array_t<T>> out) -> py::array_t<T> {
            auto np = py::module_::import("numpy");

            // NOTE: here we are repeating several checks which are redundant with
            // checks already performed on the C++ side, with the goal of providing better
            // error messages.
            return std::visit(
                [&]<typename U>(py::array_t<U> &in_arr) {
                    // Checks on the input array.
                    const auto in_ndim = in_arr.ndim();
                    if (in_ndim != 1 && in_ndim != 2) [[unlikely]] {
                        py_throw(
                            PyExc_ValueError,
                            fmt::format(
                                "A times/dates array with 1 or 2 dimensions is expected as an input for the call "
                                "operator of an sgp4 propagator, but an array with {} dimensions was provided instead",
                                in_ndim)
                                .c_str());
                    }

                    if (in_ndim == 1) {
                        // In scalar mode, the number of elements must match the number of satellites.
                        if (in_arr.shape(0) != boost::numeric_cast<py::ssize_t>(prop.get_nsats())) [[unlikely]] {
                            py_throw(PyExc_ValueError,
                                     fmt::format(
                                         "Invalid times/dates array detected as an input for the call operator of "
                                         "an sgp4 propagator: the number of satellites inferred from the "
                                         "times/dates array is {}, but the propagator contains {} satellite(s) instead",
                                         in_arr.shape(0), prop.get_nsats())
                                         .c_str());
                        }
                    } else {
                        // In batch mode, the number of input columns must match the number of satellites.
                        if (in_arr.shape(1) != boost::numeric_cast<py::ssize_t>(prop.get_nsats())) [[unlikely]] {
                            py_throw(PyExc_ValueError,
                                     fmt::format(
                                         "Invalid times/dates array detected as an input for the call operator of "
                                         "an sgp4 propagator in batch mode: the number of satellites inferred from the "
                                         "times/dates array is {}, but the propagator contains {} satellite(s) instead",
                                         in_arr.shape(1), prop.get_nsats())
                                         .c_str());
                        }
                    }

                    // We need a C array in both scalar and batch mode.
                    if (!is_npy_array_carray(in_arr)) [[unlikely]] {
                        py_throw(PyExc_ValueError,
                                 "Invalid times/dates array detected as an input for the call operator of "
                                 "an sgp4 propagator: the array is not C-style contiguous");
                    }

                    // Establish whether we are operating in scalar or batch mode.
                    std::optional<std::size_t> n_evals;
                    if (in_ndim == 2) {
                        // Batch mode. Set the number of evaluations.
                        n_evals.emplace(boost::numeric_cast<std::size_t>(in_arr.shape(0)));
                    }

                    // Build or fetch the output array.
                    if (out) {
                        // Array provided by the user. We need to check that:
                        //
                        // - it is a writable C array,
                        // - it has no memory overlap with the inputs array,
                        // - it has the correct shape.
                        if (!is_npy_array_carray(*out, true)) [[unlikely]] {
                            py_throw(PyExc_ValueError,
                                     "Invalid output array detected in the call operator of "
                                     "an sgp4 propagator: the array is not C-style contiguous and writeable");
                        }

                        if (may_share_memory(*out, in_arr)) [[unlikely]] {
                            py_throw(PyExc_ValueError, "Invalid input/output arrays detected in the call operator of "
                                                       "an sgp4 propagator: the input/outputs arrays may overlap");
                        }

                        if (n_evals) {
                            // Batch mode.
                            if (out->ndim() != 3) [[unlikely]] {
                                py_throw(PyExc_ValueError,
                                         fmt::format("Invalid output array detected in the call operator of "
                                                     "an sgp4 propagator in batch mode: the array has {} dimension(s), "
                                                     "but 3 dimensions are expected instead",
                                                     out->ndim())
                                             .c_str());
                            }

                            if (out->shape(0) != boost::numeric_cast<py::ssize_t>(*n_evals)) [[unlikely]] {
                                py_throw(
                                    PyExc_ValueError,
                                    fmt::format(
                                        "Invalid output array detected in the call operator of "
                                        "an sgp4 propagator in batch mode: the first dimension has a size of {}, but a "
                                        "size of {} (i.e., equal to the number of evaluations) is required instead",
                                        out->shape(0), *n_evals)
                                        .c_str());
                            }

                            if (out->shape(1) != boost::numeric_cast<py::ssize_t>(prop.get_nouts())) [[unlikely]] {
                                py_throw(PyExc_ValueError,
                                         fmt::format("Invalid output array detected in the call operator of "
                                                     "an sgp4 propagator in batch mode: the second dimension has a "
                                                     "size of {}, but a "
                                                     "size of {} (i.e., equal to the number of outputs for each "
                                                     "propagation) is required instead",
                                                     out->shape(1), prop.get_nouts())
                                             .c_str());
                            }

                            if (out->shape(2) != boost::numeric_cast<py::ssize_t>(prop.get_nsats())) [[unlikely]] {
                                py_throw(PyExc_ValueError,
                                         fmt::format("Invalid output array detected in the call operator of "
                                                     "an sgp4 propagator in batch mode: the third dimension has a "
                                                     "size of {}, but a "
                                                     "size of {} (i.e., equal to the total number of satellites) is "
                                                     "required instead",
                                                     out->shape(2), prop.get_nsats())
                                             .c_str());
                            }
                        } else {
                            // Scalar mode.
                            if (out->ndim() != 2) [[unlikely]] {
                                py_throw(PyExc_ValueError,
                                         fmt::format("Invalid output array detected in the call operator of "
                                                     "an sgp4 propagator: the array has {} dimension(s), "
                                                     "but 2 dimensions are expected instead",
                                                     out->ndim())
                                             .c_str());
                            }

                            if (out->shape(0) != boost::numeric_cast<py::ssize_t>(prop.get_nouts())) [[unlikely]] {
                                py_throw(PyExc_ValueError,
                                         fmt::format("Invalid output array detected in the call operator of "
                                                     "an sgp4 propagator: the first dimension has a "
                                                     "size of {}, but a "
                                                     "size of {} (i.e., equal to the number of outputs for each "
                                                     "propagation) is required instead",
                                                     out->shape(0), prop.get_nouts())
                                             .c_str());
                            }

                            if (out->shape(1) != boost::numeric_cast<py::ssize_t>(prop.get_nsats())) [[unlikely]] {
                                py_throw(PyExc_ValueError,
                                         fmt::format("Invalid output array detected in the call operator of "
                                                     "an sgp4 propagator: the second dimension has a "
                                                     "size of {}, but a "
                                                     "size of {} (i.e., equal to the total number of satellites) is "
                                                     "required instead",
                                                     out->shape(1), prop.get_nsats())
                                             .c_str());
                            }
                        }
                    } else {
                        // Construct an output array.
                        if (n_evals) {
                            // Batch mode.
                            out.emplace(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(*n_evals),
                                                                  boost::numeric_cast<py::ssize_t>(prop.get_nouts()),
                                                                  boost::numeric_cast<py::ssize_t>(prop.get_nsats())});
                        } else {
                            // Scalar mode.
                            out.emplace(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(prop.get_nouts()),
                                                                  boost::numeric_cast<py::ssize_t>(prop.get_nsats())});
                        }
                    }

                    // Create the spans and invoke the call operator.
                    if (n_evals) {
                        // Batch mode.
                        typename prop_t::out_3d out_span(out->mutable_data(),
                                                         boost::numeric_cast<std::size_t>(out->shape(0)),
                                                         boost::numeric_cast<std::size_t>(out->shape(1)),
                                                         boost::numeric_cast<std::size_t>(out->shape(2)));

                        typename prop_t::template in_2d<U> in_span(in_arr.data(),
                                                                   boost::numeric_cast<std::size_t>(in_arr.shape(0)),
                                                                   boost::numeric_cast<std::size_t>(in_arr.shape(1)));

                        // NOTE: release the GIL during propagation.
                        py::gil_scoped_release release;

                        prop(out_span, in_span);
                    } else {
                        // Scalar mode.
                        typename prop_t::out_2d out_span(out->mutable_data(),
                                                         boost::numeric_cast<std::size_t>(out->shape(0)),
                                                         boost::numeric_cast<std::size_t>(out->shape(1)));

                        typename prop_t::template in_1d<U> in_span(in_arr.data(),
                                                                   boost::numeric_cast<std::size_t>(in_arr.shape(0)));

                        // NOTE: release the GIL during propagation.
                        py::gil_scoped_release release;

                        prop(out_span, in_span);
                    }

                    // Return the result.
                    return std::move(*out);
                },
                tm_arr);
        },
        "times"_a.noconvert(), "out"_a.noconvert() = py::none{},
        docstrings::sgp4_propagator_call(suffix, std::same_as<T, double> ? "float" : "numpy.single").c_str());

    // Copy/deepcopy.
    prop_cl.def("__copy__", copy_wrapper<prop_t>);
    prop_cl.def("__deepcopy__", deepcopy_wrapper<prop_t>, "memo"_a);

    // Pickle support.
    prop_cl.def(py::pickle(&pickle_getstate_wrapper<prop_t>, &pickle_setstate_wrapper<prop_t>));

    // Repr.
    prop_cl.def("__repr__", [](const prop_t &prop) {
        return fmt::format("SGP4 propagator\nN of satellites: {}\n", prop.get_nsats());
    });
}

} // namespace

} // namespace detail

void expose_sgp4_propagators(py::module_ &m)
{
    detail::expose_sgp4_propagator_impl<float>(m, "flt");
    detail::expose_sgp4_propagator_impl<double>(m, "dbl");
}

} // namespace heyoka_py
