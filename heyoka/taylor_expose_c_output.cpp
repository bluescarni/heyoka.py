// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <sstream>
#include <string>
#include <type_traits>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/taylor.hpp>

#include "common_utils.hpp"
#include "pickle_wrappers.hpp"
#include "taylor_expose_c_output.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;

namespace detail
{

namespace
{

// Small helper to refer to mppp::real128,
// if it exists.
using fp_128_t =
#if defined(HEYOKA_HAVE_REAL128)
    mppp::real128
#else
    void
#endif
    ;

// Exposition for the scalar continuous output.
template <typename T>
void expose_c_output_impl(py::module &m, const std::string &suffix)
{
    using namespace pybind11::literals;

    using c_output_t = hey::continuous_output<T>;
    using array_arg_t = std::conditional_t<std::is_same_v<T, fp_128_t>, py::iterable, py::array_t<T>>;

    const auto name = fmt::format("continuous_output_{}", suffix);

    py::class_<c_output_t> c_out_c(m, name.c_str(), py::dynamic_attr{});

    c_out_c.def(py::init<>())
        .def(
            "__call__",
            [](py::object &o, T tm) {
                auto *c_out = py::cast<c_output_t *>(o);

                // NOTE: run the computation first, so that if c_out
                // is def-cted an exception will be thrown.
                (*c_out)(tm);

                if constexpr (std::is_same_v<T, fp_128_t>) {
                    return py::array(py::cast(c_out->get_output()));
                } else {
                    auto ret = py::array_t<T>(
                        py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(c_out->get_output().size())},
                        c_out->get_output().data(), o);

                    // Ensure the returned array is read-only.
                    ret.attr("flags").attr("writeable") = false;

                    return ret;
                }
            },
            "time"_a)
        .def(
            "__call__",
            [](py::object &o, const array_arg_t &tm) {
                auto *c_out = py::cast<c_output_t *>(o);

                // Number of columns in the return value.
                const auto ncols = boost::numeric_cast<py::ssize_t>(c_out->get_output().size());

                if constexpr (std::is_same_v<T, fp_128_t>) {
                    // Compute the number of rows.
                    const auto nrows = py::len(tm);

                    // The output vector.
                    std::vector<T> out;

                    // Do the computation.
                    for (const auto &t : tm) {
                        (*c_out)(t.template cast<fp_128_t>());

                        out.insert(out.end(), c_out->get_output().begin(), c_out->get_output().end());
                    }

                    auto ret = py::array(py::cast(out));

                    ret.attr("shape") = py::make_tuple(nrows, ncols);

                    return ret;
                } else {
                    // NOTE: investigate about GIL release here. Can we use unchecked
                    // access to tm without holding the GIL?

                    // Check the shape of tm.
                    if (tm.ndim() != 1) {
                        py_throw(PyExc_ValueError,
                                 fmt::format("Invalid time array passed to a continuous_output object: the "
                                             "number of dimensions must be 1, but it is "
                                             "{} instead",
                                             tm.ndim())
                                     .c_str());
                    }

                    // Compute the number of rows.
                    const auto nrows = tm.shape(0);

                    // Unchecked accessor to tm.
                    auto u_tm = tm.template unchecked<1>();

                    // Prepare the output object.
                    auto ret = py::array_t<T>(py::array::ShapeContainer{nrows, ncols});

                    // Fetch a pointer for writing.
                    auto ret_ptr = ret.mutable_data();

                    // Do the computation.
                    for (py::ssize_t i = 0; i < nrows; ++i) {
                        (*c_out)(u_tm(i));

                        std::copy(c_out->get_output().begin(), c_out->get_output().end(), ret_ptr + i * ncols);
                    }

                    return ret;
                }
            },
            "time"_a)
        .def_property_readonly(
            "output",
            [](const py::object &o) -> py::object {
                auto *c_out = py::cast<const c_output_t *>(o);

                if (c_out->get_output().empty()) {
                    return py::none{};
                } else {
                    if constexpr (std::is_same_v<T, fp_128_t>) {
                        return py::array(py::cast(c_out->get_output()));
                    } else {
                        auto ret = py::array_t<T>(
                            py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(c_out->get_output().size())},
                            c_out->get_output().data(), o);

                        // Ensure the returned array is read-only.
                        ret.attr("flags").attr("writeable") = false;

                        return std::move(ret);
                    }
                }
            })
        .def_property_readonly(
            "times",
            [](const py::object &o) -> py::object {
                auto *c_out = py::cast<const c_output_t *>(o);

                if (c_out->get_times().empty()) {
                    return py::none{};
                } else {
                    if constexpr (std::is_same_v<T, fp_128_t>) {
                        return py::array(py::cast(c_out->get_times()));
                    } else {
                        auto ret = py::array_t<T>(
                            py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(c_out->get_times().size())},
                            c_out->get_times().data(), o);

                        // Ensure the returned array is read-only.
                        ret.attr("flags").attr("writeable") = false;

                        return std::move(ret);
                    }
                }
            })
        .def_property_readonly("tcs",
                               [](const py::object &o) -> py::object {
                                   auto *c_out = py::cast<const c_output_t *>(o);

                                   if (c_out->get_tcs().empty()) {
                                       return py::none{};
                                   } else {
                                       const auto tc_size = c_out->get_tcs().size();
                                       const auto n_steps = c_out->get_n_steps();
                                       const auto nvars = c_out->get_output().size();

                                       assert(tc_size % (n_steps * nvars) == 0u);

                                       // NOTE: this is the number of coefficients per
                                       // state variable, i.e., the Taylor order + 1.
                                       const auto ncoeffs = tc_size / (n_steps * nvars);

                                       if constexpr (std::is_same_v<T, fp_128_t>) {
                                           auto ret = py::array(py::cast(c_out->get_tcs()));

                                           ret.attr("shape") = py::make_tuple(n_steps, nvars, ncoeffs);

                                           return std::move(ret);
                                       } else {
                                           auto ret = py::array_t<T>(
                                               py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(n_steps),
                                                                         boost::numeric_cast<py::ssize_t>(nvars),
                                                                         boost::numeric_cast<py::ssize_t>(ncoeffs)},
                                               c_out->get_tcs().data(), o);

                                           // Ensure the returned array is read-only.
                                           ret.attr("flags").attr("writeable") = false;

                                           return std::move(ret);
                                       }
                                   }
                               })
        .def_property_readonly("bounds", &c_output_t::get_bounds)
        .def_property_readonly("n_steps", &c_output_t::get_n_steps)
        // Repr.
        .def("__repr__",
             [](const c_output_t &c) {
                 std::ostringstream oss;
                 oss << c;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", [](const c_output_t &c) { return c; })
        .def(
            "__deepcopy__", [](const c_output_t &c, py::dict) { return c; }, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pickle_getstate_wrapper<c_output_t>, &pickle_setstate_wrapper<c_output_t>));

    // Expose the llvm state getter.
    expose_llvm_state_property(c_out_c);
}

// Exposition for the batch continuous output.
template <typename T>
void expose_c_output_batch_impl(py::module &m, const std::string &suffix)
{
    using namespace pybind11::literals;

    using c_output_t = hey::continuous_output_batch<T>;

    const auto name = fmt::format("continuous_output_batch_{}", suffix);

    py::class_<c_output_t> c_out_c(m, name.c_str(), py::dynamic_attr{});

    c_out_c.def(py::init<>())
        .def(
            "__call__",
            [](py::object &o, const py::array_t<T> &tm) {
                auto *c_out = py::cast<c_output_t *>(o);

                if (c_out->get_output().empty()) {
                    py_throw(PyExc_ValueError, "Cannot use a default-constructed continuous_output_batch object");
                }

                const auto batch_size = c_out->get_batch_size();
                assert(batch_size > 0u);
                assert(c_out->get_output().size() % batch_size == 0u);
                const auto dim = c_out->get_output().size() / batch_size;

                // The shape of tm must be either
                // - (batch_size, ) (i.e., compute for a single time batch), or
                // - (n, batch_size) (i.e., compute for a n time batches).
                if (tm.ndim() != 1 && tm.ndim() != 2) {
                    py_throw(PyExc_ValueError,
                             fmt::format("Invalid time array passed to a continuous_output_batch object: the "
                                         "number of dimensions must be 1 or 2, but it is "
                                         "{} instead",
                                         tm.ndim())
                                 .c_str());
                }

                if (tm.ndim() == 1) {
                    // Single time batch.
                    if (tm.shape(0) != boost::numeric_cast<py::ssize_t>(batch_size)) {
                        py_throw(PyExc_ValueError,
                                 fmt::format("Invalid time array passed to a continuous_output_batch object: the "
                                             "length must be {} but it is {} instead",
                                             batch_size, tm.shape(0))
                                     .c_str());
                    }

                    // Check if tm is contiguous.
                    // NOTE: it looks like c_style does not necessarily imply well-aligned:
                    // https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_FromAny.NPY_ARRAY_CARRAY
                    // If this becomes a problem, we can tap into the NumPy C API and do additional
                    // flag checking:
                    // https://numpy.org/devdocs/reference/c-api/array.html#c.PyArray_CHKFLAGS
                    const auto tm_cont = static_cast<bool>(tm.flags() & py::array::c_style);

                    if (tm_cont) {
                        // tm is contiguous.
                        (*c_out)(tm.data());
                    } else {
                        // tm is not contiguous.
                        auto tm_copy = py::cast<std::vector<T>>(tm);
                        (*c_out)(tm_copy);
                    }

                    auto ret = py::array_t<T>(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(dim),
                                                                        boost::numeric_cast<py::ssize_t>(batch_size)},
                                              c_out->get_output().data(), o);

                    // Ensure the returned array is read-only.
                    ret.attr("flags").attr("writeable") = false;

                    return ret;
                } else {
                    // Multiple time batches.
                    if (tm.shape(1) != boost::numeric_cast<py::ssize_t>(batch_size)) {
                        py_throw(PyExc_ValueError,
                                 fmt::format("Invalid time array passed to a continuous_output_batch object: the "
                                             "number of columns must be {} but it is {} instead",
                                             batch_size, tm.shape(1))
                                     .c_str());
                    }

                    // Fetch the number of rows.
                    const auto nrows = tm.shape(0);

                    // Setup the return value.
                    auto ret = py::array_t<T>(py::array::ShapeContainer{nrows, boost::numeric_cast<py::ssize_t>(dim),
                                                                        boost::numeric_cast<py::ssize_t>(batch_size)});

                    // Fetch a pointer for writing.
                    auto ret_ptr = ret.mutable_data();

                    // Unchecked accessor to tm.
                    auto u_tm = tm.template unchecked<2>();

                    // A temporary buffer that we will use in the calls to c_out.
                    std::vector<T> tmp_buffer;
                    tmp_buffer.resize(boost::numeric_cast<decltype(tmp_buffer.size())>(batch_size));

                    // NOTE: investigate about GIL release here. Can we use unchecked
                    // access to tm without holding the GIL?

                    // Run the computation.
                    for (py::ssize_t i = 0; i < nrows; ++i) {
                        // Copy the current time batch to tmp_buffer.
                        for (py::ssize_t j = 0; j < boost::numeric_cast<py::ssize_t>(batch_size); ++j) {
                            tmp_buffer.data()[j] = u_tm(i, j);
                        }

                        // Compute the continuous output.
                        (*c_out)(tmp_buffer);

                        // Copy over to ret.
                        std::copy(c_out->get_output().begin(), c_out->get_output().end(),
                                  ret_ptr
                                      + i * boost::numeric_cast<py::ssize_t>(dim)
                                            * boost::numeric_cast<py::ssize_t>(batch_size));
                    }

                    return ret;
                }
            },
            "time"_a)
        .def_property_readonly("output",
                               [](const py::object &o) -> py::object {
                                   auto *c_out = py::cast<const c_output_t *>(o);

                                   if (c_out->get_output().empty()) {
                                       return py::none{};
                                   } else {
                                       const auto batch_size = c_out->get_batch_size();
                                       assert(batch_size > 0u);
                                       assert(c_out->get_output().size() % batch_size == 0u);
                                       const auto dim = c_out->get_output().size() / batch_size;

                                       auto ret = py::array_t<T>(
                                           py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(dim),
                                                                     boost::numeric_cast<py::ssize_t>(batch_size)},
                                           c_out->get_output().data(), o);

                                       // Ensure the returned array is read-only.
                                       ret.attr("flags").attr("writeable") = false;

                                       return std::move(ret);
                                   }
                               })
        .def_property_readonly("times",
                               [](const py::object &o) -> py::object {
                                   auto *c_out = py::cast<const c_output_t *>(o);

                                   if (c_out->get_times().empty()) {
                                       return py::none{};
                                   } else {
                                       const auto batch_size = c_out->get_batch_size();
                                       assert(batch_size > 0u);
                                       assert(c_out->get_times().size() % batch_size == 0u);
                                       auto dim = c_out->get_times().size() / batch_size;
                                       assert(dim > 0u);
                                       // NOTE: the times vector contains padding, remove it
                                       // in the return value.
                                       --dim;

                                       auto ret = py::array_t<T>(
                                           py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(dim),
                                                                     boost::numeric_cast<py::ssize_t>(batch_size)},
                                           c_out->get_times().data(), o);

                                       // Ensure the returned array is read-only.
                                       ret.attr("flags").attr("writeable") = false;

                                       return std::move(ret);
                                   }
                               })
        .def_property_readonly("tcs",
                               [](const py::object &o) -> py::object {
                                   auto *c_out = py::cast<const c_output_t *>(o);

                                   if (c_out->get_tcs().empty()) {
                                       return py::none{};
                                   } else {
                                       const auto batch_size = c_out->get_batch_size();
                                       assert(batch_size > 0u);

                                       const auto tc_size = c_out->get_tcs().size();

                                       // NOTE: get_n_steps() accounts for padding.
                                       const auto n_steps = c_out->get_n_steps();
                                       assert(c_out->get_output().size() % batch_size == 0u);

                                       const auto nvars = c_out->get_output().size() / batch_size;

                                       // NOTE: this is the number of coefficients per
                                       // state variable, i.e., the Taylor order + 1.
                                       assert(tc_size % (n_steps * nvars * batch_size) == 0u);
                                       const auto ncoeffs = tc_size / (n_steps * nvars * batch_size);

                                       auto ret = py::array_t<T>(
                                           py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(n_steps),
                                                                     boost::numeric_cast<py::ssize_t>(nvars),
                                                                     boost::numeric_cast<py::ssize_t>(ncoeffs),
                                                                     boost::numeric_cast<py::ssize_t>(batch_size)},
                                           c_out->get_tcs().data(), o);

                                       // Ensure the returned array is read-only.
                                       ret.attr("flags").attr("writeable") = false;

                                       return std::move(ret);
                                   }
                               })
        .def_property_readonly("bounds",
                               [](const c_output_t &c_out) {
                                   auto ret = c_out.get_bounds();

                                   return py::make_tuple(py::array_t<T>(py::cast(ret.first)),
                                                         py::array_t<T>(py::cast(ret.second)));
                               })
        .def_property_readonly("n_steps", &c_output_t::get_n_steps)
        .def_property_readonly("batch_size", &c_output_t::get_batch_size)
        // Repr.
        .def("__repr__",
             [](const c_output_t &c) {
                 std::ostringstream oss;
                 oss << c;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", [](const c_output_t &c) { return c; })
        .def(
            "__deepcopy__", [](const c_output_t &c, py::dict) { return c; }, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pickle_getstate_wrapper<c_output_t>, &pickle_setstate_wrapper<c_output_t>));

    // Expose the llvm state getter.
    expose_llvm_state_property(c_out_c);
}

} // namespace

} // namespace detail

void taylor_expose_c_output(py::module &m)
{
    // Expose the scalar versions.
    detail::expose_c_output_impl<double>(m, "dbl");
    detail::expose_c_output_impl<long double>(m, "ldbl");

#if defined(HEYOKA_HAVE_REAL128)

    detail::expose_c_output_impl<mppp::real128>(m, "f128");

#endif

    // Expose the batch versions.
    detail::expose_c_output_batch_impl<double>(m, "dbl");
}

} // namespace heyoka_py
