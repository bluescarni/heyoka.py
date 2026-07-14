// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <exception>
#include <optional>
#include <ranges>
#include <string>
#include <utility>

#include <boost/align/is_aligned.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/number.hpp>
#include <heyoka/sw_data.hpp>

#include "common_utils.hpp"
#include "custom_casters.hpp"

namespace heyoka_py
{

py::object builtins()
{
    return py::module_::import("builtins");
}

py::object type(const py::handle &o)
{
    return builtins().attr("type")(o);
}

std::string str(const py::handle &o)
{
    return py::cast<std::string>(py::str(o));
}

void py_throw(PyObject *type, const char *msg)
{
    PyErr_SetString(type, msg);
    throw py::error_already_set();
}

// Detect if o is a callable object.
bool callable(const py::handle &o)
{
    return py::cast<bool>(builtins().attr("callable")(o));
}

bool may_share_memory(const py::array &a, const py::array &b)
{
    return py::module_::import("numpy").attr("may_share_memory")(a, b).cast<bool>();
}

// Helper to check if a numpy array is a NPY_ARRAY_CARRAY (i.e., C-style contiguous and with storage properly aligned
// for the array's dtype). The flag signals whether the array must also be writeable or not.
bool is_npy_array_carray(const py::array &arr, const bool writeable)
{
    assert(PyObject_IsInstance(arr.ptr(), reinterpret_cast<PyObject *>(&PyArray_Type)));

    // NOTE: NPY_ARRAY_CARRAY is NPY_ARRAY_CARRAY_RO + writeable flag.
    const auto flags = writeable ? NPY_ARRAY_CARRAY : NPY_ARRAY_CARRAY_RO;
    return PyArray_CHKFLAGS(reinterpret_cast<const PyArrayObject *>(arr.ptr()), flags) != 0;
}

namespace detail
{

bool with_pybind11_eh_impl()
{
    auto &local_exception_translators = py::detail::get_local_internals().registered_exception_translators;
    if (py::detail::apply_exception_translators(local_exception_translators)) {
        return true;
    }
    auto &exception_translators = py::detail::get_internals().registered_exception_translators;
    if (py::detail::apply_exception_translators(exception_translators)) {
        return true;
    }

    PyErr_SetString(PyExc_SystemError, "Exception escaped from default exception translator!");
    return true;
}

} // namespace detail

std::pair<heyoka::dtens::v_idx_t, heyoka::expression>
dtens_t_it::operator()(const std::pair<heyoka::dtens::sv_idx_t, heyoka::expression> &p) const
{
    const auto &[sv_idx, ex] = p;

    return std::make_pair(
        sparse_to_dense(sv_idx, boost::numeric_cast<heyoka::dtens::v_idx_t::size_type>(dt->get_nargs())), ex);
}

heyoka::dtens::v_idx_t dtens_t_it::sparse_to_dense(const heyoka::dtens::sv_idx_t &sv_idx,
                                                   heyoka::dtens::v_idx_t::size_type nargs)
{
    // Init the dense vector from the component index.
    heyoka::dtens::v_idx_t ret{sv_idx.first};

    // Transform the sparse index/order pairs into dense format.
    // NOTE: no overflow check needed on ++idx because dtens ensures that
    // the number of variables can be represented by std::uint32_t.
    std::uint32_t idx = 0;
    for (auto it = sv_idx.second.begin(); it != sv_idx.second.end(); ++idx) {
        if (it->first == idx) {
            // The current index shows up in the sparse vector,
            // fetch the corresponding order and move to the next
            // element of the sparse vector.
            ret.push_back(it->second);
            assert(it->second != 0u);
            ++it;
        } else {
            // The current index does not show up in the sparse
            // vector, set the order to zero.
            ret.push_back(0);
        }
    }

    // Sanity check on the number of diff variables
    // inferred from the sparse vector.
    assert(ret.size() - 1u <= nargs);

    // Pad missing values at the end of ret.
    ret.resize(boost::safe_numerics::safe<decltype(ret.size())>(nargs) + 1);

    return ret;
}

// Small helper to facilitate the conversion of an iterable into a contiguous aligned NumPy array of type dt.
py::array as_carray(const py::iterable &v, int dt)
{
    using namespace pybind11::literals;

    // NOTE: use numpy.require() (rather than numpy.ascontiguousarray()) so that we guarantee not only C-contiguity, but
    // also that the resulting array's memory buffer's alignment is the one specified by the input dtype.
    py::array ret
        = py::module_::import("numpy").attr("require")(v, py::dtype(dt), py::make_tuple("C_CONTIGUOUS", "ALIGNED"));

    assert(ret.dtype().num() == dt);
    assert(is_npy_array_carray(ret));

    return ret;
}

// Common helper for the implementation of the ctor for the EOP/SW data classes.
template <typename Data>
Data eop_sw_ctor(const char *descr, const std::optional<py::array> &data, const std::optional<std::string> &timestamp,
                 const std::optional<std::string> &identifier)
{
    const auto with_data = static_cast<bool>(data);
    const auto with_ts = static_cast<bool>(timestamp);
    const auto with_id = static_cast<bool>(identifier);

    if (with_data && with_ts && with_id) {
        // Fetch the structured dtype corresponding to the row type of Data.
        const auto dt = make_aligned_dtype<typename Data::row_type>();

        // Check the dtype.
        //
        // NOTE: this will check structural equality of the layout, but not other attributes of the dtype (e.g., whether
        // it is aligned or not).
        if (!data->dtype().equal(dt)) [[unlikely]] {
            py_throw(PyExc_TypeError, fmt::format("Unable to construct an {} dataset: the dtype of the input NumPy "
                                                  "array is {}, but it should be {} instead",
                                                  descr, str(data->dtype()), str(dt))
                                          .c_str());
        }

        // Check the array dimensionality.
        if (data->ndim() != 1) [[unlikely]] {
            py_throw(PyExc_ValueError, fmt::format("Unable to construct an {} dataset: the input data array must have "
                                                   "1 dimension, but instead {} dimensions were detected",
                                                   descr, data->ndim())
                                           .c_str());
        }

        // Ensure that the data is C-contiguous *and* correctly aligned: reinterpreting the raw buffer as a range of
        // row_type below requires both.
        //
        // NOTE: because dt is an aligned dtype, np.require() will make an aligned copy if the input is not already
        // aligned - a merely contiguous buffer (e.g. a view into a byte buffer at an odd offset) can be misaligned,
        // which would make the reinterpretation UB.
        //
        // NOTE: this is essentially the same as as_carray(), but we cannot use it directly because at this time it
        // requires a builtin scalar type and here we are dealing with a structured dtype instead. Perhaps we can
        // consider an additional as_carray() overload in the future.
        const auto cdata = py::module_::import("numpy")
                               .attr("require")(*data, dt, py::make_tuple("C_CONTIGUOUS", "ALIGNED"))
                               .template cast<py::array>();

        // Fetch the begin/end iterators to the raw data.
        assert(boost::alignment::is_aligned(cdata.data(), alignof(typename Data::row_type)));
        const auto *const begin = static_cast<typename Data::row_type const *>(cdata.data());
        const auto *const end = begin + cdata.shape(0);

        return Data{std::ranges::subrange(begin, end), *timestamp, *identifier};
    } else if (!with_data && !with_ts && !with_id) {
        return Data{};
    } else [[unlikely]] {
        py_throw(PyExc_TypeError, fmt::format("Unable to construct an {} dataset: either none or all of the three "
                                              "construction arguments must be provided",
                                              descr)
                                      .c_str());
    }
}

// Explicit instantiations.
template heyoka::eop_data eop_sw_ctor(const char *, const std::optional<py::array> &,
                                      const std::optional<std::string> &, const std::optional<std::string> &);
template heyoka::sw_data eop_sw_ctor(const char *, const std::optional<py::array> &, const std::optional<std::string> &,
                                     const std::optional<std::string> &);

} // namespace heyoka_py
