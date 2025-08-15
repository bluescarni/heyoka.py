// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <string>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

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
#include <heyoka/expression.hpp>
#include <heyoka/number.hpp>

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

// Helper to check if a numpy array is a NPY_ARRAY_CARRAY (i.e., C-style
// contiguous and with properly aligned storage). The flag signals whether
// the array must also be writeable or not.
bool is_npy_array_carray(const py::array &arr, bool writeable)
{
    assert(PyObject_IsInstance(arr.ptr(), reinterpret_cast<PyObject *>(&PyArray_Type)));

    // NOTE: NPY_ARRAY_CARRAY is NPY_ARRAY_CARRAY_RO + writeable flag.
    if (writeable) {
        return PyArray_CHKFLAGS(reinterpret_cast<const PyArrayObject *>(arr.ptr()), NPY_ARRAY_CARRAY) != 0;
    } else {
        return PyArray_CHKFLAGS(reinterpret_cast<const PyArrayObject *>(arr.ptr()), NPY_ARRAY_CARRAY_RO) != 0;
    }
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

// Small helper to facilitate the conversion of an iterable into
// a contiguous NumPy array of type dt.
py::array as_carray(const py::iterable &v, int dt)
{
    using namespace pybind11::literals;

    py::array ret = py::module_::import("numpy").attr("ascontiguousarray")(v, "dtype"_a = py::dtype(dt));

    assert(ret.dtype().num() == dt);
    assert(is_npy_array_carray(ret));

    return ret;
}

} // namespace heyoka_py
