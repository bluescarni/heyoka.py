// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_LONG_DOUBLE_CASTER_HPP
#define HEYOKA_PY_LONG_DOUBLE_CASTER_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Python.h>

#include "common_utils.hpp"

namespace pybind11::detail
{

template <>
struct type_caster<long double> {
    PYBIND11_TYPE_CASTER(long double, _("long double"));
    bool load(handle src, bool)
    {
        namespace heypy = heyoka_py;

        if (!heypy::type(src).is(module_::import("numpy").attr("longdouble"))) {
            return false;
        }

        // We can reach the byte representation of the long double
        // object via its 'data' member.
        memoryview mv = src.attr("data");
        auto buffer = PyMemoryView_GET_BUFFER(mv.ptr());
        assert(PyBuffer_IsContiguous(buffer, 'C'));

        if (boost::numeric_cast<std::size_t>(buffer->len) != sizeof(long double)) {
            heypy::py_throw(
                PyExc_RuntimeError,
                ("error while converting a numpy.longdouble to a C++ long double: the size of the bytes array ("
                 + std::to_string(buffer->len) + ") does not match the size of the long double type ("
                 + std::to_string(sizeof(long double)) + ")")
                    .c_str());
        }

        auto start = static_cast<const unsigned char *>(buffer->buf);
        std::copy(start, start + sizeof(long double), reinterpret_cast<unsigned char *>(&value));

        return true;
    }
    static handle cast(const long double &src, return_value_policy, handle)
    {
        // NOTE: this is a bit hackish: we create
        // a numpy array of size 1 and return its sum.
        array_t<long double> tmp(array::ShapeContainer{1});
        tmp.mutable_at(0) = src;

        auto ret = tmp.attr("sum")();

        return ret.release();
    }
};

} // namespace pybind11::detail

#endif
