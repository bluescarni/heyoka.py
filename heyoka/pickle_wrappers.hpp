// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_PICKLE_WRAPPERS_HPP
#define HEYOKA_PY_PICKLE_WRAPPERS_HPP

#include <sstream>
#include <string>
#include <utility>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <fmt/format.h>

#include <pybind11/pybind11.h>

#include <Python.h>

#include "common_utils.hpp"

// Wrappers to implement Python pickling for exposed C++ classes
// via Boost.Serialization.

namespace heyoka_py
{

namespace py = pybind11;

template <typename T>
inline py::tuple pickle_getstate_wrapper(const py::object &self)
{
    auto &x = py::cast<const T &>(self);

    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oa(oss);
        oa << x;
    }

    return py::make_tuple(py::bytes(oss.str()), self.attr("__dict__"));
}

template <typename T>
inline std::pair<T, py::dict> pickle_setstate_wrapper(py::tuple state)
{
    if (py::len(state) != 2) {
        py_throw(PyExc_ValueError, (fmt::format("The state tuple passed to the deserialization wrapper "
                                                "must have 2 elements, but instead it has {} element(s)",
                                                py::len(state)))
                                       .c_str());
    }

    auto *ptr = PyBytes_AsString(state[0].ptr());
    if (!ptr) {
        py_throw(PyExc_TypeError, "A bytes object is needed in the deserialization wrapper");
    }

    std::istringstream iss;
    iss.str(std::string(ptr, ptr + py::len(state[0])));
    T x;
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> x;
    }

    return std::make_pair(std::move(x), state[1].cast<py::dict>());
}

} // namespace heyoka_py

#endif
