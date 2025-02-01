// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_COMMON_UTILS_HPP
#define HEYOKA_PY_COMMON_UTILS_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <variant>

#if defined(__GLIBCXX__)

#include <cxxabi.h>

#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

// NOTE: implementation of Py_SET_TYPE() for Python < 3.9. See:
// https://docs.python.org/3.11/whatsnew/3.11.html

#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)

static inline void _Py_SET_TYPE(PyObject *ob, PyTypeObject *type)
{
    ob->ob_type = type;
}
#define Py_SET_TYPE(ob, type) _Py_SET_TYPE((PyObject *)(ob), type)

#endif

namespace heyoka_py
{

namespace py = pybind11;

py::object builtins();

py::object type(const py::handle &);

std::string str(const py::handle &);

[[noreturn]] void py_throw(PyObject *, const char *);

bool callable(const py::handle &);

// Helper to expose the llvm_state getter
// for a Taylor integrator.
template <typename T>
inline void expose_llvm_state_property_ta(py::class_<T> &c)
{
    c.def_property_readonly("llvm_state", [](const T &self) {
        const auto &st = self.get_llvm_state();

        using ret_t = std::variant<std::reference_wrapper<const heyoka::llvm_state>,
                                   std::reference_wrapper<const heyoka::llvm_multi_state>>;

        return std::visit([](const auto &v) -> ret_t { return std::cref(v); }, st);
    });
}

// Helper to expose the llvm_state getter
// for a generic object.
template <typename T>
inline void expose_llvm_state_property(py::class_<T> &c)
{
    c.def_property_readonly("llvm_state", &T::get_llvm_state);
}

// NOTE: these are wrappers for the implementation of
// copy/deepcopy semantics for exposed C++ classes.
// Doing a simple C++ copy and casting it to Python
// won't work because it ignores dynamic Python
// attributes that might have been set on the input
// object o. Thus, the strategy is to first make
// a C++ copy of the original object and then attach
// to it copies of the dynamic attributes that were
// added to the original object from Python.
template <typename T>
py::object copy_wrapper(py::object o)
{
    // Fetch a pointer to the C++ copy.
    auto *o_cpp = py::cast<const T *>(o);

    // Copy the C++ object and transform it into
    // a Python object.
    // NOTE: no room for GIL unlock here, due
    // to possible copy of Pythonic event callbacks.
    py::object ret = py::cast(T(*o_cpp));

    // Fetch the list of attributes from the original
    // object and turn it into a set.
    auto orig_dir = py::set(builtins().attr("dir")(o));

    // Fetch the list of attributes form the copy
    // and turn it into a set.
    auto new_dir = py::set(builtins().attr("dir")(ret));

    // Compute the difference.
    // NOTE: this will be the list of attributes that
    // are in o but not in its copy.
    auto set_diff = orig_dir.attr("difference")(new_dir);

    // Iterate over the difference and assign the
    // missing attributes.
    for (auto attr_name : set_diff) {
        py::setattr(ret, attr_name, o.attr(attr_name));
    }

    return ret;
}

template <typename T>
py::object deepcopy_wrapper(py::object o, py::dict memo)
{
    // Fetch a pointer to the C++ copy.
    auto *o_cpp = py::cast<const T *>(o);

    // Copy the C++ object and transform it into
    // a Python object.
    // NOTE: no room for GIL unlock here, due
    // to possible copy of Pythonic event callbacks.
    py::object ret = py::cast(T(*o_cpp));

    // Fetch the list of attributes from the original
    // object and turn it into a set.
    auto orig_dir = py::set(builtins().attr("dir")(o));

    // Fetch the list of attributes form the copy
    // and turn it into a set.
    auto new_dir = py::set(builtins().attr("dir")(ret));

    // Compute the difference.
    // NOTE: this will be the list of attributes that
    // are in o but not in its copy.
    auto set_diff = orig_dir.attr("difference")(new_dir);

    // Iterate over the difference and deep copy the
    // missing attributes.
    auto copy_func = py::module_::import("copy").attr("deepcopy");
    for (auto attr_name : set_diff) {
        py::setattr(ret, attr_name, copy_func(o.attr(attr_name), memo));
    }

    return ret;
}

// Helper to check if a list of arrays may share any memory with each other.
// Quadratic complexity.
bool may_share_memory(const py::array &, const py::array &);

template <typename... Args>
bool may_share_memory(const py::array &a, const py::array &b, const Args &...args)
{
    const std::array args_arr = {std::cref(a), std::cref(b), std::cref(args)...};
    const auto nargs = args_arr.size();

    for (std::size_t i = 0; i < nargs; ++i) {
        for (std::size_t j = i + 1u; j < nargs; ++j) {
            if (may_share_memory(args_arr[i].get(), args_arr[j].get())) {
                return true;
            }
        }
    }

    return false;
}

// Helper to check if a numpy array is a NPY_ARRAY_CARRAY (i.e., C-style
// contiguous and with properly aligned storage). The flag signals whether
// the array must also be writeable or not.
bool is_npy_array_carray(const py::array &, bool = false);

namespace detail
{

bool with_pybind11_eh_impl();

} // namespace detail

// This function will invoke the function object f,
// wrapping its execution in the pybind11 C++ -> Python
// exception translation logic. If a C++ exception is
// thrown by the execution of f, the Python error flag is set
// and true is returned. Otherwise, false will be returned.
// The return value of f is ignored.
template <typename F>
bool with_pybind11_eh(const F &f)
{
    try {
        f();

        return false;
    } catch (py::error_already_set &e) {
        e.restore();
        return true;
#ifdef __GLIBCXX__
    } catch (abi::__forced_unwind &) {
        throw;
#endif
    } catch (...) {
        return detail::with_pybind11_eh_impl();
    }
}

// Functor to transform on-the-fly the content of a dtens
// from sparse format into dense format.
struct dtens_t_it {
    const heyoka::dtens *dt = nullptr;

    std::pair<heyoka::dtens::v_idx_t, heyoka::expression>
    operator()(const std::pair<heyoka::dtens::sv_idx_t, heyoka::expression> &) const;

    // Helper to implement the conversion from sparse to dense format.
    static heyoka::dtens::v_idx_t sparse_to_dense(const heyoka::dtens::sv_idx_t &, heyoka::dtens::v_idx_t::size_type);
};

py::array as_carray(const py::iterable &, int);

// Macros to avoid repetitions in commonly-used keyword arguments.
#define HEYOKA_PY_LLVM_STATE_ARGS                                                                                      \
    "opt_level"_a.noconvert() = 3, "force_avx512"_a.noconvert() = false, "slp_vectorize"_a.noconvert() = false,        \
    "fast_math"_a.noconvert() = false, "code_model"_a.noconvert() = heyoka::code_model::small,                         \
    "parjit"_a.noconvert() = heyoka::detail::default_parjit

#define HEYOKA_PY_CFUNC_ARGS(default_cm)                                                                               \
    "high_accuracy"_a.noconvert() = false, "compact_mode"_a.noconvert() = default_cm,                                  \
    "parallel_mode"_a.noconvert() = false, "batch_size"_a.noconvert() = 0, "prec"_a.noconvert() = 0

} // namespace heyoka_py

#endif
